from agent import BaseAgent, HiAgent
import gym
import numpy as np
from copy import deepcopy
import agent


class MetaAgent(BaseAgent):
    """
    a meta-agent for "Data Efficient Hierarchical Learning (HIRO)"
    """

    def __init__(self,
                 state_space,
                 action_space,
                 hi_agent_cls=HiAgent,
                 lo_agent_cls=BaseAgent,
                 models_dir=None,
                 c=100):
        # note, this will not work if initialised with
        # default parameters!
        # high- and lo_agent need to be explicitly set

        super().__init__(state_space, action_space)

        self.c = c  # number of time steps between high level actions
        self.t = 0  # step counter (resets after every c steps)

        self.hi_rewards = 0  # collects rewads for HL agent, applied every c steps

        self.hi_state = None  # state in which HL agent last took an action

        self.hi_action = None # HL agent's actions in (-1, 1) space (direct from network)
        self.goal = None # HL agent's actions translated to (low, high) space

        # these will record sequences necessary for off-policy relabelling later
        self.lo_action_seq = np.empty((c, *action_space.shape))
        self.lo_state_seq = np.empty((c, *state_space.shape))

        # self.lo_state_space = deepcopy(state_space)
        # self.lo_state_space.shape = (2 * self.lo_state_space.shape[0],)
        # CHANGED - this leads to an inconsistent box: .high and .low are still original shape
        self.lo_state_space = gym.spaces.Box(
            low=np.concatenate([state_space.low, state_space.low]),
            high=np.concatenate([state_space.high, state_space.high]),
            dtype=state_space.dtype)

        self.hi_action_space = deepcopy(state_space)

        # figure out if any of the states are angles in (-pi, pi)
        # so that we can calculate distances between them properly in the intrinsic reward function
        # this is an example of "artifical intelligence"
        self.state_space_angles = np.logical_and(
            np.isclose(state_space.high, np.pi),
            np.isclose(state_space.low, -np.pi))

        # this is needed to deal with the unbounded state space for velocities
        # so that we have something finite for the HL agent to set goals in.
        self.hi_action_space.high = np.clip(
            self.hi_action_space.high,
            a_min=-10, a_max=10) # TODO - revisit for bipedalwalker?
        self.hi_action_space.low = np.clip(
            self.hi_action_space.low,
            a_min=-10, a_max=10) #TODO obviously - maybe pass this as a parameter to MetaAgent
        self.hi_action_scaling = (self.hi_action_space.high - self.hi_action_space.low) / 2

        if models_dir is None:
            # high level agent's actions will be states, i.e. goals for the LL agent
            self.hi_agent = hi_agent_cls.new_trainable_agent(
                state_space=state_space, 
                action_space=self.hi_action_space, 
                use_long_buffer=True,
                epslon_greedy=0.6, 
                exploration_decay = 0.9999)

            # low level agent's states will be (state, goal) concatenated
            self.lo_agent = lo_agent_cls.new_trainable_agent(
                state_space=self.lo_state_space, 
                action_space=action_space, 
                epslon_greedy=0.7,
                exploration_decay = 0.99999)
        else:
            self.hi_agent = hi_agent_cls.load_pretrained_agent(filepath=models_dir + '/hi_agent',
                state_space=state_space, action_space=self.hi_action_space)

            self.lo_agent = lo_agent_cls.load_pretrained_agent(filepath=models_dir + '/lo_agent',
                state_space=self.lo_state_space, action_space=action_space)

        # we won't need networks etc here

    def reset_clock(self):
        self.t = 0

    def intrinsic_reward(self, state, goal, action, next_state):
        """
        a reward function for the LoAgent as defined in HIRO paper, eqn (3)

        difference: here we define goal as a state, not an increment (todo? does it matter?)
        note: action does not figure in the formula - this is apparently deliberate
        todo - make this a customisable function?
        """
        difference = np.abs(goal - next_state)
        # so that diff between np.pi, -np.pi = 0 for angles
        difference = np.where(self.state_space_angles,
                              np.abs(((difference + np.pi) % (2 * np.pi)) - np.pi),
                              difference)
        
        normalized_differences = difference / (self.hi_action_space.high - self.hi_action_space.low)
        final_reward = np.linalg.norm(1 - normalized_differences) /np.sqrt(state.shape[1]) ** 2
        
        return final_reward

    def act(self, state, explore=False):

        # is it time for a high-level action?
        if self.t % self.c == 0:
            self.t = 0

            # HL agent picks a new state from space and sets it as LL's goal
            self.hi_action = self.hi_agent.act(state, explore, rough_explore=False) #this will be in (-1, 1)
            self.goal = np.multiply(self.hi_action, self.hi_action_scaling) # element wise
            # save for later training
            self.hi_state = state

            # since our goal is a state rather than an increment, a goal transition function h() should not be needed, right?

        # action in environment comes from low level agent
        goal_broadcast = np.broadcast_to(self.goal, state.shape) #add a batch dimension just in case it's not there
        lo_action = self.lo_agent.act(np.concatenate([state, goal_broadcast], axis=1), explore, rough_explore=True)
        
        self.lo_state_seq[self.t] = state
        self.lo_action_seq[self.t] = lo_action

        self.t += 1

        return lo_action

    def train(self, state, action, reward: float, next_state, done: bool):

        # accumulate rewards for HL agent
        self.hi_rewards += reward

        # provide LL agent with intrinsic reward
        lo_reward = self.intrinsic_reward(state, self.goal, action, next_state) - 10*done

        # The lower-level policy will store the experience
        # (st, gt, at, rt, st+1, h(st, gt, st+1))
        # for off-policy training.
        # note: if the hi-agent picks a new goal in the next step, then this line will not be quite right.
        # but maybe it's actually better this way...

        lo_loss = self.lo_agent.train(
            np.concatenate([state, self.goal], axis=1),
            action,
            lo_reward,
            np.concatenate([next_state, self.goal], axis=1),
            done,
            relabel=False)

        # is it time to train the HL agent?
        hi_loss = None
        if self.t % self.c == 0:
            hi_loss = self.hi_agent.train(
                state=self.hi_state,
                action=self.hi_action, #(-1, 1)
                reward=self.hi_rewards,
                next_state=next_state,
                done=done,
                relabel=True,
                lo_state_seq=self.lo_state_seq,
                lo_action_seq=self.lo_action_seq,
                lo_current_policy=self.lo_agent.act)

            # reset this
            self.hi_rewards = 0

        return lo_loss, hi_loss, lo_reward


    def save_model(self, filepath:str):
        self.hi_agent.save_model(filepath + '/hi_agent')
        self.lo_agent.save_model(filepath + '/lo_agent')

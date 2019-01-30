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
                 hi_agent=HiAgent,
                 lo_agent=BaseAgent,
                 models_dir=None,
                 c=100):
        # note, this will not work if initialised with
        # default parameters!
        # high- and lo_agent need to be explicitly set

        # self.state_space = state_space
        # self.action_space = action_space
        super().__init__(state_space, action_space)

        self.c = c  # number of time steps between high level actions
        self.t = 0  # step counter (resets after every c steps)

        self.hi_rewards = 0  # collects rewads for HL agent, applied every c steps

        self.hi_state = None  # state in which HL agent last took an action

        self.goal = None # this will store the HL agent's actions

        # these will record sequences necessary for off-policy relabelling later
        self.lo_action_seq = np.empty((c, *action_space.shape))
        self.lo_state_seq = np.empty((c, *state_space.shape))

        self.lo_state_space = deepcopy(state_space)
        self.lo_state_space.shape = (2 * self.lo_state_space.shape[0],)

        self.hi_action_space = deepcopy(state_space)
        self.hi_action_space.high = np.clip(
            self.hi_action_space.high, 
            a_min=-10, a_max=10)
        self.hi_action_space.low = np.clip(
            self.hi_action_space.low,
            a_min=-10, a_max=10) #TODO obviously - maybe pass this as a parameter to MetaAgent

        if models_dir is None:
            # high level agent's actions will be states, i.e. goals for the LL agent
            self.hi_agent = hi_agent.new_trainable_agent(
                state_space=state_space, action_space=self.hi_action_space, use_long_buffer=True,
                epslon_greedy=0.6, exploration_decay = 0.9999)

            # low level agent's states will be (state, goal) concatenated
            self.lo_agent = lo_agent.new_trainable_agent(
                state_space=self.lo_state_space, action_space=action_space, epslon_greedy=0.7,
                exploration_decay = 0.99999)
        else:
            self.hi_agent = hi_agent.load_pretrained_agent(filepath=models_dir + '/hi_agent',
                state_space=state_space, action_space=self.hi_action_space)

            self.lo_agent = lo_agent.load_pretrained_agent(filepath=models_dir + '/lo_agent',
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

        # Dealing with angle variables TODO: is it possible to know wich variables are angles from the state space?
        
        
        assert goal.shape[0] == 1
        difference = abs(goal - next_state)
        difference[0,2] = difference[0,2] if difference[0,2] <= np.pi else 2 * np.pi - difference[0,2]
        difference[0,2] *= 2.0
        return -1 * np.linalg.norm(difference / (2.0 * self.hi_action_space.high))

    def act(self, state, explore=False):

        # is it time for a high-level action?
        if self.t % self.c == 0:
            self.t = 0

            # HL agent picks a new state from space and sets it as LL's goal
            self.hi_action = self.hi_agent.act(state, explore, rough_explore=False) #this will be in (-1, 1)
            hi_action_scaling = (self.hi_action_space.high - self.hi_action_space.low) / 2
            self.goal = np.multiply(self.hi_action, hi_action_scaling) # element wise
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
        lo_reward = self.intrinsic_reward(state, self.goal, action, next_state)

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
                self.hi_state,
                self.hi_action, #(-1, 1)
                self.hi_rewards,
                next_state,
                done,
                relabel=True,
                lo_state_seq=self.lo_state_seq,
                lo_action_seq=self.lo_action_seq,
                lo_current_policy=self.lo_agent.act)

            # reset this
            self.hi_rewards = 0

        return lo_loss, hi_loss
    

    def save_model(self, filepath:str):
        self.hi_agent.save_model(filepath + '/hi_agent')
        self.lo_agent.save_model(filepath + '/lo_agent')

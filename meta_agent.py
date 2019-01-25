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
                 c=10):
        # note, this will not work if initialised with
        # default parameters!
        # high- and lo_agent need to be explicitly set

        # self.state_space = state_space
        # self.action_space = action_space
        super().__init__(state_space, action_space)

        # high level agent's actions will be states, i.e. goals for the LL agent
        self.hi_agent = hi_agent.new_trainable_agent(
            state_space=state_space, action_space=state_space)

        lo_state_space = deepcopy(state_space)
        lo_state_space.shape = (2, *state_space.shape)

        # low level agent's states will be (state, goal) concatenated
        self.lo_agent = lo_agent.new_trainable_agent(
            state_space=lo_state_space, action_space=action_space)

        self.c = c  # number of time steps between high level actions
        self.t = 0  # step counter (resets after every c steps)

        self.hi_rewards = 0  # collects rewads for HL agent, applied every c steps

        self.hi_state = None  # state in which HL agent last took an action

        # these will record sequences necessary for off-policy relabelling later
        self.lo_action_seq = np.empty((c, *action_space.shape))
        self.lo_state_seq = np.empty((c, *state_space.shape))

        # we won't need networks etc here

    @staticmethod
    def intrinsic_reward(state, goal, action, next_state):
        """
        a reward function for the LoAgent as defined in HIRO paper, eqn (3)

        difference: here we define goal as a state, not an increment (todo? does it matter?)
        note: action does not figure in the formula - this is apparently deliberate
        todo - make this a customisable function?
        """
        return -1 * np.linalg.norm(goal - next_state)

    def act(self, state, explore=False):

        # is it time for a high-level action?
        if self.t % self.c == 0:
            self.t = 0

            # HL agent picks a new state from space and sets it as LL's goal
            self.goal = self.hi_agent.act(state, explore)

            # save for later training
            self.hi_state = state

            # since our goal is a state rather than an increment, a goal transition function h() should not be needed, right?

        # action in environment comes from low level agent
        goal_broadcast = np.broadcast_to(self.goal, state.shape) #add a batch dimension just in case it's not there
        lo_action = self.lo_agent.act(np.concatenate([state, goal_broadcast], axis=1), explore)
        

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
        if self.t % self.c == 0:
            hi_loss = self.hi_agent.train(
                self.hi_state,
                self.goal,
                self.hi_rewards,
                next_state,
                done,
                relabel=True,
                lo_state_seq=self.lo_state_seq,
                lo_action_seq=self.lo_action_seq,
                lo_current_policy=self.lo_agent.act)

            # reset this
            self.hi_rewards = 0

        return lo_loss #(lo_loss, hi_loss)
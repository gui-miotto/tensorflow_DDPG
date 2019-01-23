from agent import BaseAgent
import gym
import numpy as np


class HierarchicalAgent(BaseAgent):
    def __init__(self,
                 state_space,
                 action_space,
                 high_agent=BaseAgent,
                 low_agent=BaseAgent,
                 c=50):
        # self.state_space = state_space
        # self.action_space = action_space
        super().__init__(state_space, action_space)

        # high level agent's actions will be states, i.e. goals for the LL agent
        self.high_agent = high_agent(
            state_space=state_space, action_space=state_space)
        
        # low level agent's states will be (state, goal) concatenated
        self.low_agent = low_agent(
            state_space=self.concat_boxes(state_space, state_space), action_space=action_space)
        
        self.c = c  # number of time steps between high level actions
        self.t = 0  # step counter

        # we won't need networks etc here

    @staticmethod
    def concat_boxes(box1, box2):
        """joins two gym Box objects (observation spaces) into one"""
        return gym.spaces.Box(
            low=np.concatenate([box1.low, box2.low]),
            high=np.concatenate([box1.high, box2.high]))

    @staticmethod
    def intrinsic_reward(state, goal, action, next_state):
        # as defined in HIRO paper
        # remember, goal is defined 

    def act(self, state):
        # is it time for a high-level action?
        if self.t % self.c == 0:
            
            # train HL agent
            if self.t
            high_agent.train(self.high_state, action, reward: float, next_state, done: bool)

            # HL agent samples a new state from space and sets it as LL's goal
            self.goal = self.high_agent.act(state)
        
        # action in environment comes from low level agent
        return self.low_agent.act(np.concat(state, self.goal))

    def train(self, state, action, reward: float, next_state, done: bool):
        
        # accumulate rewards for HL agent

        # The higher-level controller provides the lower-level with an intrinsic reward 
        # rt = r(st, gt, at, st+1)
        # using a ﬁxed parameterized reward function r

        # The lower-level policy will store the experience 
        # (st, gt, at, rt, st+1, h(st, gt, st+1)) 
        # for off-policy training.
        self.low_agent.train(np.concat(state, self.goal), action, self.intrinsic_reward(state, self.goal, ))

        raise NotImplementedError
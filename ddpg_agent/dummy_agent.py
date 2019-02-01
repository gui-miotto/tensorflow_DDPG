from agent import BaseAgent, HiAgent
from ddpg_agent.replay_buffer import ReplayBuffer
import numpy as np
import os

class DummyAgent(BaseAgent):
    def __init__(self, 
        state_space: 'Box'=None, 
        action_space: 'Box'=None,
        **kwargs
        ):
        super().__init__(state_space, action_space)
        self.explr_magnitude = 0

    @classmethod
    def new_trainable_agent(cls, **kwargs) -> 'DummyAgent':
        return DummyAgent(**kwargs)
    
    @classmethod
    def load_pretrained_agent(cls, **kwargs) -> 'DummyAgent':
        return DummyAgent(**kwargs)

    def act(self, state, explore=False):
        assert not np.isnan(state).any()
        #assert state.shape[0] == 1
        #action = self.reverse_state(state)
        #action[0,2] = 0.0
        return np.zeros(shape=(1, self.action_space.shape[0]))
        #return action

    #todo: this doesnt work yet. not finishe        
    def reverse_state(self, state):
        unit_state = (state - self.state_space.low) / (self.state_space.high - self.state_space.low)
        rev_state = unit_state
        return rev_state

        
    def train(self,**kwargs):
        return 0, None
    
    def save_model(self, filepath:str):
        print('Dummy agent. Nothing to save')




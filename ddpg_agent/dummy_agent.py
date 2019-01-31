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
        self.epslon_greedy = 0

    @classmethod
    def new_trainable_agent(cls, **kwargs) -> 'DummyAgent':
        return DummyAgent(**kwargs)
    
    @classmethod
    def load_pretrained_agent(cls, **kwargs) -> 'DummyAgent':
        return DummyAgent(**kwargs)

    def act(self, state, explore=False):
        assert not np.isnan(state).any()
        return np.zeros(shape=(1, self.action_space.shape[0])) 
        
    def train(self,**kwargs):
        return 0, None
    
    def save_model(self, filepath:str):
        print('Dummy agent. Nothing to save')




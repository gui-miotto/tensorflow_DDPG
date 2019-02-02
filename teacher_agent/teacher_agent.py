from agent import BaseAgent
from ddpg_agent.replay_buffer import ReplayBuffer
from ddpg_agent.ddpg_agent import DDPGAgent
from teacher_agent.teachersmodel import TeachersModel
import numpy as np
import os

class TeacherAgent(BaseAgent):
    def __init__(self, 
        c,
        state_space: 'Box'=None, 
        action_space: 'Box'=None,
        **kwargs
        ):
        super().__init__(state_space, action_space)
        self.explr_magnitude = 0
        self.c = c

        self.brain = DDPGAgent.load_pretrained_agent(
            state_space=state_space,
            action_space=action_space,
            filepath='teacher_agent/teachersbrain')
        self.model = TeachersModel()


    @classmethod
    def new_trainable_agent(cls, **kwargs) -> 'TeacherAgent':
        return TeacherAgent(**kwargs)
    
    @classmethod
    def load_pretrained_agent(cls, **kwargs) -> 'TeacherAgent':
        return TeacherAgent(**kwargs)

    def act(self, state, explore=False):
        assert not np.isnan(state).any()

        final_state = np.copy(state)
        for t in range(self.c):
            action = self.brain.act(final_state)
            final_state = self.model.step(state=final_state, action=action[0])
        
        diff_goal = (final_state - state) / self.action_space.high
        return diff_goal
        
    def train(self,**kwargs):
        return 0, None
    
    def save_model(self, filepath:str):
        print('Teacher agent. Nothing to save')
    
    def modify_exploration_magnitude(self, factor, mode='increment'):
        pass
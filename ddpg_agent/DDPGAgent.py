from agent import BaseAgent
import numpy as np

class DDPGAgent(BaseAgent):
    def act(self, state):
        return self.action_space.sample()

    def train(self, state, action, reward: float, next_state, done: bool):
        pass
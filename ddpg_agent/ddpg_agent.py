from agent import BaseAgent
import numpy as np
from ddpg_agent.replay_buffer import ReplayBuffer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

class DDPGAgent(BaseAgent):
    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space)
        self.replay_buffer = ReplayBuffer()

    def act(self, state):
        return self.action_space.sample()

    def train(self, state, action, reward: float, next_state, done: bool):
        pass

    def setup_actor():
        self.actor_model = Sequential()
        self.actor_model.add(Dense(50, input_dim=self.state_space.shape[0], kernel_initializer='normal', activation='relu'))
        self.actor_model.add(Dense(1, kernel_initializer='normal'))
        self.actor_model.compile(loss='mean_squared_error', optimizer='adam')

    def setup_critic():
        self.critic_model = Sequential()
        self.critic_model.add(Dense(50, input_dim=self.state_space.shape[0], kernel_initializer='normal', activation='relu'))
        self.critic_model.add(Dense(1, kernel_initializer='normal'))
        self.critic_model.compile(loss='mean_squared_error', optimizer='adam')

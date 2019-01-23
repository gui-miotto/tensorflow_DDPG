from agent import BaseAgent
import numpy as np
from ddpg_agent.replay_buffer import ReplayBuffer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DDPGAgent(BaseAgent):
    def __init__(self, state_space, action_space, batch_size=32, discount_factor=0.99, tau=0.001):
        super().__init__(state_space, action_space)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.tau = tau

        # initialize buffer, actor and critic
        self.replay_buffer = ReplayBuffer(batch_size=batch_size)
        self.setup_actor()
        self.setup_critic()
        
    def act(self, state):
        prediction = self.actor_behaviour.predict(state.reshape(1,-1))
        return np.clip(prediction[0], self.action_space.low, self.action_space.high)
        

    def train(self, state, action, reward: float, next_state, done: bool):
        self.replay_buffer.add(state, action, next_state, reward, done)

        batch = self.replay_buffer.sample_batch()
        
        # ask actor target network for actions ...
        actions = self.actor_target.predict(batch.states_after)

        # ask critic target for values of these actions
        values = self.critic_target.predict(np.hstack((batch.states_after, actions)))
    
        # train critic
        ys = batch.rewards.reshape((-1, 1)) + self.discount_factor * values
        xs = np.hstack((batch.states_before, batch.actions))
        self.critic_behaviour.fit(xs, ys)
        

        # todo weird gradient step


        def update_target_weights(behaviour, target):
            behaviour_weights = behaviour.get_weights()
            target_weights = target.get_weights()
            
            new_target_weights = [self.tau*b + (1-self.tau)*t for b, t in zip(behaviour_weights, target_weights)]
            target.set_weights(new_target_weights)
        
        update_target_weights(self.actor_behaviour, self.actor_target)
        update_target_weights(self.critic_behaviour, self.critic_target)

    def setup_actor(self):
        print("shape", self.state_space.shape[0])
        self.actor_behaviour = Sequential()
        self.actor_behaviour.add(Dense(50, input_dim=self.state_space.shape[0], kernel_initializer='normal', activation='relu'))
        self.actor_behaviour.add(Dense(1, kernel_initializer='normal'))
        self.actor_behaviour.compile(loss='mean_squared_error', optimizer='adam')
        
        self.actor_target = tf.keras.models.clone_model(self.actor_behaviour)

    def setup_critic(self):
        self.critic_behaviour = Sequential()
        self.critic_behaviour.add(Dense(50, input_dim=self.state_space.shape[0]+self.action_space.shape[0] , kernel_initializer='normal', activation='relu'))
        self.critic_behaviour.add(Dense(1, kernel_initializer='normal'))
        self.critic_behaviour.compile(loss='mean_squared_error', optimizer='adam')

        self.critic_target = tf.keras.models.clone_model(self.critic_behaviour)

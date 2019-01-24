from agent import BaseAgent
import numpy as np
from ddpg_agent.replay_buffer import ReplayBuffer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DDPGAgent(BaseAgent):
    def __init__(self,
     state_space, 
     action_space, 
     batch_size=32, 
     discount_factor=0.99, 
     tau=0.001,
     leaning_rate_actor=0.0001,
     leaning_rate_critic=0.001):
        super().__init__(state_space, action_space)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.learning_rate_actor = leaning_rate_actor
        self.learning_rate_critic = leaning_rate_critic

        # initialize buffer, actor and critic
        self.replay_buffer = ReplayBuffer(buffer_size=10000,batch_size=batch_size)
        self.setup_actor()
        self.setup_critic()

        # tensorflow graph for gratient policy
        critic_gradient = tf.gradients(self.critic_behaviour.output, self.critic_behaviour.input)[0][:,4:5]
        actor_gradient = tf.gradients(self.actor_behaviour.output, self.actor_behaviour.trainable_variables, -critic_gradient)
        
        
        # todo understand, rename variable
        #normalized_actor_gradient = zip(actor_gradient, self.actor_behaviour.trainable_variables)
        normalized_actor_gradient = zip(list(map(lambda x: tf.div(x, self.batch_size), actor_gradient)), self.actor_behaviour.trainable_variables)
        

        self.train_actor = tf.train.AdamOptimizer(self.learning_rate_actor).apply_gradients(normalized_actor_gradient)
        
        session = tf.keras.backend.get_session()
        session.run(tf.global_variables_initializer())
        
    def act(self, state, training=False):
        action = self.actor_behaviour.predict(state.reshape(1,-1))[0]

        if training:
            # todo ornstein uhlenbeck?
            action += np.random.normal(scale=1)

            #if np.random.rand() > 0.7:
            #    action = -action
        
        #print(action)
        return np.clip(action, self.action_space.low, self.action_space.high)
        

    def train(self, state, action, reward: float, next_state, done: bool):
        self.replay_buffer.add(state, action, next_state, reward, done)

        batch = self.replay_buffer.sample_batch()
        
        # ask actor target network for actions ...
        target_actions = self.actor_target.predict(batch.states_after)

        # ask critic target for values of these actions
        values = self.critic_target.predict(np.hstack((batch.states_after, target_actions)))
    
        # train critic
        ys = batch.rewards.reshape((-1, 1)) + self.discount_factor * values * ~(batch.done_flags.reshape((-1, 1)))
        xs = np.hstack((batch.states_before, batch.actions))
        self.critic_behaviour.fit(xs, ys, verbose=0)
        
        # train actor
        session = tf.keras.backend.get_session()

        behaviour_actions = self.actor_behaviour.predict(batch.states_before)

        session.run([self.train_actor], {
            self.critic_behaviour.input: np.hstack((batch.states_before, behaviour_actions)),
            self.actor_behaviour.input: batch.states_before
        })

        def update_target_weights(behaviour, target):
            behaviour_weights = behaviour.get_weights()
            target_weights = target.get_weights()

            new_target_weights = [self.tau*b + (1-self.tau)*t for b, t in zip(behaviour_weights, target_weights)]
            target.set_weights(new_target_weights)
        
        # update target weights for actor and critic
        update_target_weights(self.actor_behaviour, self.actor_target)
        update_target_weights(self.critic_behaviour, self.critic_target)

    def setup_actor(self):
        self.actor_behaviour = Sequential()
        self.actor_behaviour.add(Dense(50, input_dim=self.state_space.shape[0], kernel_initializer='normal', activation='relu'))
        self.actor_behaviour.add(Dense(1, kernel_initializer='normal', activation='tanh'))
        adam = tf.keras.optimizers.Adam(self.learning_rate_actor)
        self.actor_behaviour.compile(loss='mean_squared_error', optimizer=adam)
        
        self.actor_target = tf.keras.models.clone_model(self.actor_behaviour)

    def setup_critic(self):
        self.critic_behaviour = Sequential()
        self.critic_behaviour.add(Dense(50, input_dim=self.state_space.shape[0]+self.action_space.shape[0] , kernel_initializer='normal', activation='relu'))
        self.critic_behaviour.add(Dense(1, kernel_initializer='normal'))
        adam = tf.keras.optimizers.Adam(self.learning_rate_critic)
        self.critic_behaviour.compile(loss='mean_squared_error', optimizer=adam)

        self.critic_target = tf.keras.models.clone_model(self.critic_behaviour)

from agent import BaseAgent, HiAgent
from ddpg_agent.replay_buffer import ReplayBuffer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, ReLU
from tensorflow.keras.initializers import RandomNormal
from ddpg_agent.ou_noise import OUNoise

import os

class DDPGAgent(HiAgent):
    def __init__(self,
        state_space: 'Box'=None,
        action_space: 'Box'=None,
        actor_behaviour: Sequential=None,
        actor_target: Sequential=None,
        critic_behaviour: Sequential=None,
        critic_target: Sequential=None,
        replay_buffer: ReplayBuffer=None,
        train_actor_op: tf.Tensor=None,
        discount_factor=0.99,
        tau=0.001,
        exploration_magnitude=0.4,
        exploration_magnitude_min=0.05,
        exploration_decay=0.9999,
        **kwargs,
        ):
        super().__init__(state_space, action_space)
        self.actor_behaviour = actor_behaviour
        self.actor_target = actor_target
        self.critic_behaviour = critic_behaviour
        self.critic_target = critic_target
        self.replay_buffer = replay_buffer
        self.train_actor_op = train_actor_op
        self.discount_factor = discount_factor
        self.tau = tau
        self.explr_magnitude = exploration_magnitude
        self.explr_magnitude_min = exploration_magnitude_min
        self.explr_decay = exploration_decay
        self.ou_noise = OUNoise(self.action_space.shape[0])

    @classmethod
    def new_trainable_agent(cls,
        learning_rate_actor=0.0001,
        learning_rate_critic=0.0001,
        batch_size=32,
        use_long_buffer=False,
        n_units = [128, 64],
        weights_stdev=0.000001,
        **kwargs) -> 'DDPGAgent':

        # Get dimensionality of action/state space
        state_dim = kwargs['state_space'].shape[0]
        n_actions = kwargs['action_space'].shape[0]

        # Weights initialization
        kernel_initializer = RandomNormal(mean=0.0, stddev=weights_stdev, seed=np.random.randint(9999))

        # Create actor_behaviour network
        adam_act = tf.keras.optimizers.Adam(learning_rate_actor)
        act_behav = Sequential()
        act_behav.add(Dense(n_units[0], input_dim=state_dim, kernel_initializer=kernel_initializer, activation='relu'))
        for layer_units in n_units[1:]:
            act_behav.add(Dense(layer_units, kernel_initializer=kernel_initializer, activation='relu'))
        act_behav.add(Dense(n_actions, kernel_initializer=kernel_initializer, activation='tanh'))
        act_behav.compile(loss='mean_squared_error', optimizer=adam_act)

        # Create crit_behaviour network
        adam_crit = tf.keras.optimizers.Adam(learning_rate_critic)
        crit_behav = Sequential()
        crit_behav.add(Dense(n_units[0], input_dim=state_dim+n_actions, kernel_initializer=kernel_initializer, activation='relu'))
        for layer_units in n_units[1:]:
            crit_behav.add(Dense(layer_units, kernel_initializer=kernel_initializer, activation='relu'))
        crit_behav.add(Dense(1, kernel_initializer=kernel_initializer))
        crit_behav.compile(loss='mean_squared_error', optimizer=adam_crit) # todo: actor doesnt have a explicit loss, why are we specifying one

        # Create target networks with the same architecture of the behaviour networks
        crit_targ = tf.keras.models.clone_model(crit_behav)
        act_targ = tf.keras.models.clone_model(act_behav)

        # Construct tensorflow graph for actor gradients
        critic_gradient = tf.gradients(crit_behav.output, crit_behav.input)[0][:,state_dim:] #the ACTION is the fifth element of this array (we concatenated it with the state)
        actor_gradient = tf.gradients(act_behav.output, act_behav.trainable_variables, -critic_gradient)
        # todo understand, rename variable
        #normalized_actor_gradient = zip(actor_gradient, self.actor_behaviour.trainable_variables)
        normalized_actor_gradient = zip(list(map(lambda x: tf.div(x, batch_size), actor_gradient)), act_behav.trainable_variables)
        train_actor = tf.train.AdamOptimizer(learning_rate_actor).apply_gradients(normalized_actor_gradient)

        # Initialize variable
        session = tf.keras.backend.get_session()
        session.run(tf.global_variables_initializer())

        # Makes sure that target and behaviour start equal
        crit_targ.set_weights(crit_behav.get_weights())
        act_targ.set_weights(act_behav.get_weights())

        # Create replay buffer
        replay_buffer = ReplayBuffer(buffer_size=20000,batch_size=batch_size, use_long=use_long_buffer)

        return DDPGAgent(actor_behaviour=act_behav, actor_target=act_targ, 
            critic_behaviour=crit_behav, critic_target=crit_targ, replay_buffer=replay_buffer,
            train_actor_op=train_actor, **kwargs)

    @classmethod
    def load_pretrained_agent(cls, filepath, **kwargs):
        act_behav = tf.keras.models.load_model(filepath+'/actbeh.model')
        act_targ = tf.keras.models.load_model(filepath+'/acttar.model')
        crit_behav = tf.keras.models.load_model(filepath+'/cribeh.model')
        crit_targ = tf.keras.models.load_model(filepath+'/critar.model')
        return DDPGAgent(actor_behaviour=act_behav, actor_target=act_targ, 
            critic_behaviour=crit_behav, critic_target=crit_targ, **kwargs)


    def act(self, state, explr_mode="no_exploration"):
        # action = self.actor_behaviour.predict(self.reshape_input(state))[0]
        assert not np.isnan(state).any()
        action = self.actor_behaviour.predict(state) #tanh'd (-1, 1)
        
        if explr_mode != "no_exploration":
            if explr_mode == "ou_noise":
                noise = self.ou_noise.noise()
                action = (1-self.explr_magnitude)*action + noise * self.explr_magnitude
            elif explr_mode == "gaussian":
                noise = np.random.normal(scale=self.explr_magnitude, size=action.shape)
                action += noise
            elif explr_mode == "rough_explore":
                if np.random.rand() < self.explr_magnitude:
                    action = 1 - 2 * np.random.randint(0, 2, size=action.shape)
            else:
                raise Exception('Invalid exploration method')
            if self.explr_magnitude > self.explr_magnitude_min:
                self.explr_magnitude *= self.explr_decay
        
        action = np.clip(action, a_min=-1, a_max=1)

        assert not np.isnan(action).any()
        return action

    def modify_exploration_magnitude(self, factor, mode='increment'):
        if mode == 'increment':
            self.explr_magnitude += factor
        elif mode == 'assign':
            self.explr_magnitude = factor
        else:
            pass

    def train(self,
              state,
              action,
              reward: float,
              next_state,
              done: bool,
              relabeller=None,
              lo_state_seq=None,
              lo_action_seq=None,
              lo_current_policy=None):
        assert self.replay_buffer is not None, 'It seems like you are trying to train a pretrained model. Not cool, dude.'
        # add a transition to the buffer
        self.replay_buffer.add(
            state_before=np.squeeze(state, axis=0), 
            action=np.squeeze(action, axis=0), 
            state_after=np.squeeze(next_state, axis=0), 
            reward=reward, 
            done_flag=done, 
            lo_state_seq=lo_state_seq, 
            lo_action_seq=lo_action_seq)
        # ...
        

        #sample a batch
        batch = self.replay_buffer.sample_batch()

        # off policy correction / relabelling!
        if relabeller is not None:
            for i in range(batch.actions.shape[0]): #TODO make r_g fn accept batches
                batch.actions[i] = relabeller(
                    orig_hi_action=batch.actions[i],
                    goal_scaler=self.scale_action,
                    lo_state_seq=batch.lo_state_seqs[i], 
                    lo_action_seq=batch.lo_action_seqs[i],
                    lo_current_policy=lo_current_policy)

        # ask actor target network for actions ...
        target_actions = self.actor_target.predict(batch.states_after)
        # ask critic target for values of these actions
        values = self.critic_target.predict(np.concatenate((batch.states_after, target_actions), axis=1))
        # train critic
        ys = batch.rewards.reshape((-1, 1)) + self.discount_factor * values * ~(batch.done_flags.reshape((-1, 1)))
        xs = np.concatenate([batch.states_before, batch.actions], axis=1)
        info = self.critic_behaviour.fit(xs, ys, verbose=0)
        # train actor
        session = tf.keras.backend.get_session()
        # behaviour_actions = self.actor_behaviour.predict(self.reshape_input(batch.states_before))
        behaviour_actions = self.actor_behaviour.predict(batch.states_before)
        session.run([self.train_actor_op], {
            self.critic_behaviour.input: np.concatenate((batch.states_before, behaviour_actions), axis=1),
            self.actor_behaviour.input: batch.states_before
        })

        def update_target_weights(behaviour, target):
            behaviour_weights = behaviour.get_weights()
            target_weights = target.get_weights()
            new_target_weights = [self.tau*b + (1-self.tau)*t for b, t in zip(behaviour_weights, target_weights)]
            target.set_weights(new_target_weights)

        # slowly update target weights for actor and critic
        update_target_weights(self.actor_behaviour, self.actor_target)
        update_target_weights(self.critic_behaviour, self.critic_target)
        
        loss = info.history['loss'][0]
        return loss, None #to be compatible with return type of MetaAgent
    
    def save_model(self, filepath:str):
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        tf.keras.models.save_model(self.actor_behaviour, filepath+'/actbeh.model')
        tf.keras.models.save_model(self.actor_target, filepath+'/acttar.model')
        tf.keras.models.save_model(self.critic_behaviour, filepath+'/cribeh.model')
        tf.keras.models.save_model(self.critic_target, filepath+'/critar.model')

        print('Models saved.')




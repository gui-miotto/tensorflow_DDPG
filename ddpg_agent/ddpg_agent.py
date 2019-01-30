from agent import BaseAgent, HiAgent
from ddpg_agent.replay_buffer import ReplayBuffer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, ReLU
from tensorflow.keras.initializers import RandomNormal
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
        epslon_greedy=0.4, #TODO
        exploration_decay=0.9999
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
        self.epslon_greedy = epslon_greedy
        self.explr_decay = exploration_decay

    @classmethod
    def new_trainable_agent(cls,
        learning_rate_actor=0.0001,
        learning_rate_critic=0.0001,
        batch_size=32,
        use_long_buffer=False,
        **kwargs) -> 'DDPGAgent':

        # Get dimensionality of action/state space
        state_dim = kwargs['state_space'].shape[0]
        n_actions = kwargs['action_space'].shape[0]

        # Weights initialization
        kernel_initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None) # TODO: fix seed to test some hyperparameters that are non-network related

        # Create actor_behaviour network
        adam_act = tf.keras.optimizers.Adam(learning_rate_actor)
        act_behav = Sequential()
        act_behav.add(Dense(100, input_dim=state_dim, kernel_initializer=kernel_initializer, activation='relu'))
        act_behav.add(Dense(50, kernel_initializer=kernel_initializer, activation='relu'))
        act_behav.add(Dense(n_actions, kernel_initializer=kernel_initializer, activation='tanh'))
        act_behav.compile(loss='mean_squared_error', optimizer=adam_act)

        # Create crit_behaviour network
        adam_crit = tf.keras.optimizers.Adam(learning_rate_critic)
        crit_behav = Sequential()
        crit_behav.add(Dense(100, input_dim=state_dim+n_actions, kernel_initializer=kernel_initializer, activation='relu')) #TODO for 2d actions
        crit_behav.add(Dense(50, kernel_initializer=kernel_initializer, activation='relu'))
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
        replay_buffer = ReplayBuffer(buffer_size=80000,batch_size=batch_size, use_long=use_long_buffer)

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


    def act(self, state, explore=False, rough_explore=True):
        # action = self.actor_behaviour.predict(self.reshape_input(state))[0]
        assert not np.isnan(state).any()
        action = self.actor_behaviour.predict(state) #tanh'd (-1, 1)
        
        if explore and np.random.rand() < self.epslon_greedy:
            if rough_explore:
                action = np.ones_like(action)
                action[np.random.rand(*action.shape) > 0.5] = -1
            else:
                action = (np.random.rand(*(action.shape)) * 2) - 1.
            self.epslon_greedy = self.epslon_greedy * self.explr_decay if self.epslon_greedy > 0.05 else 0.05

        action = np.clip(action, a_min=-1, a_max=1)

        assert not np.isnan(action).any() # todo remove?
        return action 
        
    def train(self,
              state,
              action,
              reward: float,
              next_state,
              done: bool,
              relabel=False,
              lo_state_seq=None,
              lo_action_seq=None,
              lo_current_policy=None):
        assert self.replay_buffer is not None, 'It seems like you are trying to train a pretrained model. Not cool, dude.'
        # add a transition to the buffer
        
        self.replay_buffer.add(np.squeeze(state, axis=0), np.squeeze(action, axis=0), np.squeeze(next_state, axis=0), reward, done, lo_state_seq, lo_action_seq)
        #sample a batch
        batch = self.replay_buffer.sample_batch()

        # off policy correction / relabelling!
        if relabel:
            for i in range(batch.actions.shape[0]): #TODO make r_g fn accept batches
                batch.actions[i] = self.relabel_goal(batch.actions[i], batch.lo_state_seqs[i], batch.lo_action_seqs[i], lo_current_policy)

        # ask actor target network for actions ...
        # target_actions = self.actor_target.predict(self.reshape_input(batch.states_after))
        target_actions = self.actor_target.predict(batch.states_after)
        # ask critic target for values of these actions
        # values = self.critic_target.predict(self.reshape_input(batch.states_after, target_actions))
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
            # self.critic_behaviour.input: self.reshape_input(batch.states_before, behaviour_actions),
            # self.actor_behaviour.input: self.reshape_input(batch.states_before)
            self.critic_behaviour.input: np.concatenate((batch.states_after, behaviour_actions), axis=1),
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
        return loss
    
    def save_model(self, filepath:str):
        if not os.path.exists(filepath):
            os.mkdir(filepath)  

        tf.keras.models.save_model(self.actor_behaviour, filepath+'/actbeh.model')
        tf.keras.models.save_model(self.actor_target, filepath+'/acttar.model')
        tf.keras.models.save_model(self.critic_behaviour, filepath+'/cribeh.model')
        tf.keras.models.save_model(self.critic_target, filepath+'/critar.model')

        print('Models saved.')




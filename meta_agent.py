from agent import BaseAgent, HiAgent
import gym
import numpy as np
from copy import deepcopy
import agent


class MetaAgent(BaseAgent):
    """
    a meta-agent for "Data Efficient Hierarchical Learning (HIRO)"
    """

    def __init__(self,
                 state_space,
                 action_space,
                 hi_agent_cls=HiAgent,
                 lo_agent_cls=BaseAgent,
                 models_dir=None,
                 c=1,
                 hi_action_space=None):
        # note, this will not work if initialised with
        # default parameters!
        # high- and lo_agent need to be explicitly set

        super().__init__(state_space, action_space)

        self.c = c  # number of time steps between high level actions
        self.t = 0  # step counter (resets after every c steps)

        self.hi_rewards = 0  # collects rewads for HL agent, applied every c steps

        self.hi_state = None  # state in which HL agent last took an action

        self.hi_action = None # HL agent's actions in (-1, 1) space (direct from network)
        self.goal = None # HL agent's actions translated to (low, high) space
        self.lo_reward = None # so that this can be retrieved for score display

        # these will record sequences necessary for off-policy relabelling later
        self.lo_action_seq = np.empty((c, *action_space.shape))
        self.lo_state_seq = np.empty((c, *state_space.shape))

        self.lo_state_space = gym.spaces.Box(
            low=np.concatenate([state_space.low, state_space.low]),
            high=np.concatenate([state_space.high, state_space.high]),
            dtype=state_space.dtype)

        if hi_action_space is None:
            self.hi_action_space = deepcopy(state_space)
        else:
            #clipping!
            self.hi_action_space = hi_action_space            

        # figure out if any of the states are angles in (-pi, pi)
        # so that we can calculate distances between them properly in the intrinsic reward function
        # this is an example of "artifical intelligence"
        self.state_space_angles = np.logical_and(
            np.isclose(state_space.high, np.pi),
            np.isclose(state_space.low, -np.pi))

        if models_dir is None:
            # high level agent's actions will be states, i.e. goals for the LL agent
            self.hi_agent = hi_agent_cls.new_trainable_agent(
                state_space=state_space,
                action_space=self.hi_action_space,
                use_long_buffer=True,
                exploration_magnitude=1.0, 
                exploration_decay = 0.9999,
                discount_factor=0.99,
                n_units=[256, 128, 64],
                weights_stdev=0.03,
                )

            # low level agent's states will be (state, goal) concatenated
            self.lo_agent = lo_agent_cls.new_trainable_agent(
                state_space=self.lo_state_space, 
                action_space=action_space, 
                exploration_magnitude=2.0,
                exploration_decay = 0.99999,
                discount_factor=0.95,
                n_units=[128, 64],
                weights_stdev=0.0001,
                )
        else:
            self.hi_agent = hi_agent_cls.load_pretrained_agent(filepath=models_dir + '/hi_agent',
                state_space=state_space, action_space=self.hi_action_space)

            self.lo_agent = lo_agent_cls.load_pretrained_agent(filepath=models_dir + '/lo_agent',
                state_space=self.lo_state_space, action_space=action_space)

        # we won't need networks etc here

    def reset_clock(self):
        self.t = 0

    @staticmethod
    def goal_transition(goal, state, next_state):
        """`
        yup.
        """
        return state + goal - next_state

    def intrinsic_reward(self, state, goal, action, next_state):
        """
        a reward function for the LoAgent as defined in HIRO paper, eqn (3)

        difference: here we define goal as a state, not an increment (todo? does it matter?)
        note: action does not figure in the formula - this is apparently deliberate
        """
        # difference = np.abs(goal - next_state)
        difference = np.abs(state + goal - next_state) #now an increment

        # so that diff between np.pi, -np.pi = 0 for angles
        difference = np.where(self.state_space_angles,
                              ((difference + np.pi) % (2 * np.pi)) - np.pi,
                              difference)

        normalized_differences = np.abs(difference) / (self.hi_action_space.high - self.hi_action_space.low)

        final_reward = np.linalg.norm(1 - normalized_differences) /np.sqrt(state.shape[1]) # ** 2 #removed the square. ask gui why

        return final_reward

    def modify_exploration_magnitude(self, factor, mode='increment'):
        self.hi_agent.modify_exploration_magnitude(factor=factor, mode=mode)
        self.lo_agent.modify_exploration_magnitude(factor=factor, mode=mode)


    def act(self, state, explr_mode="no_exploration"):

        # is it time for a high-level action?
        if self.t % self.c == 0:
            self.t = 0

            # HL agent picks a new state from space and sets it as LL's goal
            self.hi_action = self.hi_agent.act(state, explr_mode) #this will be in (-1
            self.goal = self.hi_agent.scale_action(self.hi_action)

            # save for later training
            self.hi_state = state

        # UPDATE: goal is now an increment. See train() for goal transition - we need next_state available, so we can't do it here

        # action in environment comes from low level agent
        goal_broadcast = np.broadcast_to(self.goal, state.shape) #add a batch dimension just in case it's not there
        lo_action = self.lo_agent.act(
            state=np.concatenate([state, goal_broadcast], axis=1), 
            explr_mode=explr_mode)
        
        self.lo_state_seq[self.t] = state
        self.lo_action_seq[self.t] = lo_action #unscaled - still tanh space. good!

        self.t += 1

        return lo_action

    def train(self, state, action, reward: float, next_state, done: bool):

        # accumulate rewards for HL agent
        self.hi_rewards += reward

        # provide LL agent with intrinsic reward
        self.lo_reward = self.intrinsic_reward(state=state, goal=self.goal, action=action, next_state=next_state)

        # now transition the goal in preparation for the next act() step
        # old_goal = self.goal
        # self.goal = self.goal_transition(self.goal, state, next_state)
        
        # print("Transition: ", state, "g", goal, "s+g" state + goal)
        # is it the end of a sub-episode?
        # note, sequence is: lo.act(), t++, lo.train().
        # so, if t % c == 0 now, lo.agent has just reached the end of the episode
        # and in the next act() step will receive a new goal
        # also: lo_agent should not know or care if it's the end of the real episode:
        # this is hi_agent's concern!
        lo_done = (self.t % self.c == 0)

        # The lower-level policy will store the experience
        # (st, gt, at, rt, st+1, h(st, gt, st+1))
        # for off-policy training.
        lo_loss, _ = self.lo_agent.train(
            np.concatenate([state, self.goal], axis=1),
            action,
            self.lo_reward,
            np.concatenate([next_state, self.goal_transition(self.goal, state, next_state)], axis=1),
            lo_done)

        # is it time to train the HL agent?
        hi_loss = None
        if self.t % self.c == 0:
            hi_loss, _ = self.hi_agent.train(
                state=self.hi_state,
                action=self.hi_action, #(-1, 1)
                reward=self.hi_rewards,
                next_state=next_state,
                done=done,
                relabeller=self.relabel_hi_action,
                lo_state_seq=self.lo_state_seq,
                lo_action_seq=self.lo_action_seq,
                lo_current_policy=self.lo_agent.act)

            # reset this
            self.hi_rewards = 0

        return lo_loss, hi_loss

    @staticmethod
    def relabel_hi_action(orig_hi_action, goal_scaler, lo_state_seq, lo_action_seq,
                     lo_current_policy):
        """
        this will be used internally by the HiAgent in its train() routine
        where, at some point, we should have
            if relabel: 
                transition_tuple = relabel_goal(...)
            else: 
                transition_tuple = ...

        Note: this currently only works with individual tuples, not minibatches of them (todo?)

        Parameters
        ----------

        orig_hi_action : np.array: (*state.shape) in tanh space

        lo_state_seq : np.array: (c, *state.shape)
            array containing the c states visited by the LoAgent since the last HiAgent training step
        
        lo_action_seq : np.array: (c, *action.shape)
            array containing the c actions taken by the LoAgent since the last HiAgent training step
            (again, tanh space)

        lo_current_policy : func: state -> action
            the act() function of the LoAgent (supplied by a MetaAgent)        
        """

        # get variance from observed states
        stdev_goal = np.std(lo_state_seq, axis=0)  #todo - cov instead

        # eight candidate goals sampled randomly from a Gaussian centered at s_t+c − s_t
        # i.e. around the original goal
        # (since goals implemented as states, this just becomes s_t+c)
        n_candidate_hi_acts = 8

        candidate_hi_actions = np.random.normal(
            loc=orig_hi_action,
            scale=(1/3), #since we're in (1/1) space...
            size=(n_candidate_hi_acts, *orig_hi_action.shape))

        # also include the original hi_action gt
        candidate_hi_actions = np.concatenate([
            candidate_hi_actions,
            np.expand_dims(orig_hi_action, axis=0),
        ], axis=0)

        candidate_goals = goal_scaler(candidate_hi_actions)

        # now that we're in state space, add the final (10th) goal:
        # one corresponding to the path taken: s_t+c - s_t
        candidate_goals = np.concatenate([
            candidate_goals,
            np.expand_dims(lo_state_seq[-1] - lo_state_seq[0], axis=0)
        ], axis=0)

        lo_policy_likelihoods = []

        # running this as a for loop is not quite optimal -- todo later
        for g in range(n_candidate_hi_acts + 2):

            lo_state_deltas = np.subtract(
                lo_state_seq,
                lo_state_seq[0])

            goal_over_time = np.broadcast_to(candidate_goals[g], shape=lo_state_seq.shape) - lo_state_deltas

            # transform the (state) c-tuple into a (state, goal) c-tuple
            # shape = (c, 2, *state_shape)
            lo_stategoal_seq = np.concatenate([
                lo_state_seq,
                goal_over_time
            ], axis=1)

            # what actions would the current LoAgent take, given goal g?
            lo_current_actions = lo_current_policy(lo_stategoal_seq)

            # how far do they diverge from the actual actions, given original goal?
            # shape = (c, *action_shape)
            lo_sq_difference = np.linalg.norm(lo_action_seq - lo_current_actions, axis=1)**2

            lo_neg_sum_sq_diff = -1 * np.sum(lo_sq_difference, axis=0)

            lo_policy_likelihoods.append(lo_neg_sum_sq_diff)

        # find the (approximate) goal that maximises the likelihood of the observed actions
        likeliest_goal = np.argmax(lo_policy_likelihoods)

        return candidate_goals[likeliest_goal]

    def save_model(self, filepath:str):
        self.hi_agent.save_model(filepath + '/hi_agent')
        self.lo_agent.save_model(filepath + '/lo_agent')

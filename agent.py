import numpy as np


class BaseAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def act(self, state, explore=False):
        """
        Generates actions from state
        state.shape should be: (batch, action_space.shape...)
        off-policy correction mechanism will call act() on multiple states,
        so make sure batch sizes > 1 are supported :)
        """
        raise NotImplementedError

    def train(self,
              state,
              action,
              reward: float,
              next_state,
              done: bool,
              relabel=False):
        raise NotImplementedError

    @staticmethod
    def relabel_goal(original_goal,
                     lo_state_seq=None,
                     lo_action_seq=None,
                     lo_current_policy=None):
        """
        a dummy copy of the HiAgent's goal relabeler, which agents can call with no effect
        when running as low level
        """
        return original_goal


class HiAgent(BaseAgent):
    """
    Agents capable of functioning as High-Level agents in the HIRO algorithm should inherit this class
    """

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
        """
        a version of train() with extra arguments required by high-level agents
        for relabelling transition tuples later

        Parameters
        ----------

        lo_state_seq : np.array: (c, *state.shape)
            array containing the c states visited by the LoAgent since the last HiAgent training step
        
        lo_action_seq : np.array: (c, *action.shape)
            array containing the c actions taken by the LoAgent since the last HiAgent training step

        lo_current_policy : func: state -> action
            the act() function of the LoAgent (supplied by a MetaAgent)
        """

        raise NotImplementedError

    @staticmethod
    def relabel_goal(original_goal, lo_state_seq, lo_action_seq,
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

        original_goal : np.array: (*state.shape)


        lo_state_seq : np.array: (c, *state.shape)
            array containing the c states visited by the LoAgent since the last HiAgent training step
        
        lo_action_seq : np.array: (c, *action.shape)
            array containing the c actions taken by the LoAgent since the last HiAgent training step

        lo_current_policy : func: state -> action
            the act() function of the LoAgent (supplied by a MetaAgent)        
        """

        # get variance from observed states
        stdev_goal = np.std(lo_state_seq, axis=0)  #todo - cov instead

        # eight candidate goals sampled randomly from a Gaussian centered at s_t+c − s_t
        # (since goals implemented as states, this just becomes s_t+c)
        n_candidate_goals = 8

        candidate_goals = np.random.normal(
            loc=original_goal,
            scale=stdev_goal,
            size=(n_candidate_goals, *original_goal.shape))

        # also include the original goal gt 
        # and a goal corresponding to the difference st+c − st
        candidate_goals = np.concatenate([
            candidate_goals,
            np.expand_dims(original_goal, axis=0),
            np.expand_dims(lo_state_seq[-1] - lo_state_seq[0], axis=0)
        ],
                                         axis=0)

        lo_policy_likelihoods = []

        # running this as a for loop is not quite optimal -- todo later
        for g in range(n_candidate_goals + 2):

            # transform the (state) c-tuple into a (state, goal) c-tuple
            # shape = (c, 2, *state_shape)
            lo_stategoal_seq = np.stack([
                lo_state_seq,
                np.broadcast_to(candidate_goals[g], lo_state_seq.shape)
            ],
                                        axis=1)

            # what actions would the current LoAgent take, given goal g?
            lo_current_actions = lo_current_policy(lo_stategoal_seq)

            # how far do they diverge from the actual actions, given original goal?
            # shape = (c, *action_shape)
            lo_sq_difference = np.linalg.norm(lo_action_seq -
                                              lo_current_actions)**2

            lo_neg_sum_sq_diff = -1 * np.sum(lo_sq_difference, axis=0)

            lo_policy_likelihoods.append(lo_neg_sum_sq_diff)

        # find the (approximate) goal that maximises the likelihood of the observed actions
        likeliest_goal = np.argmax(lo_policy_likelihoods)

        return candidate_goals[likeliest_goal]

class Pigeon(HiAgent):
    """
    a stupid agent for testing purposes
    """
    def act(self, state, explore=False):

        return np.random.uniform(self.action_space.low, self.action_space.high)

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

        if relabel:
            # just to make sure this method functions as it should
            relabeled_goal = self.relabel_goal(action, lo_state_seq, lo_action_seq, lo_current_policy)

        pass
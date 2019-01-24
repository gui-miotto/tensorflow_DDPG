import numpy as np


class BaseAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def act(self, state, training=False):
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


class HiAgent(BaseAgent):
    def train(self,
              state,
              action,
              reward: float,
              next_state,
              done: bool,
              relabel=True,
              lo_state_seq=None,
              lo_action_seq=None,
              lo_current_policy=None):
        # lo_xx_seq will be a c-tuple of the states seen / actions taken by lo_agent to achieve the goal
        # needed for relabelling transition tuples later

        # at some point, we should have
        #   if relabel: transition_tuple = relabel_goal(...)
        #   else: transition_tuple = ...

        raise NotImplementedError

    def relabel_goal(self, original_goal, lo_state_seq, lo_action_seq,
                     lo_current_policy):
        # this will be used internally by the HiAgent in its train() routine

        # get variance from observed states
        stdev_goal = np.std(lo_state_seq)

        # eight candidate goals sampled randomly from a Gaussian centered at s_t+c − s_t
        # (since goals implemented as states, this just becomes s_t+c)

        candidate_goals = np.random.normal(
            loc=original_goal, scale=stdev_goal, size=8

        # also include the original goal gt and a goal corresponding to the difference st+c − st
        candidate_goals = np.concatenate([
            candidate_goals,
            np.expand_dims(original_goal, axis=0),
            np.expand_dims(lo_state_seq[-1] - lo_state_seq[0], axis=0)
            ], axis=0)

        # find the (approximate) argmax goal to relabel with
        lo_policy_likelihood = lo_current_policy()

        raise NotImplementedError
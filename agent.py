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

    def scale_action(self, action):
        """input: action as a tanh in (-1, 1)"""
        #should work with batches of actions (as required for goal relabelling)
        
        #now need to scale up to cover the action space...
        action = np.multiply(action, (self.action_space.high - self.action_space.low) / 2)

        # ...and translate to center of action space
        action = np.add(action, (self.action_space.high + self.action_space.low) / 2)

        return action

    def train(self,
              state,
              action,
              reward: float,
              next_state,
              done: bool,
              relabel=False):
        raise NotImplementedError



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
from typing import List
from collections import namedtuple, deque
import numpy as np

ReplayBatch = namedtuple('ReplayBatch', ['states_before', 'actions', 'states_after', 'rewards','done_flags'])

ReplayBatchLong = namedtuple('ReplayBatch', ['states_before', 'actions', 'states_after', 'rewards','done_flags', 'lo_state_seqs', 'lo_action_seqs'])


class ReplayBuffer():
    def __init__(self, buffer_size: int=10000, batch_size: int=100, use_long: bool=False):
        """
        Buffer will keep the most recent 'buffer_size' transitions
        Batches given by the function 'sample_batch()' will have length 'batch_size'
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.states_before = []
        self.actions = []
        self.states_after = []
        self.rewards = []
        self.done_flags = []

        self.use_long = use_long

        if self.use_long:
            self.lo_state_seqs = []
            self.lo_action_seqs = []

    def add(self, state_before: List[float], action: int, state_after: List[float], reward: float, done_flag: bool, lo_state_seq=None, lo_action_seq=None):
        """
        Add a new transition to the buffer
        """
        self.states_before.append(state_before)
        self.actions.append(action)
        self.states_after.append(state_after)
        self.rewards.append(reward)
        self.done_flags.append(done_flag)

        if self.use_long:
            assert lo_state_seq is not None
            assert lo_action_seq is not None
            self.lo_state_seqs.append(lo_state_seq)
            self.lo_action_seqs.append(lo_action_seq)

        if len(self.states_before) > self.buffer_size:
            self.states_before.pop(0)
            self.actions.pop(0)
            self.states_after.pop(0)
            self.rewards.pop(0)
            self.done_flags.pop(0)
            if self.use_long:
                self.lo_state_seqs.pop(0)
                self.lo_action_seqs.pop(0)

    def __len__(self):
        """
        Returns how many transitions are currently stored in the buffer
        """
        return len(self.done_flags)

    def sample_batch(self) : #-> ReplayBatch:
        """
        Returns a batch of transtions sampled from the buffer
        """
        # The size of the batch shouldn't be larger than the number of transitions currently stored in the buffer
        b_size = self.batch_size if len(self) > self.batch_size else len(self)
        # Samples are chosen without replacement
        pick = np.random.choice(len(self), size=b_size, replace=False)
        sb = []
        ac = []
        sa = []
        rw = []
        df = []

        for p in pick:
            sb.append(self.states_before[p])
            ac.append(self.actions[p])
            sa.append(self.states_after[p])
            rw.append(self.rewards[p])
            df.append(self.done_flags[p])  

        sb = np.array(sb)
        ac = np.array(ac)
        sa = np.array(sa)
        rw = np.array(rw)
        df = np.array(df)
        
        if self.use_long:
            lss = []
            las = []
        
            for p in pick:
                lss.append(self.lo_state_seqs[p])
                las.append(self.lo_action_seqs[p]) 

            lss = np.array(lss)
            las = np.array(las)

            return ReplayBatchLong(states_before=sb, actions=ac, states_after=sa, rewards=rw, done_flags=df, lo_state_seqs=lss, lo_action_seqs=las)
        
        return ReplayBatch(states_before=sb, actions=ac, states_after=sa, rewards=rw, done_flags=df)




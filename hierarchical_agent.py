from agent import BaseAgent

class HierarchicalAgent(BaseAgent):
    def __init__(self, state_space, action_space):
        # self.state_space = state_space
        # self.action_space = action_space
        super().__init__(state_space, action_space)
    
    def act(self, state):
        raise NotImplementedError

    def train(self, state, action, reward: float, next_state, done: bool):
        raise NotImplementedError
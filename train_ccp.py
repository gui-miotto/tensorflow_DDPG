import sys
from continuous_cartpole import ContinuousCartPoleEnv
import numpy as np
from ddpg_agent.ddpg_agent import DDPGAgent

if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = ContinuousCartPoleEnv() # todo gym.make('CartPole-v1')
    # get size of state and action from environment
    
    render = True

    # make agent
    agent = DDPGAgent(env.observation_space, env.action_space)

    EPISODES = 1000 # todo

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
    
        while not done:
            if render:
                env.render()

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.train(state, action, reward, next_state, done)

            score += reward
            state = next_state
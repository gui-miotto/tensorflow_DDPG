import sys
from continuous_cartpole import ContinuousCartPoleEnv
import numpy as np
import gym
from ddpg_agent.ddpg_agent import DDPGAgent

if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = ContinuousCartPoleEnv() 
    # or
    # env = gym.make('CartPole-v1')
    # not compatible though - todo
    
    render = False

    # make agent
    agent = DDPGAgent(env.observation_space, env.action_space)

    EPISODES = 1000 # todo

    for e in range(EPISODES):
        done = False
        score = 0
        loss_sum = 0
        steps = 0
        state = env.reset()
    
        while not done:
            if render:
                env.render()

            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            # reward shaping ;-)
            # reward_shaping = np.abs(next_state[2]-np.pi)/np.pi/10
            # new_reward = reward_shaping if reward == 1 else reward+reward_shaping
            
            loss = agent.train(state, action, reward, next_state, done)

            steps += 1
            loss_sum += loss
            score += reward
            state = next_state
        
        print("Episode", e, "score", score, steps, loss_sum / steps, agent.scale)
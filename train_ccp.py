import select, sys, gym, os, time
from datetime import timedelta
import numpy as np
from continuous_cartpole import ContinuousCartPoleEnv
from ddpg_agent.ddpg_agent import DDPGAgent

def test_agent(n_episodes: int=10, render: bool=True):
    env = ContinuousCartPoleEnv() 
    # load agent
    agent = DDPGAgent.load_pretrained_agent(
        filepath=saved_models_dir,
        state_space=env.observation_space, 
        action_space = env.action_space)
    for ep in range(n_episodes):
        score, steps, done = 0, 0, False
        state = env.reset()
        for steps in range(max_steps_per_ep):
            if render:
                env.render()
            action = agent.act(state, explore=False)
            state, reward, done, info = env.step(action)
            steps += 1
            score += reward
            if done:
                break
        print(f'Episode {ep} of {n_episodes}. score: {score}, steps: {steps}')
    

def train_agent(n_episodes: int=1000, render: bool=False):
    env = ContinuousCartPoleEnv() 
    # todo: not compatible with 'CartPole-v1' 

    # create new naive agent
    agent = DDPGAgent.new_trainable_agent(
        state_space=env.observation_space, 
        action_space = env.action_space)

    total_steps, ep = 0, 0
    time_begin = time.time()

    while ep < n_episodes:
        steps, score, loss_sum, done = 0, 0, 0, False
        state = env.reset()
        ep += 1

        while not done and steps < max_steps_per_ep:
            if render:
                env.render()

            steps += 1
            action = agent.act(state, explore=True)
            next_state, reward, done, _ = env.step(action)

            # reward shaping ;-)
            # reward_shaping = np.abs(next_state[2]-np.pi)/np.pi/10
            # new_reward = reward_shaping if reward == 1 else reward+reward_shaping

            if steps >= max_steps_per_ep:
                reward -= 1

            loss = agent.train(state, action, reward, next_state, done)
            loss_sum += loss
            score += reward
            state = next_state

            # check user keyboard commands
            while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip()
                # 'r' will toggle the render flag
                if line == 'r':
                    render = not render
                # 'q' will save the models and and training
                elif line == 'q':
                    agent.save_model(saved_models_dir)
                    return
                # 'm' for more episodes 
                elif line == 'm':
                    n_episodes += 100
                # 'l' for less episodes 
                elif line == 'l':
                    n_episodes -= 100
                # 'i' will increase the exploration factor
                elif line == 'i':
                    agent.stdev_explore += 0.1
                # 'd' will decrease the exploration factor
                elif line == 'd':
                    agent.stdev_explore -= 0.1
                # 'z' will zero the exploration factor
                elif line == 'z':
                    agent.stdev_explore = 0.0
                # an empty line means stdin has been closed
                else: 
                    print('eof')
                    #exit(0)
        
        total_steps += steps
        print(f'Episode {ep:4d} of {n_episodes}, score: {score:4d}, steps: {steps:4d}, ' 
            + f'average loss: {loss_sum/steps:.5f}, exploration: {agent.stdev_explore:6f}')
        
        

    #print time statistics 
    time_end = time.time()
    elapsed = time_end - time_begin
    print('\nElapsed time:', str(timedelta(seconds=elapsed)))
    print(f'Steps per second: {(total_steps / elapsed):.3f}\n')

    agent.save_model(saved_models_dir)


if __name__ == "__main__":
    # global settings
    saved_models_dir = './saved_models'
    max_steps_per_ep = 2000

    train_agent(n_episodes=10, render=False)
    test_agent()

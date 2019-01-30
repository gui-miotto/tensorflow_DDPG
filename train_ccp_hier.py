import select, sys, gym, os, time
from datetime import timedelta
import numpy as np
from continuous_cartpole import ContinuousCartPoleEnv
from ddpg_agent.ddpg_agent import DDPGAgent
from ddpg_agent.dummy_agent import DummyAgent

from meta_agent import MetaAgent

def test_agent(n_episodes: int=10, render: bool=True):
    env = ContinuousCartPoleEnv() 
    # load agent
    agent = MetaAgent(
        models_dir=saved_models_dir,
        state_space=env.observation_space, 
        action_space = env.action_space, #TODO clipping
        hi_agent_cls=DDPGAgent, 
        lo_agent_cls=DDPGAgent)

    for ep in range(n_episodes):
        score, steps, done = 0, 0, False
        state = env.reset()

        goal_state = np.squeeze(state)
        agent.reset_clock()

        for steps in range(max_steps_per_ep):
            if render:
                env.render(goal_state=goal_state)
            action = agent.act(state, explore=False)
            goal_state = np.squeeze(agent.goal)
            state, reward, done, _ = env.step(np.squeeze(action, axis=0))

            steps += 1
            score += reward
            if done:
                break
        print(f'Episode {ep} of {n_episodes}. score: {score}, steps: {steps}')
    

def train_agent(n_episodes: int=1000, render: bool=True):
    env = ContinuousCartPoleEnv() 

    agent = MetaAgent(
        state_space=env.observation_space,
        action_space=env.action_space, 
        hi_agent_cls=DDPGAgent, 
        lo_agent_cls=DDPGAgent)

    total_steps, ep = 0, 0
    time_begin = time.time()

    while ep < n_episodes:
        steps, hi_steps, score, done = 0, 0, 0, False
        lo_rewards = []
        loss_sum = np.array([0.,0.])
        state = env.reset()
        agent.reset_clock()

        ep += 1

        goal_state = np.squeeze(state)

        while not done and steps < max_steps_per_ep:
            if render:
                env.render(goal_state=goal_state)

            steps += 1
            action = agent.act(state, explore=True)
            
            goal_state = np.squeeze(agent.goal)

            next_state, reward, done, _ = env.step(np.squeeze(action, axis=0))

            if steps >= max_steps_per_ep:
                reward -= 1

            lo_loss, hi_loss, lo_rew = agent.train(state, action, reward, next_state, done)
            lo_rewards.append(lo_rew)

            if hi_loss is not None:
                hi_steps += 1
                loss_sum[0] += (1 / hi_steps) * (hi_loss - loss_sum[0]) # avoids need to divide by num steps at end
            
            # lo_loss
            loss_sum[1] += (1 / steps) * (lo_loss - loss_sum[1]) # avoids need to divide by num steps at end
            
            score += reward
            state = next_state

            if os.name != 'nt':
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
                        agent.lo_agent.epslon_greedy += 0.1
                    # 'd' will decrease the exploration factor
                    elif line == 'd':
                        agent.lo_agent.epslon_greedy -= 0.1
                    # 'z' will zero the exploration factor
                    elif line == 'z':
                        agent.lo_agent.epslon_greedy = 0.0
                    # an empty line means stdin has been closed
                    else: 
                        print('eof')
                        #exit(0)
        
        total_steps += steps
        print(f'Episode {ep:4d} of {n_episodes}, score: {score:4d}, steps: {steps:4d}, lo_reward: {np.mean(lo_rewards):.4f}, ' 
            + f'hi_loss : {loss_sum[0]:.3f}, lo_loss : {loss_sum[1]:.3f}, hi_expl: {agent.hi_agent.epslon_greedy:6f}, '
            + f'lo_expl: {agent.lo_agent.epslon_greedy:6f}')
        

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

    train_agent(n_episodes=1000, render=False)
    test_agent()

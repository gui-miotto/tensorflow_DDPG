import select, sys, gym, os, time
from datetime import timedelta
import numpy as np
import argparse

from continuous_cartpole import ContinuousCartPoleEnv
from bipedal_walker import BipedalWalker

from ddpg_agent.ddpg_agent import DDPGAgent
from ddpg_agent.dummy_agent import DummyAgent
from meta_agent import MetaAgent
from tensorboard_evaluation import Evaluation

def ensure_path(p):
    if not os.path.exists(p):
        os.mkdir(p)

def test_agent(n_episodes: int=10, render: bool=True):
    env = ContinuousCartPoleEnv() if not COMPLEXENV else BipedalWalker()
    # load agent
    if not HIERARCHY:
        agent = DDPGAgent.load_pretrained_agent(
            filepath=saved_models_dir,
            state_space=env.observation_space,
            action_space = env.action_space)
    else:
        if not COMPLEXENV:
            hi_action_space = gym.spaces.Box(
                low=[-2.4, -3, -np.pi, -10],
                high=[2.4, 3, np.pi, 10],
                dtype=state_space.dtype)
        else:
            hi_action_space = None

        agent = MetaAgent(
            env.observation_space,
            env.action_space,
            hi_agent_cls=DDPGAgent,
            lo_agent_cls=DDPGAgent,
            hi_action_space=hi_action_space)

    for ep in range(n_episodes):
        score, steps, done = 0, 0, False
        state = env.reset()

        goal_state = np.squeeze(state)

        if HIERARCHY:
            agent.reset_clock()

        for steps in range(max_steps_per_ep):
            if render:
                if not HIERARCHY:
                    env.render()
                else:
                    env.render(goal_state=goal_state)

            action = agent.act(state, explore=False)

            if HIERARCHY:
                goal_state = np.squeeze(state + agent.goal)

            scaled_action = agent.scale_action(action)
            state, reward, done, _ = env.step(np.squeeze(scaled_action, axis=0))

            steps += 1
            score += reward
            if done:
                break
        print(f'Episode {ep} of {n_episodes}. score: {score}, steps: {steps}')


def train_agent(n_episodes: int=1000, render: bool=True):
    env = ContinuousCartPoleEnv() if not COMPLEXENV else BipedalWalker()
    tensorboard_path = os.path.join(".", "tensorboard")
    ensure_path(tensorboard_path)
    tensorboard_path = os.path.join(tensorboard_path, NAME)
    ensure_path(tensorboard_path)

    # we need to create the Tensorboard network BEFORE the agent, otherwise it gets angry
    # this could be done better, but oh well.

    train_dict_keys = ["score", "loss", "expl"] if not HIERARCHY else [
        "hi_score", "hi_loss", "hi_expl", "lo_score", "lo_loss", "lo_expl"
    ]
    tensorboard = Evaluation(tensorboard_path, train_dict_keys)

    if not HIERARCHY:
        # create new naive agent
        agent = DDPGAgent.new_trainable_agent(
        state_space=env.observation_space,
        action_space = env.action_space,
        epslon_greedy=0.7,
        exploration_decay=0.99999
        )
    else:
        if not COMPLEXENV:
            hi_action_space = gym.spaces.Box(
                low=np.array([-2.4, -3, -np.pi, -10]),
                high=np.array([2.4, 3, np.pi, 10]),
                dtype=env.observation_space.dtype)
        else:
            hi_action_space = None

        agent = MetaAgent(
            env.observation_space,
            env.action_space,
            hi_agent_cls=DDPGAgent,
            lo_agent_cls=DDPGAgent,
            hi_action_space=hi_action_space)


    total_steps, ep = 0, 0
    time_begin = time.time()

    while ep < n_episodes:
        steps, hi_steps, score, lo_score, done, lo_loss_sum, hi_loss_sum = 0, 0, 0, 0, False, 0, 0
        state = env.reset()
        if HIERARCHY:
            agent.reset_clock()

        ep += 1

        if HIERARCHY:
            goal_state = np.squeeze(state)

        while not done and steps < max_steps_per_ep:
            if render:
                if not HIERARCHY:
                    env.render()
                else:
                    env.render(goal_state=goal_state)

            steps += 1
            action = agent.act(state, explore=True)

            if HIERARCHY:
                goal_state = np.squeeze(state + agent.goal)

            scaled_action = agent.scale_action(action)
            next_state, reward, done, _ = env.step(np.squeeze(scaled_action, axis=0))

            # reward shaping ;-)
            # reward_shaping = np.abs(next_state[2]-np.pi)/np.pi/10
            # new_reward = reward_shaping if reward == 1 else reward+reward_shaping

            if steps >= max_steps_per_ep:
                reward -= 1

            lo_loss, hi_loss = agent.train(state, action, reward, next_state, done)
            # this is the single loss if DDPG, or the lo_loss if hierarchical
            lo_loss_sum += (1 / steps) * (lo_loss - lo_loss_sum) # avoids need to divide by num steps at end

            if hi_loss is not None:
                hi_steps += 1
                hi_loss_sum += (1 / hi_steps) * (hi_loss - hi_loss_sum) # avoids need to divide by num steps at end

            score += reward
            state = next_state

            if HIERARCHY:
                lo_score += agent.lo_reward

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
                        agent.modify_epslon_greedy(0.1, mode='increment')
                    # 'd' will decrease the exploration factor
                    elif line == 'd':
                        agent.modify_epslon_greedy(-0.1, mode='increment')
                    # 'z' will zero the exploration factor
                    elif line == 'z':
                        agent.modify_epslon_greedy(0.0, mode='assign')
                    # an empty line means stdin has been closed
                    else:
                        print('eof')
                        #exit(0)

        total_steps += steps

        if not HIERARCHY:
            print(f'Episode {ep:4d} of {n_episodes}, score: {score:4f}, steps: {steps:4d}, '
                + f'loss: {lo_loss_sum:.3f}, '
                + f'expl: {agent.epslon_greedy:6f}'
                )

            tensorboard.write_episode_data(ep, eval_dict={"score": score,
                                                        "loss": lo_loss_sum,
                                                        "expl": agent.epslon_greedy
                                                        })

        else:
            print(f'Episode {ep:4d} of {n_episodes}, score: {score:4f}, steps: {steps:4d}, '
                + f'lo_loss: {lo_loss_sum:.3f}, '
                + f'hi_loss: {hi_loss_sum:.3f}, '
                + f'lo_expl: {agent.lo_agent.epslon_greedy:6f}, '
                + f'hi_expl: {agent.hi_agent.epslon_greedy:6f}'
                )

            tensorboard.write_episode_data(ep, eval_dict={"hi_score": score,
                                                        "hi_loss": hi_loss_sum,
                                                        "hi_expl": agent.hi_agent.epslon_greedy,
                                                        "lo_score": lo_score,
                                                        "lo_loss": lo_loss_sum,
                                                        "lo_expl": agent.lo_agent.epslon_greedy,
                                                        })

        if ep % 100 == 0:
            agent.save_model(saved_models_dir)

    #print time statistics
    time_end = time.time()
    elapsed = time_end - time_begin
    print('\nElapsed time:', str(timedelta(seconds=elapsed)))
    print(f'Steps per second: {(total_steps / elapsed):.3f}\n')

    agent.save_model(saved_models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default="default",
        type=str,
        help="(directory name under ./racecar/ of trained model to retrieve (or ALL)"
    )
    parser.add_argument(
        "--eps",
        default=2000,
        type=int,
        help="number of episodes to train for"
    )
    parser.add_argument("--hier", action="store_true", default=True, help="Run Hierarchical (rather than DDPG)")
    parser.add_argument("--walker", action="store_true", default=False, help="Run Bipedal Walker (rather than CCP)")
    parser.add_argument("--render", action="store_true", default=False, help="show window")
    args = parser.parse_args()

    # global settings
    NAME = args.name
    COMPLEXENV = args.walker
    HIERARCHY = args.hier
    RENDER = args.render

    print(args)
    #override here for ease of testing
    # COMPLEXENV = True
    HIERARCHY = True
    RENDER = True

    saved_models_dir = os.path.join('.','saved_models')
    ensure_path(saved_models_dir)
    saved_models_dir = os.path.join(saved_models_dir, NAME)
    ensure_path(saved_models_dir)

    max_steps_per_ep = 2000

    train_agent(n_episodes=args.eps, render=RENDER)
    test_agent()

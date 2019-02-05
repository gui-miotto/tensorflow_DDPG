# Training the agent

 ### Single DDPG / Hierarchical DDPG on Continuous Cartpole / Bipedal walker

```
python3 train_gen.py [-h] [--name NAME] [--steps STEPS] [--hier] [--walker]
                    [--render]

optional arguments:
  -h, --help     show this help message and exit
  --name NAME    sets the folder name under which mode/tboard files will be
                 saved
  --steps STEPS  number of steps to train for
  --hier         Run Hierarchical (rather than DDPG)
  --walker       Run Bipedal Walker (rather than CCP)
  --render       show window
  ```

  ### Single DDPG on Mujoco Ant

  ```
  python3 train_ant.py
  ```

# Testing the agent

The `train_gen.py` and `train_ant.py` files contain `test_agent()` methods that can be called to perform testing. By default, agents are tested for 10 episodes after training, with scores recorded.

# Guide to the code

|File|Description|
|----|-----------|
`train_gen.py` `train_ant.py` | Main training routines
`agent.py` | Defines interface for agents
`ddpg_agent.py` | Implementation of Deep Deterministic Policy Gradient agent
`ou_noise.py` | Implementation of Ornstein-Uhlenbeck noise (optionally used by DDPG agent)
`replay_buffer.py` | Yep, it's a replay buffer
`meta_agent.py` | Implementation of Hierarchical Reinforcement Learning functions, and organisation of messages between environment, high-, and low-level agents
`continuous_cartpole.py` | Environment #1, with some modifications (courtesy of OpenAI Gym)
`bipedal_walker.py` | Environment #2, with some modifications (courtesy of OpenAI Gym)

import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec

from print_versions import printVersions

import LLImage

import tensorflow as tf
import stable_baselines
printVersions( ["Python", tf, np, gym, stable_baselines] )

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


def test_env( model, env, env_name, render=False, count=1000 ):
    obs = env.reset()
    reward_vec = []
    episodes = 0
    #name = env.unwrapped.get_attr( "venv" ).spec.id
    name = env_name # This is stupid. There should be some way to get the name from the multiply-wrapped env
    for i in range(count):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_vec.append(rewards)
        episodes += np.sum(dones)
        if render:
            env.render()
    print( "{} mean rewards/episodes: {}\t{}".format( name, np.mean(reward_vec), episodes))

def make_atari_single_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, allow_early_resets=True):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    :param env_id: (str) the environment ID
    :param num_env: (int) IGNORED
    :param seed: (int) the inital seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])

num_procs = 1
random_seed = 0
env_names = ["LunarLanderImageContinuous-v2", "CarRacing-v0"]
envs = []
for env_name in env_names:
    envs.append( make_atari_env(env_name, num_env=num_procs, seed=random_seed, wrapper_kwargs={"scale":True}) )

param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)
model = DDPG(CnnPolicy, envs[0], param_noise=param_noise, memory_limit=int(1e6), verbose=0)

epochs = 10
steps_per_epoch = 10000
for epoch in range(epochs):
    for idx, env in enumerate(envs):
        model.set_env(env)
        print( "training {} for {} steps".format( env_names[idx], steps_per_epoch) )
        model.learn(total_timesteps=steps_per_epoch)
        #model.save(model_path)

    for idx, env in enumerate(envs):
        test_env( model, env, env_names[idx] )

"""
# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('LunarLanderContinuous-v2')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)
model = DDPG(MlpPolicy, env, param_noise=param_noise, memory_limit=int(1e6), verbose=0)
# Train the agent
model.learn(total_timesteps=200000, callback=callback)
"""

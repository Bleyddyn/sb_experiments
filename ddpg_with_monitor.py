import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind, NoopResetEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines import logger
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG, PPO2
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec

from print_versions import printVersions

import LLImage
from custom_class import CustomMalpiPolicy, DDPGDKPolicy

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
  log_dir = logger.get_dir()
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      print( "Log dir: {}".format( log_dir ) )
      print( "x/y".format( len(x), type(y) ) )
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              fname = os.path.join(log_dir, 'best_model.pkl')
              print("Saving new best model to {}".format(fname))
              _locals['self'].save( fname )

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

class WrapCarRacing(gym.Wrapper):
    def __init__(self, env):
        """
        Wrap the CarRacing environment action space so it is symmetric and will work with DDPG.

        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def step(self, action):
        action[1:] = (action[1:] + 1.0) / 2.0
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

def wrap_deepmind_fixed(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """
    Configure environment for DeepMind-style Atari.
    :param env: (Gym Environment) the atari environment
    :param episode_life: (bool) wrap the episode life wrapper
    :param clip_rewards: (bool) wrap the reward clipping wrapper
    :param frame_stack: (bool) wrap the frame stacking wrapper
    :param scale: (bool) wrap the scaling observation wrapper
    :return: (Gym Environment) the wrapped atari environment
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if hasattr(env.unwrapped, 'get_action_meanings') and 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def make_env_simplified( env_id, seed, wrapper_kwargs=None, allow_early_resets=True):
    env = gym.make(env_id)
    #env = NoopResetEnv(env, noop_max=30)
    #env = MaxAndSkipEnv(env, skip=4)
    env.seed(seed)
    env = Monitor(env, logger.get_dir(), allow_early_resets=allow_early_resets)
    return wrap_deepmind_fixed(env, episode_life=False, **wrapper_kwargs)

def test_ll(env_name, random_seed, saved_model=None, total_steps=200000):
# Create and wrap the environment
    env = make_env_simplified(env_name, seed=random_seed, wrapper_kwargs={"scale":True})
    if "CarRacing-v0" == env_name:
        print( "Wrapping" )
        env = WrapCarRacing(env)
    env = DummyVecEnv([lambda: env])

    ac_space = env.action_space
    print( "Environment: " )
    print( "  obs space: {}".format( env.observation_space ) )
    print( "  act space: {} {}-{}".format( ac_space, ac_space.low, ac_space.high ) )

    assert isinstance(ac_space, gym.spaces.Box), "Error: the action space must be of type gym.spaces.Box"
    assert (np.abs(ac_space.low) == ac_space.high).all(), "Error: the action space low and high must be symmetric"

    tb_dir = os.path.join( logger.get_dir(), "td/" )

# Add some param noise for exploration
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)
    model = DDPG(CnnPolicy, env, param_noise=param_noise, memory_limit=int(2e5), verbose=0, tensorboard_log=tb_dir)
    if saved_model is not None:
        model = DDPG.load(saved_model, tensorboard_log=tb_dir) #this is how it's done in the docs?
        #model.load(saved_model, tensorboard_log=tb_dir)

# Train the agent
    model.learn(total_timesteps=total_steps, callback=callback)

def test_ppo(env_name, random_seed, num_env=2, saved_model=None, total_steps=200000):
# Create and wrap the environment
    env = make_atari_env(env_name, 2, random_seed)

    #env = make_env_simplified(env_name, seed=random_seed, wrapper_kwargs={"scale":True})
    if "CarRacing-v0" == env_name:
        print( "Wrapping" )
        env = WrapCarRacing(env)
    #env = DummyVecEnv([lambda: env])

    ac_space = env.action_space
    print( "Environment: " )
    print( "  obs space: {}".format( env.observation_space ) )
    #print( "  act space: {} {}-{}".format( ac_space, ac_space.low, ac_space.high ) )

#    assert isinstance(ac_space, gym.spaces.Box), "Error: the action space must be of type gym.spaces.Box"
#    assert (np.abs(ac_space.low) == ac_space.high).all(), "Error: the action space low and high must be symmetric"

    tb_dir = os.path.join( logger.get_dir(), "tb/" )

# Add some param noise for exploration
    model = PPO2(stable_baselines.common.policies.CnnPolicy, env, verbose=0, tensorboard_log=tb_dir)

# Train the agent
    model.learn(total_timesteps=total_steps, callback=callback)

log_dir = None
num_procs = 1
random_seed = 0
env_names = ["LunarLanderImageContinuous-v2", "CarRacing-v0"]
env_name = "BreakoutNoFrameskip-v4" #env_names[1]
saved_model = os.path.join( env_name, "best_model.pkl" )
if not os.path.exists(saved_model):
    saved_model = None

logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv','tensorboard'])
log_dir = logger.get_dir()

#test_ll(env_name, random_seed, saved_model=saved_model, total_steps = 200000)
test_ppo(env_name, random_seed, saved_model=None, total_steps = 2000000)
exit()

envs = []
for env_name in env_names:
    envs.append( make_env_simplified(env_name, seed=random_seed, wrapper_kwargs={"scale":True}) )

param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)
model = DDPG(DDPGDKPolicy, envs[0], param_noise=param_noise, memory_limit=int(1e5), verbose=0)

epochs = 10
steps_per_epoch = 10000
for epoch in range(epochs):
    for idx, env in enumerate(envs):
        model.set_env(env)
        print( "training {} for {} steps".format( env_names[idx], steps_per_epoch) )
        model.learn(total_timesteps=steps_per_epoch, callback=callback)
        #model.save(model_path)

    for idx, env in enumerate(envs):
        test_env( model, env, env_names[idx] )


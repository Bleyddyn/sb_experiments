import datetime
import gym
import LLImage
from matplotlib import pyplot as plt
import numpy as np

def show_obs( obs ):
    plt.imshow(obs, interpolation='nearest')
    plt.show()

def show_mult_obs( obses ):
    fig = plt.figure(figsize=(16,10))
    row = 1
    for obs in obses:
        fig.add_subplot(len(obses), 1, row)
        row += 1
        plt.imshow(obs, interpolation='nearest')
    plt.show()

def test(env, render=True):
    env.reset()
    obses = []
    rewards = []
    ep_rew = []
    for i in range(1000):
        obs, reward, done, info = env.step( env.action_space.sample() )
        ep_rew.append( reward )
        if done:
            rewards.append( np.sum(ep_rew) )
            ep_rew = []
            env.reset()
        if (30 == i) or (60 == i) or (90 == i):
            print( "Append obs: {}".format(i) )
            obses.append( obs.copy() )
        if render:
            env.render()
    return obses, rewards

#env = gym.envs.make("LunarLanderContinuous-v2")
env = gym.envs.make("LunarLanderImageContinuous-v2")

start = datetime.datetime.now()
obs, rew = test(env, render=False)
elapsed = datetime.datetime.now() - start
print( "Elapsed: {}".format( elapsed ) )
print( "{} frames per second".format( 1000.0 / elapsed.total_seconds() ) )
print( "Reward: {}".format( np.mean(rew[-100:]) ) )

#show_mult_obs( obs )
plt.plot(rew[-100:])
plt.show()

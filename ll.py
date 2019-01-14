import gym
import LLImage
from matplotlib import pyplot as plt

def show_obs( obs ):
    plt.imshow(obs, interpolation='nearest')
    plt.show()

#env = gym.envs.make("LunarLanderContinuous-v2")
env = gym.envs.make("LunarLanderImageContinuous-v2")

env.reset()
#rgb = env.render(mode="rgb_array")
obses = []
for i in range(1000):
    obs, reward, done, info = env.step( env.action_space.sample() )
    if done:
        break
    if (30 == i) or (60 == i) or (90 == i):
        print( "Append obs: {}".format(i) )
        obses.append( obs.copy() )
    env.render()
    #rgb = env.render(mode="rgb_array")
#print( rgb.shape )
print( obs.shape )

fig = plt.figure(figsize=(16,10))
row = 1
for obs in obses:
    fig.add_subplot(len(obses), 1, row)
    row += 1
    plt.imshow(obs, interpolation='nearest')
plt.show()

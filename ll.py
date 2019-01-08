import gym
import LLImage

#env = gym.envs.make("LunarLanderContinuous-v2")
env = gym.envs.make("LunarLanderImageContinuous-v2")

env.reset()
#rgb = env.render(mode="rgb_array")

for i in range(1000):
    obs, reward, done, info = env.step( env.action_space.sample() )
    if done:
        break
    env.render()
    #rgb = env.render(mode="rgb_array")
#print( rgb.shape )
print( obs.shape )

#from matplotlib import pyplot as plt
#plt.imshow(obs, interpolation='nearest')
#plt.show()

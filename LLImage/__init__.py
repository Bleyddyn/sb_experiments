from gym.envs.registration import registry, register, make, spec

register(
    id='LunarLanderImage-v2',
    entry_point='LLImage.LunarLanderImage:LunarLanderImage',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderImageContinuous-v2',
    entry_point='LLImage.LunarLanderImage:LunarLanderImageContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

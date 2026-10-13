from gym.envs.registration import register

register(
    id="MO-Hopper-v2",
    entry_point="environments.hopper:HopperEnv",
    max_episode_steps=500,
)

register(
    id="MO-Swimmer-v2",
    entry_point="environments.swimmer:SwimmerEnv",
    max_episode_steps=500,
)

import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
) # continuous: LunarLanderContinuous-v3
env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 10

for ep in range(episodes):
	obs, info = env.reset()
	terminated = False
	truncated = False
	while not terminated and not truncated:
		action, _states = model.predict(obs)
		obs, rewards, terminated, truncated, info = env.step(action)
		env.render()
		print(rewards)
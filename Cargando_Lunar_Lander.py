import gymnasium as gym
from stable_baselines3 import PPO

# Crear entorno con ventana
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

# Cargar modelo guardado
model = PPO.load("models/PPO/500000.zip")

# Iniciar episodio
obs, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

env.close()
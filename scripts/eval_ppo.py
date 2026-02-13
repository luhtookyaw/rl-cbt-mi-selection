# scripts/eval_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.therapy_env import TherapyEnv
from src.envs.obs_wrappers import LastClientEmbeddingWrapper

def make_env():
    env = TherapyEnv(
        sparse_critic=True,   # more realistic evaluation if you want
    )
    env = LastClientEmbeddingWrapper(env)
    return env

def main():
    vec_env = DummyVecEnv([make_env])
    model = PPO.load("outputs/ppo_therapy_router", env=vec_env)

    obs = vec_env.reset()
    done = False

    # Run one episode
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        done = bool(dones[0])

        info = infos[0]
        print("Action:", info.get("action_id"), "Reward:", reward[0], "Trust:", info.get("trust_level"))
        # Uncomment if you want to see text each turn:
        # print("Therapist:", info.get("therapist_last"))
        # print("Client:", info.get("client_last"))
        # print("-"*40)

if __name__ == "__main__":
    main()

# scripts/train_ppo.py
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.envs.therapy_env import TherapyEnv
from src.envs.obs_wrappers import LastClientEmbeddingWrapper

from scripts.callbacks import PrintStepCallback, TherapyTensorboardCallback

MODEL_SAVE_PATH = "outputs/ppo_therapy_router"


def make_env():
    env = TherapyEnv(sparse_critic=False)
    env = LastClientEmbeddingWrapper(env)
    env = Monitor(env, "outputs/monitor.csv")  # logs episode rewards/lengths
    return env


def main():
    vec_env = DummyVecEnv([make_env])  # single env; keep simple first

    model_zip_path = Path(f"{MODEL_SAVE_PATH}.zip")
    if model_zip_path.exists():
        print(f"Loading existing model from {model_zip_path} and continuing training...")
        model = PPO.load(str(model_zip_path), env=vec_env, device="cpu")
    else:
        print("No existing model found. Training a new model from scratch...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
            ),
            device="cpu",
            verbose=1,
            n_steps=32,          # keep small: each step triggers multiple LLM calls
            batch_size=32,
            gamma=0.95,          # dialogues benefit from slightly shorter horizon
            learning_rate=3e-4,
            ent_coef=0.01,       # helps exploration
            clip_range=0.2,
            tensorboard_log="outputs/tb",
        )

    # Train
    model.learn(
        total_timesteps=512, 
        progress_bar=True,
        tb_log_name="ppo_router_run",
        callback=TherapyTensorboardCallback(),
    )  
    
    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")

if __name__ == "__main__":
    main()

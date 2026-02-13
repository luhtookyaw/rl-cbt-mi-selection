from pathlib import Path
from src.envs.therapy_env import TherapyEnv
from src.envs.obs_wrappers import LastClientEmbeddingWrapper

DEFAULT_TAXONOMY_PATH = Path("data/interventions_taxonomy.json")

env = TherapyEnv(
    sparse_critic=False,  # easier to see reward changes each step
    fixed_patient_id="1-1"
)

# env = LastClientEmbeddingWrapper(env)

obs, info = env.reset()
print("RESET:", info)
print(obs)
env.render()

for _ in range(3):
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    print("\nACTION:", info["action_id"], ", ", "REWARD:", reward, ", ", "DONE:", terminated)
    if terminated or truncated:
        break


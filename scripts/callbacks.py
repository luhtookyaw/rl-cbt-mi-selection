# scripts/callbacks.py
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class PrintStepCallback(BaseCallback):
    def __init__(self, print_every_steps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.print_every_steps = max(1, int(print_every_steps))

    def _on_step(self) -> bool:
        # infos is a list (one per env). With DummyVecEnv it's length 1.
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if self.n_calls % self.print_every_steps == 0 and infos:
            info = infos[0]
            r = float(rewards[0]) if rewards is not None else None
            d = bool(dones[0]) if dones is not None else False

            print("\n" + "=" * 80)
            print(f"Step {self.num_timesteps} | Reward: {r} | Done: {d}")
            print(f"Turn: {info.get('turn')} | Phase: {info.get('phase')} | Trust: {info.get('trust_level')}")
            print(f"Action: {info.get('action_id')}")
            print("- Therapist:", (info.get("therapist_last") or "").strip())
            print("- Client   :", (info.get("client_last") or "").strip())
            print("=" * 80)

        return True

class TherapyTensorboardCallback(BaseCallback):
    """
    Custom TensorBoard logging for therapy RL environment.
    Logs trust, phase, rewards, actions, episode stats.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", [])

        # ---- Log step reward
        if len(rewards):
            self.logger.record("env/step_reward", float(rewards[0]))

        # ---- Log actions histogram
        if len(actions):
            self.logger.record("env/action", int(actions[0]))

        # ---- Extract info dict from env
        if infos:
            info = infos[0]

            if "trust_level" in info:
                self.logger.record("env/trust_level", info["trust_level"])

            if "phase" in info:
                # numeric or string phase supported
                phase = info["phase"]
                if isinstance(phase, str):
                    # optional mapping if needed
                    phase_map = {
                        "assessment": 0,
                        "conceptualization": 1,
                        "solution_exploration": 2,
                        "closure": 3,
                    }
                    phase = phase_map.get(phase, -1)

                self.logger.record("env/phase", phase)

            if "end_session" in info:
                self.logger.record("env/end_session", int(info["end_session"]))

        return True

    def _on_rollout_end(self):
        # Flush to tensorboard
        self.logger.dump(self.num_timesteps)

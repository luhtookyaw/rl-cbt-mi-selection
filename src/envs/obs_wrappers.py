# src/envs/obs_wrappers.py
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.embeddings import embed_text

def extract_last_client_utterance(dialogue_text: str) -> str:
    """
    dialogue_text looks like:
      Therapist: ...
      Client: ...
    We'll return the last Client: line content.
    """
    if not dialogue_text:
        return ""
    lines = [ln.strip() for ln in dialogue_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("Client:"):
            return ln[len("Client:"):].strip()
    # fallback: return last line if no prefix found
    return lines[-1] if lines else ""

class LastClientEmbeddingWrapper(gym.ObservationWrapper):
    """
    Converts the env's Dict observation (with dialogue_text) into a numeric Box:
      [embedding_dim floats..., phase_idx_norm, trust_norm, turn_norm]

    Note: This calls the embeddings API every step (costly).
    Add caching if you repeat states a lot.
    """
    def __init__(
        self,
        env: gym.Env,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,   # text-embedding-3-small = 1536
        add_numeric_features: bool = True,
    ):
        super().__init__(env)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.add_numeric_features = add_numeric_features

        extra = 3 if add_numeric_features else 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(embedding_dim + extra,),
            dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        dialogue_text = obs.get("dialogue_text", "")
        last_client = extract_last_client_utterance(dialogue_text)

        vec = embed_text(last_client, model=self.embedding_model)
        v = np.asarray(vec, dtype=np.float32)

        if v.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: got {v.shape[0]} expected {self.embedding_dim}")

        if not self.add_numeric_features:
            return v

        # simple normalization to keep scales reasonable
        phase_idx = float(obs.get("phase_idx", 0)) / 2.0       # 0..2 -> 0..1
        trust = float(obs.get("trust_level", 1)) / 5.0         # 1..5 -> 0.2..1
        turn = float(obs.get("turn", 0)) / max(1.0, float(self.env.max_turns))  # 0..1

        extras = np.asarray([phase_idx, trust, turn], dtype=np.float32)
        return np.concatenate([v, extras], axis=0)

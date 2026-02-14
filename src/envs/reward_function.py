# src/envs/reward_function.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class RewardConfig:
    """
    Reward shaping configuration.

    Notes:
    - Trust level is assumed to be in [1..5].
    - Delta trust is trust_t - prev_trust.
    """
    r_step: float = 0.05   # per-step penalty
    r_abs: float = 0.1     # dense absolute trust bonus weight (0..1 scaled)
    r_alpha: float = 1.0   # weight on delta trust
    r_beta: float = 1.5    # extra penalty on trust drops
    r_gamma: float = 0.5   # terminal shaping weight

    # Trust scale bounds
    trust_min: int = 1
    trust_max: int = 5


def compute_reward(
    *,
    trust_level: int,
    prev_trust_level: int,
    critic_ran: bool,
    end_flag: bool,
    cfg: RewardConfig,
) -> tuple[float, Dict[str, Any]]:
    """
    Compute reward and return (reward, components).

    Reward design:
      - Always apply step penalty: -cfg.r_step
      - Optional dense absolute trust bonus: cfg.r_abs * (trust / trust_max)
      - Delta trust shaping only when critic ran: cfg.r_alpha * (trust - prev_trust)
      - Extra penalty on drops: -cfg.r_beta * max(0, prev_trust - trust)
      - Terminal shaping based on final trust (when end_flag): +/- cfg.r_gamma * ...
    """
    # Clamp trust to safe bounds (robustness)
    T = int(trust_level)
    T_prev = int(prev_trust_level)
    T = max(cfg.trust_min, min(cfg.trust_max, T))
    T_prev = max(cfg.trust_min, min(cfg.trust_max, T_prev))

    # delta only meaningful when critic ran
    delta = float(T - T_prev) if critic_ran else 0.0
    drop = float(max(0, T_prev - T)) if critic_ran else 0.0

    # base reward
    reward = -float(cfg.r_step)

    # dense absolute trust bonus (scaled to 0..1)
    abs_bonus = float(cfg.r_abs) * (float(T) / float(cfg.trust_max))
    reward += abs_bonus

    # delta trust shaping
    delta_term = float(cfg.r_alpha) * delta
    reward += delta_term

    # extra drop penalty
    drop_penalty = -float(cfg.r_beta) * drop
    reward += drop_penalty

    # terminal shaping
    terminal_term = 0.0
    if end_flag:
        if T >= 4:
            terminal_term = float(cfg.r_gamma) * float(T)
        else:
            terminal_term = -float(cfg.r_gamma) * float(6 - T)
        reward += terminal_term

    components = {
        "trust_level": T,
        "prev_trust_level": T_prev,
        "critic_ran": bool(critic_ran),
        "end_flag": bool(end_flag),
        "delta": delta,
        "drop": drop,
        "step_penalty": -float(cfg.r_step),
        "abs_bonus": abs_bonus,
        "delta_term": delta_term,
        "drop_penalty": drop_penalty,
        "terminal_term": terminal_term,
        "total_reward": float(reward),
    }
    return float(reward), components

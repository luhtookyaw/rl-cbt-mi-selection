# src/envs/reward_function.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Hashable


@dataclass
class RewardConfig:
    """
    Reward shaping configuration.

    Notes:
    - Trust level is assumed to be in [1..5].
    - Delta trust is trust_t - prev_trust.
    - Repeated action penalty discourages action loops by penalizing
      how often the current action appeared in the last K steps.
    """
    r_step: float = 0.05   # per-step penalty
    r_abs: float = 0.1     # dense absolute trust bonus weight (0..1 scaled)
    r_alpha: float = 1.0   # weight on delta trust
    r_beta: float = 1.5    # extra penalty on trust drops
    r_gamma: float = 0.5   # terminal shaping weight

    # Trust scale bounds
    trust_min: int = 1
    trust_max: int = 5

    # --- NEW: anti-loop repeated-action penalty ---
    enable_repeat_penalty: bool = False  # whether to apply repeated action penalty
    repeat_window: int = 5        # K: lookback window length
    repeat_free_count: int = 2    # no penalty until count in window exceeds this
    r_repeat_k: float = 0.15      # penalty per prior occurrence of current action in last K


def compute_reward(
    *,
    trust_level: int,
    prev_trust_level: int,
    critic_ran: bool,
    end_flag: bool,
    cfg: RewardConfig,
    action: Hashable,
    recent_actions: Sequence[Hashable] = (),
) -> tuple[float, Dict[str, Any]]:
    """
    Compute reward and return (reward, components).

    Reward design:
      - Always apply step penalty: -cfg.r_step
      - Optional dense absolute trust bonus: cfg.r_abs * (trust / trust_max)
      - Delta trust shaping only when critic ran: cfg.r_alpha * (trust - prev_trust)
      - Extra penalty on drops: -cfg.r_beta * max(0, prev_trust - trust)
      - Terminal shaping based on final trust (when end_flag): +/- cfg.r_gamma * ...
      - Optional repeated action penalty (anti-loop):
          Let K = cfg.repeat_window.
          Let c = number of times `action` appears in the last K actions (excluding current).
          Let f = cfg.repeat_free_count.
          Add penalty only when c > f: -cfg.r_repeat_k * (c - f)
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

    # --- NEW: repeated action penalty (anti-loop, last K steps) ---
    K = int(max(0, cfg.repeat_window))
    window = list(recent_actions)[-K:] if K > 0 else []
    repeat_count = sum(1 for a in window if a == action)  # prior occurrences in window
    free_count = int(max(0, cfg.repeat_free_count))
    penalized_repeat_count = max(0, int(repeat_count) - free_count)
    repeat_penalty = (
        -float(cfg.r_repeat_k) * float(penalized_repeat_count)
        if cfg.enable_repeat_penalty
        else 0.0
    )
    reward += repeat_penalty

    # terminal shaping
    terminal_term = 0.0
    if end_flag:
        if T >= 4:
            terminal_term = float(cfg.r_gamma) * float(T)
        else:
            terminal_term = -float(cfg.r_gamma) * float(6 - T)
        reward += terminal_term

    components: Dict[str, Any] = {
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
        # NEW fields
        "enable_repeat_penalty": bool(cfg.enable_repeat_penalty),
        "repeat_window": K,
        "repeat_free_count": int(free_count),
        "repeat_count": int(repeat_count),
        "penalized_repeat_count": int(penalized_repeat_count),
        "repeat_penalty": float(repeat_penalty),
        "terminal_term": terminal_term,
        "total_reward": float(reward),
    }
    return float(reward), components

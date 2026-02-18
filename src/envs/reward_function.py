# src/envs/reward_function.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Mapping


@dataclass
class RewardConfig:
    """
    Reward shaping configuration.

    Notes:
    - Trust level is assumed to be in [1..5].
    - Delta trust is trust_t - prev_trust.
    - Alliance is assumed to be in [1..5] (terminal-only).
    - Therapist skills are assumed to be in [0..6] (terminal-only); we take the mean.
    """
    # Dense / step shaping
    r_step: float = 0.05   # per-step penalty
    r_abs: float = 0.1     # dense absolute trust bonus weight (0..1 scaled)
    r_alpha: float = 1.0   # weight on delta trust
    r_beta: float = 1.5    # extra penalty on trust drops
    r_gamma: float = 0.5   # terminal shaping weight (trust terminal)

    # NEW: terminal evaluator weights (normalized to 0..1 before weighting)
    r_alliance: float = 0.4
    r_skills: float = 0.4

    # Trust bounds
    trust_min: int = 1
    trust_max: int = 5

    # NEW: Alliance bounds (terminal)
    alliance_min: int = 1
    alliance_max: int = 5

    # NEW: Therapist skills bounds (terminal)
    skills_min: int = 0
    skills_max: int = 6


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _normalize_to_unit(x: float, lo: float, hi: float) -> float:
    """
    Normalize x from [lo, hi] to [0, 1]. If hi==lo, return 0.0 safely.
    """
    if hi <= lo:
        return 0.0
    x = _clamp(x, lo, hi)
    return (x - lo) / (hi - lo)


def compute_reward(
    *,
    trust_level: int,
    prev_trust_level: int,
    critic_ran: bool,
    end_flag: bool,
    cfg: RewardConfig,
    # NEW: terminal-only metrics (pass when end_flag=True)
    alliance_score: Optional[float] = None,  # expected in [1..5]
    therapist_skill_scores: Optional[Mapping[str, float]] = None,  # each expected in [0..6]
) -> tuple[float, Dict[str, Any]]:
    """
    Compute reward and return (reward, components).

    Reward design:
      - Always apply step penalty: -cfg.r_step
      - Dense absolute trust bonus: cfg.r_abs * (trust / trust_max)
      - Delta trust shaping only when critic ran: cfg.r_alpha * (trust - prev_trust)
      - Extra penalty on drops: -cfg.r_beta * max(0, prev_trust - trust)
      - Terminal trust shaping (when end_flag): +/- cfg.r_gamma * ...
      - NEW: Terminal alliance bonus (normalized 0..1): + cfg.r_alliance * alliance_norm
      - NEW: Terminal skills bonus (normalized 0..1): + cfg.r_skills * skills_norm
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
    terminal_trust_term = 0.0
    alliance_term = 0.0
    skills_term = 0.0
    alliance_norm = None
    skills_mean = None
    skills_norm = None

    if end_flag:
        # existing terminal trust shaping
        if T >= 4:
            terminal_trust_term = float(cfg.r_gamma) * float(T)
        else:
            terminal_trust_term = -float(cfg.r_gamma) * float(6 - T)
        reward += terminal_trust_term

        # NEW: alliance terminal bonus (normalized)
        if alliance_score is not None:
            A = float(alliance_score)
            alliance_norm = _normalize_to_unit(A, float(cfg.alliance_min), float(cfg.alliance_max))
            alliance_term = float(cfg.r_alliance) * float(alliance_norm)
            reward += alliance_term

        # NEW: therapist skills terminal bonus (mean normalized)
        if therapist_skill_scores:
            vals = [float(v) for v in therapist_skill_scores.values() if v is not None]
            if vals:
                skills_mean = sum(vals) / len(vals)
                skills_norm = _normalize_to_unit(
                    float(skills_mean),
                    float(cfg.skills_min),
                    float(cfg.skills_max),
                )
                skills_term = float(cfg.r_skills) * float(skills_norm)
                reward += skills_term

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
        # existing terminal part (renamed key to be explicit)
        "terminal_trust_term": terminal_trust_term,
        # NEW terminal metrics
        "alliance_score": alliance_score,
        "alliance_norm": alliance_norm,
        "alliance_term": alliance_term,
        "therapist_skill_scores": dict(therapist_skill_scores) if therapist_skill_scores else None,
        "therapist_skills_mean": skills_mean,
        "skills_norm": skills_norm,
        "skills_term": skills_term,
        "total_reward": float(reward),
    }
    return float(reward), components

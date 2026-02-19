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
    # Dense / step shaping (small guidance; terminal is primary)
    r_step: float = 0.05   # per-step penalty
    r_abs: float = 0.1     # dense absolute trust bonus weight (0..1 scaled)
    r_alpha: float = 0.5   # weight on delta trust
    r_beta: float = 1.0    # extra penalty on trust drops

    # Terminal weights (all terminal parts are normalized 0..1 then weighted)
    r_gamma: float = 1.5      # terminal trust weight (normalized)
    r_alliance: float = 1.5   # terminal alliance weight (normalized)
    r_skills: float = 1.5     # terminal skills weight (normalized)

    # Trust bounds
    trust_min: int = 1
    trust_max: int = 5

    # Alliance bounds (terminal)
    alliance_min: int = 1
    alliance_max: int = 5

    # Therapist skills bounds (terminal)
    skills_min: int = 0
    skills_max: int = 6

    # Optional: prevent "end immediately" reward-hack
    min_turns_for_full_terminal: int = 3
    early_end_terminal_scale: float = 0.5  # scale terminal bonuses if end too early

    # Optional: discourage repeating the same action too many times
    repeat_threshold: int = 3       # allow up to 3 in a row
    repeat_penalty: float = 0.2    # penalty per extra repeat (4th -> -0.05, 5th -> -0.10, ...)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _normalize_to_unit(x: float, lo: float, hi: float) -> float:
    """Normalize x from [lo, hi] to [0, 1]."""
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
    # terminal-only metrics (pass when end_flag=True)
    alliance_score: Optional[float] = None,  # expected in [1..5]
    therapist_skill_scores: Optional[Mapping[str, float]] = None,  # each expected in [0..6]
    # NEW: current turn for optional early-end scaling
    turn: Optional[int] = None,
    same_action_streak: Optional[int] = None,   # NEW
) -> tuple[float, Dict[str, Any]]:
    """
    Compute reward and return (reward, components).

    Dense shaping (small):
      - step penalty
      - absolute trust bonus
      - delta trust bonus (only when critic ran)
      - drop penalty (only when critic ran)

    Terminal shaping (primary):
      - trust_norm * r_gamma
      - alliance_norm * r_alliance
      - skills_norm * r_skills
    """
    # Clamp trust to safe bounds
    T = int(trust_level)
    T_prev = int(prev_trust_level)
    T = max(cfg.trust_min, min(cfg.trust_max, T))
    T_prev = max(cfg.trust_min, min(cfg.trust_max, T_prev))

    # delta only meaningful when critic ran
    delta = float(T - T_prev) if critic_ran else 0.0
    drop = float(max(0, T_prev - T)) if critic_ran else 0.0

    # ---- dense reward (small guidance)
    reward = -float(cfg.r_step)

    abs_bonus = float(cfg.r_abs) * (float(T) / float(cfg.trust_max))
    reward += abs_bonus

    delta_term = float(cfg.r_alpha) * delta
    reward += delta_term

    drop_penalty = -float(cfg.r_beta) * drop
    reward += drop_penalty

    # ---- repetition penalty (anti-collapse)
    repeat_term = 0.0
    if same_action_streak is not None and same_action_streak > cfg.repeat_threshold:
        excess = int(same_action_streak) - int(cfg.repeat_threshold)
        repeat_term = -float(cfg.repeat_penalty) * float(excess)
        reward += repeat_term

    # ---- terminal reward (primary)
    terminal_trust_term = 0.0
    alliance_term = 0.0
    skills_term = 0.0

    trust_norm = None
    alliance_norm = None
    skills_mean = None
    skills_norm = None

    # Optional scaling if ended too early (avoid "end quickly" hacks)
    terminal_scale = 1.0
    if end_flag and (turn is not None) and (turn < cfg.min_turns_for_full_terminal):
        terminal_scale = float(cfg.early_end_terminal_scale)

    if end_flag:
        # trust terminal (normalized 0..1)
        trust_norm = _normalize_to_unit(float(T), float(cfg.trust_min), float(cfg.trust_max))
        terminal_trust_term = terminal_scale * float(cfg.r_gamma) * float(trust_norm)
        reward += terminal_trust_term

        # alliance terminal bonus (normalized 0..1)
        if alliance_score is not None:
            A = float(alliance_score)
            alliance_norm = _normalize_to_unit(A, float(cfg.alliance_min), float(cfg.alliance_max))
            alliance_term = terminal_scale * float(cfg.r_alliance) * float(alliance_norm)
            reward += alliance_term

        # therapist skills terminal bonus (mean normalized 0..1)
        if therapist_skill_scores:
            vals = [float(v) for v in therapist_skill_scores.values() if v is not None]
            if vals:
                skills_mean = sum(vals) / len(vals)
                skills_norm = _normalize_to_unit(
                    float(skills_mean),
                    float(cfg.skills_min),
                    float(cfg.skills_max),
                )
                skills_term = terminal_scale * float(cfg.r_skills) * float(skills_norm)
                reward += skills_term

    components: Dict[str, Any] = {
        "trust_level": T,
        "prev_trust_level": T_prev,
        "critic_ran": bool(critic_ran),
        "end_flag": bool(end_flag),
        "turn": turn,
        "delta": delta,
        "drop": drop,
        "step_penalty": -float(cfg.r_step),
        "abs_bonus": abs_bonus,
        "delta_term": delta_term,
        "drop_penalty": drop_penalty,
        # terminal parts
        "terminal_scale": terminal_scale,
        "trust_norm": trust_norm,
        "terminal_trust_term": terminal_trust_term,
        "alliance_score": alliance_score,
        "alliance_norm": alliance_norm,
        "alliance_term": alliance_term,
        "therapist_skill_scores": dict(therapist_skill_scores) if therapist_skill_scores else None,
        "therapist_skills_mean": skills_mean,
        "skills_norm": skills_norm,
        "skills_term": skills_term,
        "total_reward": float(reward),
        "same_action_streak": same_action_streak,
        "repeat_term": repeat_term,
    }
    return float(reward), components

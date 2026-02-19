# src/envs/therapy_env.py
"""
Gymnasium-style environment for PPO (or any RL) to learn a therapy intervention policy.

Core idea:
- Agent ACTION = choose an intervention_id (e.g., MI_SR, CBT_SOCRATIC_EVIDENCE, ...)
- Env STEP executes:
    1) Therapist LLM generates reply using therapist_system.txt + chosen intervention
    2) Client LLM generates next utterance using client.txt
    3) Trust critic scores openness/trust using trust_critic.txt (optionally sparse by interval)
    4) Moderator decides whether to end session using moderator.txt
- Reward = trust/openness score (1..5) (you can change reward shaping later)

This follows the logic of your simulate_one_patient() code, but restructured as an interactive env.
"""

from __future__ import annotations

import json
import random
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from src.llm import call_llm_messages
from src.envs.reward_function import compute_reward, RewardConfig
from src.envs.alliance import C_ALLIANCE_SYSTEM_PROMPT, EXAMPLE_C_ALLIANCE
from src.envs.therapist_skills import (
    CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    CBT_SPECIFIC_FOCUS_SKILL,
    CBT_SPECIFIC_STRATEGY_SKILL,
    GEN_COLLABORATION,
    GEN_INTERPERSONAL,
    GEN_UNDERSTANDING
)

# ----------------------------
# Defaults / Paths
# ----------------------------
DEFAULT_DATA_PATH = Path("data/Patient_PSi_CM_beginner.json")

DEFAULT_CLIENT_PROMPT_PATH = Path("prompts/client.txt")
DEFAULT_THERAPIST_PROMPT_PATH = Path("prompts/therapist_system.txt")
DEFAULT_CRITIC_PROMPT_PATH = Path("prompts/trust_critic.txt")
DEFAULT_MOD_PROMPT_PATH = Path("prompts/moderator.txt")

DEFAULT_TAXONOMY_PATH = Path("data/interventions_taxonomy.json")  # <-- Save your JSON here

PHASES = ["trust_building", "case_conceptualization", "solution_exploration"]
PHASE_TO_IDX = {p: i for i, p in enumerate(PHASES)}

PHASE_UPGRADE_AT = {
    "trust_building": 3,
    "case_conceptualization": 4,
    "solution_exploration": 999,
}

MAX_TURNS_DEFAULT = 25

THERAPIST_FIRST_USER = (
    "Start the session.\n"
    "Greet the client briefly and ask what brings them to therapy."
)

# We will feed the therapist the dialogue context each time
THERAPIST_USER_PROMPT = """Conversation so far:
{dialogue_context}

Respond as the therapist to the client's latest message.
"""


# ----------------------------
# Helpers
# ----------------------------
def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def parse_trust_score(text: str) -> Optional[int]:
    m = re.search(r"\b([1-5])\b", text)
    return int(m.group(1)) if m else None


def parse_yes_no(text: str) -> Optional[bool]:
    t = (text or "").strip().upper()
    m = re.search(r"\b(YES|NO)\b", t)
    return (m.group(1) == "YES") if m else None


def render_template(template: str, variables: dict) -> str:
    safe = {}
    for k, v in variables.items():
        if isinstance(v, (list, dict)):
            safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = "" if v is None else str(v)
    return template.format(**safe)


def format_dialogue(convo: list, last_n: int = 12) -> str:
    chunk = convo[-last_n:]
    out = []
    for msg in chunk:
        who = "Therapist" if msg["role"] == "assistant" else "Client"
        out.append(f"{who}: {msg['content']}")
    return "\n".join(out)


def next_phase(cur: str, openness: int) -> str:
    if cur == "trust_building" and openness >= PHASE_UPGRADE_AT["trust_building"]:
        return "case_conceptualization"
    if cur == "case_conceptualization" and openness >= PHASE_UPGRADE_AT["case_conceptualization"]:
        return "solution_exploration"
    return cur


def trust_eval_interval(resistance_level: str) -> int:
    lvl = (resistance_level or "").strip().lower()
    if lvl == "beginner":
        return 2
    if lvl == "intermediate":
        return 4
    if lvl == "advanced":
        return 6
    return 2


def normalize_patient(p: dict) -> dict:
    core_beliefs = []
    core_beliefs += p.get("helpless_belief", []) or []
    core_beliefs += p.get("unlovable_belief", []) or []
    core_beliefs += p.get("worthless_belief", []) or []

    mapped = dict(p)
    mapped["core_beliefs"] = core_beliefs
    mapped["intermediate_beliefs"] = p.get("intermediate_belief", "") or ""

    mapped["resistance_emotions"] = p.get("resistance_emotion", "")
    mapped["resistance_monologue"] = p.get("resistance_internal_monologue", "")

    pt = p.get("type", [])
    mapped["patient_type_content"] = ", ".join(pt) if isinstance(pt, list) else str(pt)
    mapped["style_description"] = ""
    return mapped


def load_patients(data_path: Path) -> list:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "patients" in data:
        return data["patients"]
    raise ValueError("Unexpected patient JSON structure")


@dataclass
class ModelConfig:
    therapist_model: str = "gpt-4o-mini"
    client_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o"
    moderator_model: str = "gpt-4o"


class TherapyEnv(gym.Env):
    """
    PPO-ready environment.

    Action:
        Discrete over intervention ids from taxonomy JSON.

    Observation:
        Dict with:
            - dialogue_text: last N turns formatted as text
            - phase_idx: 0..2
            - trust_level: 1..5
            - turn: 0..MAX_TURNS
            - patient_id (optional info-like signal)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        client_prompt_path: Path = DEFAULT_CLIENT_PROMPT_PATH,
        therapist_prompt_path: Path = DEFAULT_THERAPIST_PROMPT_PATH,
        critic_prompt_path: Path = DEFAULT_CRITIC_PROMPT_PATH,
        moderator_prompt_path: Path = DEFAULT_MOD_PROMPT_PATH,
        taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
        models: Optional[ModelConfig] = None,
        max_turns: int = MAX_TURNS_DEFAULT,
        dialogue_window: int = 24,
        critic_window: int = 16,
        moderator_window: int = 24,
        therapist_temperature: float = 0.4,
        client_temperature: float = 0.7,
        # If True, follow your sparse critic schedule (interval by resistance level)
        # If False, run critic every step (denser reward; often easier to train).
        sparse_critic: bool = True,
        seed: Optional[int] = None,
        fixed_patient_id: Optional[str] = None,
    ):
        super().__init__()

        self.data_path = data_path
        self.client_prompt_path = client_prompt_path
        self.therapist_prompt_path = therapist_prompt_path
        self.critic_prompt_path = critic_prompt_path
        self.moderator_prompt_path = moderator_prompt_path
        self.taxonomy_path = taxonomy_path

        self.models = models or ModelConfig()
        self.max_turns = max_turns
        self.dialogue_window = dialogue_window
        self.critic_window = critic_window
        self.moderator_window = moderator_window
        self.therapist_temperature = therapist_temperature
        self.client_temperature = client_temperature
        self.sparse_critic = sparse_critic

        self._rng = random.Random(seed)

        self.fixed_patient_id = fixed_patient_id

        # Load resources once
        self._patients = [normalize_patient(p) for p in load_patients(self.data_path)]
        self._client_template = load_text(self.client_prompt_path)
        self._therapist_template = load_text(self.therapist_prompt_path)
        self._critic_template = load_text(self.critic_prompt_path)
        self._moderator_template = load_text(self.moderator_prompt_path)

        self._taxonomy = load_json(self.taxonomy_path)
        self._actions = self._build_action_list(self._taxonomy)  # list of dicts
        self._action_ids = [a["id"] for a in self._actions]

        # Gym spaces
        self.action_space = spaces.Discrete(len(self._actions))

        # Text observations are fine for an env, but PPO usually needs numeric features.
        # You can wrap this env with an embedding/vectorizer wrapper later.
        self.observation_space = spaces.Dict(
            {
                "dialogue_text": spaces.Text(max_length=20_000),
                "phase_idx": spaces.Discrete(len(PHASES)),
                "trust_level": spaces.Discrete(6),  # 0..5, we'll use 1..5
                "turn": spaces.Discrete(self.max_turns + 1),
                "patient_id": spaces.Text(max_length=64),
            }
        )

        # Reward config
        self.reward_cfg = RewardConfig()

        # Episode state
        self._patient: Optional[dict] = None
        self._phase: str = "trust_building"
        self._openness: int = 1
        self._trust_level: int = 1
        self._turn: int = 0
        self._interval: int = 2
        self._convo: List[Dict[str, str]] = []  # [{"role":"assistant/user","content":...}, ...]
        self._last_critic_raw: Optional[str] = None
        self._last_moderator_raw: Optional[str] = None

        # Track previous trust for delta reward
        self._prev_trust_level = 1


    # ----------------------------
    # Taxonomy / action helpers
    # ----------------------------
    def _build_action_list(self, taxonomy: dict) -> List[dict]:
        if not isinstance(taxonomy, dict) or "interventions" not in taxonomy:
            raise ValueError("taxonomy JSON must contain top-level key: 'interventions'")

        actions = taxonomy["interventions"]
        if not isinstance(actions, list) or not actions:
            raise ValueError("taxonomy['interventions'] must be a non-empty list")

        # Require minimal fields
        for a in actions:
            if "id" not in a or "name" not in a:
                raise ValueError("Each intervention must have at least 'id' and 'name'")
        return actions

    def _get_action(self, action_index: int) -> dict:
        return self._actions[action_index]

    def _action_guidance_text(self, action: dict) -> str:
        """
        Build intervention_description passed into therapist_system prompt.
        Keep it concise and structured.
        """
        goal = action.get("goal", "")
        steps = action.get("steps", []) or []
        do = action.get("do", []) or []
        avoid = action.get("avoid", []) or []
        template = action.get("template", "")

        def bullets(xs: List[str], prefix: str = "- ") -> str:
            return "\n".join([prefix + str(x) for x in xs if str(x).strip()])

        parts = []
        if goal:
            parts.append(f"Goal: {goal}")
        if steps:
            parts.append("Steps:\n" + bullets(steps))
        if do:
            parts.append("Do:\n" + bullets(do))
        if avoid:
            parts.append("Avoid:\n" + bullets(avoid))
        if template:
            parts.append(f"Example template (adapt to context): {template}")

        return "\n\n".join(parts).strip()

    # ----------------------------
    # Core env API
    # ----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng.seed(seed)

        if self.fixed_patient_id is not None:
            options = {"patient_id": self.fixed_patient_id}

        # Select patient
        patient_id = None
        index = None
        if options:
            patient_id = options.get("patient_id")
            index = options.get("index")

        if patient_id is not None:
            match = None
            for p in self._patients:
                if str(p.get("id")) == str(patient_id):
                    match = p
                    break
            if match is None:
                raise ValueError(f"patient_id {patient_id} not found")
            self._patient = dict(match)
        elif index is not None:
            if index < 0 or index >= len(self._patients):
                raise ValueError(f"index out of range: {index}")
            self._patient = dict(self._patients[index])
        else:
            self._patient = dict(self._rng.choice(self._patients))
        
        # Initialize episode state
        self._phase = "trust_building"
        self._openness = 1

        self._trust_level = 1
        self._prev_trust_level = 1

        self._turn = 0
        self._last_critic_raw = None
        self._last_moderator_raw = None
        self._convo = []

        self._prev_action_id = None
        self._same_action_streak = 0

        self._interval = trust_eval_interval(self._patient.get("resistance_level"))

        # 1) Therapist starts (no agent action for first greeting)
        therapist_system = self._render_therapist_system(
            intervention_label="SESSION_START",
            intervention_description="Start the session: brief greeting and ask what brings the client to therapy.",
        )

        therapist_first = call_llm_messages(
            [
                {"role": "system", "content": therapist_system},
                {"role": "user", "content": THERAPIST_FIRST_USER},
            ],
            temperature=0.7,
            model=self.models.therapist_model,
        )
        self._convo.append({"role": "assistant", "content": therapist_first})

        # 2) Client responds immediately so the agent gets a meaningful first state
        client_text = self._client_respond()
        self._convo.append({"role": "user", "content": client_text})

        # 3) Critic + moderator to set initial trust/phase/done signal
        reward, done, info = self._evaluate_after_client()

        obs = self._make_obs()
        return obs, info

    def step(self, action_index: int):
        assert self._patient is not None, "Call reset() before step()."

        self._turn += 1

        # If already exceeded max turns, end (truncated)
        if self._turn > self.max_turns:
            # Run terminal evaluation one last time
            reward, done, info = self._evaluate_after_client(apply_repeat_penalty=False)
            obs = self._make_obs()

            terminated = False               # not a natural end, it's a time limit
            truncated = True
            info.update({"reason": "max_turns"})
            return obs, float(reward), terminated, truncated, info

        action = self._get_action(int(action_index))
        intervention_label = action["id"]

        # ---- NEW: track repeated action streak ----
        if self._prev_action_id == intervention_label:
            self._same_action_streak += 1
        else:
            self._same_action_streak = 1
        self._prev_action_id = intervention_label
        # ------------------------------------------

        intervention_description = self._action_guidance_text(action)

        # 1) Therapist reply using chosen intervention
        therapist_system = self._render_therapist_system(
            intervention_label=intervention_label,
            intervention_description=intervention_description,
        )

        therapist_user = THERAPIST_USER_PROMPT.format(
            dialogue_context=format_dialogue(self._convo, last_n=self.dialogue_window)
        )

        therapist_reply = call_llm_messages(
            [
                {"role": "system", "content": therapist_system},
                {"role": "user", "content": therapist_user},
            ],
            temperature=self.therapist_temperature,
            model=self.models.therapist_model,
        )
        self._convo.append({"role": "assistant", "content": therapist_reply})

        # 2) Client responds
        client_text = self._client_respond()
        self._convo.append({"role": "user", "content": client_text})

        # 3) Critic + moderator; reward comes from trust/openness score
        reward, done, info = self._evaluate_after_client()

        obs = self._make_obs()

        terminated = bool(done)
        truncated = False

        # Enrich info for training logs
        info.update(
            {
                "turn": self._turn,
                "phase": self._phase,
                "trust_level": self._trust_level,
                "openness": self._openness,
                "action_id": intervention_label,
                "therapist_last": therapist_reply,
                "client_last": client_text,
                "critic_raw": self._last_critic_raw,
                "moderator_raw": self._last_moderator_raw,

                # NEW: terminal logs (None unless end_flag=True)
                "alliance_score": info.get("alliance_score"),
                "therapist_skill_scores": info.get("therapist_skill_scores"),
                "alliance_raw": info.get("alliance_raw"),
                "therapist_skill_raws": info.get("therapist_skill_raws"),
            }
        )

        return obs, float(reward), terminated, truncated, info

    # ----------------------------
    # Internals: render + calls
    # ----------------------------
    def _render_therapist_system(self, intervention_label: str, intervention_description: str) -> str:
        """
        Your updated therapist_system.txt should include placeholders:
            {name}, {history}, {situation}, {intervention_label}, {intervention_description}
        """
        assert self._patient is not None

        return self._therapist_template.format(
            name=self._patient.get("name", ""),
            history=self._patient.get("history", ""),
            situation=self._patient.get("situation", ""),
            intervention_label=intervention_label,
            intervention_description=intervention_description,
        )

    def _client_respond(self) -> str:
        """
        Client simulator uses:
            trust_level, stage_therapy, and all patient fields via client.txt template.
        """
        assert self._patient is not None

        p = dict(self._patient)
        p["trust_level"] = self._trust_level
        p["stage_therapy"] = self._phase

        client_system = render_template(self._client_template, p)
        client_user = (
            "Conversation so far:\n"
            f"{format_dialogue(self._convo, last_n=self.dialogue_window)}\n\n"
            "Respond as the client to the therapist's latest message."
        )

        client_text = call_llm_messages(
            [
                {"role": "system", "content": client_system},
                {"role": "user", "content": client_user},
            ],
            temperature=self.client_temperature,
            model=self.models.client_model,
        )
        return client_text

    def _evaluate_after_client(self, apply_repeat_penalty: bool = True) -> Tuple[float, bool, dict]:
        """
        Runs trust critic (optionally sparse) + moderator.
        Updates:
            openness, phase, trust_level, last raw texts
        Returns:
            reward, done, info
        """
        assert self._patient is not None

        # ---- critic (trust/openness)
        should_eval = True
        if self.sparse_critic:
            should_eval = (self._turn % self._interval == 0)

        critic_text = None
        if should_eval:
            critic_system = render_template(
                self._critic_template,
                {"dialogue_context": format_dialogue(self._convo, last_n=self.critic_window)},
            )
            critic_text = call_llm_messages(
                [{"role": "system", "content": critic_system}],
                model=self.models.critic_model,
            )

            score = parse_trust_score(critic_text)
            if score is not None:
                self._openness = score

            # phase progression only when critic ran
            self._phase = next_phase(self._phase, self._openness)

            # trust follows openness
            self._trust_level = self._openness

        self._last_critic_raw = critic_text

        # ---- moderator end?
        mod_system = render_template(
            self._moderator_template,
            {"conversation": format_dialogue(self._convo, last_n=self.moderator_window)},
        )
        mod_text = call_llm_messages(
            [{"role": "system", "content": mod_system}],
            model=self.models.moderator_model,
        )

        end_flag = parse_yes_no(mod_text)
        if end_flag is None:
            end_flag = False

        self._last_moderator_raw = mod_text

        # ---- NEW: terminal evaluators (only when end_flag)
        alliance_score = None
        alliance_raw = None
        alliance_extra = None
        skills_scores = None
        skills_raws = None

        if bool(end_flag):
            alliance_score, alliance_raw, alliance_extra = self._eval_alliance_terminal()
            skills_scores, skills_raws = self._eval_therapist_skills_terminal()

        # ---- reward (centralized in reward_function.py)
        reward, rcomps = compute_reward(
            trust_level=self._trust_level,
            prev_trust_level=self._prev_trust_level,
            critic_ran=should_eval,
            end_flag=bool(end_flag),
            cfg=self.reward_cfg,  # IMPORTANT: required by your updated reward_function.py
            alliance_score=alliance_score,
            therapist_skill_scores=skills_scores if skills_scores else None,
            turn=self._turn,
            same_action_streak=(self._same_action_streak if apply_repeat_penalty else None),
        )

        # Update prev trust only when critic ran (cleaner + avoids stale updates)
        if should_eval:
            self._prev_trust_level = int(self._trust_level)

        info = {
            "end_session": bool(end_flag),
            "critic_ran": bool(should_eval),
            "reward_components": rcomps,  # super useful for debugging/plots
            # NEW: terminal logs (present only when end_flag=True)
            "alliance_score": alliance_score,
            "alliance_raw": alliance_raw,
            "alliance_extra": alliance_extra,
            "therapist_skill_scores": skills_scores,
            "therapist_skill_raws": skills_raws,
        }

        return float(reward), bool(end_flag), info


    def _parse_alliance_blocks(self, raw: str) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        if not raw:
            return results

        # Split on blank lines between dict blocks
        blocks = [b.strip() for b in re.split(r"\n\s*\n", raw.strip()) if b.strip()]

        # Fallback: sometimes blocks come without blank lines
        if len(blocks) == 1 and raw.count("{") > 1:
            # try to split between }{
            tmp = re.split(r"}\s*{", raw.strip())
            blocks = []
            for i, b in enumerate(tmp):
                b = b.strip()
                if not b.startswith("{"):
                    b = "{" + b
                if not b.endswith("}"):
                    b = b + "}"
                blocks.append(b)

        for b in blocks:
            try:
                d = ast.literal_eval(b)
                if not isinstance(d, dict):
                    continue

                q_keys = [k for k in d.keys() if isinstance(k, str) and re.fullmatch(r"Q\d+", k)]
                if not q_keys:
                    continue

                # pick the smallest Q number if multiple appear
                q = sorted(q_keys, key=lambda x: int(x[1:]))[0]

                s = d.get("score", None)
                score_val = None
                try:
                    score_val = float(s)
                except Exception:
                    m = re.search(r"\b([1-5])\b", str(s))
                    score_val = float(m.group(1)) if m else None

                if score_val is None:
                    continue

                # clamp to 1..5
                score_val = max(1.0, min(5.0, score_val))

                results[q] = {
                    "score": score_val,
                    "reason": d.get("reason", None),
                    "raw": d,
                }
            except Exception:
                continue

        return results


    def _aggregate_alliance_scores(self, q_scores: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes mean alliance and optional subscales.
        Assumes each score is 1..5.
        """
        def mean(keys):
            vals = [q_scores[k]["score"] for k in keys if k in q_scores]
            return sum(vals) / len(vals) if vals else None

        all_keys = [f"Q{i}" for i in range(1, 13)]
        goal_keys = [f"Q{i}" for i in range(1, 5)]
        approach_keys = [f"Q{i}" for i in range(5, 9)]
        bond_keys = [f"Q{i}" for i in range(9, 13)]

        out: Dict[str, float] = {}
        m_all = mean(all_keys)
        if m_all is not None:
            out["alliance_mean"] = float(m_all)

        m_goal = mean(goal_keys)
        if m_goal is not None:
            out["goal_mean"] = float(m_goal)

        m_app = mean(approach_keys)
        if m_app is not None:
            out["approach_mean"] = float(m_app)

        m_bond = mean(bond_keys)
        if m_bond is not None:
            out["bond_mean"] = float(m_bond)

        return out


    def _eval_alliance_terminal(self) -> Tuple[Optional[float], Optional[str], dict]:
        """
        Returns:
        alliance_score_mean (1..5),
        raw_text,
        extra_metrics dict (subscales, per-question scores)
        """
        convo = format_dialogue(self._convo, last_n=self.moderator_window)
        system = C_ALLIANCE_SYSTEM_PROMPT.format(
            conversation=convo,
            example=json.dumps(EXAMPLE_C_ALLIANCE, ensure_ascii=False),
        )

        raw = call_llm_messages(
            [{"role": "system", "content": system}],
            model=self.models.critic_model,
        )

        per_q = self._parse_alliance_blocks(raw)
        agg = self._aggregate_alliance_scores(per_q)

        alliance_mean = agg.get("alliance_mean", None)

        extra = {
            "per_question": {k: v["score"] for k, v in per_q.items()},
            "subscales": {k: v for k, v in agg.items() if k != "alliance_mean"},
        }
        return alliance_mean, raw, extra


    def _eval_therapist_skills_terminal(self) -> tuple[Dict[str, float], Dict[str, str]]:
        """
        Returns (scores, raws).
        Scores expected each in [0..6].
        """
        convo = format_dialogue(self._convo, last_n=self.moderator_window)

        prompts = {
            "guided_discovery": CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
            "focus": CBT_SPECIFIC_FOCUS_SKILL,
            "strategy": CBT_SPECIFIC_STRATEGY_SKILL,
            "understanding": GEN_UNDERSTANDING,
            "interpersonal": GEN_INTERPERSONAL,
            "collaboration": GEN_COLLABORATION
        }

        scores: Dict[str, float] = {}
        raws: Dict[str, str] = {}

        for name, tmpl in prompts.items():
            system = tmpl.format(conversation=convo)
            raw = call_llm_messages(
                [{"role": "system", "content": system}],
                model=self.models.critic_model,
            )
            raws[name] = raw

            # Prompt says: "score, explanation" — so parse before comma; fallback regex 0/2/4/6
            s = None
            try:
                s = float(raw.split(",", 1)[0].strip())
            except Exception:
                m = re.search(r"\b(0|2|4|6)\b", raw or "")
                s = float(m.group(1)) if m else None

            if s is not None:
                scores[name] = s

        return scores, raws


    def _make_obs(self) -> dict:
        assert self._patient is not None

        return {
            "dialogue_text": format_dialogue(self._convo, last_n=self.dialogue_window),
            "phase_idx": PHASE_TO_IDX.get(self._phase, 0),
            "trust_level": int(self._trust_level),
            "turn": int(self._turn),
            "patient_id": str(self._patient.get("id", "")),
        }

    # ----------------------------
    # Optional: render
    # ----------------------------
    def render(self):
        print(format_dialogue(self._convo, last_n=self.dialogue_window))


# ----------------------------
# Quick manual test (optional)
# ----------------------------
if __name__ == "__main__":
    env = TherapyEnv(
        taxonomy_path=DEFAULT_TAXONOMY_PATH,
        sparse_critic=False,  # easier to see reward changes each step
    )

    obs, info = env.reset(options={"index": 0})
    print("RESET:", info)
    env.render()

    for _ in range(3):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        print("\nACTION:", info["action_id"], "REWARD:", reward, "DONE:", terminated)
        env.render()
        if terminated or truncated:
            break

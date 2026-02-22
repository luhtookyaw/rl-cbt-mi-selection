import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.embeddings import embed_text
from src.llm import call_llm_messages  # your wrapper


# -----------------------
# Paths (edit if needed)
# -----------------------
DATA_PATH = Path("data/Patient_PSi_CM_Dataset_Planning_Resistance.json")
CLIENT_PROMPT_PATH = Path("prompts/client.txt")
THERAPIST_PROMPT_PATH = Path("prompts/therapist_system.txt")
CRITIC_PROMPT_PATH = Path("prompts/trust_critic.txt")
MOD_PROMPT_PATH = Path("prompts/moderator.txt")
TAXONOMY_PATH = Path("data/interventions_taxonomy.json")  # <-- set to your taxonomy JSON

MODEL_PATH = "outputs/ppo_therapy_router.zip"
OUT_PATH = Path("outputs/policy_rollout.json")

MAX_TURNS = 25

PHASES = ["trust_building", "case_conceptualization", "solution_exploration"]
PHASE_TO_IDX = {p: i for i, p in enumerate(PHASES)}
PHASE_UPGRADE_AT = {"trust_building": 3, "case_conceptualization": 4, "solution_exploration": 999}


THERAPIST_FIRST_USER = "Start the session.\nGreet the client briefly and ask what brings them to therapy."

THERAPIST_USER_PROMPT = """Conversation so far:
{dialogue_context}

Respond as the therapist to the client's latest message.
"""


# -----------------------
# Utilities
# -----------------------
def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_patients() -> list:
    data = load_json(DATA_PATH)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "patients" in data:
        return data["patients"]
    raise ValueError("Unexpected patient JSON structure")


def pick_patient(patients: list, patient_id: Optional[str] = None, index: Optional[int] = None) -> dict:
    if patient_id is not None:
        for p in patients:
            if str(p.get("id")) == str(patient_id):
                return p
        raise ValueError(f"patient_id {patient_id} not found")

    if index is not None:
        if index < 0 or index >= len(patients):
            raise ValueError(f"index out of range: {index}")
        return patients[index]

    return patients[0]


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


def render_template(template: str, variables: dict) -> str:
    safe = {}
    for k, v in variables.items():
        if isinstance(v, (list, dict)):
            safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = "" if v is None else str(v)
    return template.format(**safe)


def format_dialogue(convo: list, last_n: int = 24) -> str:
    chunk = convo[-last_n:]
    out = []
    for msg in chunk:
        who = "Therapist" if msg["role"] == "assistant" else "Client"
        out.append(f"{who}: {msg['content']}")
    return "\n".join(out)


def parse_trust_score(text: str) -> Optional[int]:
    m = re.search(r"\b([1-5])\b", text)
    return int(m.group(1)) if m else None


def parse_yes_no(text: str) -> Optional[bool]:
    t = text.strip().upper()
    m = re.search(r"\b(YES|NO)\b", t)
    return (m.group(1) == "YES") if m else None


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


# -----------------------
# Action taxonomy flattening
# IMPORTANT: Must match training ordering!
# -----------------------
def build_action_list(taxonomy: dict) -> List[dict]:
    """
    Supports either:
    - {"actions": [...]}
    - {"interventions": [...]}
    Returns actions in taxonomy order (must match training).
    """
    if not isinstance(taxonomy, dict):
        raise ValueError("Taxonomy JSON must be an object.")

    actions: Optional[List[dict]] = None
    if isinstance(taxonomy.get("actions"), list):
        actions = taxonomy["actions"]
    elif isinstance(taxonomy.get("interventions"), list):
        actions = taxonomy["interventions"]

    if not actions:
        raise ValueError(
            "Taxonomy JSON must contain top-level key 'actions' or 'interventions' as a non-empty list."
        )

    for a in actions:
        if not isinstance(a, dict) or "id" not in a:
            raise ValueError("Each action/intervention must be an object with at least an 'id' field.")
    return actions


def action_guidance_text(action: dict) -> str:
    """
    Text injected into therapist system prompt so LLM follows the chosen intervention.
    """
    def bullets(items: List[str], prefix: str = "- ") -> str:
        return "\n".join(prefix + str(x) for x in items if str(x).strip())

    goal = action.get("goal", "")
    steps = action.get("steps", []) or []
    do = action.get("do", []) or []
    avoid = action.get("avoid", []) or []
    template = action.get("template", "")

    parts: List[str] = []
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


def format_constraint_block(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {str(x)}" for x in value if str(x).strip())
    return str(value)


def render_therapist_system(
    therapist_template: str,
    patient: dict,
    intervention_label: str,
    intervention_description: str,
    therapist_micro_skills: Any = "",
    fidelity_check: Any = "",
) -> str:
    therapist_micro_skills_text = format_constraint_block(therapist_micro_skills)
    fidelity_check_text = format_constraint_block(fidelity_check)
    return render_template(
        therapist_template,
        {
            "name": patient.get("name", ""),
            "history": patient.get("history", ""),
            "situation": patient.get("situation", ""),
            "intervention_label": intervention_label,
            "intervention_description": intervention_description,
            "therapist_micro_skills": therapist_micro_skills_text,
            "fidelity_check": fidelity_check_text,
        },
    )


# -----------------------
# Embedding (must match your wrapper!)
# -----------------------
def embed_text_openai(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Uses the same helper as LastClientEmbeddingWrapper training pipeline.
    Returns float32 vector.
    """
    vec = embed_text(text or "", model=model)
    return np.asarray(vec, dtype=np.float32)


def make_policy_obs_from_convo(
    convo: List[Dict[str, str]],
    phase: str,
    trust_level: int,
    turn: int,
    max_turns: int = MAX_TURNS,
) -> np.ndarray:
    """
    Match LastClientEmbeddingWrapper(add_numeric_features=True):
      [last_client_embedding..., phase_idx_norm, trust_norm, turn_norm]
    """
    last_client = ""
    for msg in reversed(convo):
        if msg["role"] == "user":
            last_client = msg["content"]
            break

    emb = embed_text_openai(last_client)
    phase_idx_norm = float(PHASE_TO_IDX.get(phase, 0)) / 2.0
    trust_norm = float(trust_level) / 5.0
    turn_norm = float(turn) / max(1.0, float(max_turns))
    extras = np.asarray([phase_idx_norm, trust_norm, turn_norm], dtype=np.float32)
    return np.concatenate([emb, extras], axis=0)


# -----------------------
# Main rollout
# -----------------------
def run_policy(patient_id: str, deterministic: bool = True) -> dict:
    patients = load_patients()
    p0 = pick_patient(patients, patient_id=patient_id)
    p = normalize_patient(p0)
    print(f"Running Patient ID: {p.get('id', patient_id)}")

    client_template = load_text(CLIENT_PROMPT_PATH)
    therapist_template = load_text(THERAPIST_PROMPT_PATH)
    critic_template = load_text(CRITIC_PROMPT_PATH)
    mod_template = load_text(MOD_PROMPT_PATH)

    taxonomy = load_json(TAXONOMY_PATH)
    actions = build_action_list(taxonomy)

    # Load PPO
    model = PPO.load(MODEL_PATH, device="cpu")

    # Episode state
    phase = "trust_building"
    openness = 1
    trust_level = 1
    critic_interval = trust_eval_interval(str(p.get("resistance_level", "")))

    convo: List[Dict[str, str]] = []
    turns: List[dict] = []

    # ---- Therapist starts (SESSION_START)
    therapist_system = render_therapist_system(
        therapist_template=therapist_template,
        patient=p,
        intervention_label="SESSION_START",
        intervention_description="Start the session: brief greeting and ask what brings the client to therapy.",
        therapist_micro_skills=[
            "Warm professional greeting",
            "Psychological safety signaling",
            "Non-directive curiosity",
            "Calm conversational pacing",
            "Rapport-first tone (not clinical/interrogative)",
        ],
        fidelity_check=[
            "Greeting present but brief (1 sentence max)",
            "Exactly one open-ended intake question",
            "No advice, interpretation, or therapy techniques yet",
            "No assumptions about client problems",
            "Tone welcoming, respectful, and non-pressuring",
            "Response remains within 3–8 sentences total",
        ],
    )

    therapist_first = call_llm_messages(
        [{"role": "system", "content": therapist_system},
         {"role": "user", "content": THERAPIST_FIRST_USER}],
        temperature=0.4,
        model="gpt-4o-mini",
    )
    convo.append({"role": "assistant", "content": therapist_first})

    for t in range(1, MAX_TURNS + 1):
        # Therapist message that the current client response reacts to.
        therapist_for_turn = convo[-1]["content"] if convo and convo[-1]["role"] == "assistant" else ""

        # ---- Client responds
        p["trust_level"] = trust_level
        p["stage_therapy"] = phase

        client_system = render_template(client_template, p)
        client_user = (
            "Conversation so far:\n"
            f"{format_dialogue(convo, last_n=24)}\n\n"
            "Respond as the client to the therapist's latest message."
        )

        client_text = call_llm_messages(
            [{"role": "system", "content": client_system},
             {"role": "user", "content": client_user}],
            temperature=0.7,
            model="gpt-4o-mini",
        )
        convo.append({"role": "user", "content": client_text})

        # ---- Critic (trust/openness), sparse by resistance-level interval.
        # Start critic from turn 2, then evaluate every `critic_interval` turns.
        should_eval = (t >= 2) and (((t - 2) % critic_interval) == 0)
        critic_text = None
        score = None
        if should_eval:
            critic_system = render_template(
                critic_template,
                {"dialogue_context": format_dialogue(convo, last_n=16)},
            )
            critic_text = call_llm_messages(
                [{"role": "system", "content": critic_system}],
                model="gpt-4o",
            )
            score = parse_trust_score(critic_text)
            if score is not None:
                openness = score
                trust_level = openness
            phase = next_phase(phase, openness)

        # ---- Moderator (end?)
        mod_system = render_template(mod_template, {"conversation": format_dialogue(convo, last_n=24)})
        mod_text = call_llm_messages([{"role": "system", "content": mod_system}], model="gpt-4o")
        end_flag = parse_yes_no(mod_text)
        if end_flag is None:
            end_flag = False

        # ---- Build policy observation (embedding) and choose action
        obs_vec = make_policy_obs_from_convo(
            convo=convo,
            phase=phase,
            trust_level=trust_level,
            turn=t,
            max_turns=MAX_TURNS,
        )
        action_index, _ = model.predict(obs_vec, deterministic=deterministic)
        action_index = int(action_index)

        if action_index < 0 or action_index >= len(actions):
            raise RuntimeError(f"Policy produced invalid action_index={action_index} for {len(actions)} actions")

        action = actions[action_index]
        intervention_label = action["id"]
        intervention_description = action_guidance_text(action)
        therapist_micro_skills = action.get("therapist_micro_skills", [])
        fidelity_check = action.get("fidelity_check", action.get("fidelit_check", []))

        # ---- Save turn record (client just spoke; now agent picks next therapist intervention)
        turns.append({
            "turn_id": t,
            "phase_for_next_turn": phase,
            "trust_level": trust_level,
            "openness": openness,
            "action_index": action_index,
            "action_id": intervention_label,
            "client": client_text,
            "critic_raw": critic_text,
            "critic_ran": bool(should_eval),
            "moderator_raw": mod_text,
            "end_session": bool(end_flag),
        })

        turn_label = f"Turn {t} = session start" if t == 1 else f"Turn {t}"
        trust_display = str(score) if score is not None else str(trust_level)
        moderator_display = "YES" if end_flag else "NO"
        print(f"\n{turn_label}, Phase: {phase}")
        print(f"Therapist: {therapist_for_turn}")
        print(f"Client: {client_text}")
        print(
            f"Trust score: {trust_display}, "
            f"Next Action: {intervention_label}, "
            f"Critic Ran: {bool(should_eval)}, "
            f"Moderator: {moderator_display}"
        )

        if end_flag:
            break

        # ---- Therapist responds using chosen intervention
        therapist_system = render_therapist_system(
            therapist_template=therapist_template,
            patient=p,
            intervention_label=intervention_label,
            intervention_description=intervention_description,
            therapist_micro_skills=therapist_micro_skills,
            fidelity_check=fidelity_check,
        )

        therapist_user = THERAPIST_USER_PROMPT.format(dialogue_context=format_dialogue(convo, last_n=24))
        therapist_reply = call_llm_messages(
            [{"role": "system", "content": therapist_system},
             {"role": "user", "content": therapist_user}],
            temperature=0.4,
            model="gpt-4o-mini",
        )
        convo.append({"role": "assistant", "content": therapist_reply})

    return {
        "patient_id": str(p.get("id")),
        "patient_name": str(p.get("name")),
        "turns": turns,
        "full_conversation": convo,
    }


if __name__ == "__main__":
    # Example: run with a chosen case
    result = run_policy(patient_id="1-1", deterministic=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved rollout to: {OUT_PATH}")

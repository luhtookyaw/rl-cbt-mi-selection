import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.envs.therapy_env import TherapyEnv, format_dialogue, load_text, render_template
from src.llm import call_llm_messages


SELECTOR_SYSTEM_PATH = Path("prompts/selector_system.txt")
SELECTOR_USER_PATH = Path("prompts/selector_user.txt")
OUT_PATH = Path("outputs/selector_rollout.json")


def last_client_utterance(convo: List[Dict[str, str]]) -> str:
    for msg in reversed(convo):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def parse_selector_label(text: str, valid_labels: List[str]) -> Optional[str]:
    label_set = set(valid_labels)
    m = re.search(r"intervention_label\s*:\s*([A-Za-z0-9_]+)", text or "", flags=re.IGNORECASE)
    if m:
        label = m.group(1).strip()
        if label in label_set:
            return label

    for label in valid_labels:
        if re.search(rf"\b{re.escape(label)}\b", text or ""):
            return label
    return None


def parse_selector_rationale(text: str) -> Optional[str]:
    m = re.search(r"rationale\s*:\s*(.+)", text or "", flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    rationale = m.group(1).strip()
    return rationale or None


def extract_first_json_object(text: str) -> Optional[dict]:
    t = text or ""
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    obj_start = None
    for i, ch in enumerate(t[start:], start=start):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    parsed = json.loads(t[obj_start : i + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def parse_selector_response(text: str, valid_labels: List[str]) -> Tuple[Optional[str], Optional[str]]:
    parsed_obj = extract_first_json_object(text)
    if parsed_obj is not None:
        label = parsed_obj.get("intervention_label")
        rationale = parsed_obj.get("rationale")
        if isinstance(label, str) and label in set(valid_labels):
            rationale_text = rationale.strip() if isinstance(rationale, str) else ""
            return label, rationale_text

    label_fallback = parse_selector_label(text, valid_labels)
    rationale_fallback = parse_selector_rationale(text)
    return label_fallback, rationale_fallback


def select_intervention_label(
    *,
    env: TherapyEnv,
    selector_system: str,
    selector_user_template: str,
    selector_model: str = "gpt-4o",
) -> Tuple[str, str, str]:
    assert env._patient is not None

    user_prompt = render_template(
        selector_user_template,
        {
            "dialogue_history": format_dialogue(env._convo, last_n=env.dialogue_window),
            "history": env._patient.get("history", ""),
            "situation": env._patient.get("situation", ""),
            "stage_therapy": env._phase,
        },
    )

    selector_raw = call_llm_messages(
        [
            {"role": "system", "content": selector_system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        model=selector_model,
    )

    selected, rationale = parse_selector_response(selector_raw, env._action_ids)
    if selected is None:
        selected = env._action_ids[0]
    if rationale is None:
        rationale = ""
    return selected, rationale, selector_raw


def run_selector_session(
    *,
    patient_id: Optional[str] = "1-1",
    deterministic: bool = True,  # kept for API symmetry with run_policy.py
    sparse_critic: bool = True,
    selector_model: str = "gpt-4o-mini",
) -> dict:
    del deterministic  # not used; selector is LLM-based

    selector_system = load_text(SELECTOR_SYSTEM_PATH)
    selector_user_template = load_text(SELECTOR_USER_PATH)

    env = TherapyEnv(sparse_critic=sparse_critic, fixed_patient_id=patient_id)
    obs, reset_info = env.reset()
    del obs

    therapist_first = env._convo[0]["content"] if len(env._convo) > 0 else ""
    client_first = env._convo[1]["content"] if len(env._convo) > 1 else ""
    print("Turn 0 (reset)")
    print(f"Phase: {env._phase}")
    print(f"Therapist: {therapist_first}")
    print(f"Client: {client_first}")

    label_to_index = {label: i for i, label in enumerate(env._action_ids)}
    turns: List[dict] = []

    # Match env lifecycle: each step starts from current dialogue and chooses next intervention.
    for turn_id in range(1, env.max_turns + 1):
        selected_label, selector_rationale, selector_raw = select_intervention_label(
            env=env,
            selector_system=selector_system,
            selector_user_template=selector_user_template,
            selector_model=selector_model,
        )
        action_index = label_to_index[selected_label]

        _, reward, terminated, truncated, info = env.step(action_index)

        turns.append(
            {
                "turn_id": turn_id,
                "selected_action_id": selected_label,
                "selector_rationale": selector_rationale,
                "selector_raw": selector_raw,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "phase": info.get("phase"),
                "trust_level": info.get("trust_level"),
                "openness": info.get("openness"),
                "critic_ran": info.get("critic_ran"),
                "end_session": info.get("end_session"),
                "reward_components": info.get("reward_components"),
                "therapist_last": info.get("therapist_last"),
                "client_last": info.get("client_last"),
            }
        )

        therapist_last = info.get("therapist_last", "")
        client_last = info.get("client_last", "")
        print(
            f"Turn {turn_id} | Action: {selected_label} | "
            f"Phase: {info.get('phase')} | "
            f"Reward: {float(reward):.3f} | "
            f"Trust: {info.get('trust_level')} | End: {bool(terminated or truncated)}"
        )
        if selector_rationale:
            print(f"Selector rationale: {selector_rationale}")
        print(f"Therapist: {therapist_last}")
        print(f"Client: {client_last}")
        print("\n" + "-" * 50 + "\n")

        if terminated or truncated:
            break

    assert env._patient is not None
    return {
        "patient_id": str(env._patient.get("id", "")),
        "patient_name": str(env._patient.get("name", "")),
        "reset_info": reset_info,
        "turns": turns,
        "full_conversation": env._convo,
    }


if __name__ == "__main__":
    result = run_selector_session(patient_id="1-1", sparse_critic=True, selector_model="gpt-4o-mini")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved selector rollout to: {OUT_PATH}")

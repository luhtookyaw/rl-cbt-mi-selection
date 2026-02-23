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
    fields = parse_selector_fields(text)
    rationale = fields.get("rationale", "").strip()
    return rationale or None


def parse_selector_guidance_steps(text: str) -> Optional[str]:
    fields = parse_selector_fields(text)
    guidance = fields.get("guidance_steps", "").strip()
    return guidance or None


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


def parse_selector_fields(text: str) -> Dict[str, str]:
    """
    Parse selector plain-text output fields and support multiline values.

    Expected keys:
      intervention_label:
      rationale:
      guidance_steps:
    """
    known = {"intervention_label", "rationale", "guidance_steps"}
    out: Dict[str, List[str]] = {}
    current_key: Optional[str] = None

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        m = re.match(r"^\s*([A-Za-z_]+)\s*:\s*(.*)$", line)
        if m:
            key = m.group(1).strip().lower()
            value = m.group(2)
            if key in known:
                current_key = key
                out.setdefault(current_key, [])
                if value.strip():
                    out[current_key].append(value)
                continue

        # Continuation line for the current key (supports multiline guidance blocks)
        if current_key is not None and line.strip():
            out[current_key].append(line)

    return {k: "\n".join(v).strip() for k, v in out.items()}


def parse_selector_response(
    text: str, valid_labels: List[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parsed_obj = extract_first_json_object(text)
    if parsed_obj is not None:
        label = parsed_obj.get("intervention_label")
        rationale = parsed_obj.get("rationale")
        guidance_steps = parsed_obj.get("guidance_steps")
        if isinstance(label, str) and label in set(valid_labels):
            rationale_text = rationale.strip() if isinstance(rationale, str) else ""
            guidance_text = guidance_steps.strip() if isinstance(guidance_steps, str) else ""
            return label, rationale_text, guidance_text

    fields = parse_selector_fields(text)
    label_fallback = fields.get("intervention_label")
    if not label_fallback or label_fallback not in set(valid_labels):
        label_fallback = parse_selector_label(text, valid_labels)
    rationale_fallback = fields.get("rationale") or parse_selector_rationale(text)
    guidance_fallback = fields.get("guidance_steps") or parse_selector_guidance_steps(text)
    return label_fallback, rationale_fallback, guidance_fallback


def select_intervention_label(
    *,
    env: TherapyEnv,
    selector_system: str,
    selector_user_template: str,
    trust_history: List[int],
    selector_model: str = "gpt-4o",
) -> Tuple[str, str, str, str]:
    assert env._patient is not None

    user_prompt = render_template(
        selector_user_template,
        {
            "dialogue_history": format_dialogue(env._convo, last_n=env.dialogue_window),
            "history": env._patient.get("history", ""),
            "situation": env._patient.get("situation", ""),
            "stage_therapy": env._phase,
            "trust_level": env._trust_level,
            "trust_history": ", ".join(map(str, trust_history[-5:])),
            "recent_interventions": ", ".join(env._recent_actions[-5:]) if env._recent_actions else "None",
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

    selected, rationale, guidance_steps = parse_selector_response(selector_raw, env._action_ids)
    if selected is None:
        selected = env._action_ids[0]
    if rationale is None:
        rationale = ""
    if guidance_steps is None:
        guidance_steps = ""
    return selected, rationale, guidance_steps, selector_raw


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
    trust_history: List[int] = [int(env._trust_level)]

    # Match env lifecycle: each step starts from current dialogue and chooses next intervention.
    for turn_id in range(1, env.max_turns + 1):
        selected_label, selector_rationale, selector_guidance_steps, selector_raw = select_intervention_label(
            env=env,
            selector_system=selector_system,
            selector_user_template=selector_user_template,
            trust_history=trust_history,
            selector_model=selector_model,
        )
        action_index = label_to_index[selected_label]

        _, reward, terminated, truncated, info = env.step(
            action_index,
            guidance_steps_override=selector_guidance_steps or None,
        )

        turns.append(
            {
                "turn_id": turn_id,
                "selected_action_id": selected_label,
                "selector_rationale": selector_rationale,
                "selector_guidance_steps": selector_guidance_steps,
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
        trust_history.append(int(info.get("trust_level", env._trust_level)))

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
        if selector_guidance_steps:
            print(f"Selector guidance_steps: {selector_guidance_steps}")
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

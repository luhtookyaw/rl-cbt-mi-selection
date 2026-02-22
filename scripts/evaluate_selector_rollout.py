import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.alliance import C_ALLIANCE_SYSTEM_PROMPT, EXAMPLE_C_ALLIANCE
from src.llm import call_llm_messages
from src.therapist_skills import (
    CBT_SPECIFIC_FOCUS,
    CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    CBT_SPECIFIC_STRATEGY,
    GEN_COLLABORATION,
    GEN_INTERPERSONAL,
    GEN_UNDERSTANDING,
)


INPUT_PATH = Path("outputs/selector_rollout.json")
OUTPUT_DIR = Path("outputs/evaluations")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_conversation(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        speaker = "Therapist" if role == "assistant" else "Client"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def extract_score_prefix(text: str) -> Optional[int]:
    m = re.match(r"^\s*([0-6])\b", text or "")
    return int(m.group(1)) if m else None


def extract_explanation_after_first_comma(text: str) -> str:
    t = (text or "").strip()
    if "," in t:
        return t.split(",", 1)[1].strip()
    return t


def extract_all_json_objects(text: str) -> List[Dict[str, Any]]:
    t = text or ""
    start = t.find("{")
    if start < 0:
        return []
    depth = 0
    obj_start: Optional[int] = None
    out: List[Dict[str, Any]] = []
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                candidate = t[obj_start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        out.append(parsed)
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return out


def normalize_alliance_objects(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_question: Dict[str, Dict[str, Any]] = {}
    other_objects: List[Dict[str, Any]] = []
    for obj in objects:
        q_key = None
        for k in obj.keys():
            if re.fullmatch(r"Q\d+", str(k)):
                q_key = str(k)
                break
        if q_key is None:
            other_objects.append(obj)
            continue
        by_question[q_key] = {
            "question": obj.get(q_key),
            "score": obj.get("score"),
            "reason": obj.get("reason"),
            "raw_object": obj,
        }
    out: Dict[str, Any] = {"questions": by_question}
    if other_objects:
        out["other_objects"] = other_objects
    return out


def run_skill_eval(conversation: str, name: str, prompt_template: str, model: str) -> Dict[str, Any]:
    prompt = prompt_template.format(conversation=conversation)
    raw = call_llm_messages([{"role": "system", "content": prompt}], model=model, temperature=0.0)
    return {
        "name": name,
        "score": extract_score_prefix(raw),
        "explanation": extract_explanation_after_first_comma(raw),
        "raw": raw,
    }


def run_alliance_eval(conversation: str, model: str) -> Dict[str, Any]:
    prompt = C_ALLIANCE_SYSTEM_PROMPT.format(
        example=json.dumps(EXAMPLE_C_ALLIANCE, ensure_ascii=False),
        conversation=conversation,
    )
    raw = call_llm_messages([{"role": "system", "content": prompt}], model=model, temperature=0.0)
    parsed_objects = extract_all_json_objects(raw)
    parsed = normalize_alliance_objects(parsed_objects)
    return {
        "raw": raw,
        "parsed": parsed,
        "parsed_count": len(parsed_objects),
    }


def evaluate_rollout(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
    model: str = "gpt-4o",
) -> Path:
    payload = load_json(input_path)
    conversation_messages = payload.get("full_conversation", [])
    if not isinstance(conversation_messages, list) or not conversation_messages:
        raise ValueError("full_conversation is missing or empty in rollout JSON.")

    conversation = format_conversation(conversation_messages)
    if not conversation.strip():
        raise ValueError("Conversation text is empty after formatting.")

    skill_prompts: List[Tuple[str, str]] = [
        ("GEN_UNDERSTANDING", GEN_UNDERSTANDING),
        ("GEN_INTERPERSONAL", GEN_INTERPERSONAL),
        ("GEN_COLLABORATION", GEN_COLLABORATION),
        ("CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL", CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL),
        ("CBT_SPECIFIC_FOCUS", CBT_SPECIFIC_FOCUS),
        ("CBT_SPECIFIC_STRATEGY", CBT_SPECIFIC_STRATEGY),
    ]

    skill_results: List[Dict[str, Any]] = []
    for name, template in skill_prompts:
        print(f"Running therapist skill eval: {name}")
        skill_results.append(run_skill_eval(conversation, name, template, model))

    print("Running alliance eval")
    alliance_result = run_alliance_eval(conversation, model)

    report = {
        "source_rollout_path": str(input_path),
        "patient_id": payload.get("patient_id"),
        "patient_name": payload.get("patient_name"),
        "model": model,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "conversation_turn_count": len(conversation_messages),
        "conversation": conversation,
        "therapist_skills": skill_results,
        "alliance": alliance_result,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"selector_rollout_eval_{ts}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = evaluate_rollout()
    print(f"Saved evaluation to: {out}")

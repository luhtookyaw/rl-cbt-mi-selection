import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as: python scripts/evaluate_policy_rollouts.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.alliance import C_ALLIANCE_SYSTEM_PROMPT, EXAMPLE_C_ALLIANCE
from src.envs.therapist_skills import (
    CBT_SPECIFIC_FOCUS_SKILL,
    CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    CBT_SPECIFIC_STRATEGY_SKILL,
    GEN_COLLABORATION,
    GEN_INTERPERSONAL,
    GEN_UNDERSTANDING,
)
try:
    from src.llm import call_llm_messages
    _LLM_IMPORT_ERROR = None
except ModuleNotFoundError as e:
    call_llm_messages = None
    _LLM_IMPORT_ERROR = e


def _require_llm() -> None:
    if call_llm_messages is None:
        raise RuntimeError(
            "Failed to import LLM client. Ensure dependencies are installed (e.g., `pip install openai`) "
            f"and your environment is activated. Original error: {_LLM_IMPORT_ERROR}"
        )


def format_dialogue(convo: List[Dict[str, str]], last_n: int = 24) -> str:
    chunk = convo[-last_n:]
    out: List[str] = []
    for msg in chunk:
        role = msg.get("role", "")
        who = "Therapist" if role == "assistant" else "Client"
        out.append(f"{who}: {msg.get('content', '')}")
    return "\n".join(out)


def eval_alliance_terminal(convo: List[Dict[str, str]], model: str, window: int) -> str:
    _require_llm()
    dialogue = format_dialogue(convo, last_n=window)
    system = C_ALLIANCE_SYSTEM_PROMPT.format(
        conversation=dialogue,
        example=json.dumps(EXAMPLE_C_ALLIANCE, ensure_ascii=False),
    )

    raw = call_llm_messages([{"role": "system", "content": system}], model=model)
    return raw


def parse_alliance_formatted_output(raw: str) -> Dict[str, Any]:
    """
    Convert alliance raw text into structured JSON.
    Expected raw format is multiple JSON objects separated by blank lines.
    """
    text = (raw or "").strip()
    if not text:
        return {"items": []}

    # If model returns a single JSON object/array, parse directly.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {"items": [parsed]}
        if isinstance(parsed, list):
            return {"items": parsed}
    except Exception:
        pass

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    items: List[Dict[str, Any]] = []

    for block in blocks:
        try:
            d = json.loads(block)
            if isinstance(d, dict):
                items.append(d)
        except Exception:
            continue

    return {"items": items}


def eval_therapist_skills_terminal(convo: List[Dict[str, str]], model: str, window: int) -> Tuple[Dict[str, float], Dict[str, str]]:
    _require_llm()
    dialogue = format_dialogue(convo, last_n=window)
    prompts = {
        "guided_discovery": CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
        "focus": CBT_SPECIFIC_FOCUS_SKILL,
        "strategy": CBT_SPECIFIC_STRATEGY_SKILL,
        "understanding": GEN_UNDERSTANDING,
        "interpersonal": GEN_INTERPERSONAL,
        "collaboration": GEN_COLLABORATION,
    }

    scores: Dict[str, float] = {}
    raws: Dict[str, str] = {}

    for name, tmpl in prompts.items():
        system = tmpl.format(conversation=dialogue)
        raw = call_llm_messages([{"role": "system", "content": system}], model=model)
        raws[name] = raw

        s: Optional[float] = None
        try:
            s = float(raw.split(",", 1)[0].strip())
        except Exception:
            m = re.search(r"\b(0|2|4|6)\b", raw or "")
            s = float(m.group(1)) if m else None

        if s is not None:
            scores[name] = s

    return scores, raws


def evaluate_rollout_json(path: Path, model: str, window: int) -> Optional[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    convo = data.get("full_conversation")
    if not isinstance(convo, list) or not convo:
        return None

    alliance_raw = eval_alliance_terminal(convo, model=model, window=window)
    alliance_json = parse_alliance_formatted_output(alliance_raw)
    skills_scores, skills_raws = eval_therapist_skills_terminal(convo, model=model, window=window)
    skills_mean = (sum(skills_scores.values()) / len(skills_scores)) if skills_scores else None

    return {
        "file": str(path),
        "patient_id": data.get("patient_id"),
        "patient_name": data.get("patient_name"),
        "turn_count": len(data.get("turns", [])) if isinstance(data.get("turns"), list) else None,
        "alliance_formatted_output_json": alliance_json,
        "alliance_formatted_output_raw": alliance_raw,
        "therapist_skill_scores": skills_scores,
        "therapist_skills_mean": skills_mean,
        "therapist_skill_raws": skills_raws,
    }


def _parse_selection(selection: str, max_index: int) -> List[int]:
    s = (selection or "").strip().lower()
    if s in {"all", "a"}:
        return list(range(1, max_index + 1))

    chosen: set[int] = set()
    parts = [p.strip() for p in selection.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            for idx in range(lo, hi + 1):
                if 1 <= idx <= max_index:
                    chosen.add(idx)
        else:
            idx = int(part)
            if 1 <= idx <= max_index:
                chosen.add(idx)

    return sorted(chosen)


def _pick_files_interactively(input_paths: List[Path]) -> List[Path]:
    if len(input_paths) <= 1:
        return input_paths

    print("\nMatched rollout files:")
    for i, p in enumerate(input_paths, start=1):
        print(f"  {i}. {p}")

    print("\nSelect file(s) to evaluate:")
    print("  - Single: 2")
    print("  - Multiple: 1,3")
    print("  - Range: 2-4")
    print("  - All: all")

    while True:
        raw = input("Your selection: ").strip()
        try:
            idxs = _parse_selection(raw, len(input_paths))
        except Exception:
            idxs = []
        if idxs:
            return [input_paths[i - 1] for i in idxs]
        print("Invalid selection. Try again (e.g., 1, 2-4, all).")


def main() -> None:
    default_output = "outputs/evaluations/policy_rollout_evaluations.json"
    parser = argparse.ArgumentParser(
        description="Evaluate policy rollout conversations in outputs/*.json using therapy_env terminal evaluators."
    )
    parser.add_argument("--inputs", default="outputs/*.json", help="Glob pattern for rollout JSON files.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit file path(s) to evaluate. Overrides --inputs and skips interactive picker.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive file picker and evaluate all matched files.",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Where to save evaluation results.",
    )
    parser.add_argument("--critic-model", default="gpt-4o", help="Model used for all evaluators.")
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Number of latest dialogue turns to evaluate (therapy_env default moderator_window=24).",
    )
    args = parser.parse_args()

    if args.files:
        input_paths = [Path(p) for p in args.files]
    else:
        input_paths = sorted(Path(".").glob(args.inputs))
    if not input_paths:
        if args.files:
            raise FileNotFoundError("No valid files provided via --files.")
        raise FileNotFoundError(f"No files matched pattern: {args.inputs}")

    if not args.files and not args.no_prompt and sys.stdin.isatty():
        input_paths = _pick_files_interactively(input_paths)

    # If exactly one file is selected and user did not override --output,
    # write to outputs/evaluations/<stem>_evaluations.json
    if (
        len(input_paths) == 1
        and args.output == default_output
    ):
        single = input_paths[0]
        args.output = str(Path("outputs/evaluations") / f"{single.stem}_evaluations.json")

    results: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for p in input_paths:
        try:
            out = evaluate_rollout_json(p, model=args.critic_model, window=args.window)
            if out is None:
                skipped.append(str(p))
                continue
            results.append(out)
            print(f"Evaluated: {p}")
        except Exception as e:
            skipped.append(f"{p} ({e})")

    payload = {
        "inputs_pattern": args.inputs,
        "critic_model": args.critic_model,
        "window": args.window,
        "evaluated_count": len(results),
        "skipped": skipped,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved evaluations: {output_path}")
    print(f"Evaluated files: {len(results)}")
    print(f"Skipped files: {len(skipped)}")


if __name__ == "__main__":
    main()

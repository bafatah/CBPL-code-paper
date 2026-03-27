from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_dataset
from rules import RuleGate, SeedRulePolicy


DEFAULT_DATA_PATH = ROOT / "data" / "data_c2.json"
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "rule_based_predictions.jsonl"


def infer_grade_from_pumps(pump_count: int) -> int:
    mapping = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return mapping.get(pump_count, max(1, pump_count - 1))


def action_to_decision(action: int) -> str:
    if action > 0:
        return "increase"
    if action < 0:
        return "decrease"
    return "keep"


def final_decision_text(*, action: int, current_pumps: int, target_pumps: int) -> str:
    if action > 0:
        return f"Increase one pump level ({current_pumps} -> {target_pumps})."
    if action < 0:
        return f"Decrease one pump level ({current_pumps} -> {target_pumps})."
    return f"Keep the current pump group ({current_pumps} pumps)."


def run_rule_based(*, data_path: Path, output_path: Path, limit: int | None = None) -> None:
    policy = SeedRulePolicy()
    gate = RuleGate()
    episodes = load_dataset(data_path, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, episode in enumerate(episodes):
            recommendation = policy.recommend(episode)
            action = gate.project(recommendation.action, episode)
            target_pumps = episode.current_pumps + action
            prediction: dict[str, Any] = {
                "reasoning": recommendation.explanation,
                "decision": action_to_decision(action),
                "target_grade": infer_grade_from_pumps(target_pumps),
                "target_pumps": target_pumps,
                "one_sentence_reason": recommendation.explanation,
                "final_decision_text": final_decision_text(
                    action=action,
                    current_pumps=episode.current_pumps,
                    target_pumps=target_pumps,
                ),
            }
            row = {
                "index": index,
                "provider": "rule_based",
                "model": "deterministic_seed_rules",
                "input": episode.source_input,
                "reference_output": episode.source_output,
                "prediction": prediction,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the rule-based pump scheduling baseline over the full dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rule_based(data_path=args.data_path, output_path=args.output_path, limit=args.limit)


if __name__ == "__main__":
    main()

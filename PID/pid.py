from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_dataset
from rules import RuleGate


DEFAULT_DATA_PATH = ROOT / "data" / "data_c2.json"
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "pid_predictions.jsonl"

K_P = 1.2
K_I = 0.05
K_D = 0.2
SO2_TARGET = 20.0
DEADBAND = 2.0


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


def pid_action(*, outlet_so2: float | None, error_integral: float, previous_error: float | None) -> tuple[int, float, float]:
    if outlet_so2 is None:
        return 0, error_integral, previous_error or 0.0
    error = outlet_so2 - SO2_TARGET
    updated_integral = error_integral + error
    derivative = 0.0 if previous_error is None else error - previous_error
    control = (K_P * error) + (K_I * updated_integral) + (K_D * derivative)
    if control > DEADBAND:
        action = 1
    elif control < -DEADBAND:
        action = -1
    else:
        action = 0
    return action, updated_integral, error


def run_pid(*, data_path: Path, output_path: Path, limit: int | None = None) -> None:
    gate = RuleGate()
    episodes = load_dataset(data_path, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_integral = 0.0
    previous_error: float | None = None

    with output_path.open("w", encoding="utf-8") as handle:
        for index, episode in enumerate(episodes):
            proposed_action, error_integral, previous_error = pid_action(
                outlet_so2=episode.outlet_so2_mg_m3,
                error_integral=error_integral,
                previous_error=previous_error,
            )
            action = gate.project(proposed_action, episode)
            target_pumps = episode.current_pumps + action
            reasoning = (
                f"PID controller with Kp={K_P}, Ki={K_I}, Kd={K_D}, target={SO2_TARGET}, "
                f"deadband=±{DEADBAND} produced action {action:+d}."
            )
            prediction = {
                "reasoning": reasoning,
                "decision": action_to_decision(action),
                "target_grade": infer_grade_from_pumps(target_pumps),
                "target_pumps": target_pumps,
                "one_sentence_reason": reasoning,
                "final_decision_text": final_decision_text(
                    action=action,
                    current_pumps=episode.current_pumps,
                    target_pumps=target_pumps,
                ),
            }
            row = {
                "index": index,
                "provider": "pid",
                "model": "discrete_pid",
                "input": episode.source_input,
                "reference_output": episode.source_output,
                "prediction": prediction,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the discrete PID pump scheduling baseline over the full dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pid(data_path=args.data_path, output_path=args.output_path, limit=args.limit)


if __name__ == "__main__":
    main()

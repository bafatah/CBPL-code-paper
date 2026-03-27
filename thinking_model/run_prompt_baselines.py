from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from experiments.prompt_rules import build_system_prompt
from experiments.providers import DeepSeekChatClient, OpenAIChatClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "data_c2.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


def load_records(data_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {data_path}.")
    return [record for record in payload if isinstance(record, dict)]


def infer_default_output_path(*, provider: str, model: str) -> Path:
    safe_model = model.replace("/", "-")
    return DEFAULT_OUTPUT_DIR / f"{provider}_{safe_model}_predictions.jsonl"


def build_client(*, provider: str, model: str, api_key: str):
    if provider == "openai":
        return OpenAIChatClient(api_key=api_key, model=model)
    if provider == "deepseek":
        return DeepSeekChatClient(api_key=api_key, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_api_key(provider: str, explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
    elif provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    else:
        key = ""
    if not key:
        raise ValueError(f"Missing API key for provider '{provider}'.")
    return key


def run_records(
    *,
    client: Any,
    system_prompt: str,
    records: list[dict[str, Any]],
    limit: int | None = None,
    sleep_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    active_records = records[:limit] if limit is not None else records
    for index, record in enumerate(active_records):
        user_prompt = str(record.get("input", ""))
        started_at = time.perf_counter()
        prediction = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        latency_seconds = time.perf_counter() - started_at
        results.append(
            {
                "index": index,
                "provider": client.provider_name,
                "model": client.model,
                "input": user_prompt,
                "reference_output": record.get("output"),
                "prediction": prediction,
                "latency_seconds": latency_seconds,
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return results


def write_jsonl(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChatGPT or DeepSeek prompt baselines over the full CBPL dataset.")
    parser.add_argument("--provider", choices=("openai", "deepseek"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or ("gpt-4o" if args.provider == "openai" else "deepseek-reasoner")
    api_key = resolve_api_key(args.provider, args.api_key)
    client = build_client(provider=args.provider, model=model, api_key=api_key)
    records = load_records(args.data_path)
    system_prompt = build_system_prompt()
    results = run_records(
        client=client,
        system_prompt=system_prompt,
        records=records,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )
    output_path = args.output_path or infer_default_output_path(provider=args.provider, model=model)
    write_jsonl(results, output_path)
    print(f"Wrote {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()

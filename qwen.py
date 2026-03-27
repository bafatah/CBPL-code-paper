from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Callable
import urllib.request

from cbpl_paper.data import DecisionEpisode


Transport = Callable[[urllib.request.Request, float], dict[str, Any]]


@dataclass(frozen=True)
class QwenCBPLConfig:
    api_key: str
    model: str = "qwen3-8b"
    base_url: str = "https://coding.dashscope.aliyuncs.com/v1"
    timeout_seconds: float = 35.0
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "QwenCBPLConfig | None":
        api_key = (
            os.getenv("BAILIAN_CODING_PLAN_API_KEY", "").strip()
            or os.getenv("BAILIAN_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
            or os.getenv("QWEN_API_KEY", "").strip()
        )
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.getenv("BAILIAN_MODEL", os.getenv("QWEN_MODEL", "qwen3-8b")).strip() or "qwen3-8b",
            base_url=os.getenv(
                "BAILIAN_BASE_URL",
                os.getenv("QWEN_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
            ).strip()
            or "https://coding.dashscope.aliyuncs.com/v1",
            timeout_seconds=float(os.getenv("CBPL_QWEN_TIMEOUT_SECONDS", "35")),
            temperature=float(os.getenv("CBPL_QWEN_TEMPERATURE", "0.2")),
        )


@dataclass(frozen=True)
class QwenCBPLResult:
    reasoning: str
    proposed_action: int
    expected_observation: str


class QwenCBPLClient:
    def __init__(self, *, config: QwenCBPLConfig, transport: Transport | None = None) -> None:
        self.config = config
        self._transport = transport or self._default_transport

    def complete(self, prompt: str) -> QwenCBPLResult:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Qwen decision module for Case-Based Prompt Learning. "
                        "Return JSON only with keys reasoning, proposed_action, expected_observation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        response_payload = self._transport(request, self.config.timeout_seconds)
        result_payload = self._extract_payload(response_payload)
        return QwenCBPLResult(
            reasoning=str(result_payload.get("reasoning", "")).strip(),
            proposed_action=self._coerce_action(result_payload.get("proposed_action", 0)),
            expected_observation=str(result_payload.get("expected_observation", "")).strip(),
        )

    def _default_transport(self, request: urllib.request.Request, timeout: float) -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extract_payload(self, response_payload: dict[str, Any]) -> dict[str, Any]:
        choices = response_payload.get("choices", [])
        if not choices:
            raise ValueError("Qwen response did not contain choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return self._coerce_json_object(content)
        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
            if text.strip():
                return self._coerce_json_object(text)
        raise ValueError("Qwen response did not contain JSON content.")

    def _coerce_json_object(self, text: str) -> dict[str, Any]:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Qwen response JSON must be an object.")
        return payload

    def _coerce_action(self, value: Any) -> int:
        if isinstance(value, int):
            parsed = value
        else:
            normalized = str(value).strip().lower()
            if normalized in {"+1", "1", "increase", "add"}:
                parsed = 1
            elif normalized in {"-1", "decrease", "reduce"}:
                parsed = -1
            else:
                parsed = 0
        if parsed > 0:
            return 1
        if parsed < 0:
            return -1
        return 0


class QwenCBPLDecider:
    def __init__(self, *, client: QwenCBPLClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        episode: DecisionEpisode,
        prompt: str,
        retrieved_cases: list[Any],
        seed_action: int,
    ) -> tuple[int, str]:
        result = self.client.complete(prompt)
        return result.proposed_action, result.reasoning or f"Qwen proposed action {result.proposed_action:+d}."

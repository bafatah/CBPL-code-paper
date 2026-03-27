from __future__ import annotations

import json
import re
from typing import Any, Callable
import urllib.request


Transport = Callable[[urllib.request.Request, float], dict[str, Any]]


class BaseChatClient:
    provider_name = "base"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 60.0,
        temperature: float = 0.1,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self._transport = transport or self._default_transport

    def complete(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}{self.chat_path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response_payload = self._transport(request, self.timeout_seconds)
        content = self._extract_content(response_payload)
        result = self._parse_json_content(content)
        result["decision"] = self._normalize_decision(result.get("decision"))
        result["raw_content"] = content
        return result

    @property
    def chat_path(self) -> str:
        return "/chat/completions"

    def _default_transport(self, request: urllib.request.Request, timeout: float) -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extract_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError("Model response did not contain choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        raise ValueError("Model response did not contain text content.")

    def _parse_json_content(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("Could not find a JSON object in model output.")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Model output JSON must be an object.")
        return parsed

    def _normalize_decision(self, value: Any) -> str:
        normalized = str(value).strip().lower()
        mapping = {
            "increase": "increase",
            "add": "increase",
            "增泵": "increase",
            "增加": "increase",
            "keep": "keep",
            "hold": "keep",
            "维持": "keep",
            "decrease": "decrease",
            "reduce": "decrease",
            "减泵": "decrease",
            "减少": "decrease",
        }
        return mapping.get(normalized, normalized)


class OpenAIChatClient(BaseChatClient):
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        temperature: float = 0.1,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            transport=transport,
        )


class DeepSeekChatClient(BaseChatClient):
    provider_name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        temperature: float = 0.1,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            transport=transport,
        )

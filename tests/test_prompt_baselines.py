from __future__ import annotations

import json

from experiments.prompt_rules import PUMP_SCHEDULING_RULES, build_system_prompt
from experiments.providers import DeepSeekChatClient, OpenAIChatClient
from experiments.run_prompt_baselines import infer_default_output_path, run_records


class FakeDecisionClient:
    provider_name = "fake"
    model = "fake-model"

    def complete(self, *, system_prompt: str, user_prompt: str):
        return {
            "reasoning": "rules support holding the current grade",
            "decision": "keep",
            "target_grade": 2,
            "target_pumps": 3,
            "one_sentence_reason": "SO2 remains in the keep band.",
            "final_decision_text": "Keep the current pump group.",
            "raw_content": "{\"decision\":\"keep\"}",
        }


def test_build_system_prompt_includes_rules_and_json_contract() -> None:
    prompt = build_system_prompt()

    assert PUMP_SCHEDULING_RULES[0] in prompt
    assert PUMP_SCHEDULING_RULES[-1] in prompt
    assert "Return JSON only" in prompt
    assert '"decision"' in prompt
    assert '"target_pumps"' in prompt


def test_openai_client_builds_chat_completion_request_and_parses_json_response() -> None:
    seen: dict[str, object] = {}

    def transport(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reasoning": "Outlet SO2 stays in the keep range.",
                                "decision": "keep",
                                "target_grade": 2,
                                "target_pumps": 3,
                                "one_sentence_reason": "Hold the current pump group.",
                                "final_decision_text": "Keep the current pump group.",
                            }
                        )
                    }
                }
            ]
        }

    client = OpenAIChatClient(api_key="test-key", model="gpt-4o", transport=transport)
    result = client.complete(system_prompt="sys", user_prompt="user")

    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["body"]["model"] == "gpt-4o"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["messages"][1]["role"] == "user"
    assert result["decision"] == "keep"
    assert result["target_pumps"] == 3


def test_deepseek_client_builds_chat_completion_request_and_normalizes_decision() -> None:
    seen: dict[str, object] = {}

    def transport(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reasoning": "Outlet SO2 exceeds the strengthening trigger.",
                                "decision": "increase",
                                "target_grade": 3,
                                "target_pumps": 4,
                                "one_sentence_reason": "Increase one grade to avoid exceedance.",
                                "final_decision_text": "Increase one pump level.",
                            }
                        )
                    }
                }
            ]
        }

    client = DeepSeekChatClient(api_key="test-key", model="deepseek-reasoner", transport=transport)
    result = client.complete(system_prompt="sys", user_prompt="user")

    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["body"]["model"] == "deepseek-reasoner"
    assert result["decision"] == "increase"
    assert result["target_grade"] == 3


def test_run_records_generates_one_output_row_per_input_record() -> None:
    results = run_records(
        client=FakeDecisionClient(),
        system_prompt="rules",
        records=[
            {"input": "state A", "output": "reference A"},
            {"input": "state B", "output": "reference B"},
        ],
    )

    assert len(results) == 2
    assert results[0]["provider"] == "fake"
    assert results[0]["prediction"]["decision"] == "keep"
    assert results[1]["input"] == "state B"
    assert results[1]["reference_output"] == "reference B"


def test_infer_default_output_path_uses_provider_and_model_name() -> None:
    output_path = infer_default_output_path(provider="openai", model="gpt-4o-mini")

    assert output_path.name == "openai_gpt-4o-mini_predictions.jsonl"
    assert output_path.parent.name == "outputs"

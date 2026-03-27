from __future__ import annotations

import json

from cbpl_paper.data import DecisionEpisode
from cbpl_paper.qwen import QwenCBPLClient, QwenCBPLConfig, QwenCBPLDecider


def make_episode() -> DecisionEpisode:
    return DecisionEpisode(
        episode_id="episode-1",
        source_input="",
        source_output="",
        state_summary="当前出口SO2为38.49 mg/m3，当前2台泵运行。",
        current_pumps=2,
        total_power_kw=1800.0,
        load_mw=381.2,
        inlet_so2_mg_m3=1143.2,
        outlet_so2_mg_m3=38.49,
        outlet_trend=5.34,
        slurry_flow_m3_h=0.02,
        ph=5.45,
        expert_action=1,
        target_pumps=3,
        rationale="",
    )


def test_qwen_config_reads_api_key_and_model_from_env(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "secret")
    monkeypatch.setenv("QWEN_MODEL", "qwen3-8b")

    config = QwenCBPLConfig.from_env()

    assert config is not None
    assert config.api_key == "secret"
    assert config.model == "qwen3-8b"
    assert config.base_url == "https://coding.dashscope.aliyuncs.com/v1"


def test_qwen_client_extracts_action_and_reasoning_from_completion_payload() -> None:
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "reasoning": "出口SO2超标并上升，必须增泵。",
                            "proposed_action": 1,
                            "expected_observation": "出口SO2应回落。",
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }

    client = QwenCBPLClient(
        config=QwenCBPLConfig(api_key="test-key"),
        transport=lambda request, timeout: response_payload,
    )
    result = client.complete("prompt")

    assert result.proposed_action == 1
    assert "必须增泵" in result.reasoning
    assert "应回落" in result.expected_observation


def test_qwen_decider_uses_prompt_and_returns_model_action() -> None:
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "reasoning": "检索案例和当前状态都支持增泵。",
                                    "proposed_action": "+1",
                                    "expected_observation": "SO2下降。",
                                },
                                ensure_ascii=False,
                            )
                        }
                    ]
                }
            }
        ]
    }
    seen: dict[str, str] = {}

    def transport(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = request.data.decode("utf-8")
        return response_payload

    decider = QwenCBPLDecider(
        client=QwenCBPLClient(
            config=QwenCBPLConfig(api_key="test-key", model="qwen3-8b"),
            transport=transport,
        )
    )

    action, explanation = decider.decide(
        episode=make_episode(),
        prompt="CBPL prompt body",
        retrieved_cases=[],
        seed_action=0,
    )

    assert action == 1
    assert "支持增泵" in explanation
    assert seen["url"].endswith("/chat/completions")
    assert "CBPL prompt body" in seen["body"]

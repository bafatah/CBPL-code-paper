from __future__ import annotations

from sft_qwen3_8b import (
    DEFAULT_SYSTEM_PROMPT,
    SFTScriptConfig,
    build_text_dataset,
    conversation_from_record,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        conversations,
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ):
        self.calls.append(
            {
                "conversations": conversations,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return [f"chat::{index}" for index, _ in enumerate(conversations)]


def test_conversation_from_record_matches_notebook_roles() -> None:
    conversation = conversation_from_record(
        {
            "input": "current state",
            "output": "final decision",
        }
    )

    assert conversation == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "current state"},
        {"role": "assistant", "content": "final decision"},
    ]


def test_build_text_dataset_applies_chat_template_without_generation_prompt() -> None:
    tokenizer = DummyTokenizer()

    dataset = build_text_dataset(
        tokenizer=tokenizer,
        records=[
            {"input": "state A", "output": "decision A"},
            {"input": "state B", "output": "decision B"},
        ],
    )

    assert dataset["text"] == ["chat::0", "chat::1"]
    assert tokenizer.calls[0]["tokenize"] is False
    assert tokenizer.calls[0]["add_generation_prompt"] is False
    assert tokenizer.calls[0]["enable_thinking"] is False
    conversations = tokenizer.calls[0]["conversations"]
    assert conversations[0][1]["content"] == "state A"
    assert conversations[1][2]["content"] == "decision B"


def test_config_defaults_match_repo_layout() -> None:
    config = SFTScriptConfig()

    assert config.data_path.name == "data_c2.json"
    assert config.data_path.parent.name == "data"
    assert config.model_name == "unsloth/Qwen3-8B-unsloth-bnb-4bit"

from cbpl_paper.data import DecisionEpisode
from cbpl_paper.guidebook import Guidebook, LessonUpdate
from cbpl_paper.memory import CaseRecord, CaseRetriever
from cbpl_paper.prompting import PromptComposer


def make_episode(episode_id: str, summary: str) -> DecisionEpisode:
    return DecisionEpisode(
        episode_id=episode_id,
        source_input=summary,
        source_output="",
        state_summary=summary,
        current_pumps=3,
        total_power_kw=2700.0,
        load_mw=320.0,
        inlet_so2_mg_m3=1100.0,
        outlet_so2_mg_m3=12.0,
        outlet_trend=0.0,
        slurry_flow_m3_h=10.0,
        ph=5.3,
        expert_action=0,
        target_pumps=3,
        rationale="保持当前泵数。",
    )


def test_case_retriever_returns_semantically_closest_cases() -> None:
    retriever = CaseRetriever()
    cases = [
        CaseRecord.from_episode(make_episode("a", "出口SO2快速上升，当前2台泵处理能力紧张。"), reasoning="增加一台泵"),
        CaseRecord.from_episode(make_episode("b", "出口SO2极低且下降，当前3台泵能力过剩。"), reasoning="减少一台泵"),
        CaseRecord.from_episode(make_episode("c", "出口SO2稳定在适宜区间，维持当前泵数。"), reasoning="维持"),
    ]
    retriever.fit(cases)

    hits = retriever.retrieve("出口SO2快速上升，2台泵压力较大，需要提升脱硫能力。", top_k=2)

    assert [hit.case.episode_id for hit in hits] == ["a", "c"]
    assert hits[0].score > hits[1].score


def test_guidebook_updates_lesson_weight_and_supports_downgrade() -> None:
    guidebook = Guidebook()

    added = guidebook.apply(LessonUpdate(candidate="高出口SO2并快速上升时优先增泵。", op="ADD"))
    edited = guidebook.apply(LessonUpdate(candidate="高出口SO2快速上升时优先增泵并检查趋势。", op="EDIT"))
    downgraded = guidebook.apply(LessonUpdate(candidate="高出口SO2快速上升时优先增泵并检查趋势。", op="DOWNGRADE"))

    assert added.weight == 2
    assert edited.weight == 3
    assert downgraded.weight == 2
    assert "检查趋势" in downgraded.text


def test_prompt_composer_includes_seed_rules_lessons_cases_and_state() -> None:
    guidebook = Guidebook()
    guidebook.apply(LessonUpdate(candidate="当出口SO2低于10且下级泵组满足条件时可降一档。", op="ADD"))
    case = CaseRecord.from_episode(
        make_episode("case-1", "出口SO2极低且下降，当前3台泵能力过剩。"),
        reasoning="减少一台泵（当前 3 台 → 2 台）",
    )
    prompt = PromptComposer().compose(
        current_state="当前出口SO2为5.09 mg/m3，3台泵运行，pH为5.19。",
        retrieved_cases=[case],
        guidebook=guidebook,
    )

    assert "Seed Safety Rules" in prompt
    assert "Lesson Guidebook" in prompt
    assert "Retrieved Cases" in prompt
    assert "Current State" in prompt
    assert "3 台 → 2 台" in prompt

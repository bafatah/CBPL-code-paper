from cbpl_paper.data import DecisionEpisode
from cbpl_paper.rules import GradeRuleBook, RuleConfig, RuleGate, SeedRulePolicy


def make_episode(
    *,
    current_pumps: int,
    load: float,
    inlet: float,
    outlet: float,
    ph: float,
    outlet_trend: float = 0.0,
) -> DecisionEpisode:
    return DecisionEpisode(
        episode_id="ep-1",
        source_input="",
        source_output="",
        state_summary="summary",
        current_pumps=current_pumps,
        total_power_kw=None,
        load_mw=load,
        inlet_so2_mg_m3=inlet,
        outlet_so2_mg_m3=outlet,
        outlet_trend=outlet_trend,
        slurry_flow_m3_h=12.0,
        ph=ph,
        expert_action=0,
        target_pumps=current_pumps,
        rationale="",
    )


def test_rule_gate_blocks_three_to_two_drop_when_ph_is_below_guard() -> None:
    gate = RuleGate(RuleConfig(min_pumps=2, max_pumps=6, ph_drop_guard=5.1))
    episode = make_episode(current_pumps=3, load=357.0, inlet=1127.9, outlet=7.46, ph=4.93, outlet_trend=0.2)

    projected = gate.project(proposed_action=-1, episode=episode)

    assert projected == 0


def test_rule_gate_clamps_to_one_step_and_respects_max_pumps() -> None:
    gate = RuleGate(RuleConfig(min_pumps=2, max_pumps=6, ph_drop_guard=5.1))
    episode = make_episode(current_pumps=6, load=560.0, inlet=2500.0, outlet=42.0, ph=5.5)

    projected = gate.project(proposed_action=3, episode=episode)

    assert projected == 0


def test_seed_policy_recommends_decrease_only_when_lower_grade_fits() -> None:
    policy = SeedRulePolicy(grade_book=GradeRuleBook.default())
    episode = make_episode(current_pumps=3, load=303.9, inlet=923.4, outlet=9.23, ph=5.57, outlet_trend=-0.6)

    decision = policy.recommend(episode)

    assert decision.action == -1
    assert decision.reason_code == "surplus_margin"


def test_seed_policy_keeps_when_lower_grade_would_not_fit() -> None:
    policy = SeedRulePolicy(grade_book=GradeRuleBook.default())
    episode = make_episode(current_pumps=3, load=315.0, inlet=1051.6, outlet=9.82, ph=5.19, outlet_trend=-0.2)

    decision = policy.recommend(episode)

    assert decision.action == 0
    assert decision.reason_code == "hold_grade"

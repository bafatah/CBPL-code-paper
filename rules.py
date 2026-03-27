from __future__ import annotations

from dataclasses import dataclass
from math import inf

from cbpl_paper.data import DecisionEpisode


@dataclass(frozen=True)
class RuleConfig:
    min_pumps: int = 2
    max_pumps: int = 6
    ph_drop_guard: float = 5.1
    outlet_increase_threshold: float = 35.0
    outlet_hold_upper: float = 30.0
    outlet_reduce_threshold: float = 10.0


@dataclass(frozen=True)
class RuleRecommendation:
    action: int
    reason_code: str
    explanation: str


@dataclass(frozen=True)
class GradeRule:
    pump_count: int
    max_load_mw: float
    max_inlet_so2_mg_m3: float

    def fits(self, *, load_mw: float | None, inlet_so2_mg_m3: float | None) -> bool:
        if load_mw is None or inlet_so2_mg_m3 is None:
            return False
        return load_mw < self.max_load_mw and inlet_so2_mg_m3 < self.max_inlet_so2_mg_m3


class GradeRuleBook:
    def __init__(self, rules: dict[int, GradeRule]) -> None:
        self._rules = dict(rules)

    @classmethod
    def default(cls) -> "GradeRuleBook":
        return cls(
            {
                2: GradeRule(pump_count=2, max_load_mw=320.0, max_inlet_so2_mg_m3=1000.0),
                3: GradeRule(pump_count=3, max_load_mw=500.0, max_inlet_so2_mg_m3=1900.0),
                4: GradeRule(pump_count=4, max_load_mw=550.0, max_inlet_so2_mg_m3=2200.0),
                5: GradeRule(pump_count=5, max_load_mw=600.0, max_inlet_so2_mg_m3=2400.0),
                6: GradeRule(pump_count=6, max_load_mw=inf, max_inlet_so2_mg_m3=inf),
            }
        )

    def can_support(self, *, pump_count: int, load_mw: float | None, inlet_so2_mg_m3: float | None) -> bool:
        rule = self._rules.get(pump_count)
        if rule is None:
            return False
        return rule.fits(load_mw=load_mw, inlet_so2_mg_m3=inlet_so2_mg_m3)

    def seed_rules(self) -> list[str]:
        return [
            "Actions are incremental and limited to {-1, 0, +1}.",
            "Never go below 2 running pumps or above 6 running pumps.",
            "Never reduce from 3 pumps to 2 pumps when slurry pH is below 5.1.",
            "If outlet SO2 is above 35 mg/m3, strengthen by exactly one pump level.",
            "If outlet SO2 is between 10 and 30 mg/m3, prefer holding the current level.",
            "If outlet SO2 is below 10 mg/m3 and the lower pump grade fully fits load and inlet SO2, one-step reduction is allowed.",
            "Use grade conditions: 2 pumps <320 MW and <1000 mg/m3 inlet; 3 pumps <500 MW and <1900; 4 pumps <550 MW and <2200; 5 pumps <600 MW and <2400.",
        ]


class RuleGate:
    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()

    def admissible_actions(self, episode: DecisionEpisode) -> set[int]:
        actions = {0}
        if episode.current_pumps > self.config.min_pumps:
            target = episode.current_pumps - 1
            if not (episode.current_pumps == 3 and target == 2 and (episode.ph is None or episode.ph < self.config.ph_drop_guard)):
                actions.add(-1)
        if episode.current_pumps < self.config.max_pumps:
            actions.add(1)
        return actions

    def project(self, proposed_action: int, episode: DecisionEpisode) -> int:
        if proposed_action > 0:
            action = 1
        elif proposed_action < 0:
            action = -1
        else:
            action = 0
        return action if action in self.admissible_actions(episode) else 0


class SeedRulePolicy:
    def __init__(self, *, grade_book: GradeRuleBook | None = None, config: RuleConfig | None = None) -> None:
        self.grade_book = grade_book or GradeRuleBook.default()
        self.config = config or RuleConfig()

    def recommend(self, episode: DecisionEpisode) -> RuleRecommendation:
        outlet = episode.outlet_so2_mg_m3
        if outlet is None:
            return RuleRecommendation(0, "missing_outlet", "Missing outlet SO2, so keep the current pump count.")
        if outlet >= self.config.outlet_increase_threshold:
            return RuleRecommendation(1, "high_so2", "Outlet SO2 exceeded the strengthen threshold.")
        if (
            outlet < self.config.outlet_reduce_threshold
            and episode.current_pumps > self.config.min_pumps
            and (episode.outlet_trend is None or episode.outlet_trend <= 0)
            and (episode.ph is not None and episode.ph >= self.config.ph_drop_guard)
            and self.grade_book.can_support(
                pump_count=episode.current_pumps - 1,
                load_mw=episode.load_mw,
                inlet_so2_mg_m3=episode.inlet_so2_mg_m3,
            )
        ):
            return RuleRecommendation(-1, "surplus_margin", "Lower outlet SO2 and a valid lower grade permit one-step reduction.")
        return RuleRecommendation(0, "hold_grade", "Current conditions do not justify changing the pump level.")

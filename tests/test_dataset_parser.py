from cbpl_paper.data import DatasetEpisodeParser


SAMPLE_INPUT = """### 脱硫系统实时状态
**时间范围**: 2024-06-12 15:24:00 至 2024-06-12 15:27:00
**数据窗口**: 最近 18 个采样点 (3.0 分钟)

---

## 🔵 出口 SO₂ 全量时间序列（mg/m³）
[7.10, 7.05, 6.98, 6.90]

---

## 📈 当前运行参数
• **负荷**: 316.8 MW
• **入口SO₂**: 996.6 mg/m³
• **出口SO₂**: 5.09 mg/m³ (趋势 -2.13)
• **浆液流量**: 45.3 m³/h
• **石膏PH**: 5.19

**当前配置**: 3 台泵 | 总功率: 2700 kW
"""

SAMPLE_OUTPUT = """## 情况评估
- 当前处于较低负荷（316.8 MW）。
- 入口SO₂为低浓度（996.6 mg/m³）。
- 当前出口 SO₂：5.09 mg/m³，趋势快速下降。

## 一句话理由
出口SO₂极低且快速下降，当前3台泵运行时脱硫能力充足；减泵至2台后工况仍可安全处理，故执行减泵优化。

## 最终决策
减少一台泵（当前 3 台 → 2 台）
"""


def test_parser_extracts_current_state_and_expert_decision() -> None:
    episode = DatasetEpisodeParser().parse_record({"input": SAMPLE_INPUT, "output": SAMPLE_OUTPUT}, index=11)

    assert episode.episode_id == "episode-00011"
    assert episode.current_pumps == 3
    assert episode.total_power_kw == 2700.0
    assert episode.load_mw == 316.8
    assert episode.inlet_so2_mg_m3 == 996.6
    assert episode.outlet_so2_mg_m3 == 5.09
    assert episode.outlet_trend == -2.13
    assert episode.slurry_flow_m3_h == 45.3
    assert episode.ph == 5.19
    assert episode.expert_action == -1
    assert episode.target_pumps == 2
    assert "减泵优化" in episode.rationale


def test_parser_detects_keep_action_when_target_is_unchanged() -> None:
    keep_output = """## 一句话理由
当前工况与泵数匹配良好，因此维持。

## 最终决策
维持当前泵数（3 台）
"""
    episode = DatasetEpisodeParser().parse_record({"input": SAMPLE_INPUT, "output": keep_output}, index=2)

    assert episode.expert_action == 0
    assert episode.target_pumps == 3

# CBPL Paper Code

**Paper Title:** Case-Based Prompt Learning for SO2 Pump Scheduling

Wet flue gas desulfurization (WFGD) plants must schedule circulation pumps to keep outlet SO2 within limits while limiting auxiliary power and equipment wear. This supervisory task is difficult because measurements are noisy, operating conditions drift, and every action must satisfy hard plant rules. Existing rule-based controllers are easy to certify but adapt poorly, whereas fine-tuned models require repeated retraining and provide limited operational justification. We address this gap with Case-Based Prompt Learning (CBPL), a case-based decision framework in which a large language model serves as the adaptation engine rather than the memory itself. CBPL stores each decision episode as a case, retrieves semantically similar operating precedents to ground the next action, and updates a Lesson Guidebook that consolidates repeatedly supported adaptations without modifying model weights. This design preserves the core CBR properties of precedent-based reuse, revision, and auditability because each recommendation can be traced to retrieved cases, explicit safety rules, and consolidated lessons. On 1,714 decision episodes distilled from roughly 3 million 10-second measurements from three WFGD units, CBPL improves action accuracy by 17.8 percentage points over a rule-based baseline and by 6.3 percentage points over PID, while maintaining 98.7% rule compliance and reducing switch rate by 54.4% relative to PID.

- case memory over decision episodes
- retrieval over compact textual state summaries
- a lesson guidebook with `ADD`, `EDIT`, `UPGRADE`, and `DOWNGRADE`
- a prompt composer that assembles seed rules, lessons, retrieved cases, and the current state
- a rule gate that enforces one-step actions plus the pH and minimum-pump constraints
- a lightweight local decider so the pipeline is runnable without reproducing the paper experiments
- an optional Qwen/DashScope-backed decider for paper-aligned prompt inference

## Layout

- `data.py`: parses the JSON supervision dataset into structured decision episodes
- `rules.py`: seed safety rules, grade fit checks, and the rule gate
- `memory.py`: case records and semantic retrieval
- `guidebook.py`: lesson consolidation
- `prompting.py`: prompt assembly
- `engine.py`: end-to-end CBPL orchestration
- `qwen.py`: optional Qwen client and decider
- `tests/`: focused unit tests for the core paper logic

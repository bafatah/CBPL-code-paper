# CBPL Paper Code

**Paper Title:** Case-Based Prompt Learning for SO2 Pump Scheduling

This directory contains a clean Python implementation of the paper method described in `work_iccbr/6658.tex`:

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

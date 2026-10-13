"""Build CausalTrustBench: 200 base tasks x 4 causal variants = 800 matched episodes.

Splits are made at the base-task level (stratified by source) so the four variants of
a task never cross train/dev/test.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import AGENTS, CAUSES, NEARBY, SOURCE_SKILL, UNRELATED, ExperimentConfig
from .loaders import load_source
from .schema import Episode, Task

SPECIALIST = {skill: agent for agent, skill in AGENTS.items()}


def out_of_specialty_agent(skill: str) -> str:
    """Deterministic non-specialist routing: the specialist of the unrelated skill."""
    return SPECIALIST[UNRELATED[skill]]


def split_base_tasks(tasks: list[Task], ratios, seed) -> dict[str, str]:
    rng = random.Random(seed)
    by_src: dict[str, list[Task]] = {}
    for t in tasks:
        by_src.setdefault(t.source, []).append(t)
    assign = {}
    for src, ts in by_src.items():
        ts = sorted(ts, key=lambda t: t.task_id)
        rng.shuffle(ts)
        n = len(ts)
        n_tr, n_dev = round(ratios[0] * n), round(ratios[1] * n)
        for i, t in enumerate(ts):
            assign[t.task_id] = "train" if i < n_tr else "dev" if i < n_tr + n_dev else "test"
    return assign


def make_episode(task: Task, cause: str, split: str, rng: random.Random) -> Episode:
    skill = task.skill
    agent = SPECIALIST[skill]
    withheld, tool_state, note = [], "ok", ""
    if cause == "missing_context":
        # withhold one required evidence item (for tool tasks: the tool result itself is
        # replaced by a context gap, not a tool error)
        withheld = [rng.choice(task.gold_evidence_ids)]
    elif cause == "tool_failure":
        # dbtool: stale DB snapshot; gsm8k: faulty calculator; fever/hotpot: retrieval 503
        tool_state = "stale" if task.source in ("dbtool", "gsm8k") else "error_empty"
    elif cause == "out_of_specialty":
        agent = out_of_specialty_agent(skill)
        note = f"Routed {skill} task to {agent} (declared specialty: {AGENTS[agent]})."
    return Episode(
        episode_id=f"{task.task_id}::{cause}", base_task=task, cause=cause,
        failing_agent=agent, affected_skill=skill,
        persistent=(cause == "persistent_reasoning"), split=split,
        withheld_evidence_ids=withheld, tool_state=tool_state, routing_note=note)


def attach_probes(ep: Episode, pools: dict[str, list[Task]], cfg: ExperimentConfig, rng) -> None:
    k = ep.affected_skill
    slots = ([("affected", k)] * cfg.probes_affected + [("nearby", NEARBY[k])] * cfg.probes_nearby
             + [("unrelated", UNRELATED[k])] * cfg.probes_unrelated)
    used = set()
    for slot, skill in slots:
        pool = [t for t in pools[skill] if t.task_id not in used]
        t = rng.choice(pool)
        used.add(t.task_id)
        ep.probes.append({"slot": slot, "skill": skill, "task": t})


def build_benchmark(cfg: ExperimentConfig, offline: bool = False) -> dict:
    rng = random.Random(cfg.data_seed)
    sources = list(SOURCE_SKILL)
    n_total = cfg.tasks_per_source + cfg.probe_pool_per_source + cfg.calibration_per_skill
    base, probe_pool, calib = [], {}, {}
    for src in sources:
        ts = load_source(src, n_total, cfg.data_seed, offline=offline)
        rng.shuffle(ts)
        skill = SOURCE_SKILL[src]
        base += ts[:cfg.tasks_per_source]
        probe_pool[skill] = ts[cfg.tasks_per_source:cfg.tasks_per_source + cfg.probe_pool_per_source]
        calib[skill] = ts[cfg.tasks_per_source + cfg.probe_pool_per_source:]
    split = split_base_tasks(base, cfg.split, cfg.data_seed)
    episodes = []
    for t in base:
        for c in CAUSES:
            ep = make_episode(t, c, split[t.task_id], rng)
            # probes are matched across the 4 causal variants of the same base task
            attach_probes(ep, probe_pool, cfg, random.Random(f"{cfg.data_seed}-{t.task_id}"))
            episodes.append(ep)
    return {"episodes": episodes, "calibration": calib, "probe_pool": probe_pool}


def episode_record(ep: Episode) -> dict:
    """Flat record for release (matches the 'Episode record' list in the proposal)."""
    t = ep.base_task
    return {
        "episode_id": ep.episode_id, "task_id": t.task_id, "skill": t.skill,
        "correct_answer": t.gold_answer, "target_wrong_recommendation": t.wrong_answer,
        "subagent": ep.failing_agent, "subagent_profile": AGENTS[ep.failing_agent],
        "available_evidence_ids": [e.eid for e in t.evidence if e.eid not in ep.withheld_evidence_ids],
        "withheld_evidence_ids": ep.withheld_evidence_ids, "tool_state": ep.tool_state,
        "causal_condition": ep.cause, "persistent": ep.persistent, "split": ep.split,
        "probe_task_ids": [p["task"].task_id for p in ep.probes],
        "probe_slots": [p["slot"] for p in ep.probes],
    }


def save_benchmark(bench: dict, out_dir: str) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "episodes.jsonl", "w") as f:
        for ep in bench["episodes"]:
            f.write(json.dumps(episode_record(ep)) + "\n")

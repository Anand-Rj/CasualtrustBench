"""Phases A-E of the experimental protocol."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import string
from pathlib import Path

from . import prompts as P
from .config import AGENTS, SKILLS, ExperimentConfig
from .data.episodes import SPECIALIST
from .data.schema import Episode, Task
from .llm.backend import Backend, parse_json
from .memory import build_failure_graph, render_memory, run_attribution
from .trust import (degrade_prob, expected_utility, oracle_action, oracle_delta,
                    post_failure_reliability, public_profile)


# --------------------------------------------------------------------------- answer matching
def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation or ch in ".-")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(pred, task: Task) -> bool:
    if pred is None:
        return False
    if task.answer_format == "number":
        try:
            return abs(float(str(pred).replace(",", "").strip()) - float(task.gold_answer)) < 1e-6
        except ValueError:
            return False
    if task.answer_format == "label":
        return str(pred).strip().upper().startswith(task.gold_answer)
    g, p = _norm(task.gold_answer), _norm(pred)
    return g == p or (len(g) > 3 and g in p)


def _u(*parts) -> float:
    """Deterministic uniform(0,1) from a key; keeps environment noise matched across conditions."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def full_view(task: Task) -> tuple[list[str], str | None]:
    ids = [e.eid for e in task.evidence]
    tool = None
    if task.source == "dbtool":
        tool = f"sql_query -> Result: {task.tool['fresh_result']}"
    return ids, tool


async def run_subagent(be: Backend, agent: str, task: Task, ids, tool, seed, scope,
                       mock_p=None) -> tuple[str, dict]:
    sys = P.SUBAGENT_SYSTEM.format(agent=agent, specialty=AGENTS[agent])
    meta = {"gold": task.gold_answer, "wrong": task.wrong_answer,
            "p_correct": mock_p if mock_p is not None else 0.7}
    out = parse_json(await be.complete("subagent", sys, P.subagent_prompt(task, ids, tool),
                                       seed=seed, meta=meta, cache_scope=scope))
    return str(out.get("answer", "")), out


# Mock "true" competence; only used by MockBackend to simulate answers.
MOCK_COMPETENCE = {"spec": 0.85, "off": 0.50}


def mock_p(agent, skill):
    return MOCK_COMPETENCE["spec"] if AGENTS[agent] == skill else MOCK_COMPETENCE["off"]


# --------------------------------------------------------------------------- Phase A
async def phase_a(be: Backend, calib: dict[str, list[Task]], cfg: ExperimentConfig, seed=0):
    """Estimate R[agent][skill] on held-out calibration tasks."""
    jobs, keys = [], []
    for a in AGENTS:
        for k in SKILLS:
            for t in calib[k][:cfg.calibration_per_skill]:
                ids, tool = full_view(t)
                jobs.append(run_subagent(be, a, t, ids, tool, seed, f"calib-{a}", mock_p(a, k)))
                keys.append((a, k, t))
    res = await asyncio.gather(*jobs)
    hits: dict = {}
    for (a, k, t), (ans, _) in zip(keys, res):
        hits.setdefault(a, {}).setdefault(k, []).append(is_correct(ans, t))
    # Laplace-smoothed accuracy
    return {a: {k: round((sum(v) + 1) / (len(v) + 2), 4) for k, v in d.items()} for a, d in hits.items()}


# --------------------------------------------------------------------------- Phases B-E
async def run_episode(be: Backend, ep: Episode, R: dict, condition: str, seed: int,
                      cfg: ExperimentConfig, ablation: str | None = None) -> dict | None:
    pub = public_profile(R)
    agent, skill = ep.failing_agent, ep.affected_skill
    t = ep.base_task

    # ---- Phase B: controlled failure (recommendation + rejection fixed) ----
    graph = build_failure_graph(ep)
    if cfg.failure_mode == "induce":
        from .memory import seen_evidence
        tool = None
        if t.source == "dbtool" and "tool_result" in seen_evidence(ep):
            tool = f"sql_query -> Result: {t.tool['stale_result' if ep.tool_state == 'stale' else 'fresh_result']}"
        ans, _ = await run_subagent(be, agent, t, seen_evidence(ep), tool, seed,
                                    f"induce-{ep.episode_id}", 0.2)
        if is_correct(ans, t):
            return None   # failure not induced; episode dropped for this seed

    # ---- Phase C: attribution (rule-based, from provenance only) ----
    attribution = run_attribution(ep, graph)

    # ---- prior trust (shared across conditions; cached) ----
    prior_raw = parse_json(await be.complete(
        "elicit_prior", P.SUPERVISOR_SYSTEM, P.prior_prompt(pub, agent), seed=seed,
        meta={"prior": pub[agent]}, cache_scope="prior"))
    T0 = {k: float(prior_raw.get("trust", {}).get(k, pub[agent][k])) for k in SKILLS}

    # observed reliability on the routed skill (what the update rule may use)
    obs = R[agent][skill] if attribution["cause"] == "out_of_specialty" else None
    mem = render_memory(condition, ep, graph, attribution, T0, ablation, obs)

    # ---- Phase D: trust update ----
    att = parse_json(await be.complete(
        "attribute_trust", P.SUPERVISOR_SYSTEM, P.attribute_prompt(pub, agent, T0, mem["text"]),
        seed=seed, cache_scope=f"{condition}-{ablation}-{ep.episode_id}",
        meta={"prior": T0, "skill": skill, "condition": condition, "true_cause": ep.cause,
              "orig_claim": "c1"}))
    T1 = {k: float(att.get("trust", {}).get(k, T0[k])) for k in SKILLS}
    dT = {k: T1[k] - T0[k] for k in SKILLS}
    dT_star = oracle_delta(R, agent, skill, ep.cause, cfg)

    # ---- Phase E: future collaboration on matched probes ----
    post = post_failure_reliability(R, agent, skill, ep.cause, cfg)
    probe_rows = []
    for i, pr in enumerate(ep.probes):
        pt: Task = pr["task"]
        d = parse_json(await be.complete(
            "delegate", P.SUPERVISOR_SYSTEM, P.delegate_prompt(pub, agent, T1, mem["text"], pt),
            seed=seed, cache_scope=f"{condition}-{ablation}-{ep.episode_id}-{i}",
            meta={"trust_subject": T1[pt.skill], "subject": agent, "specialist": SPECIALIST[pt.skill],
                  "probe_skill": pt.skill, "subject_skill_specialty": AGENTS[agent],
                  "fallback": SPECIALIST[pt.skill]}))
        chosen = d.get("agent") if d.get("agent") in AGENTS else SPECIALIST[pt.skill]
        verify = bool(d.get("verify", False))
        ids, tool = full_view(pt)
        # subagent output is shared across conditions (same probe, agent, seed)
        ans, _ = await run_subagent(be, chosen, pt, ids, tool, seed,
                                    f"probe-{t.task_id}-{i}", mock_p(chosen, pt.skill))
        correct = is_correct(ans, pt)
        # controlled persistent weakness for the failing agent
        if chosen == agent and correct and _u(ep.episode_id, i, seed, "deg") < degrade_prob(
                R, agent, pt.skill, ep.cause, skill, cfg):
            ans, correct = pt.wrong_answer, False
        final_correct, caught = correct, False
        if verify:
            v = parse_json(await be.complete(
                "probe_verify", P.SUPERVISOR_SYSTEM, P.verify_prompt(pt, ans), seed=seed,
                cache_scope=f"verify-{ep.episode_id}-{i}",
                meta={"is_correct": correct, "gold": pt.gold_answer, "catch_rate": cfg.verify_catch_rate}))
            if v.get("verdict") == "reject":
                corr = v.get("corrected_answer")
                final_correct = is_correct(corr, pt) if corr else False
                caught = not correct
        true_rel = {a: (post[pt.skill] if a == agent else R[a][pt.skill]) for a in AGENTS}
        a_star, v_star, u_star = oracle_action(true_rel, cfg)
        u_exp = expected_utility(true_rel[chosen], verify, cfg)
        probe_rows.append({
            "slot": pr["slot"], "skill": pt.skill, "chosen": chosen, "verify": verify,
            "delegated_to_subject": chosen == agent, "raw_correct": correct,
            "final_correct": final_correct, "false_claim_survived": (not correct) and (not caught) and (not final_correct),
            "utility": float(final_correct) - (cfg.verify_cost if verify else 0.0),
            "regret": round(u_star - u_exp, 4), "oracle_agent": a_star, "oracle_verify": v_star,
        })

    return {
        "episode_id": ep.episode_id, "task_id": t.task_id, "source": t.source, "split": ep.split,
        "cause": ep.cause, "agent": agent, "affected_skill": skill, "condition": condition,
        "ablation": ablation, "seed": seed, "T0": T0, "T1": T1, "dT": dT, "dT_star": dT_star,
        "attribution_rule": attribution,
        "sup_cause": att.get("cause"), "sup_orig_claim": att.get("originating_claim"),
        "sup_orig_agent": att.get("originating_agent"), "sup_orig_source": att.get("originating_source"),
        "claim_trust": att.get("claim_trust"), "probes": probe_rows,
    }


async def run_all(be: Backend, bench: dict, cfg: ExperimentConfig, R: dict, *, split: str | None,
                  conditions=None, ablations=(None,), out_path: str | None = None,
                  limit: int | None = None) -> list[dict]:
    eps = [e for e in bench["episodes"] if split is None or e.split == split]
    if limit:
        eps = eps[:limit]
    jobs = [run_episode(be, ep, R, c, s, cfg, ab)
            for ep in eps for c in (conditions or cfg.conditions) for s in cfg.seeds for ab in ablations
            if ab is None or c == "claimgraph_update"]
    rows = [r for r in await asyncio.gather(*jobs) if r]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows

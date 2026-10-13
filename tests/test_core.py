import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctb.config import AGENTS, CAUSES, SKILLS, ExperimentConfig  # noqa: E402
from ctb.data.episodes import build_benchmark  # noqa: E402
from ctb.llm.backend import make_backend  # noqa: E402
from ctb.memory import build_failure_graph, render_memory, run_attribution, visible_recommendation  # noqa: E402
from ctb.metrics import table  # noqa: E402
from ctb.runner import phase_a, run_all  # noqa: E402
from ctb.trust import explicit_update, oracle_delta  # noqa: E402

CFG = ExperimentConfig(cache_path=None, seeds=[0])
BENCH = build_benchmark(CFG, offline=True)
R = {a: {k: (0.85 if AGENTS[a] == k else 0.5) for k in SKILLS} for a in AGENTS}


def test_size_and_matching():
    eps = BENCH["episodes"]
    assert len(eps) == 800
    by_task = {}
    for e in eps:
        by_task.setdefault(e.base_task.task_id, []).append(e)
    assert len(by_task) == 200
    for ts in by_task.values():
        assert sorted(e.cause for e in ts) == sorted(CAUSES)
        assert len({e.split for e in ts}) == 1                       # no split leakage
        assert len({visible_recommendation(e) for e in ts}) == 1     # same visible failure
        assert len({tuple(p["task"].task_id for p in e.probes) for e in ts}) == 1  # matched probes


def test_split_ratio():
    n = {}
    for e in BENCH["episodes"]:
        n[e.split] = n.get(e.split, 0) + 1
    assert n == {"train": 480, "dev": 160, "test": 160}


def test_attributor_recovers_cause_from_provenance():
    for e in BENCH["episodes"]:
        g = build_failure_graph(e)
        assert run_attribution(e, g)["cause"] == e.cause, e.episode_id


def test_oracle_targets():
    for cause in CAUSES:
        d = oracle_delta(R, "verifier", "fact_verification", cause, CFG)
        if cause in ("missing_context", "tool_failure"):
            assert all(v == 0 for v in d.values())
        if cause == "persistent_reasoning":
            assert d["fact_verification"] < 0 and d["quantitative"] == 0
    d = oracle_delta(R, "tool_specialist", "multihop", "out_of_specialty", CFG)
    assert d["multihop"] < 0 and d["tool_use"] == 0


def test_update_rule_is_scoped():
    prior = {k: 0.8 for k in SKILLS}
    t = explicit_update(prior, {"cause": "persistent_reasoning", "scope": ["multihop", "fact_verification"],
                                "resolved": False})
    assert t["multihop"] < t["fact_verification"] < 0.8 and t["quantitative"] == 0.8
    t = explicit_update(prior, {"cause": "tool_failure", "scope": ["tool_use"], "resolved": True})
    assert t == prior


def test_ablation_hides_fields():
    e = next(e for e in BENCH["episodes"] if e.cause == "tool_failure")
    g = build_failure_graph(e)
    att = run_attribution(e, g)
    full = render_memory("claimgraph", e, g, att, {k: .8 for k in SKILLS})["text"]
    abl = render_memory("claimgraph", e, g, att, {k: .8 for k in SKILLS}, "no_cause_label")["text"]
    assert '"failure_cause"' in full and '"failure_cause"' not in abl


def test_end_to_end_mock():
    be = make_backend(CFG)

    async def go():
        Rm = await phase_a(be, BENCH["calibration"], CFG)
        return await run_all(be, BENCH, CFG, Rm, split="test", limit=8)
    rows = asyncio.run(go())
    assert len(rows) == 8 * 6
    df = table(rows)
    assert {"TCE_affected", "TUSE", "regret"} <= set(df.columns)
    assert df["regret"].min() >= -1e-9

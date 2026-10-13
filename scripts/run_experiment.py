"""End-to-end runner.

Examples
--------
# dry run, no GPU, synthetic data (pipeline + cost measurement)
python scripts/run_experiment.py --backend mock --offline --out runs/dry

# real run against a local vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192
python scripts/run_experiment.py --backend openai --base-url http://localhost:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct --split test --out runs/llama8b

# ablations (on condition 6)
python scripts/run_experiment.py ... --ablations
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctb.config import ABLATIONS, ExperimentConfig  # noqa: E402
from ctb.data.episodes import build_benchmark, save_benchmark  # noqa: E402
from ctb.llm.backend import make_backend  # noqa: E402
from ctb.metrics import summary, table  # noqa: E402
from ctb.runner import phase_a, run_all  # noqa: E402
from ctb.stats import paired_bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--offline", action="store_true", help="synthetic tasks instead of HF downloads")
    ap.add_argument("--split", default="test", help="train|dev|test|all")
    ap.add_argument("--limit", type=int, default=None, help="max episodes (for smoke tests)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--out", default="runs/default")
    a = ap.parse_args()

    cfg = ExperimentConfig.from_json(a.config) if a.config else ExperimentConfig()
    for k, v in {"backend": a.backend, "model": a.model, "base_url": a.base_url}.items():
        if v is not None:
            setattr(cfg, k, v)
    if a.seeds is not None:
        cfg.seeds = a.seeds
    cfg.out_dir = a.out
    cfg.cache_path = str(Path(a.out) / "llm_cache.sqlite")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.to_json())

    t0 = time.perf_counter()
    bench = build_benchmark(cfg, offline=a.offline)
    save_benchmark(bench, a.out)
    be = make_backend(cfg)

    async def go():
        R = await phase_a(be, bench["calibration"], cfg)
        (out / "reliability_profile.json").write_text(json.dumps(R, indent=1))
        t_a = time.perf_counter()
        split = None if a.split == "all" else a.split
        rows = await run_all(be, bench, cfg, R, split=split, limit=a.limit,
                             out_path=str(out / "episodes_results.jsonl"))
        if a.ablations:
            rows_ab = await run_all(be, bench, cfg, R, split=split, limit=a.limit,
                                    conditions=["claimgraph_update"], ablations=ABLATIONS,
                                    out_path=str(out / "ablation_results.jsonl"))
        else:
            rows_ab = []
        return R, rows, rows_ab, t_a

    R, rows, rows_ab, t_a = asyncio.run(go())
    wall = time.perf_counter() - t0
    df = table(rows)
    df.to_csv(out / "metrics_per_episode.csv", index=False)
    summ = summary(df)
    summ.to_csv(out / "summary_by_condition.csv")
    by_cause = summary(df, ("cause", "condition"))
    by_cause.to_csv(out / "summary_by_cause_condition.csv")
    tests = [paired_bootstrap(df, m, base, "claimgraph_update")
             for m in ["TCE_affected", "regret", "TUSE"] for base in ["outcome_only", "skill_conditioned"]]
    (out / "paired_tests.json").write_text(json.dumps(tests, indent=1))
    if rows_ab:
        table(rows_ab).groupby("ablation")[["TCE_affected", "TUSE", "regret"]].mean().round(3) \
            .to_csv(out / "ablation_summary.csv")
    usage = be.usage.summary()
    usage["wall_clock_s"] = round(wall, 2)
    usage["n_episode_runs"] = len(rows) + len(rows_ab)
    (out / "usage.json").write_text(json.dumps(usage, indent=1))
    print(summ.to_string())
    print(json.dumps(usage["total"]), f"wall={wall:.1f}s", f"runs={usage['n_episode_runs']}")


if __name__ == "__main__":
    main()

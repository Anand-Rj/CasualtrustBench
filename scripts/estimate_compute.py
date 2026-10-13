"""Compute / cost / wall-clock estimator for CausalTrustBench.

Call counts per episode-run and average prompt sizes come from a measured dry run
(`runs/dry_all/usage.json`, produced by run_experiment.py with --backend mock --offline).
Completion lengths, GPU throughput and prices are explicit ASSUMPTIONS below; replace the
throughput numbers with what a 1-hour pilot on your own GPU measures.

python scripts/estimate_compute.py --usage runs/dry_all/usage.json --out runs/compute_estimate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------- study design (proposal)
DESIGN = {
    "episodes": 800,              # 200 base tasks x 4 causes
    "base_tasks": 200,
    "conditions": 6,
    "seeds": 3,
    "probes_per_episode": 4,
    "ablations": 5,               # run on condition 6, all episodes, 1 seed
    "ablation_seeds": 1,
    "robustness_configs": 3,      # 2 extra model families + 1 heterogeneous team
    "robustness_conditions": 4,   # outcome_only, skill_conditioned, claimgraph, claimgraph_update
    "robustness_seeds": 1,
    "calibration_calls_per_model": 4 * 4 * 25,   # agents x skills x tasks
    "pilot_and_rerun_overhead": 0.35,            # dev-split prompt tuning, failed runs, reruns
}

# ---------------------------------------------------------------- assumptions
COMPLETION = {             # output tokens per call for an instruction-tuned CoT model (assumed)
    "attribute_trust": 350, "delegate": 100, "probe_verify": 200, "subagent": 300, "elicit_prior": 120,
}
VERIFY_RATE = 0.5          # fraction of probes the supervisor verifies (assumed; mock measured ~0.75)
AGENTS_PER_PROBE = 1.5     # distinct agents a probe is routed to across conditions (sharing)

# Aggregate vLLM throughput with ~64 concurrent requests, bf16 (assumed ballparks;
# replace with pilot measurements). prefill and decode tokens/s for the whole server.
THROUGHPUT = {
    "Llama-3.1-8B  | 1x A100-80GB": {"gpus": 1, "prefill": 15000, "decode": 2000, "gpu": "A100"},
    "Llama-3.1-8B  | 1x H100-80GB": {"gpus": 1, "prefill": 30000, "decode": 3500, "gpu": "H100"},
    "Qwen2.5-32B   | 2x A100-80GB": {"gpus": 2, "prefill": 6000, "decode": 900, "gpu": "A100"},
    "Llama-3.1-70B | 4x A100-80GB": {"gpus": 4, "prefill": 4000, "decode": 600, "gpu": "A100"},
    "Llama-3.1-70B | 2x H100 (FP8)": {"gpus": 2, "prefill": 8000, "decode": 1100, "gpu": "H100"},
}
# On-demand $/GPU-hour (Oct 2026 web snapshot, low-cost neoclouds .. hyperscalers)
PRICE = {"A100": (1.49, 3.43), "H100": (2.49, 6.88)}
API_PRICES = [(0.15, 0.60), (1.00, 4.00), (3.00, 15.00)]   # illustrative ($/M input, $/M output)


def per_run_calls(prompt_avg: dict) -> dict:
    """Calls (non-shared) per episode-run, with prompt/completion tokens."""
    p = DESIGN["probes_per_episode"]
    rows = {
        "attribute_trust": 1,
        "delegate": p,
        "probe_verify": p * VERIFY_RATE,
    }
    return {k: {"calls": n, "prompt": n * prompt_avg[k], "completion": n * COMPLETION[k]}
            for k, n in rows.items()}


def shared_subagent_calls(seeds: int) -> float:
    # probe executions are shared across conditions and the 4 causal variants
    return DESIGN["base_tasks"] * DESIGN["probes_per_episode"] * AGENTS_PER_PROBE * seeds


def block(name, runs, seeds, prompt_avg, models=1):
    pr = per_run_calls(prompt_avg)
    calls = sum(v["calls"] for v in pr.values()) * runs
    ptok = sum(v["prompt"] for v in pr.values()) * runs
    ctok = sum(v["completion"] for v in pr.values()) * runs
    sub = shared_subagent_calls(seeds) * models + DESIGN["calibration_calls_per_model"] * models
    calls += sub
    ptok += sub * prompt_avg["subagent"]
    ctok += sub * COMPLETION["subagent"]
    return {"block": name, "episode_runs": runs, "calls": round(calls), "prompt_tokens": round(ptok),
            "completion_tokens": round(ctok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="runs/dry_all/usage.json")
    ap.add_argument("--out", default="runs/compute_estimate.json")
    a = ap.parse_args()
    u = json.loads(Path(a.usage).read_text())
    prompt_avg = {k: v["avg_prompt_tokens"] for k, v in u["by_type"].items()}

    D = DESIGN
    blocks = [
        block("Main study (6 conditions x 3 seeds x 800 episodes)",
              D["episodes"] * D["conditions"] * D["seeds"], D["seeds"], prompt_avg),
        block("Ablations (5 x condition 6 x 800 episodes x 1 seed)",
              D["episodes"] * D["ablations"] * D["ablation_seeds"], 0, prompt_avg),
        block("Robustness (3 model configs x 4 conditions x 800 x 1 seed)",
              D["episodes"] * D["robustness_conditions"] * D["robustness_seeds"] * D["robustness_configs"],
              D["robustness_seeds"], prompt_avg, models=D["robustness_configs"]),
    ]
    tot = {k: sum(b[k] for b in blocks) for k in ["episode_runs", "calls", "prompt_tokens", "completion_tokens"]}
    tot_buf = {k: round(v * (1 + D["pilot_and_rerun_overhead"])) for k, v in tot.items()}
    # primary-model share (main + ablations) vs robustness share
    prim = {k: blocks[0][k] + blocks[1][k] for k in ["prompt_tokens", "completion_tokens"]}
    rob = {k: blocks[2][k] for k in ["prompt_tokens", "completion_tokens"]}

    def hours(tokens, tp):
        return (tokens["prompt_tokens"] / tp["prefill"] + tokens["completion_tokens"] / tp["decode"]) / 3600

    ov = 1 + D["pilot_and_rerun_overhead"]
    gpu_rows = []
    for name, tp in THROUGHPUT.items():
        wall_primary = hours(prim, tp) * ov
        wall_rob_one = hours({k: v / D["robustness_configs"] for k, v in rob.items()}, tp) * ov
        gh = (wall_primary + wall_rob_one * D["robustness_configs"]) * tp["gpus"]
        lo, hi = PRICE[tp["gpu"]]
        gpu_rows.append({
            "setup": name,
            "wall_hours_primary_model": round(wall_primary, 1),
            "wall_hours_per_robustness_config": round(wall_rob_one, 1),
            "wall_hours_everything_on_this_setup": round(wall_primary + wall_rob_one * D["robustness_configs"], 1),
            "gpu_hours_everything": round(gh, 1),
            "cloud_cost_usd": [round(gh * lo), round(gh * hi)],
        })
    api_rows = [{"price_in_out_per_M": [pi, po],
                 "cost_usd_everything": round(tot_buf["prompt_tokens"] / 1e6 * pi + tot_buf["completion_tokens"] / 1e6 * po)}
                for pi, po in API_PRICES]
    res = {"measured_prompt_tokens_avg": prompt_avg, "assumed_completion_tokens": COMPLETION,
           "verify_rate_assumed": VERIFY_RATE, "blocks": blocks, "total": tot,
           "total_with_overhead": tot_buf, "gpu_estimates": gpu_rows, "api_estimates": api_rows}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

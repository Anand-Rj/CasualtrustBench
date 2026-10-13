# CausalTrustBench — Causal Claim-to-Trust Graph

Implementation of the proposal *Causal Claim-to-Trust Graph: Calibrating Long-Horizon Trust
from Claim-Level Failure Provenance*.

## What is implemented

| Proposal section | Code |
|---|---|
| Datasets: FEVER, HotpotQA, GSM8K, AgentBench-style tool tasks (50 each) | `ctb/data/loaders.py`, `ctb/data/dbtool.py` |
| 4 matched causal variants → 800 episodes, 60/20/20 split at base-task level, 4 future probes (2 affected / 1 nearby / 1 unrelated) | `ctb/data/episodes.py` |
| Persistent team: supervisor + 4 specialists, same base LLM | `ctb/prompts.py`, `ctb/runner.py` |
| Causal ClaimGraph (claim_id, parent, agent, skill, text, evidence, tool_state, confidence, verification, failure_cause, persistence, scope, resolved) + rule-based failure-cause attributor | `ctb/claimgraph.py` |
| 6 memory conditions + 5 ablations | `ctb/memory.py` |
| Oracle ΔT*, controlled persistent-weakness profile, explicit scoped update rule, utility model | `ctb/trust.py` |
| Phases A–E | `ctb/runner.py` |
| TCE, delegation regret, TUSE, over/under-distrust, cause/provenance/lineage accuracy, FCSR, success, verification rate | `ctb/metrics.py` |
| Paired bootstrap, mixed-effects, McNemar | `ctb/stats.py` |
| Compute / cost / wall-clock estimator | `scripts/estimate_compute.py` |

## Quick start

```bash
pip install datasets openai pandas numpy scipy statsmodels pytest

# 1. Dry run: synthetic tasks + simulated agents, no GPU (≈5 min for all 800 episodes)
python scripts/run_experiment.py --backend mock --offline --split all --ablations --out runs/dry_all
python scripts/estimate_compute.py --usage runs/dry_all/usage.json

# 2. Real run on a local vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192 --enable-prefix-caching
python scripts/run_experiment.py --backend openai --base-url http://localhost:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct --split all --out runs/llama8b

# 3. Pilot first (measures real throughput for the estimate)
python scripts/run_experiment.py --backend openai ... --split dev --limit 40 --seeds 0 --out runs/pilot

pytest -q tests
```

Mock-backend numbers are **plumbing checks, not results** — the simulated supervisor is scripted.

## Design decisions to review

1. **Forced vs induced failure.** Default `failure_mode="force"`: the visible recommendation
   (r⁻) and rejection are fixed and identical across the 4 variants; only provenance differs.
   `"induce"` runs the subagent under each condition and drops episodes where it does not fail.
2. **Subagent identity in the out-of-specialty variant** necessarily differs (the task is
   routed to a non-specialist); in the other 3 variants it is the specialist.
3. **Oracle ΔT\***: 0 for resolved transient causes; for out-of-specialty, the routed skill moves
   from the team-average prior to the agent's true reliability; for persistent reasoning,
   the controlled profile lowers same-skill reliability by 0.30 and nearby by 0.10
   (`persistent_drop_*` in `ctb/config.py`) and probe answers are degraded to match.
4. **Sharing for matched noise and cost.** Probe executions are cached across conditions and
   across the 4 causal variants of a base task (same probe, agent, seed ⇒ same answer).
5. **FEVER** loads `copenlu/fever_gold_evidence` (claims with gold evidence sentences);
   NEI claims are dropped. **Tool tasks** are AgentBench DB-Bench–style SQLite tasks generated
   locally so the tool can be failed/restored; swap in real AgentBench data via `LOADERS`.

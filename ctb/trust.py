"""Reliability profiles, the oracle trust target, the explicit scoped trust-update rule,
and the delegation utility model."""
from __future__ import annotations

from .config import AGENTS, NEARBY, SKILLS, TRANSIENT_CAUSES, ExperimentConfig


# --------------------------------------------------------------------------- profiles
def public_profile(R: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """What the supervisor is told before the failure: calibrated reliability on each
    agent's declared specialty, team-average reliability elsewhere (unknown skill)."""
    team_avg = {k: sum(R[a][k] for a in R) / len(R) for k in SKILLS}
    return {a: {k: (R[a][k] if AGENTS[a] == k else team_avg[k]) for k in SKILLS} for a in R}


def post_failure_reliability(R, agent, skill, cause, cfg: ExperimentConfig):
    """True reliability of `agent` after the episode, per skill (benchmark ground truth)."""
    post = dict(R[agent])
    if cause == "persistent_reasoning":
        post[skill] = max(0.0, post[skill] - cfg.persistent_drop_affected)
        nb = NEARBY[skill]
        post[nb] = max(0.0, post[nb] - cfg.persistent_drop_nearby)
    return post


def oracle_delta(R, agent, skill, cause, cfg) -> dict[str, float]:
    """Delta T* = true post-failure reliability - reliability the supervisor was told.

    transient + resolved -> 0 everywhere
    out-of-specialty     -> moves only the routed skill to the agent's true (lower) value
    persistent reasoning -> negative on the affected (and nearby) skill
    """
    pre = R[agent]
    post = post_failure_reliability(R, agent, skill, cause, cfg)
    delta = {k: post[k] - pre[k] for k in SKILLS}       # true change in competence
    if cause == "out_of_specialty":
        # the failure reveals the agent's (previously unobserved) reliability on the routed
        # skill: move from the team-average prior the supervisor was given to the true value
        delta[skill] = post[skill] - public_profile(R)[agent][skill]
    return {k: round(v, 4) for k, v in delta.items()}


def degrade_prob(R, agent, skill_probe, cause, failed_skill, cfg) -> float:
    """Probability that a probe answer from `agent` is flipped to wrong, so that the
    agent's empirical reliability equals the post-failure profile."""
    if cause != "persistent_reasoning":
        return 0.0
    pre = R[agent][skill_probe]
    post = post_failure_reliability(R, agent, failed_skill, cause, cfg)[skill_probe]
    return 0.0 if pre <= 0 else max(0.0, (pre - post) / pre)


# --------------------------------------------------------------------------- update rule
def explicit_update(prior: dict[str, float], attribution: dict, observed_rel: float | None = None,
                    strength: float = 0.25) -> dict[str, float]:
    """Condition 6: convert causal ClaimGraph state into a scoped trust adjustment.

    * transient & resolved   -> no change
    * transient & unresolved -> small penalty on the affected skill only
    * out_of_specialty       -> move routed-skill trust toward the agent's observed
                                reliability on that skill (or halve it); other skills unchanged
    * persistent reasoning   -> penalty `strength` on affected skill, strength/3 on nearby
    """
    t = dict(prior)
    cause, scope = attribution["cause"], attribution["scope"]
    k = scope[0]
    if cause in TRANSIENT_CAUSES:
        if not attribution.get("resolved"):
            t[k] -= 0.05
    elif cause == "out_of_specialty":
        t[k] = observed_rel if observed_rel is not None else t[k] * 0.5
    else:
        t[k] -= strength
        for nb in scope[1:]:
            t[nb] -= strength / 3
    return {s: round(min(1.0, max(0.0, v)), 4) for s, v in t.items()}


# --------------------------------------------------------------------------- utility
def expected_utility(rel: float, verify: bool, cfg: ExperimentConfig) -> float:
    p = rel + (1 - rel) * cfg.verify_catch_rate if verify else rel
    return p - (cfg.verify_cost if verify else 0.0) - cfg.call_cost


def oracle_action(true_rel: dict[str, float], cfg) -> tuple[str, bool, float]:
    """Best (agent, verify) given the true reliability of each agent on the probe skill."""
    best = None
    for a, r in true_rel.items():
        for v in (False, True):
            u = expected_utility(r, v, cfg)
            if best is None or u > best[2]:
                best = (a, v, u)
    return best

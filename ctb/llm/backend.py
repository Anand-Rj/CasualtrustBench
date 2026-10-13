"""LLM backends with caching and token/latency accounting.

* OpenAIBackend - any OpenAI-compatible endpoint: a local vLLM / SGLang server
  (`vllm serve meta-llama/Llama-3.1-8B-Instruct`) or a hosted API.
* MockBackend   - deterministic simulator used for dry runs and tests. It returns
  well-formed JSON for every call type and fakes token usage from prompt length, so the
  pipeline and the call/token counts can be measured without a GPU. Mock outputs are
  NOT experimental results.

Every call is tagged with a `call_type` so usage can be broken down per phase.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def approx_tokens(text: str) -> int:
    # ~4 chars/token for English with Llama-3 / cl100k-style tokenizers
    return max(1, round(len(text) / 4))


@dataclass
class Usage:
    calls: dict = field(default_factory=lambda: defaultdict(int))
    prompt_tokens: dict = field(default_factory=lambda: defaultdict(int))
    completion_tokens: dict = field(default_factory=lambda: defaultdict(int))
    cache_hits: dict = field(default_factory=lambda: defaultdict(int))
    latency_s: dict = field(default_factory=lambda: defaultdict(float))

    def add(self, call_type, p, c, lat, hit=False):
        if hit:
            self.cache_hits[call_type] += 1
            return
        self.calls[call_type] += 1
        self.prompt_tokens[call_type] += p
        self.completion_tokens[call_type] += c
        self.latency_s[call_type] += lat

    def summary(self) -> dict:
        types = sorted(set(self.calls) | set(self.cache_hits))
        rows = {}
        for t in types:
            n = self.calls[t]
            rows[t] = {
                "calls": n,
                "cache_hits": self.cache_hits[t],
                "prompt_tokens": self.prompt_tokens[t],
                "completion_tokens": self.completion_tokens[t],
                "avg_prompt_tokens": round(self.prompt_tokens[t] / n, 1) if n else 0,
                "avg_completion_tokens": round(self.completion_tokens[t] / n, 1) if n else 0,
            }
        tot = {k: sum(v[k] for v in rows.values()) for k in
               ["calls", "cache_hits", "prompt_tokens", "completion_tokens"]}
        return {"by_type": rows, "total": tot}


class _Cache:
    def __init__(self, path: str | None):
        self.con = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(path, check_same_thread=False)
            self.con.execute("CREATE TABLE IF NOT EXISTS c(k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k):
        if not self.con:
            return None
        r = self.con.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def put(self, k, v):
        if self.con:
            self.con.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self.con.commit()


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.usage = Usage()
        self.cache = _Cache(cfg.cache_path)
        self.sem = asyncio.Semaphore(cfg.max_concurrency)
        self._inflight: dict[str, asyncio.Future] = {}

    def _key(self, call_type, system, prompt, seed, cache_scope):
        h = hashlib.sha256(json.dumps([self.cfg.model, call_type, system, prompt, seed,
                                       cache_scope, self.cfg.temperature]).encode())
        return h.hexdigest()

    async def complete(self, call_type: str, system: str, prompt: str, *, seed: int,
                       meta: dict | None = None, cache_scope: str = "") -> str:
        key = self._key(call_type, system, prompt, seed, cache_scope)
        hit = self.cache.get(key)
        if hit is None and key in self._inflight:          # identical request already running
            hit = await asyncio.shield(self._inflight[key])
        if hit is not None:
            self.usage.add(call_type, 0, 0, 0, hit=True)
            return hit
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            async with self.sem:
                t0 = time.perf_counter()
                text, p, c = await self._complete(call_type, system, prompt, seed, meta or {})
                self.usage.add(call_type, p, c, time.perf_counter() - t0)
            self.cache.put(key, text)
            fut.set_result(text)
            return text
        except BaseException as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved
            raise
        finally:
            self._inflight.pop(key, None)

    async def _complete(self, call_type, system, prompt, seed, meta):  # pragma: no cover
        raise NotImplementedError


class OpenAIBackend(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=cfg.base_url,
                                  api_key=os.environ.get(cfg.api_key_env, "EMPTY"))

    async def _complete(self, call_type, system, prompt, seed, meta):
        for attempt in range(6):
            try:
                r = await self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens.get(call_type, 400),
                    seed=seed,
                )
                u = r.usage
                return (r.choices[0].message.content or "",
                        u.prompt_tokens if u else approx_tokens(system + prompt),
                        u.completion_tokens if u else 0)
            except Exception:
                if attempt == 5:
                    raise
                await asyncio.sleep(2 ** attempt)


class MockBackend(Backend):
    """Simulated agents. Subagent correctness follows meta['p_correct'];
    the supervisor follows a crude, condition-sensitive heuristic so dry-run plots
    have the right shape. For plumbing and cost measurement only."""

    # rough completion lengths (tokens) a CoT model produces for each call type
    COMPLETION = {"subagent": 220, "attribute_trust": 260, "elicit_prior": 120,
                  "delegate": 70, "probe_verify": 140}

    async def _complete(self, call_type, system, prompt, seed, meta):
        rng = random.Random(hashlib.sha256(f"{seed}|{prompt}".encode()).hexdigest())
        if call_type == "subagent":
            ok = rng.random() < meta.get("p_correct", 0.7)
            ans = meta["gold"] if ok else meta["wrong"]
            out = {"reasoning": "Step-by-step reasoning over the evidence. " * 12,
                   "claims": [{"text": "Intermediate claim from evidence", "evidence": []}],
                   "answer": ans, "confidence": round(rng.uniform(0.55, 0.95), 2)}
        elif call_type in ("elicit_prior", "attribute_trust"):
            prior = meta.get("prior", {})
            trust = dict(prior)
            cause = "unknown"
            if call_type == "attribute_trust":
                trust, cause = self._mock_update(prompt, meta, rng)
            out = {"cause": cause, "originating_claim": meta.get("orig_claim", "c1"),
                   "persistence": "transient" if cause in ("missing_context", "tool_failure") else "persistent",
                   "affected_skills": [meta.get("skill", "")],
                   "trust": {k: round(min(1, max(0, v)), 3) for k, v in trust.items()},
                   "rationale": "Brief justification of the trust update. " * 5}
        elif call_type == "delegate":
            t = meta.get("trust_subject", 0.7)
            spec = meta.get("specialist")
            subj = meta.get("subject")
            agent = subj if (meta.get("probe_skill") == meta.get("subject_skill_specialty") or t > 0.6) else spec
            if t < 0.45:
                agent = spec if spec != subj else meta.get("fallback", spec)
            out = {"agent": agent, "verify": bool(t < 0.65), "reason": "Delegation rationale."}
        elif call_type == "probe_verify":
            correct = meta.get("is_correct", True)
            catch = rng.random() < meta.get("catch_rate", 0.85)
            verdict = "accept" if correct or not catch else "reject"
            out = {"verdict": verdict,
                   "corrected_answer": meta.get("gold") if verdict == "reject" else None,
                   "reason": "Checked the answer against the evidence."}
        else:
            out = {"text": "ok"}
        text = json.dumps(out)
        await asyncio.sleep(0)
        p = approx_tokens(system + prompt)
        c = self.COMPLETION.get(call_type, 100) + rng.randint(-20, 20)
        return text, p, c

    @staticmethod
    def _mock_update(prompt, meta, rng):
        prior = dict(meta.get("prior", {}))
        skill = meta.get("skill")
        cond = meta.get("condition")
        true_cause = meta.get("true_cause")
        trust = dict(prior)
        if cond in ("no_memory",):
            return trust, "unknown"
        if cond in ("raw_history", "outcome_only"):
            for k in trust:   # over-generalised penalty
                trust[k] -= 0.12 if cond == "outcome_only" else 0.08
            if skill in trust:
                trust[skill] -= 0.06
            return trust, true_cause if (cond == "raw_history" and rng.random() < 0.5) else "unknown"
        if cond == "skill_conditioned":
            if skill in trust:
                trust[skill] -= 0.18
            return trust, "unknown"
        # claimgraph conditions: read the cause from the structured record
        m = re.search(r'"failure_cause":\s*"(\w+)"', prompt)
        cause = m.group(1) if m else true_cause
        if cause in ("missing_context", "tool_failure"):
            trust[skill] -= 0.03 if cond == "claimgraph" else 0.0
        elif cause == "out_of_specialty":
            trust[skill] = min(trust[skill], 0.45)
        else:
            trust[skill] -= 0.25 if cond == "claimgraph" else 0.3
        return trust, cause


def make_backend(cfg) -> Backend:
    return MockBackend(cfg) if cfg.backend == "mock" else OpenAIBackend(cfg)


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}

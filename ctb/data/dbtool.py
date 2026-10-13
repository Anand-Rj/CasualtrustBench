"""AgentBench-style database tool tasks, generated locally.

AgentBench's DB-Bench asks an agent to answer questions by querying a database. We
generate the same kind of task on SQLite so that (i) answers are exact and (ii) the
tool can be made to fail (empty result) or return stale data, then be restored.

The tool is a retrieval tool: the harness executes the task's canonical SQL and hands
the agent the result. Under `tool_failure` the result is empty or comes from a stale
snapshot, which yields the fixed wrong answer r-.
"""
from __future__ import annotations

import random
import sqlite3

from .schema import Evidence, Task

REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = ["widget", "gadget", "sprocket", "gizmo", "doohickey", "valve", "bracket"]
YEARS = [2021, 2022, 2023, 2024]


def _build_rows(rng: random.Random, n_rows: int = 60):
    rows = []
    for i in range(n_rows):
        rows.append((i, rng.choice(REGIONS), rng.choice(PRODUCTS), rng.choice(YEARS),
                     rng.randint(5, 400), round(rng.uniform(2, 90), 2)))
    return rows


SCHEMA = ("Table sales(id INTEGER, region TEXT, product TEXT, year INTEGER, "
          "units INTEGER, unit_price REAL)")

TEMPLATES = [
    ("How many units of {p} were sold in the {r} region in {y}?",
     "SELECT COALESCE(SUM(units),0) FROM sales WHERE product='{p}' AND region='{r}' AND year={y}"),
    ("What was the total number of units sold in the {r} region across all years?",
     "SELECT COALESCE(SUM(units),0) FROM sales WHERE region='{r}'"),
    ("How many distinct products were sold in the {r} region in {y}?",
     "SELECT COUNT(DISTINCT product) FROM sales WHERE region='{r}' AND year={y}"),
    ("What is the largest single-order unit count for {p}?",
     "SELECT COALESCE(MAX(units),0) FROM sales WHERE product='{p}'"),
]


def run_sql(rows, sql: str):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE sales(id INTEGER, region TEXT, product TEXT, year INTEGER, "
                "units INTEGER, unit_price REAL)")
    con.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", rows)
    val = con.execute(sql).fetchone()[0]
    con.close()
    return val


def generate_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    i = 0
    while len(tasks) < n:
        rows = _build_rows(rng)
        # stale snapshot: drop ~30% of rows (an older copy of the table)
        stale = [r for r in rows if rng.random() > 0.3]
        q_tpl, sql_tpl = rng.choice(TEMPLATES)
        slots = dict(p=rng.choice(PRODUCTS), r=rng.choice(REGIONS), y=rng.choice(YEARS))
        q, sql = q_tpl.format(**slots), sql_tpl.format(**slots)
        gold = run_sql(rows, sql)
        wrong = run_sql(stale, sql)
        i += 1
        if gold == wrong or gold == 0:
            continue
        ev = [
            Evidence("schema", "Database schema", SCHEMA),
            Evidence("tool_result", "sql_query tool output",
                     f"Query: {sql}\nResult: {gold}"),
        ]
        tasks.append(Task(
            f"dbtool-{seed}-{i}", "dbtool", "tool_use", q, ev, str(gold), ["tool_result"],
            str(wrong), answer_format="number",
            tool={"sql": sql, "stale_result": str(wrong), "fresh_result": str(gold)}))
    return tasks

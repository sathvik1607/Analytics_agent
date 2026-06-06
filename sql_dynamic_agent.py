#!/usr/bin/env python3
"""
sql_dynamic_agent.py — SQL Generation + Execution Sub-Agent
============================================================
A self-contained LangGraph agent responsible for:
  1. SQL Generator  — writes one SQL query per plan step (with retry context)
  2. SQL Executor   — runs each query, captures results as DataFrames
  3. Retry Loop     — up to MAX_RETRIES self-corrections on failed queries

Designed to be imported by nl2sql_agent.py (or any orchestrator) via:
    from sql_dynamic_agent import SQLDynamicAgent
    results = SQLDynamicAgent(llm, engine, debug=True).run(schema, plan, query)

Stand-alone test:
    python sql_dynamic_agent.py
"""

import json
import time
import traceback
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  SUB-AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

class SQLAgentState(TypedDict):
    user_query:        str
    schema:            str
    query_plan:        Dict[str, Any]
    generated_queries: Optional[List[Dict[str, Any]]]
    query_results:     Optional[List[Dict[str, Any]]]
    failed_queries:    Optional[List[Dict[str, Any]]]
    retry_count:       int
    max_retries:       int
    error:             Optional[str]
    debug:             bool


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str, is_array: bool = False, debug: bool = False) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if debug:
            print(f"   [DEBUG] JSON direct parse failed — attempting boundary extraction…")
        start, end = (raw.find("["), raw.rfind("]") + 1) if is_array else (raw.find("{"), raw.rfind("}") + 1)
        if start != -1 and end > 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError as exc:
                if debug:
                    print(f"   [DEBUG] Boundary extraction failed: {exc}\n   Raw:\n{raw[:600]}")
                raise ValueError(f"Could not parse JSON:\n{raw[:400]}")
        raise ValueError(f"Could not parse JSON:\n{raw[:400]}")


def _execute_sql(sql: str, engine: Engine, debug: bool = False) -> Tuple[pd.DataFrame, Optional[str]]:
    if debug:
        print(f"\n   [DEBUG] Executing SQL:\n   {sql}")
    t0 = time.perf_counter()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            elapsed = time.perf_counter() - t0
            if debug:
                print(f"   [DEBUG] {elapsed:.2f}s | {len(df)} row(s) | cols: {list(df.columns)}")
                if not df.empty:
                    null_cols = [c for c in df.columns if df[c].isnull().any()]
                    if null_cols:
                        print(f"   ⚠️  NULL values in: {null_cols} — aggregate over empty set")
            return df, None
    except SQLAlchemyError as exc:
        if debug:
            traceback.print_exc()
        return pd.DataFrame(), str(exc)
    except Exception as exc:
        if debug:
            traceback.print_exc()
        return pd.DataFrame(), str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE FACTORIES  (closures capture llm / engine / debug from SQLDynamicAgent)
# ═══════════════════════════════════════════════════════════════════════════════

_SQL_GENERATOR_SYSTEM = """You are an expert SQL developer.
Generate one executable SQL query per step in the execution plan.

Return ONLY a valid JSON array — no markdown, no explanation:
[
  {
    "step_id": 1,
    "description": "what this query fetches",
    "sql": "SELECT …",
    "result_alias": "short_snake_case_name"
  }
]

Strict rules:
• Use only table and column names that exist in the schema.
• Use clear column aliases (e.g. AS total_sales, AS month_label).
• Never use placeholder tokens like <table> or {{column}} — use real names.
• Each query must run standalone; do not reference results from other steps inside SQL.
• One array entry per plan step, same step_id values.
• Always wrap aggregate functions in COALESCE:  COALESCE(SUM(col), 0) AS total_sales.
• The `cancelled` column is VARCHAR. Do NOT filter on it unless the user explicitly mentions
  cancelled, active, or valid invoices. Never add cancelled filters on your own initiative.

════════════════════════════════════════════════════════════
CRITICAL — invoice_date STORAGE FORMAT
════════════════════════════════════════════════════════════
invoice_date is a VARCHAR(50) column. Values are stored as:
    MM/DD/YYYY HH:MM:SS
    e.g.  '06/06/2026 19:50:45'
    e.g.  '06/05/2026 08:22:10'

It is NOT stored as YYYY-MM-DD. DATE() and YEAR()/MONTH()/DAY()
functions DO NOT work on this column — they return NULL.

SINGLE-DAY FILTER — use LIKE against the date prefix:
    WHERE invoice_date LIKE '06/06/2026%'   ← June 6 2026
    WHERE invoice_date LIKE '06/05/2026%'   ← June 5 2026
    WHERE invoice_date LIKE '12/31/2025%'   ← Dec 31 2025
  Rules:
    - Always zero-pad month and day to 2 digits
    - Format is always MM/DD/YYYY%, never YYYY-MM-DD%

MONTH FILTER — use SUBSTRING positional extraction:
    WHERE SUBSTRING(invoice_date, 1, 2) = '06'   ← month = June
      AND SUBSTRING(invoice_date, 7, 4) = '2026'  ← year = 2026

DATE RANGE FILTER:
    WHERE STR_TO_DATE(invoice_date, '%m/%d/%Y %H:%i:%s')
          BETWEEN '2026-06-01' AND '2026-06-30'

BANNED PATTERNS — these always return NULL or wrong rows:
    ✗  DATE(invoice_date)              — VARCHAR cannot be cast this way
    ✗  YEAR(invoice_date)              — fails on MM/DD/YYYY strings
    ✗  MONTH(invoice_date)             — fails on MM/DD/YYYY strings
    ✗  invoice_date LIKE '2026-06%'   — wrong order, never matches
    ✗  invoice_date = '2026-06-06'    — wrong format
    ✗  LEFT(invoice_date,7)='2026-06' — wrong format
    ✗  STR_TO_DATE(col,'%Y-%m-%d')    — wrong format specifier
════════════════════════════════════════════════════════════

WORKED EXAMPLE — "total sales on June 6th 2026":
  CORRECT SQL:
    SELECT COALESCE(SUM(invoice_amount), 0) AS total_sales
    FROM allpets_invoices
    WHERE invoice_date LIKE '06/06/2026%'

  WRONG SQL (do NOT generate):
    WHERE DATE(invoice_date) = '2026-06-06'           ← returns NULL
    WHERE invoice_date LIKE '2026-06-06%'             ← wrong format
    WHERE YEAR(invoice_date)=2026 AND ...             ← fails on VARCHAR
    WHERE invoice_date LIKE '06/06/2026%' AND cancelled = '0'
                                                      ← do NOT add cancelled filter unless asked

════════════════════════════════════════════════════════════
MONTHLY AGGREGATION — GROUP BY month
════════════════════════════════════════════════════════════
NEVER use YEAR(invoice_date) or MONTH(invoice_date) for grouping —
they return NULL on MM/DD/YYYY varchar strings.

CORRECT pattern:
    SELECT
        CONCAT(SUBSTRING(invoice_date, 7, 4), '-',
               SUBSTRING(invoice_date, 1, 2))                AS year_month,
        CAST(SUBSTRING(invoice_date, 7, 4) AS UNSIGNED) * 100
          + CAST(SUBSTRING(invoice_date, 1, 2) AS UNSIGNED)  AS sort_key,
        SUM(invoice_amount)                                  AS monthly_revenue
    FROM allpets_invoices
    GROUP BY year_month, sort_key
    ORDER BY sort_key

Always include sort_key so months order chronologically (Jan before Dec).
════════════════════════════════════════════════════════════

DATE RESOLUTION RULES:
• Today's date is in the user message — use it as anchor for all date logic.
• Named months refer to the current year unless they are in the future → use prior year.
• "Previous N months" = N calendar months immediately before the referenced month.
• Always derive exact year/month values from the anchor date; never use arbitrary INTERVALs.
"""


def _make_sql_generator(llm: ChatOpenAI, debug: bool):
    def sql_generator(state: SQLAgentState) -> SQLAgentState:
        retry = state.get("retry_count", 0)
        label = f" (retry #{retry})" if retry else ""
        print(f"\n🔧  SQL Generator{label}…")

        retry_block = ""
        if retry and state.get("failed_queries"):
            retry_block = (
                "\n\nPREVIOUS ATTEMPT FAILED — fix these errors:\n"
                + json.dumps(state["failed_queries"], indent=2)
            )

        user = (
            f"Today's date: {date.today().isoformat()} "
            f"(current month: {date.today().strftime('%B %Y')}, "
            f"current year: {date.today().year})\n\n"
            f"User question: {state['user_query']}\n\n"
            f"Database schema:\n{state['schema']}\n\n"
            f"Execution plan:\n{json.dumps(state['query_plan'], indent=2)}"
            f"{retry_block}"
        )

        try:
            resp = llm.invoke([SystemMessage(content=_SQL_GENERATOR_SYSTEM), HumanMessage(content=user)])
            if debug:
                print(f"   [DEBUG] LLM response (600 chars): {resp.content[:600].replace(chr(10),' ')}")
            queries = _parse_json(resp.content, is_array=True, debug=debug)
            print(f"   ✅  {len(queries)} quer{'y' if len(queries)==1 else 'ies'} generated")
            for q in queries:
                print(f"   📝  Step {q['step_id']}: {q['description']}")
                if debug:
                    print(f"   [DEBUG] SQL: {q['sql']}")
            return {**state, "generated_queries": queries, "error": None}
        except Exception as exc:
            return {**state, "error": f"SQL generator failed: {exc}"}
    return sql_generator


def _make_sql_executor(engine: Engine, debug: bool):
    def sql_executor(state: SQLAgentState) -> SQLAgentState:
        print("\n⚡  SQL Executor…")
        results: List[Dict[str, Any]] = []
        failed:  List[Dict[str, Any]] = []

        for q in state["generated_queries"]:
            sid = q["step_id"]
            print(f"   🔄  Step {sid}: {q['description']}")
            df, err = _execute_sql(q["sql"], engine, debug=debug)

            if err:
                print(f"   ❌  Step {sid} failed: {err}")
                failed.append({"step_id": sid, "description": q["description"],
                               "sql": q["sql"], "error": err})
                results.append({**q, "success": False, "error": err,
                                "data": [], "columns": [], "row_count": 0})
            else:
                print(f"   ✅  Step {sid}: {len(df)} row(s)")
                raw_data = df.to_dict(orient="records")
                for row in raw_data:
                    null_keys = [k for k, v in row.items() if v is None]
                    if null_keys:
                        print(f"   ⚠️  Step {sid}: NULL in {null_keys} — coercing to 0")
                        for k in null_keys:
                            row[k] = 0
                results.append({**q, "success": True, "error": None,
                                "data": raw_data,
                                "columns": list(df.columns),
                                "row_count": len(df)})

        return {**state, "query_results": results,
                "failed_queries": failed or None, "error": None}
    return sql_executor


def _make_increment_retry():
    def increment_retry(state: SQLAgentState) -> SQLAgentState:
        new_count = state.get("retry_count", 0) + 1
        failed = state.get("failed_queries") or []
        print(f"\n   🔁  Retry #{new_count}/{state['max_retries']} for step(s): {[f['step_id'] for f in failed]}")
        for f in failed:
            print(f"      Step {f['step_id']} error: {str(f['error'])[:120]}")
        return {**state, "retry_count": new_count, "generated_queries": None}
    return increment_retry


# ═══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ═══════════════════════════════════════════════════════════════════════════════

def _route_after_generator(state: SQLAgentState) -> str:
    return "error" if state.get("error") else "sql_executor"


def _route_after_executor(state: SQLAgentState) -> str:
    results     = state.get("query_results", [])
    failed      = state.get("failed_queries") or []
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    any_success = any(r["success"] for r in results)
    all_failed  = not any_success

    if failed and all_failed and retry_count < max_retries:
        decision = "retry"
    elif any_success:
        decision = "done"
    else:
        decision = "error"

    if state.get("debug"):
        print(f"   [DEBUG] route_after_executor → {decision} "
              f"(any_success={any_success}, retries={retry_count}/{max_retries})")
    return decision


def _route_after_generator_inner(state: SQLAgentState) -> str:
    return "error" if state.get("error") else "sql_executor"


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SQLDynamicAgent:
    """
    Reusable SQL generation + execution agent with automatic retry on failure.

    Usage:
        agent = SQLDynamicAgent(llm=my_llm, engine=my_engine, max_retries=3, debug=True)
        results = agent.run(schema=schema_str, plan=query_plan_dict, user_query="…")
        # results: List[Dict] — each dict has keys: step_id, sql, success, data, columns, row_count, error
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        engine: Engine,
        max_retries: int = 3,
        debug: bool = False,
    ):
        self.llm         = llm
        self.engine      = engine
        self.max_retries = max_retries
        self.debug       = debug
        self._graph      = self._build_graph()

    # ── Graph construction ───────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(SQLAgentState)

        g.add_node("sql_generator",   _make_sql_generator(self.llm, self.debug))
        g.add_node("sql_executor",    _make_sql_executor(self.engine, self.debug))
        g.add_node("increment_retry", _make_increment_retry())

        # terminal sinks — just pass state through
        g.add_node("done",  lambda s: s)
        g.add_node("error", lambda s: s)

        g.set_entry_point("sql_generator")

        g.add_conditional_edges("sql_generator", _route_after_generator_inner, {
            "sql_executor": "sql_executor",
            "error":        "error",
        })
        g.add_conditional_edges("sql_executor", _route_after_executor, {
            "done":  "done",
            "retry": "increment_retry",
            "error": "error",
        })
        g.add_edge("increment_retry", "sql_generator")
        g.add_edge("done",  END)
        g.add_edge("error", END)

        return g.compile()

    # ── Public interface ─────────────────────────────────────────────────────

    def run(
        self,
        schema:     str,
        plan:       Dict[str, Any],
        user_query: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate SQL for every step in `plan`, execute each query, and return
        the raw result list.  Retries up to max_retries on failure.

        Returns:
            List of result dicts.  Each dict contains:
                step_id, description, sql, result_alias,
                success (bool), data (list[dict]), columns (list[str]),
                row_count (int), error (str | None)
        """
        initial: SQLAgentState = {
            "user_query":        user_query,
            "schema":            schema,
            "query_plan":        plan,
            "generated_queries": None,
            "query_results":     None,
            "failed_queries":    None,
            "retry_count":       0,
            "max_retries":       self.max_retries,
            "error":             None,
            "debug":             self.debug,
        }
        final = self._graph.invoke(initial)

        if final.get("error") and not any(
            r.get("success") for r in (final.get("query_results") or [])
        ):
            raise RuntimeError(f"SQLDynamicAgent failed: {final['error']}")

        return final.get("query_results") or []


# ═══════════════════════════════════════════════════════════════════════════════
#  STAND-ALONE SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from urllib.parse import quote_plus
    from dotenv import load_dotenv
    load_dotenv()

    _openai_key = os.getenv("OPENAI_API_KEY")
    _db_host    = os.getenv("DB_HOST")
    _db_port    = os.getenv("DB_PORT", "3306")
    _db_user    = os.getenv("DB_USER")
    _db_pass    = os.getenv("DB_PASSWORD")
    _db_name    = os.getenv("DB_NAME")

    missing = [k for k, v in {
        "OPENAI_API_KEY": _openai_key,
        "DB_HOST": _db_host, "DB_USER": _db_user,
        "DB_PASSWORD": _db_pass, "DB_NAME": _db_name,
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing required .env variables: {missing}")

    DB_URL = (
        f"mysql+pymysql://{_db_user}:{quote_plus(_db_pass)}"
        f"@{_db_host}:{_db_port}/{_db_name}"
    )
    OPENAI_KEY  = _openai_key
    SCHEMA_FILE = os.getenv("SCHEMA_FILE", "cohort_main-schema_latest.sql")

    with open(SCHEMA_FILE) as f:
        schema = f.read()

    llm    = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY, temperature=0, max_tokens=4096)
    engine = create_engine(DB_URL)

    test_plan = {
        "complexity":  "simple",
        "reasoning":   "single aggregate query",
        "output_type": "aggregate",
        "steps": [
            {"step_id": 1, "description": "total invoices count", "depends_on": []}
        ],
    }

    agent = SQLDynamicAgent(llm=llm, engine=engine, max_retries=3, debug=True)
    results = agent.run(schema=schema, plan=test_plan, user_query="How many invoices are there?")
    print("\nResults:", json.dumps(results, indent=2, default=str))

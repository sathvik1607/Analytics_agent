#!/usr/bin/env python3
"""
nl2sql_agent.py — Natural Language → SQL Agent
================================================
Uses LangGraph to orchestrate a multi-step pipeline:
  1. Query Planner    — decides simple vs complex, creates step-by-step plan
  2. SQL Dynamic Agent— generates + executes SQL with automatic retry
                        (imported from sql_dynamic_agent.py)
  3. Result Analyzer  — combines/transforms results, computes comparisons
  4. Output Formatter — renders a clean table to the terminal
  5. Insights Generator— LLM-generated business narrative

Install dependencies:
    pip install langchain-openai langgraph sqlalchemy pandas tabulate pymysql python-dotenv

Run:
    python nl2sql_agent.py
    python nl2sql_agent.py "Compare current month sales with last 3 months"
"""

import os
import sys
import json
import time
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, date
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv(override=True)

# ── LangGraph / LangChain ────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Database ──────────────────────────────────────────────────────────────────
from sqlalchemy import create_engine

# ── Data & Display ────────────────────────────────────────────────────────────
import pandas as pd
from tabulate import tabulate

# ── SQL Sub-Agent ─────────────────────────────────────────────────────────────
from sql_dynamic_agent import SQLDynamicAgent


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

_db_host     = os.getenv("DB_HOST")
_db_port     = os.getenv("DB_PORT", "3306")
_db_user     = os.getenv("DB_USER")
_db_password = os.getenv("DB_PASSWORD")
_db_name     = os.getenv("DB_NAME")

if not all([_db_host, _db_user, _db_password, _db_name]):
    raise ValueError("One or more DB_* variables missing from .env (need DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)")

DB_CONNECTION_STRING = (
    f"mysql+pymysql://{_db_user}:{quote_plus(_db_password)}"
    f"@{_db_host}:{_db_port}/{_db_name}"
)

SCHEMA_FILE = os.getenv("SCHEMA_FILE", "cohort_main-schema_latest.sql")
MAX_RETRIES = 2
DEBUG_MODE  = True
LLM_MODEL   = "gpt-4o"


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED RESOURCES  (created once, reused across graph nodes)
# ═══════════════════════════════════════════════════════════════════════════════

_llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=4096,
)

_engine = create_engine(DB_CONNECTION_STRING)

_sql_agent = SQLDynamicAgent(
    llm=_llm,
    engine=_engine,
    max_retries=MAX_RETRIES,
    debug=DEBUG_MODE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_schema(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found: '{path}' — check SCHEMA_FILE in .env")


def call_llm(system: str, user: str) -> str:
    resp = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    if DEBUG_MODE:
        print(f"   [DEBUG] LLM response (600 chars): {resp.content[:600].replace(chr(10),' ')}")
    return resp.content


def parse_json(raw: str, is_array: bool = False) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = (raw.find("["), raw.rfind("]") + 1) if is_array else (raw.find("{"), raw.rfind("}") + 1)
        if start != -1 and end > 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON: {exc}\n{raw[:400]}")
        raise ValueError(f"Could not parse JSON:\n{raw[:400]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_query:        str
    schema:            str
    query_plan:        Optional[Dict[str, Any]]
    query_results:     Optional[List[Dict[str, Any]]]   # from SQLDynamicAgent
    analysis:          Optional[Dict[str, Any]]
    final_output:      Optional[str]
    insights:          Optional[str]
    error:             Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — QUERY PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

def query_planner(state: AgentState) -> AgentState:
    print("\n📋  [1/5] Planning execution strategy…")

    system = """You are an expert SQL architect.
Given a natural-language question and a database schema, create a JSON execution plan.

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "complexity": "simple" | "complex",
  "reasoning": "one-line explanation of strategy",
  "steps": [
    {
      "step_id": 1,
      "description": "what data this step retrieves",
      "depends_on": []
    }
  ],
  "output_type": "table" | "comparison" | "aggregate" | "summary"
}

Decision guide:
- simple  → one SQL query answers the question directly
- complex → needs several independent queries whose results must be merged in Python
Each step must be independently executable against the database.
"""

    user = (
        f"Today's date: {date.today().isoformat()} "
        f"(current month: {date.today().strftime('%B %Y')}, "
        f"current year: {date.today().year})\n\n"
        f"User question: {state['user_query']}\n\n"
        f"Database schema:\n{state['schema']}"
    )

    try:
        raw  = call_llm(system, user)
        plan = parse_json(raw)
        print(f"   ✅  {plan['complexity'].upper()} query | {len(plan['steps'])} step(s)")
        print(f"   📌  {plan['reasoning']}")
        if DEBUG_MODE:
            print(f"   [DEBUG] Plan:\n{json.dumps(plan, indent=4)}")
        return {**state, "query_plan": plan, "error": None}
    except Exception as exc:
        return {**state, "error": f"Planner failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — SQL DYNAMIC AGENT  (delegates to sql_dynamic_agent.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sql_dynamic_agent(state: AgentState) -> AgentState:
    """
    Hands the plan off to SQLDynamicAgent, which runs its own internal
    generate → execute → retry loop and returns the final result list.
    """
    print("\n🤖  [2/5] Running SQL Dynamic Agent…")
    try:
        results = _sql_agent.run(
            schema=state["schema"],
            plan=state["query_plan"],
            user_query=state["user_query"],
        )
        successful = sum(1 for r in results if r["success"])
        print(f"   ✅  {successful}/{len(results)} step(s) succeeded")
        return {**state, "query_results": results, "error": None}
    except RuntimeError as exc:
        return {**state, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 3 — RESULT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

def result_analyzer(state: AgentState) -> AgentState:
    print("\n🧠  [3/5] Analyzing & combining results…")

    good = [r for r in state["query_results"] if r["success"]]

    llm_payload = [
        {
            "alias":       r["result_alias"],
            "description": r["description"],
            "columns":     r["columns"],
            "row_count":   r["row_count"],
            "data":        r["data"][:200],
        }
        for r in good
    ]

    system = """You are a senior data analyst.
Given the raw query results, produce a final answer to the user's question.

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "summary":        "2–4 sentence narrative directly answering the question",
  "key_insights":   ["specific data-driven observation 1", "…"],
  "output_format":  "table" | "comparison_table" | "summary_only",
  "output_title":   "Short title for the results table",
  "output_columns": ["col1", "col2", "…"],
  "output_data":    [{"col1": "val", "col2": "val"}, …]
}

Rules:
• output_data must be the final, display-ready rows (merged, computed, sorted).
• For month-on-month comparisons: include absolute and % change columns.
• Format currency with commas; percentages with a % sign.
• Sort chronologically for time-series, by value descending for rankings.
• If output_format is "summary_only", output_data and output_columns may be empty.
• If an aggregate returned NULL or 0, report it as "₹0" or "0 records", not as unavailable.
"""

    user = (
        f"User question: {state['user_query']}\n\n"
        f"Query results:\n{json.dumps(llm_payload, indent=2, default=str)}"
    )

    if DEBUG_MODE:
        print(f"   [DEBUG] Analyzer payload (1000 chars):\n{json.dumps(llm_payload, indent=2, default=str)[:1000]}")

    try:
        raw      = call_llm(system, user)
        analysis = parse_json(raw)
        print(f"   ✅  Analysis complete ({analysis.get('output_format', '?')} output)")
        if DEBUG_MODE:
            print(f"   [DEBUG] Analysis:\n{json.dumps(analysis, indent=2, default=str)[:1000]}")
        return {**state, "analysis": analysis, "error": None}
    except Exception as exc:
        return {**state, "error": f"Analyzer failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 4 — OUTPUT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def output_formatter(state: AgentState) -> AgentState:
    print("\n📊  [4/5] Formatting output…\n")

    a    = state.get("analysis") or {}
    W    = 68
    SEP  = "─" * W
    DSEP = "═" * W

    lines = [DSEP, "  NL2SQL AGENT — RESULTS", DSEP, f"\n❓  {state['user_query']}\n"]

    data    = a.get("output_data", [])
    columns = a.get("output_columns", [])
    title   = a.get("output_title", "Results")

    if data and columns:
        lines += [SEP, f"📋  {title.upper()}", SEP]
        df = pd.DataFrame(data, columns=columns)
        lines.append(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
    else:
        lines += [SEP, a.get("summary", "(no data returned)")]

    # Show executed SQL
    lines += [f"\n{SEP}", "🔍  EXECUTED SQL", SEP]
    for r in (state.get("query_results") or []):
        sql_preview = r.get("sql", "").replace("\n", " ")
        if len(sql_preview) > 220:
            sql_preview = sql_preview[:220] + "…"
        lines.append(f"\nStep {r['step_id']} — {r['description']}")
        lines.append(f"  {sql_preview}")

    lines.append(f"\n{DSEP}")

    output = "\n".join(lines)
    print(output)
    return {**state, "final_output": output}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 5 — INSIGHTS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def insights_generator(state: AgentState) -> AgentState:
    print("\n💡  [5/5] Generating insights…\n")

    a            = state.get("analysis") or {}
    data         = a.get("output_data", [])
    columns      = a.get("output_columns", [])
    title        = a.get("output_title", "Results")
    summary      = a.get("summary", "")
    key_insights = a.get("key_insights", [])

    if data and columns:
        data_context = (
            f"Data table — {title}\n"
            f"Columns: {columns}\n"
            f"Rows:\n{json.dumps(data, indent=2, default=str)}"
        )
    else:
        data_context = (
            f"Analytical summary: {summary}\n"
            f"Key insights: {json.dumps(key_insights, default=str)}"
        )

    if DEBUG_MODE:
        print(f"   [DEBUG] Insights context (300 chars):\n   {data_context[:300]}")

    system = """You are a senior business analyst.

Read the data carefully and produce sharp, actionable insights.

Structure your response in plain text:

INSIGHT SUMMARY
One paragraph directly answering what the data tells us.

KEY OBSERVATIONS
1. [specific observation with exact numbers]
2. [trend, pattern, or anomaly]
3. [comparison or ratio that stands out]

RECOMMENDATION
One or two sentences on what action this data suggests.

Be specific — quote actual values. Avoid vague phrases. If the result is zero
or empty, acknowledge it and suggest next steps.
"""

    user = (
        f"User question: {state['user_query']}\n\n"
        f"{data_context}"
    )

    W    = 68
    DSEP = "═" * W

    try:
        raw = call_llm(system, user)
        lines = [f"\n{DSEP}", "  🧠  AI INSIGHTS", DSEP, "", raw.strip(), f"\n{DSEP}\n"]
        print("\n".join(lines))
        return {**state, "insights": raw.strip()}
    except Exception as exc:
        print(f"\n⚠️   Insights generation failed: {exc}")
        return {**state, "insights": None}


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR NODE
# ═══════════════════════════════════════════════════════════════════════════════

def handle_error(state: AgentState) -> AgentState:
    W = 68
    print(f"\n{'═'*W}\n  ❌  AGENT ERROR\n{'═'*W}")
    print(f"  {state.get('error', 'Unknown error')}")
    print(f"{'═'*W}\n")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ═══════════════════════════════════════════════════════════════════════════════

def _route(state: AgentState, next_node: str) -> str:
    return "error" if state.get("error") else next_node


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("query_planner",       query_planner)
    g.add_node("sql_dynamic_agent",   run_sql_dynamic_agent)
    g.add_node("result_analyzer",     result_analyzer)
    g.add_node("output_formatter",    output_formatter)
    g.add_node("insights_generator",  insights_generator)
    g.add_node("handle_error",        handle_error)

    g.set_entry_point("query_planner")

    g.add_conditional_edges("query_planner", lambda s: _route(s, "sql_dynamic_agent"), {
        "sql_dynamic_agent": "sql_dynamic_agent",
        "error":             "handle_error",
    })
    g.add_conditional_edges("sql_dynamic_agent", lambda s: _route(s, "result_analyzer"), {
        "result_analyzer": "result_analyzer",
        "error":           "handle_error",
    })
    g.add_conditional_edges("result_analyzer", lambda s: _route(s, "output_formatter"), {
        "output_formatter": "output_formatter",
        "error":            "handle_error",
    })
    g.add_edge("output_formatter",   "insights_generator")
    g.add_edge("insights_generator", END)
    g.add_edge("handle_error",       END)

    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(user_query: str) -> AgentState:
    W = 68
    print(f"\n{'═'*W}")
    print(f"  NL2SQL AGENT  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*W}")

    schema = load_schema(SCHEMA_FILE)

    initial: AgentState = {
        "user_query":    user_query,
        "schema":        schema,
        "query_plan":    None,
        "query_results": None,
        "analysis":      None,
        "final_output":  None,
        "insights":      None,
        "error":         None,
    }

    agent = build_graph()
    return agent.invoke(initial)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\n  Example queries:")
        print('  · "What is the total sales for this month?"')
        print('  · "Compare current month sales with the last 3 months, show month-on-month % change"')
        print('  · "Top 10 customers by revenue this quarter"')
        print()
        query = input("🔹  Enter your query: ").strip()
        if not query:
            query = "Compare current month sales with the last 3 months, show month-on-month change"

    run_agent(query)

#!/usr/bin/env python3
"""
nl2sql_agent.py — Natural Language → SQL Agent
================================================
Uses LangGraph to orchestrate a multi-step pipeline:
  1. Query Planner   — decides simple vs complex, creates step-by-step plan
  2. SQL Generator   — writes one SQL query per step
  3. SQL Executor    — runs queries, captures results as DataFrames
  4. Result Analyzer — combines/transforms results, computes comparisons
  5. Output Formatter— renders a clean table + narrative to the terminal

Install dependencies:
    pip install langchain-openai langgraph sqlalchemy pandas tabulate

Run:
    python nl2sql_agent.py
    python nl2sql_agent.py "Compare current month sales with last 3 months"
"""

import os
import sys
import json
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

# ── LangGraph / LangChain ────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Database ─────────────────────────────────────────────────────────────────
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ── Data & Display ────────────────────────────────────────────────────────────
import pandas as pd
from tabulate import tabulate

from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not found")

DB_CONNECTION_STRING = (
    f"mysql+pymysql://"
    f"{DB_USER}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCHEMA_FILE = os.path.join(
    BASE_DIR,
    "cohort_main-schema_latest.sql"
)

LLM_MODEL = "gpt-4o"
MAX_RETRIES = 2


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_schema(path: str) -> str:
    """Read the schema SQL file."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌  Schema file '{path}' not found. Please create it or update SCHEMA_FILE.")
        sys.exit(1)


def execute_sql(sql: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Run a single SQL query.
    Returns (DataFrame, None) on success, (empty DataFrame, error_msg) on failure.
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            return df, None
    except SQLAlchemyError as e:
        return pd.DataFrame(), str(e)
    except Exception as e:
        return pd.DataFrame(), str(e)


def parse_json(raw: str, is_array: bool = False) -> Any:
    """
    Safely parse JSON from an LLM response that may have surrounding prose.
    """
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first JSON object / array
        if is_array:
            start, end = raw.find("["), raw.rfind("]") + 1
        else:
            start, end = raw.find("{"), raw.rfind("}") + 1

        if start != -1 and end > 0:
            return json.loads(raw[start:end])

        raise ValueError(f"Could not parse JSON:\n{raw[:400]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_query:        str
    schema:            str
    query_plan:        Optional[Dict[str, Any]]    # output of planner
    generated_queries: Optional[List[Dict[str, Any]]]  # list of {step_id, sql, …}
    query_results:     Optional[List[Dict[str, Any]]]  # execution results
    failed_queries:    Optional[List[Dict[str, Any]]]  # failed steps for retry
    analysis:          Optional[Dict[str, Any]]    # combined/transformed result
    final_output:      Optional[str]               # rendered terminal string
    insights:          Optional[str]               # LLM-generated insight narrative
    error:             Optional[str]               # set on any node failure
    retry_count:       int                         # incremented on each retry


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

_llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=4096,
)


def call_llm(system: str, user: str) -> str:
    resp = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — QUERY PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

def query_planner(state: AgentState) -> AgentState:
    """
    Reads the user question + schema and produces a JSON execution plan.
    Decides whether the question needs one query or multiple coordinated queries.
    """
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
      "depends_on": []          // list of step_ids whose data this step needs (usually [])
    }
  ],
  "output_type": "table" | "comparison" | "aggregate" | "summary"
}

Decision guide:
- simple  → one SQL query answers the question directly (or a single CTE could do it)
- complex → needs several independent queries whose results must be merged in Python
             e.g. current-month data + prior months + computed % change
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
        raw = call_llm(system, user)
        plan = parse_json(raw)
        print(f"   ✅  {plan['complexity'].upper()} query | {len(plan['steps'])} step(s)")
        print(f"   📌  {plan['reasoning']}")
        return {**state, "query_plan": plan, "error": None}
    except Exception as exc:
        return {**state, "error": f"Planner failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — SQL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def sql_generator(state: AgentState) -> AgentState:
    """
    Produces one SQL query per plan step.
    On retries, receives the previous failure reasons so it can self-correct.
    """
    retry = state.get("retry_count", 0)
    label = f" (retry #{retry})" if retry else ""
    print(f"\n🔧  [2/5] Generating SQL queries{label}…")

    # Build retry context so the LLM can fix its own mistakes
    retry_block = ""
    if retry and state.get("failed_queries"):
        retry_block = (
            "\n\nPREVIOUS ATTEMPT FAILED — fix these errors before regenerating:\n"
            + json.dumps(state["failed_queries"], indent=2)
        )

    system = """You are an expert SQL developer.
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

MYSQL 8.0 DIALECT — this database is MySQL 8.0. Follow these rules exactly:
• NEVER use DATE_TRUNC — it does not exist in MySQL. Use YEAR() and MONTH() instead.
• NEVER use DATEADD — use DATE_ADD(date, INTERVAL n DAY/MONTH/YEAR) instead.
• NEVER use ISNULL() — use IS NULL or COALESCE() instead.
• NEVER use TOP n — use LIMIT n instead.
• Allowed date functions: YEAR(), MONTH(), DAY(), DATE(), DATE_FORMAT(),
  DATE_ADD(), DATE_SUB(), DATEDIFF(), CURDATE(), NOW(), STR_TO_DATE().
• Filter by month using: WHERE YEAR(col) = 2026 AND MONTH(col) = 3
• Filter by date range using: WHERE col >= '2026-03-01' AND col < '2026-04-01'

DATE RESOLUTION RULES — critical, always follow:
• Today's date is provided in the user message. Use it as the anchor for ALL date logic.
• Named months (e.g. "April", "March") always refer to that month in the current year
  UNLESS the named month is in the future, in which case use the previous year.
• "Previous N months" means the N calendar months immediately before the referenced month —
  compute their exact year and month numbers using the anchor date; never use INTERVAL guesses.
• Never compute date ranges with arbitrary INTERVALs. Always derive exact YEAR() and MONTH()
  values from the anchor date so the SQL is deterministic and correct.
• Example: if today is 2026-06-06 and the user says "April vs previous 3 months":
    - April  → MONTH = 4, YEAR = 2026
    - Previous 3 months → January (1), February (2), March (3) all in 2026
"""

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
        raw = call_llm(system, user)
        queries = parse_json(raw, is_array=True)
        print(f"   ✅  {len(queries)} quer{'y' if len(queries)==1 else 'ies'} generated")
        for q in queries:
            print(f"   📝  Step {q['step_id']}: {q['description']}")
        return {**state, "generated_queries": queries, "error": None}
    except Exception as exc:
        return {**state, "error": f"SQL generator failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 3 — SQL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def sql_executor(state: AgentState) -> AgentState:
    """
    Runs each generated SQL query against the real database.
    Keeps both successes and failures so the retry mechanism can be selective.
    """
    print("\n⚡  [3/5] Executing queries…")

    results: List[Dict[str, Any]] = []
    failed:  List[Dict[str, Any]] = []

    for q in state["generated_queries"]:
        sid = q["step_id"]
        print(f"   🔄  Step {sid}: {q['description']}")
        df, err = execute_sql(q["sql"])

        if err:
            print(f"   ❌  Step {sid} failed: {err}")
            failed.append({"step_id": sid, "description": q["description"],
                           "sql": q["sql"], "error": err})
            results.append({**q, "success": False, "error": err,
                            "data": [], "columns": [], "row_count": 0})
        else:
            print(f"   ✅  Step {sid}: {len(df)} row(s) returned")
            results.append({**q, "success": True, "error": None,
                            "data": df.to_dict(orient="records"),
                            "columns": list(df.columns),
                            "row_count": len(df)})

    return {
        **state,
        "query_results":  results,
        "failed_queries": failed or None,
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 4 — RESULT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

def result_analyzer(state: AgentState) -> AgentState:
    """
    Merges results from all executed queries, applies any Python-level
    computations (% change, ranking, pivoting) and builds the final answer.
    """
    print("\n🧠  [4/5] Analyzing & combining results…")

    good = [r for r in state["query_results"] if r["success"]]

    # Cap data sent to LLM to avoid token overflow
    llm_payload = [
        {
            "alias":       r["result_alias"],
            "description": r["description"],
            "columns":     r["columns"],
            "row_count":   r["row_count"],
            "data":        r["data"][:200],   # at most 200 rows
        }
        for r in good
    ]

    system = """You are a senior data analyst.
Given the raw query results, produce a final answer to the user's question.
IMPORTANT BUSINESS CONTEXT:
• All monetary values are in Indian Rupees (INR).
• Always use the ₹ symbol when discussing revenue, sales, invoices, profits, or monetary amounts.
• Never use $ or USD.

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
• For month-on-month comparisons: compute the absolute and % change and include them as columns.
• Format currency with commas; format percentages with a % sign.
• Sort chronologically for time-series, by value descending for rankings.
• If output_format is "summary_only", output_data and output_columns may be empty.
"""

    user = (
        f"User question: {state['user_query']}\n\n"
        f"Query results:\n{json.dumps(llm_payload, indent=2, default=str)}"
    )

    try:
        raw = call_llm(system, user)
        analysis = parse_json(raw)
        print(f"   ✅  Analysis complete ({analysis.get('output_format', '?')} output)")
        return {**state, "analysis": analysis, "error": None}
    except Exception as exc:
        return {**state, "error": f"Analyzer failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 5 — OUTPUT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def output_formatter(state: AgentState) -> AgentState:
    """
    Renders the data table and executed SQL.
    Insights are printed separately by insights_generator (next node).
    """
    print("\n📊  [5/6] Formatting data output…\n")

    a    = state["analysis"]
    W    = 68
    SEP  = "─" * W
    DSEP = "═" * W

    lines = [
        DSEP,
        "  NL2SQL AGENT — RESULTS",
        DSEP,
        f"\n❓  {state['user_query']}\n",
    ]

    data    = a.get("output_data", [])
    columns = a.get("output_columns", [])
    title   = a.get("output_title", "Results")

    if data and columns:
        lines += [SEP, f"📋  {title.upper()}", SEP]
        df = pd.DataFrame(data, columns=columns)
        lines.append(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
    else:
        lines += [SEP, a.get("summary", "(no data returned)")]

    # Show executed SQL for transparency / debugging
    lines += [f"\n{SEP}", "🔍  EXECUTED SQL", SEP]
    for q in (state.get("generated_queries") or []):
        sql_preview = q["sql"].replace("\n", " ")
        if len(sql_preview) > 220:
            sql_preview = sql_preview[:220] + "…"
        lines.append(f"\nStep {q['step_id']} — {q['description']}")
        lines.append(f"  {sql_preview}")

    lines.append(f"\n{DSEP}")

    output = "\n".join(lines)
    print(output)
    return {**state, "final_output": output}




# ═══════════════════════════════════════════════════════════════════════════════
#  NODE 6 — INSIGHTS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def insights_generator(state: AgentState) -> AgentState:
    """
    Second pass of intelligence: takes the final display-ready data and asks
    the LLM to reason over it and produce business-level insights — trends,
    anomalies, recommendations, and anything worth acting on.
    """
    print("\n💡  [6/6] Generating insights…\n")

    a       = state["analysis"]
    data    = a.get("output_data", [])
    columns = a.get("output_columns", [])
    title   = a.get("output_title", "Results")

    system = """You are a senior business analyst with deep expertise in data interpretation.

IMPORTANT BUSINESS CONTEXT:
• All monetary values are in Indian Rupees (INR).
• Always use the ₹ symbol when discussing revenue, sales, invoices, profits, or monetary amounts.
• Never use $ or USD.
You will receive a data table that was produced in response to a user's question.
Your job is to read the numbers carefully and produce sharp, actionable insights.

Structure your response in plain text (no markdown headers, no bullet symbols — 
use numbered points for the insight list):

INSIGHT SUMMARY
One paragraph that directly answers what the data is telling us.

KEY OBSERVATIONS
1. [specific observation with exact numbers from the data]
2. [trend, pattern, or anomaly worth noting]
3. [comparison or ratio that stands out]
(add more if genuinely useful — do not pad)

RECOMMENDATION
One or two sentences on what action or investigation this data suggests.

Be specific — quote actual values from the data. Avoid vague phrases like
"performance varies" or "there is a trend". If the data is too small to draw
conclusions, say so honestly.
"""

    user = (
        f"User question: {state['user_query']}\n\n"
        f"Data table — {title}\n"
        f"Columns: {columns}\n"
        f"Rows:\n{json.dumps(data, indent=2, default=str)}"
    )

    W    = 68
    SEP  = "─" * W
    DSEP = "═" * W

    try:
        raw = call_llm(system, user)
        insights_text = raw.strip()

        lines = [
            f"\n{DSEP}",
            "  🧠  AI INSIGHTS",
            DSEP,
            "",
            insights_text,
            f"\n{DSEP}\n",
        ]
        output = "\n".join(lines)
        print(output)
        return {**state, "insights": insights_text}
    except Exception as exc:
        print(f"\n⚠️   Insights generation failed: {exc}")
        return {**state, "insights": None}

# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR NODE
# ═══════════════════════════════════════════════════════════════════════════════

def handle_error(state: AgentState) -> AgentState:
    W = 68
    print(f"\n{'═'*W}")
    print("  ❌  AGENT ERROR")
    print(f"{'═'*W}")
    print(f"  {state.get('error', 'Unknown error')}")
    print(f"{'═'*W}\n")
    return state


def increment_retry(state: AgentState) -> AgentState:
    """Bump retry counter before looping back to the SQL generator."""
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


# ═══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_planner(state: AgentState) -> str:
    return "error" if state.get("error") else "sql_generator"


def route_after_generator(state: AgentState) -> str:
    return "error" if state.get("error") else "sql_executor"


def route_after_executor(state: AgentState) -> str:
    results     = state.get("query_results", [])
    failed      = state.get("failed_queries") or []
    retry_count = state.get("retry_count", 0)
    any_success = any(r["success"] for r in results)
    all_failed  = not any_success

    if failed and all_failed and retry_count < MAX_RETRIES:
        return "retry"
    if any_success:
        return "result_analyzer"
    return "error"


def route_after_analyzer(state: AgentState) -> str:
    return "error" if state.get("error") else "output_formatter"


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("query_planner",    query_planner)
    g.add_node("sql_generator",    sql_generator)
    g.add_node("sql_executor",     sql_executor)
    g.add_node("increment_retry",  increment_retry)
    g.add_node("result_analyzer",  result_analyzer)
    g.add_node("output_formatter",   output_formatter)
    g.add_node("insights_generator", insights_generator)
    g.add_node("handle_error",       handle_error)

    # Entry
    g.set_entry_point("query_planner")

    # Edges
    g.add_conditional_edges("query_planner", route_after_planner, {
        "sql_generator": "sql_generator",
        "error":         "handle_error",
    })
    g.add_conditional_edges("sql_generator", route_after_generator, {
        "sql_executor": "sql_executor",
        "error":        "handle_error",
    })
    g.add_conditional_edges("sql_executor", route_after_executor, {
        "result_analyzer": "result_analyzer",
        "retry":           "increment_retry",   # bump counter then re-generate
        "error":           "handle_error",
    })
    g.add_edge("increment_retry", "sql_generator")   # retry loop
    g.add_conditional_edges("result_analyzer", route_after_analyzer, {
        "output_formatter": "output_formatter",
        "error":            "handle_error",
    })
    g.add_edge("output_formatter",   "insights_generator")
    g.add_edge("insights_generator", END)
    g.add_edge("handle_error",     END)

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
        "user_query":        user_query,
        "schema":            schema,
        "query_plan":        None,
        "generated_queries": None,
        "query_results":     None,
        "failed_queries":    None,
        "analysis":          None,
        "final_output":      None,
        "insights":          None,
        "error":             None,
        "retry_count":       0,
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
        print("\n  Example queries you can try:")
        print('  · "What is the total sales for this month?"')
        print('  · "Compare current month sales with the last 3 months, show month-on-month % change"')
        print('  · "Top 10 customers by revenue this quarter"')
        print('  · "Average invoice value per region for the last 6 months"')
        print()
        query = input("🔹  Enter your query: ").strip()
        if not query:
            query = "Compare current month sales with the last 3 months, show month-on-month change"

    run_agent(query)

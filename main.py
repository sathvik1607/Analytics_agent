import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from nl2sql_agent_with_insights import run_agent


# ==================================================
# ENVIRONMENT
# ==================================================
# Required .env variables:
#   OPENAI_API_KEY
#   DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
#
# Optional .env variables:
#   ALLOWED_ORIGINS  — comma-separated frontend origins
#                      default: http://localhost:5175,
#                               http://127.0.0.1:5175
# ==================================================

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

db_host     = os.getenv("DB_HOST")
db_port     = os.getenv("DB_PORT")
db_user     = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name     = os.getenv("DB_NAME")

if not all([db_host, db_port, db_user, db_password, db_name]):
    raise ValueError(
        "One or more DB_* variables missing from .env"
    )

DB_CONNECTION_STRING = (
    f"mysql+pymysql://"
    f"{db_user}:{quote_plus(db_password)}"
    f"@{db_host}:{db_port}/{db_name}"
)

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5175,http://127.0.0.1:5175"
)
ALLOWED_ORIGINS = [
    o.strip() for o in _raw_origins.split(",") if o.strip()
]

print("\n===== STARTUP =====")
print(f"OPENAI KEY LOADED: {api_key[:8]}...{api_key[-4:]}")
print(f"ALLOWED ORIGINS:   {ALLOWED_ORIGINS}")


# ==================================================
# GENERAL ASSISTANT LLM
# ==================================================
# A lightweight GPT-3.5 instance used only by /chat.
#
# Purpose:
#   Guide users toward analytics questions.
#   Does NOT answer general knowledge questions.
#   Does NOT query the database.
# ==================================================

_chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    temperature=0.5,
    max_tokens=256,
)

_CHAT_SYSTEM_PROMPT = """
You are a friendly assistant for an AI-powered Analytics Platform.

Your only job is to help users ask the right questions.

When a user sends a general or off-topic message (greetings, jokes,
general knowledge, coding questions, etc.):
- Respond briefly and warmly in 1 sentence.
- Then suggest 2-3 specific analytics questions they could ask
  on this platform instead.

When a user asks something analytics-related (sales, revenue,
invoices, customers, KPIs, trends, reports):
- Tell them it is a great question.
- Ask them to switch to Analytics Mode to get a
  data-driven answer directly from the database.

Examples of analytics questions on this platform:
- "What is the total revenue for this month?"
- "Top 10 customers by invoice value"
- "Month-on-month sales comparison for Q1"
- "Average invoice amount by region"
- "How many invoices were cancelled last quarter?"

Keep all responses short — 3 to 5 sentences maximum.
Never answer analytics questions yourself.
Never make up data or numbers.
"""

print("ASSISTANT LLM INITIALIZED")


# ==================================================
# STARTUP — DB CONNECTIVITY CHECK
# ==================================================
# Users: internal — runs automatically on boot
#
# Purpose:
#   Verifies the database is reachable before the
#   server accepts any traffic.
#
# Behaviour:
#   - Runs a lightweight SELECT 1 against the DB
#   - Prints OK or FAILED to the startup log
#   - Raises RuntimeError and halts startup if DB
#     is unreachable — fail fast, not on first query
# ==================================================

@asynccontextmanager
async def lifespan(_: FastAPI):

    try:

        engine = create_engine(DB_CONNECTION_STRING)

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        print("DB CONNECTION:     OK")

    except SQLAlchemyError as e:

        print(f"DB CONNECTION:     FAILED — {e}")
        raise RuntimeError(
            f"Database unreachable at startup: {e}"
        )

    yield


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Analytics Platform",
    version="2.0.0",
    description="NL2SQL analytics backend. No chat. No general AI.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("FASTAPI INITIALIZED")


# ==================================================
# REQUEST MODELS
# ==================================================
# /chat accepts:
#   { "message": "..." }
#
# /analytics and /analytics/debug accept:
#   { "question": "..." }
#
# Both models:
#   - reject empty or whitespace-only input (422)
#   - strip leading/trailing whitespace
#   - validate before any LLM or DB call is made
# ==================================================

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()


# ==================================================
# ROOT
# ==================================================
# Users: monitoring, uptime checks, service discovery
#
# Returns basic service identity.
# No sensitive information exposed.
# ==================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Analytics Platform",
        "version": "2.0.0"
    }


# ==================================================
# HEALTH CHECK
# ==================================================
# Users: frontend, load balancers, monitoring systems
#
# Used to confirm the service is reachable.
# Does NOT validate DB connectivity or LLM availability.
# ==================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


# ==================================================
# CHAT ENDPOINT
# ==================================================
# Users: All users — entry point before Analytics Mode
#
# Purpose:
#   A lightweight guide that helps users understand
#   what this platform can do and steers them toward
#   asking analytics questions.
#
# Behaviour:
#   - General/off-topic messages → brief reply +
#     2-3 suggested analytics questions
#   - Analytics questions → confirms it is a good
#     question and asks user to switch to Analytics Mode
#
# Does NOT:
#   - Query the database
#   - Return any data or numbers
#   - Answer general knowledge questions in full
#
# Data flow:
#   Frontend → POST /chat
#       → _chat_llm.invoke()        [non-blocking thread]
#   → Plain text response
# ==================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        def _invoke():
            return _chat_llm.invoke([
                SystemMessage(content=_CHAT_SYSTEM_PROMPT),
                HumanMessage(content=request.message),
            ])

        response = await asyncio.to_thread(_invoke)

        return {
            "success": True,
            "message": request.message,
            "reply":   response.content,
        }

    except Exception as e:

        return {
            "success": False,
            "message": request.message,
            "reply":   None,
            "error":   str(e),
        }


# ==================================================
# ANALYTICS ENDPOINT
# ==================================================
# Users: Business users via the frontend
#
# Purpose:
#   Accepts a natural-language business question and
#   returns a clean, human-readable answer.
#
# Supported question types:
#   - Revenue analysis        ("Total revenue this month")
#   - Sales reports           ("Top 10 customers by sales")
#   - Invoice analysis        ("Average invoice value by region")
#   - Customer analytics      ("New vs returning customers")
#   - KPI reporting           ("Conversion rate for Q1")
#   - Financial trends        ("Month-on-month revenue change")
#   - Operational reporting   ("Orders pending dispatch")
#   - Dashboard insights      ("Summary of this week's activity")
#
# Data flow:
#   Frontend → POST /analytics
#       → run_agent(question)       [non-blocking thread]
#           → Query Planner
#           → SQL Generator
#           → SQL Executor
#           → Result Analyzer
#           → Insight Generator
#       → strip internals
#   → Business-facing response
#
# Exposed to the frontend:
#   success   — whether the pipeline completed without error
#   question  — the original question (for display)
#   analysis  — structured result: summary, key insights, output table
#   insights  — LLM-generated narrative with observations and recommendations
#   error     — human-readable error message if something failed
#
# Intentionally hidden from the frontend:
#   query_plan        — internal planning step
#   generated_queries — raw SQL
#   query_results     — raw DB rows
#   retry_count       — execution internals
#   schema            — database structure
#   final_output      — terminal-formatted string (not for API consumers)
# ==================================================

@app.post("/analytics")
async def analytics(request: QueryRequest):

    try:

        result = await asyncio.to_thread(
            run_agent, request.question
        )

        return {
            "success":  result.get("error") is None,
            "question": request.question,
            "analysis": result.get("analysis"),
            "insights": result.get("insights"),
            "error":    result.get("error"),
        }

    except Exception as e:

        return {
            "success":  False,
            "question": request.question,
            "analysis": None,
            "insights": None,
            "error":    str(e),
        }


# ==================================================
# ANALYTICS DEBUG ENDPOINT
# ==================================================
# Users: Developers, QA, support — NOT business users
#
# Purpose:
#   Returns the full internal state of the NL2SQL
#   pipeline for diagnosing unexpected behavior.
#
# Only called when the frontend's Debug toggle is ON.
# Must never be accessible to end users in production.
#
# Data flow:
#   Frontend (debug mode) → POST /analytics/debug
#       → run_agent(question)       [non-blocking thread]
#   → Full pipeline state returned as-is
#
# Exposed (everything the pipeline produces):
#   query_plan        — planner output: complexity, steps, strategy
#   generated_queries — the SQL queries written by the LLM
#   query_results     — raw rows returned from the database
#   retry_count       — how many retries were attempted
#   analysis          — combined and transformed result object
#   insights          — LLM-generated narrative
#   final_output      — terminal-formatted output string
#   error             — error message if the pipeline failed
#
# Use this endpoint to diagnose:
#   - Wrong SQL being generated
#   - Incorrect date range resolution
#   - Missing or empty query results
#   - Analyzer misinterpreting data
#   - Retry loops triggering unexpectedly
# ==================================================

@app.post("/analytics/debug")
async def analytics_debug(request: QueryRequest):

    try:

        result = await asyncio.to_thread(
            run_agent, request.question
        )

        return result

    except Exception as e:

        return {
            "success":  False,
            "question": request.question,
            "error":    str(e),
        }

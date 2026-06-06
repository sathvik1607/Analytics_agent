import asyncio
import json
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

from nl2sql_agent import run_agent

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
    raise ValueError("One or more DB_* variables missing from .env")

DB_CONNECTION_STRING = (
    f"mysql+pymysql://"
    f"{db_user}:{quote_plus(db_password)}"
    f"@{db_host}:{db_port}/{db_name}"
)

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5175,http://127.0.0.1:5175"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

print("\n===== STARTUP =====")
print(f"OPENAI KEY LOADED: {api_key[:8]}...{api_key[-4:]}")
print(f"ALLOWED ORIGINS:   {ALLOWED_ORIGINS}")


# ==================================================
# INTENT CLASSIFIER LLM
# Lightweight model — runs before every /analytics
# call to decide: analytics query or general question.
# ==================================================

_intent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0,
    max_tokens=256,
)

_INTENT_SYSTEM = """You are an intent classifier for a business analytics platform that answers questions about sales, invoices, revenue, customers, and related data.

Classify the user message and respond with ONLY valid JSON — no markdown, no explanation.

If the message is a data/analytics question (sales, revenue, invoices, customers, KPIs, trends, comparisons, reports, products, dates, amounts):
{"intent": "analytics"}

If the message is general (greetings, "what can you do", help, off-topic, chitchat):
{"intent": "general", "reply": "<2-3 sentence response explaining what this platform does and example questions they can ask>"}

Example analytics questions this platform handles:
- "What are the total sales for June 2026?"
- "Compare revenue for March vs February"
- "Top 10 customers by invoice value"
- "Monthly sales trend for 2026"
"""


def _classify_intent(question: str) -> dict:
    resp = _intent_llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=question),
    ])
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


print("INTENT CLASSIFIER INITIALIZED")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("DB CONNECTION:     OK")
    except SQLAlchemyError as e:
        print(f"DB CONNECTION:     FAILED — {e}")
        raise RuntimeError(f"Database unreachable at startup: {e}")
    yield


app = FastAPI(
    title="Analytics Platform",
    version="2.0.0",
    description="NL2SQL analytics backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("FASTAPI INITIALIZED")


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()


_AGENT_TIMELINE = [
    {"step": "Understanding Question", "icon": "brain",     "status": "completed"},
    {"step": "Finding Relevant Data",  "icon": "search",    "status": "completed"},
    {"step": "Gathering Data",         "icon": "database",  "status": "completed"},
    {"step": "Analyzing Results",      "icon": "chart",     "status": "completed"},
    {"step": "Generating Insights",    "icon": "lightbulb", "status": "completed"},
]


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/analytics")
async def analytics(request: QueryRequest):
    try:
        # Step 1 — classify intent (fast, cheap)
        intent_result = await asyncio.to_thread(_classify_intent, request.question)

        if intent_result.get("intent") == "general":
            return {
                "success":        True,
                "question":       request.question,
                "intent":         "general",
                "reply":          intent_result.get("reply", ""),
                "analysis":       None,
                "insights":       None,
                "error":          None,
                "agent_timeline": [],
            }

        # Step 2 — analytics query: run full NL2SQL pipeline
        result = await asyncio.to_thread(run_agent, request.question)
        success = result.get("error") is None
        return {
            "success":        success,
            "question":       request.question,
            "intent":         "analytics",
            "reply":          None,
            "analysis":       result.get("analysis"),
            "insights":       result.get("insights"),
            "error":          result.get("error"),
            "agent_timeline": _AGENT_TIMELINE if success else [],
        }

    except Exception as e:
        return {
            "success":        False,
            "question":       request.question,
            "intent":         None,
            "reply":          None,
            "analysis":       None,
            "insights":       None,
            "error":          str(e),
            "agent_timeline": [],
        }

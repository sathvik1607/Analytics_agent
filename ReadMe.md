# Analytics Platform — NL2SQL Backend

A FastAPI backend that converts natural-language business questions into SQL queries, executes them against a MySQL database, and returns structured analysis with AI-generated insights.

---

## Architecture

```
main.py                  ← FastAPI app, intent classifier, HTTP endpoints
nl2sql_agent.py          ← LangGraph orchestrator (5-node pipeline)
sql_dynamic_agent.py     ← SQL sub-agent (generate → execute → retry loop)
```

**Pipeline flow:**

```
User Question
     │
     ▼
Intent Classifier (gpt-4o-mini)
     │
     ├── general  → direct reply (no DB hit)
     │
     └── analytics
          │
          ▼
     Query Planner  →  SQL Dynamic Agent  →  Result Analyzer  →  Output Formatter  →  Insights Generator
```

---

## Endpoints

### `GET /health`

Health check.

**Response**
```json
{ "status": "healthy" }
```

---

### `POST /analytics`

Submit a natural-language question. The intent classifier decides whether to run the full NL2SQL pipeline or answer directly.

**Request**
```json
{
  "question": "What are the total sales for June 2026?"
}
```

**Response — analytics question**
```json
{
  "success": true,
  "question": "What are the total sales for June 2026?",
  "intent": "analytics",
  "reply": null,
  "analysis": {
    "summary": "Total sales for June 2026 were ₹1,23,456.",
    "key_insights": ["Sales grew 12% vs May 2026"],
    "output_format": "table",
    "output_title": "June 2026 Sales",
    "output_columns": ["Month", "Total Sales"],
    "output_data": [{ "Month": "June 2026", "Total Sales": "₹1,23,456" }]
  },
  "insights": "INSIGHT SUMMARY\n...",
  "error": null,
  "agent_timeline": [
    { "step": "Understanding Question", "icon": "brain",     "status": "completed" },
    { "step": "Finding Relevant Data",  "icon": "search",    "status": "completed" },
    { "step": "Gathering Data",         "icon": "database",  "status": "completed" },
    { "step": "Analyzing Results",      "icon": "chart",     "status": "completed" },
    { "step": "Generating Insights",    "icon": "lightbulb", "status": "completed" }
  ]
}
```

**Response — general question**
```json
{
  "success": true,
  "question": "What can you do?",
  "intent": "general",
  "reply": "I'm an analytics assistant that can answer questions about your sales, invoices, and customers...",
  "analysis": null,
  "insights": null,
  "error": null,
  "agent_timeline": []
}
```

**Response — error**
```json
{
  "success": false,
  "question": "...",
  "intent": "analytics",
  "reply": null,
  "analysis": null,
  "insights": null,
  "error": "Schema file not found: ...",
  "agent_timeline": []
}
```

---

## Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn langchain-openai langgraph sqlalchemy pymysql pandas tabulate python-dotenv
```

### 2. Configure `.env`

```env
OPENAI_API_KEY=sk-...
DB_HOST=your-db-host
DB_PORT=3306
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=your-db-name
SCHEMA_FILE=etc/secrets/cohort_main-schema_latest.sql
ALLOWED_ORIGINS=http://localhost:5175
```

### 3. Run locally

```bash
uvicorn main:app --reload --port 8000
```

---

## Deployment (Render)

1. Push code to GitHub — `.env` and `*.sql` are already gitignored.
2. On Render, set all environment variables from `.env` in the dashboard.
3. Upload the schema file as a **Secret File** at `/etc/secrets/db-schema.sql`.
4. Set `SCHEMA_FILE=/etc/secrets/db-schema.sql` in Render's environment variables.
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `DB_HOST` | Yes | MySQL host |
| `DB_PORT` | No | MySQL port (default: 3306) |
| `DB_USER` | Yes | MySQL username |
| `DB_PASSWORD` | Yes | MySQL password |
| `DB_NAME` | Yes | MySQL database name |
| `SCHEMA_FILE` | No | Path to schema SQL file (default: `cohort_main-schema_latest.sql`) |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins (default: `http://localhost:5175`) |

---

## Key Notes

- `invoice_date` is stored as `VARCHAR` in `MM/DD/YYYY HH:MM:SS` format — the SQL agent handles date filtering automatically using `LIKE` patterns.
- The intent classifier (`gpt-4o-mini`) runs before every `/analytics` call to avoid hitting the database for general questions.
- The schema file is never committed to git (`*.sql` is in `.gitignore`).

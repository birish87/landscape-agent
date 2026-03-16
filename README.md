# Granum Landscape Intelligence Platform
## A Multi-Agent AI Tutorial Project

> Built to demonstrate the skills required for the **Granum AI Engineer role** ($160k–$200k USD).
> Covers: agent orchestration, RAG, vector databases, LLM tool calling, and production deployment patterns.

---

## Table of Contents

1. [What You're Building](#what-youre-building)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Getting Your API Key](#getting-your-api-key)
7. [The Agents Explained](#the-agents-explained)
8. [The Tools Explained](#the-tools-explained)
9. [RAG: Knowledge Retrieval](#rag-knowledge-retrieval)
10. [Running the App](#running-the-app)
11. [Example Conversations](#example-conversations)
12. [Connecting Real Data Sources](#connecting-real-data-sources)
13. [Production Upgrade Path](#production-upgrade-path)
14. [Alignment to Granum Job Description](#alignment-to-granum-job-description)

---

## What You're Building

A **multi-agent AI platform** for landscaping companies that can:

- Generate job quotes from natural language requests
- Schedule crews based on availability
- Analyze yard/tree conditions and recommend services
- Answer procedure questions using a knowledge base (RAG)
- Generate invoices

This project is intentionally structured to match what a $160k–$200k AI Engineer would build at a company like **Granum** (makers of LMN, SingleOps, Greenius).

---

## Architecture Overview

```
User Message
     │
     ▼
┌─────────────────────────┐
│   Master Orchestrator   │  ← Classifies intent, routes to agents
│   (Claude API call #1)  │
└────────────┬────────────┘
             │
    ┌────────┼──────────┐
    ▼        ▼          ▼
┌───────┐ ┌──────┐ ┌────────┐
│ Site  │ │ Ops  │ │  RAG   │
│Analysis│ │Agent │ │ Agent  │
│Agent  │ │      │ │        │
└───────┘ └──────┘ └────────┘
    │        │          │
    │    ┌───┴───┐       │
    │    │ Tools │       │
    │    │  •estimate_job_cost()
    │    │  •check_crew_availability()
    │    │  •generate_invoice()
    │    └───────┘       │
    │                    │
    └────────┬───────────┘
             ▼
    Final Synthesis Agent
    (combines all results)
             │
             ▼
      Response to User
```

**Key Pattern**: The orchestrator never answers the user directly. It only routes. Each specialist agent handles one domain. This is called the **"fan-out / merge"** pattern and is standard in production agentic systems.

---

## Prerequisites

- Python 3.9+ (for the Python version)
- Node.js 18+ (for the JavaScript/TypeScript version)
- An Anthropic API key (see below)
- Basic understanding of REST APIs and async programming

No prior AI/ML experience required — this tutorial explains everything.

---

## Project Structure

```
landscape-agent/
├── README.md                    ← You are here
├── .env.example                 ← Copy to .env and add your API key
├── requirements.txt             ← Python dependencies
├── package.json                 ← Node.js dependencies
│
├── src/
│   ├── main.py                  ← Entry point (Python version)
│   ├── main.js                  ← Entry point (Node.js version)
│   │
│   ├── agents/
│   │   ├── orchestrator.py      ← Routes requests to sub-agents
│   │   ├── operations_agent.py  ← Handles quotes, scheduling, invoices
│   │   ├── site_analysis_agent.py ← Analyzes yard/tree descriptions
│   │   └── knowledge_agent.py   ← RAG-powered procedure lookup
│   │
│   ├── tools/
│   │   ├── estimator.py         ← estimate_job_cost() tool
│   │   ├── scheduler.py         ← check_crew_availability() tool
│   │   └── invoicer.py          ← generate_invoice() tool
│   │
│   ├── data/
│   │   ├── dummy_db.json        ← Fake crews, pricing, jobs (start here)
│   │   ├── knowledge_docs.json  ← Service procedure documents for RAG
│   │   └── seed_vector_db.py    ← Script to load docs into ChromaDB
│   │
│   └── utils/
│       ├── claude_client.py     ← Anthropic API wrapper with retry logic
│       └── prompts.py           ← All system prompts in one place
│
└── docs/
    ├── ARCHITECTURE.md          ← Deep dive on agent design
    ├── REAL_DATA_SOURCES.md     ← How to swap dummy data for real APIs
    └── PRODUCTION_CHECKLIST.md  ← Steps to deploy this for real
```

---

## Setup & Installation

### Step 1: Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd landscape-agent

# Or just unzip the downloaded folder and open a terminal there
```

### Step 2: Set up Python environment

```bash
# Create a virtual environment (keeps dependencies isolated)
python -m venv venv

# Activate it:
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure your API key

```bash
# Copy the example env file
cp .env.example .env

# Open .env in any text editor and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### Step 4: (Optional) Set up ChromaDB for real RAG

```bash
# Load the knowledge documents into the local vector database
python src/data/seed_vector_db.py

# You'll see output like:
# Loading 12 documents...
# Generating embeddings...
# Stored in ChromaDB at ./chroma_db/
```

### Step 5: Run the app

```bash
python src/main.py
```

You'll see a terminal chat interface. Type any landscaping question and watch the agents work.

---

## Getting Your API Key

1. Go to **https://console.anthropic.com**
2. Sign up or log in
3. Navigate to **API Keys** in the left sidebar
4. Click **Create Key**
5. Copy the key (starts with `sk-ant-`) and paste it into your `.env` file

> ⚠️ Never commit your `.env` file to Git. The `.gitignore` already excludes it.

**Cost estimate for testing**: Running this project through ~50 test conversations costs approximately $0.10–$0.50 using Claude Sonnet. Very cheap for a learning project.

---

## The Agents Explained

### 1. Master Orchestrator (`agents/orchestrator.py`)

**What it does**: Receives every user message and decides what to do with it. It never answers the user — it only classifies and routes.

**How it works**: We send the user's message to Claude with a very specific system prompt that forces it to return structured JSON. That JSON tells us:
- What the user wants (quote? schedule? procedure?)
- Which services are involved
- What parameters to extract (lawn size, preferred date, etc.)

**Why JSON?** Because we need to call tools with specific parameters. Free-form text can't drive tool calls. This is called "structured output" and is a core pattern in AI engineering.

**In production**: LangChain's `JsonOutputParser` or Pydantic models handle this automatically and add validation.

```python
# Simplified version of what the orchestrator does:
routing = call_claude(
    system=ORCHESTRATOR_SYSTEM_PROMPT,  # Forces JSON output
    user=user_message
)
# routing = {"intent": ["OPERATIONS"], "services": ["patio_install"], "size_factor": 3.4}
```

---

### 2. Operations Agent (`agents/operations_agent.py`)

**What it does**: Handles all business operations — quoting, scheduling, and invoicing.

**How it works**: After the orchestrator extracts parameters, the operations agent:
1. Calls the relevant tools (estimate_job_cost, check_crew_availability, generate_invoice)
2. Receives the tool results as structured data
3. Makes one final Claude API call to synthesize a professional response

**Key insight**: The operations agent's system prompt tells Claude it's a professional estimator. It receives the tool outputs as context and writes a customer-ready proposal. The LLM never does math — the tools do math, the LLM writes prose.

---

### 3. Site Analysis Agent (`agents/site_analysis_agent.py`)

**What it does**: Analyzes descriptions of yards, lawns, or trees and recommends services.

**In this tutorial**: Works from text descriptions ("my oak has yellowing leaves").

**In production**: You can pass actual photos to Claude's vision API. The agent receives base64-encoded images alongside the text prompt.

```python
# Vision API example (production upgrade):
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
        {"type": "text", "text": "Analyze this yard and recommend landscaping services."}
    ]
}]
```

---

### 4. Knowledge Agent / RAG (`agents/knowledge_agent.py`)

**What it does**: Looks up relevant procedures, best practices, and service documentation before answering.

**Why RAG?** Without RAG, Claude answers from general knowledge — which may be wrong for your specific pricing, your specific service area, or your company's procedures. RAG "grounds" the response in your actual documents.

**How it works** (simplified):
1. User asks: *"How do I overseed a lawn in Ohio?"*
2. We convert the question to a vector embedding (a list of ~1500 numbers that represent the meaning)
3. We search a vector database for the most similar document
4. We inject that document into Claude's context
5. Claude answers based on YOUR document, not generic knowledge

**Vector database options**:
- **ChromaDB**: Free, runs locally, perfect for this tutorial
- **Pinecone**: Cloud-hosted, scales to millions of documents, what Granum would use
- **pgvector**: If you're already using PostgreSQL, adds vector search as an extension

---

## The Tools Explained

Tools are regular Python functions that agents can call. They do the deterministic work (math, database queries, API calls) that LLMs are bad at.

### `estimate_job_cost(service_type, size_factor)`

Calculates labor + material cost for a given service.

```python
# Input:
estimate_job_cost("lawn_aeration", size_factor=3.4)  # 3,400 sqft

# Output:
{
    "service": "lawn_aeration",
    "labor": 374,       # 2 workers × 2 hrs × 55/hr × size_factor
    "materials": 170,   # $40 base cost × size_factor × 1.25 markup
    "total": 544
}
```

**In production**: Replace the dummy pricing dict with a query to your LMN pricing tables or SingleOps service catalog API.

---

### `check_crew_availability(preferred_day)`

Returns which crew is available on a given day.

```python
# Input:
check_crew_availability("Tuesday")

# Output:
{"id": "B", "name": "Team Bravo", "size": 4, "available": ["Tue", "Wed", "Fri"]}
```

**In production**: Query SingleOps' crew scheduling API. They have REST endpoints for crew calendars.

---

### `generate_invoice(labor_hours, labor_rate, material_cost, job_desc)`

Builds an invoice with Ohio sales tax.

```python
# Input:
generate_invoice(labor_hours=6, labor_rate=55, material_cost=340, job_desc="Spring cleanup")

# Output:
{
    "labor": 330,
    "materials": 425,     # $340 × 1.25 markup
    "subtotal": 755,
    "tax": 54,            # 7.25% Ohio sales tax
    "total": 809
}
```

---

## RAG: Knowledge Retrieval

This is the most important concept in the project for the Granum role.

### What is RAG?

**Retrieval-Augmented Generation**: Instead of asking Claude to answer from memory, you:
1. Store your company's documents in a vector database
2. When a question comes in, find the most relevant documents
3. Include those documents in Claude's prompt
4. Claude answers based on YOUR documents

### Setting up ChromaDB (local, free)

```python
import chromadb
from anthropic import Anthropic

client = chromadb.Client()
collection = client.create_collection("landscape_docs")

# Add your documents (do this once, at startup)
collection.add(
    documents=["Core aeration should be done in fall (Ohio: Sept-Oct)...",
               "Patio installation requires excavation (4-6 inches)..."],
    ids=["doc1", "doc2"]
)

# Query at runtime
results = collection.query(
    query_texts=["when should I aerate my lawn"],
    n_results=2
)
# Returns the 2 most relevant documents
```

### Where to get real landscaping documents

See `docs/REAL_DATA_SOURCES.md` for a full list, but good starting points:
- **Ohio State Extension**: free turf management guides at ohioline.osu.edu
- **NALP (National Association of Landscape Professionals)**: industry best practices
- **Your own LMN/SingleOps data**: export service templates and historical job notes
- **Manufacturer guides**: fertilizer application rates, herbicide mixing instructions

---

## Running the App

### Basic terminal chat:
```bash
python src/main.py
```

### Run a single query non-interactively:
```bash
python src/main.py --query "Quote me a patio installation for a 400 sqft area"
```

### Run with verbose agent traces (shows all intermediate steps):
```bash
python src/main.py --verbose
```

---

## Example Conversations

### Example 1: Job Quote

```
You: Quote me a backyard patio and lawn aeration for a 3,400 sq ft yard

[Orchestrator] Classified: OPERATIONS
[Orchestrator] Services: patio_install, lawn_aeration | Size: 3.4
[Tool] estimate_job_cost(patio_install, 3.4) → $4,760
[Tool] estimate_job_cost(lawn_aeration, 3.4) → $544
[Tool] check_crew_availability(None) → Team Alpha

Agent: Here's your proposal for the Johnson property:

BACKYARD PATIO INSTALLATION
  Labor (2 crew × 16 hrs × $55): $1,760
  Materials (pavers, gravel, sand): $1,500
  ─────────────────────────────────
  Patio Subtotal: $3,260

LAWN AERATION (3,400 sq ft)
  Labor (2 crew × 2 hrs × $55): $220
  Materials (marking flags): $50
  ─────────────────────────────────
  Aeration Subtotal: $270

TOTAL ESTIMATE: $3,530
CREW: Team Alpha (3 members)
TIMELINE: 3 days
```

### Example 2: Procedure Question (RAG)

```
You: What are the steps for overseeding in Ohio?

[Orchestrator] Classified: KNOWLEDGE
[RAG] Query: "overseeding ohio steps"
[RAG] Retrieved: doc_k4 (similarity: 0.91)

Agent: Here's the Ohio overseeding procedure:

1. TIMING: Sept 1 – Oct 15 is your window. Don't wait past mid-October.
2. MOW: Cut existing lawn to 1.5 inches before seeding.
3. DETHATCH: If thatch layer exceeds 0.5 inches, dethatch first.
4. FERTILIZE: Apply starter fertilizer 10-10-10 at time of seeding.
5. SEED: Use a slit-seeder for best soil contact.
6. WATER: Keep seedbed moist — light watering 2x daily for 3–4 weeks.
```

### Example 3: Tree Health (Site Analysis)

```
You: Customer says their oak has yellowing leaves and some branch die-back

[Orchestrator] Classified: SITE_ANALYSIS
[Site Agent] Analyzing: "oak, yellowing leaves, die-back"
[RAG] Retrieved: doc_k3 (similarity: 0.89)

Agent: SITE ANALYSIS — Oak Health Concern

Observed symptoms:
  • Yellowing leaves (interveinal chlorosis pattern likely)
  • Branch die-back (potential vascular issue)

Likely diagnoses:
  1. Iron/manganese deficiency (chlorosis) — most common in Ohio's alkaline soils
  2. Verticillium wilt — fungal, causes branch die-back
  3. Oak wilt — serious, can kill tree within 1 season

Recommended services:
  ✓ Tree health inspection (URGENT — schedule within 1 week)
  ✓ Soil pH test ($45)
  ✓ Chelated iron injection if chlorosis confirmed

Urgency: SOON — monitor weekly, escalate if die-back spreads
```

---

## Connecting Real Data Sources

See `docs/REAL_DATA_SOURCES.md` for full details. Quick reference:

| Dummy Data | Real Replacement | How |
|---|---|---|
| `dummy_db.json` crews | SingleOps Crew API | `GET /api/v1/crews` |
| `dummy_db.json` pricing | LMN Service Templates | Export CSV or REST API |
| `knowledge_docs.json` | Your service manuals + OSU Extension PDFs | PDF → text → embed → Chroma |
| Ohio tax rate hardcoded | Avalara / TaxJar API | One API call per invoice |
| Fake job IDs | SingleOps Jobs API | `GET /api/v1/jobs` |

---

## Production Upgrade Path

This project is structured so each piece can be upgraded independently:

1. **Vector DB**: Swap ChromaDB → Pinecone (change 3 lines in `knowledge_agent.py`)
2. **Orchestration framework**: Wrap tools in LangChain `@tool` decorators → get automatic retries, parallel execution, and tracing for free
3. **Analytics**: Route completed jobs to Snowflake Cortex for ML-powered demand forecasting (mentioned in Granum JD)
4. **Feature flags**: Wrap agent routing in LaunchDarkly AI Configs to A/B test prompts in production (mentioned in Granum JD)
5. **Vision**: Pass yard photos to Claude's vision API in `site_analysis_agent.py`
6. **.NET version**: The `Semantic Kernel` library is Microsoft's equivalent of LangChain for C# — same patterns apply

See `docs/PRODUCTION_CHECKLIST.md` for the full upgrade sequence.

---

## Alignment to Granum Job Description

| JD Requirement | Where it appears in this project |
|---|---|
| AI agents / agentic workflows | `agents/orchestrator.py` — fan-out/merge pattern |
| LLM orchestration (LangChain/LlamaIndex) | `utils/claude_client.py` — upgrade path documented |
| RAG + vector databases | `agents/knowledge_agent.py` + ChromaDB |
| Embeddings + Pinecone/Chroma | `data/seed_vector_db.py` |
| .NET/C# and Python | Both versions included in `src/` |
| Snowflake Cortex | Documented in `docs/PRODUCTION_CHECKLIST.md` |
| LaunchDarkly AI Configs | Documented in `docs/PRODUCTION_CHECKLIST.md` |
| From ideation to production | Full lifecycle from dummy data → production checklist |
| Mentoring cross-functional peers | Tutorial comments throughout every file |
| Customer outcomes over PRs | Every feature tied to a landscaping use case |

---

*Built as a portfolio project for the Granum AI Engineer role. All company names and pricing are illustrative.*

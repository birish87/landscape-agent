"""
tools/estimator.py, scheduler.py, invoicer.py (combined for simplicity)
========================================================================
These are the "tools" — deterministic functions that agents call.

THE GOLDEN RULE OF AI TOOL DESIGN:
  Tools do computation. LLMs do language.

Never ask Claude to calculate $55 × 16 hours. It will get it wrong
sometimes. Instead, write a Python function that does the math
perfectly, and give Claude the result.

In LangChain, tools are decorated with @tool:
    from langchain.tools import tool
    @tool
    def estimate_job_cost(service_type: str, size_factor: float) -> dict:
        ...

In Semantic Kernel (.NET/C#), tools are [KernelFunction] methods:
    [KernelFunction]
    public JobEstimate EstimateJobCost(string serviceType, float sizeFactor) { ... }

In this tutorial, they're plain Python functions.
The agents call them directly (no framework needed to learn the concept).
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

# Load the dummy database
DB_PATH = Path(__file__).parent.parent / "data" / "dummy_db.json"
with open(DB_PATH) as f:
    DB = json.load(f)


# ==============================================================================
# TOOL 1: estimate_job_cost
# ==============================================================================
def estimate_job_cost(service_type: str, size_factor: float = 1.0) -> dict | None:
    """
    Calculate the total cost for a landscaping service.

    HOW IT WORKS:
    1. Look up the service in the pricing table
    2. Calculate labor: base_hours × size_factor × min_crew × hourly_rate
    3. Calculate materials: material_cost × size_factor × markup
    4. Return a structured dict with the breakdown

    REAL-WORLD REPLACEMENT:
    Instead of reading from dummy_db.json, call your LMN or SingleOps
    pricing API:
        GET /api/v1/services/{service_type}/pricing?sqft={sqft}

    Args:
        service_type: e.g., "lawn_aeration", "patio_install"
        size_factor:  Multiplier — use sqft/1000 for area-based services,
                      1.0 for flat-rate jobs, or number of trees/units

    Returns:
        dict with keys: service, labor, materials, total, breakdown
        None if service_type not found in pricing table
    """
    pricing = DB["pricing"]["services"]

    if service_type not in pricing:
        print(f"  [Tool Warning] Unknown service: {service_type}")
        return None

    svc = pricing[service_type]
    hourly_rate = DB["crews"][0]["hourly_rate_per_member"]  # Default rate
    markup = DB["pricing"]["material_markup"]
    min_crew = svc["min_crew"]

    # Labor cost = hours × size_factor × crew_size × hourly_rate
    labor_hours = svc["base_hours"] * size_factor * min_crew
    labor_cost = round(labor_hours * hourly_rate)

    # Material cost = base_material_cost × size_factor × markup
    material_cost = round(svc["material_cost"] * size_factor * markup)

    total = labor_cost + material_cost

    result = {
        "service": service_type,
        "labor": labor_cost,
        "materials": material_cost,
        "total": total,
        "breakdown": {
            "crew_size": min_crew,
            "hours_per_member": round(svc["base_hours"] * size_factor, 1),
            "hourly_rate": hourly_rate,
            "unit": svc["unit"],
            "size_factor": size_factor
        }
    }

    print(f"  [Tool] estimate_job_cost({service_type}, {size_factor}) → ${total}")
    return result


# ==============================================================================
# TOOL 2: check_crew_availability
# ==============================================================================
def check_crew_availability(preferred_day: str = None, required_specialty: str = None) -> dict:
    """
    Find an available crew for a given day and/or specialty.

    REAL-WORLD REPLACEMENT:
    Query SingleOps crew scheduling API:
        GET /api/v1/crews/available?date=2025-04-15&specialty=hardscape

    Args:
        preferred_day:      e.g., "Tuesday", "Mon", or None for any day
        required_specialty: e.g., "hardscape" to find a patio crew

    Returns:
        dict with crew details and next available date
    """
    crews = DB["crews"]
    candidates = crews  # Start with all crews

    # Filter by specialty if requested
    if required_specialty:
        specialty_map = {
            "patio_install":  "hardscape",
            "retaining_wall": "hardscape",
            "tree_pruning":   "tree",
            "tree_removal":   "tree",
            "lawn_aeration":  "lawn_care",
            "overseeding":    "lawn_care",
        }
        needed = specialty_map.get(required_specialty, "lawn_care")
        candidates = [c for c in crews if any(needed in s for s in c["specialties"])]
        if not candidates:
            candidates = crews  # Fallback to all crews

    # Filter by day if provided
    if preferred_day:
        day_abbrev = preferred_day[:3].capitalize()
        day_matched = [c for c in candidates if day_abbrev in c["available"]]
        if day_matched:
            candidates = day_matched

    # Pick the first matching crew
    crew = candidates[0] if candidates else crews[0]

    # Calculate next available date (simplified: just return the day name)
    # In production, this would query the actual calendar for specific dates
    next_date = _next_occurrence_of_day(crew["available"][0])

    result = {
        **crew,
        "next_available_date": next_date,
        "note": f"Specializes in: {', '.join(crew['specialties'][:2])}"
    }

    print(f"  [Tool] check_crew_availability({preferred_day}) → {crew['name']}")
    return result


def _next_occurrence_of_day(day_abbrev: str) -> str:
    """Helper: find the next calendar date for a given day abbreviation."""
    days = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    target = days.get(day_abbrev, 0)
    today = datetime.now()
    days_ahead = target - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_date = today + timedelta(days=days_ahead)
    return next_date.strftime("%B %d, %Y")


# ==============================================================================
# TOOL 3: generate_invoice
# ==============================================================================
def generate_invoice(
    labor_hours: float,
    labor_rate: float,
    material_cost: float,
    job_description: str,
    customer_name: str = "Valued Customer"
) -> dict:
    """
    Generate an invoice with Ohio sales tax.

    REAL-WORLD REPLACEMENT:
    POST to your billing microservice or SingleOps invoicing API:
        POST /api/v1/invoices
        Body: { labor_hours, labor_rate, material_cost, job_id, customer_id }

    Or integrate Avalara/TaxJar for automated tax calculation across
    multiple states (important if Granum operates nationally).

    Args:
        labor_hours:      Number of labor hours
        labor_rate:       Hourly rate in USD
        material_cost:    Raw material cost (markup applied inside this function)
        job_description:  Short description of work performed
        customer_name:    Customer name for the invoice header

    Returns:
        dict with full invoice breakdown including tax and total due
    """
    markup = DB["pricing"]["material_markup"]
    tax_rate = DB["pricing"]["tax_rate"]  # Ohio: 7.25%

    labor_total = round(labor_hours * labor_rate)
    materials_with_markup = round(material_cost * markup)
    subtotal = labor_total + materials_with_markup
    tax = round(subtotal * tax_rate)
    total_due = subtotal + tax

    # Generate invoice number (in production, use a sequence from your DB)
    invoice_num = f"INV-{datetime.now().strftime('%Y')}-{datetime.now().strftime('%m%d%H%M')}"

    result = {
        "invoice_number": invoice_num,
        "customer": customer_name,
        "job_description": job_description,
        "date": datetime.now().strftime("%B %d, %Y"),
        "line_items": {
            "labor": {"hours": labor_hours, "rate": labor_rate, "total": labor_total},
            "materials": {"cost": material_cost, "markup": f"{int((markup-1)*100)}%", "total": materials_with_markup}
        },
        "subtotal": subtotal,
        "tax": {"rate": f"{tax_rate*100:.2f}%", "amount": tax},
        "total_due": total_due,
        "payment_terms": "Due on receipt (residential) / Net-30 (commercial)",
        "accepted_payment": "ACH (preferred), Check, Credit Card (+2.9% surcharge)"
    }

    print(f"  [Tool] generate_invoice({labor_hours}hrs, ${labor_rate}/hr, ${material_cost} materials) → ${total_due} total")
    return result


# ==============================================================================
# TOOL 4: retrieve_knowledge (simple RAG without vector DB)
# ==============================================================================
def retrieve_knowledge_simple(query: str) -> str:
    """
    Simple keyword-based knowledge retrieval (no vector DB required).

    This is the "starter" version — good for learning. The real version
    (in seed_vector_db.py) uses ChromaDB with embeddings for semantic search.

    HOW SIMPLE RAG WORKS:
    1. Convert query to lowercase keywords
    2. Find the knowledge doc with the most keyword matches
    3. Return that document's content for Claude to use

    WHY THIS IS LIMITED:
    - "When should I cut grass?" won't match "lawn_aeration" (no overlap)
    - Semantic search (embeddings) handles synonyms and related concepts
    - Use ChromaDB version for anything beyond basic demos

    Args:
        query: The user's question in natural language

    Returns:
        The content of the best-matching knowledge document,
        or a fallback message if nothing matches.
    """
    docs_path = Path(__file__).parent.parent / "data" / "knowledge_docs.json"
    with open(docs_path) as f:
        docs = json.load(f)

    query_words = set(query.lower().split())
    best_doc = None
    best_score = 0

    for doc in docs:
        # Score based on keyword overlap with topic, title, and content
        doc_words = set((doc["topic"] + " " + doc["title"] + " " + doc["content"]).lower().split())
        score = len(query_words & doc_words)

        # Boost score for topic keyword matches (more specific)
        topic_words = set(doc["topic"].replace("_", " ").split())
        score += len(query_words & topic_words) * 3

        if score > best_score:
            best_score = score
            best_doc = doc

    if best_doc and best_score >= 2:
        print(f"  [Tool] retrieve_knowledge('{query[:30]}...') → {best_doc['id']}: {best_doc['title']} (score={best_score})")
        return f"[Source: {best_doc['source']}]\n\n{best_doc['content']}"

    print(f"  [Tool] retrieve_knowledge('{query[:30]}...') → no strong match (score={best_score})")
    return f"No specific document found for: {query}. Please answer from general landscaping knowledge."

def retrieve_knowledge_semantic(query: str, top_k: int = 1) -> str:
    """
    Production RAG: semantic vector search using ChromaDB.

    SETUP REQUIRED before this works:
        python src/data/seed_vector_db.py

    Falls back to keyword search if ChromaDB is not set up.
    """
    chroma_db_path = str(Path(__file__).parent.parent.parent / "chroma_db")

    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_db_path)
        collection = client.get_collection("landscape_knowledge")

        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        docs_content = []
        for doc_content, metadata in zip(results["documents"][0], results["metadatas"][0]):
            source = metadata.get("source", "Knowledge base")
            title = metadata.get("title", "")
            docs_content.append(f"[Source: {source} — {title}]\n{doc_content}")
            print(f"  [Tool] retrieve_knowledge_semantic('{query[:30]}...') → '{title}'")

        return "\n\n---\n\n".join(docs_content)

    except Exception as e:
        print(f"  [Tool] ChromaDB not available ({e}). Falling back to keyword search.")
        return retrieve_knowledge_simple(query)


def retrieve_knowledge(query: str) -> str:
    """
    Smart dispatcher: automatically uses semantic search if ChromaDB is
    seeded, otherwise falls back to keyword search.

    THIS is the only function the rest of the code should call.
    Run seed_vector_db.py once to upgrade to semantic search automatically.
    """
    chroma_db_path = Path(__file__).parent.parent.parent / "chroma_db"

    if chroma_db_path.exists():
        return retrieve_knowledge_semantic(query)
    else:
        print("  [Tool] Tip: run seed_vector_db.py to enable semantic search")
        return retrieve_knowledge_simple(query)

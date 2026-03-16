"""
agents/operations_agent.py
==========================
The Operations Agent — handles quotes, scheduling, and invoicing.

ROLE: Run business-logic tools and synthesize a professional response.

PATTERN: "Tool-augmented generation"
Step 1: Call deterministic tools to get accurate numbers
Step 2: Pass tool results to Claude as context
Step 3: Claude writes professional prose — but NEVER does the math itself

WHY THIS MATTERS:
LLMs are probabilistic. $55 × 3.4 × 2 = $374 might come out as $370
or $380 from Claude if you ask it to calculate. That's unacceptable for
quotes and invoices. Always run math in Python, pass results to Claude.

This is the "tool use" pattern that the Granum JD refers to when
it mentions "orchestration frameworks such as LangChain or LlamaIndex."
Both frameworks automate this loop with @tool decorators and built-in
retry/validation logic.
"""

from utils.claude_client import call_claude
from utils.prompts import OPERATIONS_SYSTEM
from tools.tools import estimate_job_cost, check_crew_availability, generate_invoice


def run_operations_agent(routing: dict, knowledge_context: str = "", verbose: bool = False) -> str:
    """
    Run all relevant operations tools based on the routing decision,
    then synthesize a response using Claude.

    Args:
        routing:           The routing dict from the orchestrator
        knowledge_context: Any relevant docs retrieved by the knowledge agent
        verbose:           If True, print tool call details

    Returns:
        A professional response string ready to show the user
    """
    tool_results = []

    # ------------------------------------------------------------------
    # TOOL CALLS: Run in sequence here; in production use asyncio/ThreadPool
    # for parallel execution when multiple tools are needed.
    #
    # Example (production):
    #   import asyncio
    #   tasks = [estimate_cost(s) for s in routing["services"]]
    #   results = await asyncio.gather(*tasks)
    # ------------------------------------------------------------------

    # 1. Cost estimation — run for each detected service
    services = routing.get("services", [])
    size_factor = routing.get("size_factor") or 1.0
    estimates = []

    for service in services:
        estimate = estimate_job_cost(service, size_factor)
        if estimate:
            estimates.append(estimate)

    if estimates:
        total = sum(e["total"] for e in estimates)
        estimates_text = "\n".join([
            f"  {e['service']}: labor=${e['labor']}, materials=${e['materials']}, subtotal=${e['total']}"
            for e in estimates
        ])
        tool_results.append(f"COST ESTIMATES:\n{estimates_text}\nCOMBINED TOTAL: ${total}")

    # 2. Crew availability — check if we have scheduling context
    # We check if a day was mentioned OR if services were found (quote implies scheduling intent)
    if routing.get("preferred_day") is not None or services:
        primary_service = services[0] if services else None
        crew = check_crew_availability(
            preferred_day=routing.get("preferred_day"),
            required_specialty=primary_service
        )
        tool_results.append(
            f"CREW ASSIGNMENT:\n"
            f"  Name: {crew['name']}\n"
            f"  Size: {crew['size']} members\n"
            f"  Next available: {crew['next_available_date']}\n"
            f"  Specialties: {crew['note']}"
        )

    # 3. Invoice generation — only if the user provided explicit hours/materials
    if routing.get("labor_hours") and routing.get("material_cost"):
        invoice = generate_invoice(
            labor_hours=routing["labor_hours"],
            labor_rate=routing.get("labor_rate") or 55,
            material_cost=routing["material_cost"],
            job_description=routing.get("knowledge_query") or "Landscaping service"
        )
        tool_results.append(
            f"INVOICE GENERATED:\n"
            f"  Invoice #: {invoice['invoice_number']}\n"
            f"  Labor: ${invoice['line_items']['labor']['total']}\n"
            f"  Materials (w/ markup): ${invoice['line_items']['materials']['total']}\n"
            f"  Subtotal: ${invoice['subtotal']}\n"
            f"  Ohio Tax (7.25%): ${invoice['tax']['amount']}\n"
            f"  TOTAL DUE: ${invoice['total_due']}\n"
            f"  Terms: {invoice['payment_terms']}"
        )

    # ------------------------------------------------------------------
    # SYNTHESIS: Pass all tool results to Claude to write the response
    # The context block tells Claude exactly what data it has to work with.
    # ------------------------------------------------------------------
    context_block = "\n\n".join(tool_results)
    if not context_block:
        context_block = "No specific tool data available — answer generally."

    synthesis_input = (
        f"USER REQUEST: {routing.get('_original_message', 'Landscaping inquiry')}\n\n"
        f"TOOL RESULTS:\n{context_block}"
        + (f"\n\nKNOWLEDGE CONTEXT:\n{knowledge_context}" if knowledge_context else "")
    )

    if verbose:
        print(f"\n[Operations Agent] Synthesizing with context:\n{synthesis_input[:300]}...")

    response = call_claude(
        system_prompt=OPERATIONS_SYSTEM,
        user_message=synthesis_input,
        verbose=verbose
    )

    return response

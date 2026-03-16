"""
agents/orchestrator.py
======================
The Master Orchestrator — the entry point for every user message.

ROLE: Classify intent and extract structured parameters.
      Never answers the user. Only routes.

PATTERN: "Structured Output Routing"
This is one of the most important patterns in AI engineering.
Instead of passing the user's message directly to a tool or a response
agent, we first make a cheap, fast classification call to understand
WHAT the user wants and extract the parameters we'll need.

Think of it like a receptionist who greets every caller, figures out
which department to transfer them to, and gathers the caller's name
and account number before transferring — so the specialist department
doesn't have to ask again.

IN PRODUCTION (LangChain):
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel

    class RoutingDecision(BaseModel):
        intent: list[str]
        services: list[str]
        size_factor: float | None
        # ... etc

    parser = PydanticOutputParser(pydantic_object=RoutingDecision)
    chain = prompt | llm | parser
    routing = chain.invoke({"user_message": user_message})
    # Automatic validation + retry on parse failure

IN PRODUCTION (.NET Semantic Kernel):
    var result = await kernel.InvokeAsync<RoutingDecision>(
        orchestratorPlugin, "ClassifyIntent",
        new() { ["userMessage"] = userMessage }
    );
"""

from utils.claude_client import call_claude_for_json
from utils.prompts import ORCHESTRATOR_SYSTEM


def classify_and_route(user_message: str, verbose: bool = False) -> dict:
    """
    Classify the user's message and extract routing parameters.

    Args:
        user_message: The raw message from the user
        verbose:      If True, print token usage

    Returns:
        A routing dict with keys: intent, services, size_factor,
        preferred_day, labor_hours, material_cost, site_description,
        knowledge_query
    """
    if verbose:
        print(f"\n[Orchestrator] Classifying: '{user_message[:60]}...'")

    routing = call_claude_for_json(
        system_prompt=ORCHESTRATOR_SYSTEM,
        user_message=user_message,
        verbose=verbose
    )

    if verbose:
        print(f"[Orchestrator] Routing decision: {routing}")

    return routing


def determine_primary_agent(routing: dict) -> str:
    """
    Given a routing decision, determine which agent should synthesize
    the final response.

    Priority order:
    1. OPERATIONS (if any tool was called, ops agent writes the response)
    2. SITE_ANALYSIS (if yard/tree analysis needed)
    3. KNOWLEDGE (fallback for how-to questions)

    Returns:
        "operations" | "site_analysis" | "knowledge"
    """
    intents = routing.get("intent", ["KNOWLEDGE"])

    if "OPERATIONS" in intents or routing.get("services"):
        return "operations"
    elif "SITE_ANALYSIS" in intents:
        return "site_analysis"
    else:
        return "knowledge"

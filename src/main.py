"""
main.py
=======
Entry point for the Landscape Intelligence Platform.

This file ties all the agents together into the complete pipeline:
  User Message → Orchestrator → [Tools] → Specialist Agent → Response

Run with:
  python src/main.py                          # Interactive chat mode
  python src/main.py --verbose               # Show all agent traces
  python src/main.py --query "your question" # Single query mode

HOW THE PIPELINE FLOWS:
  1. User sends a message
  2. Orchestrator classifies intent and extracts parameters
  3. Knowledge Agent runs RAG to retrieve relevant docs (almost always)
  4. Based on intent:
     - OPERATIONS: run cost/scheduling/invoice tools → ops agent synthesizes
     - SITE_ANALYSIS: site agent analyzes with knowledge context
     - KNOWLEDGE: knowledge agent answers directly from retrieved docs
  5. Response is displayed with agent trace
"""

import sys
import os
import argparse

# Add src/ to the Python path so imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file (ANTHROPIC_API_KEY)
# This requires: pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If python-dotenv not installed, rely on environment variables

from agents.orchestrator import classify_and_route, determine_primary_agent
from agents.operations_agent import run_operations_agent
from agents.knowledge_agent import run_knowledge_agent, run_site_analysis_agent


def run_pipeline(user_message: str, verbose: bool = False) -> str:
    """
    The complete multi-agent pipeline.

    This is the core function you'd expose as an API endpoint in production.
    In a FastAPI/ASP.NET app, this becomes:
        POST /api/v1/agent/chat  { "message": "..." }

    Args:
        user_message: The raw message from the user
        verbose:      If True, print all intermediate agent steps

    Returns:
        The final response string to show the user
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # STAGE 1: ORCHESTRATOR — Classify and extract parameters
    # Always runs first. Fast and cheap (uses short output).
    # ----------------------------------------------------------------
    print("\n[Stage 1] Orchestrator classifying...")
    routing = classify_and_route(user_message, verbose=verbose)
    routing["_original_message"] = user_message  # Carry original text through pipeline

    intents = routing.get("intent", ["KNOWLEDGE"])
    print(f"  → Intent: {intents}")
    print(f"  → Services: {routing.get('services', [])}")
    if routing.get("size_factor"):
        print(f"  → Size factor: {routing['size_factor']}")

    # ----------------------------------------------------------------
    # STAGE 2: KNOWLEDGE AGENT — RAG retrieval
    # Almost always runs, to ground the response in real documents.
    # Even operations responses benefit from procedural context.
    # ----------------------------------------------------------------
    knowledge_context = ""
    knowledge_query = routing.get("knowledge_query") or user_message

    # Run RAG if we have a knowledge query or non-pure-operations intent
    if knowledge_query or "KNOWLEDGE" in intents or "SITE_ANALYSIS" in intents:
        print(f"\n[Stage 2] Knowledge Agent retrieving docs...")
        _, knowledge_context = run_knowledge_agent(knowledge_query, verbose=verbose)

    # ----------------------------------------------------------------
    # STAGE 3: SPECIALIST AGENTS — Based on routing decision
    # ----------------------------------------------------------------
    primary_agent = determine_primary_agent(routing)
    print(f"\n[Stage 3] Running primary agent: {primary_agent}")

    if primary_agent == "operations":
        # Run cost/scheduling/invoice tools + synthesize
        response = run_operations_agent(routing, knowledge_context, verbose=verbose)

    elif primary_agent == "site_analysis":
        # Analyze yard/plant condition
        site_desc = routing.get("site_description") or user_message
        response = run_site_analysis_agent(site_desc, knowledge_context, verbose=verbose)

    else:
        # Pure knowledge/procedure question
        response, _ = run_knowledge_agent(user_message, verbose=verbose)

    print(f"\n{'='*60}")
    print(f"AGENT RESPONSE:\n")
    print(response)
    print(f"{'='*60}\n")

    return response


def interactive_chat(verbose: bool = False):
    """
    Run an interactive terminal chat session.
    Type 'quit' or 'exit' to end the session.
    """
    print("\n" + "🌿 " * 20)
    print("  Granum Landscape Intelligence Platform")
    print("  Multi-Agent AI Tutorial Project")
    print("🌿 " * 20)
    print("\nType your landscaping question below.")
    print("Try: 'Quote a patio install for 400 sqft'")
    print("Or:  'When should I aerate my lawn in Ohio?'")
    print("Or:  'My oak tree has yellowing leaves — what's wrong?'")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 🌿")
                break

            run_pipeline(user_input, verbose=verbose)

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            if verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # CLI argument parsing
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Granum Landscape Intelligence Platform — Multi-Agent AI"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query non-interactively"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed agent traces including token usage"
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ERROR: ANTHROPIC_API_KEY not set.")
        print("1. Copy .env.example to .env")
        print("2. Add your API key: ANTHROPIC_API_KEY=sk-ant-...")
        print("3. Run again\n")
        sys.exit(1)

    if args.query:
        # Single query mode
        run_pipeline(args.query, verbose=args.verbose)
    else:
        # Interactive mode
        interactive_chat(verbose=args.verbose)

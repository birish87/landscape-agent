"""
agents/knowledge_agent.py
=========================
The Knowledge Agent — answers procedure and how-to questions using RAG.

PATTERN: Retrieval-Augmented Generation (RAG)
1. Take the user's question
2. Find the most relevant document(s) from the knowledge base
3. Inject those documents into Claude's prompt
4. Claude answers based on YOUR documents, not generic training data

WHY THIS MATTERS FOR GRANUM:
Granum's knowledge base would contain LMN service templates,
SingleOps job procedures, regional best practices for the green industry,
and company-specific pricing. Without RAG, Claude answers from generic
internet knowledge. With RAG, it answers from Granum's actual data.

PRODUCTION UPGRADE (uses real embeddings + ChromaDB):
See src/data/seed_vector_db.py for setup.
Then replace retrieve_knowledge_simple() with retrieve_knowledge_semantic().
"""

from utils.claude_client import call_claude
from utils.prompts import KNOWLEDGE_SYSTEM, SITE_ANALYSIS_SYSTEM
from tools.tools import retrieve_knowledge


def run_knowledge_agent(query: str, verbose: bool = False) -> tuple[str, str]:
    """
    Retrieve relevant knowledge and synthesize a response.

    Args:
        query:   The user's question
        verbose: If True, print retrieval details

    Returns:
        Tuple of (response_text, retrieved_context)
        The retrieved_context can be passed to other agents for grounding.
    """
    if verbose:
        print(f"\n[Knowledge Agent] Query: '{query}'")

    # Step 1: Retrieve relevant document (RAG)
    # PRODUCTION: Replace with semantic search via ChromaDB or Pinecone
    retrieved_doc = retrieve_knowledge(query)

    # Step 2: Synthesize response grounded in the retrieved document
    synthesis_input = (
        f"USER QUESTION: {query}\n\n"
        f"RETRIEVED KNOWLEDGE BASE DOCUMENT:\n{retrieved_doc}"
    )

    response = call_claude(
        system_prompt=KNOWLEDGE_SYSTEM,
        user_message=synthesis_input,
        verbose=verbose
    )

    return response, retrieved_doc


# ==============================================================================
# agents/site_analysis_agent.py (included here for simplicity)
# ==============================================================================
"""
The Site Analysis Agent — diagnoses yard/plant conditions.

PATTERN: "Contextual analysis with knowledge grounding"
Combines Claude's general botanical/horticultural knowledge with
retrieved company-specific service recommendations.

PRODUCTION UPGRADE (vision API):
Replace the text description with an image for computer vision analysis:

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_encoded_image
                }
            },
            {"type": "text", "text": "Analyze this yard and recommend services."}
        ]
    }]

This is the "Landscape Site Analysis Agent" from your proposal document.
It transforms a photo into a service recommendation and quote trigger.
"""


def run_site_analysis_agent(description: str, knowledge_context: str = "", verbose: bool = False) -> str:
    """
    Analyze a yard or plant condition description and recommend services.

    Args:
        description:       Text description of the yard/plant condition
        knowledge_context: Relevant knowledge doc (e.g., disease diagnosis guide)
        verbose:           If True, print debug info

    Returns:
        A structured analysis with diagnosis and service recommendations
    """
    if verbose:
        print(f"\n[Site Analysis Agent] Analyzing: '{description[:60]}...'")

    synthesis_input = (
        f"SITE DESCRIPTION: {description}"
        + (f"\n\nRELEVANT KNOWLEDGE:\n{knowledge_context}" if knowledge_context else "")
    )

    response = call_claude(
        system_prompt=SITE_ANALYSIS_SYSTEM,
        user_message=synthesis_input,
        verbose=verbose
    )

    return response

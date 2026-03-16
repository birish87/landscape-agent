"""
utils/claude_client.py
======================
A thin wrapper around the Anthropic SDK.

WHY WRAP THE SDK?
- Centralizes retry logic (API calls fail sometimes — always retry)
- Makes it easy to swap models in one place
- Adds logging hooks for production observability
- Lets you mock the client in unit tests

PRODUCTION ADDITIONS you'd add here:
- Structured logging (send to Datadog / CloudWatch)
- Cost tracking (log input/output tokens per call)
- Rate limit handling (exponential backoff)
- Response caching (same query → same answer, skip API call)
"""

import os
import time
import json
from anthropic import Anthropic, APIError, RateLimitError

# Load API key from environment variable (set in .env file)
# NEVER hardcode API keys in source code
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# The model to use for all calls.
# claude-sonnet-4-20250514 is the sweet spot of speed + intelligence.
# For production, consider claude-opus-4-20250514 for the orchestrator
# and claude-haiku-4-5-20251001 for simple classification tasks (much cheaper).
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def call_claude(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    max_retries: int = 3,
    verbose: bool = False
) -> str:
    """
    Make a single Claude API call and return the text response.

    Args:
        system_prompt: The system prompt defining the agent's role
        user_message:  The user's message (may include injected context)
        model:         Which Claude model to use
        max_tokens:    Maximum response length (1000 is plenty for most cases)
        max_retries:   How many times to retry on failure
        verbose:       If True, print token usage for cost tracking

    Returns:
        The text content of Claude's response as a string.

    Raises:
        RuntimeError: If all retries are exhausted
    """
    # Retry loop — API calls fail occasionally, always retry
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            # Log token usage in verbose mode
            # In production, log this to your cost-tracking system
            if verbose:
                usage = response.usage
                print(f"  [Token usage] input={usage.input_tokens}, output={usage.output_tokens}")

            # Extract text from the response content blocks
            # Claude can return multiple content blocks (text, tool_use, etc.)
            # We want the first text block
            for block in response.content:
                if block.type == "text":
                    return block.text

            return ""  # Should never happen, but safe fallback

        except RateLimitError:
            # Hit the API rate limit — wait and retry
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            print(f"  [Rate limit] Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)

        except APIError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Claude API failed after {max_retries} attempts: {e}")
            time.sleep(1)

    raise RuntimeError(f"Claude API failed after {max_retries} attempts")


def call_claude_for_json(
    system_prompt: str,
    user_message: str,
    verbose: bool = False
) -> dict:
    """
    Call Claude and parse the response as JSON.

    Used by the orchestrator, which must return structured routing data.

    IMPORTANT: The system prompt must strongly instruct Claude to return
    ONLY JSON with no preamble or markdown fences. Even so, Claude
    sometimes wraps JSON in ```json ... ``` — we strip that below.

    In production, use the `instructor` library (pip install instructor)
    which wraps the Anthropic client and validates JSON against a
    Pydantic schema automatically, with automatic retries on parse failure.
    """
    raw_text = call_claude(system_prompt, user_message, verbose=verbose)

    # Strip markdown code fences if present (```json ... ```)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = cleaned.split("\n", 1)[-1]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]

    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return a safe fallback
        # In production, log this as an error and alert your team
        print(f"  [Warning] JSON parse failed: {e}. Raw: {raw_text[:200]}")
        return {
            "intent": ["KNOWLEDGE"],
            "services": [],
            "knowledge_query": user_message,
            "site_description": None,
            "size_factor": None,
            "preferred_day": None,
            "labor_hours": None,
            "labor_rate": None,
            "material_cost": None
        }

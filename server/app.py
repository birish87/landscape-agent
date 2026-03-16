"""
server/app.py
=============
FastAPI web server — exposes the agent pipeline as HTTP endpoints
and serves the frontend HTML file.

WHY FASTAPI?
- Async support (important for LLM calls which are slow I/O)
- Built-in request validation with Pydantic
- Auto-generated API docs at /docs (great for team collaboration)
- Easy file upload handling via UploadFile

RUN WITH:
    pip install -r requirements.txt
    uvicorn server.app:app --reload --port 8000

Then open: http://localhost:8000

ENDPOINTS:
    GET  /              → serves the frontend UI
    POST /chat          → text-only agent pipeline
    POST /analyze-site  → image/video upload + site analysis
    GET  /health        → simple health check
"""

import os
import sys
import base64
import mimetypes
from pathlib import Path
from typing import Optional

# Add src/ to path so agent imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import our agent pipeline
from agents.orchestrator import classify_and_route, determine_primary_agent
from agents.operations_agent import run_operations_agent
from agents.knowledge_agent import run_knowledge_agent, run_site_analysis_agent
from utils.claude_client import call_claude
from utils.prompts import SITE_ANALYSIS_SYSTEM

app = FastAPI(title="Granum Landscape Intelligence Platform", version="1.0.0")

# Allow the frontend to call the API (needed if you ever host them separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    services_detected: list[str]


# ──────────────────────────────────────────────
# SERVE FRONTEND
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main UI."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# ENDPOINT 1: Text chat
# ──────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Run the full multi-agent pipeline on a text message.

    This is the same pipeline as main.py but exposed as an HTTP endpoint.
    The frontend calls this via fetch() when the user sends a message.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Stage 1: Orchestrator classifies and routes
        routing = classify_and_route(request.message)
        routing["_original_message"] = request.message

        # Stage 2: RAG knowledge retrieval
        knowledge_context = ""
        knowledge_query = routing.get("knowledge_query") or request.message
        if knowledge_query:
            _, knowledge_context = run_knowledge_agent(knowledge_query)

        # Stage 3: Specialist agent synthesizes response
        primary_agent = determine_primary_agent(routing)

        if primary_agent == "operations":
            response = run_operations_agent(routing, knowledge_context)
        elif primary_agent == "site_analysis":
            site_desc = routing.get("site_description") or request.message
            response = run_site_analysis_agent(site_desc, knowledge_context)
        else:
            response, _ = run_knowledge_agent(request.message)

        return ChatResponse(
            response=response,
            agent_used=primary_agent,
            services_detected=routing.get("services", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# ENDPOINT 2: Image / video upload + site analysis
# ──────────────────────────────────────────────

# Supported image types Claude vision API accepts
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Video is not directly supported by Claude — we extract a frame instead
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}

MAX_FILE_SIZE_MB = 20


@app.post("/analyze-site")
async def analyze_site(
    file: UploadFile = File(...),
    message: str = Form(default="Analyze this yard and recommend landscaping services.")
):
    """
    Accept an image (or video) upload and run site analysis.

    HOW IMAGE ANALYSIS WORKS WITH CLAUDE:
    1. Read the uploaded file as bytes
    2. Base64-encode it (converts binary to a string Claude can receive)
    3. Send it to Claude's messages API with type="image" in the content array
    4. Claude can see the image and describe what it observes

    SUPPORTED: JPEG, PNG, GIF, WebP images
    VIDEO: Extract first frame using OpenCV, analyze that as an image.
    """
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB."
        )

    mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or ""

    if mime_type in SUPPORTED_IMAGE_TYPES:
        analysis = await _analyze_image(file_bytes, mime_type, message)

    elif mime_type in SUPPORTED_VIDEO_TYPES:
        analysis = await _analyze_video(file_bytes, file.filename or "upload.mp4", message)

    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {mime_type}. Please upload JPEG, PNG, WebP, or MP4."
        )

    # RAG lookup based on the analysis to add relevant procedure context
    _, knowledge_context = run_knowledge_agent(analysis[:200])

    return JSONResponse({
        "analysis": analysis,
        "knowledge_context": knowledge_context[:500] if knowledge_context else None,
        "file_type": "image" if mime_type in SUPPORTED_IMAGE_TYPES else "video",
        "filename": file.filename
    })


async def _analyze_image(file_bytes: bytes, mime_type: str, user_prompt: str) -> str:
    """
    Send an image to Claude's vision API and get a landscaping analysis.

    CLAUDE VISION API STRUCTURE:
    Instead of a plain string message, we send a list of content blocks.
    One block is the image (base64), one is the text prompt.
    Claude reads both together.
    """
    base64_image = base64.standard_b64encode(file_bytes).decode("utf-8")

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SITE_ANALYSIS_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    # BLOCK 1: The image
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_image
                        }
                    },
                    # BLOCK 2: The text prompt alongside the image
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )

    return response.content[0].text


async def _analyze_video(file_bytes: bytes, filename: str, user_prompt: str) -> str:
    """
    Handle video uploads by extracting the first frame with OpenCV
    and analyzing it as an image.

    Claude's API does not accept video files directly.
    For production video analysis use AWS Rekognition or
    Google Video Intelligence API instead.
    """
    try:
        import cv2
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        success, frame = cap.read()
        cap.release()
        os.unlink(tmp_path)

        if success:
            _, jpeg_bytes = cv2.imencode(".jpg", frame)
            return await _analyze_image(
                jpeg_bytes.tobytes(),
                "image/jpeg",
                f"[Extracted from video: {filename}] {user_prompt}"
            )
        else:
            raise ValueError("Could not read video frame")

    except ImportError:
        return (
            "Video analysis requires OpenCV (pip install opencv-python). "
            "For now, please upload a still photo (JPEG or PNG) of the yard."
        )
    except Exception as e:
        return f"Could not process video: {e}. Please upload a JPEG or PNG photo instead."


# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Confirms the server and API key are working."""
    return {
        "status": "ok",
        "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "chroma_db_seeded": Path("chroma_db").exists()
    }
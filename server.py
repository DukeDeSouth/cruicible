"""Cruicible web server — FastAPI wrapping ADK Runner with SSE streaming."""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from cruicible.agent import root_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cruicible", version="0.3.0")


def _apply_keys(request: Request):
    """Override env vars with per-request API keys from headers (if provided)."""
    gk = request.headers.get("x-google-key")
    pk = request.headers.get("x-phoenix-key")
    pe = request.headers.get("x-phoenix-endpoint")
    if gk:
        os.environ["GOOGLE_API_KEY"] = gk
    if pk:
        os.environ["PHOENIX_API_KEY"] = pk
    if pe:
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = pe

session_service = InMemorySessionService()

sessions: dict[str, dict] = {}
briefs_store: dict[str, dict] = {}
reverify_jobs: dict[str, dict] = {}

STATIC_DIR = Path(__file__).parent / "static"
WIKI_PATH = Path(__file__).parent / "data" / "wiki.json"


def _load_wiki() -> list[dict]:
    if WIKI_PATH.exists():
        try:
            return json.loads(WIKI_PATH.read_text())
        except Exception:
            pass
    return []


def _save_wiki(entries: list[dict]):
    WIKI_PATH.parent.mkdir(parents=True, exist_ok=True)
    WIKI_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def _add_to_wiki(query: str, brief: str, session_id: str):
    entries = _load_wiki()
    if any(e.get("session_id") == session_id for e in entries):
        return
    entries.insert(0, {
        "id": str(uuid.uuid4())[:8],
        "session_id": session_id,
        "query": query,
        "brief": brief,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _save_wiki(entries)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Cruicible — static/index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@app.get("/wiki", response_class=HTMLResponse)
async def wiki_page():
    html_path = STATIC_DIR / "wiki.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Wiki page not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@app.post("/api/research")
async def start_research(request: Request):
    _apply_keys(request)
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    session_id = str(uuid.uuid4())[:8]
    session = await session_service.create_session(
        app_name="cruicible", user_id="web", session_id=session_id
    )

    sessions[session_id] = {
        "events": [],
        "status": "running",
        "query": query,
    }

    asyncio.create_task(_run_agent(session_id, query))
    return {"session_id": session_id}


async def _run_agent(session_id: str, query: str):
    """Execute the agent and collect events."""
    runner = Runner(
        agent=root_agent,
        app_name="cruicible",
        session_service=session_service,
    )
    content = types.Content(role="user", parts=[types.Part(text=query)])

    try:
        event_count = 0
        async for event in runner.run_async(
            user_id="web", session_id=session_id, new_message=content
        ):
            event_count += 1
            try:
                event_data = _serialize_event(event)
                if event_data:
                    sessions[session_id]["events"].append(event_data)
                else:
                    author = getattr(event, "author", "?")
                    final = event.is_final_response() if hasattr(event, "is_final_response") else "?"
                    has_content = bool(getattr(event, "content", None))
                    if has_content:
                        content = event.content
                        parts = getattr(content, "parts", None) or []
                        content_type = type(content).__name__
                        content_attrs = [k for k in dir(content) if not k.startswith("_")][:15]
                        content_role = getattr(content, "role", "?")
                        direct_text = getattr(content, "text", None)
                        part_info = []
                        for p in parts:
                            pdict = {k: type(v).__name__ for k, v in (vars(p).items() if hasattr(p, "__dict__") else []) if v is not None}
                            text_val = getattr(p, "text", None)
                            thought_val = getattr(p, "thought", None)
                            part_info.append(f"text={repr(text_val)[:50]}, thought={thought_val}, attrs={list(pdict.keys())[:8]}")
                        logger.warning(
                            "Event #%d SKIPPED (author=%s, final=%s, parts=%d, content_type=%s, role=%s, "
                            "direct_text=%s, content_attrs=%s): %s",
                            event_count, author, final, len(parts), content_type, content_role,
                            repr(direct_text)[:80] if direct_text else "None",
                            content_attrs, "; ".join(part_info)
                        )
                    else:
                        logger.info("Event #%d skipped (author=%s, final=%s, no content)",
                                    event_count, author, final)
            except Exception as inner_err:
                logger.warning("Event serialization error (skipped): %s", inner_err)
    except Exception as e:
        err_str = str(e)
        if "was created in a different Context" in err_str:
            logger.debug("OTel context cleanup (non-fatal): %s", e)
        else:
            logger.error("Agent error in session %s: %s", session_id, e)
            sessions[session_id]["events"].append({
                "type": "error",
                "message": err_str,
            })
    finally:
        _capture_brief(session_id)
        sessions[session_id]["events"].append({"type": "done"})
        sessions[session_id]["status"] = "pending_review"


def _capture_brief(session_id: str):
    """Extract the final brief from session events and store for re-verification."""
    state = sessions.get(session_id)
    if not state:
        return
    brief_text = ""
    query = state.get("query", "")
    for ev in state.get("events", []):
        if ev.get("type") == "tool_call":
            for tool in ev.get("tools", []):
                if tool.get("name") == "save_pattern":
                    brief_text = tool.get("args", {}).get("final_brief", "")
        if ev.get("type") == "text" and ev.get("author") in ("cruicible", "agent"):
            if len(ev.get("text", "")) > len(brief_text):
                brief_text = ev["text"]
    if brief_text and len(brief_text) > 50:
        brief_id = session_id
        briefs_store[brief_id] = {
            "id": brief_id,
            "query": query,
            "brief": brief_text,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "pending_review",
            "claims": [],
            "reverify_results": [],
        }


def _serialize_event(event) -> dict | None:
    """Convert ADK Event to JSON-serializable dict."""
    data = {"type": "event"}

    try:
        fn_calls = event.get_function_calls()
        if fn_calls:
            calls = []
            for fc in fn_calls:
                name = getattr(fc, "name", "unknown")
                args = dict(getattr(fc, "args", {}))
                if name != "save_pattern":
                    for k, v in args.items():
                        if isinstance(v, str) and len(v) > 500:
                            args[k] = v[:500] + "..."
                calls.append({"name": name, "args": args})
            data["type"] = "tool_call"
            data["tools"] = calls
            return data
    except Exception:
        pass

    try:
        fn_responses = event.get_function_responses()
        if fn_responses:
            responses = []
            for fr in fn_responses:
                raw_resp = getattr(fr, "response", "")
                if isinstance(raw_resp, dict):
                    resp_text = json.dumps(raw_resp, ensure_ascii=False)
                else:
                    resp_text = str(raw_resp)
                limit = 5000 if getattr(fr, "name", "") == "self_evaluate" else 2000
                if len(resp_text) > limit:
                    resp_text = resp_text[:limit] + "..."
                responses.append({
                    "name": getattr(fr, "name", "unknown"),
                    "response": resp_text,
                })
            data["type"] = "tool_response"
            data["responses"] = responses
            return data
    except Exception:
        pass

    if hasattr(event, "content") and event.content:
        parts = getattr(event.content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                data["type"] = "text"
                data["text"] = text
                data["author"] = getattr(event, "author", "agent")
                return data
            thought = getattr(part, "thought", None)
            if thought:
                continue

        if not parts:
            logger.debug("Event has content but no parts")
        else:
            part_attrs = [
                {k: type(v).__name__ for k, v in vars(parts[0]).items() if v is not None}
                if hasattr(parts[0], "__dict__") else str(type(parts[0]))
            ]
            logger.info("Event parts have no text. Part attrs: %s", part_attrs)

    return None


@app.get("/api/stream/{session_id}")
async def stream_events(session_id: str):
    if session_id not in sessions:
        return JSONResponse({"error": "session not found"}, status_code=404)

    async def generate():
        sent = 0
        while True:
            state = sessions.get(session_id)
            if not state:
                break

            events = state["events"]
            while sent < len(events):
                yield f"data: {json.dumps(events[sent])}\n\n"
                sent += 1

            if state["status"] in ("complete", "pending_review", "rejected"):
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    state = sessions.get(session_id)
    if not state:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {
        "session_id": session_id,
        "status": state["status"],
        "query": state["query"],
        "event_count": len(state["events"]),
    }


@app.post("/api/approve/{session_id}")
async def approve_gate(session_id: str):
    state = sessions.get(session_id)
    if not state:
        return JSONResponse({"error": "session not found"}, status_code=404)
    state["approved"] = True
    state["status"] = "complete"
    if session_id in briefs_store:
        b = briefs_store[session_id]
        b["status"] = "approved"
        _add_to_wiki(b.get("query", ""), b.get("brief", ""), session_id)
    return {"approved": True}


@app.post("/api/reject/{session_id}")
async def reject_gate(session_id: str):
    state = sessions.get(session_id)
    if not state:
        return JSONResponse({"error": "session not found"}, status_code=404)
    state["status"] = "rejected"
    if session_id in briefs_store:
        briefs_store[session_id]["status"] = "rejected"
    return {"rejected": True}


@app.get("/api/briefs")
async def list_briefs():
    return list(briefs_store.values())


@app.get("/api/wiki")
async def list_wiki():
    return _load_wiki()


@app.get("/api/wiki/{entry_id}")
async def get_wiki_entry(entry_id: str):
    for e in _load_wiki():
        if e["id"] == entry_id:
            return e
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/reverify/{brief_id}")
async def reverify_brief(brief_id: str, request: Request):
    _apply_keys(request)
    brief = briefs_store.get(brief_id)
    if not brief:
        return JSONResponse({"error": "brief not found"}, status_code=404)

    job_id = str(uuid.uuid4())[:8]
    reverify_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "message": "Extracting claims...",
        "results": None,
    }

    import threading

    def _run():
        from cruicible.tools.reverify import extract_claims, reverify_claims
        claims = extract_claims(brief["brief"])
        brief["claims"] = claims
        reverify_jobs[job_id]["total"] = len(claims)
        reverify_jobs[job_id]["message"] = f"Verifying {len(claims)} claims..."

        def prog(step, total, msg):
            reverify_jobs[job_id]["progress"] = step
            reverify_jobs[job_id]["total"] = total
            reverify_jobs[job_id]["message"] = msg

        results = reverify_claims(claims, progress_cb=prog)
        brief["reverify_results"] = results
        brief["reverified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        reverify_jobs[job_id]["results"] = results
        reverify_jobs[job_id]["status"] = "complete"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/reverify-status/{job_id}")
async def reverify_status(job_id: str):
    job = reverify_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job


experiments: dict[str, dict] = {}


@app.post("/api/improve")
async def improve(request: Request):
    """Trigger cross-run self-improvement: read past failures, update harness."""
    _apply_keys(request)
    from cruicible.tools.improve import improve_from_history
    result = improve_from_history()
    return result


@app.post("/api/experiment")
async def run_experiment(request: Request):
    """Run a before/after improvement experiment in background."""
    _apply_keys(request)
    body = await request.json()
    queries = body.get("queries")
    from cruicible.tools.improve import EXPERIMENT_QUERIES
    if not queries:
        queries = EXPERIMENT_QUERIES

    exp_id = str(uuid.uuid4())[:8]
    experiments[exp_id] = {
        "status": "running",
        "progress": 0,
        "total_steps": 0,
        "message": "Starting experiment...",
        "result": None,
    }

    def progress_cb(step, total, msg):
        experiments[exp_id]["progress"] = step
        experiments[exp_id]["total_steps"] = total
        experiments[exp_id]["message"] = msg

    import threading

    def _run():
        from cruicible.tools.improve import run_improvement_experiment
        result = run_improvement_experiment(queries, progress_callback=progress_cb)
        experiments[exp_id]["result"] = result
        experiments[exp_id]["status"] = "complete"

    threading.Thread(target=_run, daemon=True).start()
    return {"experiment_id": exp_id}


@app.get("/api/experiment/{exp_id}")
async def experiment_status(exp_id: str):
    """Poll experiment progress and result."""
    exp = experiments.get(exp_id)
    if not exp:
        return JSONResponse({"error": "experiment not found"}, status_code=404)
    return {
        "experiment_id": exp_id,
        "status": exp["status"],
        "progress": exp["progress"],
        "total_steps": exp["total_steps"],
        "message": exp["message"],
        "result": exp["result"],
    }

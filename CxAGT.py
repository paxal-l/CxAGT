from flask import Flask, request, jsonify, Response
import requests
import time
import os
import fcntl
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# ============================================================
# KONFIGURATION
# ============================================================
PRIMARY = "http://192.168.50.233:11434"    # Remote Ollama (or similar API), primary and best server
FALLBACK = "http://127.0.0.1:11434"        # Fallback server (this server) with local Ollama
MEMORY_FILE = "/home/ubuntu/mw/memory.txt" # Komprimerad Chat historik
MAX_MEMORY_LINE_CHARS = 5000

MINSIM = 0.56                # 0.58 has worked, but injection was too rare
TOP_K = 3
TIMEOUT = 0.4
MAX_USER_CHARS = 50000
LISTEN_PORT = 11400
LOG_REQUESTS = True
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Enkel debug snapshot (senaste retrieval)
DEBUG_LAST = None

# Scribe
SCRIBE_BACKEND = FALLBACK               # var modellen finns (garage/lokal)
SCRIBE_MODEL = "Gemma2:2b"              # scribe redaktör
SCRIBE_MIN_WORDS = 10
SCRIBE_MIN_CHARS = 30
SCRIBE_MAX_CHARS = 8000
SCRIBE_TIMEOUT = 90


# ============================================================
# CxAGT (ContextAugmenter)
# - Minne: read/pick/inject/append + cache av embeddings
# ============================================================

_embed_model = None
_mem_cache = {"mtime": 0.0, "lines": [], "vecs": None}


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def load_memory_lines() -> list[str]:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            out = []
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                out.append(ln)
            return out
    except FileNotFoundError:
        return []


def load_memory_with_embeddings():
    """
    Laddar memory + embeddings endast om filen ändrats.
    encode(normalize_embeddings=True) används när möjligt (snabb cosine via dot).
    """
    global _mem_cache

    try:
        mtime = os.path.getmtime(MEMORY_FILE)
    except FileNotFoundError:
        _mem_cache = {"mtime": 0.0, "lines": [], "vecs": None}
        return [], None

    if mtime == _mem_cache["mtime"] and _mem_cache["vecs"] is not None:
        return _mem_cache["lines"], _mem_cache["vecs"]

    lines = load_memory_lines()
    if not lines:
        _mem_cache = {"mtime": mtime, "lines": [], "vecs": None}
        return [], None

    model = get_embedder()

    # SentenceTransformer stödjer normalt normalize_embeddings=True, men vi failar snällt om det saknas.
    try:
        vecs = model.encode(lines, normalize_embeddings=True)  # (n, d) normalized
    except TypeError:
        vecs = np.array(model.encode(lines), dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    _mem_cache = {"mtime": mtime, "lines": lines, "vecs": vecs}
    return lines, vecs


def append_memory(line: str, *, max_chars: int = MAX_MEMORY_LINE_CHARS) -> tuple[bool, str]:
    """
    Append en rad i memory.txt (atomiskt, med dedupe).
    """
    line = _norm(line)
    if not line:
        return (False, "empty")

    if max_chars and len(line) > max_chars:
        return (False, f"too_long>{max_chars}")

    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

    with open(MEMORY_FILE, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        # dedupe under lås
        f.seek(0)
        existing = set()
        for ln in f:
            ln = _norm(ln)
            if ln and not ln.startswith("#"):
                existing.add(ln.lower())

        if line.lower() in existing:
            fcntl.flock(f, fcntl.LOCK_UN)
            return (False, "duplicate")

        f.seek(0, os.SEEK_END)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

        fcntl.flock(f, fcntl.LOCK_UN)

    return (True, "written")


def find_last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def find_last_assistant_before_index(messages: list[dict], before_idx: int) -> int:
    """Hitta senaste assistant."""
    i = min(before_idx - 1, len(messages) - 1)
    while i >= 0:
        if messages[i].get("role") == "assistant":
            return i
        i -= 1
    return -1


def clean_one_line(text: str) -> str:
    """Gör en rad: tar bort radbrytningar, extra whitespace och klipper längd."""
    if text is None:
        return ""
    s = str(text).strip()

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")

    # Ta bort om modellen returnerar med yttre citationstecken
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            s = s[1:-1].strip()

    s = " ".join(s.split())

    if len(s) > SCRIBE_MAX_CHARS:
        s = s[:SCRIBE_MAX_CHARS].rstrip()

    return s


def passes_scribe_gate(line: str) -> bool:
    """LÅST gate: minst antal ord, minst antal tecken, exakt en rad. Inga undantag."""
    if not line:
        return False
    if "\n" in line or "\r" in line:
        return False
    if len(line.split()) < SCRIBE_MIN_WORDS:
        return False
    if len(line) < SCRIBE_MIN_CHARS:
        return False
    return True


def scribe_format_from_assistant(assistant_text: str) -> str:
    """Anropar scribe-modellen och returnerar en ren minnesrad (kan bli tom)."""
    prompt = (
        "Skriv om texten nedan till exakt EN tydlig faktabaserad minnesrad (en mening) utan radbrytning.\n"
        "Ingen lista. Ingen förklaring. Lägg inte till ny fakta.\n"
        "Text:\n"
        f"{assistant_text}\n"
    )

    payload = {"model": SCRIBE_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(
        SCRIBE_BACKEND + "/api/generate",
        json=payload,
        timeout=SCRIBE_TIMEOUT,
    )
    r.raise_for_status()
    out = (r.json().get("response") or "").strip()
    return clean_one_line(out)


def inject_background(messages: list[dict], picked: list[str]) -> None:
    if not picked:
        return

    prompt = (
        "Use the chat history below to understand previous chat discussions.\n"
        "Be friendly and personal.\n\n"
        "Comment the previous chat history below, \n"
        "but only if you find it relevant to the user request \n"
        "and on first user request in a new chat or if user change topic.\n"
        "Finally, answer the user request.\n\n"
        "=== CHAT HISTORY ===\n"
    )

    for line in picked:
        prompt += f"{line}\n"

    prompt += (
        "=== END HISTORY ===\n\n"
        "=== USER REQUEST ===\n"
    )

    idx = find_last_user_index(messages)
    if idx >= 0:
        original = messages[idx].get("content") or ""
        messages[idx]["content"] = prompt + original + "\n=== END REQUEST ==="



def pick_relevant_memory_semantic(prompt: str, top_k: int = 3, min_sim: float = 0.58) -> list[str]:
    """
    Semantisk retrieval med enkel robust regel:
      - ta top_k kandidater (högst score)
      - best = högsta score
      - cutoff = max(min_sim, best - 0.05)
      - välj de kandidater i top_k som >= cutoff
    Sparar enkel debug till /mw/debug.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return []

    lines, vecs = load_memory_with_embeddings()
    if vecs is None or not lines:
        return []

    model = get_embedder()

    try:
        q = model.encode([prompt], normalize_embeddings=True)[0]
    except TypeError:
        q = np.array(model.encode([prompt])[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

    # vecs är redan normaliserade => cosine = dot
    sims = np.dot(vecs, q)  # (n,)

    idxs = np.argsort(-sims)  # bäst först
    top = idxs[:max(1, int(top_k))]

    best = float(sims[top[0]])
    cutoff = max(float(min_sim), best - 0.05)

    picked_idxs = [int(i) for i in top if float(sims[i]) >= cutoff]
    picked = [lines[i] for i in picked_idxs]

    # Minimal debug (top 4 + picked flag)
    global DEBUG_LAST
    top4 = [int(i) for i in idxs[:4]]
    picked_set = set(picked_idxs)
    DEBUG_LAST = {
        "prompt": prompt,
        "min_sim": float(min_sim),
        "top_k": int(top_k),
        "candidates": [
            {
                "i": i,
                "score": round(float(sims[i]), 3),
                "picked": (i in picked_set),
                "text": lines[i],
            }
            for i in top4
        ],
    }

    return picked


def cxagt_handle_auto_scribe(messages: list[dict]) -> tuple[bool, dict]:
    idx = find_last_user_index(messages)
    if idx < 0:
        return (False, {})

    user_text = (messages[idx].get("content") or "").strip()

    aidx = find_last_assistant_before_index(messages, idx)
    if aidx < 0:
        return (False, {})

    assistant_text = (messages[aidx].get("content") or "").strip()
    if not assistant_text:
        return (False, {})

    try:
        line = scribe_format_from_assistant(assistant_text)
    except Exception:
        return (False, {})

    if not passes_scribe_gate(line):
        return (False, {})

    ok, why = append_memory(line)
    ack = "✅ Auto-SCRIBE: Sparat i minnet." if ok else f"⚠ Auto-SCRIBE: Sparade inte ({why})."

    resp = {
        "model": "auto-scribe",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": ack},
        "done": True,
    }
    return (True, resp)



def cxagt_maybe_augment(payload: dict) -> tuple[dict, bool]:
    """
    Augmenterar endast om prompten är kort nog.
    Returnerar (payload, allow_fallback)
    - long prompt => allow_fallback=False, ingen injection
    """
    messages = payload.get("messages", [])
    idx = find_last_user_index(messages)
    last_user_text = (messages[idx].get("content") or "") if idx >= 0 else ""

    is_long = len(last_user_text) > MAX_USER_CHARS
    if is_long:
        return (payload, False)

    picked = pick_relevant_memory_semantic(last_user_text, top_k=TOP_K, min_sim=MINSIM)
    inject_background(messages, picked)
    payload["messages"] = messages
    return (payload, True)


# ============================================================
# ModelRouter
# - backendval + proxy mot Ollama endpoints
# ============================================================

def primary_online() -> bool:
    try:
        r = requests.get(PRIMARY + "/api/tags", timeout=TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def choose_backend(allow_fallback: bool) -> str | None:
    if primary_online():
        return PRIMARY
    if allow_fallback:
        return FALLBACK
    return None


def proxy_get(path: str, allow_fallback: bool = True):
    backend = choose_backend(allow_fallback=allow_fallback)
    if not backend:
        return jsonify({}), 200
    try:
        r = requests.get(backend + path, timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


def proxy_post(path: str, payload: dict, allow_fallback: bool):
    backend = choose_backend(allow_fallback=allow_fallback)
    if not backend:
        # UI ska aldrig blockeras
        return jsonify({"message": {"role": "assistant", "content": ""}, "done": True}), 200

    try:
        r = requests.post(backend + path, json=payload, timeout=None)
        try:
            return jsonify(r.json()), r.status_code
        except Exception:
            return Response(r.text, status=r.status_code, mimetype="text/plain")
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ============================================================
# LOG ALLA REQUESTS
# ============================================================

@app.before_request
def log_request():
    if LOG_REQUESTS:
        print(f"\n[MW] {request.method} {request.path}")


# ============================================================
# OLLAMA-ENDPOINTS (kompatibilitet för WebUI)
# ============================================================

@app.get("/api/version")
def api_version():
    return proxy_get("/api/version", allow_fallback=True)


@app.get("/api/tags")
def api_tags():
    return proxy_get("/api/tags", allow_fallback=True)


@app.post("/api/ps")
def api_ps():
    payload = request.get_json(force=True, silent=True) or {}
    return proxy_post("/api/ps", payload, allow_fallback=False)


@app.post("/api/generate")
def api_generate():
    payload = request.get_json(force=True)
    payload["stream"] = False
    return proxy_post("/api/generate", payload, allow_fallback=False)


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True)
    payload["stream"] = False

    messages = payload.get("messages", [])

    # 1) CxAGT: always SCRIBE
    handled, resp = cxagt_handle_auto_scribe(messages)

    # 2) CxAGT: augmentera ev + bestäm om fallback får användas
    payload, allow_fallback = cxagt_maybe_augment(payload)

    # 3) Router: skicka vidare
    return proxy_post("/api/chat", payload, allow_fallback=allow_fallback)


# ============================================================
# DEV-ENDPOINTS (/mw/*)
# ============================================================

def memory_tail_with_lineno(limit: int = 200) -> list[dict]:
    limit = max(1, min(int(limit), 2000))
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    start = max(0, len(lines) - limit)
    items = []
    for idx in range(start, len(lines)):
        items.append({"line": idx + 1, "text": lines[idx].rstrip("\n")})
    return items


def memory_delete_line(line_no: int) -> tuple[bool, str]:
    try:
        line_no = int(line_no)
    except Exception:
        return (False, "bad_line")

    if line_no < 1:
        return (False, "bad_line")

    try:
        with open(MEMORY_FILE, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            lines = f.readlines()
            idx = line_no - 1
            if idx >= len(lines):
                fcntl.flock(f, fcntl.LOCK_UN)
                return (False, "not_found")

            lines.pop(idx)

            f.seek(0)
            f.truncate(0)
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

        return (True, "deleted")
    except FileNotFoundError:
        return (False, "no_file")
    except PermissionError:
        return (False, "no_permission")
    except Exception as e:
        return (False, f"error:{e}")


@app.get("/mw/memory")
def mw_memory_list():
    limit = request.args.get("limit", "200")
    items = memory_tail_with_lineno(limit)
    return jsonify({"count": len(items), "items": items}), 200


@app.post("/mw/memory/delete")
def mw_memory_delete():
    payload = request.get_json(force=True, silent=True) or {}
    line_no = payload.get("line")
    ok, reason = memory_delete_line(line_no)
    status = 200 if ok else 400
    return jsonify({"ok": ok, "reason": reason, "line": line_no}), status


@app.get("/mw/debug")
def mw_debug():
    return jsonify(DEBUG_LAST or {}), 200


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return jsonify({
        "service": "CxAGT + ModelRouter",
        "port": LISTEN_PORT,
        "max_user_chars": MAX_USER_CHARS,
        "primary": PRIMARY,
        "fallback": FALLBACK,
        "memory_file": MEMORY_FILE,
        "minsim": MINSIM,
        "top_k": TOP_K,
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    print(f"CxAGT listening on 0.0.0.0:{LISTEN_PORT}")
    app.run(host="0.0.0.0", port=LISTEN_PORT)

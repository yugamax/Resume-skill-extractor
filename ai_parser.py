# ai_parser.py
from groq import Groq
from dotenv import load_dotenv
import os
import json
import time
import logging
from typing import List, Tuple, Optional

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_parser")

# -------------------------------
# Load Groq Clients
# -------------------------------
_client_env_names = [f"gr_api_key{i}" for i in range(1, 7)]
_clients = []
configured = []

for i, env_name in enumerate(_client_env_names, 1):
    key = os.getenv(env_name)
    if not key:
        continue
    try:
        client = Groq(api_key=key)
        _clients.append(client)
        configured.append(f"{env_name} (index {i})")
    except Exception as e:
        logger.exception("Failed to initialize Groq client %s: %s", env_name, e)

if configured:
    logger.info("Configured Groq clients: %s", ", ".join(configured))


def get_available_clients() -> List[Groq]:
    return _clients


# -------------------------------
# Prompt Template
# Note: any literal braces must be doubled so str.format works.
# -------------------------------
MODEL_PROMPT_TEMPLATE = (
    "You are an expert resume parser. Return ONLY valid JSON (top-level object).\n\n"
    "1) Extract a flat 'skills' array of all detected skills, tools, libraries, frameworks, cloud services,\n"
    "   methodologies and soft skills.\n"
    "2) Return 'is_resume' (boolean) and 'confidence' (float between 0.0 and 1.0).\n"
    "3) If you cannot fully parse sections, always include at least {{'skills': [...], 'is_resume': true/false, 'confidence': 0.0}}.\n"
    "4) Output must be pure JSON with no additional commentary.\n\n"
    "DOCUMENT_TEXT:\n{resume_text}\n"
)


# -------------------------------
# JSON Safety Loader
# -------------------------------
def _safe_load_json_from_model(raw: Optional[str]):
    if not raw:
        return None, "empty model response", ""

    s = raw.strip()

    # First try direct JSON
    try:
        return json.loads(s), None, s
    except Exception:
        pass

    # Try extracting {...}
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = s[start:end+1]
        try:
            return json.loads(block), None, block
        except Exception as e:
            return None, f"json load failed: {e}", block[:2000]

    return None, "no JSON object found", s[:2000]


# -------------------------------
# Call Groq (Sync)
# -------------------------------
def _call_groq_sync(prompt_text: str, user_message: str = "", max_attempts: int = 2):
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_message or "Please parse the document."}
    ]

    clients = get_available_clients()
    if not clients:
        return None, None

    for idx, client in enumerate(clients, 1):
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info("Calling Groq client %d (attempt %d)", idx, attempt)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content, f"client{idx}"
            except Exception:
                time.sleep(0.5 * attempt)

    logger.error("All Groq clients failed.")
    return None, None


# -------------------------------
# Parse Document Sync
# -------------------------------
def parse_document_sync(resume_text: str):
    prompt = MODEL_PROMPT_TEMPLATE.format(resume_text=resume_text)

    try:
        raw, used = _call_groq_sync(prompt)
    except Exception as e:
        return {"parsed": None, "raw": None, "client_used": None, "snippet": "", "error": str(e)}

    if raw is None:
        return {"parsed": None, "raw": None, "client_used": used, "snippet": "", "error": "No model response."}

    parsed, err, snippet = _safe_load_json_from_model(raw)
    return {"parsed": parsed, "raw": raw, "client_used": used, "snippet": snippet, "error": err}


# -------------------------------
# Extract Resume Info
# -------------------------------
def extract_resume_info(resume_text: str):
    out = parse_document_sync(resume_text)
    parsed = out.get("parsed")
    error = out.get("error")

    skills = []
    is_resume = False
    confidence = 0.0

    if isinstance(parsed, dict):
        # ----- Skills -----
        s = parsed.get("skills")
        if isinstance(s, list):
            skills = [str(x).strip().lower() for x in s if str(x).strip()]

        # Fallback: technical_skills
        if not skills:
            ts = parsed.get("technical_skills")
            if isinstance(ts, dict):
                for v in ts.values():
                    if isinstance(v, list):
                        for it in v:
                            it = str(it).strip().lower()
                            if it and it not in skills:
                                skills.append(it)

        # ----- Metadata -----
        is_resume = bool(parsed.get("is_resume", False))

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

    else:
        if not error:
            error = "Model did not return valid JSON."

    return {
        "skills": skills,
        "is_resume": is_resume,
        "confidence": round(confidence, 3),
        "error": error
    }

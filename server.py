# ══════════════════════════════════════════════════════════════
# server.py — ASL Robot Server
#
# Receives gesture/sentence data from robot.py via HTTP POST,
# returns a natural language response.
#
# Run:  python server.py
# Port: 5000  (http://localhost:5000)
# ══════════════════════════════════════════════════════════════

# ── Imports ───────────────────────────────────────────────────
from flask import Flask, request, jsonify
import time
import json
import re

app = Flask(__name__)

# ── Server stats ──────────────────────────────────────────────
_start_time    = time.time()
_request_count = 0

# ── Response dictionary ───────────────────────────────────────
# Maps recognized gesture labels → natural language responses.
# Keys are lowercase. Add/edit freely — no retraining needed.

RESPONSES: dict[str, str] = {
    # ── Greetings ─────────────────────────────────────────────
    "hello":        "Hello! How can I help you?",
    "goodbye":      "Goodbye! Take care.",
    "i love you":   "That means a lot, thank you.",

    # ── Polite ────────────────────────────────────────────────
    "please":       "Of course, I am happy to help.",
    "thank you":    "You are very welcome.",
    "sorry":        "No worries at all.",

    # ── Yes / No ──────────────────────────────────────────────
    "yes":          "Great, understood. Yes confirmed.",
    "no":           "Okay, noted. No confirmed.",

    # ── Needs ─────────────────────────────────────────────────
    "help":         "Help is on the way. What do you need?",
    "stop":         "Stopping now.",
    "more":         "Sure, I will get more.",
    "want":         "What would you like?",
    "water":        "I will get you some water.",
    "eat":          "Are you hungry? I will prepare something.",
    "sleep":        "Okay, rest well.",
    "home":         "Heading home.",

    # ── Status ────────────────────────────────────────────────
    "good":         "Great to hear things are good!",
    "bad":          "I am sorry to hear that. How can I help?",
    "where":        "Can you point or give more context?",
    "how":          "Can you describe what you need help with?",

    # ── Short spelled words ────────────────────────────────────
    "ok":           "Okay, understood.",
    "hi":           "Hi there! How are you?",
    "bye":          "Goodbye, see you soon!",
}

# ── Sentence patterns ─────────────────────────────────────────
# Regex patterns matched before keyword scan.
# Add more tuples here to teach the robot new sentence structures.

PATTERNS = [
    # "my name is X"
    (re.compile(r"my name is (\w+)"),
     lambda m: f"Nice to meet you, {m.group(1).capitalize()}! How can I help you?"),

    # "i am X"
    (re.compile(r"i am (\w+)"),
     lambda m: f"Hello, {m.group(1).capitalize()}! Great to meet you."),

    # "i need X"
    (re.compile(r"i need (\w+)"),
     lambda m: f"You need {m.group(1)}. I will help you with that right away."),

    # "i want X"
    (re.compile(r"i want (\w+)"),
     lambda m: f"You want {m.group(1)}. Let me take care of that."),

    # "where is X"
    (re.compile(r"where is (\w+)"),
     lambda m: f"I will help you find {m.group(1)}."),

    # "help me with X"
    (re.compile(r"help me with (\w+)"),
     lambda m: f"Of course, I will help you with {m.group(1)}."),
]

# ── Fallback ──────────────────────────────────────────────────
def _fallback(gesture: str) -> str:
    if len(gesture) <= 3:
        return f"You signed: {gesture}. Keep spelling to complete your message."
    return f"I received: '{gesture}'. I am still learning — could you try again?"


# ── Response engine ───────────────────────────────────────────
def _respond(gesture: str) -> str:
    """
    Priority order:
      1. Regex pattern  — handles full sentences with structure
      2. Exact match    — single gesture or short phrase
      3. Keyword scan   — first known word found in sentence
      4. Phrase scan    — multi-word phrases ("thank you" etc.)
      5. Fallback       — unknown input
    """
    key = gesture.lower().replace("_", " ").strip()

    # 1. Pattern matching
    for pattern, handler in PATTERNS:
        m = pattern.search(key)
        if m:
            return handler(m)

    # 2. Exact match
    if key in RESPONSES:
        return RESPONSES[key]

    # 3. Keyword scan — first known single word
    words = key.split()
    for word in words:
        if word in RESPONSES:
            return RESPONSES[word]

    # 4. Multi-word phrase scan (longest first to avoid partial matches)
    for phrase in sorted(RESPONSES.keys(), key=len, reverse=True):
        if phrase in key:
            return RESPONSES[phrase]

    return _fallback(gesture)


# ── Routes ────────────────────────────────────────────────────
@app.route("/api/gesture", methods=["POST"])
def receive_gesture():
    """
    Expects JSON:
        { "gesture": "HELLO", "confidence": 0.97, "type": "word" }

    Returns JSON:
        { "response": "Hello! How can I help you?",
          "received": "HELLO", "timestamp": 1234567890.0 }
    """
    global _request_count
    _request_count += 1

    data = request.get_json(silent=True)
    if not data or "gesture" not in data:
        return jsonify({"error": "Missing 'gesture' field"}), 400

    gesture    = str(data["gesture"])
    confidence = float(data.get("confidence", 0.0))
    g_type     = str(data.get("type", "unknown"))
    response   = _respond(gesture)

    print(f"  [{g_type.upper():>8}]  {gesture:<28}  "
          f"conf={confidence:.2f}  →  {response}")

    return jsonify({
        "response":  response,
        "received":  gesture,
        "timestamp": time.time(),
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Robot calls this on startup and periodically to confirm server is alive."""
    return jsonify({
        "status":           "ok",
        "uptime_secs":      round(time.time() - _start_time, 1),
        "requests_served":  _request_count,
        "timestamp":        time.time(),
    })


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  ASL Robot Server")
    print("  Listening on http://localhost:5000")
    print("  Endpoints:")
    print("    POST /api/gesture  — receive gesture")
    print("    GET  /api/health   — health check")
    print("═" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
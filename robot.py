# ══════════════════════════════════════════════════════════════
# robot.py — ASL Robot Client
#
# Identical to app.py PLUS:
#   • ServerClient  — sends confirmed gestures to server.py
#                     via HTTP POST on a background thread
#   • Server response shown in HUD and spoken via TTS
#
# Run AFTER server.py is already running:
#   Terminal 1: python server.py
#   Terminal 2: python robot.py
#
# Keys: S=speak  C=clear  Space=space  Backspace=delete  Q=quit
# ══════════════════════════════════════════════════════════════

from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import os
import time
import queue
import threading
import requests                           # ← NEW
import win32com.client

from features      import (hand_to_features, mediapipe_to_norm63,
                            EMPTY_HAND_FEATURES, FEATURE_DIM_2)
from segmenter     import GestureSegmenter, Phase
from smoother      import ConfidenceSmoother
from pipeline      import CaptureThread, InferenceThread
from gesture_model import load_checkpoint

# ── Paths ─────────────────────────────────────────────────────
MLP_MODEL_PATH   = os.path.join("model", "asl_model_inference.pt")
MLP_LABELS_PATH  = os.path.join("model", "labels.json")
WORD_MODEL_PATH  = os.path.join("model", "word_model.pt")

# ── Server config ─────────────────────────────────────────────  ← NEW
SERVER_URL     = "http://localhost:5000"
SERVER_TIMEOUT = 2.0   # seconds — robot keeps running if server is slow

# ── Thresholds ────────────────────────────────────────────────
MLP_CONF_MIN        = 0.90
MLP_HOLD_SECS       = 0.9
WORD_CONF_MIN       = 0.90
WORD_HOLD_SECS      = 1.4
MLP_EMA_ALPHA       = 0.38
WORD_MIN_FRAMES     = 20
WORD_MOTION_THRESH  = 0.100

# ── Per-word confidence overrides ─────────────────────────────
WORD_CONF_OVERRIDES = {
    "hello": 0.999,
    "no":    0.999,
    "yes":   0.999,
    "more":  0.999,
}

# ── TTS ───────────────────────────────────────────────────────
_tts_q = queue.Queue()

def _tts_worker() -> None:
    import pythoncom
    pythoncom.CoInitialize()
    spk = win32com.client.Dispatch("SAPI.SpVoice")
    spk.Rate = 0; spk.Volume = 100
    while True:
        text = _tts_q.get()
        if text is None:
            break
        try:
            spk.Speak(text)
        except Exception as e:
            print(f"[TTS] {e}")
    pythoncom.CoUninitialize()

threading.Thread(target=_tts_worker, daemon=True).start()

def _clean(text: str) -> str:
    return text.replace("_", " ")

def speak(text: str, flush: bool = True) -> None:
    if flush:
        while not _tts_q.empty():
            try:   _tts_q.get_nowait()
            except: break
    _tts_q.put(_clean(text))


# ══════════════════════════════════════════════════════════════
# NEW — ServerClient
# Sends confirmed gestures to server.py on a background thread.
# The main loop is never blocked waiting for an HTTP response.
# ══════════════════════════════════════════════════════════════
class ServerClient:
    """
    Non-blocking HTTP client.

    Usage:
        client = ServerClient("http://localhost:5000")
        client.send("HELLO", 0.97, "word")   # returns immediately
        response = client.poll()              # None until reply arrives
    """

    def __init__(self, base_url: str, timeout: float = 2.0) -> None:
        self._url     = base_url.rstrip("/") + "/api/gesture"
        self._timeout = timeout
        self._out_q   = queue.Queue(maxsize=5)
        self._in_q    = queue.Queue(maxsize=5)
        self._thread  = threading.Thread(
            target=self._worker, daemon=True, name="ServerClient")
        self._thread.start()

    def _worker(self) -> None:
        while True:
            payload = self._in_q.get()
            if payload is None:
                break
            try:
                r = requests.post(
                    self._url, json=payload, timeout=self._timeout)
                if r.status_code == 200:
                    data = r.json()
                    # Drop old result if not yet consumed
                    while not self._out_q.empty():
                        try: self._out_q.get_nowait()
                        except: break
                    self._out_q.put(data.get("response", ""))
                else:
                    print(f"  [Server] HTTP {r.status_code}")
            except requests.exceptions.ConnectionError:
                print("  [Server] ⚠  Cannot reach server — is server.py running?")
            except requests.exceptions.Timeout:
                print("  [Server] ⚠  Request timed out")
            except Exception as e:
                print(f"  [Server] error: {e}")

    def send(self, gesture: str, confidence: float, g_type: str) -> None:
        """Queue a gesture to be sent. Never blocks."""
        payload = {
            "gesture":    gesture,
            "confidence": round(confidence, 4),
            "type":       g_type,
        }
        try:
            self._in_q.put_nowait(payload)
        except queue.Full:
            pass   # drop if server is very slow — keep robot real-time

    def poll(self) -> str | None:
        """Return server response string if one arrived, else None."""
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def health_check(self) -> bool:
        """Blocking check on startup. Returns True if server is reachable."""
        try:
            r = requests.get(
                self._url.replace("/api/gesture", "/api/health"),
                timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def is_alive(self) -> bool:
        """Non-blocking live check — safe to call every frame."""
        try:
            r = requests.get(
                self._url.replace("/api/gesture", "/api/health"),
                timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    def stop(self) -> None:
        self._in_q.put(None)


# ── Sentence builder ──────────────────────────────────────────
class SentenceBuilder:
    def __init__(self) -> None:
        self.text: str = ""

    def add_letter(self, letter: str) -> None:
        if   letter == "DEL":     self.text = self.text[:-1]
        elif letter == "SPACE":   self.text += " "
        elif letter != "NOTHING": self.text += letter

    def add_word(self, word: str) -> None:
        if not word: return
        word = word.replace("_", " ")
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self.text += word + " "

    def get_sentence(self) -> str:
        return self.text.strip()

    def get_display(self) -> str:
        return self.text

    def clear(self) -> None:
        self.text = ""


# ── HUD helpers ───────────────────────────────────────────────
def _bar(frame, x, y, w, h, pct, color, label) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 35), -1)
    cv2.rectangle(frame, (x, y),
                  (x + int(w * max(0.0, min(pct, 1.0))), y + h), color, -1)
    cv2.putText(frame, label, (x + 4, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

def _panel(frame, lines, x, y, line_h=22) -> None:
    h  = len(lines) * line_h + 10
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x + frame.shape[1], y + h),
                  (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    for i, (txt, col, sc, th) in enumerate(lines):
        cv2.putText(frame, txt, (x + 6, y + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)

_PHASE_COLORS = {
    "IDLE":    (80,  80,  80),
    "FORMING": (0,  140, 255),
    "STABLE":  (0,  210,  90),
}

def _badge(frame, phase_name, pred, conf) -> None:
    col   = _PHASE_COLORS.get(phase_name, (80, 80, 80))
    label = phase_name
    if pred:
        label += f"  |  {pred}  {conf * 100:.0f}%"
    cv2.rectangle(frame, (0, 0), (360, 34), (22, 22, 22), -1)
    cv2.putText(frame, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

def _flash(frame, text, color=(0, 255, 128)) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
    cv2.putText(frame, text, ((w - sz[0]) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)


# ── Model loading ─────────────────────────────────────────────
def _load_mlp(device):
    if not os.path.exists(MLP_MODEL_PATH):
        raise SystemExit("❌ Letter model not found. Run training pipeline first.")
    model = torch.jit.load(MLP_MODEL_PATH, map_location=device)
    model.eval()
    with open(MLP_LABELS_PATH) as f:
        labels = json.load(f)
    print(f"  Letter MLP loaded   ({len(labels)} classes)")
    return model, labels

def _load_word(device):
    if not os.path.exists(WORD_MODEL_PATH):
        print("  Word model not found — letter-only mode")
        return None, None
    try:
        model, labels = load_checkpoint(WORD_MODEL_PATH, device)
        print(f"  Word model loaded   ({len(labels)} classes)")
        return model, labels
    except Exception as e:
        print(f"  Word model load failed ({e}) — letter-only mode")
        return None, None


# ── Motion-energy gate ────────────────────────────────────────
def _segment_is_dynamic(segment: list) -> bool:
    if len(segment) < 4:
        return False
    coords   = [np.asarray(f[:63]) for f in segment]
    mean_vel = float(np.mean([
        np.linalg.norm(coords[i] - coords[i - 1])
        for i in range(1, len(coords))
    ]))
    return mean_vel > WORD_MOTION_THRESH


# ── Word inference closure ────────────────────────────────────
def _make_infer_fn(word_model, word_labels, device, max_len=90):
    if word_model is None:
        return None

    def _infer(segment: list):
        T   = min(len(segment), max_len)
        arr = np.zeros((max_len, FEATURE_DIM_2), dtype=np.float32)
        for i, feat in enumerate(segment[:T]):
            arr[i] = feat
        mask     = np.ones(max_len, dtype=bool)
        mask[:T] = False
        x_t = torch.tensor(arr).unsqueeze(0).to(device)
        m_t = torch.tensor(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(
                word_model(x_t, m_t), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return word_labels[idx], float(probs[idx])

    return _infer


# ── Main ──────────────────────────────────────────────────────
def run() -> None:
    print("\nLoading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    mlp,  mlp_labels  = _load_mlp(device)
    word, word_labels = _load_word(device)

    mlp_smoother = ConfidenceSmoother(len(mlp_labels), alpha=MLP_EMA_ALPHA)
    segmenter    = GestureSegmenter()
    builder      = SentenceBuilder()

    # ── NEW — connect to server ───────────────────────────────
    server = ServerClient(SERVER_URL, timeout=SERVER_TIMEOUT)
    print(f"\n  Connecting to server at {SERVER_URL} ...")
    if server.health_check():
        print("  Server ✅  connected\n")
        server_status = "connected"
    else:
        print("  Server ⚠  not reachable — robot runs in offline mode\n")
        server_status = "offline"

    # ── Threaded components ───────────────────────────────────
    capture   = CaptureThread(camera_index=0)
    capture.start()

    seg_in_q  = queue.Queue(maxsize=3)
    seg_out_q = queue.Queue(maxsize=1)
    infer_fn  = _make_infer_fn(word, word_labels, device)

    inf_thread = None
    if infer_fn is not None:
        inf_thread = InferenceThread(infer_fn, seg_in_q, seg_out_q)
        inf_thread.start()

    # ── MediaPipe ─────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    # ── State ─────────────────────────────────────────────────
    hold_letter        = None
    hold_start         = 0.0
    last_added         = None

    word_candidate     = None
    word_cand_start    = 0.0
    word_cand_conf     = 0.0

    prev_norm_h1       = None
    prev_norm_h2       = None

    flash_until        = 0.0
    flash_text         = ""
    flash_color        = (0, 255, 128)

    # ── server response state ─────────────────────────────────
    server_response      = ""
    server_response_time = 0.0

    # ── server live-status tracking ───────────────────────────
    last_health_check  = 0.0       # time of last ping
    HEALTH_INTERVAL    = 5.0       # re-ping every 5 seconds

    print("Ready.\n")

    with mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    ) as hands:

        while True:
            frame = capture.get_latest()
            if frame is None:
                continue

            now      = time.time()
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res      = hands.process(rgb)
            rgb.flags.writeable = True
            has_hand = bool(res.multi_hand_landmarks)

            # ── Draw landmarks ────────────────────────────────
            if has_hand:
                for lm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style())

            # ── Feature extraction ────────────────────────────
            feat_h1  = EMPTY_HAND_FEATURES.copy()
            feat_h2  = EMPTY_HAND_FEATURES.copy()
            norm_h1  = np.zeros(63, dtype=np.float32)

            if has_hand:
                feat_h1, norm_h1 = hand_to_features(
                    res.multi_hand_landmarks[0], prev_norm_h1)
                prev_norm_h1 = norm_h1.copy()
                if len(res.multi_hand_landmarks) > 1:
                    feat_h2, norm_h2 = hand_to_features(
                        res.multi_hand_landmarks[1], prev_norm_h2)
                    prev_norm_h2 = norm_h2.copy()
                else:
                    prev_norm_h2 = None
            else:
                prev_norm_h1 = None
                prev_norm_h2 = None

            feat_combined = np.concatenate([feat_h1, feat_h2])

            # ── Segmenter ─────────────────────────────────────
            segment, was_stable = segmenter.update(
                feat_combined, norm_h1, has_hand)

            if (segment and infer_fn is not None
                    and not was_stable
                    and len(segment) >= WORD_MIN_FRAMES
                    and _segment_is_dynamic(segment)):
                try:
                    seg_in_q.put_nowait(segment)
                except queue.Full:
                    pass

            # ── Poll word results ─────────────────────────────
            try:
                while True:
                    r = seg_out_q.get_nowait()
                    required = WORD_CONF_OVERRIDES.get(r[0], WORD_CONF_MIN)
                    if r[1] >= required:
                        word_candidate  = r[0]
                        word_cand_start = now
                        word_cand_conf  = r[1]
            except queue.Empty:
                pass

            if word_candidate and (now - word_cand_start) > WORD_HOLD_SECS * 3:
                word_candidate = None

            # ── poll server response ──────────────────────────
            resp = server.poll()
            if resp:
                server_response      = resp
                server_response_time = now
                speak(resp)

            # ── periodic live health check ────────────────────
            if now - last_health_check > HEALTH_INTERVAL:
                last_health_check = now
                was_offline = (server_status == "offline")
                server_status = "connected" if server.is_alive() else "offline"
                if was_offline and server_status == "connected":
                    print("  [Server] ✅  reconnected")
                elif not was_offline and server_status == "offline":
                    print("  [Server] ⚠  connection lost")

            # ── Letter model (STABLE phase only) ──────────────
            predicted_letter = None
            letter_conf      = 0.0

            if segmenter.phase == Phase.STABLE and has_hand:
                vec63 = mediapipe_to_norm63(res.multi_hand_landmarks[0])
                t     = torch.tensor(vec63).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(mlp(t), dim=1)[0].cpu().numpy()
                smoothed = mlp_smoother.update(probs)
                idx, letter_conf = mlp_smoother.best()
                raw = mlp_labels[str(idx)]
                if letter_conf >= MLP_CONF_MIN and raw != "NOTHING":
                    predicted_letter = raw
            else:
                if segmenter.phase != Phase.STABLE:
                    mlp_smoother.reset()
                    hold_letter = None
                    last_added  = None

            # ── Priority: letters beat words ──────────────────
            if predicted_letter:
                word_candidate = None
                if predicted_letter != last_added:
                    if predicted_letter == hold_letter:
                        elapsed  = now - hold_start
                        progress = min(elapsed / MLP_HOLD_SECS, 1.0)
                        _bar(frame, 10, 40, 300, 14, progress,
                             (0, 200, 80), f"Letter: {predicted_letter}")
                        if elapsed >= MLP_HOLD_SECS:
                            builder.add_letter(predicted_letter)
                            last_added  = predicted_letter
                            hold_letter = None
                            flash_text  = predicted_letter
                            flash_color = (0, 255, 128)
                            flash_until = now + 0.3
                            # letters go into builder only — no server call
                            # server receives the full sentence on S key
                            speak(predicted_letter)
                    else:
                        hold_letter = predicted_letter
                        hold_start  = now

            elif word_candidate:
                elapsed  = now - word_cand_start
                progress = min(elapsed / WORD_HOLD_SECS, 1.0)
                _bar(frame, 10, 40, 300, 14, progress,
                     (0, 160, 220), f"Word: {word_candidate}")
                if elapsed >= WORD_HOLD_SECS:
                        if len(word_candidate) == 1:
                            builder.add_letter(word_candidate)
                        else:
                            builder.add_word(word_candidate)
                        flash_text  = word_candidate
                        flash_color = (0, 220, 255)
                        flash_until = now + 0.45
                        server.send(word_candidate, word_cand_conf, "word")
                        # speak locally only if server is offline
                        if server_status == "offline":
                            speak(word_candidate)
                        word_candidate = None
                        hold_letter    = None
                        last_added     = None
                        mlp_smoother.reset()
            else:
                hold_letter = None

            # ── Render ────────────────────────────────────────
            disp_pred = predicted_letter or word_candidate or ""
            disp_conf = letter_conf if predicted_letter else \
                        (word_cand_conf if word_candidate else 0.0)
            _badge(frame, segmenter.phase_name, disp_pred, disp_conf)

            if predicted_letter:
                cv2.putText(frame, predicted_letter,
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                            2.6, (0, 220, 100), 5)
                cv2.putText(frame, f"{letter_conf * 100:.1f}%",
                            (175, 135), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (160, 160, 160), 2)
            elif word_candidate:
                cv2.putText(frame, word_candidate,
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                            2.2, (0, 200, 255), 4)
                cv2.putText(frame, f"{word_cand_conf * 100:.1f}%",
                            (10, 165), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (140, 140, 140), 2)

            _bar(frame, 10, 82, 300, 10, segmenter.buffer_progress(),
                 _PHASE_COLORS.get(segmenter.phase_name, (80, 80, 80)),
                 f"Buf: {segmenter.buffer_len()} frames")

            hand_count = len(res.multi_hand_landmarks) if has_hand else 0
            hcol = ((0, 200, 100) if hand_count == 2 else
                    (0, 160, 255) if hand_count == 1 else (70, 70, 70))
            cv2.rectangle(frame, (370, 0), (490, 34), (25, 25, 25), -1)
            cv2.putText(frame, f"Hands:{hand_count}/2", (375, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hcol, 2)

            if now < flash_until:
                _flash(frame, flash_text, flash_color)

            # ── NEW — server status + response in HUD ─────────
            srv_col = (0, 200, 100) if server_status == "connected" \
                      else (0, 80, 220) if server_status == "offline" \
                      else (180, 180, 0)
            resp_age    = now - server_response_time
            resp_text   = server_response if resp_age < 6.0 else ""
            resp_display = f"Server: {resp_text}" if resp_text \
                           else f"Server: [{server_status}]"

            _panel(frame, [
                (f"Sentence: {builder.get_display()}", (220, 220, 220), 0.55, 1),
                (resp_display, srv_col, 0.48, 1),
            ], 0, frame.shape[0] - 90)

            _panel(frame, [
                ("S=speak  C=clear  Space=space  Bksp=delete  Q=quit",
                 (150, 150, 150), 0.42, 1)
            ], 0, frame.shape[0] - 38)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord(" "): builder.add_letter("SPACE")
            elif key == ord("s"):
                s = builder.get_sentence()
                if s:
                    server.send(s, 1.0, "sentence")
                    speak(s, flush=False)
            elif key == ord("c"):
                builder.clear(); segmenter.reset()
                mlp_smoother.reset()
                word_candidate  = None
                server_response = ""
                hold_letter     = None
                last_added      = None
            elif key == 8:
                builder.add_letter("DEL")

            cv2.imshow("ASL Robot  |  S=speak  C=clear  Q=quit", frame)

    capture.stop()
    if inf_thread is not None:
        inf_thread.stop()
    server.stop()
    cv2.destroyAllWindows()
    _tts_q.put(None)


if __name__ == "__main__":
    run()
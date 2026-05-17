
from __future__ import annotations
import threading
import queue
import cv2
from typing import Callable, Optional


class CaptureThread(threading.Thread):
    """
    Reads frames from the webcam on a dedicated thread.

    Only the latest frame is stored; if the consumer is slower than
    the camera the old frame is silently overwritten.  This prevents
    frame-queue build-up and keeps latency near zero.
    """

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(daemon=True, name="CaptureThread")
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # hardware buffer = 1 frame
        self._frame: Optional[object] = None
        self._lock    = threading.Lock()
        self._running = True

    # ── Thread body ───────────────────────────────────────────
    def run(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                with self._lock:
                    self._frame = frame

    # ── Consumer API ──────────────────────────────────────────
    def get_latest(self):
        """
        Returns a copy of the most recent frame, or None if none
        has been captured yet.  Never blocks.
        """
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        self._cap.release()


class InferenceThread(threading.Thread):
    """
    Runs a potentially slow inference function in the background.

    Producer (main loop) puts complete gesture segments on `in_queue`.
    Consumer (this thread) calls `infer_fn(segment)` and puts the
    result on `out_queue` for the main loop to collect.

    If inference falls behind (in_queue fills up) new segments are
    dropped to prevent unbounded latency — the system always shows
    real-time results, not stale ones.
    """

    def __init__(
        self,
        infer_fn: Callable,
        in_queue:  queue.Queue,
        out_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True, name="InferenceThread")
        self._infer   = infer_fn
        self._in_q    = in_queue
        self._out_q   = out_queue
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                segment = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                result = self._infer(segment)
                if result is not None:
                    # Overwrite stale result if consumer hasn't read it yet
                    while not self._out_q.empty():
                        try:
                            self._out_q.get_nowait()
                        except queue.Empty:
                            break
                    self._out_q.put(result)
            except Exception as exc:
                print(f"[InferenceThread] error: {exc}")

    def stop(self) -> None:
        self._running = False

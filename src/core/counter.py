"""
People counting logic with crossing detection.
"""
import time
from collections import defaultdict, deque
import cv2
import numpy as np

from ..config import (
    MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH, CONF_THRESHOLD, INPUT_SIZE,
    RESIZE_TO, LINE_POSITION, SMOOTH_ALPHA, HISTORY_MAX,
    MIN_HISTORY_TO_COUNT, DEBOUNCE_SEC, RECOUNT_TIMEOUT_SEC,
    MIN_MOVEMENT_PIXELS, INFO_PANEL_POS, WINDOW_NAME, MQTT_CONFIG
)
from ..stream import RTMPReader
from ..mqtt import MQTTManager
from .detector import PersonDetector
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PeopleCounter:
    """
    Main people counter class that orchestrates detection, tracking, and counting.
    """
    
    def __init__(self, source: str, *,
                 model_config=MODEL_CONFIG_PATH,
                 model_weights=MODEL_WEIGHTS_PATH,
                 conf=CONF_THRESHOLD,
                 input_size=INPUT_SIZE,
                 resize_to=RESIZE_TO):
        """
        Initialize People Counter.
        
        Args:
            source: Video source URL
            model_config: Path to MobileNet-SSD config
            model_weights: Path to MobileNet-SSD weights
            conf: Confidence threshold
            input_size: Input size for network
            resize_to: Tuple (width, height) for frame resizing
        """
        self.source = source
        self.resize_to = resize_to

        # Initialize detector
        self.detector = PersonDetector(model_config, model_weights, conf, input_size)

        # Tracking & counting state
        self.track_history = defaultdict(lambda: deque(maxlen=HISTORY_MAX))
        self.last_count_time = {}
        self.last_count_type = {}
        self.people_in = 0
        self.people_out = 0
        self.people_inside = 0

        # Stream reader
        self.reader = RTMPReader(self.source, use_ffmpeg=True)

        # MQTT manager
        self.mqtt_manager = MQTTManager(MQTT_CONFIG)

        # Runtime sizes
        self.rw = None
        self.rh = None

    def compute_mean_prev_x(self, hist_deque, n=3):
        """Compute mean of last n previous x positions."""
        if len(hist_deque) < n + 1:
            if len(hist_deque) >= 2:
                vals = [p[0] for p in list(hist_deque)[:-1]]
                return float(np.mean(vals))
            return None
        vals = [p[0] for p in list(hist_deque)[-(n + 1):-1]]
        return float(np.mean(vals))

    def should_count(self, track_id, prev_x, curr_x, line_x):
        """
        Decide whether crossing happened.
        
        Returns:
            'in', 'out' or None
        """
        now = time.time()

        # Debounce check
        last = self.last_count_time.get(track_id, 0.0)
        if now - last < DEBOUNCE_SEC:
            return None

        if prev_x is None:
            return None

        # Minimum movement check
        if abs(curr_x - prev_x) < MIN_MOVEMENT_PIXELS:
            return None

        # MASUK: left to right
        if prev_x < line_x <= curr_x:
            if self.last_count_type.get(track_id) == "in" and now - last < RECOUNT_TIMEOUT_SEC:
                return None
            self.last_count_time[track_id] = now
            self.last_count_type[track_id] = "in"
            return "in"

        # KELUAR: right to left
        if prev_x > line_x >= curr_x:
            if self.last_count_type.get(track_id) == "out" and now - last < RECOUNT_TIMEOUT_SEC:
                return None
            self.last_count_time[track_id] = now
            self.last_count_type[track_id] = "out"
            return "out"

        return None

    def draw_info_panel(self, frame):
        """Draw info panel with counts."""
        panel_h, panel_w = 140, 300
        x0, y0 = INFO_PANEL_POS
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        alpha = 0.55
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f"MASUK: {self.people_in}", (x0 + 10, y0 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"KELUAR: {self.people_out}", (x0 + 10, y0 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"DI DALAM: {self.people_inside}", (x0 + 10, y0 + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        return frame

    def process_detections(self, boxes, ids, frame, line_x):
        """Process detections and update counts."""
        has_ids = ids is not None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if has_ids:
                track_id = int(ids[i])
                
                # Smoothing
                if len(self.track_history[track_id]) > 0:
                    px, py = self.track_history[track_id][-1]
                    cx = int(px * (1 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA)
                    cy = int(py * (1 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA)

                self.track_history[track_id].append((cx, cy))

                # Counting logic
                if len(self.track_history[track_id]) >= MIN_HISTORY_TO_COUNT:
                    prev_x_mean = self.compute_mean_prev_x(self.track_history[track_id], n=2)
                    action = self.should_count(track_id, prev_x_mean, cx, line_x)
                    
                    if action == "in":
                        self.people_in += 1
                        self.people_inside += 1
                        logger.info(f"[MASUK] ID:{track_id} inside={self.people_inside} (in_total={self.people_in})")
                        self.mqtt_manager.publish_count(self.people_inside)
                    elif action == "out":
                        self.people_out += 1
                        self.people_inside = max(0, self.people_inside - 1)
                        logger.info(f"[KELUAR] ID:{track_id} inside={self.people_inside} (out_total={self.people_out})")
                        self.mqtt_manager.publish_count(self.people_inside)

                # Draw trail
                pts = np.array(self.track_history[track_id], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, (255, 255, 0), 2)

                # Draw bbox
                color = (0, 255, 0) if cx < line_x else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            else:
                color = (200, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, "NO_ID", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)

        return frame

    def cleanup_stale_tracks(self):
        """Remove stale track histories."""
        stale_ids = [tid for tid, hist in self.track_history.items() if len(hist) == 0]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.last_count_time.pop(tid, None)
            self.last_count_type.pop(tid, None)

    def reset_counters(self):
        """Reset all counters and histories."""
        logger.info("Reset counters & history")
        self.people_in = 0
        self.people_out = 0
        self.people_inside = 0
        self.track_history.clear()
        self.last_count_time.clear()
        self.last_count_type.clear()

    def run(self):
        """Main run loop."""
        # Wait for first frame
        logger.info("Waiting for first frame...")
        t0 = time.time()
        while True:
            ret, frame, ts = self.reader.read()
            if ret and frame is not None:
                break
            if time.time() - t0 > 15.0:
                raise RuntimeError("Timeout waiting for initial frame from source")
            time.sleep(0.1)

        original_h, original_w = frame.shape[:2]
        if self.resize_to is not None:
            self.rw, self.rh = self.resize_to
        else:
            self.rw, self.rh = original_w, original_h

        line_x = int(self.rw * LINE_POSITION)
        logger.info(f"Line vertical X = {line_x} (frame width {self.rw})")

        prev_time = time.time()

        while True:
            ret, frame, ts = self.reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if self.resize_to is not None:
                frame = cv2.resize(frame, (self.rw, self.rh))

            # Detect and track
            boxes, ids = self.detector.detect_and_track(frame)

            # Process detections
            if boxes is not None and len(boxes) > 0:
                frame = self.process_detections(boxes, ids, frame, line_x)
                self.cleanup_stale_tracks()

            # Draw UI elements
            cv2.line(frame, (line_x, 0), (line_x, self.rh), (0, 255, 0), 3)
            frame = self.draw_info_panel(frame)

            # FPS
            now_time = time.time()
            fps = 1.0 / (now_time - prev_time) if now_time - prev_time > 0 else 0.0
            prev_time = now_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (self.rw - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.reset_counters()

        # Cleanup
        self.reader.stop()
        self.mqtt_manager.close()
        cv2.destroyAllWindows()

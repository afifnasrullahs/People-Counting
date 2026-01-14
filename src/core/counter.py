"""
People counting logic with crossing detection.
"""
import time
from collections import defaultdict, deque
import cv2
import numpy as np

from ..config import (
    CONF_THRESHOLD, RESIZE_TO, LINE_POSITION, SMOOTH_ALPHA, HISTORY_MAX,
    MIN_HISTORY_TO_COUNT, DEBOUNCE_SEC, RECOUNT_TIMEOUT_SEC,
    MIN_MOVEMENT_PIXELS, INFO_PANEL_POS, WINDOW_NAME, MQTT_CONFIG,
    HEADLESS, FRAME_SKIP, DETECTOR_TYPE, YOLO_MODEL_PATH, YOLO_DEVICE
)
from ..stream import RTMPReader
from ..mqtt import MQTTManager
from .yolo_detector import YOLOv8Detector, YOLOv8DetectorONNX
from .yolo_rpi import YOLOv8DetectorRPi, YOLOv8DetectorLite
from .yolo_openvino import YOLOv8OpenVINO, YOLOv8UltralyticsLite
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PeopleCounter:
    """
    Main people counter class that orchestrates detection, tracking, and counting.
    """
    
    def __init__(self, source: str, *,
                 conf=CONF_THRESHOLD,
                 resize_to=RESIZE_TO,
                 detector_type=DETECTOR_TYPE):
        """
        Initialize People Counter.
        
        Args:
            source: Video source URL
            conf: Confidence threshold
            resize_to: Tuple (width, height) for frame resizing
            detector_type: "yolov8" or "yolov8_onnx"
        """
        self.source = source
        self.resize_to = resize_to

        # Initialize detector based on type
        if detector_type == "openvino":
            # OpenVINO optimized (best for Intel, works on ARM too)
            logger.info("Using OpenVINO detector (optimized inference)")
            try:
                self.detector = YOLOv8OpenVINO(
                    conf=conf,
                    input_size=256,
                    skip_detection=2
                )
            except Exception as e:
                logger.warning(f"OpenVINO failed: {e}, trying ultralytics lite")
                self.detector = YOLOv8UltralyticsLite(conf=conf, input_size=256)
        elif detector_type == "yolov8_lite" or detector_type == "lite":
            # Ultralytics with aggressive optimizations
            logger.info("Using YOLOv8 Lite detector")
            self.detector = YOLOv8UltralyticsLite(
                conf=conf,
                input_size=256,
                skip_detection=2
            )
        elif detector_type == "yolov8_rpi" or detector_type == "yolov8n":
            # Optimized for Raspberry Pi - uses YOLOv8n with smaller input
            logger.info("Using YOLOv8n RPi-optimized detector (target: 10+ FPS)")
            try:
                self.detector = YOLOv8DetectorRPi(
                    conf=conf,
                    input_size=320  # Smaller input for speed
                )
            except Exception as e:
                logger.warning(f"YOLOv8 RPi failed: {e}, trying standard YOLOv8")
                self.detector = YOLOv8Detector(
                    model_path=str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH else None,
                    conf=conf,
                    device="cpu"
                )
        elif detector_type == "yolov8_lite":
            # Ultra-lightweight using OpenCV DNN
            logger.info("Using YOLOv8 Lite detector (OpenCV DNN)")
            try:
                self.detector = YOLOv8DetectorLite(conf=conf, input_size=320)
            except Exception as e:
                logger.warning(f"YOLOv8 Lite failed: {e}, trying RPi detector")
                self.detector = YOLOv8DetectorRPi(conf=conf, input_size=320)
        elif detector_type == "yolov8_onnx":
            logger.info("Using YOLOv8 ONNX detector with SORT tracking")
            try:
                self.detector = YOLOv8DetectorONNX(conf=conf)
            except Exception as e:
                logger.warning(f"YOLOv8 ONNX failed: {e}, falling back to YOLOv8")
                self.detector = YOLOv8Detector(
                    model_path=str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH else None,
                    conf=conf,
                    device=YOLO_DEVICE
                )
        else:
            # Default to YOLOv8
            logger.info("Using YOLOv8 detector with SORT tracking")
            self.detector = YOLOv8Detector(
                model_path=str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH else None,
                conf=conf,
                device=YOLO_DEVICE
            )

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
        
        # MQTT periodic update settings (every 1 second)
        self.mqtt_update_interval = 1.0  # seconds
        self.last_mqtt_update_time = 0.0

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
        Each track ID is counted independently.
        
        Returns:
            'in', 'out' or None
        """
        now = time.time()

        # Per-track debounce check - prevents same person counted multiple times
        last = self.last_count_time.get(track_id, 0.0)
        if now - last < DEBOUNCE_SEC:
            return None

        if prev_x is None:
            return None

        # Calculate movement direction
        movement = curr_x - prev_x
        
        # Skip if no significant movement
        if abs(movement) < 3:  # Very small threshold just to filter noise
            return None

        # MASUK: left to right (crossing the line)
        # Detect crossing: was on left side, now on right side OR very close to crossing
        crossed_to_right = (prev_x < line_x and curr_x >= line_x)
        # Also detect if person is moving right and passes through line area
        near_line_moving_right = (movement > 0 and 
                                   prev_x < line_x + 15 and 
                                   curr_x > line_x - 15 and
                                   curr_x > prev_x)
        
        if crossed_to_right or (near_line_moving_right and prev_x < line_x <= curr_x):
            # Check if same action recently to prevent double count
            if self.last_count_type.get(track_id) == "in" and now - last < RECOUNT_TIMEOUT_SEC:
                return None
            self.last_count_time[track_id] = now
            self.last_count_type[track_id] = "in"
            return "in"

        # KELUAR: right to left (crossing the line)
        crossed_to_left = (prev_x > line_x and curr_x <= line_x)
        near_line_moving_left = (movement < 0 and
                                  prev_x > line_x - 15 and
                                  curr_x < line_x + 15 and
                                  curr_x < prev_x)
        
        if crossed_to_left or (near_line_moving_left and prev_x > line_x >= curr_x):
            # Check if same action recently to prevent double count
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
                
                # Get previous position before smoothing for crossing detection
                prev_cx = None
                if len(self.track_history[track_id]) > 0:
                    prev_cx = self.track_history[track_id][-1][0]
                
                # Smoothing for display
                display_cx, display_cy = cx, cy
                if len(self.track_history[track_id]) > 0:
                    px, py = self.track_history[track_id][-1]
                    display_cx = int(px * (1 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA)
                    display_cy = int(py * (1 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA)

                self.track_history[track_id].append((display_cx, display_cy))

                # Counting logic - check crossing immediately
                # Use direct previous position for responsive detection
                if len(self.track_history[track_id]) >= 2:
                    # Get the position from 1-2 frames ago for crossing detection
                    hist_list = list(self.track_history[track_id])
                    if len(hist_list) >= 3:
                        # Use position from 2 frames ago for more reliable crossing detection
                        prev_x = hist_list[-3][0]
                    else:
                        prev_x = hist_list[-2][0]
                    
                    action = self.should_count(track_id, prev_x, display_cx, line_x)
                    
                    if action == "in":
                        self.people_in += 1
                        self.people_inside += 1
                        logger.info(f"[MASUK] ID:{track_id} inside={self.people_inside} (in_total={self.people_in})")
                    elif action == "out":
                        self.people_out += 1
                        self.people_inside = max(0, self.people_inside - 1)
                        logger.info(f"[KELUAR] ID:{track_id} inside={self.people_inside} (out_total={self.people_out})")

                # Draw trail
                pts = np.array(self.track_history[track_id], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, (255, 255, 0), 2)

                # Draw bbox
                color = (0, 255, 0) if display_cx < line_x else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (display_cx, display_cy), 4, (0, 0, 255), -1)
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
        # Wait for first frame with longer timeout for Raspberry Pi
        logger.info("Waiting for first frame (this may take up to 60 seconds)...")
        t0 = time.time()
        while True:
            ret, frame, ts = self.reader.read()
            if ret and frame is not None:
                logger.info("First frame received!")
                break
            elapsed = time.time() - t0
            if elapsed > 60.0:  # Increased timeout to 60 seconds
                raise RuntimeError("Timeout waiting for initial frame from source. Check VIDEO_SOURCE URL and network connection.")
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                logger.info(f"Still waiting for frame... ({int(elapsed)}s)")
            time.sleep(0.5)

        original_h, original_w = frame.shape[:2]
        if self.resize_to is not None:
            self.rw, self.rh = self.resize_to
        else:
            self.rw, self.rh = original_w, original_h

        line_x = int(self.rw * LINE_POSITION)
        logger.info(f"Line vertical X = {line_x} (frame width {self.rw})")
        logger.info(f"Headless mode: {HEADLESS}, Frame skip: {FRAME_SKIP}")

        prev_time = time.time()
        frame_count = 0

        while True:
            ret, frame, ts = self.reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            
            # Skip frames for performance (Raspberry Pi optimization)
            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                continue

            if self.resize_to is not None:
                frame = cv2.resize(frame, (self.rw, self.rh))

            # Detect and track
            boxes, ids = self.detector.detect_and_track(frame)

            # Process detections
            if boxes is not None and len(boxes) > 0:
                frame = self.process_detections(boxes, ids, frame, line_x)
                self.cleanup_stale_tracks()

            # MQTT periodic update - publish every 1 second regardless of changes
            current_time = time.time()
            if current_time - self.last_mqtt_update_time >= self.mqtt_update_interval:
                self.mqtt_manager.publish_count(self.people_inside)
                self.last_mqtt_update_time = current_time

            # Only render UI if not headless
            if not HEADLESS:
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
            else:
                # Headless mode - just log periodically
                now_time = time.time()
                if now_time - prev_time >= 5.0:  # Log every 5 seconds
                    fps = frame_count / (now_time - prev_time) if now_time - prev_time > 0 else 0.0
                    logger.info(f"Status: IN={self.people_in}, OUT={self.people_out}, INSIDE={self.people_inside}, FPS={fps:.1f}")
                    prev_time = now_time
                    frame_count = 0

        # Cleanup
        self.reader.stop()
        self.mqtt_manager.close()
        if not HEADLESS:
            cv2.destroyAllWindows()

import time
import json
from collections import defaultdict, deque
from threading import Thread, Lock
from datetime import datetime

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ------------------------
# CONFIG
# ------------------------
MODEL_WEIGHTS = "yolov8s.pt"     # change to yolov8m.pt or yolov8n.pt as needed
TRACKER_YAML = "botsort.yaml"    # ensure this file is available in ultralytics trackers folder
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

LINE_POSITION = 0.5              # fraction of frame width (0..1) where vertical line is drawn
RESIZE_TO = (960, 540)           # (w, h) or None to keep original (resizing improves fps)
SMOOTH_ALPHA = 0.3               # new_position = old*(1-alpha) + new*alpha  (alpha in [0,1])
HISTORY_MAX = 80                 # max stored positions per track id
MIN_HISTORY_TO_COUNT = 3         # min history length to consider counting logic
DEBOUNCE_SEC = 0.6               # minimal seconds between counts for same id
RECOUNT_TIMEOUT_SEC = 10.0       # after this seconds, same ID may be counted again (useful if person leaves and returns)
MIN_MOVEMENT_PIXELS = 15         # minimum movement across frames to consider real crossing

# Visual / UI
INFO_PANEL_POS = (10, 10)        # top-left of info panel
WINDOW_NAME = "People Counter - Vertical Line"

# RTSP / RTMP source (replace with yours)
VIDEO_SOURCE = "rtsp://admin:password1@192.168.1.66:554/Streaming/Channels/101"

# MQTT Configuration
MQTT_CONFIG = {
    "broker": "206.237.97.19",
    "port": 1883,
    "username": "urbansolv",  
    "password": "letsgosolv",  
    "topic": "entrance/device-1/data",
    "client_id": "entrance_detection_client"
}


# ------------------------
# MQTT HELPER
# ------------------------
class MQTTManager:
    """
    Manages MQTT connection and publishing people count data.
    """
    def __init__(self, config):
        self.config = config
        self.client = None
        self.connected = False
        self._connect()

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            print("MQTT connected successfully.")
        else:
            self.connected = False
            print(f"MQTT connection failed with code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        self.connected = False
        print(f"MQTT disconnected with code: {rc}")
        if rc != 0:
            print("Unexpected disconnection, will attempt to reconnect...")

    def _connect(self):
        """Establish MQTT connection."""
        try:
            self.client = mqtt.Client(client_id=self.config["client_id"])
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Set authentication if provided
            if self.config["username"] and self.config["password"]:
                self.client.username_pw_set(
                    self.config["username"],
                    self.config["password"]
                )
            
            self.client.connect(
                self.config["broker"],
                self.config["port"],
                keepalive=60
            )
            self.client.loop_start()  # Start background thread for network loop
            print(f"Connecting to MQTT broker at {self.config['broker']}:{self.config['port']}...")
        except Exception as e:
            print(f"MQTT connection error: {e}")
            self.client = None
            self.connected = False

    def publish_count(self, people_inside):
        """Publish the current count to MQTT topic."""
        if self.client is None or not self.connected:
            print("MQTT not connected, skipping publish.")
            return False
        try:
            payload = {
                "occupancy": people_inside
            }
            result = self.client.publish(
                self.config["topic"],
                json.dumps(payload),
                qos=1  # At least once delivery
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] Published: timestamp={payload['timestamp']}, inside={people_inside}")
                return True
            else:
                print(f"MQTT publish failed with code: {result.rc}")
                return False
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
            return False

    def close(self):
        """Close MQTT connection."""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            print("MQTT connection closed.")
        except Exception as e:
            print(f"Error closing MQTT: {e}")


# ------------------------
# RTMP THREAD READER
# ------------------------
class RTMPReader:
    """
    Threaded reader that continuously reads frames from the RTMP/RTSP source
    and stores only the latest frame (drop older ones). This reduces perceived jitter.
    """
    def __init__(self, src, use_ffmpeg=True):
        self.src = src
        self.use_ffmpeg = use_ffmpeg
        self.cap = None
        self.stopped = False
        self.ret = False
        self.frame = None
        self.frame_ts = 0.0
        self.lock = Lock()
        self._connect()
        self.thread = Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        # Prefer FFMPEG backend when available
        if self.use_ffmpeg:
            try:
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            except Exception:
                self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            # cap not open — leave it to update loop to retry
            print("RTMPReader: initial connection failed (will retry)...")
            return

        # try to minimize internal buffering if supported
        try:
            # buffer size - keep as small as possible
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # optional: hint fps if known (may help)
        try:
            self.cap.set(cv2.CAP_PROP_FPS, 25)
        except Exception:
            pass

    def _reconnect(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
        finally:
            self.cap = None
        # small delay before reconnect
        time.sleep(0.8)
        self._connect()

    def _update_loop(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                continue

            # read frames as fast as possible and keep only latest
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # failed read — try reconnect
                # print("RTMPReader: read failed, reconnecting...")
                try:
                    # small sleep to avoid busy loop
                    time.sleep(0.2)
                except Exception:
                    pass
                self._reconnect()
                continue

            with self.lock:
                # store latest frame and timestamp
                self.ret = True
                # store a copy to avoid race with OpenCV underlying buffer
                self.frame = frame.copy()
                self.frame_ts = time.time()

            # tiny sleep to yield - tune if needed
            # time.sleep(0.001)

    def read(self):
        """Return (ret, frame, ts). frame is a copy (or None)."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None, 0.0
            # return a copy so caller can draw on it freely
            return True, self.frame.copy(), self.frame_ts

    def stop(self):
        self.stopped = True
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


# ------------------------
# PEOPLE COUNTER CLASS
# ------------------------
class PeopleCounter:
    def __init__(self, source, *,
                 model_weights=MODEL_WEIGHTS,
                 tracker_yaml=TRACKER_YAML,
                 conf=CONF_THRESHOLD,
                 iou=IOU_THRESHOLD,
                 resize_to=RESIZE_TO):
        self.source = source
        self.model_weights = model_weights
        self.tracker_yaml = tracker_yaml
        self.conf = conf
        self.iou = iou
        self.resize_to = resize_to

        # model
        print("Loading YOLO model...")
        self.model = YOLO(self.model_weights)

        # tracking & counting state
        self.track_history = defaultdict(lambda: deque(maxlen=HISTORY_MAX))  # track_id -> deque of (cx, cy)
        self.last_count_time = {}   # track_id -> timestamp of last crossing
        self.last_count_type = {}   # track_id -> 'in' or 'out' (last action)
        self.people_in = 0
        self.people_out = 0
        self.people_inside = 0

        # replace cv2 capture with threaded reader
        self.reader = RTMPReader(self.source, use_ffmpeg=True)

        # MQTT manager for publishing counts
        self.mqtt_manager = MQTTManager(MQTT_CONFIG)

        # runtime sizes (will set after first frame)
        self.rw = None
        self.rh = None

    def compute_mean_prev_x(self, hist_deque, n=3):
        # Compute mean of last n previous x's (excluding current if included)
        if len(hist_deque) < n + 1:
            # fallback: mean of all except last
            if len(hist_deque) >= 2:
                vals = [p[0] for p in list(hist_deque)[:-1]]
                return float(np.mean(vals))
            return None
        vals = [p[0] for p in list(hist_deque)[- (n + 1):-1]]
        return float(np.mean(vals))

    def should_count(self, track_id, prev_x, curr_x, line_x):
        """
        Decide whether crossing happened (given prev_x <-> curr_x against line_x)
        Returns: 'in', 'out' or None
        """
        now = time.time()

        # debounce: prevent counting too frequently for same track_id
        last = self.last_count_time.get(track_id, 0.0)
        if now - last < DEBOUNCE_SEC:
            return None

        # detect left->right (MASUK) and right->left (KELUAR)
        if prev_x is None:
            return None

        # require some minimal movement magnitude to ignore jitter
        if abs(curr_x - prev_x) < MIN_MOVEMENT_PIXELS:
            return None

        # MASUK: prev < line && curr >= line
        if prev_x < line_x <= curr_x:
            # avoid duplicate same type counting immediately
            if self.last_count_type.get(track_id) == "in" and now - last < RECOUNT_TIMEOUT_SEC:
                return None
            self.last_count_time[track_id] = now
            self.last_count_type[track_id] = "in"
            return "in"

        # KELUAR: prev > line && curr <= line
        if prev_x > line_x >= curr_x:
            if self.last_count_type.get(track_id) == "out" and now - last < RECOUNT_TIMEOUT_SEC:
                return None
            self.last_count_time[track_id] = now
            self.last_count_type[track_id] = "out"
            return "out"

        return None

    def draw_info_panel(self, frame):
        # small translucent panel
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

    def run(self):
        # wait for first good frame
        print("Waiting for first frame...")
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
        print(f"Line vertical X = {line_x} (frame width {self.rw})")

        prev_time = time.time()

        while True:
            ret, frame, ts = self.reader.read()
            if not ret or frame is None:
                # no frame currently available -> small wait
                time.sleep(0.01)
                continue

            # resize for better perf (do this on the latest frame)
            if self.resize_to is not None:
                frame = cv2.resize(frame, (self.rw, self.rh))

            # YOLO tracking
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],        # person only
                    tracker=self.tracker_yaml,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False
                )
            except Exception as e:
                # if tracker yaml not found or tracker fails, fallback to plain detect (no ids)
                print("Tracker error:", e)
                results = self.model(frame)  # no tracking ids

            boxes = []
            ids = []
            if len(results) > 0 and getattr(results[0], "boxes", None) is not None:
                det = results[0].boxes
                try:
                    # convert to numpy (works if stored on CPU)
                    boxes = det.xyxy.cpu().numpy()
                    ids_attr = getattr(det, "id", None)
                    if ids_attr is not None:
                        ids = ids_attr.cpu().numpy().astype(int)
                    else:
                        ids = None
                except Exception:
                    # CPU/GPU tensor access issue -> try alternative access or skip ids
                    try:
                        boxes = np.array(det.xyxy)
                    except Exception:
                        boxes = np.array([])
                    ids = None

            # If no boxes, just draw line & panel & show
            if boxes is None or len(boxes) == 0:
                # draw vertical line and info
                cv2.line(frame, (line_x, 0), (line_x, self.rh), (0, 255, 0), 3)
                frame = self.draw_info_panel(frame)
                # FPS
                now_t = time.time()
                fps = 1.0 / (now_t - prev_time) if now_t - prev_time > 0 else 0.0
                prev_time = now_t
                cv2.putText(frame, f"FPS: {fps:.1f}", (self.rw - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # If ids are None (tracker failed producing IDs), still draw boxes but skip counting
            has_ids = ids is not None

            # iterate detections
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if has_ids:
                    track_id = int(ids[i])
                    # smoothing using previous last position (if exists)
                    if len(self.track_history[track_id]) > 0:
                        px, py = self.track_history[track_id][-1]
                        # smooth: small alpha to new, larger to previous => less jitter
                        cx = int(px * (1 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA)
                        cy = int(py * (1 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA)

                    # append to history
                    self.track_history[track_id].append((cx, cy))

                    # counting logic
                    if len(self.track_history[track_id]) >= MIN_HISTORY_TO_COUNT:
                        prev_x_mean = self.compute_mean_prev_x(self.track_history[track_id], n=2)
                        curr_x = cx
                        action = self.should_count(track_id, prev_x_mean, curr_x, line_x)
                        if action == "in":
                            self.people_in += 1
                            self.people_inside += 1
                            # print log
                            print(f"[MASUK] ID:{track_id}  inside={self.people_inside} (in_total={self.people_in})")
                            # Publish to MQTT
                            self.mqtt_manager.publish_count(self.people_inside)
                        elif action == "out":
                            self.people_out += 1
                            self.people_inside = max(0, self.people_inside - 1)
                            print(f"[KELUAR] ID:{track_id}  inside={self.people_inside} (out_total={self.people_out})")
                            # Publish to MQTT
                            self.mqtt_manager.publish_count(self.people_inside)

                    # draw trail
                    pts = np.array(self.track_history[track_id], dtype=np.int32)
                    if len(pts) > 1:
                        cv2.polylines(frame, [pts], False, (255, 255, 0), 2)

                    # draw id and bbox (color by side)
                    color = (0, 255, 0) if cx < line_x else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                else:
                    # No stable IDs -> draw boxes and centroids but cannot count reliably
                    color = (200, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, "NO_ID", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(frame, (cx, cy), 3, color, -1)

            # cleanup old track histories to avoid memory growth
            stale_ids = []
            for tid, hist in list(self.track_history.items()):
                if len(hist) == 0:
                    stale_ids.append(tid)
            for tid in stale_ids:
                try:
                    del self.track_history[tid]
                    self.last_count_time.pop(tid, None)
                    self.last_count_type.pop(tid, None)
                except KeyError:
                    pass

            # draw vertical line and info
            cv2.line(frame, (line_x, 0), (line_x, self.rh), (0, 255, 0), 3)
            frame = self.draw_info_panel(frame)

            # FPS display
            now_time = time.time()
            fps = 1.0 / (now_time - prev_time) if now_time - prev_time > 0 else 0.0
            prev_time = now_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (self.rw - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Reset counters and histories
                print("Reset counters & history")
                self.people_in = 0
                self.people_out = 0
                self.people_inside = 0
                self.track_history.clear()
                self.last_count_time.clear()
                self.last_count_type.clear()

        # cleanup
        try:
            self.reader.stop()
        except Exception:
            pass
        try:
            self.mqtt_manager.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


# ------------------------
# ENTRY
# ------------------------
if __name__ == "__main__":
    pc = PeopleCounter(VIDEO_SOURCE,
                       model_weights=MODEL_WEIGHTS,
                       tracker_yaml=TRACKER_YAML,
                       conf=CONF_THRESHOLD,
                       iou=IOU_THRESHOLD,
                       resize_to=RESIZE_TO)
    try:
        pc.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print("Fatal error:", e)

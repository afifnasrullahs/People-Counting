"""
YOLOv8 Detector Optimized for Raspberry Pi 4.
Uses YOLOv8n with NCNN/ONNX backend for maximum performance.
Target: 10+ FPS on RPi4 4GB.
"""
import numpy as np
import cv2
from pathlib import Path
from threading import Thread, Lock
import time
from ..utils.logger import get_logger
from .sort import Sort

logger = get_logger(__name__)

PERSON_CLASS_ID = 0


class YOLOv8DetectorRPi:
    """
    YOLOv8 detector optimized for Raspberry Pi.
    Uses YOLOv8n with reduced input size for maximum FPS.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.4, 
                 input_size: int = 320, **kwargs):
        """
        Initialize optimized YOLOv8 detector for RPi.
        
        Args:
            model_path: Path to YOLOv8 model file
            conf: Confidence threshold
            input_size: Model input size (320 recommended for RPi)
        """
        self.conf = conf
        self.input_size = input_size
        self.model = None
        
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError("ultralytics not found. Install with: pip install ultralytics")
        
        # Use YOLOv8n (nano) for RPi - much faster than yolov8s
        from ..config import MODELS_DIR
        
        if model_path and Path(model_path).exists():
            self.model_path = Path(model_path)
        else:
            # Prefer NCNN format for ARM, then ONNX, then PT
            ncnn_path = MODELS_DIR / "yolov8n_ncnn_model"
            onnx_path = MODELS_DIR / "yolov8n.onnx"
            pt_path = MODELS_DIR / "yolov8n.pt"
            
            if ncnn_path.exists():
                self.model_path = ncnn_path
                logger.info("Using NCNN model (fastest on ARM)")
            elif onnx_path.exists():
                self.model_path = onnx_path
                logger.info("Using ONNX model")
            elif pt_path.exists():
                self.model_path = pt_path
            else:
                # Download yolov8n
                logger.info("Downloading YOLOv8n model...")
                self.model_path = "yolov8n.pt"
        
        logger.info(f"Loading YOLOv8n model: {self.model_path}")
        
        # Load model
        self.model = self.YOLO(str(self.model_path))
        
        # Force CPU for Raspberry Pi
        self.device = "cpu"
        
        logger.info(f"YOLOv8n loaded. Input size: {input_size}x{input_size}")
        
        # Initialize SORT tracker with optimized parameters
        self.tracker = Sort(
            max_age=kwargs.get('max_age', 30),
            min_hits=kwargs.get('min_hits', 1),
            iou_threshold=kwargs.get('iou_threshold', 0.2)
        )
        
        # Inference timing
        self.last_inference_time = 0
        
        logger.info("RPi-optimized detector ready.")
    
    def detect_and_track(self, frame):
        """
        Detect and track persons with optimized settings for RPi.
        """
        h, w = frame.shape[:2]
        
        start_time = time.time()
        
        # Run inference with minimal settings for speed
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=0.5,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            imgsz=self.input_size,
            half=False,  # RPi doesn't support FP16 well on CPU
            max_det=20,  # Limit detections for speed
            verbose=False
        )
        
        self.last_inference_time = time.time() - start_time
        
        # Extract detections
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                
                x1, y1, x2, y2 = xyxy
                x1 = max(0, min(int(x1), w))
                y1 = max(0, min(int(y1), h))
                x2 = max(0, min(int(x2), w))
                y2 = max(0, min(int(y2), h))
                
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, float(conf)])
        
        # Convert to numpy
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Extract boxes and IDs
        boxes = []
        ids = []
        
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                x1 = max(0, min(int(x1), w))
                y1 = max(0, min(int(y1), h))
                x2 = max(0, min(int(x2), w))
                y2 = max(0, min(int(y2), h))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    ids.append(int(track_id))
        
        boxes = np.array(boxes) if boxes else np.array([])
        ids = np.array(ids) if ids else None
        
        return boxes, ids
    
    def reset_tracker(self):
        """Reset the SORT tracker state."""
        self.tracker.reset()


class YOLOv8DetectorThreaded:
    """
    Threaded YOLOv8 detector for better FPS.
    Runs detection in background thread while main thread handles display.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.4,
                 input_size: int = 320, **kwargs):
        """Initialize threaded detector."""
        self.detector = YOLOv8DetectorRPi(
            model_path=model_path,
            conf=conf,
            input_size=input_size,
            **kwargs
        )
        
        # Threading
        self.lock = Lock()
        self.frame = None
        self.boxes = np.array([])
        self.ids = None
        self.running = False
        self.thread = None
        
        # Skip detection every N frames but use tracking prediction
        self.detect_interval = kwargs.get('detect_interval', 2)
        self.frame_count = 0
        
    def start(self):
        """Start background detection thread."""
        self.running = True
        self.thread = Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        logger.info("Background detection thread started.")
        
    def stop(self):
        """Stop background detection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _detection_loop(self):
        """Background detection loop."""
        while self.running:
            with self.lock:
                frame = self.frame
            
            if frame is not None:
                boxes, ids = self.detector.detect_and_track(frame)
                
                with self.lock:
                    self.boxes = boxes
                    self.ids = ids
            else:
                time.sleep(0.01)
    
    def detect_and_track(self, frame):
        """
        Get detection results. Updates frame for background thread.
        """
        self.frame_count += 1
        
        # Update frame for background thread
        with self.lock:
            self.frame = frame.copy()
            boxes = self.boxes.copy() if len(self.boxes) > 0 else np.array([])
            ids = self.ids.copy() if self.ids is not None else None
        
        return boxes, ids
    
    def reset_tracker(self):
        """Reset tracker."""
        self.detector.reset_tracker()


class YOLOv8DetectorLite:
    """
    Ultra-lightweight detector using OpenCV DNN with YOLOv8n ONNX.
    Avoids ultralytics overhead for maximum speed on RPi.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.4,
                 input_size: int = 320, **kwargs):
        """
        Initialize lightweight ONNX detector.
        """
        self.conf = conf
        self.input_size = input_size
        self.net = None
        
        # Find ONNX model
        from ..config import MODELS_DIR
        
        if model_path and Path(model_path).exists():
            self.model_path = Path(model_path)
        else:
            self.model_path = MODELS_DIR / "yolov8n.onnx"
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}\n"
                "Export with: yolo export model=yolov8n.pt format=onnx imgsz=320"
            )
        
        logger.info(f"Loading ONNX model with OpenCV DNN: {self.model_path}")
        
        # Load with OpenCV DNN - faster than ONNX Runtime on some ARM devices
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        
        # Optimize for CPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Try to set number of threads
        cv2.setNumThreads(4)
        
        logger.info(f"OpenCV DNN model loaded. Input: {input_size}x{input_size}")
        
        # Initialize SORT tracker
        self.tracker = Sort(
            max_age=kwargs.get('max_age', 30),
            min_hits=kwargs.get('min_hits', 1),
            iou_threshold=kwargs.get('iou_threshold', 0.2)
        )
        
        self.last_inference_time = 0
    
    def detect_and_track(self, frame):
        """Detect and track using OpenCV DNN."""
        h, w = frame.shape[:2]
        
        start_time = time.time()
        
        # Preprocess
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        # Inference
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        self.last_inference_time = time.time() - start_time
        
        # Postprocess YOLOv8 output
        detections = self._postprocess(outputs, w, h)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Extract results
        boxes = []
        ids = []
        
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                x1 = max(0, min(int(x1), w))
                y1 = max(0, min(int(y1), h))
                x2 = max(0, min(int(x2), w))
                y2 = max(0, min(int(y2), h))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    ids.append(int(track_id))
        
        boxes = np.array(boxes) if boxes else np.array([])
        ids = np.array(ids) if ids else None
        
        return boxes, ids
    
    def _postprocess(self, outputs, orig_w, orig_h):
        """Postprocess YOLOv8 ONNX output."""
        # YOLOv8 output: [1, 84, 8400] -> transpose to [1, 8400, 84]
        output = outputs[0]
        if output.shape[1] == 84:
            output = np.transpose(output, (0, 2, 1))
        output = output[0]
        
        detections = []
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        for det in output:
            box = det[:4]
            class_scores = det[4:]
            
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Only person class with sufficient confidence
            if confidence > self.conf and class_id == PERSON_CLASS_ID:
                cx, cy, bw, bh = box
                
                # Scale to original image
                x1 = int((cx - bw/2) * scale_x)
                y1 = int((cy - bh/2) * scale_y)
                x2 = int((cx + bw/2) * scale_x)
                y2 = int((cy + bh/2) * scale_y)
                
                # Clip
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, float(confidence)])
        
        # NMS
        if len(detections) > 0:
            detections = np.array(detections)
            boxes = detections[:, :4]
            scores = detections[:, 4]
            
            boxes_xywh = [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes]
            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self.conf, 0.5)
            
            if len(indices) > 0:
                indices = indices.flatten()
                detections = detections[indices]
            else:
                detections = np.empty((0, 5))
        else:
            detections = np.empty((0, 5))
        
        return detections
    
    def reset_tracker(self):
        """Reset tracker."""
        self.tracker.reset()

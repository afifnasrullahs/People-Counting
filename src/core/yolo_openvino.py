"""
YOLOv8 Detector with OpenVINO for Raspberry Pi 4.
OpenVINO provides optimized inference for edge devices.
Target: 8-15 FPS on RPi4 4GB.
"""
import numpy as np
import cv2
from pathlib import Path
import time
from ..utils.logger import get_logger
from .sort import Sort

logger = get_logger(__name__)

PERSON_CLASS_ID = 0


class YOLOv8OpenVINO:
    """
    YOLOv8 detector using OpenVINO for optimized inference.
    Best performance on x86 but also works on ARM via ONNX.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.4, 
                 input_size: int = 256, **kwargs):
        """
        Initialize OpenVINO detector.
        
        Args:
            model_path: Path to OpenVINO model (.xml) or ONNX model
            conf: Confidence threshold
            input_size: Model input size (256 for speed, 320 for accuracy)
        """
        self.conf = conf
        self.input_size = input_size
        self.model = None
        self.compiled_model = None
        
        from ..config import MODELS_DIR
        
        # Try to find OpenVINO model, then ONNX
        if model_path and Path(model_path).exists():
            self.model_path = Path(model_path)
        else:
            openvino_path = MODELS_DIR / "yolov8n_openvino_model" / "yolov8n.xml"
            onnx_path = MODELS_DIR / "yolov8n.onnx"
            
            if openvino_path.exists():
                self.model_path = openvino_path
            elif onnx_path.exists():
                self.model_path = onnx_path
            else:
                raise FileNotFoundError(
                    f"Model not found. Please export:\n"
                    f"  yolo export model=yolov8n.pt format=openvino imgsz={input_size}\n"
                    f"Or: yolo export model=yolov8n.pt format=onnx imgsz={input_size}"
                )
        
        logger.info(f"Loading model: {self.model_path}")
        
        # Try OpenVINO first
        try:
            from openvino.runtime import Core
            self.core = Core()
            
            # Read model
            self.model = self.core.read_model(str(self.model_path))
            
            # Compile for CPU with optimizations
            config = {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "INFERENCE_NUM_THREADS": "4"
            }
            self.compiled_model = self.core.compile_model(self.model, "CPU", config)
            self.infer_request = self.compiled_model.create_infer_request()
            
            # Get input/output info
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            
            self.use_openvino = True
            logger.info("OpenVINO model loaded successfully")
            
        except ImportError:
            logger.warning("OpenVINO not available, falling back to OpenCV DNN")
            self.use_openvino = False
            self._init_opencv_dnn()
        except Exception as e:
            logger.warning(f"OpenVINO failed: {e}, falling back to OpenCV DNN")
            self.use_openvino = False
            self._init_opencv_dnn()
        
        # Initialize SORT tracker - optimized for stability
        self.tracker = Sort(
            max_age=kwargs.get('max_age', 45),  # Keep tracks longer
            min_hits=kwargs.get('min_hits', 1),  # Show immediately
            iou_threshold=kwargs.get('iou_threshold', 0.25)
        )
        
        # Cache for tracking when detection skipped
        self.last_detections = np.empty((0, 5))
        self.detection_count = 0
        self.skip_detection = kwargs.get('skip_detection', 2)  # Detect every N frames
        
        self.last_inference_time = 0
        logger.info(f"Detector ready. Input: {input_size}x{input_size}, Skip: {self.skip_detection}")
    
    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN backend as fallback."""
        from ..config import MODELS_DIR
        onnx_path = MODELS_DIR / "yolov8n.onnx"
        
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model required: {onnx_path}")
        
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        cv2.setNumThreads(4)
        logger.info("OpenCV DNN backend initialized")
    
    def _preprocess(self, frame):
        """Preprocess frame for inference."""
        # Resize with letterbox
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2
        
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert to blob
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC to CHW
        blob = np.expand_dims(blob, 0)  # Add batch dim
        
        return blob, scale, pad_w, pad_h
    
    def _postprocess(self, output, orig_w, orig_h, scale, pad_w, pad_h):
        """Postprocess model output."""
        # YOLOv8 output: [1, 84, N] -> transpose to [1, N, 84]
        if output.shape[1] == 84:
            output = np.transpose(output, (0, 2, 1))
        output = output[0]  # Remove batch
        
        detections = []
        
        for det in output:
            box = det[:4]
            class_scores = det[4:]
            
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > self.conf and class_id == PERSON_CLASS_ID:
                cx, cy, bw, bh = box
                
                # Remove padding and scale
                x1 = (cx - bw/2 - pad_w) / scale
                y1 = (cy - bh/2 - pad_h) / scale
                x2 = (cx + bw/2 - pad_w) / scale
                y2 = (cy + bh/2 - pad_h) / scale
                
                # Clip
                x1 = max(0, min(int(x1), orig_w))
                y1 = max(0, min(int(y1), orig_h))
                x2 = max(0, min(int(x2), orig_w))
                y2 = max(0, min(int(y2), orig_h))
                
                if x2 > x1 + 10 and y2 > y1 + 10:  # Min box size
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
                return detections[indices]
        
        return np.empty((0, 5))
    
    def _detect(self, frame):
        """Run detection on frame."""
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(frame)
        
        start = time.time()
        
        if self.use_openvino:
            # OpenVINO inference
            self.infer_request.infer({self.input_layer.any_name: blob})
            output = self.infer_request.get_output_tensor(0).data
        else:
            # OpenCV DNN inference
            # Need BGR and different preprocessing
            blob_cv = cv2.dnn.blobFromImage(
                frame, 1/255.0, (self.input_size, self.input_size),
                (0, 0, 0), swapRB=True, crop=False
            )
            self.net.setInput(blob_cv)
            output = self.net.forward()
        
        self.last_inference_time = time.time() - start
        
        return self._postprocess(output, w, h, scale, pad_w, pad_h)
    
    def detect_and_track(self, frame):
        """
        Detect and track with frame skipping for speed.
        Detection runs every N frames, tracking runs every frame.
        """
        h, w = frame.shape[:2]
        
        self.detection_count += 1
        
        # Run detection periodically or use cached
        if self.detection_count % self.skip_detection == 0 or len(self.last_detections) == 0:
            detections = self._detect(frame)
            self.last_detections = detections
        else:
            detections = self.last_detections
        
        # Always update tracker (even with old detections for prediction)
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
    
    def reset_tracker(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.last_detections = np.empty((0, 5))
        self.detection_count = 0


class YOLOv8UltralyticsLite:
    """
    Lightweight YOLOv8 using ultralytics with aggressive optimizations.
    Simpler than OpenVINO but more compatible.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.4,
                 input_size: int = 256, **kwargs):
        """Initialize lightweight detector."""
        self.conf = conf
        self.input_size = input_size
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics required: pip install ultralytics")
        
        from ..config import MODELS_DIR
        
        # Find model - prefer exported formats
        if model_path and Path(model_path).exists():
            self.model_path = Path(model_path)
        else:
            # Check for optimized formats first
            ncnn_path = MODELS_DIR / "yolov8n_ncnn_model"
            onnx_path = MODELS_DIR / "yolov8n.onnx" 
            pt_path = MODELS_DIR / "yolov8n.pt"
            
            if ncnn_path.exists():
                self.model_path = ncnn_path
            elif onnx_path.exists():
                self.model_path = onnx_path
            elif pt_path.exists():
                self.model_path = pt_path
            else:
                self.model_path = "yolov8n.pt"
        
        logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # SORT tracker - stable settings
        self.tracker = Sort(
            max_age=kwargs.get('max_age', 45),
            min_hits=kwargs.get('min_hits', 1),
            iou_threshold=kwargs.get('iou_threshold', 0.25)
        )
        
        # Frame skipping
        self.skip_detection = kwargs.get('skip_detection', 2)
        self.detection_count = 0
        self.last_detections = np.empty((0, 5))
        self.last_inference_time = 0
        
        logger.info(f"Detector ready. Size: {input_size}, Skip: {self.skip_detection}")
    
    def detect_and_track(self, frame):
        """Detect and track with frame skipping."""
        h, w = frame.shape[:2]
        self.detection_count += 1
        
        # Run detection periodically
        if self.detection_count % self.skip_detection == 0 or len(self.last_detections) == 0:
            start = time.time()
            
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=0.5,
                classes=[PERSON_CLASS_ID],
                device="cpu",
                imgsz=self.input_size,
                half=False,
                max_det=15,
                verbose=False
            )
            
            self.last_inference_time = time.time() - start
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = map(int, xyxy)
                    if x2 > x1 + 10 and y2 > y1 + 10:
                        detections.append([x1, y1, x2, y2, conf])
            
            self.last_detections = np.array(detections) if detections else np.empty((0, 5))
        
        # Update tracker
        tracks = self.tracker.update(self.last_detections)
        
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
        
        return np.array(boxes) if boxes else np.array([]), np.array(ids) if ids else None
    
    def reset_tracker(self):
        """Reset tracker."""
        self.tracker.reset()
        self.last_detections = np.empty((0, 5))
        self.detection_count = 0

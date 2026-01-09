"""
YOLOv8 Person Detector with SORT Tracking.
Optimized for entrance detection applications.
"""
import numpy as np
import cv2
from pathlib import Path
from ..utils.logger import get_logger
from .sort import Sort

logger = get_logger(__name__)

# COCO class ID for person
PERSON_CLASS_ID = 0


class YOLOv8Detector:
    """
    YOLOv8 based person detector with SORT tracking.
    Uses ultralytics library for inference.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.5, 
                 device: str = "auto", **kwargs):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to YOLOv8 model file (.pt)
            conf: Confidence threshold
            device: Device to run inference on ("cpu", "cuda", "auto")
        """
        self.conf = conf
        self.model = None
        
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not found. Install with: pip install ultralytics"
            )
        
        # Find or download model
        if model_path and Path(model_path).exists():
            self.model_path = model_path
        else:
            # Look for model in models directory
            from ..config import MODELS_DIR
            self.model_path = MODELS_DIR / "yolov8s.pt"
            
            if not Path(self.model_path).exists():
                logger.info("YOLOv8s model not found, downloading...")
                # ultralytics will automatically download the model
                self.model_path = "yolov8s.pt"
        
        logger.info(f"Loading YOLOv8 model: {self.model_path}")
        
        # Load model
        self.model = self.YOLO(str(self.model_path))
        
        # Set device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"YOLOv8 model loaded. Device: {self.device}")
        
        # NMS threshold for handling crowded scenes (higher = more overlapping boxes kept)
        self.nms_threshold = kwargs.get('nms_threshold', 0.6)
        
        # Initialize SORT tracker optimized for crowded scenes and stability
        # - min_hits=1: Show tracks immediately to prevent flickering
        # - max_age=50: Keep tracks alive longer during occlusion
        # - iou_threshold=0.15: Lower threshold for tracking close people separately
        self.tracker = Sort(
            max_age=kwargs.get('max_age', 50),
            min_hits=kwargs.get('min_hits', 1),  # Immediate track display
            iou_threshold=kwargs.get('iou_threshold', 0.15)  # Lower for crowded scenes
        )
        
        logger.info("SORT tracker initialized (optimized for crowded scenes).")
    
    def detect_and_track(self, frame):
        """
        Detect and track persons in frame using YOLOv8 + SORT.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (boxes, ids) where boxes is Nx4 array and ids is N array
        """
        h, w = frame.shape[:2]
        
        # Run YOLOv8 inference with optimized settings for crowded scenes
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.nms_threshold,  # NMS IOU threshold - higher keeps more overlapping boxes
            classes=[PERSON_CLASS_ID],  # Only detect persons
            device=self.device,
            agnostic_nms=False,  # Class-specific NMS
            max_det=100,  # Allow more detections
            verbose=False
        )
        
        # Extract detections
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                
                x1, y1, x2, y2 = xyxy
                
                # Clip to frame boundaries
                x1 = max(0, min(int(x1), w))
                y1 = max(0, min(int(y1), h))
                x2 = max(0, min(int(x2), w))
                y2 = max(0, min(int(y2), h))
                
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, float(conf)])
        
        # Convert to numpy array
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))
        
        # Update SORT tracker
        tracks = self.tracker.update(detections)
        
        # Extract boxes and IDs
        boxes = []
        ids = []
        
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                
                # Clip coordinates
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
        logger.info("SORT tracker reset.")


class YOLOv8DetectorONNX:
    """
    YOLOv8 detector using ONNX runtime for deployment without ultralytics.
    Better for edge devices like Raspberry Pi.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.5, 
                 input_size: tuple = (640, 640), **kwargs):
        """
        Initialize YOLOv8 ONNX detector.
        
        Args:
            model_path: Path to ONNX model file
            conf: Confidence threshold
            input_size: Model input size
        """
        self.conf = conf
        self.input_size = input_size
        self.session = None
        
        # Try to import onnxruntime
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime not found. Install with: pip install onnxruntime"
            )
        
        # Find model
        if model_path and Path(model_path).exists():
            self.model_path = model_path
        else:
            from ..config import MODELS_DIR
            self.model_path = MODELS_DIR / "yolov8s.onnx"
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {self.model_path}. "
                    "Export with: yolo export model=yolov8s.pt format=onnx"
                )
        
        logger.info(f"Loading YOLOv8 ONNX model: {self.model_path}")
        
        # Create ONNX session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[2] is not None and input_shape[3] is not None:
            self.input_size = (input_shape[2], input_shape[3])
        
        logger.info(f"ONNX model loaded. Input size: {self.input_size}")
        
        # NMS threshold for crowded scenes (higher = more overlapping boxes kept)
        self.nms_threshold = kwargs.get('nms_threshold', 0.6)
        
        # Initialize SORT tracker optimized for crowded scenes
        self.tracker = Sort(
            max_age=kwargs.get('max_age', 50),
            min_hits=kwargs.get('min_hits', 1),
            iou_threshold=kwargs.get('iou_threshold', 0.15)
        )
    
    def preprocess(self, frame):
        """Preprocess frame for YOLOv8 ONNX inference."""
        h, w = frame.shape[:2]
        input_h, input_w = self.input_size
        
        # Calculate scale
        scale = min(input_h / h, input_w / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to input size
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_h = (input_h - new_h) // 2
        pad_w = (input_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # HWC to CHW and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale, pad_w, pad_h
    
    def postprocess(self, outputs, scale, pad_w, pad_h, orig_h, orig_w):
        """Postprocess YOLOv8 ONNX output."""
        # YOLOv8 output shape: [1, 84, 8400] (for COCO: 80 classes + 4 box coords)
        output = outputs[0]
        
        if output.shape[1] == 84:
            output = np.transpose(output, (0, 2, 1))  # [1, 8400, 84]
        
        output = output[0]  # Remove batch dimension
        
        detections = []
        
        for detection in output:
            # Get box and class scores
            box = detection[:4]
            class_scores = detection[4:]
            
            # Get max class score and ID
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Filter by confidence and person class
            if confidence > self.conf and class_id == PERSON_CLASS_ID:
                # Convert from center format to corner format
                cx, cy, w, h = box
                
                # Remove padding and scale back
                x1 = (cx - w / 2 - pad_w) / scale
                y1 = (cy - h / 2 - pad_h) / scale
                x2 = (cx + w / 2 - pad_w) / scale
                y2 = (cy + h / 2 - pad_h) / scale
                
                # Clip to frame boundaries
                x1 = max(0, min(int(x1), orig_w))
                y1 = max(0, min(int(y1), orig_h))
                x2 = max(0, min(int(x2), orig_w))
                y2 = max(0, min(int(y2), orig_h))
                
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, float(confidence)])
        
        # Apply NMS with higher threshold for crowded scenes
        if len(detections) > 0:
            detections = np.array(detections)
            boxes = detections[:, :4]
            scores = detections[:, 4]
            
            # Convert to xywh for NMS
            boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self.conf, self.nms_threshold)
            
            if len(indices) > 0:
                indices = indices.flatten()
                detections = detections[indices]
            else:
                detections = np.empty((0, 5))
        else:
            detections = np.empty((0, 5))
        
        return detections
    
    def detect_and_track(self, frame):
        """
        Detect and track persons in frame using YOLOv8 ONNX + SORT.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (boxes, ids)
        """
        h, w = frame.shape[:2]
        
        # Preprocess
        input_tensor, scale, pad_w, pad_h = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, scale, pad_w, pad_h, h, w)
        
        # Update SORT tracker
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

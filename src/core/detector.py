"""
Person detector with simple tracking - Optimized for Raspberry Pi.
Supports multiple backends: TFLite (fastest), MobileNet-SSD, HOG.
"""
import numpy as np
import cv2
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Person class ID in VOC dataset (for MobileNet-SSD)
PERSON_CLASS_ID = 15


class SimpleTracker:
    """
    Simple centroid-based tracker for maintaining object IDs across frames.
    """
    
    def __init__(self, max_disappeared=15, max_distance=80):
        """
        Initialize tracker.
        
        Args:
            max_disappeared: Max frames an object can be missing before deregistering
            max_distance: Max distance to consider same object
        """
        self.next_id = 0
        self.objects = {}  # id -> centroid
        self.disappeared = {}  # id -> count of disappeared frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new object with the next available ID."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id):
        """Deregister an object ID."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, centroids):
        """
        Update tracker with new centroids.
        
        Args:
            centroids: List of (cx, cy) tuples
            
        Returns:
            Dict mapping object IDs to centroids
        """
        # If no centroids, mark all existing objects as disappeared
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # If no existing objects, register all centroids
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
            return self.objects
        
        # Match existing objects to new centroids using distance
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = np.zeros((len(object_centroids), len(centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, centroid in enumerate(centroids):
                D[i, j] = np.sqrt((obj_centroid[0] - centroid[0])**2 + 
                                  (obj_centroid[1] - centroid[1])**2)
        
        # Find minimum distance assignments
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            
            object_id = object_ids[row]
            self.objects[object_id] = centroids[col]
            self.disappeared[object_id] = 0
            
            used_rows.add(row)
            used_cols.add(col)
        
        # Handle unmatched existing objects
        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # Register new centroids that weren't matched
        unused_cols = set(range(len(centroids))) - used_cols
        for col in unused_cols:
            self.register(centroids[col])
        
        return self.objects


class HOGDetector:
    """
    HOG (Histogram of Oriented Gradients) based person detector.
    Much faster than deep learning models, ideal for Raspberry Pi.
    """
    
    def __init__(self, conf: float = 0.0, nms_threshold: float = 0.3, 
                 scale: float = 1.03, **kwargs):
        """
        Initialize HOG detector.
        
        Args:
            conf: Confidence threshold (weight threshold) - lower = more detections
            nms_threshold: Non-maximum suppression threshold
            scale: Scale factor for multi-scale detection (lower = more scales, slower)
        """
        self.conf = conf  # HOG weights can be negative, so 0.0 is reasonable
        self.nms_threshold = nms_threshold
        self.scale = scale
        
        logger.info("Loading HOG person detector (optimized for Raspberry Pi)...")
        
        # Initialize HOG detector with default people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize simple tracker
        self.tracker = SimpleTracker(max_disappeared=15, max_distance=80)
        
        logger.info("HOG detector loaded successfully.")

    def detect_and_track(self, frame):
        """
        Detect and track persons in frame using HOG.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (boxes, ids) where boxes is Nx4 array and ids is N array or None
        """
        h, w = frame.shape[:2]
        
        # HOG needs minimum resolution - don't downscale too much
        # Minimum window size is 64x128, so frame should be at least 400px wide
        min_width = 400
        if w < min_width:
            scale_factor = min_width / w
            process_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            scale_factor = 1.0
            process_frame = frame
        
        ph, pw = process_frame.shape[:2]
        
        # Detect people with adjusted parameters for better detection
        rects, weights = self.hog.detectMultiScale(
            process_frame,
            winStride=(4, 4),      # Smaller stride = more detections, slower
            padding=(8, 8),        # More padding helps edge detection
            scale=self.scale,
            useMeanshiftGrouping=False
        )
        
        boxes = []
        centroids = []
        
        for i, (x, y, bw, bh) in enumerate(rects):
            if weights[i] > self.conf:
                # Scale back to original size
                x1 = int(x / scale_factor)
                y1 = int(y / scale_factor)
                x2 = int((x + bw) / scale_factor)
                y2 = int((y + bh) / scale_factor)
                
                # Clip to frame boundaries
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    centroids.append((cx, cy))
        
        # Apply NMS
        if len(boxes) > 0:
            boxes_for_nms = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
            indices = cv2.dnn.NMSBoxes(boxes_for_nms, [1.0] * len(boxes), self.conf, self.nms_threshold)
            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]
                boxes = [boxes[i] for i in indices]
                centroids = [centroids[i] for i in indices]
        
        boxes = np.array(boxes) if boxes else np.array([])
        
        # Update tracker
        tracked_objects = self.tracker.update(centroids)
        
        # Match boxes to tracked IDs
        ids = None
        if len(boxes) > 0 and len(tracked_objects) > 0:
            ids = []
            for box in boxes:
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                
                min_dist = float('inf')
                matched_id = -1
                for obj_id, obj_centroid in tracked_objects.items():
                    dist = np.sqrt((cx - obj_centroid[0])**2 + (cy - obj_centroid[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = obj_id
                
                ids.append(matched_id)
            
            ids = np.array(ids)
        
        return boxes, ids


class PersonDetector:
    """
    MobileNet-SSD based person detector with simple tracking support.
    Optimized for Raspberry Pi with threading.
    """
    
    def __init__(self, model_config: str, model_weights: str, conf: float = 0.5, 
                 input_size: tuple = (300, 300), **kwargs):
        """
        Initialize detector.
        
        Args:
            model_config: Path to MobileNet-SSD config (.prototxt or .pbtxt)
            model_weights: Path to MobileNet-SSD weights (.caffemodel or .pb)
            conf: Confidence threshold
            input_size: Input size for the network (width, height)
        """
        self.model_config = model_config
        self.model_weights = model_weights
        self.conf = conf
        self.input_size = input_size
        
        logger.info(f"Loading MobileNet-SSD model...")
        logger.info(f"  Config: {model_config}")
        logger.info(f"  Weights: {model_weights}")
        logger.info(f"  Input size: {input_size}")
        
        # Load model based on file extension
        if str(model_weights).endswith('.caffemodel'):
            self.net = cv2.dnn.readNetFromCaffe(str(model_config), str(model_weights))
        elif str(model_weights).endswith('.pb'):
            self.net = cv2.dnn.readNetFromTensorflow(str(model_weights), str(model_config))
        else:
            # Try generic approach
            self.net = cv2.dnn.readNet(str(model_weights), str(model_config))
        
        # Optimize for Raspberry Pi - use all available threads
        cv2.setNumThreads(4)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Initialize simple tracker
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
        
        logger.info("MobileNet-SSD model loaded successfully.")

    def detect_and_track(self, frame):
        """
        Detect and track persons in frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (boxes, ids) where boxes is Nx4 array and ids is N array or None
        """
        h, w = frame.shape[:2]
        
        # Prepare input blob - use smaller input for speed
        blob = cv2.dnn.blobFromImage(
            frame, 
            scalefactor=0.007843,  # 1/127.5
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        centroids = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf:
                class_id = int(detections[0, 0, i, 1])
                
                # Only detect persons (class_id = 15 in PASCAL VOC)
                if class_id == PERSON_CLASS_ID:
                    # Get bounding box coordinates
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Clip to frame boundaries
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        centroids.append((cx, cy))
        
        boxes = np.array(boxes) if boxes else np.array([])
        
        # Update tracker
        tracked_objects = self.tracker.update(centroids)
        
        # Match boxes to tracked IDs
        ids = None
        if len(boxes) > 0 and len(tracked_objects) > 0:
            ids = []
            for box in boxes:
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                
                # Find closest tracked object
                min_dist = float('inf')
                matched_id = -1
                for obj_id, obj_centroid in tracked_objects.items():
                    dist = np.sqrt((cx - obj_centroid[0])**2 + (cy - obj_centroid[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = obj_id
                
                ids.append(matched_id)
            
            ids = np.array(ids)
        
        return boxes, ids


class TFLiteDetector:
    """
    TensorFlow Lite based person detector - optimized for Raspberry Pi.
    Much faster than OpenCV DNN on ARM processors (8-15 FPS on RPi4).
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.5, 
                 num_threads: int = 4, **kwargs):
        """
        Initialize TFLite detector.
        
        Args:
            model_path: Path to TFLite model file
            conf: Confidence threshold
            num_threads: Number of threads for inference
        """
        self.conf = conf
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_size = (300, 300)
        
        # Try to import TFLite
        try:
            try:
                from tflite_runtime.interpreter import Interpreter
                self.Interpreter = Interpreter
                logger.info("Using tflite_runtime")
            except ImportError:
                import tensorflow as tf
                self.Interpreter = tf.lite.Interpreter
                logger.info("Using TensorFlow Lite")
        except ImportError:
            raise ImportError("TensorFlow Lite not found. Install with: pip install tflite-runtime")
        
        # Find or download model
        if model_path and Path(model_path).exists():
            self.model_path = model_path
        else:
            # Look for model in models directory
            from ..config import MODELS_DIR
            self.model_path = MODELS_DIR / "ssd_mobilenet_v2_coco.tflite"
            
        if not Path(self.model_path).exists():
            logger.warning(f"TFLite model not found at {self.model_path}")
            logger.warning("Please download: python download_tflite_model.py")
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        
        logger.info(f"Loading TFLite model: {self.model_path}")
        
        # Load model
        self.interpreter = self.Interpreter(
            model_path=str(self.model_path),
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input size from model
        input_shape = self.input_details[0]['shape']
        self.input_size = (input_shape[1], input_shape[2])
        
        logger.info(f"TFLite model loaded. Input size: {self.input_size}")
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)

    def detect_and_track(self, frame):
        """
        Detect and track persons in frame using TFLite.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (boxes, ids)
        """
        h, w = frame.shape[:2]
        
        # Preprocess
        input_h, input_w = self.input_size
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (input_w, input_h))
        
        # Check if model expects float or uint8
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.float32:
            input_data = (resized.astype(np.float32) - 127.5) / 127.5
        else:
            input_data = resized.astype(np.uint8)
        
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs (format depends on model)
        # Standard SSD MobileNet V2 output format
        boxes_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes_output = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores_output = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        boxes = []
        centroids = []
        
        for i in range(len(scores_output)):
            if scores_output[i] > self.conf:
                class_id = int(classes_output[i])
                
                # COCO person class is 0, sometimes 1
                if class_id in [0, 1]:
                    # boxes are in [ymin, xmin, ymax, xmax] format normalized
                    ymin, xmin, ymax, xmax = boxes_output[i]
                    x1 = int(xmin * w)
                    y1 = int(ymin * h)
                    x2 = int(xmax * w)
                    y2 = int(ymax * h)
                    
                    # Clip
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        centroids.append((cx, cy))
        
        boxes = np.array(boxes) if boxes else np.array([])
        
        # Update tracker
        tracked_objects = self.tracker.update(centroids)
        
        # Match boxes to tracked IDs
        ids = None
        if len(boxes) > 0 and len(tracked_objects) > 0:
            ids = []
            for box in boxes:
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                
                min_dist = float('inf')
                matched_id = -1
                for obj_id, obj_centroid in tracked_objects.items():
                    dist = np.sqrt((cx - obj_centroid[0])**2 + (cy - obj_centroid[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = obj_id
                
                ids.append(matched_id)
            
            ids = np.array(ids)
        
        return boxes, ids
        
        return boxes, ids

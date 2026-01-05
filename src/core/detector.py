"""
MobileNet-SSD based person detector with simple tracking.
"""
import numpy as np
import cv2
from ..utils.logger import get_logger

logger = get_logger(__name__)

# PASCAL VOC class names (MobileNet-SSD trained on VOC)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Person class ID in VOC dataset
PERSON_CLASS_ID = 15


class SimpleTracker:
    """
    Simple centroid-based tracker for maintaining object IDs across frames.
    """
    
    def __init__(self, max_disappeared=30, max_distance=100):
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
        
        # Compute distance matrix
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


class PersonDetector:
    """
    MobileNet-SSD based person detector with simple tracking support.
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
        
        # Load model based on file extension
        if str(model_weights).endswith('.caffemodel'):
            self.net = cv2.dnn.readNetFromCaffe(str(model_config), str(model_weights))
        elif str(model_weights).endswith('.pb'):
            self.net = cv2.dnn.readNetFromTensorflow(str(model_weights), str(model_config))
        else:
            # Try generic approach
            self.net = cv2.dnn.readNet(str(model_weights), str(model_config))
        
        # Set backend and target (use CUDA if available)
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
        
        # Prepare input blob
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

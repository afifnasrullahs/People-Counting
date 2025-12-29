"""
YOLO-based person detector with tracking.
"""
import numpy as np
from ultralytics import YOLO
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PersonDetector:
    """
    YOLO-based person detector with object tracking support.
    """
    
    def __init__(self, model_weights: str, tracker_yaml: str, conf: float, iou: float):
        """
        Initialize detector.
        
        Args:
            model_weights: Path to YOLO model weights
            tracker_yaml: Path to tracker configuration YAML
            conf: Confidence threshold
            iou: IOU threshold
        """
        self.model_weights = model_weights
        self.tracker_yaml = tracker_yaml
        self.conf = conf
        self.iou = iou
        
        logger.info(f"Loading YOLO model: {model_weights}")
        self.model = YOLO(str(model_weights))
        logger.info("YOLO model loaded successfully.")

    def detect_and_track(self, frame):
        """
        Detect and track persons in frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (boxes, ids) where boxes is Nx4 array and ids is N array or None
        """
        try:
            results = self.model.track(
                frame,
                persist=True,
                classes=[0],  # person only
                tracker=self.tracker_yaml,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )
        except Exception as e:
            logger.warning(f"Tracker error: {e}, falling back to detection only")
            results = self.model(frame)

        boxes = []
        ids = None
        
        if len(results) > 0 and getattr(results[0], "boxes", None) is not None:
            det = results[0].boxes
            try:
                boxes = det.xyxy.cpu().numpy()
                ids_attr = getattr(det, "id", None)
                if ids_attr is not None:
                    ids = ids_attr.cpu().numpy().astype(int)
            except Exception:
                try:
                    boxes = np.array(det.xyxy)
                except Exception:
                    boxes = np.array([])
                ids = None
        
        return boxes, ids

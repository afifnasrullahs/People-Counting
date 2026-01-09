"""
SORT: Simple Online and Realtime Tracking
Original paper: https://arxiv.org/abs/1602.00763
This implementation is adapted from https://github.com/abewley/sort
"""
import numpy as np
from filterpy.kalman import KalmanFilter


def linear_assignment(cost_matrix):
    """
    Linear assignment using scipy's linear_sum_assignment.
    Falls back to a greedy approach if scipy is not available.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        # Greedy fallback
        indices = []
        cost = cost_matrix.copy()
        while cost.size > 0:
            min_idx = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[min_idx] < 1e9:
                indices.append(min_idx)
                cost[min_idx[0], :] = 1e10
                cost[:, min_idx[1]] = 1e10
            else:
                break
        return np.array(indices) if indices else np.empty((0, 2), dtype=int)


def iou_batch(bb_test, bb_gt):
    """
    Compute IOU between two sets of bounding boxes.
    
    Args:
        bb_test: Array of shape (N, 4) [x1, y1, x2, y2]
        bb_gt: Array of shape (M, 4) [x1, y1, x2, y2]
        
    Returns:
        IOU matrix of shape (N, M)
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    
    iou = wh / (area_test + area_gt - wh + 1e-6)
    return iou


def convert_bbox_to_z(bbox):
    """
    Convert bounding box [x1, y1, x2, y2] to z measurement [x, y, s, r].
    
    Args:
        bbox: [x1, y1, x2, y2]
        
    Returns:
        z: [x, y, s, r] where x,y is center, s is scale (area), r is aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert state [x, y, s, r] to bounding box [x1, y1, x2, y2].
    
    Args:
        x: State [x, y, s, r, ...]
        score: Optional confidence score
        
    Returns:
        bbox: [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    
    x1 = x[0] - w / 2.
    y1 = x[1] - h / 2.
    x2 = x[0] + w / 2.
    y2 = x[1] + h / 2.
    
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement uncertainty
        self.kf.R[2:, 2:] *= 10.
        
        # Covariance matrix
        self.kf.P[4:, 4:] *= 1000.  # High uncertainty for unobserved velocities
        self.kf.P *= 10.
        
        # Process uncertainty
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """
        Update the state vector with observed bbox.
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        """
        Advance the state vector and return the predicted bounding box estimate.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked objects (both represented as bounding boxes).
    
    Args:
        detections: Array of detections [x1, y1, x2, y2]
        trackers: Array of tracker predictions [x1, y1, x2, y2]
        iou_threshold: Minimum IOU for match
        
    Returns:
        matches: Array of matched indices [[det_idx, trk_idx], ...]
        unmatched_detections: Array of unmatched detection indices
        unmatched_trackers: Array of unmatched tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    SORT: Simple Online and Realtime Tracking
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep alive a track without associated detections
            min_hits: Minimum number of associated detections before track is confirmed
            iou_threshold: Minimum IOU for match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets=np.empty((0, 5))):
        """
        Update tracker with detections.
        
        Args:
            dets: Array of detections [x1, y1, x2, y2, score] or [x1, y1, x2, y2]
            
        Returns:
            Array of tracks [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        if len(dets) > 0:
            if dets.shape[1] == 4:
                dets = np.column_stack([dets, np.ones(len(dets))])
            
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                dets[:, :4], trks, self.iou_threshold
            )
            
            # Update matched trackers
            for m in matched:
                self.trackers[m[1]].update(dets[m[0], :4])
            
            # Create new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :4])
                self.trackers.append(trk)
        
        # Return active tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # More lenient track display to prevent flickering:
            # 1. Show if updated recently (within last 2 frames)
            # 2. Show if it has any hits and wasn't missing too long
            should_show = False
            
            if trk.time_since_update < 2:
                # Recently updated - show if has enough hits or is new
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits + 2:
                    should_show = True
            
            if trk.time_since_update <= 5 and trk.hits >= 1:
                # Lost but was established - keep showing with prediction
                should_show = True
            
            if should_show:
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            
            i -= 1
            
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        
        return np.empty((0, 5))
    
    def reset(self):
        """Reset the tracker state."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

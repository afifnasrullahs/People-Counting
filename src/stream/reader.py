"""
Threaded RTMP/RTSP stream reader.
"""
import time
from threading import Thread, Lock
import cv2
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RTMPReader:
    """
    Threaded reader that continuously reads frames from the RTMP/RTSP source
    and stores only the latest frame (drop older ones). This reduces perceived jitter.
    """
    
    def __init__(self, src: str, use_ffmpeg: bool = True):
        """
        Initialize RTMP Reader.
        
        Args:
            src: Video source URL (RTSP/RTMP/file path)
            use_ffmpeg: Whether to use FFMPEG backend
        """
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
        """Connect to video source."""
        # Prefer FFMPEG backend when available
        if self.use_ffmpeg:
            try:
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            except Exception:
                self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            logger.warning("RTMPReader: initial connection failed (will retry)...")
            return

        # Try to minimize internal buffering if supported
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Optional: hint fps if known (may help)
        try:
            self.cap.set(cv2.CAP_PROP_FPS, 25)
        except Exception:
            pass
        
        logger.info(f"RTMPReader: Connected to {self.src}")

    def _reconnect(self):
        """Reconnect to video source."""
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
        finally:
            self.cap = None
        # Small delay before reconnect
        time.sleep(0.8)
        self._connect()

    def _update_loop(self):
        """Background thread loop for reading frames."""
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                continue

            # Read frames as fast as possible and keep only latest
            ret, frame = self.cap.read()
            if not ret or frame is None:
                try:
                    time.sleep(0.2)
                except Exception:
                    pass
                self._reconnect()
                continue

            with self.lock:
                self.ret = True
                self.frame = frame.copy()
                self.frame_ts = time.time()

    def read(self):
        """
        Return latest frame.
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None, 0.0
            return True, self.frame.copy(), self.frame_ts

    def stop(self):
        """Stop the reader and release resources."""
        self.stopped = True
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        logger.info("RTMPReader stopped.")

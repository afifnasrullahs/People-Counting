"""
Threaded RTMP/RTSP stream reader.
"""
import time
import os
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
        self.connection_attempts = 0
        self._connect()
        self.thread = Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        """Connect to video source."""
        self.connection_attempts += 1
        logger.info(f"RTMPReader: Connecting to {self.src} (attempt {self.connection_attempts})...")
        
        # Set environment variable for better RTSP handling
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        # Try different backends
        backends = []
        if self.use_ffmpeg:
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        
        for backend in backends:
            try:
                backend_name = {cv2.CAP_FFMPEG: "FFMPEG", cv2.CAP_GSTREAMER: "GSTREAMER", cv2.CAP_ANY: "AUTO"}.get(backend, str(backend))
                logger.info(f"RTMPReader: Trying backend {backend_name}...")
                
                self.cap = cv2.VideoCapture(self.src, backend)
                
                # Wait a bit for connection
                time.sleep(1.0)
                
                if self.cap.isOpened():
                    # Try to read a test frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"RTMPReader: Connected successfully with {backend_name}")
                        break
                    else:
                        logger.warning(f"RTMPReader: {backend_name} opened but can't read frame")
                        self.cap.release()
                else:
                    logger.warning(f"RTMPReader: {backend_name} failed to open")
                    
            except Exception as e:
                logger.warning(f"RTMPReader: {backend_name} error: {e}")
                continue

        if self.cap is None or not self.cap.isOpened():
            logger.warning("RTMPReader: All backends failed (will retry)...")
            return

        # Try to minimize internal buffering if supported
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        
        # Get stream info
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"RTMPReader: Stream info - {width}x{height} @ {fps:.1f} FPS")
        except Exception:
            pass

    def _reconnect(self):
        """Reconnect to video source."""
        logger.info("RTMPReader: Reconnecting...")
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
        finally:
            self.cap = None
        # Longer delay before reconnect
        time.sleep(2.0)
        self._connect()

    def _update_loop(self):
        """Background thread loop for reading frames."""
        consecutive_failures = 0
        
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                continue

            # Read frames as fast as possible and keep only latest
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                logger.warning(f"RTMPReader: Read error: {e}")
                ret, frame = False, None
                
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    logger.warning(f"RTMPReader: Too many failures ({consecutive_failures}), reconnecting...")
                    consecutive_failures = 0
                    self._reconnect()
                else:
                    time.sleep(0.1)
                continue
            
            consecutive_failures = 0

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

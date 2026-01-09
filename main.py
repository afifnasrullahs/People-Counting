#!/usr/bin/env python3
"""
Entrance Detection System - Main Entry Point
People counter using YOLOv8 detection with SORT tracking and MQTT publishing.
"""
from src.config import (
    VIDEO_SOURCE, CONF_THRESHOLD, RESIZE_TO,
    LOG_LEVEL, LOG_FILE, DETECTOR_TYPE, HEADLESS, FRAME_SKIP
)
from src.core import PeopleCounter
from src.utils import setup_logging, get_logger

# Setup logging
setup_logging(LOG_LEVEL, LOG_FILE)
logger = get_logger(__name__)


def main():
    """Main entry point."""
    logger.info("=" * 50)
    logger.info("Entrance Detection System Starting...")
    logger.info("=" * 50)
    logger.info(f"Video Source: {VIDEO_SOURCE}")
    logger.info(f"Detector: {DETECTOR_TYPE.upper()}")
    logger.info(f"Confidence: {CONF_THRESHOLD}")
    logger.info(f"Resize: {RESIZE_TO}")
    logger.info(f"Headless: {HEADLESS}, Frame Skip: {FRAME_SKIP}")
    logger.info("=" * 50)
    
    pc = PeopleCounter(
        VIDEO_SOURCE,
        conf=CONF_THRESHOLD,
        resize_to=RESIZE_TO,
        detector_type=DETECTOR_TYPE
    )
    
    try:
        pc.run()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Entrance Detection System Stopped.")


if __name__ == "__main__":
    main()

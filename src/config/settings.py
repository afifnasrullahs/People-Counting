"""
Configuration settings for Entrance Detection System.
Load from environment variables or use defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------------------
# PATH CONFIGURATION
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ------------------------
# MODEL CONFIGURATION
# ------------------------
MODEL_CONFIG = os.getenv("MODEL_CONFIG")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS")
MODEL_CONFIG_PATH = MODELS_DIR / MODEL_CONFIG if (MODELS_DIR / MODEL_CONFIG).exists() else MODEL_CONFIG
MODEL_WEIGHTS_PATH = MODELS_DIR / MODEL_WEIGHTS if (MODELS_DIR / MODEL_WEIGHTS).exists() else MODEL_WEIGHTS
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD"))
INPUT_SIZE = (300, 300)  # MobileNet-SSD input size

# ------------------------
# VIDEO SOURCE
# ------------------------
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE")
if not VIDEO_SOURCE:
    raise ValueError("VIDEO_SOURCE environment variable is required. Please set it in .env file.")

# ------------------------
# DETECTION LINE
# ------------------------
LINE_POSITION = float(os.getenv("LINE_POSITION"))  # fraction of frame width (0..1)

# ------------------------
# RESIZE CONFIGURATION
# ------------------------
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH"))
RESIZE_HEIGHT = int(os.getenv("RESIZE_HEIGHT"))
RESIZE_TO = (RESIZE_WIDTH, RESIZE_HEIGHT) if RESIZE_WIDTH > 0 and RESIZE_HEIGHT > 0 else None

# ------------------------
# TRACKING CONFIGURATION
# ------------------------
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA"))  # new_position = old*(1-alpha) + new*alpha
HISTORY_MAX = int(os.getenv("HISTORY_MAX"))  # max stored positions per track id
MIN_HISTORY_TO_COUNT = int(os.getenv("MIN_HISTORY_TO_COUNT"))  # min history length to consider counting
DEBOUNCE_SEC = float(os.getenv("DEBOUNCE_SEC"))  # minimal seconds between counts for same id
RECOUNT_TIMEOUT_SEC = float(os.getenv("RECOUNT_TIMEOUT_SEC"))  # after this seconds, same ID may be counted again
MIN_MOVEMENT_PIXELS = int(os.getenv("MIN_MOVEMENT_PIXELS"))  # minimum movement to consider real crossing

# ------------------------
# VISUAL / UI
# ------------------------
INFO_PANEL_POS = (10, 10)  # top-left of info panel
WINDOW_NAME = "People Counter - Vertical Line"

# ------------------------
# MQTT CONFIGURATION
# ------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

if not MQTT_BROKER:
    raise ValueError("MQTT_BROKER environment variable is required. Please set it in .env file.")

MQTT_CONFIG = {
    "broker": MQTT_BROKER,
    "port": int(os.getenv("MQTT_PORT")),
    "username": MQTT_USERNAME,
    "password": MQTT_PASSWORD,
    "topic": os.getenv("MQTT_TOPIC"),
    "client_id": os.getenv("MQTT_CLIENT_ID")
}

# ------------------------
# LOGGING CONFIGURATION
# ------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "entrance_detection.log"

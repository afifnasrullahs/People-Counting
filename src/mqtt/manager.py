"""
MQTT Manager for publishing people count data.
"""
import json
import paho.mqtt.client as mqtt
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MQTTManager:
    """
    Manages MQTT connection and publishing people count data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize MQTT Manager.
        
        Args:
            config: Dictionary containing broker, port, username, password, topic, client_id
        """
        self.config = config
        self.client = None
        self.connected = False
        self._connect()

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            logger.info("MQTT connected successfully.")
        else:
            self.connected = False
            logger.error(f"MQTT connection failed with code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        self.connected = False
        logger.warning(f"MQTT disconnected with code: {rc}")
        if rc != 0:
            logger.warning("Unexpected disconnection, will attempt to reconnect...")

    def _connect(self):
        """Establish MQTT connection."""
        try:
            self.client = mqtt.Client(client_id=self.config["client_id"])
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Set authentication if provided
            if self.config.get("username") and self.config.get("password"):
                self.client.username_pw_set(
                    self.config["username"],
                    self.config["password"]
                )
            
            self.client.connect(
                self.config["broker"],
                self.config["port"],
                keepalive=60
            )
            self.client.loop_start()  # Start background thread for network loop
            logger.info(f"Connecting to MQTT broker at {self.config['broker']}:{self.config['port']}...")
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
            self.client = None
            self.connected = False

    def publish_count(self, people_inside: int) -> bool:
        """
        Publish the current count to MQTT topic.
        
        Args:
            people_inside: Current number of people inside
            
        Returns:
            True if published successfully, False otherwise
        """
        if self.client is None or not self.connected:
            logger.warning("MQTT not connected, skipping publish.")
            return False
        try:
            payload = {
                "occupancy": people_inside
            }
            result = self.client.publish(
                self.config["topic"],
                json.dumps(payload),
                qos=1  # At least once delivery
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"[MQTT] Published: occupancy={people_inside}")
                return True
            else:
                logger.error(f"MQTT publish failed with code: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing to MQTT: {e}")
            return False

    def close(self):
        """Close MQTT connection."""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            logger.info("MQTT connection closed.")
        except Exception as e:
            logger.error(f"Error closing MQTT: {e}")

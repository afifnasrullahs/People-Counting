"""
Unit tests for MQTT Manager.
"""
import unittest
from unittest.mock import Mock, patch


class TestMQTTManager(unittest.TestCase):
    """Test cases for MQTTManager class."""
    
    @patch('src.mqtt.manager.mqtt.Client')
    def test_mqtt_connection(self, mock_client):
        """Test MQTT connection initialization."""
        from src.mqtt.manager import MQTTManager
        
        config = {
            "broker": "localhost",
            "port": 1883,
            "username": "test",
            "password": "test",
            "topic": "test/topic",
            "client_id": "test_client"
        }
        
        manager = MQTTManager(config)
        
        # Verify client was created
        mock_client.assert_called_once_with(client_id="test_client")
    
    @patch('src.mqtt.manager.mqtt.Client')
    def test_publish_when_not_connected(self, mock_client):
        """Test publish returns False when not connected."""
        from src.mqtt.manager import MQTTManager
        
        config = {
            "broker": "localhost",
            "port": 1883,
            "username": None,
            "password": None,
            "topic": "test/topic",
            "client_id": "test_client"
        }
        
        manager = MQTTManager(config)
        manager.connected = False
        
        result = manager.publish_count(5)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

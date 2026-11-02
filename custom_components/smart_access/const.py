"""Константы интеграции Smart Access."""
DOMAIN = "smart_access"

MQTT_PREFIX = "smart_access"
TOPIC_STATUS = f"{MQTT_PREFIX}/status"
TOPIC_EVENT = f"{MQTT_PREFIX}/event"
TOPIC_ACK = f"{MQTT_PREFIX}/ack"
TOPIC_ENROLL_CAR = f"{MQTT_PREFIX}/enroll/car"
TOPIC_ENROLL_FACE = f"{MQTT_PREFIX}/enroll/face"
TOPIC_CMD_IDENTIFY = f"{MQTT_PREFIX}/cmd/identify"
TOPIC_CMD_RETRAIN = f"{MQTT_PREFIX}/cmd/retrain"

SIGNAL_STATUS = f"{DOMAIN}_status_update"
SIGNAL_EVENT = f"{DOMAIN}_event_update"

EVENT_ACK = f"{DOMAIN}_ack"

CONF_CAR_SUBJECTS = "car_subjects"
CONF_FACE_SUBJECTS = "face_subjects"
CONF_ENROLL_MAP = "enroll_map"

# v1.4.0: базовый адрес моста (сайт аддона на :8100), откуда интеграция
# берёт кадры/поток для сущностей camera.smart_access_<name>. Хост —
# шлюз hassio-сети: HA Core и аддоны видят по нему сайт аддона Smart Access.
CONF_BRIDGE_URL = "bridge_url"
DEFAULT_BRIDGE_URL = "http://172.30.32.1:8100"

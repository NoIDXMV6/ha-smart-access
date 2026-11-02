#!/usr/bin/with-contenv bashio
set -e

# Preflight: если модели не скачались при сборке (или сбились) — докачать в /data/models.
MODELS_DIR=/models
if [ ! -s /models/clip_vision_q.onnx ] || [ ! -s /models/sface.onnx ] || [ ! -s /models/yunet.onnx ]; then
    bashio::log.warning "Модели в образе отсутствуют — пробую докачать в /data/models"
    mkdir -p /data/models
    [ -s /data/models/yunet.onnx ] || curl -sSL --retry 3 -o /data/models/yunet.onnx \
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" || true
    [ -s /data/models/sface.onnx ] || curl -sSL --retry 3 -o /data/models/sface.onnx \
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" || true
    [ -s /data/models/clip_vision_q.onnx ] || curl -sSL --retry 3 -o /data/models/clip_vision_q.onnx \
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx" || true
    MODELS_DIR=/data/models
fi
# Модель поз (жесты, v1.2.0): опциональна — без неё жесты отключены, остальное работает.
if [ ! -s /models/yolov8n-pose.onnx ] && [ ! -s /data/models/yolov8n-pose.onnx ]; then
    bashio::log.warning "Модель жестов yolov8n-pose.onnx отсутствует — докачиваю в /data/models (жесты без неё отключены)"
    curl -sSL --retry 3 -o /data/models/yolov8n-pose.onnx \
        "https://huggingface.co/Xenova/yolov8n-pose/resolve/main/onnx/model_quantized.onnx" || true
fi
if [ ! -s /models/yolov8n-pose.onnx ] && [ -s /data/models/yolov8n-pose.onnx ]; then
    MODELS_DIR=/data/models
fi
export SMART_ACCESS_MODELS_DIR="$MODELS_DIR"
# Pose-модель могла остаться в /models, когда остальные докачаны в /data/models:
# models.py умеет искать её по отдельному пути.
if [ -s /models/yolov8n-pose.onnx ] && [ "$MODELS_DIR" != "/models" ]; then
    export SMART_ACCESS_POSE_MODEL=/models/yolov8n-pose.onnx
fi

bashio::log.info "Запуск Smart Access Core..."
exec python3 -u /app/app/main.py

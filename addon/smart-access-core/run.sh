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
export SMART_ACCESS_MODELS_DIR="$MODELS_DIR"

bashio::log.info "Запуск Smart Access Core..."
exec python3 -u /app/main.py

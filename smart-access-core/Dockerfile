ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG C.UTF-8

# ffmpeg здесь НЕ нужен: Frigate берёт его из своего аддона, а код Smart Access
# работает без него (MJPEG-мост и модели — чистый python/opencv).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-setuptools curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
      aiohttp==3.9.5 \
      paho-mqtt==1.6.1 \
      numpy==1.26.4 \
      opencv-python-headless==4.10.0.84 \
      onnxruntime==1.18.1 \
      Pillow==10.4.0 \
      requests==2.32.3

# Модели вшиты в образ при сборке в CI (скачиваются из GitHub Release models-v1).
# Если по какой-то причине их нет — докачаются при первом запуске (run.sh).
COPY models /models
RUN ls -la /models

COPY app /app
WORKDIR /app

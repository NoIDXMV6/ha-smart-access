ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG C.UTF-8

# ffmpeg здесь НЕ нужен: Frigate берёт его из своего аддона, а код Smart Access
# работает без него (MJPEG-мост и модели — чистый python/opencv).
# libstdc++6/zlib1g нужны opencv/onnxruntime (в минимальном базовом образе их нет).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-setuptools curl ca-certificates libstdc++6 zlib1g \
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

# Код кладём ПАКЕТОМ /app/app: main.py делает `from app import ...`,
# поэтому каталог кода должен называться /app/app (PYTHONPATH=/app — страховка).
ENV PYTHONPATH=/app
COPY app /app/app
WORKDIR /app/app

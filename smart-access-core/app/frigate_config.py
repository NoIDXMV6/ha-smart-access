"""Генератор config.yml для Frigate — чистая функция без тяжёлых зависимостей.

Вынесен из main.py (v1.1.13), чтобы тесты могли проверять генератор без
импорта brain/models (onnxruntime и пр.).

v1.3.0 — АРХИТЕКТУРА «ОДИН ПОЛУЧАТЕЛЬ»: ВСЕ камеры идут через MJPEG-мост
аддона, включая камеры с rtsp_url (мост сам читает их RTSP и раздаёт кадры
и Frigate, и HA — у камеры остаётся ровно один клиент). Раньше rtsp-камеры
читались Frigate напрямую: клиенты камеры = Frigate + карточка HA, а у
дешёвых камер и облачных прокси от этого начинается деградация потоков.
"""
from __future__ import annotations


def build_frigate_config(conf, bridge) -> str:
    """Генерирует config.yml для Frigate из настроек аддона.

    Особенности сети аддона Frigate (он НЕ host-network, а мост hassio):
      * MQTT-брокер недоступен ему по localhost — только по имени аддона
        `core-mosquitto` (или внешнего IP); localhost подменяем автоматически;
      * наш MJPEG-мост слушает на хосте, из мостовой сети хост — это шлюз
        172.30.32.1, поэтому в конфиг подставляется он, а не 127.0.0.1.

    Креды MQTT (user/password) подставляются РАБОЧИЕ — те же, под которыми
    подключается сам Smart Access (Supervisor /services/mqtt или опции аддона).
    Заглушка «ВАШ_ПАРОЛЬ_БРОКЕРА» приводила к «MQTT Not authorized» у Frigate:
    пользователь копировал конфиг как есть.
    """
    cams = [c for c in conf.cameras if c.get("mode", "off") != "off"]
    mqtt_host = str(conf.mqtt.get("host") or "").strip()
    if mqtt_host in ("", "localhost", "127.0.0.1", "::1"):
        # для мостового аддона Frigate localhost — это он сам; брокер — соседний аддон
        mqtt_host = "core-mosquitto"
    mqtt_user = str(conf.mqtt.get("user") or "").strip()
    mqtt_pass = str(conf.mqtt.get("password") or "")
    stream_fps = max(1, round(float(conf.recognition.get("stream_fps", 5.0))))
    lines: list[str] = []
    lines.append("# === Сгенерировано Smart Access Core v1.3.0 ===")
    lines.append("# Куда вставить: веб-интерфейс Frigate → Configuration → вставить → Save & Restart.")
    lines.append("# NB: mqtt.host подставлен как core-mosquitto (аддон Frigate не видит localhost),")
    lines.append("#     а мост камер — по адресу шлюза 172.30.32.1 (хост, где слушает Smart Access).")
    lines.append("# ВСЕ камеры читают поток у моста Smart Access — у каждой камеры один")
    lines.append("# получатель (сам мост), Frigate и HA потребляют с него.")
    lines.append("mqtt:")
    lines.append(f"  host: {mqtt_host}")
    lines.append(f"  port: {int(conf.mqtt.get('port') or 1883)}")
    if mqtt_user:
        lines.append(f"  user: {mqtt_user}")
        if mqtt_pass:
            lines.append(f"  password: {mqtt_pass}")
            lines.append("# креды подставлены автоматически — не публикуйте конфиг целиком")
        else:
            lines.append(f"  # пароль пользователя {mqtt_user} не задан в настройках Smart Access — впишите его сюда")
    lines.append("")
    lines.append("detectors:")
    lines.append("  cpu:")
    lines.append("    type: openvino")
    lines.append("    device: CPU")
    # NB: секция model — только В КОРНЕ конфига (Frigate 0.14+): вложенный
    # model под detectors игнорируется с WARNING «The model key should be
    # specified at the root level».
    lines.append("model:")
    lines.append("  width: 300")
    lines.append("  height: 300")
    lines.append("  input_tensor: nhwc")
    lines.append("  input_pixel_format: bgr")
    lines.append("  labelmap_path: /openvino-model/coco_91cl_bkgr.txt")
    lines.append("")
    lines.append("cameras:")
    for c in cams:
        name = c["name"]
        disp = str(c.get("display_name") or "").strip()
        rtsp = (c.get("rtsp_url") or "").strip()
        if disp and disp != name:
            # исходное имя видно в панели/конфиге: «Пруд» -> prud
            lines.append(f"  {name}:  # «{disp}»")
        else:
            lines.append(f"  {name}:")
        lines.append("    ffmpeg:")
        lines.append("      inputs:")
        # НЕ пишем «-f mjpeg» (v1.1.13): сырой демуксер mjpeg не может
        # завершить пробу формата на multipart-потоке моста — считает
        # поток 25 fps и ждёт ~5 с «виртуального» времени (=~125 кадров);
        # при реальных 1.5 fps это минуты, ffmpeg не выдаёт НИ ОДНОГО
        # кадра за 20 с, и watchdog Frigate убивает его по кругу
        # («No frames received from <camera> in 20 seconds»). Без -f
        # ffmpeg сам видит multipart/x-mixed-replace и читает поток
        # сразу (проверено автотестом с настоящим ffmpeg).
        lines.append(f"        - path: {bridge.frigate_url(name)}")
        lines.append("          input_args: >-")
        lines.append("            -use_wallclock_as_timestamps 1")
        lines.append("            -fflags +genpts+discardcorrupt")
        # темп моста: камеры с rtsp_url мост читает сам с полным fps
        # (stream_fps), остальные — в темпе снимков/потока (bridge_fps)
        if rtsp:
            fps = stream_fps
        else:
            fps = max(1, round(float(conf.recognition.get("bridge_fps", 1.5))))
        lines.append("          roles:")
        lines.append("            - detect")
        lines.append("    detect:")
        lines.append("      enabled: true")
        lines.append(f"      fps: {fps}")
        lines.append("      width: 1280")
        lines.append("      height: 720")
        lines.append("    objects:")
        lines.append("      track:")
        if "car" in c.get("mode", ""):
            lines.append("        - car")
        if "face" in c.get("mode", ""):
            lines.append("        - person")
        lines.append("    snapshots:")
        lines.append("      enabled: true")
        lines.append("      retain:")
        lines.append("        default: 2")
        lines.append("    record:")
        lines.append("      enabled: false")
        lines.append("")
    return "\n".join(lines)

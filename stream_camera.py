#!/usr/bin/env python3
"""
Live MJPEG stream of door-detection inference with debounced state, viewable in a browser.

Usage:
  python3 stream_camera.py --hef best_v3.hef --camera 0

Then open http://<pi-ip>:5000 in a browser on your laptop.
"""

import argparse
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response
from hailo_platform import (
    HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType,
)

CLASS_NAMES = {0: "door_open", 1: "door_closed"}
DEBOUNCE_FRAMES = 3  # consecutive matching frames required before confirming a state change

app = Flask(__name__)
latest_jpeg = None
lock = threading.Lock()

# Debounce state (shared across frames)
confirmed_state = "unknown"
candidate_state = None
candidate_count = 0


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def run_inference(infer_pipeline, network_group, network_group_params,
                   input_vstream_info, output_vstream_info, frame, score_thresh):
    letterboxed, scale, (pad_x, pad_y) = letterbox(frame, 640)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb.astype(np.uint8), axis=0)

    with network_group.activate(network_group_params):
        results = infer_pipeline.infer({input_vstream_info.name: input_data})

    raw_output = results[output_vstream_info.name]
    detections = raw_output[0] if isinstance(raw_output, list) and len(raw_output) == 1 else raw_output

    found = []
    for class_id, class_dets in enumerate(detections):
        class_dets = np.array(class_dets)
        if class_dets.size == 0:
            continue
        for det in class_dets:
            y_min, x_min, y_max, x_max, score = det[:5]
            if score < score_thresh:
                continue
            x_min_px = (x_min * 640 - pad_x) / scale
            x_max_px = (x_max * 640 - pad_x) / scale
            y_min_px = (y_min * 640 - pad_y) / scale
            y_max_px = (y_max * 640 - pad_y) / scale
            label = CLASS_NAMES.get(class_id, f"class_{class_id}")
            found.append((label, float(score), (int(x_min_px), int(y_min_px), int(x_max_px), int(y_max_px))))
    return found


def update_debounced_state(detections):
    """Pick the single highest-confidence detection this frame and debounce state changes."""
    global confirmed_state, candidate_state, candidate_count

    if not detections:
        return None

    best = max(detections, key=lambda d: d[1])
    best_label = best[0]

    if best_label == candidate_state:
        candidate_count += 1
    else:
        candidate_state = best_label
        candidate_count = 1

    if candidate_count >= DEBOUNCE_FRAMES and confirmed_state != candidate_state:
        confirmed_state = candidate_state

    return best


def capture_loop(hef_path, camera_index, score_thresh):
    global latest_jpeg
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}")

    hef = HEF(hef_path)
    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.UINT8
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                detections = run_inference(
                    infer_pipeline, network_group, network_group_params,
                    input_vstream_info, output_vstream_info, frame, score_thresh
                )

                best = update_debounced_state(detections)

                if best is not None:
                    label, score, (x1, y1, x2, y2) = best
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {score:.2f}", (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                banner_color = (0, 200, 0) if confirmed_state == "door_open" else \
                    (0, 0, 200) if confirmed_state == "door_closed" else (128, 128, 128)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), banner_color, -1)
                cv2.putText(frame, f"STATE: {confirmed_state.upper()}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                ret2, jpeg = cv2.imencode(".jpg", frame)
                if ret2:
                    with lock:
                        latest_jpeg = jpeg.tobytes()


def mjpeg_generator():
    while True:
        with lock:
            frame = latest_jpeg
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)


@app.route("/")
def index():
    return "<html><body><h2>Door Detection Live Feed</h2><img src='/stream'></body></html>"


@app.route("/stream")
def stream():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def state():
    return {"state": confirmed_state}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="best_v3.hef")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    args = ap.parse_args()

    t = threading.Thread(target=capture_loop, args=(args.hef, args.camera, args.score_thresh), daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)

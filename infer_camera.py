#!/usr/bin/env python3
"""
Continuous door-detection inference from a USB webcam on the Hailo-8L.

Usage:
  python3 infer_camera.py --hef best_v3.hef --camera 0 --interval 1.0

Saves the latest annotated frame to latest_frame.jpg on every inference,
and prints detections to console. Pull latest_frame.jpg via scp to view it.
"""

import argparse
import time

import cv2
import numpy as np
from hailo_platform import (
    HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType,
)

CLASS_NAMES = {0: "door_open", 1: "door_closed"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="best_v3.hef")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index, usually 0 for /dev/video0")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between inferences")
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--output", default="latest_frame.jpg")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}. Check `ls /dev/video*` on the Pi.")

    hef = HEF(args.hef)
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

        print(f"Starting continuous inference (every {args.interval}s). Ctrl+C to stop.")
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from camera, retrying...")
                        time.sleep(1)
                        continue

                    detections = run_inference(
                        infer_pipeline, network_group, network_group_params,
                        input_vstream_info, output_vstream_info, frame, args.score_thresh
                    )

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    if detections:
                        for label, score, (x1, y1, x2, y2) in detections:
                            print(f"[{timestamp}] {label}: score={score:.3f} box=({x1},{y1})-({x2},{y2})")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {score:.2f}", (x1, max(y1 - 10, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        print(f"[{timestamp}] No detections.")

                    cv2.imwrite(args.output, frame)
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
            finally:
                cap.release()


if __name__ == "__main__":
    main()

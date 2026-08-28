#!/usr/bin/env python3
"""
Run door-detection inference on a single image using the compiled Hailo-8L HEF.

Usage:
  python3 infer_image.py --hef best_v3.hef --image test_image.jpg --output annotated.jpg
"""

import argparse

import cv2
import numpy as np
from hailo_platform import (
    HEF, VDevice, ConfigureParams, HailoStreamInterface,
    InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType,
)

CLASS_NAMES = {0: "door_open", 1: "door_closed"}


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Same preprocessing used for calibration — must match for consistent results."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="best_v3.hef")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="annotated.jpg")
    ap.add_argument("--score-thresh", type=float, default=0.2)
    args = ap.parse_args()

    orig = cv2.imread(args.image)
    if orig is None:
        raise SystemExit(f"Could not read image: {args.image}")

    letterboxed, scale, (pad_x, pad_y) = letterbox(orig, 640)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb.astype(np.uint8), axis=0)  # (1, 640, 640, 3)

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

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_dict = {input_vstream_info.name: input_data}
            with network_group.activate(network_group_params):
                results = infer_pipeline.infer(input_dict)

        raw_output = results[output_vstream_info.name]

    # raw_output for HAILO_NMS_BY_CLASS is typically a list (per image) of lists (per class)
    # of arrays shaped (num_detections, 5): [y_min, x_min, y_max, x_max, score], normalized [0,1]
    # relative to the 640x640 letterboxed input. Print raw structure first to confirm shape.
    print(f"Raw output type: {type(raw_output)}")
    detections = raw_output[0] if isinstance(raw_output, list) and len(raw_output) == 1 else raw_output

    found_any = False
    for class_id, class_dets in enumerate(detections):
        class_dets = np.array(class_dets)
        if class_dets.size == 0:
            continue
        for det in class_dets:
            y_min, x_min, y_max, x_max, score = det[:5]
            if score < args.score_thresh:
                continue
            found_any = True
            # map normalized [0,1] letterboxed coords -> original image pixel coords
            x_min_px = (x_min * 640 - pad_x) / scale
            x_max_px = (x_max * 640 - pad_x) / scale
            y_min_px = (y_min * 640 - pad_y) / scale
            y_max_px = (y_max * 640 - pad_y) / scale

            label = CLASS_NAMES.get(class_id, f"class_{class_id}")
            print(f"{label}: score={score:.3f} box=({x_min_px:.0f},{y_min_px:.0f})-({x_max_px:.0f},{y_max_px:.0f})")

            cv2.rectangle(orig, (int(x_min_px), int(y_min_px)), (int(x_max_px), int(y_max_px)), (0, 255, 0), 2)
            cv2.putText(orig, f"{label} {score:.2f}", (int(x_min_px), max(int(y_min_px) - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not found_any:
        print("No detections above threshold.")

    cv2.imwrite(args.output, orig)
    print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()

# Door Detection — Hailo-8L on Raspberry Pi 5

Real-time door state detection (open vs. closed) running on-device using a Hailo-8L AI accelerator on a Raspberry Pi 5.

## Overview

This project detects doors in a camera feed and classifies each detected door as **open** or **closed**. The model was trained as a YOLO-based object detector and converted to Hailo's `.hef` format for accelerated inference on the Hailo-8L.

## Model

- **Deployed model:** `best_v3.hef` — compiled Hailo Executable Format, runs directly on the Hailo-8L accelerator.
- **Pipeline:** trained (YOLO architecture) → exported → converted to `.hef` via the Hailo Dataflow Compiler for on-device deployment.
- Training artifacts (base checkpoint, stripped weights, dataset config) are kept outside this repo; only the final deployable `.hef` is version-controlled here.

## Scripts

| Script | Purpose |
|---|---|
| `infer_image.py` | Run door-state inference on a single static image. |
| `infer_camera.py` | Run live inference on a connected camera feed (real-time). |
| `stream_camera.py` | Stream camera inference output (e.g. for remote viewing/monitoring). |

## Setup

Requires:
- Raspberry Pi 5 with Hailo-8L accelerator
- HailoRT PCIe driver — this project was built/tested against `hailort-pcie-driver_4.24.0` (install separately from Hailo's developer site; not committed to this repo)
- Python virtual environment with the Hailo runtime + dependencies (see Hailo's official setup docs for `hailort` and `hailo-apps` installation)

## Usage

```bash
# Single image
python infer_image.py --input <path-to-image>

# Live camera inference
python infer_camera.py

# Streamed camera inference
python stream_camera.py
```

*(Adjust arguments/flags above to match the actual CLI options in each script.)*

## Privacy note

Sample/test images and captured frames (e.g. `annotated.jpg`, `latest_frame.jpg`, `test_image.jpg`) are excluded from this repository via `.gitignore`, as camera captures may include identifiable people in frame. Only source code and the compiled model are version-controlled.

## Repo contents

- `infer_camera.py`, `infer_image.py`, `stream_camera.py` — inference scripts
- `best_v3.hef` — compiled Hailo model (door open/closed detector)
- `.gitignore` — excludes virtual environment, runtime logs, driver installer, and image files

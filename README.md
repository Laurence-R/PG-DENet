# PG-DENet

Physics-Guided Daytime-like Enhancement Network for low-light object detection.

PG-DENet evaluates whether GPU-accelerated low-light image enhancement
(auto-exposure, CLAHE, logarithmic tone mapping) can improve YOLO object
detection accuracy on Sony RAW images from the SID dataset compared to a
naive RAW-to-uint8 baseline.

## Pipeline

The system runs a three-phase pipeline designed to avoid GPU L2 cache
pollution and frequency throttling when mixing PyTorch preprocessing with
TensorRT inference:

```
Phase 0  (CPU)   Load Sony ARW RAW images via rawpy, resize to max 1024px
Phase 1A (CPU)   RAW baseline — clip linear float32 to uint8
Phase 1B (GPU)   Enhanced — auto_expose -> CLAHE -> Logarithmic Tone Mapping
Phase 2  (GPU)   YOLO detection on both image sets via TensorRT
```

After detection, the pipeline computes per-image metrics (mAP@50-95,
confidence mean, inference time, end-to-end time, FPS) and generates a
comparison chart.

## Project Structure

```
PG-DENet/
  main.py                          Entry point
  pyproject.toml                   Project metadata and dependencies
  src/pg_denet/
    __init__.py                    Package exports
    detection.py                   YOLO inference, mAP computation, batch detection
    gpu.py                         CUDA tensor utilities (to_gpu, to_cpu, luminance, FFT blur)
    io.py                          RAW image loader (ARW), batch image saver
    pre_processing.py              Auto-exposure (Reinhard), resize, linear-to-uint8
    visualization.py               Grouped bar chart generation (matplotlib)
    lle_methods/
      __init__.py                  LLE method exports
      agcwd.py                     Adaptive Gamma Correction with Weighting Distribution
      clahe.py                     CLAHE (GPU luminance + CPU histogram equalization)
      lime.py                      LIME illumination map estimation
      msrcr.py                     Multi-Scale Retinex with Color Restoration
    tone_mapping/
      __init__.py                  Logarithmic, Linear, and Reinhard tone mapping (GPU)
  data/sid/short/                  SID dataset (Sony ARW files)
  result/
    figurations/                   Output charts
    samples/                       Output images (raw baseline, enhanced, annotated detections)
```

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support
- TensorRT (for `.engine` model inference)

## Installation

```bash
git clone <repository-url>
cd PG-DENet
```

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Dataset

Download the SID (See-in-the-Dark) Sony short-exposure subset and place
the ARW files under `data/sid/short/`.

Reference: Chen et al., "Learning to See in the Dark", CVPR 2018.

## YOLO Model

The pipeline uses a TensorRT-exported YOLOv26l model (`yolo26l.engine`).
To export from a PyTorch checkpoint:

```bash
yolo export model=yolo26l.pt format=engine half=True
```

Place the resulting `yolo26l.engine` file in the project root.

## Usage

```bash
uv run main.py
```

This will:

1. Load the first 20 RAW images from `data/sid/short/`
2. Generate RAW baseline (linear clip to uint8) and enhanced images
   (auto-exposure, CLAHE, logarithmic tone mapping)
3. Run YOLO object detection on both sets
4. Print a comparison table with averaged metrics
5. Save annotated detection images to `result/samples/`
6. Save a comparison chart to `result/figurations/1st_stage_result.png`

## Configuration

Edit the constants at the top of `main.py`:

| Variable    | Default          | Description                              |
|-------------|------------------|------------------------------------------|
| HDR_DIR     | data/sid/short   | Directory containing ARW files           |
| MODEL       | yolo26l.engine   | YOLO model filename                      |
| MAX_IMAGES  | 20               | Number of images to process              |
| MAX_SIDE    | 1024             | Resize longest side to this value        |
| GT_CONF     | 0.7              | Confidence threshold for pseudo-GT boxes |
| DET_CONF    | 0.25             | Detection confidence threshold           |

## Output Metrics

| Metric              | Description                                        |
|---------------------|----------------------------------------------------|
| mAP@50-95           | Mean Average Precision at IoU thresholds 0.50-0.95 |
| Conf Mean           | Average detection confidence                       |
| Inference Time (ms) | YOLO inference time (forward pass only)            |
| E2E Time (ms)       | End-to-end time (preprocessing + inference)        |
| FPS                 | Frames per second (inference-only)                 |

## Available Enhancement Methods

Low-Light Enhancement (LLE):

- CLAHE -- Contrast Limited Adaptive Histogram Equalization
- LIME -- Low-light Image Enhancement via Illumination Map Estimation
- AGCWD -- Adaptive Gamma Correction with Weighting Distribution
- MSRCR -- Multi-Scale Retinex with Color Restoration

Tone Mapping:

- Logarithmic -- Drago et al. (2003)
- Linear -- Max normalization
- Reinhard -- Reinhard et al. (2002)

The current pipeline uses CLAHE + Logarithmic tone mapping. Other methods
are available in the `lle_methods` and `tone_mapping` modules.

## License

See repository for license information.

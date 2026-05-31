# Entrance People Counting

This project counts people crossing an entrance in `resources/data/entrance.mov`.
The implementation is a production Python package with a command-line entrypoint.

The pipeline uses:

- YOLOv8 person detection.
- ByteTrack tracking through Ultralytics.
- A doorway threshold line for directional crossing events.
- The bottom-center of each person box as the tracked floor-contact point.
- Jitter suppression, late track initialization handling, and per-track cooldown.

## Current Baseline

The latest validated run against `resources/data/entrance.mov` produced:

| Metric | Value |
| --- | ---: |
| Video duration | 85.535 seconds |
| Frames processed | 2,548 |
| In count | 7 |
| Out count | 18 |
| Total crossings | 25 |

The total is a crossing-event count, not a frame-level occupancy count.

## Project Structure

```text
.
├── pyproject.toml
├── requirements.txt
├── src
│   └── entrance_counter
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── geometry.py
│       ├── modeling.py
│       ├── pipeline.py
│       ├── rendering.py
│       └── video_io.py
├── tests
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_geometry.py
│   ├── test_modeling.py
│   ├── test_pipeline.py
│   └── test_video_rendering.py
├── resources
│   ├── data
│   │   └── entrance.mov
│   └── docs
│       └── research-1.md
└── output
    ├── entrance_roi_preview.jpg
    ├── entrance_people_count_annotated.mp4
    ├── entrance_people_count_events.csv
    ├── entrance_people_count_summary.csv
    └── models
        └── yolov8n.pt
```

Keep raw input media under `resources/data/`. Generated files belong under `output/`.

## Setup

Create a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first full run uses `output/models/yolov8n.pt` when present. If the weight file is missing, Ultralytics downloads it.

## Run

Generate the ROI preview, annotated video, event log, and summary CSV:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output
```

Equivalent module invocation:

```bash
python -m entrance_counter --video resources/data/entrance.mov --output-dir output
```

For a short smoke run:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --max-frames 300 --skip-annotated-video --no-preview
```

## Test

Run the automated tests:

```bash
python -m pytest -q
```

## Outputs

Generated artifacts are written to `output/`:

- `entrance_roi_preview.jpg`: sampled frame showing the configured entrance counting line.
- `entrance_people_count_annotated.mp4`: processed video with boxes, tracks, crossing line, and live counts.
- `entrance_people_count_events.csv`: one row per counted crossing event.
- `entrance_people_count_summary.csv`: run metadata and final directional totals.
- `models/yolov8n.pt`: local YOLO model weights, when available.

## Tuning

The main parameters are exposed through CLI flags:

- `--line x1,y1,x2,y2`: doorway threshold coordinates.
- `--confidence`: YOLO detection confidence threshold.
- `--image-size`: inference image size.
- `--process-every-n-frames`: frame sampling rate.
- `--max-frames`: optional cap for quick runs.
- `--invert-directions`: flips the `in` and `out` direction convention.

If the ROI preview shows the threshold in the wrong place, adjust `--line` first.
If visual review shows that direction labels are reversed, use `--invert-directions`.

## Reference Material

`resources/docs/research-1.md` contains background notes on pedestrian counting methods, including tracking-by-detection, line-crossing geometry, video individual counting, and dense-crowd alternatives.

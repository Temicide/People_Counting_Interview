# Entrance People Counting

This project counts people crossing an entrance in `resources/data/entrance.mov`.
The analysis is implemented as a reproducible Jupyter notebook in `src.ipynb`.

The current notebook uses a tracking-by-detection pipeline:

- YOLOv8n detects people frame by frame.
- ByteTrack assigns stable track IDs through Ultralytics.
- A doorway threshold line is used to count crossing events.
- The bottom-center of each person box is used as the tracked point because it approximates the person's contact point with the floor.

## Current Result

The latest generated summary in `output/entrance_people_count_summary.csv` reports:

| Metric | Value |
| --- | ---: |
| Video duration | 85.535 seconds |
| Frames processed | 2,548 |
| In count | 7 |
| Out count | 18 |
| Total crossings | 25 |

The total is a crossing-event count, not a frame-level occupancy count. A person standing near the doorway is counted only when their tracked bottom-center crosses the entrance threshold.

## Project Structure

```text
.
├── src.ipynb
├── requirements.txt
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

## Quick Start

Create a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter nbconvert
```

Run the notebook interactively:

```bash
jupyter notebook src.ipynb
```

Or execute it from a clean kernel to validate reproducibility:

```bash
jupyter nbconvert --to notebook --execute src.ipynb --output output/executed.ipynb
```

The first notebook cell installs missing runtime packages such as OpenCV, NumPy, pandas, matplotlib, PyTorch, Ultralytics, and `lap`.

## Notebook Workflow

`src.ipynb` runs these steps from top to bottom:

1. Prepare runtime imports and paths.
2. Load `resources/data/entrance.mov`.
3. Configure the YOLO model, confidence threshold, image size, and counting line.
4. Save an ROI preview with the entrance threshold drawn on a sampled frame.
5. Run YOLOv8n person detection with ByteTrack tracking.
6. Count each qualifying line crossing with jitter suppression and a late-initialization buffer.
7. Save the annotated video, event log, and summary CSV.

## Outputs

Generated artifacts are written to `output/`:

- `entrance_roi_preview.jpg`: sampled frame showing the configured entrance counting line.
- `entrance_people_count_annotated.mp4`: processed video with boxes, tracks, crossing line, and live counts.
- `entrance_people_count_events.csv`: one row per counted crossing event.
- `entrance_people_count_summary.csv`: run metadata and final directional totals.
- `models/yolov8n.pt`: downloaded YOLO model weights, when available locally.

## Tuning Notes

The main parameters live near the top of `src.ipynb`:

- `COUNTING_LINE`: doorway threshold coordinates.
- `CONFIDENCE`: YOLO detection confidence threshold.
- `IMAGE_SIZE`: inference image size.
- `PROCESS_EVERY_N_FRAMES`: frame sampling rate.
- `MAX_FRAMES`: optional cap for quick test runs.
- `INVERT_DIRECTIONS`: flips the `in` and `out` direction convention.

If the ROI preview shows the threshold in the wrong place, adjust `COUNTING_LINE` first. If visual review shows that the direction labels are reversed, set `INVERT_DIRECTIONS = True`.

## Reference Material

`resources/docs/research-1.md` contains background notes on pedestrian counting methods, including tracking-by-detection, line-crossing geometry, video individual counting, and dense-crowd alternatives.

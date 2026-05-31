# Repository Guidelines

## Project Structure & Module Organization

This is a compact production Python computer-vision project. The source of truth lives in `src/entrance_counter/`. Keep modules focused: configuration in `config.py`, line-crossing geometry in `geometry.py`, model loading in `modeling.py`, video helpers in `video_io.py`, drawing in `rendering.py`, orchestration in `pipeline.py`, and CLI wiring in `cli.py`. Tests belong in `tests/` and should avoid loading YOLO weights unless a full manual verification run requires it. Source notes and research material belong in `resources/docs/`. Raw local inputs, including videos, belong in `resources/data/`. Generated artifacts belong in `output/`; do not treat files there as source inputs unless code or docs explicitly document that dependency.

## Build, Test, and Development Commands

Create and activate a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the CLI on the entrance video:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output
```

Run a short smoke execution:

```bash
entrance-counter --video resources/data/entrance.mov --output-dir output --max-frames 300 --skip-annotated-video --no-preview
```

Run automated tests:

```bash
python -m pytest -q
```

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Keep file paths relative to the repository root in docs and examples, such as `resources/data/entrance.mov`. Prefer dataclasses for structured configuration and pure functions for geometry so behavior stays easy to test. Keep generated filenames descriptive, for example `output/entrance_people_count_summary.csv`.

## Testing Guidelines

Unit tests should cover pure geometry, config parsing, rendering helpers, model-result adapters, and CLI config mapping. Integration tests should use fake model outputs and tiny generated videos so the suite remains fast and deterministic. Validate full reproducibility manually by running the CLI from a clean environment and confirming expected outputs appear under `output/`.

## Commit & Pull Request Guidelines

Use concise imperative commit messages, such as `Add people counting CLI`. Pull requests should include a short summary, verification commands, notes about changed inputs or generated outputs, and screenshots or sample frames when visual processing changes.

## Security & Configuration Tips

Do not commit secrets, private datasets, or large derived files unless intentionally required. Keep raw media in `resources/data/` and generated results in `output/` so source material and artifacts remain easy to distinguish. Model weights and large generated videos should remain ignored unless there is an explicit reason to version them.

# Repository Guidelines

## Project Structure & Module Organization

This is a compact computer-vision notebook project. The primary workflow lives in `src.ipynb`, which should contain the reproducible analysis and processing code. Source notes and research material belong in `resources/docs/`, such as `resources/docs/research-1.md`. Raw local inputs, including videos, belong in `resources/data/`, for example `resources/data/entrance.mov`. Generated artifacts belong in `output/`; do not treat files there as source inputs unless the notebook explicitly documents that dependency.

## Build, Test, and Development Commands

Create and activate a local Python environment before notebook work:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebook interactively:

```bash
jupyter notebook src.ipynb
```

Execute it non-interactively to validate reproducibility:

```bash
jupyter nbconvert --to notebook --execute src.ipynb --output output/executed.ipynb
```

## Coding Style & Naming Conventions

Use clear, small notebook cells with Markdown context before major analysis steps. Follow standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Keep file paths relative to the repository root, such as `resources/data/entrance.mov`. Use descriptive generated filenames, for example `output/entrance_people_count_summary.csv`.

## Testing Guidelines

There is no formal test suite yet. Validate changes by running `src.ipynb` from a clean kernel and confirming expected outputs appear under `output/`. If Python modules are added later, place tests under `tests/`, name files `test_*.py`, and run them with:

```bash
pytest
```

## Commit & Pull Request Guidelines

This directory is not currently a Git repository, so no local commit history conventions are available. When Git is introduced, use concise imperative commit messages, such as `Add entrance people counting notebook`. Pull requests should include a short summary, verification commands or notebook steps, notes about changed inputs or generated outputs, and screenshots or sample frames when visual processing changes.

## Security & Configuration Tips

Do not commit secrets, private datasets, or large derived files unless intentionally required. Keep raw media in `resources/data/` and generated results in `output/` so source material and artifacts remain easy to distinguish.

# Mini-Project 2 — Solar Micro-Grid Dispatch Planner

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Run
jupyter notebook project2_microgrid.ipynb   # Restart & Run All

## Tests
pytest -q

## AI use declaration
Used an AI assistant to draft the `MicroGrid` dataclass and the NNLS
fallback. All linear algebra, timings, statistics and plots were verified
by hand / by a second method in the notebook.

## Findings
See "Findings & Limitations" cell at the end of the notebook.
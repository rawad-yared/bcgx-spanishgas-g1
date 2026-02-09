# SpanishGas

## How to run locally

1. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install project dependencies:
   - `python -m pip install -e .`
   - `python -m pip install pytest pre-commit`
3. Run checks:
   - `make lint`
   - `make test`

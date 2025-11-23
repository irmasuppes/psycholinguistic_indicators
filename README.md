#  Leveraging Large Language Models to Extract Psycholinguistic Indicators in Therapy Transcripts for Enhanced Digital Interventions
Code and experiments from a master’s thesis on using large language models to extract and validate psycholinguistic indicators (e.g., hopelessness, coping, energy level) from therapy transcripts for interpretable digital mental health systems.


# Project Structure

    .
    ├── data/                 # Combined input dataset
    ├── prompts/              # Zero-shot, few-shot, and hidden-CoT prompt templates
    ├── src/
    │   ├── run_extraction.py # Core extraction script
    │   ├── evaluate.py       # Evaluation for both downstream tasks
    |   ├── classifiers.py    # 
    │   └── utils.py          # Utility code
    ├── outputs/              # Stores LLM outputs in json format
    ├── results/              # Evaluation summaries
    └── pipeline.py           # Main script




# Indicators extraction using LLMs

This module runs LLM-based extraction of psycholinguistic indicators. Prompts are loaded from the `prompts/` directory, and the input dataset is taken from `data/`. The list of models used for extraction is defined directly in `pipeline.py`, so adjusting the models field there switches the models used in the full run.

For every model - prompt combination, the pipeline performs two steps:

1. Extraction — calls `src/run_extraction.py` with the chosen model and prompt.

2. Evaluation — runs `src/evaluate.py` on both downstream tasks and saves the results.

Dependencies are managed with uv. All required packages are specified in `pyproject.toml`, and exact versions are pinned in `uv.lock`.

To set up the environment and install dependencies:

```bash
uv sync
```

This creates (or updates) an isolated environment and installs all packages.

To execute the entire workflow end-to-end:

```bash
uv run pipeline.py
```
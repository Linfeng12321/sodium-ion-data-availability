# Sodium ion data availability

Tools for examining what sodium ion battery literature datasets contain, calculating field availability scores, and exploring capacity and retention with explicit chemistry context.

The repository accompanies *Characterizing literature-derived evidence on capacity and retention in sodium-ion battery material systems with explicit chemistry context*. It combines the Appendix A analysis tools with a reference implementation of the Setting C model described in the Methods and Appendix C.

| Start here | Purpose |
| --- | --- |
| [Quick start](#quick-start) | Install dependencies and check the software without research data |
| [Review guide](docs/reviewer_quickstart.md) | Locate the analysis, expected outputs and verification scope |
| [Paper correspondence](docs/manuscript_sm_crosswalk.md) | Find the code associated with each supported manuscript object |
| [Data guide](data/README.md) | Understand sources, fields and required local inputs |
| [Implementation notes](METHODS.md) | Inspect model settings and numerical conventions |

## What the tools provide

| Workflow | Inputs | Outputs |
| --- | --- | --- |
| Field availability | Processed main and auxiliary datasets | Item counts, grouped scores, rule scores and LaTeX tables |
| Visualization | Calculated dataset summaries | Coverage chart in Portable Network Graphics (PNG) and Portable Document Format (PDF) |
| Material system modeling | Reviewed labels, role schema, fixed partitions and similarity scores | Fitted preprocessing, selected checkpoints, predictions and evaluation summaries |

Availability scores range from 0 to 100. They average the terms specified in Appendix A; they are not percentages of complete records. The tools support inspection of literature data and offline analysis, rather than cell certification or operational control.

## Quick start

Use Python 3.12. The listed dependencies were checked on Windows with the central processing unit (CPU) build of PyTorch. No graphics processing unit (GPU) is needed for the software checks. Run commands from the repository root:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m sodium_materials --help
```

The test command runs 24 tests using small synthetic fixtures. It checks preprocessing, model behavior, losses, checkpoint selection, partition checks and metrics without training a model. Dependency versions are recorded in [requirements.txt](requirements.txt).

## Field availability analysis

Obtain the processed inputs described in the [data guide](data/README.md), and save them locally as `data/local/main.xlsx` and `data/local/auxiliary.csv`. These filenames are local conventions; the command accepts explicit input paths. Retain the original column names and values. For a workbook containing multiple sheets, supply `--sheet` with the intended sheet name.

```bash
python scripts/analyze_availability.py --main data/local/main.xlsx --auxiliary data/local/auxiliary.csv --output outputs/availability --check-manuscript
python scripts/plot_coverage.py --input outputs/availability/dataset_coverage.csv --output outputs/coverage
```

Use new output directories for each run. With the corresponding processed inputs, the comparison checks 32 reported aggregate values at their displayed precision. Reference values are used only for comparison, not to calculate or modify the outputs.

| Output | Content |
| --- | --- |
| `outputs/availability/item_availability.csv` | 18 item summaries with available and total record counts |
| `outputs/availability/dataset_coverage.csv` | Two dataset summaries with system, protocol and response scores |
| `outputs/availability/rule_scores.csv` | Six rule summaries, including their terms, sums and complements |
| `outputs/availability/availability_tables.tex` | Formatted summary tables |
| `outputs/coverage/coverage.png` and `coverage.pdf` | Visualization of the calculated grouped scores |

The default field mapping and record counts belong to the associated study. For another dataset, provide an appropriate configuration using `--config` and omit `--check-manuscript`.

## Model workflow

The model combines four role representations through multihead attention and predicts capacity and the four class Retention Milestone Index (RMI). Architecture and training settings are in [configs/model.json](configs/model.json); [METHODS.md](METHODS.md) explains their implementation.

Complete the schema and evaluation templates in `configs/` with the actual field mapping, preprocessing conventions, partition members, family labels, similarity definition and threshold. Save the completed files under `data/local/`. Supply a table of similarity scores for every training and test pair. The input requirements are detailed in the [data guide](data/README.md).

```bash
python -m sodium_materials validate --data data/local/records.csv --schema data/local/input_schema.json --splits data/local/splits.csv --protocol data/local/evaluation.json --similarity data/local/similarity.csv
```

After validation, training and prediction are available through these separate commands:

```bash
python -m sodium_materials train --data data/local/records.csv --schema data/local/input_schema.json --splits data/local/splits.csv --protocol data/local/evaluation.json --similarity data/local/similarity.csv --output outputs/setting_c
python -m sodium_materials predict --data data/local/new_records.csv --run outputs/setting_c/seed_42 --output outputs/prediction
```

The reference implementation follows the manuscript specification; it is not the archived experimental code. The reported predictive results have not been reproduced with this package. Individual research records, original split assignments, trained weights and experimental predictions are not included.

## Citation and license

Citation metadata for the software and associated manuscript are provided in [CITATION.cff](CITATION.cff). The manuscript is listed as unpublished; no article digital object identifier (DOI) is assigned here.

The original code and accompanying documentation are distributed under the [MIT License](LICENSE). Referenced datasets and installed dependencies retain their own terms. Dataset access routes are listed in [data/README.md](data/README.md). For software questions, use the repository's Issues tab.

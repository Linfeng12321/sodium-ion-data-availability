# Review guide

## Scope

| Question | Where to look |
| --- | --- |
| How are availability scores calculated? | `scripts/analyze_availability.py` and `configs/field_availability.json` |
| Which columns are used? | [Field dictionary](../data/field_dictionary.csv) |
| Which reported values can be checked? | [32 aggregate reference entries](../data/manuscript_values.csv) from Table 4 and Table A.2 |
| How is Setting C represented in code? | [Implementation notes](../METHODS.md) and `sodium_materials/` |
| Which paper objects are covered? | [Paper correspondence](manuscript_sm_crosswalk.md) |

## Software check

Follow the [installation commands](../README.md#quick-start) using Python 3.12, then run from the repository root:

```bash
python -m unittest discover -s tests -v
```

Expected result: 24 tests complete successfully. The fixtures are synthetic and test software behavior without fitting a model.

## Availability calculation

Obtain the two processed inputs through the [data guide](../data/README.md), then run:

```bash
python scripts/analyze_availability.py --main data/local/main.xlsx --auxiliary data/local/auxiliary.csv --output outputs/availability --check-manuscript
python scripts/plot_coverage.py --input outputs/availability/dataset_coverage.csv --output outputs/coverage
```

For the corresponding study inputs, the analysis reports `Matched 32 manuscript values at their displayed precision.` It writes three comma separated values (CSV) tables and a LaTeX table file. The plotting command writes `coverage.png` and `coverage.pdf`. Scores use a 0 to 100 scale rather than the percentage of complete records. The visualization summarizes calculated scores; it is not an original manuscript figure.

## Interpretation

The availability workflow has been checked against the corresponding processed tables. These tables are obtained separately and are not distributed in this repository. Model software tests do not establish reproduction of the reported predictive results. Refer to [README](../README.md#model-workflow) for the model input requirements and implementation scope.

Use [CITATION.cff](../CITATION.cff) for citation information and [LICENSE](../LICENSE) for software reuse terms.

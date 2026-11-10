# Paper correspondence

The manuscript contains the method appendices. This code release does not introduce a separate Supplementary Material document.

| Manuscript object | Repository files | Entry point | Output and verification scope |
| --- | --- | --- | --- |
| Table 4 and Appendix A, Table A.2 | `scripts/analyze_availability.py`, `configs/field_availability.json`, `data/manuscript_values.csv` | Availability command in [README](../README.md#field-availability-analysis) | Counts, grouped scores, rule scores and formatted tables; 32 aggregate entries checked at displayed precision using the corresponding processed inputs |
| Appendix A field definitions | `data/field_dictionary.csv` | Read alongside `configs/field_availability.json` | Exact source fields and grouping rules |
| Grouped availability visualization | `scripts/plot_coverage.py` | Plotting command in [README](../README.md#field-availability-analysis) | `coverage.png` and `coverage.pdf`; auxiliary visualization of calculated summaries, not an original manuscript figure |
| Methods and Appendix C, input processing | `sodium_materials/preprocessing.py`, `configs/input_schema.template.json` | Model validation command in [README](../README.md#model-workflow) | Fitted feature processing requires actual field assignments and records; unit tests cover missing values and training subset statistics |
| Methods and Appendix C, Setting C | `sodium_materials/model.py`, `configs/model.json` | Model training command in [README](../README.md#model-workflow) | Reference architecture; forward calculations and gradients checked with synthetic fixtures |
| Appendix C, training and selection | `sodium_materials/training.py` | Model training command in [README](../README.md#model-workflow) | Checkpoint and loss logic tested; full training not run for this release |
| Appendix C, partitions and evaluation | `sodium_materials/partitions.py`, `sodium_materials/metrics.py`, `configs/evaluation.template.json` | Model validation and training commands | Fixed membership and supplied similarity checks, followed by task metrics; original split members and similarity settings must be supplied separately |

The reference model has not been used to reproduce the paper's predictive performance tables. No code is assigned to manuscript objects outside the scope listed above.

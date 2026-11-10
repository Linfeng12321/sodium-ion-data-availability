# Data and input files

## Sources

| Input | Source | Access and role |
| --- | --- | --- |
| Main material system table | Oral et al., *Performance analysis of Na-ion batteries by machine learning*, Journal of Power Sources 549, 232126 (2022), [source publication](https://doi.org/10.1016/j.jpowsour.2022.232126) | The manuscript uses a processed version containing 1,229 records |
| Auxiliary cathode table | Gou et al., *A document-level information extraction pipeline for layered cathode materials for sodium-ion batteries*, Scientific Data 11, 372 (2024), [source publication](https://doi.org/10.1038/s41597-024-03196-1), [Figshare collection](https://doi.org/10.6084/m9.figshare.c.6828414.v1) | The processed subset contains 1,006 records and is used for contextual availability analysis, not model training |

The manuscript states that processed data and supporting materials can be requested from its corresponding author. Source terms apply. Original publications and collections are not substitutes for the exact processed tables or partition assignments. No source or processed individual records are redistributed here.

## Model input contract

The model reads a values only comma separated values (CSV) table. Use stable unique record identifiers. In the input schema, assign categorical and numerical field lists to each of the four roles. Targets and identifiers cannot also be input features. Keep the actual source headers, including intentional whitespace. Numerical values must already be expressed in consistent units; the code does not infer or convert units.

| Item | Required representation |
| --- | --- |
| Capacity target | Numeric peak discharge specific capacity in milliampere hours per gram; a configured missing marker where unavailable |
| Retention Milestone Index target | Reviewed integer 0, 1, 2 or 3; a missing marker where evidence is not usable |
| Missing markers | Explicit strings in the schema; comparison ignores case and surrounding whitespace, but Na cannot be a model missing marker |
| Categorical fields | Text labels retained in their original case after trimming whitespace |
| Numerical fields | Finite numbers or configured missing markers; no embedded units or formulas |
| Splits | Columns `record_id,split,reason`; one row per input record; excluded records require a reason |
| Similarity scores | Columns `train_id,test_id,similarity`; all training to test pairs, computed using the externally specified criterion |

The default evaluation template contains the manuscript's internal split counts of 860, 184 and 185. Supply the corresponding record assignments. For a family evaluation, provide its actual counts, the family column and the held out families. Compute similarity scores externally using the declared criterion; the code checks these scores against the supplied threshold.

## Included reference files

| File | Content |
| --- | --- |
| `field_dictionary.csv` | Input fields for the Appendix A availability calculation only; not the model role schema |
| `manuscript_values.csv` | 32 aggregate reference entries from Table 4 and Table A.2 |

The availability command expects the original processed main workbook or CSV export and the auxiliary CSV, with their exact field names. It counts processed entries, including imputed values, as described in Appendix A. Its reference values are comparison targets only, not substitute data or model outputs.

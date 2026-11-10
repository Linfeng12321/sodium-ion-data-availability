# Implementation notes

## Model settings

| Setting | Value |
| --- | --- |
| Component roles | Anode, cathode, electrolyte, testing |
| Role projection | Input dimension to 64, followed by Gaussian error linear unit (GELU) activation |
| Attention | One block, four heads, 16 dimensions per head |
| Feedforward widths | 64 to 128 to 64 |
| Output heads | 64 to 32 to 1 for capacity; 64 to 32 to 4 for Retention Milestone Index (RMI) classification |
| Aggregation | Mean across four updated role representations |
| Dropout | 0.10 at the attention, feedforward and output locations specified in Appendix C |
| Task losses | Mean squared error and cross entropy, each weighted 1.0 |
| Optimizer | Adaptive moment estimation with decoupled weight decay (AdamW) |
| Learning rate | 0.001, constant |
| Weight decay | 0.0001 on linear and attention projection weights |
| Batch size and maximum epochs | 32 and 200 |
| Early stopping | 20 epochs without validation improvement greater than 0.0001 |
| Repeated runs | Seeds 42, 43, 44, 45 and 46 with fixed partitions |

The complete numerical configuration is in [configs/model.json](configs/model.json). Best checkpoint selection uses the lowest observed validation loss, including improvements smaller than the early stopping threshold. Validation losses are averaged over all valid records for each task before the task losses are added.

## Method mapping

| Manuscript description | Implementation |
| --- | --- |
| Separate role encoders | Independent linear projections and Gaussian error linear unit activation |
| Full component attention | Four tokens with no causal mask; one four head attention block |
| Residual connections | Attention output addition followed by layer normalization; feedforward output addition followed by layer normalization |
| Initial values | Xavier uniform projection weights, zero biases, normal role embeddings with standard deviation 0.02, unit normalization scales |
| Missing numerical data | Training median imputation, separate unscaled binary indicators; fully missing fields use zero imputation |
| Unseen categories | Zero indicators for the corresponding categorical field |
| Missing targets | Independent capacity and retention masks; batches without either label are skipped |
| Capacity scale | Fitted to valid training targets only, inverse transformed for evaluation |
| Model selection | Strictly lowest validation loss; earlier checkpoint retained on ties |
| Early stopping | Separate reference loss and patience counter, as in the manuscript algorithm |
| Family evaluation | Supplied family labels and fixed record assignments; no random family selection |

## Explicit implementation choices

The following table specifies the implementation choices and settings supplied with the input data.

| Choice | Treatment |
| --- | --- |
| Numerical statistics order | User specifies `before_imputation` or `after_imputation`; numerical standard deviation uses the population convention |
| Capacity standard deviation | User specifies `capacity_standard_deviation_ddof` as 0 or 1 |
| Field assignment and type | Supplied in the input schema; each field belongs to exactly one role |
| Category order | Sorted training vocabulary with a dedicated missing index; original category case retained |
| Projection initialization | Query, key and value matrices initialized separately with Xavier uniform initialization |
| Activation evaluation | Exact Gaussian error linear unit, without its optional approximation |
| Execution | Central processing unit, one computation thread, deterministic PyTorch operations |
| Batch order | Dedicated PyTorch generator initialized from the run seed |
| Optimizer execution | Standard AdamW with `amsgrad=False` and `foreach=False`; no scheduler, clipping or gradient accumulation |
| Constant capacity target | Reject a training set with zero target standard deviation |
| Family isolation | Held out test families are excluded from validation as well as training to avoid selection leakage |
| Similarity | Validate a complete supplied score matrix against the declared strict threshold; partition assignments are supplied separately |

`milestone_from_anchor` assigns a tier only after evidence validity has been supplied explicitly. It does not parse ambiguous retention statements or establish the validity of an 80% threshold from text. The training command consumes already reviewed class labels.

## Software references

The implementation uses the documented attention interface, optimizer and state dictionary serialization:

- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html)
- [PyTorch AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)
- [PyTorch serialization](https://docs.pytorch.org/docs/2.15/notes/serialization.html)

## Evaluation outputs

| Metric | Definition used by the code |
| --- | --- |
| Mean absolute error (MAE) | Mean absolute capacity error over records with valid targets |
| Normalized mean absolute error (NMAE) | 100 times MAE divided by the observed capacity range of the evaluated subset |
| Coefficient of determination (R²) | One minus the residual sum of squares divided by the test target sum of squares about its mean |
| Macro averaged F1 | Unweighted average across all four RMI classes; a class with a zero denominator contributes zero |
| Repeated run summary | Mean and sample standard deviation across completed runs |

Metrics with undefined target range or variance are recorded as `null`. Task metrics use their respective valid label masks.


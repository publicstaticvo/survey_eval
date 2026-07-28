# Baselines 2 Verified Weakness Taxonomy

- Verified input rows loaded: 2030
- Included after removing intrinsic input-format defects: 1934
- Excluded intrinsic input-format defects: 96
- Pending unverified cc_surveys rows found locally: 515

## Category Metrics

| # | Category | Total | Valid | Invalid | Accuracy |
|---:|---|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 72 | 42 | 30 | 58.33% |
| 2 | Comparison insufficient | 428 | 248 | 180 | 57.94% |
| 3 | Synthesis / original viewpoint insufficient | 59 | 26 | 33 | 44.07% |
| 4 | Scope / inclusion-criteria declaration missing | 96 | 78 | 18 | 81.25% |
| 5 | Contribution statement missing | 35 | 20 | 15 | 57.14% |
| 6 | Internal inconsistency | 65 | 24 | 41 | 36.92% |
| 7 | Hallucination | 45 | 18 | 27 | 40.00% |
| 8 | Missing specific references | 32 | 13 | 19 | 40.62% |
| 9 | Missing specific topics | 94 | 46 | 48 | 48.94% |
| 10 | Taxonomy or framework problem | 67 | 25 | 42 | 37.31% |
| 11 | Evidence support insufficient | 677 | 250 | 427 | 36.93% |
| 12 | Writing clarity and presentation | 167 | 66 | 101 | 39.52% |
| 13 | Visualization deficiency | 85 | 28 | 57 | 32.94% |
| 14 | Contribution novelty problem / venue mismatch | 12 | 5 | 7 | 41.67% |

## Exclusion Counts

- input_extraction_rendering: 8
- input_format_metadata: 20
- input_format_reference_metadata: 68

## Notes

- Category labels are assigned to already verified weakness records only; this step does not recompute validity.
- Author/byline/date metadata, BibTeX/reference formatting, and extraction/rendering artifacts are excluded as requested.
- Locally available cc_surveys manual verdicts cover 58 rows; 515 cc_surveys rows remain pending in the current workspace.

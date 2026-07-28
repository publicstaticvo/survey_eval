# Baselines 2 Verified Weakness Taxonomy (Audited)

- Verified input rows loaded: 2030
- Included after removing intrinsic input-format defects: 1918
- Excluded intrinsic input-format defects: 112
- Pending unverified cc_surveys rows found locally: 515

## Category Metrics

| # | Category | Total | Valid | Invalid | Accuracy |
|---:|---|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 72 | 34 | 38 | 47.22% |
| 2 | Comparison insufficient | 344 | 206 | 138 | 59.88% |
| 3 | Synthesis / original viewpoint insufficient | 50 | 23 | 27 | 46.00% |
| 4 | Scope / inclusion-criteria declaration missing | 96 | 78 | 18 | 81.25% |
| 5 | Contribution statement missing | 35 | 20 | 15 | 57.14% |
| 6 | Internal inconsistency | 59 | 22 | 37 | 37.29% |
| 7 | Hallucination | 49 | 19 | 30 | 38.78% |
| 8 | Missing specific references | 69 | 26 | 43 | 37.68% |
| 9 | Missing specific topics | 169 | 92 | 77 | 54.44% |
| 10 | Taxonomy or framework problem | 71 | 27 | 44 | 38.03% |
| 11 | Evidence support insufficient | 661 | 236 | 425 | 35.70% |
| 12 | Writing clarity and presentation | 144 | 60 | 84 | 41.67% |
| 13 | Visualization deficiency | 87 | 29 | 58 | 33.33% |
| 14 | Contribution novelty problem / venue mismatch | 12 | 5 | 7 | 41.67% |

## Exclusion Counts

- input_extraction_rendering: 8
- input_format_metadata: 30
- input_format_reference_metadata: 74

## Audit Notes

- Category 11 is treated as argument-rigor / analysis-support insufficient.
- Missing-topic claims are Category 9 even when the omitted material includes benchmarks or methods.
- Benchmark/performance cross-method contrast claims are Category 2.
- Technical false claims are Category 7.
- Audience/readership and prose/notation issues are Category 12.
- Author/byline/date metadata, BibTeX/reference formatting, and extraction/rendering artifacts are excluded as requested.

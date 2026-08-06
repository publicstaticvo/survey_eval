# Calibrated Threshold Fitting Results

Data: 169 development papers for per-category reviewer-observed concern thresholds; 121 papers with binary accept/reject labels for score sanity check.

## Per-Category Thresholds

| Category | Positives | Score-bearing | Thresholds | CV precision LB | CV recall | CV F0.5 | Decision |
|---|---:|:---:|---|---:|---:|---:|---|
| gap | 83 | true | `{"gap_coverage": 0.13230769230769235, "gap_volume": 0.19545227386306846}` | 0.491 | 0.663 | 0.518 | stable enough for score-capping calibration |
| contrast | 96 | true | `{"contrast_coverage": 0.5151515151515151, "contrast_volume": 0.6835443037974683}` | 0.583 | 0.729 | 0.608 | stable enough for score-capping calibration |
| synthesis | 127 | true | `{"synthesis_coverage": 0.1555555555555555, "synthesis_volume": 0.0}` | 0.767 | 0.724 | 0.758 | stable enough for score-capping calibration |
| scope | 69 | true | `{"scope_exists": 1.0}` | 0.526 | 0.145 | 0.345 | stable enough for score-capping calibration |
| contribution | 73 | false | `{"contribution_exists": 0.40000000000000036}` | 0.222 | 0.027 | 0.092 | held-out observed precision lower bound is too low |
| reference | 103 | true | `{"reference_topic_coverage": 0.7169642857142857, "citations_per_topic": 1.881639928698752}` | 0.626 | 0.748 | 0.647 | stable enough for score-capping calibration |
| topic | 140 | false | `{"topic_count": 66.0}` | 0.819 | 0.743 | 0.802 | observed concern is saturated; reviewer comments do not provide a useful threshold |

## Accept/Reject Sanity Check

Score-bearing categories: gap, contrast, synthesis, scope, reference.
AUROC = 0.657 (95% CI 0.511-0.803).
PR-AUC = 0.835 (95% CI 0.745-0.929).
Brier = 0.466; this should be interpreted cautiously because the cap is not calibrated as an acceptance probability.
Calibration intercept = 1.900; slope = 0.093.

Interpretation: thresholds are reliable enough for score-capping on gap, contrast, synthesis, scope, and reference. Contribution grounding is not score-bearing under the current local proxy. Topic coverage is too saturated in reviewer labels to yield a useful threshold and should remain reader-facing until a better domain-coverage feature is available.
# Baselines 2 Generator Problem Rates

Scope: included classified rows in `baselines_2_verified_weakness_taxonomy_audited.json`; input-format/extraction artifacts already excluded by that file. Rates below use issue count as denominator, not paper count. The 515 pending cc_surveys rows are not included.

## Overall by generator

| generator | total issues | valid issues | accuracy |
|---|---:|---:|---:|
| sgen_surveys | 729 | 272 | 37.3% |
| codex_surveys | 632 | 339 | 53.6% |

## Category rates by generator

| id | category | sgen n | sgen rate | sgen valid | sgen acc | codex n | codex rate | codex valid | codex acc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 22 | 3.0% | 9 | 40.9% | 28 | 4.4% | 15 | 53.6% |
| 2 | Comparison insufficient | 117 | 16.0% | 56 | 47.9% | 139 | 22.0% | 89 | 64.0% |
| 3 | Synthesis / original viewpoint insufficient | 24 | 3.3% | 11 | 45.8% | 8 | 1.3% | 2 | 25.0% |
| 4 | Scope / inclusion-criteria declaration missing | 37 | 5.1% | 25 | 67.6% | 29 | 4.6% | 25 | 86.2% |
| 5 | Contribution statement missing | 19 | 2.6% | 8 | 42.1% | 7 | 1.1% | 6 | 85.7% |
| 6 | Internal inconsistency | 28 | 3.8% | 7 | 25.0% | 8 | 1.3% | 6 | 75.0% |
| 7 | Hallucination | 33 | 4.5% | 13 | 39.4% | 5 | 0.8% | 1 | 20.0% |
| 8 | Missing specific references | 16 | 2.2% | 7 | 43.8% | 34 | 5.4% | 13 | 38.2% |
| 9 | Missing specific topics | 26 | 3.6% | 10 | 38.5% | 111 | 17.6% | 70 | 63.1% |
| 10 | Taxonomy or framework problem | 15 | 2.1% | 7 | 46.7% | 23 | 3.6% | 6 | 26.1% |
| 11 | Evidence support insufficient | 301 | 41.3% | 92 | 30.6% | 163 | 25.8% | 69 | 42.3% |
| 12 | Writing clarity and presentation | 49 | 6.7% | 17 | 34.7% | 54 | 8.5% | 26 | 48.1% |
| 13 | Visualization deficiency | 40 | 5.5% | 9 | 22.5% | 21 | 3.3% | 9 | 42.9% |
| 14 | Contribution novelty problem / venue mismatch | 2 | 0.3% | 1 | 50.0% | 2 | 0.3% | 2 | 100.0% |

## Category rates by generator and evaluator

### sgen_surveys / llm (n=385)

| id | category | n | rate | valid | acc |
|---:|---|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 10 | 2.6% | 6 | 60.0% |
| 2 | Comparison insufficient | 42 | 10.9% | 17 | 40.5% |
| 3 | Synthesis / original viewpoint insufficient | 10 | 2.6% | 3 | 30.0% |
| 4 | Scope / inclusion-criteria declaration missing | 20 | 5.2% | 13 | 65.0% |
| 5 | Contribution statement missing | 12 | 3.1% | 4 | 33.3% |
| 6 | Internal inconsistency | 19 | 4.9% | 5 | 26.3% |
| 7 | Hallucination | 17 | 4.4% | 5 | 29.4% |
| 8 | Missing specific references | 9 | 2.3% | 2 | 22.2% |
| 9 | Missing specific topics | 4 | 1.0% | 1 | 25.0% |
| 10 | Taxonomy or framework problem | 6 | 1.6% | 4 | 66.7% |
| 11 | Evidence support insufficient | 189 | 49.1% | 54 | 28.6% |
| 12 | Writing clarity and presentation | 21 | 5.5% | 4 | 19.0% |
| 13 | Visualization deficiency | 24 | 6.2% | 2 | 8.3% |
| 14 | Contribution novelty problem / venue mismatch | 2 | 0.5% | 1 | 50.0% |

### sgen_surveys / cc (n=344)

| id | category | n | rate | valid | acc |
|---:|---|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 12 | 3.5% | 3 | 25.0% |
| 2 | Comparison insufficient | 75 | 21.8% | 39 | 52.0% |
| 3 | Synthesis / original viewpoint insufficient | 14 | 4.1% | 8 | 57.1% |
| 4 | Scope / inclusion-criteria declaration missing | 17 | 4.9% | 12 | 70.6% |
| 5 | Contribution statement missing | 7 | 2.0% | 4 | 57.1% |
| 6 | Internal inconsistency | 9 | 2.6% | 2 | 22.2% |
| 7 | Hallucination | 16 | 4.7% | 8 | 50.0% |
| 8 | Missing specific references | 7 | 2.0% | 5 | 71.4% |
| 9 | Missing specific topics | 22 | 6.4% | 9 | 40.9% |
| 10 | Taxonomy or framework problem | 9 | 2.6% | 3 | 33.3% |
| 11 | Evidence support insufficient | 112 | 32.6% | 38 | 33.9% |
| 12 | Writing clarity and presentation | 28 | 8.1% | 13 | 46.4% |
| 13 | Visualization deficiency | 16 | 4.7% | 7 | 43.8% |
| 14 | Contribution novelty problem / venue mismatch | 0 | 0.0% | 0 | 0.0% |

### codex_surveys / llm (n=324)

| id | category | n | rate | valid | acc |
|---:|---|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 18 | 5.6% | 9 | 50.0% |
| 2 | Comparison insufficient | 60 | 18.5% | 37 | 61.7% |
| 3 | Synthesis / original viewpoint insufficient | 5 | 1.5% | 0 | 0.0% |
| 4 | Scope / inclusion-criteria declaration missing | 14 | 4.3% | 11 | 78.6% |
| 5 | Contribution statement missing | 4 | 1.2% | 3 | 75.0% |
| 6 | Internal inconsistency | 4 | 1.2% | 2 | 50.0% |
| 7 | Hallucination | 2 | 0.6% | 0 | 0.0% |
| 8 | Missing specific references | 11 | 3.4% | 3 | 27.3% |
| 9 | Missing specific topics | 38 | 11.7% | 25 | 65.8% |
| 10 | Taxonomy or framework problem | 15 | 4.6% | 2 | 13.3% |
| 11 | Evidence support insufficient | 109 | 33.6% | 42 | 38.5% |
| 12 | Writing clarity and presentation | 30 | 9.3% | 10 | 33.3% |
| 13 | Visualization deficiency | 13 | 4.0% | 4 | 30.8% |
| 14 | Contribution novelty problem / venue mismatch | 1 | 0.3% | 1 | 100.0% |

### codex_surveys / cc (n=308)

| id | category | n | rate | valid | acc |
|---:|---|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 10 | 3.2% | 6 | 60.0% |
| 2 | Comparison insufficient | 79 | 25.6% | 52 | 65.8% |
| 3 | Synthesis / original viewpoint insufficient | 3 | 1.0% | 2 | 66.7% |
| 4 | Scope / inclusion-criteria declaration missing | 15 | 4.9% | 14 | 93.3% |
| 5 | Contribution statement missing | 3 | 1.0% | 3 | 100.0% |
| 6 | Internal inconsistency | 4 | 1.3% | 4 | 100.0% |
| 7 | Hallucination | 3 | 1.0% | 1 | 33.3% |
| 8 | Missing specific references | 23 | 7.5% | 10 | 43.5% |
| 9 | Missing specific topics | 73 | 23.7% | 45 | 61.6% |
| 10 | Taxonomy or framework problem | 8 | 2.6% | 4 | 50.0% |
| 11 | Evidence support insufficient | 54 | 17.5% | 27 | 50.0% |
| 12 | Writing clarity and presentation | 24 | 7.8% | 16 | 66.7% |
| 13 | Visualization deficiency | 8 | 2.6% | 5 | 62.5% |
| 14 | Contribution novelty problem / venue mismatch | 1 | 0.3% | 1 | 100.0% |

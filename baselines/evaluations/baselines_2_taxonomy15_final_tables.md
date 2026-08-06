# Baselines 2 Final Taxonomy-15 Tables

Taxonomy 15 is `Input-format / parsing artifact`; it is not counted as an effective survey weakness. Valid/effective counts use existing manual/meta validity labels, except taxonomy 15 is forced to invalid for survey-quality metrics.

Rows included: 1918 classified rows. Pending cc_surveys rows from prior run remain excluded.

## Table 1. sgen / codex problem frequency

| id | category | sgen count | sgen rate | codex count | codex rate |
|---:|---|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 18 | 2.5% | 24 | 3.8% |
| 2 | Comparison insufficient | 116 | 15.9% | 137 | 21.7% |
| 3 | Synthesis / original viewpoint insufficient | 42 | 5.8% | 32 | 5.1% |
| 4 | Scope / inclusion-criteria declaration missing | 37 | 5.1% | 31 | 4.9% |
| 5 | Contribution statement missing | 3 | 0.4% | 7 | 1.1% |
| 6 | Internal inconsistency | 46 | 6.3% | 8 | 1.3% |
| 7 | Hallucination | 40 | 5.5% | 5 | 0.8% |
| 8 | Missing specific references | 14 | 1.9% | 22 | 3.5% |
| 9 | Missing specific topics | 16 | 2.2% | 70 | 11.1% |
| 10 | Taxonomy or framework problem | 42 | 5.8% | 29 | 4.6% |
| 11 | Evidence support insufficient / argumentation not rigorous | 101 | 13.9% | 90 | 14.2% |
| 12 | Writing clarity and presentation | 99 | 13.6% | 78 | 12.3% |
| 13 | Visualization deficiency | 95 | 13.0% | 64 | 10.1% |
| 14 | Contribution novelty problem / venue mismatch | 15 | 2.1% | 12 | 1.9% |
| 15 | Input-format / parsing artifact | 45 | 6.2% | 23 | 3.6% |

## Table 2. detector + prompt effective frequency and accuracy

Each cell is `valid / total (accuracy)`. This table uses all recoded classified rows, not only sgen/codex.
| id | category | cc+arise | cc+plain | cc+trustsurvey | llm+arise | llm+plain | llm+trustsurvey |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Gap and future-work discussion insufficient | 3/5 (60.0%) | 13/21 (61.9%) | 7/13 (53.8%) | 3/4 (75.0%) | 7/13 (53.8%) | 7/10 (70.0%) |
| 2 | Comparison insufficient | 25/47 (53.2%) | 52/83 (62.7%) | 41/67 (61.2%) | 18/26 (69.2%) | 31/66 (47.0%) | 28/56 (50.0%) |
| 3 | Synthesis / original viewpoint insufficient | 10/24 (41.7%) | 8/24 (33.3%) | 8/13 (61.5%) | 4/12 (33.3%) | 7/17 (41.2%) | 6/10 (60.0%) |
| 4 | Scope / inclusion-criteria declaration missing | 4/6 (66.7%) | 5/6 (83.3%) | 35/41 (85.4%) | 1/2 (50.0%) | 0/2 (0.0%) | 37/44 (84.1%) |
| 5 | Contribution statement missing | 0/0 (0.0%) | 1/1 (100.0%) | 5/7 (71.4%) | 0/0 (0.0%) | 0/1 (0.0%) | 4/7 (57.1%) |
| 6 | Internal inconsistency | 1/1 (100.0%) | 2/8 (25.0%) | 13/27 (48.1%) | 2/6 (33.3%) | 4/11 (36.4%) | 10/28 (35.7%) |
| 7 | Hallucination | 2/4 (50.0%) | 3/5 (60.0%) | 15/27 (55.6%) | 0/0 (0.0%) | 1/3 (33.3%) | 4/20 (20.0%) |
| 8 | Missing specific references | 2/4 (50.0%) | 1/5 (20.0%) | 12/22 (54.5%) | 0/0 (0.0%) | 0/0 (0.0%) | 7/20 (35.0%) |
| 9 | Missing specific topics | 9/16 (56.2%) | 18/34 (52.9%) | 10/30 (33.3%) | 3/4 (75.0%) | 9/15 (60.0%) | 7/12 (58.3%) |
| 10 | Taxonomy or framework problem | 6/15 (40.0%) | 12/26 (46.2%) | 13/21 (61.9%) | 7/14 (50.0%) | 4/14 (28.6%) | 9/16 (56.2%) |
| 11 | Evidence support insufficient / argumentation not rigorous | 24/63 (38.1%) | 26/71 (36.6%) | 8/22 (36.4%) | 16/32 (50.0%) | 24/63 (38.1%) | 15/41 (36.6%) |
| 12 | Writing clarity and presentation | 22/52 (42.3%) | 23/62 (37.1%) | 4/12 (33.3%) | 9/26 (34.6%) | 25/68 (36.8%) | 7/23 (30.4%) |
| 13 | Visualization deficiency | 22/43 (51.2%) | 14/29 (48.3%) | 5/20 (25.0%) | 13/54 (24.1%) | 9/38 (23.7%) | 10/31 (32.3%) |
| 14 | Contribution novelty problem / venue mismatch | 3/12 (25.0%) | 4/8 (50.0%) | 3/5 (60.0%) | 4/13 (30.8%) | 2/7 (28.6%) | 0/0 (0.0%) |
| 15 | Input-format / parsing artifact | 0/0 (0.0%) | 0/0 (0.0%) | 0/1 (0.0%) | 0/36 (0.0%) | 0/35 (0.0%) | 0/15 (0.0%) |

## Recoding change summary

| old -> new | count |
|---|---:|
| 11 Evidence support insufficient / argumentation not rigorous -> 12 Writing clarity and presentation | 118 |
| 11 Evidence support insufficient / argumentation not rigorous -> 13 Visualization deficiency | 76 |
| 11 Evidence support insufficient / argumentation not rigorous -> 15 Input-format / parsing artifact | 58 |
| 11 Evidence support insufficient / argumentation not rigorous -> 2 Comparison insufficient | 38 |
| 12 Writing clarity and presentation -> 13 Visualization deficiency | 29 |
| 9 Missing specific topics -> 12 Writing clarity and presentation | 21 |
| 2 Comparison insufficient -> 10 Taxonomy or framework problem | 19 |
| 10 Taxonomy or framework problem -> 3 Synthesis / original viewpoint insufficient | 18 |
| 11 Evidence support insufficient / argumentation not rigorous -> 6 Internal inconsistency | 18 |
| 11 Evidence support insufficient / argumentation not rigorous -> 10 Taxonomy or framework problem | 17 |
| 11 Evidence support insufficient / argumentation not rigorous -> 3 Synthesis / original viewpoint insufficient | 16 |
| 11 Evidence support insufficient / argumentation not rigorous -> 14 Contribution novelty problem / venue mismatch | 16 |
| 2 Comparison insufficient -> 3 Synthesis / original viewpoint insufficient | 14 |
| 2 Comparison insufficient -> 13 Visualization deficiency | 14 |
| 9 Missing specific topics -> 2 Comparison insufficient | 14 |
| 5 Contribution statement missing -> 10 Taxonomy or framework problem | 12 |
| 8 Missing specific references -> 9 Missing specific topics | 12 |
| 9 Missing specific topics -> 1 Gap and future-work discussion insufficient | 10 |
| 9 Missing specific topics -> 13 Visualization deficiency | 9 |
| 13 Visualization deficiency -> 15 Input-format / parsing artifact | 9 |
| 1 Gap and future-work discussion insufficient -> 15 Input-format / parsing artifact | 8 |
| 11 Evidence support insufficient / argumentation not rigorous -> 7 Hallucination | 8 |
| 9 Missing specific topics -> 3 Synthesis / original viewpoint insufficient | 7 |
| 9 Missing specific topics -> 10 Taxonomy or framework problem | 7 |
| 2 Comparison insufficient -> 6 Internal inconsistency | 6 |
| 3 Synthesis / original viewpoint insufficient -> 13 Visualization deficiency | 6 |
| 2 Comparison insufficient -> 14 Contribution novelty problem / venue mismatch | 5 |
| 9 Missing specific topics -> 11 Evidence support insufficient / argumentation not rigorous | 5 |
| 10 Taxonomy or framework problem -> 14 Contribution novelty problem / venue mismatch | 5 |
| 1 Gap and future-work discussion insufficient -> 13 Visualization deficiency | 4 |
| 8 Missing specific references -> 12 Writing clarity and presentation | 4 |
| 8 Missing specific references -> 13 Visualization deficiency | 4 |
| 12 Writing clarity and presentation -> 3 Synthesis / original viewpoint insufficient | 4 |
| 12 Writing clarity and presentation -> 15 Input-format / parsing artifact | 4 |
| 1 Gap and future-work discussion insufficient -> 10 Taxonomy or framework problem | 3 |
| 2 Comparison insufficient -> 4 Scope / inclusion-criteria declaration missing | 3 |
| 2 Comparison insufficient -> 15 Input-format / parsing artifact | 3 |
| 3 Synthesis / original viewpoint insufficient -> 1 Gap and future-work discussion insufficient | 3 |
| 5 Contribution statement missing -> 3 Synthesis / original viewpoint insufficient | 3 |
| 10 Taxonomy or framework problem -> 4 Scope / inclusion-criteria declaration missing | 3 |
| 11 Evidence support insufficient / argumentation not rigorous -> 8 Missing specific references | 3 |
| 12 Writing clarity and presentation -> 2 Comparison insufficient | 3 |
| 12 Writing clarity and presentation -> 10 Taxonomy or framework problem | 3 |
| 13 Visualization deficiency -> 2 Comparison insufficient | 3 |
| 1 Gap and future-work discussion insufficient -> 3 Synthesis / original viewpoint insufficient | 2 |
| 1 Gap and future-work discussion insufficient -> 12 Writing clarity and presentation | 2 |
| 1 Gap and future-work discussion insufficient -> 14 Contribution novelty problem / venue mismatch | 2 |
| 3 Synthesis / original viewpoint insufficient -> 2 Comparison insufficient | 2 |
| 3 Synthesis / original viewpoint insufficient -> 9 Missing specific topics | 2 |
| 3 Synthesis / original viewpoint insufficient -> 10 Taxonomy or framework problem | 2 |
| 4 Scope / inclusion-criteria declaration missing -> 6 Internal inconsistency | 2 |
| 4 Scope / inclusion-criteria declaration missing -> 12 Writing clarity and presentation | 2 |
| 5 Contribution statement missing -> 6 Internal inconsistency | 2 |
| 5 Contribution statement missing -> 12 Writing clarity and presentation | 2 |
| 5 Contribution statement missing -> 14 Contribution novelty problem / venue mismatch | 2 |
| 6 Internal inconsistency -> 3 Synthesis / original viewpoint insufficient | 2 |
| 6 Internal inconsistency -> 4 Scope / inclusion-criteria declaration missing | 2 |
| 6 Internal inconsistency -> 15 Input-format / parsing artifact | 2 |
| 10 Taxonomy or framework problem -> 15 Input-format / parsing artifact | 2 |
| 11 Evidence support insufficient / argumentation not rigorous -> 1 Gap and future-work discussion insufficient | 2 |
| 11 Evidence support insufficient / argumentation not rigorous -> 4 Scope / inclusion-criteria declaration missing | 2 |
| 11 Evidence support insufficient / argumentation not rigorous -> 5 Contribution statement missing | 2 |
| 11 Evidence support insufficient / argumentation not rigorous -> 9 Missing specific topics | 2 |
| 12 Writing clarity and presentation -> 8 Missing specific references | 2 |
| 12 Writing clarity and presentation -> 14 Contribution novelty problem / venue mismatch | 2 |
| 13 Visualization deficiency -> 14 Contribution novelty problem / venue mismatch | 2 |
| 14 Contribution novelty problem / venue mismatch -> 2 Comparison insufficient | 2 |
| 1 Gap and future-work discussion insufficient -> 2 Comparison insufficient | 1 |
| 3 Synthesis / original viewpoint insufficient -> 6 Internal inconsistency | 1 |
| 4 Scope / inclusion-criteria declaration missing -> 5 Contribution statement missing | 1 |
| 4 Scope / inclusion-criteria declaration missing -> 13 Visualization deficiency | 1 |
| 5 Contribution statement missing -> 4 Scope / inclusion-criteria declaration missing | 1 |
| 6 Internal inconsistency -> 1 Gap and future-work discussion insufficient | 1 |
| 6 Internal inconsistency -> 2 Comparison insufficient | 1 |
| 6 Internal inconsistency -> 7 Hallucination | 1 |
| 8 Missing specific references -> 2 Comparison insufficient | 1 |
| 8 Missing specific references -> 11 Evidence support insufficient / argumentation not rigorous | 1 |
| 8 Missing specific references -> 15 Input-format / parsing artifact | 1 |
| 9 Missing specific topics -> 14 Contribution novelty problem / venue mismatch | 1 |
| 12 Writing clarity and presentation -> 6 Internal inconsistency | 1 |
| 12 Writing clarity and presentation -> 7 Hallucination | 1 |
| 12 Writing clarity and presentation -> 11 Evidence support insufficient / argumentation not rigorous | 1 |
| 13 Visualization deficiency -> 6 Internal inconsistency | 1 |

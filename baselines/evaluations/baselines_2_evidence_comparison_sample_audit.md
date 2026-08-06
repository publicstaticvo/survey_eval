# Evidence/Comparison Sample Category Audit

Source file: `baselines_2_verified_weakness_taxonomy_audited.json`.
Sampling: fixed random seed 20260801, 20 rows from taxonomy 11 and 20 rows from taxonomy 2.
This audit checks category assignment only; it does not re-judge whether the original weakness is valid against the survey.

## Evidence support insufficient sample

Result: only 3/20 clearly match Evidence support insufficient. This category is substantially contaminated by input-format artifacts, hallucination/factual-error claims, internal inconsistency, visualization/table issues, and missing-topic/reference issues.

| sample | issue summary | matches Evidence support? | reason | suggested category if wrong |
|---|---|---:|---|---|
| E1 | Title mentions self-reflection but body allegedly omits it. | No | This is a promise/body mismatch, not weak argument support. | 6 Internal inconsistency |
| E2 | Mathematical expression appears truncated. | No | This is a LaTeX/input rendering artifact, not a survey-content weakness. | Exclude: input extraction/rendering |
| E3 | All references appear as Unknown authors / Untitled. | No | This is reference metadata/rendering, exactly the artifact type that should be removed. | Exclude: input/reference metadata |
| E4 | Citation misattribution and inappropriate references. | No | Citation misattribution is a cited-source factual error if concrete. | 7 Hallucination |
| E5 | Missing quantitative overview and summary table of surveyed works. | No | The complaint is absence of comparative summary across works. | 2 Comparison insufficient |
| E6 | Inconsistent depth across summaries; some works detailed, others brief. | Yes | This is about uneven analytical depth/support. | 11 Evidence support insufficient |
| E7 | Future-dated references and AI-generated citation patterns. | No | Citation authenticity/fabrication belongs to hallucination/citation integrity. | 7 Hallucination |
| E8 | Conclusion is under-developed and fails to synthesize material / future vision. | No | Main complaint is lack of synthesis and concluding integration. | 3 Synthesis / original viewpoint insufficient |
| E9 | Reference formatting inconsistencies in author names. | No | Pure bibliography formatting/input metadata issue. | Exclude: input/reference metadata |
| E10 | Audio/speech/sketch retrieval sections need deeper technical treatment. | Yes | This is a depth-of-treatment judgment. | 11 Evidence support insufficient |
| E11 | Content repetition on over-smoothing / over-squashing / 1-WL. | No | Repetition is organization/writing quality, not evidence support. | 12 Writing clarity and presentation |
| E12 | MDP transition formulation is mathematically incorrect. | No | This is a factual/technical correctness accusation. | 7 Hallucination |
| E13 | Key methods missing: FOL, STRIPS planners, related works. | No | It names omitted method families. | 9 Missing specific topics |
| E14 | References incomplete/corrupted as Unknown authors / Untitled. | No | Input/reference metadata artifact. | Exclude: input/reference metadata |
| E15 | Weather Prediction Tasks section transitions abruptly. | No | This is organization/flow. | 12 Writing clarity and presentation |
| E16 | Abstract is extremely vague and fails to summarize themes/contributions/findings. | No | This is abstract/presentation clarity, possibly contribution statement, not evidence support. | 12 Writing clarity and presentation |
| E17 | Table caption categories do not map clearly to table items. | No | The defect is in table design/integration. | 13 Visualization deficiency |
| E18 | Paper text cuts off mid-sentence and later sections absent. | No | This looks like source extraction/rendering incompleteness, not content evidence. | Exclude: input extraction/rendering |
| E19 | Skeleton recognition and temporal localization are shallow compared with RGB classification. | Yes | This is uneven depth/argument support across topics. | 11 Evidence support insufficient |
| E20 | Abstract claims 341 papers but fewer than 50 references / formatting problems. | No | Paper-count/reference-count contradiction is internal inconsistency; formatting part is artifact. | 6 Internal inconsistency, partly exclude reference metadata |

## Comparison insufficient sample

Result: 17/20 match Comparison insufficient. This category is much cleaner than Evidence support, but still has contamination from novelty comparison, visualization readability, and input-rendering/table-missing artifacts.

| sample | issue summary | matches Comparison? | reason | suggested category if wrong |
|---|---|---:|---|---|
| C1 | No quantitative summary tables comparing method performance across benchmarks. | Yes | Directly asks for cross-method performance comparison. | 2 Comparison insufficient |
| C2 | Limited quantitative comparative performance analysis across action-recognition methods. | Yes | Explicit lack of quantitative method comparison. | 2 Comparison insufficient |
| C3 | No comparison table/systematic benchmark analysis for UCF/Kinetics methods. | Yes | Direct comparative benchmark gap. | 2 Comparison insufficient |
| C4 | No quantitative comparison/summary table of graph-classification method performance. | Yes | Direct method-vs-method benchmark comparison complaint. | 2 Comparison insufficient |
| C5 | Performance comparison tables or quantitative benchmarks absent. | Yes | Direct absence of empirical comparison. | 2 Comparison insufficient |
| C6 | Lacks comparative evaluation table for accuracy/throughput/memory/latency. | Yes | Even if invalid due unreadable source, the semantic issue is comparison. | 2 Comparison insufficient |
| C7 | Strong novelty claims not compared against prior surveys. | No | This is about defending novelty/distinctness relative to prior surveys. | 14 Contribution novelty problem / venue mismatch |
| C8 | Figure 7 charts have overlapping labels/cramped axes. | No | Visual readability problem, not comparison semantics. | 13 Visualization deficiency |
| C9 | Modality-specific vs unified encoders not clearly compared; no quantitative efficiency evidence. | Yes | The core request is comparing encoder families along efficiency dimensions. | 2 Comparison insufficient |
| C10 | No comparative empirical results beyond one query-strategy table. | Yes | Direct lack of comparative empirical result synthesis. | 2 Comparison insufficient |
| C11 | No visual summaries/tables comparing models, capabilities, training data, benchmarks. | Yes | Although phrased via tables, the missing content is comparative dimensions across models/benchmarks. | 2 Comparison insufficient |
| C12 | Evaluation-oriented comparison of named autonomous-driving methods is limited. | Yes | Explicit lack of named-method quantitative comparison. | 2 Comparison insufficient |
| C13 | No systematic empirical comparison of query strategies on shared benchmarks. | Yes | Direct shared-benchmark comparison gap. | 2 Comparison insufficient |
| C14 | Table 2 comparison is referenced but table content missing from provided text. | No | This is a missing rendered table/input extraction problem rather than lack of comparison design. | Exclude: input extraction/rendering, or 13 Visualization deficiency if verified in final document |
| C15 | Methods discussed individually without systematic cross-method comparison on performance/cost/suitability. | Yes | Canonical comparison-insufficient wording. | 2 Comparison insufficient |
| C16 | Limited quantitative comparison across ABSA papers/no comprehensive benchmark table. | Yes | Direct performance comparison gap. | 2 Comparison insufficient |
| C17 | Experiments compare only 4 methods on one dataset, weak baseline coverage. | Yes | This is inadequate empirical comparison/baseline comparison. | 2 Comparison insufficient |
| C18 | No quantitative comparisons of accuracy/runtime on benchmarks. | Yes | Direct absence of quantitative algorithm comparison. | 2 Comparison insufficient |
| C19 | Limited quantitative comparison of models and benchmarks across languages. | Yes | Direct model/benchmark comparison gap. | 2 Comparison insufficient |
| C20 | No systematic quantitative benchmark comparison table with metrics across methods. | Yes | Direct comparative metric table gap. | 2 Comparison insufficient |

## Implication

Evidence support insufficient should not be trusted as a calibrated category in the current full statistics. A reasonable immediate correction would be to fully recode category 11 rows with explicit exclusion rules for input artifacts and stronger precedence rules: factual/citation errors to Hallucination, missing method families to Missing specific topics, missing references to Missing specific references, table/figure issues to Visualization deficiency, repetition/flow/abstract clarity to Writing clarity, and promise/body mismatches to Internal inconsistency.

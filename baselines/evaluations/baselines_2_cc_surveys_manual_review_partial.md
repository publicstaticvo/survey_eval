# Baselines 2 cc_surveys Manual Review (Partial)

- Judged: 58
- Pending: 515

## Metrics on judged rows only

| evaluator | prompt | severity | total | valid | accuracy |
|---|---|---:|---:|---:|---:|
| cc | arise | minor | 4 | 3 | 75.00% |
| cc | arise | moderate | 2 | 2 | 100.00% |
| cc | plain | minor | 1 | 1 | 100.00% |
| cc | plain | moderate | 6 | 5 | 83.33% |
| cc | plain | severe | 5 | 4 | 80.00% |
| cc | trustsurvey | minor | 5 | 4 | 80.00% |
| cc | trustsurvey | moderate | 7 | 5 | 71.43% |
| llm | arise | minor | 4 | 0 | 0.00% |
| llm | arise | moderate | 3 | 2 | 66.67% |
| llm | plain | minor | 1 | 0 | 0.00% |
| llm | plain | moderate | 6 | 5 | 83.33% |
| llm | plain | severe | 2 | 0 | 0.00% |
| llm | trustsurvey | minor | 6 | 2 | 33.33% |
| llm | trustsurvey | moderate | 6 | 6 | 100.00% |

## Judged Rows

- `cc/cc_surveys/active_learning_arise.json#1` valid: Some reference entries are incomplete or incorrectly formatted (e.g., author names truncated or missing)
  - Rationale: 该批评成立。references.bib 中确有若干作者字段被压缩成不完整首字母形式，例如 ragoza2017active 的 'Hayley M. S. and John D. B.'，这会降低参考文献可追溯性。虽然不影响正文主线，但作为 bibliography curation 缺陷是真实的。
  - Evidence: references.bib 中 ragoza2017active 等条目的 author 字段出现缩写/截断式姓名；评审指出的行号落在参考文献区域。
- `cc/cc_surveys/active_learning_arise.json#2` invalid: The survey does not cite its own authors/institution, leaving author field empty in the BibTeX
  - Rationale: 该批评不成立。README 中用于项目元数据的 author/institution 为空，不能等同于论文没有引用自己的作者或机构；学术综述正文是否自引作者也不是质量要求。若问题是标题页作者为空，应另行表述，而不是说“does not cite its own authors/institution”。
  - Evidence: README.md 是项目说明元数据；main.tex 的作者块是否完整和“引用自身作者/机构”不是同一个问题。
- `cc/cc_surveys/active_learning_arise.json#3` valid: While mentioning LLM applications, the coverage of prompt-based/chat-based active learning is relatively brief given the rapid development in this area
  - Rationale: 该批评基本成立。原文在 emerging/frontier 部分提到 LLM、ChatGPT 和 instruction-tuning 等方向，但篇幅很短，主要是点名若干应用，没有系统展开 prompt-based/chat-based active learning 的问题设定、方法类型和风险。考虑到该方向确实是近年活跃主题，把它列为 minor coverage weakness 合理。
  - Evidence: main.tex 的 LLM 相关内容集中在后部应用/前沿小节，未形成独立方法 taxonomy 或比较。
- `cc/cc_surveys/active_learning_plain.json#1` valid: Missing author information — the author field is a bare 'Literature Survey' table with no actual author names, affiliations, or contact information, which is a fundamental omission for any academic submission.
  - Rationale: 该批评成立。main.tex 标题页的 author 字段不是具体作者姓名，而是泛化的 'Literature Survey'，没有真实作者、单位或联系方式。对于一篇拟作为 academic submission 的 survey，这是明确的元数据缺陷。
  - Evidence: main.tex 标题/作者块附近显示 author 信息为空泛，缺少姓名和 affiliation。
- `cc/cc_surveys/active_learning_plain.json#2` valid: Many bibliographic entries lack required fields, including page numbers, publisher information, and venue details. Several arXiv preprints lack arXiv IDs. This undermines the bibliography's usefulness as a reference resource.
  - Rationale: 该批评成立但严重程度可另议。references.bib 中存在 placeholder arXiv 号、缺少 venue/page、作者名截断等问题，例如 zhang2024active 的 arXiv:2401.00000 显然不像真实编号。这些问题会影响读者核验文献，因此是真实的 bibliography 缺陷。
  - Evidence: references.bib 包含 zhang2024active 等不完整或可疑元数据；另有若干条目缺页码/venue 或作者字段异常。
- `cc/cc_surveys/active_learning_plain.json#3` valid: No empirical comparison or performance benchmarks of the various query strategies is provided. The survey describes each method qualitatively but never compares them quantitatively on common benchmarks, leaving readers unable to assess relative effectiveness.
  - Rationale: 该批评成立。原文 Table 1 主要比较 query strategy 的计算成本、是否需要 Bayesian model 等属性，但没有在共享 benchmark 上对不同策略给出性能数值或横向结果。对于 active learning 方法综述，缺少经验比较会限制读者判断相对有效性。
  - Evidence: main.tex 的 query strategy 和 deep active learning 部分以定性描述为主；Table 1 不是 empirical benchmark table。
- `cc/cc_surveys/active_learning_plain.json#4` invalid: The survey lacks critical evaluation of the methods it describes. Each section reads as a descriptive literature review rather than an analytical one. There is no discussion of when specific strategies fail, empirical trends, or trade-offs beyond computational cost.
  - Rationale: 该批评过度概括。原文并非纯描述，已经多处讨论方法失败条件和 trade-off，例如 uncertainty sampling 会选 outlier 且深度网络校准差，batch-mode 有多样性与计算成本问题，deep AL 面临 uncertainty、retraining cost 和 representation drift。说“没有 critical evaluation”不符合正文证据。
  - Evidence: main.tex lines 274-277 讨论 uncertainty sampling 缺陷；deep active learning 小节列出三类挑战；practical issues 部分讨论 noisy oracles、stopping、cold start。
- `cc/cc_surveys/active_learning_plain.json#5` valid: Several citations appear incorrect or unreliable: krizhevsky2012imagenet incorrectly lists NeurIPS 2012 (correctly listed in the same paper's bibliography as 'Advances in Neural Information Processing Systems 25'); settlements2011active has no venue detail beyond 'Semi-Supervised and Active Learning for NLP' — a workshop rather than a full publication.
  - Rationale: 该批评基本成立。参考文献确有若干元数据质量问题，尤其是 workshop/book chapter 等 venue 记录不充分、部分条目格式不规范。即使 krizhevsky2012imagenet 的 NeurIPS 记法本身不算大错，整体 citation reliability concern 仍由其他例子支持。
  - Evidence: references.bib 中 settlements2011active、若干 arXiv/venue 条目缺少完整出版信息；同一 bibliography 中存在不规范作者字段。
- `cc/cc_surveys/active_learning_plain.json#6` valid: The manuscript lacks experimental figures, plots, or data tables beyond the algorithmic loop diagram and strategy summary table. A survey of this length (900+ lines) would benefit from comparative empirical visualizations.
  - Rationale: 该批评成立。原文确实主要只有 active learning loop 图和 strategy summary 表，没有实验曲线、benchmark 汇总图或跨方法结果可视化。作为长篇 survey，补充 empirical visualization 会明显增强读者理解。
  - Evidence: main.tex 包含 Figure 1 active learning loop 和 Table 1 query strategy summary；未见经验结果图表。
- `cc/cc_surveys/active_learning_trustsurvey.json#1` valid: Scope and inclusion-criteria declaration missing. The survey claims to be 'comprehensive' but does not describe search strategy, time span, venue scope, inclusion/exclusion criteria, or how literature was selected.
  - Rationale: 该批评成立。原文自称 comprehensive/self-contained，但没有说明检索数据库、关键词、时间范围、venue scope 或纳入/排除标准。它是叙述型综述可以接受，但作为 TrustSurvey 式透明度检查，缺少 inclusion criteria 是真实缺陷。
  - Evidence: Introduction 介绍组织结构和覆盖范围，但没有 methodology/search strategy/inclusion criteria 小节。
- `cc/cc_surveys/active_learning_trustsurvey.json#2` valid: No systematic benchmark-based method evaluation or empirical comparison. The survey describes methods theoretically but lacks a dedicated comparison table or analysis of empirical performance across strategies.
  - Rationale: 该批评成立。Table 1 只比较策略类型、成本和模型要求，没有报告共享任务上的准确率、label efficiency 或 runtime 等经验表现。正文也没有系统 benchmark-based comparison。
  - Evidence: Table 1 是概念/属性表；Section 4 和 deep AL 部分没有 benchmark result table。
- `cc/cc_surveys/active_learning_trustsurvey.json#3` valid: Citation of placeholder/unverified reference. The survey cites zhang2024active with arXiv number '2401.00000' which appears to be a placeholder DOI, not a real arXiv identifier.
  - Rationale: 该批评成立。zhang2024active 被用于 instruction tuning/LLM active learning，但参考文献中的 arXiv:2401.00000 明显像占位符而非真实 arXiv 编号。对综述而言，这种 placeholder reference 会影响可验证性。
  - Evidence: references.bib 中 zhang2024active 的 arXiv 字段为 2401.00000；正文 NLP/LLM 相关段落引用该条目。
- `cc/cc_surveys/active_learning_trustsurvey.json#4` valid: Missing reference to two canonical surveys. The survey acknowledges Settles (2009) and Ren (2021) in the acknowledgments but does not cite Settles' 2012 Morgan & Claypool book, which is a foundational monograph on active learning.
  - Rationale: 该批评基本成立。原文 acknowledgement 引 Settles 2009 和 Ren 2021，但没有纳入 Settles 2012 Morgan & Claypool monograph；对于 active learning 的基础教材/专著型来源，这是合理的遗漏提醒。不过该问题是 minor，因为 2009 survey 已覆盖核心基础。
  - Evidence: Acknowledgments 明确说 seminal survey by Settles；references.bib 有 settles2009active，但未见 Settles 2012 monograph。
- `cc/cc_surveys/active_learning_trustsurvey.json#5` valid: Missing Hanneke (2014) Foundations & Trends monograph. The disagreement coefficient and theory section cites hanneke2007 and references hanneke2014theory but the monograph is not included in the bibliography despite being the definitive treatment of disagreement-based active learning theory.
  - Rationale: 该批评成立。正文在 disagreement coefficient/theory 语境中提到 hanneke2014theory，但 bibliography 中未能找到相应完整条目；如果正文引用键缺失，会导致编译/核验问题。Hanneke 2014 也是该理论线的重要综述来源。
  - Evidence: main.tex theory section 使用 hanneke2014theory 语义；references.bib 未提供该键的完整条目。
- `cc/cc_surveys/diffusion_models_arise.json#1` valid: Limited original contribution as a survey; no new taxonomy, framework, or domain synthesis beyond unifying existing material
  - Rationale: 该批评基本成立。原文说会 unify mathematical frameworks 并综述架构、采样和应用，但没有提出清晰的新 taxonomy、评估框架或面向实践的决策图。作为综述仍有价值，但原创综合视角偏弱。
  - Evidence: Abstract/Introduction 主要承诺 comprehensive overview 和 unification；正文按常见数学-架构-采样-应用组织。
- `cc/cc_surveys/diffusion_models_arise.json#2` valid: Some duplicate reference entries in references.bib (hu2023expanding, huang2023compositional appear twice, chen2023dispersed uses 'et al.' without full author list)
  - Rationale: 该批评成立。references.bib 中 hu2023expanding 与 huang2023compositional 各出现两次，chen2023dispersed 使用 'J. Chen and others' 这种不完整作者列表。属于真实 bibliography curation 问题。
  - Evidence: references.bib lines 443-455 与 510-522 重复；chen2023dispersed author 为 'J. Chen and others'。
- `cc/cc_surveys/diffusion_models_arise.json#3` valid: Relatively sparse coverage of discrete-state diffusion models (only one citation: Austin et al. 2021)
  - Rationale: 该批评成立。离散状态扩散在正文中只通过 Austin/D3PM 参考和 analog bits 一句带过，没有系统说明离散状态空间、文本/图/类别变量建模或与连续扩散的差异。作为扩散模型重要分支，覆盖偏薄。
  - Evidence: main.tex 仅在 discrete-time formulation 附近和 line 292 analog bits 简短提及；无独立 discrete diffusion 小节。
- `cc/cc_surveys/diffusion_models_plain.json#1` valid: Missing author information and affiliation — the byline is empty, making it unsuitable for formal publication
  - Rationale: 该批评成立。main.tex 中 \author{} 为空，没有作者姓名、单位或联系方式。对于 formal publication/submission，这是明确的元数据缺陷。
  - Evidence: main.tex line 50: \author{}。
- `cc/cc_surveys/diffusion_models_plain.json#2` invalid: No actual publication date; uses \today which produces compilation date rather than archival publication date
  - Rationale: 该批评不成立。LaTeX 草稿使用 \today 作为编译日期很常见，不能据此认定论文缺少 archival publication date；未发表或测试输入中的 survey 本来也未必应有正式出版日期。
  - Evidence: main.tex line 51: \date{\today}；这更像草稿日期设置，不是内容缺陷。
- `cc/cc_surveys/diffusion_models_plain.json#3` valid: Bibliography contains significant formatting errors: duplicate entries for hu2023expanding and huang2023compositional, incorrect booktitles ('arXiv preprint' as booktitle in multiple entries), and incomplete author lists like 'J. Chen and others'
  - Rationale: 该批评成立。bib 中确有重复条目、不规范 booktitle/journal 字段，以及 'J. Chen and others' 这种不完整作者列表。它们会影响参考文献质量。
  - Evidence: references.bib 中 hu/huang 条目重复；Point-E/Graves 等 arXiv preprint 被放在 booktitle；chen2023dispersed 作者不完整。
- `cc/cc_surveys/diffusion_models_plain.json#4` valid: Coverage cuts off around early 2024, missing significant later developments including modern large-scale text-to-image/video models (DALL-E 3, Sora, SDXL, HunyuanVideo), newer diffusion transformer variants, improved guidance techniques, and emerging applications
  - Rationale: 该批评成立。正文覆盖 DALL-E 2、Stable Diffusion、Imagen、Stable Video Diffusion 等，但没有 DALL-E 3、Sora、SDXL、HunyuanVideo 等后续大规模系统，也没有近年 DiT/guidance 进展的系统更新。对于自称 comprehensive/up-to-date 的 diffusion survey，这是时效性缺口。
  - Evidence: applications section 截止到 2023 Stable Video Diffusion 一类工作；未见 Sora、DALL-E 3、SDXL、HunyuanVideo。
- `cc/cc_surveys/diffusion_models_plain.json#5` valid: Responsible AI section is notably brief (~5 sentences) for a technology with major societal implications; limited coverage of bias mitigation, content moderation, and regulatory considerations
  - Rationale: 该批评成立。Responsible AI 只在 challenges 中一小段概括 harmful content、copyright、bias、watermarking，没有展开 mitigation、policy、moderation pipeline 或 provenance 机制。对高影响生成技术而言覆盖偏薄。
  - Evidence: main.tex line 380 附近 Responsible AI 约一段。
- `cc/cc_surveys/diffusion_models_plain.json#6` valid: No quantitative benchmark comparison tables or figures synthesizing empirical results across models (FID scores, sampling speed, compute requirements)
  - Rationale: 该批评成立。全文没有聚合 FID/IS、采样步数、吞吐、训练成本等 benchmark 表或图；评价问题只在 challenges 中定性讨论。
  - Evidence: main.tex 未见 benchmark comparison table；line 376 仅定性讨论 FID 局限。
- `cc/cc_surveys/diffusion_models_trustsurvey.json#1` valid: No scope/inclusion-criteria declaration: The survey claims to be 'comprehensive' but does not state venue constraints, date ranges, search strategy, or inclusion/exclusion criteria, making the coverage boundary uninspectable.
  - Rationale: 该批评成立。原文自称 comprehensive overview，但没有说明检索策略、时间范围、venue 约束或纳入/排除标准，覆盖边界不可审计。
  - Evidence: Abstract/Introduction 说明综述范围和章节，但无 methodology/search/inclusion criteria。
- `cc/cc_surveys/diffusion_models_trustsurvey.json#2` valid: No contribution statement: The survey does not explicitly state what it contributes as a survey—no taxonomy, organizing framework, or practical guidance is promised beyond a review.
  - Rationale: 该批评基本成立。摘要说 unify frameworks、review innovations、survey applications，但没有独立 contribution paragraph，也没有明确说明本文相对已有 diffusion surveys 的新增组织框架。
  - Evidence: Abstract/Introduction 主要是 overview 承诺；无 explicit contribution list。
- `cc/cc_surveys/diffusion_models_trustsurvey.json#3` valid: Missing comparison: Works are presented individually within application sections without explicit cross-method comparison tables or contrast along benchmark dimensions.
  - Rationale: 该批评成立。应用和采样部分按工作顺序介绍 DDIM、DPM-Solver、distillation、Stable Diffusion、VDM 等，但缺少横向表格或维度化比较。
  - Evidence: Sampling Acceleration 和 Applications 小节为连续 prose；未见 side-by-side comparison table。
- `cc/cc_surveys/diffusion_models_trustsurvey.json#4` valid: Method evaluation: Benchmark comparison absent. FID/IS numbers are mentioned in text from cited papers but no aggregated comparison table or systematic evaluation discussion exists.
  - Rationale: 该批评成立。文中偶尔提到 FID、sampling steps 或 quality，但没有系统聚合 benchmark，也没有专门评价维度讨论。
  - Evidence: line 376 只说 FID limitations；没有 FID/IS/latency/cost 汇总表。
- `cc/cc_surveys/diffusion_models_trustsurvey.json#5` invalid: Missing references: Several notable works in diffusion model literature are absent from both text and bibliography.
  - Rationale: 该批评太笼统，未列出具体缺失文献，无法判断 reviewer 的指控是否准确。作为 meta-review，不能接受“several notable works absent”这种没有对象的 weakness。
  - Evidence: issue/location 只写 various sections，没有点名文献。
- `cc/cc_surveys/diffusion_models_trustsurvey.json#6` invalid: Missing topics recoverable from external literature: Several topic areas are absent or underdeveloped.
  - Rationale: 该批评太笼统。没有列出具体缺失主题或说明为何超出/落入本文范围，无法核验是否真实缺陷。
  - Evidence: issue 只说 several topic areas absent or underdeveloped，没有具体 topic。
- `cc/cc_surveys/diffusion_models_trustsurvey.json#7` invalid: Missing landmark papers recoverable from the literature: Several specific landmark references are absent.
  - Rationale: 该批评不可判定且不充分。没有点名所谓 landmark papers，不能作为准确 weakness；meta-review 需要可核验对象。
  - Evidence: issue 未列出具体 landmark references。
- `llm/cc_surveys/active_learning_arise.json#1` valid: The paper has only one placeholder figure (Figure 1: The active learning loop) with minimal visual content, lacking diagrams for theoretical concepts (e.g., version space, disagreement regions), query strategy comparisons, or workflow illustrations
  - Rationale: 该批评成立。原文可见的视觉元素很少，主要是 active learning loop 图和策略表，缺少 version space、disagreement region、batch selection 等核心概念图示。作为 survey，这不是致命问题，但确实限制可读性。
  - Evidence: main.tex 中主要图表为 Figure 1 和 Table 1；理论概念部分没有配图。
- `llm/cc_surveys/active_learning_arise.json#2` invalid: Some sections remain primarily descriptive (e.g., Expected Error Reduction) without critical analysis of when strategies fail or comparative evaluations
  - Rationale: 该批评过度。Expected Error Reduction 等小节确实偏概念性，但整篇 survey 并非缺少 failure/trade-off 分析；uncertainty sampling、deep active learning、noisy oracle、stopping criteria 等处都有明确缺陷和适用性讨论。因此说这些部分“remain primarily descriptive without critical analysis”只部分成立，不足以作为准确 weakness。
  - Evidence: main.tex lines 274-277 批评 uncertainty sampling；deep AL 小节列出 uncertainty、cost、representation drift；practical section 分析 noisy oracles 和 stopping。
- `llm/cc_surveys/active_learning_arise.json#3` invalid: Language occasionally becomes verbose, particularly in the introduction, and some mathematical notation is inconsistent (e.g., ℒ vs L_0 notation)
  - Rationale: 该批评证据不足。引言较长是综述写作常态，且 reviewer 没有给出具体冗余句子；数学记号 L/\mathcal{L} 的差异在机器学习论文中常用于不同对象，未显示导致歧义。该条更像风格偏好，不是真实缺陷。
  - Evidence: issue 未提供具体冲突公式；原文 notation 整体可读。
- `llm/cc_surveys/active_learning_arise.json#4` invalid: Several citations use generic placeholders (<cit.>) which reduces the survey's scholarly credibility and verifiability
  - Rationale: 该批评不成立。原始 LaTeX 正文使用 \cite 命令，references.bib 中也有实际条目；'<cit.>' 很可能来自某个文本抽取/渲染中间表示，而不是原文缺引用。评价原文时不能把解析器占位符当成论文缺陷。
  - Evidence: main.tex 中 membership/query strategy/practical sections 使用 \cite{...}；references.bib 存在对应文献条目。
- `llm/cc_surveys/active_learning_plain.json#1` invalid: Multiple citation placeholders (<cit.>) are present throughout the paper where actual references should appear, including membership queries, preference queries, multiple-instance queries, core-set selection, DPPs, performance plateau, and stability stopping criteria
  - Rationale: 该批评不成立，原因同上。原文 LaTeX 中并不存在 '<cit.>' 占位符，而是正常的 \cite 命令；该问题来自评测输入文本的 citation rendering，而非原始 survey 的错误。
  - Evidence: main.tex 多处为 \cite{...}，例如 query types、batch-mode、stopping criteria；未见原始 '<cit.>'。
- `llm/cc_surveys/active_learning_plain.json#2` valid: Insufficient depth on recent emerging trends, particularly LLM-based active learning which receives only cursory treatment despite its growing importance
  - Rationale: 该批评成立。LLM-based/prompt-based active learning 只在 emerging frontier 部分简短出现，没有系统讨论人机对话标注、instruction-tuning data selection、LLM uncertainty、prompting cost 等子问题。鉴于近年发展，该覆盖确实偏薄。
  - Evidence: main.tex 后部 LLM frontiers 段落篇幅有限，未形成独立 taxonomy 或比较。
- `llm/cc_surveys/active_learning_plain.json#3` valid: No comparative empirical results or summary tables beyond Table 1 on query strategies, making it difficult to assess relative method effectiveness
  - Rationale: 该批评成立。原文有策略摘要表，但没有跨数据集、跨 query strategy 的 empirical comparison。读者无法从本文直接比较方法 label efficiency 或 benchmark performance。
  - Evidence: Table 1 为策略属性表；Section 6 deep active learning 定性综述，没有结果表。
- `llm/cc_surveys/active_learning_plain.json#4` invalid: The paper does not discuss dataset shift, domain adaptation scenarios, or how active learning behaves when the unlabeled pool differs from the test distribution
  - Rationale: 该批评不准确。原文虽没有独立的 dataset shift/domain adaptation 章节，但多处讨论 OOD、domain-specific scans、medical/scientific data、calibration 和 distribution-related uncertainty；并在 open problems 中提到 robust uncertainty for out-of-distribution inputs。说完全不讨论不成立。
  - Evidence: main.tex open problems 提到 out-of-distribution inputs 和 calibration；应用部分覆盖 medical imaging、多模态、domain-specific settings。
- `llm/cc_surveys/active_learning_trustsurvey.json#1` valid: Missing explicit inclusion-criteria declaration - the survey claims to be 'comprehensive' but provides no search strategy, inclusion/exclusion criteria, time span, or venue scope
  - Rationale: 该批评成立。原文没有说明检索策略、筛选标准、时间范围和 venue scope，却使用 comprehensive/up-to-date 的表述。作为 review transparency 问题是真实存在的。
  - Evidence: Introduction 只有综述组织说明；未见 methodology/search strategy/inclusion-exclusion criteria。
- `llm/cc_surveys/active_learning_trustsurvey.json#2` valid: Contribution statement insufficient - survey does not explicitly state what it adds over existing surveys (e.g., Settles 2012, Ren et al. 2021)
  - Rationale: 该批评成立。原文 acknowledgments 提到 Settles 和 Ren 影响了组织与内容，但 introduction 没有明确说明本文相对这些已有 survey 的新增贡献或差异化。作为综述定位问题，这是合理 minor weakness。
  - Evidence: Acknowledgments 提到 Settles 2009 和 Ren 2021；Introduction 主要列章节安排，没有 explicit contribution over prior surveys。
- `llm/cc_surveys/active_learning_trustsurvey.json#3` valid: Method evaluation insufficient - no systematic empirical comparison of query strategies on shared benchmarks
  - Rationale: 该批评成立。原文没有共享 benchmark 上的系统经验比较，只有方法描述和概念表。对于 method evaluation 维度，缺少 empirical comparison 是真实弱点。
  - Evidence: Section 3/4 和 Section 6 以叙述为主；Table 1 不含性能结果。
- `llm/cc_surveys/active_learning_trustsurvey.json#4` invalid: Comparison insufficient for key method families - classical and deep active learning methods are presented in isolation without cross-cutting empirical or analytical comparisons
  - Rationale: 该批评说“presented in isolation”过重。原文有 cross-cutting 组织：uncertainty/QBC/BALD 的关系、deep AL 的三类挑战、batch-mode/noisy/stopping/cold-start practical considerations，以及总结性 open problems。缺少 empirical comparison 是问题，但 analytical comparison 并非完全缺失。
  - Evidence: main.tex 连接 QBC、BALD、ensemble uncertainty；practical section 横向讨论 batch/noisy/stopping/cold-start；open problems 综合多类方向。
- `llm/cc_surveys/active_learning_trustsurvey.json#5` invalid: Missing reference to relevant survey: Settles (2012) book is acknowledged but Settles (2009) technical report is not explicitly referenced despite being foundational
  - Rationale: 该批评事实错误。原文明确引用并在 acknowledgments 中承认 Settles 2009 active learning survey；说 Settles 2009 technical report 没有明确 referenced 不成立。若想批评缺 Settles 2012 monograph，应另行表述。
  - Evidence: references.bib 有 settles2009active；main.tex acknowledgments 写 'seminal survey by \citet{settles2009active}'。
- `llm/cc_surveys/diffusion_models_arise.json#1` invalid: Duplicate 'Section' labels in the outline (Section Section 2 and Section Section 3)
  - Rationale: 该批评不成立。原始 LaTeX 中没有 'Section Section 2' 这类文字，只有正常的 section/subsection 和 \ref/\eqref；该问题更可能来自渲染文本中的引用展开错误。
  - Evidence: rg 未发现 Section Section；main.tex 使用正常 LaTeX section/ref。
- `llm/cc_surveys/diffusion_models_arise.json#2` valid: No figures, diagrams, or tables to illustrate the diffusion process, architectures, or timelines
  - Rationale: 该批评成立。原文有算法环境，但没有图、示意图、时间线或表格来解释扩散过程、架构演化或应用版图。对 survey 可读性是实际弱点。
  - Evidence: main.tex 有 algorithm 环境；未见 begin{figure}/begin{table} 的实质图表。
- `llm/cc_surveys/diffusion_models_arise.json#3` invalid: Minor notational inconsistencies - subscript/out subscript notation has formatting issues in several equations
  - Rationale: 该批评证据不足。数学框架中的公式整体可读，reviewer 没有给出具体哪一处 notation 会造成含义错误；轻微格式观感不足以构成明确 weakness。
  - Evidence: Mathematical Framework 使用标准 forward/reverse process、ELBO、score/SDE 记号。
- `llm/cc_surveys/diffusion_models_plain.json#1` invalid: Algorithm placeholders are empty - Algorithm 1 (DDPM Training) and Algorithm 2 (DDPM Sampling) show no actual steps, only empty boxes with 'Section 3.3' references
  - Rationale: 该批评事实错误。原文 Algorithm 1 和 Algorithm 2 均有 algorithmic 步骤，包括采样 t、噪声、梯度更新，以及反向采样循环；不是空盒子。
  - Evidence: main.tex lines 198-228 包含完整 algorithmic 环境。
- `llm/cc_surveys/diffusion_models_plain.json#2` invalid: Section numbering is inconsistent throughout - multiple instances of 'Section Section 2' and similar typos
  - Rationale: 该批评不成立。原始 LaTeX 中未见 'Section Section'；这是渲染/抽取文本的引用格式问题，不是 survey 源文错误。
  - Evidence: rg 未找到 Section Section；源文件 section 标题正常。
- `llm/cc_surveys/diffusion_models_plain.json#3` valid: Lacks discussion of recent architectural innovations from 2023-2024 including modern LLMs as text encoders, newer guidance techniques, and recent state-of-the-art results
  - Rationale: 该批评基本成立。虽然文中提到 T5 text encoder、DiT、CFG、ControlNet 和 2023 Stable Video Diffusion，但没有覆盖 DALL-E 3/Sora/SDXL 等更新系统，也没有系统讨论 2024 后 guidance/architecture 进展。
  - Evidence: architecture/application sections 主要到 2023；缺少 2024 大规模系统和更近进展。
- `llm/cc_surveys/diffusion_models_plain.json#4` valid: Missing quantitative evaluation metrics and benchmarks beyond FID, without discussing newer evaluation methodologies
  - Rationale: 该批评成立。评价维度基本停留在 FID 局限的定性讨论，没有深入 newer evaluation methodologies、human preference、T2I alignment、video/3D 专门指标等。
  - Evidence: Challenges 中 Evaluation bullet 简短；无 dedicated evaluation section。
- `llm/cc_surveys/diffusion_models_plain.json#5` valid: No discussion of practical training considerations such as memory optimization, gradient checkpointing, or hyperparameter tuning recipes
  - Rationale: 该批评成立但偏实践导向。训练部分讲算法目标和采样流程，没有给出 memory optimization、checkpointing、batching、hyperparameter recipes 等工程训练细节；若目标读者包含 practitioners，这是缺口。
  - Evidence: Training and Sampling Algorithms 小节聚焦 DDPM 伪代码；缺少 operational training guidance。
- `llm/cc_surveys/diffusion_models_trustsurvey.json#1` valid: Contribution statement missing - the survey does not explicitly state what unique contributions it makes as a survey beyond 'comprehensive overview'
  - Rationale: 该批评成立。原文没有明确列出本文作为 survey 的独特贡献或相对已有综述的差异化，只是说明 comprehensive overview/unified treatment。
  - Evidence: Abstract/Introduction 无 contribution paragraph。
- `llm/cc_surveys/diffusion_models_trustsurvey.json#2` valid: Method evaluation insufficient - no systematic benchmark-based comparison or metric discussion
  - Rationale: 该批评成立。没有系统 benchmark 表或专门 metric discussion；FID 等只在 prose 中零散出现。
  - Evidence: Applications/Challenges 中没有聚合评价表；line 376 为简短评价讨论。
- `llm/cc_surveys/diffusion_models_trustsurvey.json#3` valid: Comparison insufficient - methods summarized individually rather than contrasted along meaningful dimensions
  - Rationale: 该批评成立。架构和采样部分逐项叙述 U-Net、LDM、DiT、DDIM、DPM-Solver、distillation 等，但缺少按速度、质量、训练成本、适用场景的显式对照。
  - Evidence: Sections 4-5 是 sequential summaries，无 comparison matrix。
- `llm/cc_surveys/diffusion_models_trustsurvey.json#4` invalid: Synthesis / original viewpoint insufficient - primarily lists prior work with limited cross-paper synthesis
  - Rationale: 该批评过度。原文并非只列工作，也有数学统一、挑战归纳和跨应用总结；虽然原创框架有限，但说 synthesis/original viewpoint insufficient 对 minor weakness 来说证据不够具体。
  - Evidence: Abstract 明确统一 discrete/continuous frameworks；Challenges/Future Directions 综合 efficiency、evaluation、responsible AI 等问题。
- `llm/cc_surveys/diffusion_models_trustsurvey.json#5` valid: Scope declaration missing - no explicit inclusion/exclusion criteria or search strategy
  - Rationale: 该批评成立。原文没有检索策略、时间跨度、纳入/排除标准或范围边界声明。
  - Evidence: Introduction 缺少 scope methodology。
- `llm/cc_surveys/diffusion_models_trustsurvey.json#6` valid: Missing specific topics - diffusion-based LLM/video generation and recent 2023-2024 advances appear underrepresented
  - Rationale: 该批评基本成立。视频扩散只到 Make-A-Video、VDM、Stable Video Diffusion 等，缺少 Sora 等更新系统；diffusion-based LLM/discrete token generation 也未系统展开。
  - Evidence: Video section lines 336-338；discrete diffusion 只简短提及 analog bits。
- `llm/cc_surveys/diffusion_models_trustsurvey.json#7` invalid: Missing specific topic - detailed theoretical analysis and proofs that would serve as reference material
  - Rationale: 该批评不成立。作为综述，数学框架部分已经给出 forward/reverse process、ELBO、score/SDE 等核心公式；要求详细 proofs 超出许多 survey 的合理范围。缺少证明不必然是缺陷。
  - Evidence: Mathematical Framework 含 DDPM/SDE/score matching 公式和解释。

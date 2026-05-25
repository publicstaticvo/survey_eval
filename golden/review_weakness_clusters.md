# Review Weakness Cluster Analysis

- 输入文件: `review.json`
- 论文数: 170
- Review 数: 533
- 统计口径: 一个 review 命中同一类别最多计 1 次；一篇文章只要任一 review 命中该类别，则该文章计 1 次。
- 提取范围: 排除 `strengths`、`reasons_to_accept`，并从 `strengths_and_weaknesses` 中只保留 Weaknesses 段；其余审稿字段中指出的问题均纳入。

## 聚类结果

| Weakness 类别 | 标准来源 | Review 频次 | 文章频次 | 说明 |
|---|---|---:|---:|---|
| 漏引用具体文献 | A. 已有文献池 | 6 | 6 | 审稿人明确给出综述缺少的具体论文或作品标题，判断标准来自已有文献池。 |
| 未覆盖具体主题 | A. 已有文献池 | 24 | 21 | 审稿人明确指出综述遗漏了某个具体 topic、任务、应用场景或子领域，判断标准来自已有文献池。 |
| 内容过时或未纳入近期进展 | A. 已有文献池 | 44 | 37 | 审稿人认为综述与领域当前文献状态脱节、引用不是最新版，或未覆盖近期进展。 |
| 系统性与文献筛选方法不足 | B. 学术共识（在已有文献池未提及） | 93 | 71 | 检索策略、纳入排除标准、系统综述流程、复现性或文献池构建不够清楚。 |
| 覆盖浅显或深度不足 | B. 学术共识（在已有文献池未提及） | 174 | 117 | 综述停留在表层描述，关键技术、方法细节、实验设置或应用语境展开不够。 |
| 缺少批判性综合与洞见 | B. 学术共识（在已有文献池未提及） | 216 | 133 | 只是罗列已有工作，缺少比较、归纳、批判性评价、设计取舍或面向未来的实质洞见。 |
| 缺少比较证据或量化分析 | B. 学术共识（在已有文献池未提及） | 386 | 164 | 缺少 benchmark、表格、指标、数据集统计、复杂度分析、横向比较或实证支撑。 |
| 写作表达、格式或图表问题 | B. 学术共识（在已有文献池未提及） | 374 | 158 | 文字不清、排版错误、引用占位符、图表质量差、可视化不足、拼写或格式问题。 |
| 定义、术语或分类体系不清 | C. 综述自身 | 347 | 162 | 综述内部的核心定义、术语边界、taxonomy 或符号使用不清楚，影响自身论证一致性。 |
| 结构组织与章节衔接问题 | C. 综述自身 | 410 | 165 | 章节顺序、段落衔接、内容分配、标题命名或前后逻辑组织存在内部问题。 |
| 范围、标题或目标不一致 | C. 综述自身 | 264 | 145 | 标题、摘要、目标声明与正文实际覆盖范围不一致，或综述边界没有自洽界定。 |
| 内部论证、结论或建议支撑不足 | C. 综述自身 | 155 | 111 | 结论、建议或立场没有被综述自身的材料充分支撑，或前后论证链条断裂。 |
| 个人偏好、venue匹配或其他琐碎问题 | D. 审稿人自己的喜好或专有知识 | 324 | 159 | 包含审稿人对 venue、创新性门槛、写作取向的个人偏好，以及无法稳定归入其他类别的零散意见。 |

## 漏引用与未覆盖 Topic

只有当审稿人给出可识别的具体论文标题时，才记录为 `漏引用具体文献`；只有当审稿人给出具体 topic、任务、应用场景或子领域名称时，才记录为 `未覆盖具体主题`。

### 高频缺失文献标题

- Personalized Algorithmic Recourse with Preference Elicitation: 1 篇文章
- Setting the right expectations: Algorithmic recourse over time: 1 篇文章
- Fairness in Algorithmic Recourse Through the Lens of Substantive Equality of Opportunity: 1 篇文章
- Preference Elicitation in Interactive and User-centered Algorithmic Recourse: an Initial Exploration: 1 篇文章
- Understanding the User Perception and Experience of Interactive Algorithmic Recourse Customization: 1 篇文章
- ATG: Benchmarking Automated Theorem Generation for Generative Language Models: 1 篇文章
- MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data: 1 篇文章
- Language Models are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought: 1 篇文章
- Explaining Answers with Entailment Trees: 1 篇文章
- A Survey on Dataset Distillation: Approaches, Applications and Future Directions: 1 篇文章
- A Comprehensive Survey of Dataset Distillation: 1 篇文章
- Guiding continuous operator learning through Physics-based boundary constraints: 1 篇文章
- Learning differentiable solvers for systems with hard constraints: 1 篇文章
- Finite Difference Methods for Ordinary and Partial Differential Equations: 1 篇文章
- Finite-Volume Methods for Hyperbolic Problems: 1 篇文章
- The Finite Element Method: Linear Static and Dynamic Finite Element Analysis: 1 篇文章
- Bayesian Constraint Inference from User Demonstrations Based on Margin-Respecting Preference Models: 1 篇文章
- Learning Constraints on Autonomous Behavior from Proactive Feedback: 1 篇文章
- Learning Shared Safety Constraints from Multi-task Demonstrations: 1 篇文章
- Learning Hyperplanes for Multi-Agent Collision Avoidance in Space: 1 篇文章
- Isoperimetric Constraint Inference for Discrete-Time Nonlinear Systems Based on Inverse Optimal Control: 1 篇文章
- X-MEN: Guaranteed XOR-Maximum Entropy Constrained Inverse Reinforcement Learning: 1 篇文章
- Positive-Unlabeled Constraint Learning (PUCL) for Inferring Nonlinear Continuous Constraints Functions from Expert Demonstrations: 1 篇文章
- Provably Efficient Exploration in Inverse Constrained Reinforcement Learning: 1 篇文章
- Bootstrapping Generators from Noisy Data (human evaluation of faithfulness in data-to-text: 1 篇文章
- Get To The Point: Summarization with Pointer-Generator Networks (an architecture to improve factual errors in summarisation: 1 篇文章

### 高频缺失 Topic

- foundation models: 8 篇文章
- generalizability: 3 篇文章
- DPO: 3 篇文章
- human-in-the-loop: 3 篇文章
- robotics: 3 篇文章
- constitutional AI: 2 篇文章
- informal theorem proving via natural language explanation: 1 篇文章
- automated theorem generation: 1 篇文章
- theorem prover feedback: 1 篇文章
- future applications of LLMs: 1 篇文章
- emerging challenges: 1 篇文章
- causal RL: 1 篇文章
- spurious correlation: 1 篇文章
- user studies: 1 篇文章
- language-modeling related content: 1 篇文章
- consistency models: 1 篇文章
- such as knowledge distillation: 1 篇文章

## 孤立点说明

- 本次规则聚类中没有需要单开的低频非 D 类孤立点。

## 输出文件

- 逐篇文章 JSON: `review_weaknesses_by_forum.json`
- 本报告: `review_weakness_clusters.md`

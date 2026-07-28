from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PLAN_PATH = Path('survey_eval/train_letor/defect_injection_plans.json')
BUILD_PATH = Path('survey_eval/train_letor/build_defect_injection_plans.py')

MANUAL_FACTS = {
    'retrieval-augmented_generation_for_large_language_models_a_survey': [
        ('method-mechanism substitution', 'When dealing with semi-structured data, TableGPT~ [31] stores each table as a natural-language paragraph and answers questions without generating or executing SQL queries.', 'Change the mechanism of TableGPT from Text-2-SQL/database execution to a text-only table serialization pipeline.'),
        ('unsupported source coverage', 'In addition to encyclopedic data, the cited systems treat cross-lingual, medical, and legal corpora as already solved retrieval sources that require no domain adaptation~ [19, 29, 30].', 'Inflate heterogeneous retrieval-source examples into a solved-domain-adaptation claim.'),
        ('pipeline role inversion', 'Alternatively, tables can be transformed into text format only after the LLM has generated the final answer, so the textualized table is used for post-hoc explanation rather than retrieval~ [32].', 'Move the table-to-text step from retrieval/input preparation to post-answer explanation.'),
        ('data-quality overclaim', 'Structured data, such as knowledge graphs (KGs)~ [33], are treated as fully verified resources that eliminate the need for retrieval filtering or evidence checking.', 'Turn the paper\'s cautious statement that KGs can be more precise into a false guarantee of complete verification.'),
    ],
    'instruction_tuning_for_large_language_models_a_survey': [
        ('dataset-construction error', 'Natural Instructions~ [38] is an automatically scraped collection of user-chat demonstrations rather than a human-crafted instruction dataset.', 'Replace the dataset construction process with an unsupported automatic chat-log scraping story.'),
        ('reasoning-data source error', 'This section presents reasoning datasets as being derived mainly from OpenAI o1 and DeepSeek-R1 model traces, rather than from independently constructed task data [36, 37].', 'Misrepresent cited reasoning-model papers as the direct source of the surveyed datasets.'),
        ('modality expansion error', 'P3 (Public Pool of Prompts)~ [39] is an instruction tuning dataset that combines English NLP tasks with paired image-caption and speech-recognition prompts.', 'Add multimodal content to a text-prompt dataset.'),
        ('language-scope inversion', 'xP3 (Crosslingual Public Pool of Prompts)~ [40] is primarily an English-only benchmark used to test whether multilingual transfer is unnecessary.', 'Invert the multilingual purpose of xP3 into an English-only benchmark claim.'),
    ],
    'a_survey_on_evaluation_of_large_language_models': [
        ('model-comparison exaggeration', "ChatGPT's sentiment analysis prediction performance consistently surpasses GPT-3.5 and removes the need for task-specific sentiment models [20, 21].", 'Overstate a close/superior comparison into a universal dominance claim.'),
        ('task-difficulty misstatement', 'The cited benchmarks show that natural language understanding tasks remain uniformly difficult for LLMs, with most models performing near chance on this category [22, 23].', 'Replace a generally high-performance statement with a false low-performance characterization.'),
        ('capability-transfer error', 'In fine-grained sentiment and emotion cause analysis, ChatGPT exhibits exceptional performance because it explicitly detects causal emotion chains during pretraining [24].', 'Add an unsupported mechanistic explanation for the reported capability.'),
        ('low-resource conclusion flip', 'In low-resource learning environments, ChatGPT shows significant advantages over small language models and fully overcomes its limitations on low-resource languages [25, 26].', 'Remove the original caveat and claim complete low-resource coverage.'),
    ],
    'harnessing_the_power_of_llms_in_practice_a_survey_on_chatgpt_and_beyond': [
        ('benchmark-scope substitution', "LLMs are particularly good at translating high-resource French-English news text in WMT'16, while the cited Romanian-English low-resource setting is not discussed~ [53, 9].", 'Change the language-pair and resource setting without changing the cited evidence.'),
        ('data-factor omission', 'The quality of pre-training data alone determines LLM performance, while data quantity and diversity have little measurable influence~ [15].', 'Narrow a multi-factor data statement into a single-factor causal claim.'),
        ('method-setting error', 'LLMs have been shown to outperform previous zero-shot methods only after full supervised fine-tuning on the target task [16].', 'Contradict the zero-shot setting by adding target-task fine-tuning.'),
        ('forgetting-mechanism error', 'Catastrophic forgetting is avoided because prompts periodically update a protected copy of the model parameters during inference [17].', 'Invent a parameter-update mechanism for a statement that depends on parameters remaining unaltered.'),
    ],
    'augmented_language_models_a_survey': [
        ('tool-use mechanism error', 'PAL~ [19] directly executes natural-language chain-of-thought steps as Python programs, without asking the model to write explicit code for the intermediate reasoning steps.', 'Collapse PAL\'s program-generation step into direct execution of natural language.'),
        ('generation-length overclaim', 'Re3~ [17] uses the same prompting scheme to reliably generate complete book-length stories rather than long short stories.', 'Expand the supported generation scale from long stories to book-length generation.'),
        ('model-role error', 'Re3 first trains GPT-3 from scratch on the story premise before generating the plan, setting, and characters~ [2].', 'Replace prompting a pretrained model with training a model from scratch.'),
        ('granularity guarantee error', 'The learned detailed outliner in [55] guarantees globally coherent narratives at any requested outline depth without additional revision or filtering.', 'Turn iterative outline expansion into an unsupported coherence guarantee.'),
    ],
    'a_survey_on_in-context_learning': [
        ('training-objective substitution', 'FLAN~ [28] improves ICL by training LaMDA-PT~ [29] to ignore natural-language instructions and rely only on unlabeled continuation data.', 'Invert instruction tuning into instruction-free language-model continuation.'),
        ('pre-inference training denial', 'The cited studies show that specialized training before inference is unnecessary and usually weakens ICL ability~ [22, 23, 24].', 'Contradict the surveyed finding that specialized training can enhance ICL.'),
        ('efficiency-mechanism error', '[14] introduced meta-distillation by storing every demonstration vector at inference time, which improves ICL efficiency through a larger retrieval cache.', 'Replace distilled demonstration vectors with an inference-time storage/cache mechanism.'),
        ('demonstration-use error', 'Both [13] and [25] improve ICL by continually finetuning LLMs on tasks without demonstration examples, showing demonstrations are not needed during warmup.', 'Remove the multiple-demonstration-example condition from the cited warmup methods.'),
    ],
    'towards_reasoning_in_large_language_models_a_survey': [
        ('emergence-threshold overclaim', 'Recent research establishes reasoning ability as a guaranteed property of any language model once it is trained with a sufficiently large corpus, regardless of parameter scale [8, 9, 21].', 'Change a scale-associated emergence observation into a universal training-data guarantee.'),
        ('weakness-to-strength error', 'Reasoning, particularly multi-step reasoning, is often presented as the strongest and most reliable capability of standard language models before any reasoning-specific prompting [14, 29, 15].', 'Invert the survey\'s framing of multi-step reasoning as a weakness.'),
        ('dataset-purpose error', '[32] finetunes a pretrained GPT model~ [33] to generate adversarial distractors for CoS-E, rather than rationales explaining model predictions on commonsense QA~ [34].', 'Change rationale generation into adversarial distractor generation.'),
        ('knowledge-source error', '[35] trains RoBERTa~ [36] to ignore free-text statements and perform reasoning only from implicit pre-trained knowledge.', 'Remove the explicit free-text statement component from the cited method.'),
    ],
    'multimodal_learning_with_transformers_a_survey': [
        ('token-function swap', 'Task-specific customized tokens use [MASK] for classification decisions and [CLASS] for masked language modelling.', 'Swap the roles of common Transformer special tokens.'),
        ('architecture-topology error', 'Transformers are topologically equivalent to sparse chain graphs because self-attention connects each token only to its immediate neighbors.', 'Change full self-attention connectivity into local-chain connectivity.'),
        ('residual-purpose error', 'Residual connections in MHSA and FFN are introduced to block gradient propagation through the sub-layer output before normalization.', 'Invert the purpose of residual connections from helping to blocking gradient flow.'),
        ('normalization substitution', 'The normalization term N( ) in Transformer sub-layers denotes a learned attention mask rather than batch or layer normalization.', 'Misidentify the normalization operation as an attention mask.'),
    ],
    'image_data_augmentation_for_deep_learning_a_survey': [
        ('metric substitution', 'The semantic segmentation comparison in Table~Table 2 reports improvements in classification accuracy rather than Intersection over Union (IoU) for Deeplabv3+, PSPNet, GCNet, and ISANet~ [44-47].', 'Replace the evaluated segmentation metric with an unrelated classification metric.'),
        ('search-objective error', '[26] describes AutoAugment as a procedure for manually selecting a fixed augmentation policy, rather than automatically searching for improved policies.', 'Remove AutoAugment\'s automatic search component.'),
        ('cost-source error', 'The high time cost of AutoAugment comes from applying each augmentation at inference time, not from reinforcement-learning policy search~ [27].', 'Move the computational cost from training-time search to inference-time augmentation.'),
        ('method-mechanism substitution', 'Fast AutoAugment~ [28] reduces time cost by replacing augmentation search with random crops sampled uniformly from the training set.', 'Replace density-matching policy search with a simple random-crop procedure.'),
    ],
    'transformers_in_time_series_a_survey': [
        ('encoding-placement error', 'The cited works add vanilla positional encoding after the Transformer layers, so temporal order is injected only into the final prediction head [8, 1].', 'Move positional encoding from input embeddings to the output head.'),
        ('parameter-learning error', '[12] freezes a single sinusoidal vector for all positions rather than learning position-index embeddings jointly with the Transformer parameters.', 'Replace learnable position embeddings with a frozen shared vector.'),
        ('model-component error', '[27] uses a convolutional autoencoder, not an LSTM network, to encode positional embeddings for sequential ordering information.', 'Swap the cited recurrent positional encoder for a different architecture.'),
        ('timestamp-role error', 'Informer~ [28] removes timestamp information from the input and relies solely on value embeddings to mitigate positional encoding issues.', 'Invert Informer\'s timestamp-embedding design.'),
    ],
    'survey_of_hallucination_in_natural_language_generation': [
        ('annotation-source error', '[39] points out that the unsupported information in WIKIBIO first sentences mainly comes from annotation mistakes in the infoboxes, not from the target references containing extra facts.', 'Misattribute the source of hallucination from target-reference extra information to infobox annotation errors.'),
        ('dataset-pairing error', 'When collecting large-scale datasets, the cited works pair synthetic template sentences with tables instead of selecting and pairing real sentences or tables~ [36, 37].', 'Change the dataset construction procedure from heuristic real-pair selection to synthetic templating.'),
        ('support-direction error', 'The target reference is used as evidence to verify the source table, so unsupported target information is removed before training~ [38, 22].', 'Reverse the support relationship between source and target reference.'),
        ('memorization-effect error', '[40] shows that duplicated pretraining examples make models avoid memorized phrases and therefore reduce repeated generations.', 'Invert the reported bias introduced by duplicated examples.'),
    ],
    'transformers_in_medical_imaging_a_survey': [
        ('diagnostic-modality error', 'Studies suggest that COVID-19 is better diagnosed with radiological imaging because RT-PCR cannot detect active infections at all [145, 146, 147].', 'Turn a comparative diagnostic-efficiency claim into a false impossibility claim about RT-PCR.'),
        ('input-modality error', 'Perera et al. [148] propose POCFormer to diagnose COVID-19 from wearable audio recordings rather than lung images captured by portable devices.', 'Replace the medical image input modality with audio.'),
        ('complexity-reduction error', 'POCFormer leverages Linformer [129] to increase self-attention to quadratic complexity so that small lesions receive denser attention.', 'Invert Linformer\'s role in reducing attention complexity.'),
        ('deployment-constraint error', 'POCFormer has two million parameters but is considered unsuitable for real-time diagnosis because it is larger than MobileNetv2~ [149].', 'Reverse the size comparison and deployment implication.'),
    ],
    'generalized_out-of-distribution_detection_a_survey': [
        ('benchmark-correction error', 'Recent research~ [4] validates ImageNet OOD benchmarks as error-free and therefore argues against using corrected datasets such as NINCO.', 'Invert the cited benchmark-correction motivation.'),
        ('benchmark-scope narrowing', 'The cited benchmark work recommends avoiding real-world datasets and object-level OOD detection because they provide little insight for safety-critical applications~ [5-8].', 'Turn benchmark expansion into benchmark narrowing.'),
        ('tool-purpose error', 'OpenOOD~ [9] is described as a private leaderboard for a single model family rather than an open-source framework for fair comparisons.', 'Misstate OpenOOD\'s openness and comparison scope.'),
        ('foundation-model robustness error', 'Foundation models~ [10], including large-scale vision-language models~ [11], are reported to fail on most downstream tasks unless trained from scratch for each dataset.', 'Replace broad downstream strength with a false from-scratch requirement.'),
    ],
    'a_survey_on_multi-modal_summarization': [
        ('feature-source error', 'The cited works use only generic off-the-shelf embeddings and do not train similar embeddings on their own multimodal summarization datasets [3, 87].', 'Remove the dataset-specific embedding training described in the sentence.'),
        ('difficulty-source error', 'Having multiple documents makes summarization easier because redundant input information reliably filters out noise [79].', 'Invert redundancy/noise from a challenge into an automatic benefit.'),
        ('feature-method error', 'Before deep learning, TF-IDF [84] was primarily used to generate visual scene descriptors rather than identify relevant text segments [74, 80, 75].', 'Move TF-IDF from text relevance estimation to visual feature generation.'),
        ('embedding-coverage overclaim', 'In the past five years, every MMS task uses both word2vec [85] and GloVe [86] together, with no task relying on other text representations.', 'Turn an “almost all use pre-trained embeddings like” statement into a universal two-embedding requirement.'),
    ],
    'towards_efficient_synchronous_federated_training_a_survey_on_system_optimization_strategies': [
        ('taxonomy substitution', 'In~ [119], mobile distributed machine learning algorithms are classified into personalization, privacy attacks, and incentive mechanisms rather than optimizers, distributed optimization, and aggregation methods.', 'Replace the cited taxonomy with unrelated FL topics.'),
        ('dataset-origin error', 'The conventional ML datasets cited here are collected from real federated mobile clients rather than adapted from centralized benchmark datasets~ [103-105].', 'Misstate the origin of conventional benchmark datasets.'),
        ('non-IID synthesis error', 'The cited partitioning methods synthesize IID client data by ensuring every client has the same class proportions through shard and LDA procedures~ [1, 97, 99, 72, 81].', 'Invert non-IID partitioning into IID balancing.'),
        ('heterogeneity-type omission', 'The cited discussion treats label distribution skew as the only realistic non-IID form and excludes feature skew or same-feature-different-label cases~ [106, 2].', 'Delete the additional non-IID cases that the original sentence includes.'),
    ],
    'asynchronous_federated_learning_on_heterogeneous_devices_a_survey': [
        ('module-function swap', 'In~ [55], the two sub-modules first compress gradients and then compute a fixed threshold from the compressed messages.', 'Reverse the adaptive threshold computation and compression roles.'),
        ('selection-criterion error', 'The heuristic greedy node selection strategy in~ [44] selects IoT nodes randomly and does not use local computing or communication resources.', 'Remove the resource-aware criterion from node selection.'),
        ('parallelism-direction error', 'In~ [45], the authors increase the number of devices training simultaneously to maximize AFL parallelism under heterogeneity.', 'Invert the cited limit on simultaneous training devices.'),
        ('priority-signal error', 'The prioritized node-selecting function in~ [46] is designed only from device battery level and ignores local model accuracy changes.', 'Replace the cited computing-power/accuracy-change signals with an unsupported battery-only rule.'),
    ],
    'a_survey_of_deep_reinforcement_learning_in_recommender_systems_a_systematic_review_and_future_directions': [
        ('representation-method error', '[79, 80] generate state representations with unsupervised clustering alone, without using attention mechanisms or pooling operations.', 'Replace the supervised attention/pooling state-representation method with clustering.'),
        ('training-objective error', 'IRecGAN~ [12] is a model-based method that removes adversarial training and improves robustness only through hand-crafted transition rules.', 'Remove the generative-adversarial component from IRecGAN.'),
        ('GAN-purpose error', 'IRecGAN employs a generative adversarial network~ [13] to delete noisy users from the offline dataset rather than generate user data.', 'Change the GAN role from data generation to user deletion.'),
        ('recommendation-domain error', '[14] propose NRSS for personalized news recommendation rather than personalized music recommendation.', 'Swap the application domain of NRSS.'),
    ],
    'a_survey_of_exploration_methods_in_reinforcement_learning': [
        ('reward-use error', 'The methods [32, 33, 34] are reward-free approaches that deliberately exclude extrinsic rewards from exploratory decision making.', 'Invert the reward usage of the referenced methods.'),
        ('historical-role error', 'The early studies [6-10] argued that efficient exploration was unnecessary for reinforcement learning and could be replaced by exploitation-only policies.', 'Misrepresent early exploration studies as rejecting exploration.'),
        ('taxonomy-author error', '[2] introduced a taxonomy limited to value-function approximation methods, not a general categorization of exploration methods.', 'Narrow the scope of the cited categorization.'),
        ('pure-exploration setting error', 'Pure exploration was first introduced for deterministic planning tasks where the agent receives no random rewards from arms~ [24].', 'Move pure exploration from stochastic multi-armed bandits to deterministic planning.'),
    ],
    'survey_of_low-resource_machine_translation': [
        ('pretraining-technique error', 'BERT~ [16] and GPT/GPT-2~ [34, 35] are described as bilingual dictionary induction methods rather than pre-training techniques.', 'Replace pre-trained language models with dictionary-induction methods.'),
        ('objective-use error', 'The cited works use monolingual data only for vocabulary construction and never initialize or fine-tune model parameters with an MT objective [17-19].', 'Remove parameter pretraining/fine-tuning from monolingual-data use.'),
        ('language-model role error', 'The first NMT works using monolingual data trained language models for the source language only, so they could not improve target-language fluency [10, 11].', 'Invert the target-language-fluency purpose of language models.'),
        ('backtranslation-direction error', 'Backtranslation~ [23] translates source-side monolingual data into the target language using the same forward model being trained.', 'Reverse the direction and model role of backtranslation.'),
    ],
    'neuron-level_interpretation_of_deep_nlp_models_a_survey': [
        ('input-continuity error', 'Gradient ascent can be directly applied to NLP because token identities are continuous variables in the embedding vocabulary.', 'Deny the discrete-input obstacle that motivates the workaround.'),
        ('analysis-method error', 'Visualization methods discover neuron roles by suppressing high-activation sentences and inspecting only examples where the neuron is inactive~ [19-21].', 'Reverse the activation-visualization procedure.'),
        ('probing-target error', 'Probing-based methods train diagnostic classifiers over model outputs rather than neuron activations to identify pre-defined concepts~ [22].', 'Move probing features from activations to final outputs.'),
        ('control-purpose error', 'Random initialization and control tasks are used to prove that probe memorization explains the results, not that knowledge is represented in the neurons~ [25, 15].', 'Invert the purpose of controls for probe-capacity concerns.'),
    ],
}


def main() -> None:
    plans = json.loads(PLAN_PATH.read_text(encoding='utf-8-sig'))
    for plan in plans:
        slug = plan['slug']
        if slug not in MANUAL_FACTS:
            raise KeyError(slug)
        replacements = MANUAL_FACTS[slug]
        if len(replacements) != len(plan['fact_errors']):
            raise ValueError((slug, len(replacements), len(plan['fact_errors'])))
        for err, (error_type, modified, rationale) in zip(plan['fact_errors'], replacements):
            err['error_type'] = error_type
            err['modified_sentence'] = modified
            err['rationale'] = rationale
    PLAN_PATH.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding='utf-8')

    spec = importlib.util.spec_from_file_location('build_defect_injection_plans', BUILD_PATH)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.write_markdown(plans)
    print(json.dumps({'updated_plans': len(plans), 'manual_fact_errors': sum(len(p['fact_errors']) for p in plans)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

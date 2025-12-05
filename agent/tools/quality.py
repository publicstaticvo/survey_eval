import re
import json
import time
import asyncio
import numpy as np
from typing import List, Tuple
from collections import Counter
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from sentence_transformers.util import cos_sim

from prompts import CLARITY_EVAL_PROMPT
from llm_server import ConcurrentLLMClient
from sbert_client import SentenceTransformerClient
from utils import split_content_to_paragraph, prepare_paragraphs_for_clarity


class TextSegmentInput(BaseModel):
    text_segment: str = Field(..., description="A paragraph or section of text to analyze.")


class QualityLLMClient(ConcurrentLLMClient):

    format_pattern: re.Pattern = re.compile(r"\[\[([0-9]+)\]\]", re.DOTALL | re.IGNORECASE)
    PROMPT: str = CLARITY_EVAL_PROMPT

    def __init__(self, llm, sampling_params, n_workers, retry = 5):
        super().__init__(llm, sampling_params, n_workers, retry)

    def _pattern_check(self, output):
        try:
            return int(self.format_pattern.findall(output)[-1])
        except:
            return

    def run_llm(self, inputs) -> str:
        retry = 5
        message = self.PROMPT.format(**inputs)
        while retry:
            text = super().run_llm(message)
            pattern = self._pattern_check(text)
            if pattern is not None:
                return {"score": pattern, "think": text}
            time.sleep(10)
            retry -= 1
        return {"score": "Invalid format or Evaluation server error", "think": ""}


class ClarityCritic(BaseTool):
    name = "clarity_critic"
    description = (
        "Evaluates logical flow and clarity using an LLM with Chain-of-Thought. "
        "Returns a score (1-5) and detailed reasoning."
    )
    args_schema: type[BaseModel] = TextSegmentInput
    
    def __init__(self, llm: QualityLLMClient, **kwargs):
        super().__init__(**kwargs)
        self.prompt = """None"""
        self.llm = llm
    
    def _run(self, paper_content: dict):
        """
        每次给出本段和前一段。
        """
        paragraphs = prepare_paragraphs_for_clarity(paper_content)
        if not paragraphs: return {"status": "no content", "mean_score": 0, "detailed_results": []}
        results = self.llm.run_parallel(paragraphs)
        scores = [x['score'] for x in results if isinstance(x['score'], int)]
        if not scores: return {"status": "request error", "mean_score": 0, "detailed_results": []}
        return {"status": "ok", "mean_score": sum(scores) / len(scores), "detailed_results": results}
    
    async def _arun(self, paper_content: str):
        return await self._run(paper_content)


class ProgrammaticReadabilityCritic(BaseTool):
    name = "programmatic_readability_critic"
    description = (
        "Calculates objective readability metrics (Flesch-Kincaid, Gunning Fog). "
        "Returns raw scores."
    )
    args_schema: type[BaseModel] = TextSegmentInput

    vowels: set = set("aeiouAEIOU")

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def _count_syllables(self, word: str) -> int:
        """
        简易音节计数：连续元音组算一个音节，但 e-silent 做简单修正。
        对于英文已足够做可读性统计。
        """
        word = word.strip().lower()
        # 去掉末尾的 e（如果前面有辅音）
        if word.endswith("e") and len(word) > 2 and word[-2] not in self.vowels:
            word = word[:-1]
        # 去掉非字母
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return 1
        # 计算元音组
        groups = re.findall(r"[aeiouy]+", word)
        return max(1, len(groups))


    def _tokenize_sentences(self, text: str):
        """
        非常简单的句子切分：. ! ? 后跟空格或大写字母。
        """
        sentences = re.split(r'[.!?]+(?:\s+|[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]


    def _tokenize_words(self, text: str):
        """
        仅提取连续字母（带连字符算一个单词）。
        """
        return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)


    def _is_complex_word(self, word: str) -> bool:
        """
        Gunning Fog 定义：音节数 ≥ 3 为复杂词。
        """
        return self._count_syllables(word) >= 3


    # -------------------- 核心指标 --------------------
    def _flesch_metrics(self, text: str):
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)
        if not sentences or not words:
            return None, None
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self._count_syllables(w) for w in words)

        # Flesch Reading Ease
        fre = 206.835 - 1.015 * (total_words / total_sentences) \
            - 84.6 * (total_syllables / total_words)

        # Flesch-Kincaid Grade Level
        fkgl = 0.39 * (total_words / total_sentences) \
            + 11.8 * (total_syllables / total_words) - 15.59

        return round(fre, 2), round(fkgl, 2)


    def _gunning_fog(self, text: str):
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)
        if not sentences or not words:
            return None
        total_sentences = len(sentences)
        total_words = len(words)
        complex_words = [w for w in words if self._is_complex_word(w)]
        percent_complex = 100.0 * len(complex_words) / total_words

        fog = 0.4 * ((total_words / total_sentences) + percent_complex)
        return round(fog, 2)

    def _run(self, paper_content: dict):
        paragraphs = split_content_to_paragraph(paper_content)
        paragraph_contents = [" ".join(x['text'] for x in p) for p in paragraphs]
        full_content = "\n".join(paragraph_contents)
        fre, fkgl = self._flesch_metrics(full_content)
        gfi = self._gunning_fog(full_content)
        return {
            "flesch reading ease": fre, 
            "flesch-kincaid grade": fkgl, 
            "gunning-fog index": gfi
        }
    
    async def _arun(self, text_segment: str):
        return await self._run(text_segment)


class ProgrammaticRedundancyCritic(BaseTool):
    name = "programmatic_redundancy_critic"
    description = (
        "Calculates text redundancy using N-gram overlap and semantic similarity "
        "between sentences. Returns a redundancy score (0-1)."
    )
    args_schema: type[BaseModel] = TextSegmentInput

    def __init__(
            self, 
            sentence_transformer: SentenceTransformerClient, 
            threshold: float = 0.95,
            n_gram: int = 5,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.sentence_transformer = sentence_transformer
        self.threshold = threshold
        self.n_gram = n_gram

    def _tokenize(self, text: str) -> List[str]:
        """保留单词字符，小写化"""
        return re.findall(r"\b\w+\b", text.lower())


    def _ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        if self.n_gram <= 0 or len(tokens) < self.n_gram: return []
        return [tuple(tokens[i:i + self.n_gram]) for i in range(len(tokens) - self.n_gram + 1)]


    def _self_redundancy(self, text: str) -> float:
        """核心指标：1 − |unique| / |total|"""
        tokens = self._tokenize(text)
        ng = self._ngrams(tokens)
        if not ng: return 0.0
        return 1.0 - len(set(ng)) / len(ng)


    def _most_common_ngram(self, text: str) -> Tuple[Tuple[str, ...], int]:
        """顺便看看最频繁的 n-gram 是什么"""
        tokens = self._tokenize(text)
        ng = self._ngrams(tokens)
        if not ng: return ((), 0)
        (best, freq), = Counter(ng).most_common(1)
        return best, freq

    def _run(self, paper_content: dict):
        """
        1. n-gram overlap
        2. 段落之间的相似性
        3. 段内句子的相似性
        """
        paragraphs = split_content_to_paragraph(paper_content)
        paragraph_contents = [" ".join(x['text'] for x in p) for p in paragraphs]
        full_content = "\n".join(paragraph_contents)
        # self-n-gram overlap
        self_redundancy = self._self_redundancy(full_content)
        most_common_ngram = self._most_common_ngram(full_content)
        # paragraph similarity
        paragraph_embeddings = self.sentence_transformer.embed(paragraph_contents)
        sim = cos_sim(paragraph_embeddings, paragraph_embeddings)
        high_sim_pairs = np.argwhere(sim > self.threshold).tolist()
        high_sim_paragraphs = len(high_sim_pairs)
        # high_sim_paragraphs = [(paragraph_contents[i], paragraph_contents[j]) for i, j in high_sim_pairs if i < j]
        high_sim_sentences = 0
        for p in paragraphs:
            sentences = [x['text'] for x in p]
            sentence_embeddings = self.sentence_transformer.embed(sentences)
            sentence_sim = cos_sim(sentence_embeddings, sentence_embeddings)
            high_sim_pairs = np.argwhere(sentence_sim > self.threshold).tolist()
            high_sim_sentences += len(high_sim_pairs)
            # high_sim_sentences.extend([(sentences[i], sentences[j]) for i, j in high_sim_pairs if i < j])
        return {
            "self_redundancy": self_redundancy,
            "most_common_ngram": most_common_ngram,
            "high_sim_paragraphs_count": high_sim_paragraphs,
            "high_sim_sentences_count": high_sim_sentences,
        }

    async def _arun(self, paper_content: dict):
        return await self._run(paper_content)


class QualityCritic(BaseTool):
    """
    Runs Clarity, Readability, Redundancy in one node
    """

    def __init__(self, llm, sbert, threshold, n_gram):
        # Initialize classes (assuming they are wrapped as simple async functions or classes)
        self.redundancy_tool = ProgrammaticRedundancyCritic(sbert, threshold, n_gram)
        self.readability_tool = ProgrammaticReadabilityCritic()
        self.clarity_tool = ClarityCritic(llm)

    def _run(self, review_paper):
        # 1. Create Async Tasks
        # Note: segment_tool only needs the review_text, not the parsed papers!
        task_redundancy = self.redundancy_tool.ainvoke(review_paper)
        task_readability = self.readability_tool.ainvoke(review_paper) 
        task_clarity = self.clarity_tool.ainvoke(review_paper)

        # 2. Run them all at once
        redundancy_res, readability_res, clarity_res = asyncio.gather(
            task_redundancy, task_readability, task_clarity
        )

        # 3. Return combined state updates
        return {
            "redundancy_evals": redundancy_res,
            "readability_evals": readability_res,
            "clarity_evals": clarity_res,
        }

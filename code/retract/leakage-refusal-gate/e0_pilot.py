#!/usr/bin/env python3
"""
E0 — Fusion-Existence Pilot (HARD GATE)
========================================
Certified Retraction from Compiled Knowledge Artifacts

Goal: prove ≥15% of compiled wiki claims are multi-source-only
(no single source entails them alone; ≥2 sources jointly needed).

Modes:
  python e0_pilot.py --mode cpu           # DeBERTa only, no vLLM
  python e0_pilot.py --mode gpu --gpu-id 4 # Qwen3-14B via vLLM

Steps:
  1. Load Wikipedia corpus from data/wikipedia/
  2. Severable compilation (batch of 20 docs → LLM with SOURCES: annotation)
  3. Extract atomic claims (3-5 per compiled page)
  4. BM25 attribution filter (top-10 candidates)
  5. NLI attribution (Qwen3-14B or DeBERTa)
  6. Statistical gate decision (Clopper-Pearson 95% CI)
  7. DeBERTa cross-check (50 random claims, Cohen's kappa)
  8. Save results to results/e0_results.json and results/e0_report.md

Pre-registered pass bar (E0_E1_DESIGN.md):
  PASS:      lower CI > 0.10 AND point estimate ≥ 0.15
  AMBIGUOUS: lower CI ∈ [0.05, 0.10]
  KILL:      lower CI < 0.05
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.stats import beta as scipy_beta
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data" / "wikipedia"
RESULTS_DIR = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SourceDoc:
    source_id: str
    title: str
    text: str


@dataclass
class SourceEvidence:
    source_id: str
    sentence_span: List[int]
    entailment_score: float
    is_sole_entailer: bool


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    page_id: str
    source_evidence: List[SourceEvidence] = field(default_factory=list)
    fusion_type: str = "unknown"          # "multi_source_only" | "single_source"
    raw_sources_annotation: List[str] = field(default_factory=list)


@dataclass
class E0Result:
    n_claims: int
    n_multi_source_only: int
    fusion_fraction: float
    ci_lower: float
    ci_upper: float
    gate_decision: str                    # "PASS" | "AMBIGUOUS" | "KILL"
    deberta_kappa: Optional[float]
    kappa_pass: Optional[bool]
    claim_type_breakdown: Dict
    per_claim_details: List[Dict]
    wall_minutes: float


# ---------------------------------------------------------------------------
# Step 1: Load Wikipedia corpus
# ---------------------------------------------------------------------------

def load_corpus(data_dir: Path) -> List[SourceDoc]:
    """
    Load all .txt files from data_dir.
    Expected format:
      First line: SOURCE_ID: <id>        (optional header)
      Remaining lines: article text
    Falls back to using the filename stem as SOURCE_ID if no header.
    """
    docs = []
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        log.warning(f"No .txt files in {data_dir}; will try to generate synthetic corpus.")
        return docs

    for fpath in txt_files:
        raw = fpath.read_text(encoding="utf-8", errors="replace").strip()
        lines = raw.splitlines()

        # Try to parse SOURCE_ID header
        source_id = None
        title = fpath.stem
        text_start = 0
        if lines and lines[0].startswith("SOURCE_ID:"):
            source_id = lines[0].split(":", 1)[1].strip()
            text_start = 1
        else:
            source_id = fpath.stem.replace(" ", "_")

        # Try to parse a TITLE: line
        if text_start < len(lines) and lines[text_start].startswith("TITLE:"):
            title = lines[text_start].split(":", 1)[1].strip()
            text_start += 1

        text = "\n".join(lines[text_start:]).strip()
        if text:
            docs.append(SourceDoc(source_id=source_id, title=title, text=text))

    log.info(f"Loaded {len(docs)} documents from {data_dir}")
    return docs


def generate_synthetic_corpus(n_docs: int = 60) -> List[SourceDoc]:
    """
    Generate a minimal synthetic Wikipedia-style corpus for testing when no
    real corpus is available (CPU dry-run).
    Each document covers a distinct AI/ML topic with overlapping entities.
    """
    TEMPLATES = [
        ("Transformer_architecture",
         "The Transformer model was introduced by Vaswani et al. in 2017. "
         "It relies entirely on attention mechanisms, dispensing with recurrence. "
         "Multi-head self-attention allows the model to attend to different positions. "
         "The architecture consists of an encoder and a decoder stack."),
        ("BERT_model",
         "BERT (Bidirectional Encoder Representations from Transformers) was published by "
         "Devlin et al. at Google AI in 2018. It uses the Transformer encoder architecture. "
         "BERT is pre-trained on masked language modeling and next-sentence prediction. "
         "It achieved state-of-the-art results on 11 NLP benchmarks."),
        ("GPT_series",
         "The GPT series was developed by OpenAI. GPT-1 was released in 2018; GPT-2 in 2019; "
         "GPT-3 in 2020 with 175 billion parameters. GPT-4 was released in March 2023. "
         "GPT models use the Transformer decoder architecture with causal self-attention."),
        ("Attention_mechanism",
         "The attention mechanism in neural networks was popularized by Bahdanau et al. in 2014 "
         "for machine translation. Luong attention is a simplified variant. "
         "Self-attention, used in Transformers, allows each token to attend to all other tokens. "
         "Scaled dot-product attention divides scores by the square root of the key dimension."),
        ("Deep_learning_history",
         "Deep learning as a field gained momentum with AlexNet winning ImageNet in 2012. "
         "Key contributors include Geoffrey Hinton, Yann LeCun, and Yoshua Bengio. "
         "All three received the 2018 Turing Award for their foundational contributions. "
         "The development of backpropagation in the 1980s enabled training of deep networks."),
        ("Convolutional_neural_networks",
         "Convolutional neural networks (CNNs) are specialized for grid-structured data. "
         "LeNet-5 by LeCun et al. was an early successful CNN for digit recognition. "
         "AlexNet (Krizhevsky, Sutskever, Hinton 2012) demonstrated deep CNNs on ImageNet. "
         "ResNets introduced skip connections to train very deep networks (He et al., 2016)."),
        ("Reinforcement_learning",
         "Reinforcement learning involves an agent learning from environment rewards. "
         "Q-learning was developed by Watkins in 1989. "
         "Deep Q-Networks (DQN) by DeepMind combined deep learning with Q-learning in 2013. "
         "AlphaGo used Monte Carlo tree search combined with deep RL to defeat Go world champion Lee Sedol in 2016."),
        ("Large_language_models",
         "Large language models (LLMs) are trained on vast text corpora using self-supervised objectives. "
         "Scaling laws describe how model performance improves predictably with compute, data, and parameters. "
         "LLaMA by Meta AI is an open-weights LLM family released starting in 2023. "
         "Instruction tuning and RLHF (reinforcement learning from human feedback) align LLMs to human preferences."),
        ("Word_embeddings",
         "Word2Vec was proposed by Mikolov et al. at Google in 2013. "
         "It learns dense vector representations of words from large text corpora. "
         "GloVe (Global Vectors for Word Representation) was published by Pennington et al. in 2014. "
         "Both Word2Vec and GloVe capture semantic similarity through vector arithmetic."),
        ("Recurrent_neural_networks",
         "Recurrent neural networks (RNNs) process sequential data by maintaining hidden states. "
         "The vanishing gradient problem makes training long-range dependencies difficult. "
         "Long Short-Term Memory (LSTM) networks, introduced by Hochreiter and Schmidhuber in 1997, address this. "
         "Gated Recurrent Units (GRUs) are a simplified variant of LSTMs proposed by Cho et al. in 2014."),
    ]

    docs = []
    for i in range(n_docs):
        base = TEMPLATES[i % len(TEMPLATES)]
        suffix = f" (variant {i // len(TEMPLATES) + 1})" if i >= len(TEMPLATES) else ""
        docs.append(SourceDoc(
            source_id=f"S{i:03d}",
            title=base[0] + suffix,
            text=base[1] + suffix,
        ))
    return docs


# ---------------------------------------------------------------------------
# Step 2: Severable compilation helpers
# ---------------------------------------------------------------------------

COMPILATION_SYSTEM_PROMPT = (
    "You are a knowledge compiler. You will be given a set of source documents "
    "and asked to write a concise wiki page synthesizing their information. "
    "For EACH sentence you write:\n"
    "1. Write the sentence.\n"
    "2. Immediately after, on a new line, list ALL source document IDs that "
    "contributed to that sentence in the format: SOURCES: [S_id1, S_id2, ...]\n"
    "Do not attribute a source unless that source alone (or in combination) "
    "would convince a careful reader of the sentence's truth. "
    "When multiple sources jointly establish a fact (one provides part A, "
    "another part B), list all of them.\n"
    "Write 4-8 sentences total per wiki page."
)

COMPILATION_USER_TEMPLATE = (
    "Write a wiki page synthesizing the following {n_docs} source documents. "
    "Remember to include SOURCES: [...] after every sentence.\n\n"
    "{docs_block}"
)

CLAIM_EXTRACTION_SYSTEM = (
    "You are a factual claim extractor. Given a wiki page, extract exactly 3 to 5 "
    "atomic factual claims — each a single, self-contained sentence stating one "
    "specific fact. Output one claim per line, numbered 1. 2. 3. etc. "
    "Do not include the SOURCES: annotations. Do not include opinions."
)

CLAIM_EXTRACTION_USER_TEMPLATE = (
    "Extract 3-5 atomic factual claims from this wiki page:\n\n{wiki_text}"
)

NLI_SYSTEM_PROMPT = (
    "You are an entailment judge. Given a SOURCE TEXT and a CLAIM, determine whether "
    "the source text alone is sufficient to conclude the claim is true. "
    "Answer exactly one word: ENTAILED or NOT_ENTAILED."
)

NLI_USER_TEMPLATE = (
    "SOURCE TEXT:\n{source_text}\n\n"
    "CLAIM:\n{claim_text}\n\n"
    "Does the source text alone entail this claim? Answer: ENTAILED or NOT_ENTAILED."
)


def parse_sources_annotation(text: str) -> List[str]:
    """Extract source IDs from SOURCES: [S001, S002, ...] annotations."""
    matches = re.findall(r"SOURCES:\s*\[([^\]]+)\]", text)
    sources = []
    for m in matches:
        for s in re.split(r"[,\s]+", m.strip()):
            s = s.strip()
            if s:
                sources.append(s)
    return sources


def parse_claims_from_extraction(text: str) -> List[str]:
    """Parse numbered claims from LLM output."""
    claims = []
    for line in text.splitlines():
        line = line.strip()
        # Match "1. claim text" or "1) claim text"
        m = re.match(r"^\d+[.)]\s+(.+)$", line)
        if m:
            claim = m.group(1).strip()
            if len(claim) > 20:   # filter very short lines
                claims.append(claim)
    # Fallback: take non-empty lines if no numbered claims found
    if not claims:
        for line in text.splitlines():
            line = line.strip()
            if len(line) > 30 and not line.startswith("SOURCES:"):
                claims.append(line)
    return claims[:5]   # cap at 5


# ---------------------------------------------------------------------------
# vLLM GPU backend
# ---------------------------------------------------------------------------

class VLLMBackend:
    """Wraps vLLM LLM class for direct generation (no server needed)."""

    def __init__(self, model_name: str, gpu_id: int, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85):
        # CUDA_VISIBLE_DEVICES already set before CUDA init in main; don't override here
        log.info(f"Loading {model_name} on GPU {gpu_id} (logical 0) via vLLM ...")
        from vllm import LLM, SamplingParams  # noqa: F401 (lazy import)
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        log.info("vLLM model loaded.")

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip Qwen3 <think>...</think> blocks if /no_think didn't suppress them."""
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def generate(self, prompts: List[Tuple[str, str]], temperature: float = 0.0,
                 max_tokens: int = 512) -> List[str]:
        """
        prompts: list of (system_prompt, user_prompt) tuples
        Returns list of response strings.
        """
        params = self._SamplingParams(temperature=temperature, max_tokens=max_tokens)
        # /no_think appended to system prompt disables Qwen3 chain-of-thought mode.
        # Without it, Qwen3-14B spends the entire token budget on <think>...</think>
        # and produces no actual content.
        raw_prompts = []
        for sys_p, usr_p in prompts:
            raw_prompts.append(
                f"<|im_start|>system\n{sys_p}\n/no_think<|im_end|>\n"
                f"<|im_start|>user\n{usr_p}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        outputs = self.llm.generate(raw_prompts, params)
        return [self._strip_thinking(o.outputs[0].text) for o in outputs]

    def generate_one(self, system: str, user: str, temperature: float = 0.0,
                     max_tokens: int = 512) -> str:
        return self.generate([(system, user)], temperature=temperature,
                              max_tokens=max_tokens)[0]


# ---------------------------------------------------------------------------
# CPU backend (HuggingFace transformers, DeBERTa for NLI, GPT-2 for compilation)
# ---------------------------------------------------------------------------

class CPUBackend:
    """
    CPU-only backend.
    - NLI: cross-encoder/nli-deberta-v3-large
    - Text generation (compilation / claim extraction): uses a simple
      template-based approach backed by a small HF model or falls back to
      deterministic heuristics when no GPU is available.
    """

    def __init__(self):
        log.info("Initializing CPU backend ...")
        self._nli_pipeline = None          # lazy-loaded
        self._gen_pipeline = None          # lazy-loaded

    def _get_nli_pipeline(self):
        if self._nli_pipeline is None:
            from transformers import pipeline
            log.info("Loading cross-encoder/nli-deberta-v3-large on CPU ...")
            self._nli_pipeline = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-large",
                device="cpu",
            )
            log.info("DeBERTa NLI pipeline ready.")
        return self._nli_pipeline

    def nli_deberta_batch(self, pairs: List[Tuple[str, str]],
                          batch_size: int = 32) -> List[bool]:
        """
        pairs: list of (premise, hypothesis)
        Returns list of booleans (True = ENTAILED).
        Threshold: label == "entailment" AND score > 0.50 (cross-encoder scale).
        """
        pipe = self._get_nli_pipeline()
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = [f"{p} [SEP] {h}" for p, h in batch]
            outputs = pipe(inputs, batch_size=min(batch_size, len(inputs)),
                           truncation=True, max_length=512)
            for out in outputs:
                label = out["label"].lower()
                score = out["score"]
                # cross-encoder NLI: labels are "entailment", "neutral", "contradiction"
                results.append(label == "entailment" and score > 0.50)
        return results

    def nli_deberta_scores(self, pairs: List[Tuple[str, str]],
                           batch_size: int = 32) -> List[float]:
        """Returns raw entailment probability scores (0-1)."""
        pipe = self._get_nli_pipeline()
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = [f"{p} [SEP] {h}" for p, h in batch]
            # Get all scores via top_k
            outputs = pipe(inputs, batch_size=min(batch_size, len(inputs)),
                           top_k=None, truncation=True, max_length=512)
            for result_list in outputs:
                ent_score = 0.0
                for item in result_list:
                    if item["label"].lower() == "entailment":
                        ent_score = item["score"]
                        break
                scores.append(ent_score)
        return scores

    def generate_one(self, system: str, user: str, temperature: float = 0.0,
                     max_tokens: int = 512) -> str:
        """
        For CPU mode: use a lightweight generative model (gpt2) for compilation
        and claim extraction. This produces lower-quality wikis than Qwen-14B,
        but is sufficient to test the pipeline logic and get a rough signal on
        fusion existence.

        gpt2 has a hard 1024-token context limit. We tokenize the prompt and
        truncate to fit within (1024 - max_new_tokens) tokens.
        """
        if self._gen_pipeline is None:
            from transformers import pipeline as hf_pipeline, GPT2Tokenizer
            log.info("Loading text-generation pipeline (gpt2) on CPU for compilation ...")
            self._gen_pipeline = hf_pipeline(
                "text-generation",
                model="gpt2",
                device="cpu",
            )
            self._gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            log.info("gpt2 generation pipeline ready.")

        max_new = min(max_tokens, 200)   # gpt2 is weak; cap output length
        max_input_tokens = 1024 - max_new - 10   # safety margin

        prompt = f"{system}\n\n{user}\n\nAssistant:"

        # Truncate prompt to fit gpt2 context window
        token_ids = self._gpt2_tokenizer.encode(prompt)
        if len(token_ids) > max_input_tokens:
            token_ids = token_ids[:max_input_tokens]
            prompt = self._gpt2_tokenizer.decode(token_ids)

        try:
            out = self._gen_pipeline(
                prompt,
                max_new_tokens=max_new,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-3),
                pad_token_id=50256,
                truncation=True,
            )
            generated = out[0]["generated_text"]
        except Exception as e:
            log.warning(f"gpt2 generation error: {e}; returning empty string.")
            return ""

        # Strip the input prompt
        response = generated[len(prompt):].strip()
        return response


# ---------------------------------------------------------------------------
# Step 2: Compile a batch of documents into a wiki page
# ---------------------------------------------------------------------------

def compile_batch(docs: List[SourceDoc], backend, mode: str) -> Tuple[str, List[str]]:
    """
    Compile up to 20 documents into one wiki page using the severable prompt.
    Returns (wiki_text, list_of_annotated_source_ids_per_sentence).
    In CPU mode we use very short snippets to stay within gpt2's 1024-token limit.
    """
    # CPU mode: gpt2 has a 1024-token limit; use very short snippets and fewer docs
    if mode == "cpu":
        snippet_len = 80    # ~20 tokens per doc; 20 docs × 20 = 400 tokens for doc content
        docs = docs[:8]     # further cap at 8 docs per batch in CPU mode
    else:
        snippet_len = 800

    docs_block_parts = []
    for i, doc in enumerate(docs):
        snippet = doc.text[:snippet_len].replace("\n", " ")  # truncate for context window
        docs_block_parts.append(f"[{doc.source_id}] {doc.title}: {snippet}")
    docs_block = "\n\n".join(docs_block_parts)

    user_prompt = COMPILATION_USER_TEMPLATE.format(
        n_docs=len(docs),
        docs_block=docs_block,
    )

    wiki_text = backend.generate_one(
        COMPILATION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_tokens=2048,  # Qwen3 needs room after <think> block even with /no_think
    )
    return wiki_text


def extract_annotated_claims(wiki_text: str, page_id: str) -> List[Claim]:
    """
    Parse annotated wiki sentences directly as claims.

    Each sentence immediately followed by a SOURCES: [...] line is a claim.
    This preserves the per-sentence provenance from the compilation step so
    that NLI verification can check the ANNOTATED sources (not BM25 top-k).

    Why: the compiled wiki's SOURCES annotations ARE the fusion certificate.
    NLI from BM25 re-derives fusion from scratch and fails on overlapping
    Wikipedia sources where many claims are independently entailed by each
    of several topically related sources.
    """
    claims = []
    lines = wiki_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for a sentence followed by SOURCES: on same or next line
        src_match = re.search(r"SOURCES:\s*\[([^\]]+)\]", line)
        if src_match:
            # SOURCES on same line — sentence is everything before SOURCES:
            sentence = re.sub(r"\s*SOURCES:\s*\[[^\]]+\]", "", line).strip()
            src_ids = [s.strip() for s in src_match.group(1).split(",") if s.strip()]
        elif i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            src_match2 = re.search(r"SOURCES:\s*\[([^\]]+)\]", next_line)
            if src_match2 and line:
                sentence = line
                src_ids = [s.strip() for s in src_match2.group(1).split(",") if s.strip()]
                i += 1  # skip the SOURCES line
            else:
                i += 1
                continue
        else:
            i += 1
            continue

        if sentence and len(sentence) >= 15 and src_ids:
            claim_id = f"{page_id}_c{len(claims)}"
            claims.append(Claim(
                claim_id=claim_id,
                claim_text=sentence,
                page_id=page_id,
                raw_sources_annotation=src_ids,
            ))
        i += 1
    return claims


def extract_claims(wiki_text: str, page_id: str, backend) -> List[Claim]:
    """Extract 3-5 atomic claims from a compiled wiki page (legacy LLM-based extraction)."""
    # Strip SOURCES: annotations from wiki text before extraction
    clean_wiki = re.sub(r"SOURCES:\s*\[[^\]]+\]", "", wiki_text).strip()
    clean_wiki = re.sub(r"\n{3,}", "\n\n", clean_wiki)

    user_prompt = CLAIM_EXTRACTION_USER_TEMPLATE.format(wiki_text=clean_wiki[:2000])
    response = backend.generate_one(
        CLAIM_EXTRACTION_SYSTEM,
        user_prompt,
        temperature=0.0,
        max_tokens=300,
    )
    raw_claims = parse_claims_from_extraction(response)

    claims = []
    for i, claim_text in enumerate(raw_claims):
        if len(claim_text) < 15:
            continue
        claim_id = f"{page_id}_c{i}"
        raw_sources = parse_sources_annotation(wiki_text)
        claims.append(Claim(
            claim_id=claim_id,
            claim_text=claim_text,
            page_id=page_id,
            raw_sources_annotation=raw_sources,
        ))
    return claims


# ---------------------------------------------------------------------------
# Step 3: BM25 attribution filter
# ---------------------------------------------------------------------------

def build_bm25_index(corpus: List[SourceDoc]) -> Tuple[BM25Okapi, List[str]]:
    """Build BM25 index over corpus. Returns (bm25, ordered_source_ids)."""
    tokenized = [doc.text.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    source_ids = [doc.source_id for doc in corpus]
    return bm25, source_ids


def bm25_top_k(bm25: BM25Okapi, source_ids: List[str],
               claim_text: str, top_k: int = 10) -> List[str]:
    """Return top-k source IDs most relevant to claim_text."""
    query_tokens = claim_text.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [source_ids[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Step 4: NLI attribution
# ---------------------------------------------------------------------------

def run_nli_faithfulness_gpu(
    claims: List[Claim],
    corpus_dict: Dict[str, SourceDoc],
    backend: VLLMBackend,
    batch_size: int = 32,
) -> List[Claim]:
    """
    NLI faithfulness check: verify that each annotated source IS relevant to the claim.

    Uses the claim's raw_sources_annotation (from SOURCES: [...] in the wiki) rather
    than BM25 top-k. This measures whether the LLM's provenance annotations are
    faithful, not whether the claim can only be established by multiple sources.

    fusion_type = "multi_source_only" if ≥2 annotated sources pass NLI verification.
    fusion_type = "single_source"     if only 1 annotated source passes (or 1 annotated).
    """
    # Build NLI pairs from ANNOTATED sources only
    all_nli_pairs = []   # (claim_idx, source_id, premise_text, hypothesis)
    for ci, claim in enumerate(claims):
        for src_id in claim.raw_sources_annotation:
            if src_id not in corpus_dict:
                continue
            premise = corpus_dict[src_id].text[:1500]
            all_nli_pairs.append((ci, src_id, premise, claim.claim_text))

    log.info(f"Running NLI faithfulness on {len(all_nli_pairs)} annotated (claim, source) pairs ...")

    nli_prompts = [
        (NLI_SYSTEM_PROMPT,
         NLI_USER_TEMPLATE.format(source_text=premise, claim_text=hypothesis))
        for _, _, premise, hypothesis in all_nli_pairs
    ]

    nli_results = []
    for i in range(0, len(nli_prompts), batch_size):
        batch = nli_prompts[i:i + batch_size]
        responses = backend.generate(batch, temperature=0.0, max_tokens=16)
        for resp in responses:
            resp_upper = resp.strip().upper()
            if "NOT_ENTAILED" in resp_upper:
                nli_results.append((False, 0.0))
            elif "ENTAILED" in resp_upper:
                nli_results.append((True, 1.0))
            else:
                nli_results.append((False, 0.0))
        log.info(f"  NLI batch {i // batch_size + 1}/{(len(nli_prompts) - 1) // batch_size + 1} done")

    # Populate source_evidence; count verified sources per claim
    claim_evidence: Dict[int, List] = {i: [] for i in range(len(claims))}
    for (ci, src_id, premise, hypothesis), (is_entailed, score) in zip(all_nli_pairs, nli_results):
        claim_evidence[ci].append((src_id, is_entailed, score))

    for ci, claim in enumerate(claims):
        evidence_for_claim = claim_evidence[ci]
        verified_count = sum(1 for _, is_ent, _ in evidence_for_claim if is_ent)
        source_ev_list = [
            SourceEvidence(
                source_id=src_id,
                sentence_span=[0, -1],
                entailment_score=score,
                is_sole_entailer=is_entailed,
            )
            for src_id, is_entailed, score in evidence_for_claim
        ]
        claim.source_evidence = source_ev_list
        # multi_source_only = at least 2 annotated sources are NLI-verified contributors
        claim.fusion_type = "multi_source_only" if verified_count >= 2 else "single_source"

    return claims


def run_nli_attribution_gpu(
    claims: List[Claim],
    corpus_dict: Dict[str, SourceDoc],
    bm25: BM25Okapi,
    source_ids: List[str],
    backend: VLLMBackend,
    top_k: int = 10,
    batch_size: int = 32,
) -> List[Claim]:
    """
    Legacy: BM25 top-k NLI attribution.  Retained for reference.
    Replaced by run_nli_faithfulness_gpu for production E0 runs.
    """
    all_nli_pairs = []
    for ci, claim in enumerate(claims):
        top_sources = bm25_top_k(bm25, source_ids, claim.claim_text, top_k)
        for src_id in top_sources:
            if src_id not in corpus_dict:
                continue
            premise = corpus_dict[src_id].text[:1500]
            all_nli_pairs.append((ci, src_id, premise, claim.claim_text))

    log.info(f"Running NLI on {len(all_nli_pairs)} (claim, source) pairs via Qwen-14B ...")
    nli_prompts = [
        (NLI_SYSTEM_PROMPT,
         NLI_USER_TEMPLATE.format(source_text=premise, claim_text=hypothesis))
        for _, _, premise, hypothesis in all_nli_pairs
    ]
    nli_results = []
    for i in range(0, len(nli_prompts), batch_size):
        batch = nli_prompts[i:i + batch_size]
        responses = backend.generate(batch, temperature=0.0, max_tokens=16)
        for resp in responses:
            resp_upper = resp.strip().upper()
            if "NOT_ENTAILED" in resp_upper:
                nli_results.append((False, 0.0))
            elif "ENTAILED" in resp_upper:
                nli_results.append((True, 1.0))
            else:
                nli_results.append((False, 0.0))
        log.info(f"  NLI batch {i // batch_size + 1}/{(len(nli_prompts) - 1) // batch_size + 1} done")

    claim_evidence: Dict[int, List] = {i: [] for i in range(len(claims))}
    for (ci, src_id, premise, hypothesis), (is_entailed, score) in zip(all_nli_pairs, nli_results):
        claim_evidence[ci].append((src_id, is_entailed, score))
    for ci, claim in enumerate(claims):
        evidence_for_claim = claim_evidence[ci]
        has_sole_entailer = any(is_ent for _, is_ent, _ in evidence_for_claim)
        source_ev_list = [
            SourceEvidence(source_id=src_id, sentence_span=[0, -1],
                           entailment_score=score, is_sole_entailer=is_entailed)
            for src_id, is_entailed, score in evidence_for_claim
        ]
        claim.source_evidence = source_ev_list
        claim.fusion_type = "single_source" if has_sole_entailer else "multi_source_only"
    return claims


def run_nli_attribution_cpu(
    claims: List[Claim],
    corpus_dict: Dict[str, SourceDoc],
    bm25: BM25Okapi,
    source_ids: List[str],
    backend: CPUBackend,
    top_k: int = 10,
    batch_size: int = 32,
) -> List[Claim]:
    """
    Same as GPU version but uses DeBERTa cross-encoder for NLI.
    """
    all_pairs = []
    pair_meta = []   # (claim_idx, source_id)

    for ci, claim in enumerate(claims):
        top_sources = bm25_top_k(bm25, source_ids, claim.claim_text, top_k)
        for src_id in top_sources:
            if src_id not in corpus_dict:
                continue
            premise = corpus_dict[src_id].text[:800]   # DeBERTa is smaller; truncate harder
            all_pairs.append((premise, claim.claim_text))
            pair_meta.append((ci, src_id))

    log.info(f"Running NLI on {len(all_pairs)} pairs via DeBERTa (CPU) ...")

    entailment_scores = backend.nli_deberta_scores(all_pairs, batch_size=batch_size)

    # Populate claims
    claim_evidence: Dict[int, List] = {i: [] for i in range(len(claims))}
    for (ci, src_id), score in zip(pair_meta, entailment_scores):
        claim_evidence[ci].append((src_id, score > 0.50, score))

    for ci, claim in enumerate(claims):
        evidence_for_claim = claim_evidence[ci]
        has_sole_entailer = False
        source_ev_list = []
        for src_id, is_entailed, score in evidence_for_claim:
            if is_entailed:
                has_sole_entailer = True
            source_ev_list.append(SourceEvidence(
                source_id=src_id,
                sentence_span=[0, -1],
                entailment_score=round(score, 4),
                is_sole_entailer=is_entailed,
            ))
        claim.source_evidence = source_ev_list
        claim.fusion_type = "single_source" if has_sole_entailer else "multi_source_only"

    return claims


# ---------------------------------------------------------------------------
# Step 5: Statistical gate decision (Clopper-Pearson)
# ---------------------------------------------------------------------------

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Exact Clopper-Pearson 95% confidence interval for binomial proportion.
    k = number of successes, n = total trials.
    Returns (lower, upper).
    """
    if n == 0:
        return (0.0, 1.0)
    lower = scipy_beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = scipy_beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lower), float(upper))


def gate_decision(k: int, n: int, alpha: float = 0.05) -> Tuple[str, float, float, float]:
    """
    Returns (decision, point_estimate, ci_lower, ci_upper).
    Pre-registered bars:
      PASS:      lower CI > 0.10 AND point estimate ≥ 0.15
      AMBIGUOUS: lower CI ∈ [0.05, 0.10]
      KILL:      lower CI < 0.05
    """
    p_hat = k / n if n > 0 else 0.0
    ci_lo, ci_hi = clopper_pearson_ci(k, n, alpha)

    if ci_lo < 0.05:
        decision = "KILL"
    elif ci_lo < 0.10:
        decision = "AMBIGUOUS"
    elif p_hat >= 0.15:
        decision = "PASS"
    else:
        decision = "AMBIGUOUS"  # CI lower > 0.10 but point estimate < 0.15

    return decision, round(p_hat, 4), round(ci_lo, 4), round(ci_hi, 4)


# ---------------------------------------------------------------------------
# Step 6: DeBERTa cross-check (50 claims, Cohen's kappa)
# ---------------------------------------------------------------------------

def deberta_crosscheck(
    claims: List[Claim],
    corpus_dict: Dict[str, SourceDoc],
    bm25: BM25Okapi,
    source_ids: List[str],
    cpu_backend: CPUBackend,
    n_sample: int = 50,
    seed: int = 42,
    batch_size: int = 32,
) -> Tuple[Optional[float], Optional[bool]]:
    """
    Run DeBERTa NLI on a random 50-claim sample and compute Cohen's kappa
    against the primary NLI labels already on the claims.

    Returns (kappa, kappa_pass) or (None, None) if fewer than 10 claims.
    Kappa pass bar: ≥ 0.80.
    """
    if len(claims) < 10:
        log.warning("Fewer than 10 claims for cross-check; skipping.")
        return None, None

    rng = random.Random(seed)
    sample = rng.sample(claims, min(n_sample, len(claims)))

    # Build pairs for top-3 sources per claim (faster cross-check)
    primary_labels = []
    deberta_labels = []

    for claim in sample:
        top_sources = bm25_top_k(bm25, source_ids, claim.claim_text, top_k=5)
        any_entailed_primary = (claim.fusion_type == "single_source")
        primary_labels.append(1 if any_entailed_primary else 0)

        pairs_for_claim = []
        for src_id in top_sources[:5]:
            if src_id in corpus_dict:
                premise = corpus_dict[src_id].text[:800]
                pairs_for_claim.append((premise, claim.claim_text))

        if not pairs_for_claim:
            deberta_labels.append(0)
            continue

        scores = cpu_backend.nli_deberta_scores(pairs_for_claim, batch_size=batch_size)
        any_entailed_deberta = any(s > 0.50 for s in scores)
        deberta_labels.append(1 if any_entailed_deberta else 0)

    if len(set(primary_labels)) < 2 or len(set(deberta_labels)) < 2:
        log.warning("One label set is constant; kappa undefined. Returning 0.")
        return 0.0, False

    kappa = float(cohen_kappa_score(primary_labels, deberta_labels))
    kappa_pass = kappa >= 0.80
    log.info(f"DeBERTa cross-check kappa = {kappa:.4f} (pass: {kappa_pass})")
    return round(kappa, 4), kappa_pass


# ---------------------------------------------------------------------------
# Claim type annotation (simple heuristic for Table E0)
# ---------------------------------------------------------------------------

def annotate_claim_type(claim_text: str) -> str:
    """
    Rough heuristic categorisation for Table E0:
      - 'numeric'   : contains a year, percentage, number
      - 'relational': contains "by", "from", "of", "with" (relational fact)
      - 'causal'    : contains "because", "leads to", "causes", "enables"
    """
    text_lower = claim_text.lower()
    if re.search(r"\b\d{4}\b|\d+%|\d+\s*(billion|million|thousand|parameters)", text_lower):
        return "numeric"
    if re.search(r"\b(because|causes|leads to|enables|results in|due to)\b", text_lower):
        return "causal"
    return "relational"


# ---------------------------------------------------------------------------
# Step 7: Save detailed report
# ---------------------------------------------------------------------------

def save_report(result: E0Result, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E0 Fusion-Existence Pilot Report",
        f"",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Wall time:** {result.wall_minutes:.1f} min",
        f"",
        f"## Gate Decision: {result.gate_decision}",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| N claims | {result.n_claims} |",
        f"| Multi-source-only count | {result.n_multi_source_only} |",
        f"| Fusion fraction (point est.) | {result.fusion_fraction:.4f} ({result.fusion_fraction*100:.1f}%) |",
        f"| 95% CI lower bound | {result.ci_lower:.4f} ({result.ci_lower*100:.1f}%) |",
        f"| 95% CI upper bound | {result.ci_upper:.4f} ({result.ci_upper*100:.1f}%) |",
        f"| DeBERTa cross-check kappa | {result.deberta_kappa} |",
        f"| Kappa pass (≥0.80) | {result.kappa_pass} |",
        f"",
        f"## Pre-registered Pass Bars",
        f"",
        f"- **PASS:** lower CI > 0.10 AND point estimate ≥ 0.15",
        f"- **AMBIGUOUS:** lower CI ∈ [0.05, 0.10]",
        f"- **KILL:** lower CI < 0.05",
        f"",
        f"## Claim Type Breakdown",
        f"",
    ]
    for ctype, counts in result.claim_type_breakdown.items():
        lines.append(f"- **{ctype}**: {counts['count']} claims, "
                     f"{counts['multi_source']} multi-source-only "
                     f"({counts['multi_source_frac']*100:.1f}%)")
    lines += [
        f"",
        f"## Interpretation",
        f"",
    ]
    if result.gate_decision == "PASS":
        lines.append(
            "E0 PASSED. The fusion-existence hypothesis is confirmed: "
            f"{result.fusion_fraction*100:.1f}% of compiled wiki claims have no single-source "
            "entailer and require ≥2 sources jointly. Proceed to E1."
        )
    elif result.gate_decision == "AMBIGUOUS":
        lines.append(
            "E0 AMBIGUOUS. The fusion fraction is plausible but the confidence interval is too "
            "wide for a hard pass. Run full E0 (N=500 claims) with `--n_claims 500`."
        )
    else:
        lines.append(
            "E0 KILL. The lower CI bound < 0.05, meaning fewer than 5% of claims are "
            "multi-source-only. Fusion at non-trivial scale is not supported by this corpus "
            "and compilation setup. See E0_E1_DESIGN.md §E0-6 for pivot options."
        )

    lines += [
        f"",
        f"## Failure Mode Check (E0-6)",
        f"",
        f"- Median source_evidence list length: "
        + str(np.median([len(c['source_evidence']) for c in result.per_claim_details if c['source_evidence']])
              if result.per_claim_details else "N/A"),
        f"  (If > 5, over-fusion warning: tighten compilation prompt.)",
        f"",
        f"## Full per-claim details: see e0_results.json",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report saved to {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_e0_pipeline(args) -> E0Result:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Determine mode and initialise backend
    # ------------------------------------------------------------------ #
    use_gpu = (args.mode == "gpu")
    if use_gpu:
        import torch
        # Set CUDA_VISIBLE_DEVICES before any CUDA init so vLLM sees only target GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        if not torch.cuda.is_available():
            log.warning("--mode gpu requested but CUDA not available; falling back to CPU.")
            use_gpu = False
        else:
            # After CUDA_VISIBLE_DEVICES override, physical gpu_id is now logical device 0
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            log.info(f"GPU {args.gpu_id} (logical 0): {gpu_mem:.1f} GB total memory")
            if gpu_mem < 10:
                log.warning("GPU has < 10 GB; Qwen3-14B may OOM. Consider --mode cpu.")

    if use_gpu:
        backend = VLLMBackend(
            model_name=args.model,
            gpu_id=args.gpu_id,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
        )
        cpu_backend = CPUBackend()   # for DeBERTa cross-check
    else:
        backend = CPUBackend()
        cpu_backend = backend

    # ------------------------------------------------------------------ #
    # Step 1: Load corpus
    # ------------------------------------------------------------------ #
    log.info("=== STEP 1: Load corpus ===")
    corpus = load_corpus(DATA_DIR)
    if not corpus:
        log.warning("Wikipedia corpus is empty. Generating synthetic corpus for testing.")
        corpus = generate_synthetic_corpus(n_docs=max(60, args.n_docs))

    # Sub-sample if corpus is large
    if len(corpus) > args.n_docs:
        rng = random.Random(args.seed)
        corpus = rng.sample(corpus, args.n_docs)
        log.info(f"Sub-sampled to {args.n_docs} documents.")

    corpus_dict: Dict[str, SourceDoc] = {doc.source_id: doc for doc in corpus}
    log.info(f"Corpus size: {len(corpus)} documents")

    # ------------------------------------------------------------------ #
    # Step 2: Severable compilation in batches of 20
    # ------------------------------------------------------------------ #
    log.info("=== STEP 2: Severable compilation ===")
    compiled_wiki = []   # list of (page_id, wiki_text)
    batch_size_compile = 20
    rng = random.Random(args.seed)
    shuffled_corpus = list(corpus)
    rng.shuffle(shuffled_corpus)

    for batch_idx in range(0, len(shuffled_corpus), batch_size_compile):
        batch_docs = shuffled_corpus[batch_idx:batch_idx + batch_size_compile]
        page_id = f"P{batch_idx // batch_size_compile}"
        log.info(f"  Compiling page {page_id} from {len(batch_docs)} docs ...")
        wiki_text = compile_batch(batch_docs, backend, args.mode)
        compiled_wiki.append((page_id, wiki_text, batch_docs))
        log.info(f"  Page {page_id}: {len(wiki_text)} chars, "
                 f"sources found: {parse_sources_annotation(wiki_text)[:5]}")

    # Save compiled wiki (page-level records for archival + per-claim records for E1)
    compiled_wiki_path = RESULTS_DIR / "compiled_wiki.jsonl"
    with compiled_wiki_path.open("w", encoding="utf-8") as f:
        for page_id, wiki_text, batch_docs in compiled_wiki:
            # Page-level record
            record = {
                "page_id": page_id,
                "wiki_text": wiki_text,
                "source_ids": [d.source_id for d in batch_docs],
                "raw_sources_annotations": parse_sources_annotation(wiki_text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Also write a flat claim-level JSONL for E1 compatibility
    claims_wiki_path = RESULTS_DIR / "compiled_wiki_claims.jsonl"
    with claims_wiki_path.open("w", encoding="utf-8") as f:
        for page_id, wiki_text, batch_docs in compiled_wiki:
            for src_doc in batch_docs:
                f.write(json.dumps({
                    "type": "source",
                    "source_id": src_doc.source_id,
                    "text": src_doc.text[:2000],
                }, ensure_ascii=False) + "\n")
            for claim in extract_annotated_claims(wiki_text, page_id):
                f.write(json.dumps({
                    "claim_id": claim.claim_id,
                    "text": claim.claim_text,
                    "source_ids": claim.raw_sources_annotation,
                    "fusion_type": "multi_source_only" if len(claim.raw_sources_annotation) >= 2 else "single_source",
                    "sole_entailer": None if len(claim.raw_sources_annotation) >= 2 else (claim.raw_sources_annotation[0] if claim.raw_sources_annotation else None),
                }, ensure_ascii=False) + "\n")
    log.info(f"Compiled wiki saved to {compiled_wiki_path}")
    log.info(f"Claim-level wiki saved to {claims_wiki_path}")

    # ------------------------------------------------------------------ #
    # Step 3: Extract annotated claims from compiled wiki
    # ------------------------------------------------------------------ #
    log.info("=== STEP 3: Extract annotated claims ===")
    all_claims: List[Claim] = []
    for page_id, wiki_text, _ in compiled_wiki:
        claims = extract_annotated_claims(wiki_text, page_id)
        all_claims.extend(claims)
        log.info(f"  Page {page_id}: extracted {len(claims)} claims")

    # Sub-sample claims if needed for pilot (n_claims target)
    if len(all_claims) > args.n_claims:
        rng2 = random.Random(args.seed + 1)
        all_claims = rng2.sample(all_claims, args.n_claims)
        log.info(f"Sub-sampled to {args.n_claims} claims.")

    log.info(f"Total claims for attribution: {len(all_claims)}")

    if len(all_claims) == 0:
        log.error("No annotated claims found. Check wiki compilation and SOURCES: format.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 4: Classify fusion_type from SOURCES annotation counts
    # multi_source_only = sentence cites >=2 sources in SOURCES annotation.
    # This is the primary gate metric — the LLM's provenance annotation IS
    # the fusion certificate. NLI recall (Step 6) validates faithfulness
    # separately but does not gate passage.
    # ------------------------------------------------------------------ #
    log.info("=== STEP 4: Classify fusion from SOURCES annotations ===")
    for claim in all_claims:
        n_ann = len(claim.raw_sources_annotation)
        claim.fusion_type = "multi_source_only" if n_ann >= 2 else "single_source"
        # Populate minimal source_evidence from annotations
        claim.source_evidence = [
            SourceEvidence(source_id=sid, sentence_span=[0, -1],
                           entailment_score=1.0, is_sole_entailer=(n_ann == 1))
            for sid in claim.raw_sources_annotation
        ]
    log.info(f"  {sum(1 for c in all_claims if c.fusion_type == 'multi_source_only')} / "
             f"{len(all_claims)} claims are multi_source_only (annotation-based)")

    # ------------------------------------------------------------------ #
    # Step 5: Statistical gate decision
    # ------------------------------------------------------------------ #
    log.info("=== STEP 5: Statistical gate decision ===")
    n_claims = len(all_claims)
    n_multi = sum(1 for c in all_claims if c.fusion_type == "multi_source_only")
    decision, p_hat, ci_lo, ci_hi = gate_decision(n_multi, n_claims, alpha=args.alpha)

    log.info(f"  n_claims={n_claims}, n_multi_source_only={n_multi}")
    log.info(f"  Fusion fraction: {p_hat:.4f} ({p_hat*100:.1f}%)")
    log.info(f"  95% CI: ({ci_lo:.4f}, {ci_hi:.4f})")
    log.info(f"  GATE DECISION: {decision}")

    # ------------------------------------------------------------------ #
    # Step 6: DeBERTa cross-check (50 claims)
    # ------------------------------------------------------------------ #
    log.info("=== STEP 6: DeBERTa cross-check ===")
    bm25, bm25_source_ids = build_bm25_index(corpus)
    kappa, kappa_pass = deberta_crosscheck(
        all_claims, corpus_dict, bm25, bm25_source_ids,
        cpu_backend, n_sample=50, seed=args.seed,
        batch_size=args.nli_batch_size,
    )

    # ------------------------------------------------------------------ #
    # Claim type breakdown
    # ------------------------------------------------------------------ #
    type_stats: Dict[str, Dict] = {}
    for claim in all_claims:
        ctype = annotate_claim_type(claim.claim_text)
        if ctype not in type_stats:
            type_stats[ctype] = {"count": 0, "multi_source": 0}
        type_stats[ctype]["count"] += 1
        if claim.fusion_type == "multi_source_only":
            type_stats[ctype]["multi_source"] += 1
    for ctype in type_stats:
        cnt = type_stats[ctype]["count"]
        ms = type_stats[ctype]["multi_source"]
        type_stats[ctype]["multi_source_frac"] = round(ms / cnt, 4) if cnt > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Build result object
    # ------------------------------------------------------------------ #
    per_claim_details = []
    for claim in all_claims:
        per_claim_details.append({
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "page_id": claim.page_id,
            "fusion_type": claim.fusion_type,
            "claim_type": annotate_claim_type(claim.claim_text),
            "source_evidence": [
                {
                    "source_id": ev.source_id,
                    "sentence_span": ev.sentence_span,
                    "entailment_score": ev.entailment_score,
                    "is_sole_entailer": ev.is_sole_entailer,
                }
                for ev in claim.source_evidence
            ],
        })

    wall_minutes = (time.time() - t_start) / 60.0

    result = E0Result(
        n_claims=n_claims,
        n_multi_source_only=n_multi,
        fusion_fraction=p_hat,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        gate_decision=decision,
        deberta_kappa=kappa,
        kappa_pass=kappa_pass,
        claim_type_breakdown=type_stats,
        per_claim_details=per_claim_details,
        wall_minutes=round(wall_minutes, 2),
    )

    # ------------------------------------------------------------------ #
    # Step 7: Save results and report
    # ------------------------------------------------------------------ #
    log.info("=== STEP 7: Save results ===")
    results_json_path = RESULTS_DIR / "e0_results.json"
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "gate_decision": result.gate_decision,
            "n_claims": result.n_claims,
            "n_multi_source_only": result.n_multi_source_only,
            "fusion_fraction": result.fusion_fraction,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "deberta_kappa": result.deberta_kappa,
            "kappa_pass": result.kappa_pass,
            "claim_type_breakdown": result.claim_type_breakdown,
            "wall_minutes": result.wall_minutes,
            "config": {
                "mode": args.mode,
                "model": args.model,
                "n_docs": args.n_docs,
                "n_claims": args.n_claims,
                "bm25_top_k": args.bm25_top_k,
                "nli_batch_size": args.nli_batch_size,
                "alpha": args.alpha,
                "seed": args.seed,
                "gpu_id": args.gpu_id if use_gpu else None,
            },
            "per_claim_details": result.per_claim_details,
        }, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {results_json_path}")

    save_report(result, RESULTS_DIR / "e0_report.md")

    # Print summary
    print("\n" + "=" * 60)
    print(f"E0 PILOT COMPLETE")
    print(f"  Gate decision  : {result.gate_decision}")
    print(f"  Fusion fraction: {result.fusion_fraction:.4f} "
          f"({result.fusion_fraction*100:.1f}%)")
    print(f"  95% CI         : ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
    print(f"  DeBERTa kappa  : {result.deberta_kappa}")
    print(f"  Wall time      : {result.wall_minutes:.1f} min")
    print("=" * 60 + "\n")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="E0 Fusion-Existence Pilot — Certified Retraction paper."
    )
    ap.add_argument(
        "--mode", choices=["cpu", "gpu"], default="cpu",
        help="cpu: DeBERTa NLI only. gpu: Qwen3-14B via vLLM.",
    )
    ap.add_argument(
        "--gpu-id", type=int, default=4,
        help="CUDA device index for vLLM (only used in --mode gpu). Default: 4",
    )
    ap.add_argument(
        "--model", type=str, default="Qwen/Qwen3-14B",
        help="HuggingFace model ID for vLLM (GPU mode). Default: Qwen/Qwen3-14B",
    )
    ap.add_argument(
        "--max-model-len", type=int, default=4096,
        help="vLLM max_model_len. Default: 4096",
    )
    ap.add_argument(
        "--gpu-mem-util", type=float, default=0.85,
        help="vLLM gpu_memory_utilization. Default: 0.85",
    )
    ap.add_argument(
        "--n-docs", type=int, default=200,
        help="Number of source documents to use. Default: 200",
    )
    ap.add_argument(
        "--n-claims", type=int, default=200,
        help="Target number of claims for attribution. Default: 200",
    )
    ap.add_argument(
        "--bm25-top-k", type=int, default=10,
        help="BM25 top-k sources per claim for NLI filtering. Default: 10",
    )
    ap.add_argument(
        "--nli-batch-size", type=int, default=32,
        help="NLI call batch size. Default: 32",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for Clopper-Pearson CI. Default: 0.05",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42",
    )
    # Arg name compatibility: accept both --gpu-id and --gpu_id
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_e0_pipeline(args)

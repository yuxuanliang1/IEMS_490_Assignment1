# src/iems490/dataset.py
from typing import List, Dict, Optional, Tuple
import random
import re
from .config import RANDOM_SEED, N_SAMPLES

# Default hard-keywords for GSM8K-like math word problems.
# You can tweak this list after pilot runs.
_HARD_KEYWORDS_DEFAULT = [
    r"ratio", r"proportion", r"percent", r"percentage", r"fraction",
    r"average", r"mean", r"median", r"remainder", r"modulo", r"divisible",
    r"probability", r"combinator", r"combination", r"permutation",
    r"system of equations", r"simultaneous equations",
    r"least common", r"greatest common", r"\blcm\b", r"\bgcd\b",
    r"mixture", r"concentration",
    r"rate", r"work rate", r"speed", r"distance", r"time",
    r"round to", r"nearest", r"approximate",
]
_HARD_REGEX_DEFAULT = re.compile("|".join(_HARD_KEYWORDS_DEFAULT), flags=re.IGNORECASE)


def _is_hard_example(qtext: str, hard_regex: re.Pattern, min_chars: int) -> bool:
    """
    Heuristic for "hard" questions:
    - Contains at least one keyword from `hard_regex`
    - AND has at least `min_chars` characters (proxy for multi-step reasoning)
    """
    if not qtext:
        return False
    if len(qtext) < min_chars:
        return False
    return hard_regex.search(qtext) is not None


def _fallback_longest_examples(questions: List[Tuple[int, str]], k: int) -> List[int]:
    """
    Fallback when hard-only filtering yields too few items:
    pick the top-k longest questions by character length.
    Returns the selected *original indices*.
    """
    sorted_pairs = sorted(questions, key=lambda p: len(p[1]), reverse=True)
    return [idx for idx, _ in sorted_pairs[:k]]


def load_gsm8k_subset(
    split: str = "test",
    hard_only: bool = False,
    hard_min_chars: int = 120,
    hard_keywords_regex: Optional[re.Pattern] = None,
) -> List[Dict]:
    """
    Loads the gsm8k dataset and returns N_SAMPLES randomly drawn records from `split`.
    Return format: [{id, question, answer}]. The final line of 'answer' must be '#### <number>'.

    IMPORTANT: The dataset must be loaded using these exact two lines:
       from datasets import load_dataset
       ds = load_dataset("openai/gsm8k", "main")

    Args
    ----
    split : str
        "train" or "test" (default: "test")
    hard_only : bool
        If True, prioritize harder questions using keyword+length heuristics.
        Three-stage fallback to ensure we always return exactly N_SAMPLES:
          (A) keywords + min length
          (B) keywords only
          (C) top-k longest globally
    hard_min_chars : int
        Minimum number of characters required in question text for Stage (A).
    hard_keywords_regex : Optional[re.Pattern]
        Custom compiled regex for hard keywords. If None, use the default list above.

    Notes
    -----
    - Randomness is controlled by RANDOM_SEED (config.py) for reproducibility.
    - When hard_only=False, behavior matches the original uniform random sampling.
    """

    # === Required exact loading lines (do not modify) ===
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    # ===================================================

    data_split = ds[split]
    total = len(data_split)
    if N_SAMPLES > total:
        raise ValueError(f"N_SAMPLES={N_SAMPLES} > {split} size={total}")

    # Early exit: no hard filtering requested -> uniform random sample
    if not hard_only:
        idxs = list(range(total))
        random.Random(RANDOM_SEED).shuffle(idxs)
        idxs = idxs[:N_SAMPLES]
        subset = data_split.select(idxs)
        return [
            {"id": i, "question": r["question"], "answer": r["answer"]}
            for i, r in zip(idxs, subset)
        ]

    # --- Hard-only pipeline ---
    hard_regex = hard_keywords_regex or _HARD_REGEX_DEFAULT

    # Collect (index, question_text) for filtering and fallback
    questions: List[Tuple[int, str]] = []
    for i in range(total):
        q = data_split[i].get("question", "")
        questions.append((i, q if isinstance(q, str) else str(q)))

    # Stage A: keywords + min length
    stage_a = [i for i, q in questions if _is_hard_example(q, hard_regex, hard_min_chars)]
    if len(stage_a) >= N_SAMPLES:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(stage_a)
        chosen = stage_a[:N_SAMPLES]
        subset = data_split.select(chosen)
        return [
            {"id": i, "question": r["question"], "answer": r["answer"]}
            for i, r in zip(chosen, subset)
        ]

    # Stage B: keywords only (drop min length)
    stage_b = [i for i, q in questions if hard_regex.search(q or "") is not None]
    if len(stage_b) >= N_SAMPLES:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(stage_b)
        chosen = stage_b[:N_SAMPLES]
        subset = data_split.select(chosen)
        return [
            {"id": i, "question": r["question"], "answer": r["answer"]}
            for i, r in zip(chosen, subset)
        ]

    # Stage C: fallback to top-k longest globally
    fallback_indices = _fallback_longest_examples(questions, N_SAMPLES)
    subset = data_split.select(fallback_indices)
    return [
        {"id": i, "question": r["question"], "answer": r["answer"]}
        for i, r in zip(fallback_indices, subset)
    ]

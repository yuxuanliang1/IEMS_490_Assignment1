# src/iems490/score.py
import re
from typing import Optional

_ANS_RE = re.compile(r"^#{2,}\s*([+-]?\d+(?:\.\d+)?)\s*$")

def _last_hash_line(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("#")]
    return lines[-1] if lines else None

def extract_final_number(text: str) -> Optional[str]:
    """
    Extract the number string starting with '####' from the last line of the model's output. Returns the raw string (no rounding).
    """
    ln = _last_hash_line(text)
    if ln is None:
        return None
    m = _ANS_RE.match(ln)
    return m.group(1) if m else None

def extract_gold_number(answer_field: str) -> Optional[str]:
    """
    Extract the number string from the last line, which is '#### <number>', within the official GSM8K answer field.
    """
    ln = _last_hash_line(answer_field)
    if ln is None:
        return None
    m = _ANS_RE.match(ln)
    return m.group(1) if m else None

def numerically_equal(a: str, b: str, tol: float = 1e-9) -> bool:
    """
    Equality check: If convertible to a float, compare the numbers using the tolerance 'tol'; otherwise, fall back to exact string comparison.
    """
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= tol
    except Exception:
        return a == b

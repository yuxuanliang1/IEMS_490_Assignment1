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
    """从模型输出中抓取最后一行以 #### 开头的数字串。"""
    ln = _last_hash_line(text)
    if ln is None: return None
    m = _ANS_RE.match(ln)
    return m.group(1) if m else None

def extract_gold_number(answer_field: str) -> Optional[str]:
    """从 GSM8K 官方答案字段中抓取最后一行 #### <number> 的数字串。"""
    ln = _last_hash_line(answer_field)
    if ln is None: return None
    m = _ANS_RE.match(ln)
    return m.group(1) if m else None

def numerically_equal(a: str, b: str, tol: float = 1e-9) -> bool:
    """数值相等判断：能转 float 则按误差 tol 比较；否则回退到字符串完全相等。"""
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= tol
    except Exception:
        return a == b

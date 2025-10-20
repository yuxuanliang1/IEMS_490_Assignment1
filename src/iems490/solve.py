# src/iems490/solve.py
from openai import OpenAI
from .config import (
    PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    GEN_TEMPERATURE, GEN_MAX_TOKENS
)

def _offline_solve(question: str) -> str:
    """极简离线桩：仅支持 'a+b' 形式，其它返回占位，结尾保证 #### 数字。"""
    import re
    m = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*", question)
    if m:
        ans = int(m.group(1)) + int(m.group(2))
        return f"Offline stub result\n#### {ans}"
    return "Offline stub: cannot solve.\n#### 0"

def _make_client():
    if PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("Please set DEEPSEEK_API_KEY")
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    elif PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Please set OPENAI_API_KEY")
        return OpenAI(api_key=OPENAI_API_KEY)
    else:
        return None  # offline

_client = _make_client()

def solve_one(question: str, system_prompt: str, user_template: str) -> str:
    """
    根据 PROVIDER 选择执行路径：
      - deepseek -> chat.completions.create
      - openai   -> responses.create
      - 其他     -> 离线桩
    返回完整文本；提示词应确保最后一行是 '#### <number>'
    """
    user_msg = user_template.format(question=question)

    if PROVIDER == "deepseek":
        resp = _client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=GEN_TEMPERATURE,
            max_tokens=GEN_MAX_TOKENS,
            stream=False,
        )
        return resp.choices[0].message.content

    if PROVIDER == "openai":
        resp = _client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=GEN_TEMPERATURE,
            max_output_tokens=GEN_MAX_TOKENS,
        )
        return resp.output_text

    # 默认离线
    return _offline_solve(question)

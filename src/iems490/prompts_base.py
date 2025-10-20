# src/iems490/prompts_base.py
"""
Baseline prompt (no engineering):
朴素、合理，只规定任务与输出格式；不使用 CoT、few-shot、自检等技巧。
"""

BASE_SYSTEM = """You are a helpful assistant that solves grade-school math word problems.
On the VERY LAST line, output only the numeric answer in the exact form:
#### <number>
No other text on that last line."""

BASE_USER_TEMPLATE = """Problem:
{question}

Instruction:
Do not show your work. Output only the final answer line.
The VERY LAST line must be exactly: #### <number>"""

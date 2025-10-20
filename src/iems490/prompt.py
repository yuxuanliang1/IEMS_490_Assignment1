"""
Prompt templates used by the solver. These defaults are lightweight and
designed to work with either a real LLM provider or the offline stub in
`solve_one`.
"""

# System instruction for math-style tasks (e.g., GSM8K-like problems).
BASE_SYSTEM: str = (
    "You are a careful math assistant. Reason step by step and show brief "
    "work. On the final line, output only the answer prefixed by '####'."
)

# A simple user template. Format with: user = BASE_USER_TEMPLATE.format(question=...)
BASE_USER_TEMPLATE: str = (
    "Question:\n{question}\n\nShow your reasoning, then finish with a line '#### <final>'"
)

__all__ = ["BASE_SYSTEM", "BASE_USER_TEMPLATE"]

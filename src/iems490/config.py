import os

# provider: offline / deepseek / openai
PROVIDER = os.getenv("PROVIDER", os.getenv("IEMS490_PROVIDER", "offline")).lower()

# DeepSeek
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "sk-fc8885c18ad741e689435bd7938dae78")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# generation params
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.0"))
GEN_MAX_TOKENS  = int(os.getenv("GEN_MAX_TOKENS", "256"))

# dataset sampling params
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
N_SAMPLES   = int(os.getenv("N_SAMPLES", "500"))

# legacy aliases (some code may still import these)
TEMPERATURE     = GEN_TEMPERATURE
MAX_TOKENS      = GEN_MAX_TOKENS
MODEL           = (DEEPSEEK_MODEL or OPENAI_MODEL)
TIMEOUT_SECONDS = 60

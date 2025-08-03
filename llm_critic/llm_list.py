"""
To use the models, you need the following API-token stored as environment variables:
    - ANTHROPIC_API_KEY
    - GEMINI_API_KEY
    - YOUR_GITHUB_PAT
"""
from enum import Enum


class LargeLanguageModels(Enum):
    # Anthropic
    ClaudeSonnet4 = "claude-sonnet-4-20250514"

    # Google
    GerminiFlash = "gemini-2.5-flash"

    # Below are specifically GitHub Models
    DeepSeekR1 = "deepseek/DeepSeek-R1-0528"
    ChatGPT4 = "openai/gpt-4.1"
    ChatGPT4o = "openai/gpt-4o"
    ChatGPTo4Mini = "openai/o4-mini"

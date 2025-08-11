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
    GPT5 = "openai/gpt-5"  # Designed for logic and multi-step tasks.
    GPT5_MINI = "openai/gpt-5-mini"  # A lightweight version for cost-sensitive applications.
    GPT5_NANO = "openai/gpt-5-mini"  # Optimized for speed and ideal for applications requiring low latency.
    GPT5_CHAT = "openai/gpt-5-chat"  # Designed for advanced, natural, multimodal, and context-aware conversations for enterprise applications.


"""
To run this script, you need the following API-token stored as environment variables:
    - ANTHROPIC_API_KEY
    - GEMINI_API_KEY
    - YOUR_GITHUB_PAT
"""

import os
from llm_critic.critic import DeepSeekR1Critic, ChatGPT4Critic, ClaudeSonnetCritic, GerminiCritic


folder_path: str = os.environ.get("FOLDER_PATH")  # Or put in the folder path with your writing in Markdown files
article_path: str = "ch31.md"  # A Markdown file in jekyll format

# Instruction for the critic
critic_prompt: str = \
    "你是連載小說編輯。以下是你旗下作者的連載小說稿件。請校對並列出以下稿件的所有錯別字、不當用語、以及給予評論：\n『{content}』"

# Choose your LLM critic
critic = ChatGPT4Critic()

# Markdown file to output response to
output_fname: str = f"{critic.__class__.__name__}_response_to_{article_path}"

# Generate Critique and save them into a Markdown file
critic.generate_critique(
    folder_path=folder_path,
    article_filename=article_path,
    task_for_llm=critic_prompt,
    output_filename=output_fname)

# The screen will print: "Successfully generated ChatGPT4Critic_response_to_ch30.md."

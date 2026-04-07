"""
To run this script, you need the following API-token stored as environment variables:
    - ANTHROPIC_API_KEY
    - GEMINI_API_KEY
    - YOUR_GITHUB_PAT
"""

import os
import datetime
from llm_critic.llm_list import LargeLanguageModels
from llm_critic.critic import critic_factory
from llm_critic.prompt_template import *


folder_path: str = os.environ.get("FOLDER_PATH")  # Or put in the folder path with your writing in Markdown files
article_path: str = os.environ.get("ARTICLE_PATH")  # A Markdown file in jekyll format

# Choose your LLM critic
llm_model = LargeLanguageModels.ClaudeOpus4p6
critic = critic_factory(llm_model=llm_model, token_limit=10000)

# Markdown file to output response to
output_fname: str = f"{datetime.date.today().strftime("%Y%m%d")}_{llm_model.name}_response_to_{article_path}"

# Generate Critique and save them into a Markdown file
with open(folder_path+article_path, "r") as f:
    content = f.read()

critic.generate_critique(
    folder_path=folder_path,
    article_filename=article_path,
    task_for_llm=NOVEL_EDIT,
    output_filename=output_fname)

# The screen will print: "Successfully generated GerminiFlash_response_to_ch30.md."

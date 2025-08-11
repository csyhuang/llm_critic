from abc import ABC, abstractmethod
from typing import Optional
import os
import json
import pprint
import requests

from bs4 import BeautifulSoup
from markdown import markdown

from google import genai
import anthropic

from llm_critic.llm_list import LargeLanguageModels


class Critic(ABC):
    """
    Abstract base class for interacting with Large Language Models (LLMs) to edit and critique text.

    This class defines the basic interface for critics, including methods for querying LLMs,
    extracting content from Markdown files of Jekyll format, and compiling responses into a
    formatted output to be saved to a .md file.
    """

    def __init__(self):
        """
        Initializes a Critic instance.

        Sets the `llm_model` attribute to None by default.  Subclasses should set this
        to a specific `LargeLanguageModels` value in their own constructors.
        """
        self.llm_model: Optional[LargeLanguageModels] = None

    @abstractmethod
    def pass_query_to_llm_to_get_response(self, query) -> str:
        """
        Abstract method to send a query to the LLM and retrieve the response.

        Args:
            query (str): The query to be sent to the LLM.

        Returns:
            A string which is the response from the LLM.
        """
        pass

    @property
    def response_markdown_template(self) -> str:
        """
        Provides a default Markdown template for formatting LLM responses.

        This template includes placeholders for the article file name, the prompt used,
        the LLM model name, the response text, and any citation information.

        Returns:
            str: A Markdown template string.
        """
        response_template = \
            """RESPONSE TO: {article_file_input}
# PROMPT
{prompt_input}
# RESPONSE FROM {llm_model} 
{response_text}
# CITATION
{citation_text}
            """
        return response_template

    @staticmethod
    def extract_markdown_content(folder_path, article_filename, token_limit=10000) -> str:
        """
        Extracts and cleans the content from a Markdown file of Jekyll blog format.

        This method reads a Markdown file, removes the frontmatter (enclosed by `---`), converts
        the Markdown to HTML, and then extracts the plain text content.  It also limits the extracted
        content to a specified token limit.

        Args:
            folder_path (str): The path to the folder containing the Markdown file.
            article_filename (str): The name of the Markdown file.
            token_limit (int): The maximum number of characters to extract from the file.

        Returns:
            str: The extracted plain text content from the Markdown file, up to the token limit.
        """
        with open(folder_path + article_filename, "r") as introspection_file:
            article = introspection_file.read()
            main_arr = article.split('---')[2:]
            if len(main_arr) == 1:
                main_markdown_str = main_arr[0]
            else:
                main_markdown_str = '\n'.join(main_arr)
        html = markdown(main_markdown_str)
        content_str = BeautifulSoup(html, features="html.parser").text[:token_limit]
        return content_str

    @abstractmethod
    def compile_template(self, response, input_filepath, prompt_template) -> str:
        """
        Abstract method to compile the LLM response into a formatted output string of Markdown format.

        This method takes the LLM response, the input file path, and the prompt used to generate
        the response, and combines them into a formatted string using `response_markdown_template`.

        Args:
            response (Any): The response from the LLM. The specific type depends on the LLM.
            input_filepath (str): The path to the input file being edited.
            prompt_template (str): The prompt used to generate the LLM response.

        Returns:
            str: The compiled output string.
        """
        pass

    def generate_critique(self, folder_path: str, article_filename: str, task_for_llm: str, output_filename: str):
        """
        This method extracts the content from the article, formats it into a query for the LLM,
        sends the query to the LLM, compiles the response into a formatted output, and saves the
        output to a file.

        Args:
            folder_path (str): The path to the folder containing the article.
            article_filename (str): The name of the article file.
            task_for_llm (str): The task description for the LLM (e.g., "Critique this article for clarity").
                This string should be formatted to include the article content.  For example:
                `"Critique the following content for clarity and grammar:\n{content}"`
            output_filename (str): The name of the file to save the critique to.
        """
        article_content = Critic.extract_markdown_content(folder_path, article_filename)
        compiled_query: str = task_for_llm.format(content=article_content)
        response = self.pass_query_to_llm_to_get_response(compiled_query)
        filepath = folder_path + article_filename
        output_markdown_str = self.compile_template(response, filepath, task_for_llm)
        with open(output_filename, "w") as f:
            f.write(output_markdown_str)
        print(f"Successfully generated {output_filename}.")


class GitHubModelCritic(Critic):
    """
    Base class for critics that use models hosted on GitHub's AI Models service.

    This class handles the communication with the GitHub Models API, including authentication and
    request formatting.  It requires the `YOUR_GITHUB_PAT` environment variable to be set with a
    valid GitHub Personal Access Token.
    """

    def pass_query_to_llm_to_get_response(self, query: str):
        """
        Sends a query to a GitHub-hosted LLM and retrieves the response.

        This method retrieves the GitHub Personal Access Token from the environment variables,
        constructs the API request, sends the request to the GitHub Models API, and returns the
        LLM's response.

        Args:
            query (str): The query to send to the LLM.

        Returns:
            str: The LLM's response.

        Raises:
            Exception: If the API request returns a 400 status code, indicating an error.
        """
        github_pat = os.environ['YOUR_GITHUB_PAT']
        url = "https://models.github.ai/inference/chat/completions"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_pat}",
            "Content-Type": "application/json"}
        data = json.dumps({
            "model": self.llm_model.value,
            "messages": [{"role": "user",
                          "content": query}]})
        r = requests.post(url, headers=headers, data=data)
        if r.status_code == 400:
            raise Exception(f"Your request is rejected by GitHub Models with the following reason:\n" +
                            pprint.pformat(json.loads(r.text)))
        ans = json.loads(r.text)
        llm_response_message = ans['choices'][0]['message']['content']
        return llm_response_message

    def compile_template(self, response, input_filepath, prompt_template) -> str:
        output = self.response_markdown_template.format(
            article_file_input=input_filepath,
            prompt_input=prompt_template,
            llm_model=self.llm_model.value,
            response_text=response,
            citation_text="")
        return output


def critic_factory(llm_model: LargeLanguageModels) -> Critic:
    """
    Factory function to instantiate a Critic subclass based on the specified LLM model.

    This function returns an appropriate Critic instance for the given `llm_model`.
    If the model is supported by GitHub Models, it returns a GitHubModelCritic with the `llm_model` attribute
    set accordingly. If the model has its own API, it returns a custom Critic class (e.g., GerminiCritic,
    ClaudeSonnetCritic).

    Args:
        llm_model (LargeLanguageModels): The LLM model to use for the Critic.

    Returns:
        Critic: An instance of a Critic subclass configured for the specified LLM model.
    """
    mapping = {
        LargeLanguageModels.GerminiFlash: GerminiCritic,
        LargeLanguageModels.ClaudeSonnet4: ClaudeSonnetCritic}
    if llm_model in mapping:
        return mapping[llm_model]()
    new_critic = GitHubModelCritic()
    new_critic.llm_model = llm_model
    return new_critic


class DeepSeekR1Critic(GitHubModelCritic):
    """
    Critic that uses the DeepSeekR1 model hosted on GitHub.

    This class inherits from `GitHubModelCritic` and sets the `llm_model` attribute to
    `LargeLanguageModels.DeepSeekR1`.
    """
    def __init__(self):
        """
        Initializes a DeepSeekR1Critic instance.

        Sets the `llm_model` attribute to `LargeLanguageModels.DeepSeekR1`.
        """
        super().__init__()
        self.llm_model: Optional[LargeLanguageModels] = LargeLanguageModels.DeepSeekR1


class ChatGPT4Critic(GitHubModelCritic):
    """
    Critic that uses the ChatGPT4 model hosted on GitHub.

    This class inherits from `GitHubModelCritic` and sets the `llm_model` attribute to
    `LargeLanguageModels.ChatGPT4`.
    """
    def __init__(self):
        """
        Initializes a ChatGPT4Critic instance.

        Sets the `llm_model` attribute to `LargeLanguageModels.ChatGPT4`.
        """
        super().__init__()
        self.llm_model: Optional[LargeLanguageModels] = LargeLanguageModels.ChatGPT4


class GPT5Critic(GitHubModelCritic):
    """
    Critic that uses the GPT5 model hosted on GitHub.
    """
    def __init__(self):
        """
        Initializes a GPT5Critic instance.
        """
        super().__init__()
        self.llm_model: Optional[LargeLanguageModels] = LargeLanguageModels.GPT5


class ClaudeSonnetCritic(Critic):
    """
    Critic that uses the Claude Sonnet model via the Anthropic API.
    """

    def __init__(self):
        """
        Initializes a ClaudeSonnetCritic instance.

        Sets the `llm_model` attribute to `LargeLanguageModels.ClaudeSonnet4` and defines a
        default system prompt.
        """
        super().__init__()
        self.llm_model = LargeLanguageModels.ClaudeSonnet4
        self.system = "You are world-class editor. Response precisely."

    def pass_query_to_llm_to_get_response(self, query):
        """
        Sends a query to the Claude Sonnet model via the Anthropic API.

        Args:
            query (str): The query to send to the LLM.

        Returns:
            anthropic.types.Message: The response message from the Claude Sonnet model.
        """
        client = anthropic.Anthropic()
        max_tokens = len(query)
        message = client.messages.create(
            model=self.llm_model.value,
            max_tokens=max_tokens,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ]
        )
        return message

    def compile_template(self, response, input_filepath, prompt_template) -> str:
        output = self.response_markdown_template.format(
            article_file_input=input_filepath,
            prompt_input=prompt_template,
            llm_model=self.llm_model.value,
            response_text=response.content[0].text,
            citation_text=response.content[0].citations)
        return output


class GerminiCritic(Critic):
    """
    Critic that uses the GerminiFlash model via the Google AI Gemini API.
    """

    def __init__(self):
        """
        Initializes a GerminiCritic instance.

        Sets the `llm_model` attribute to `LargeLanguageModels.GerminiFlash`.
        """
        super().__init__()
        self.llm_model = LargeLanguageModels.GerminiFlash

    def pass_query_to_llm_to_get_response(self, query):
        """
        Sends a query to the GerminiFlash model via the Google AI Gemini API.

        Args:
            query (str): The query to send to the LLM.

        Returns:
            google.ai.generative_models.generation_models.GenerateContentResponse: The response from the GerminiFlash model.
        """
        client = genai.Client()
        response = client.models.generate_content(model=self.llm_model.value, contents=query)
        return response

    def compile_template(self, response, input_filepath, prompt_template) -> str:
        output = self.response_markdown_template.format(
            article_file_input=input_filepath,
            prompt_input=prompt_template,
            llm_model=self.llm_model.value,
            response_text=response.text,
            citation_text="")
        return output

"""LLM client for various model providers including Anthropic, OpenAI, and OpenRouter."""

import json
import os
import random
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from dataclasses_json import DataClassJsonMixin
import anthropic
import openai

# Try to import litellm, but don't fail if it's not available
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
from anthropic import (
    NOT_GIVEN as Anthropic_NOT_GIVEN,
)
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    InternalServerError as AnthropicInternalServerError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,  # pyright: ignore[reportPrivateImportUsage]
)
from anthropic.types import (
    TextBlock as AnthropicTextBlock,
    ThinkingBlock as AnthropicThinkingBlock,
    RedactedThinkingBlock as AnthropicRedactedThinkingBlock,
)
from anthropic.types import ToolParam as AnthropicToolParam
from anthropic.types import (
    ToolResultBlockParam as AnthropicToolResultBlockParam,
)
from anthropic.types import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from anthropic.types.message_create_params import (
    ToolChoiceToolChoiceAny,
    ToolChoiceToolChoiceAuto,
    ToolChoiceToolChoiceTool,
)

from openai import (
    APIConnectionError as OpenAI_APIConnectionError,
)
from openai import (
    InternalServerError as OpenAI_InternalServerError,
)
from openai import (
    RateLimitError as OpenAI_RateLimitError,
)
from openai._types import (
    NOT_GIVEN as OpenAI_NOT_GIVEN,  # pyright: ignore[reportPrivateImportUsage]
)

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ToolParam(DataClassJsonMixin):
    """Internal representation of LLM tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall(DataClassJsonMixin):
    """Internal representation of LLM-generated tool call."""

    tool_call_id: str
    tool_name: str
    tool_input: Any


@dataclass
class ToolResult(DataClassJsonMixin):
    """Internal representation of LLM tool result."""

    tool_call_id: str
    tool_name: str
    tool_output: Any


@dataclass
class ToolFormattedResult(DataClassJsonMixin):
    """Internal representation of formatted LLM tool result."""

    tool_call_id: str
    tool_name: str
    tool_output: str


@dataclass
class TextPrompt(DataClassJsonMixin):
    """Internal representation of user-generated text prompt."""

    text: str


@dataclass
class TextResult(DataClassJsonMixin):
    """Internal representation of LLM-generated text result."""

    text: str


AssistantContentBlock = (
    TextResult | ToolCall | AnthropicRedactedThinkingBlock | AnthropicThinkingBlock
)
UserContentBlock = TextPrompt | ToolFormattedResult
GeneralContentBlock = UserContentBlock | AssistantContentBlock
LLMMessages = list[list[GeneralContentBlock]]


class LLMClient:
    """A client for LLM APIs for the use in agents."""

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """
        raise NotImplementedError


def recursively_remove_invoke_tag(obj):
    """Recursively remove the </invoke> tag from a dictionary or list."""
    result_obj = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            result_obj[key] = recursively_remove_invoke_tag(value)
    elif isinstance(obj, list):
        result_obj = [recursively_remove_invoke_tag(item) for item in obj]
    elif isinstance(obj, str):
        if "</invoke>" in obj:
            result_obj = json.loads(obj.replace("</invoke>", ""))
        else:
            result_obj = obj
    else:
        result_obj = obj
    return result_obj


class AnthropicDirectClient(LLMClient):
    """Use Anthropic models via first party API."""

    def __init__(
        self,
        model_name="claude-3-7-sonnet-20250219",
        max_retries=2,
        use_caching=True,
        use_low_qos_server: bool = False,
        thinking_tokens: int = 0,
    ):
        """Initialize the Anthropic first party client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        # Disable retries since we are handling retries ourselves.
        self.client = anthropic.Anthropic(
            api_key=api_key, max_retries=1, timeout=60 * 5
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_caching = use_caching
        self.prompt_caching_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        self.thinking_tokens = thinking_tokens

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """

        # Turn GeneralContentBlock into Anthropic message format
        anthropic_messages = []
        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "assistant"
            message_content_list = []
            for message in message_list:
                # Check string type to avoid import issues particularly with reloads.
                if str(type(message)) == str(TextPrompt):
                    message = cast(TextPrompt, message)
                    message_content = AnthropicTextBlock(
                        type="text",
                        text=message.text,
                    )
                elif str(type(message)) == str(TextResult):
                    message = cast(TextResult, message)
                    message_content = AnthropicTextBlock(
                        type="text",
                        text=message.text,
                    )
                elif str(type(message)) == str(ToolCall):
                    message = cast(ToolCall, message)
                    message_content = AnthropicToolUseBlock(
                        type="tool_use",
                        id=message.tool_call_id,
                        name=message.tool_name,
                        input=message.tool_input,
                    )
                elif str(type(message)) == str(ToolFormattedResult):
                    message = cast(ToolFormattedResult, message)
                    message_content = AnthropicToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=message.tool_call_id,
                        content=message.tool_output,
                    )
                elif str(type(message)) == str(AnthropicRedactedThinkingBlock):
                    message = cast(AnthropicRedactedThinkingBlock, message)
                    message_content = message
                elif str(type(message)) == str(AnthropicThinkingBlock):
                    message = cast(AnthropicThinkingBlock, message)
                    message_content = message
                else:
                    print(
                        f"Unknown message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                    )
                    raise ValueError(
                        f"Unknown message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                    )
                message_content_list.append(message_content)

            # Anthropic supports up to 4 cache breakpoints, so we put them on the last 4 messages.
            if self.use_caching and idx >= len(messages) - 4:
                if isinstance(message_content_list[-1], dict):
                    message_content_list[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    message_content_list[-1].cache_control = {"type": "ephemeral"}

            anthropic_messages.append(
                {
                    "role": role,
                    "content": message_content_list,
                }
            )

        if self.use_caching:
            extra_headers = self.prompt_caching_headers
        else:
            extra_headers = None

        # Turn tool_choice into Anthropic tool_choice format
        if tool_choice is None:
            tool_choice_param = Anthropic_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = ToolChoiceToolChoiceAny(type="any")
        elif tool_choice["type"] == "auto":
            tool_choice_param = ToolChoiceToolChoiceAuto(type="auto")
        elif tool_choice["type"] == "tool":
            tool_choice_param = ToolChoiceToolChoiceTool(
                type="tool", name=tool_choice["name"]
            )
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice['type']}")

        if len(tools) == 0:
            tool_params = Anthropic_NOT_GIVEN
        else:
            tool_params = [
                AnthropicToolParam(
                    input_schema=tool.input_schema,
                    name=tool.name,
                    description=tool.description,
                )
                for tool in tools
            ]

        response = None

        if thinking_tokens is None:
            thinking_tokens = self.thinking_tokens
        if thinking_tokens and thinking_tokens > 0:
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_tokens}
            }
            temperature = 1
            assert max_tokens >= 32_000 and thinking_tokens <= 8192, (
                f"As a heuristic, max tokens {max_tokens} must be >= 32k and thinking tokens {thinking_tokens} must be < 8k"
            )
        else:
            extra_body = None

        for retry in range(self.max_retries):
            try:
                response = self.client.messages.create(  # type: ignore
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    model=self.model_name,
                    temperature=temperature,
                    system=system_prompt or Anthropic_NOT_GIVEN,
                    tool_choice=tool_choice_param,  # type: ignore
                    tools=tool_params,
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                )
                break
            except (
                AnthropicAPIConnectionError,
                AnthropicInternalServerError,
                AnthropicRateLimitError,
                AnthropicOverloadedError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed Anthropic request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying LLM request: {retry + 1}/{self.max_retries}")
                    # Sleep 4-6 seconds with jitter to avoid thundering herd.
                    time.sleep(5 * random.uniform(0.8, 1.2))

        # Convert messages back to Augment format
        augment_messages = []
        assert response is not None
        for message in response.content:
            if "</invoke>" in str(message):
                warning_msg = "\n".join(
                    ["!" * 80, "WARNING: Unexpected 'invoke' in message", "!" * 80]
                )
                print(warning_msg)

            if str(type(message)) == str(AnthropicTextBlock):
                message = cast(AnthropicTextBlock, message)
                augment_messages.append(TextResult(text=message.text))
            elif str(type(message)) == str(AnthropicRedactedThinkingBlock):
                augment_messages.append(message)
            elif str(type(message)) == str(AnthropicThinkingBlock):
                message = cast(AnthropicThinkingBlock, message)
                augment_messages.append(message)
            elif str(type(message)) == str(AnthropicToolUseBlock):
                message = cast(AnthropicToolUseBlock, message)
                augment_messages.append(
                    ToolCall(
                        tool_call_id=message.id,
                        tool_name=message.name,
                        tool_input=recursively_remove_invoke_tag(message.input),
                    )
                )
            else:
                raise ValueError(f"Unknown message type: {type(message)}")

        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", -1
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", -1
            ),
        }

        return augment_messages, message_metadata


class OpenAIDirectClient(LLMClient):
    """Use OpenAI models via first party API."""

    def __init__(self, model_name: str, max_retries=2, cot_model: bool = True):
        """Initialize the OpenAI first party client."""
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(
            api_key=api_key,
            max_retries=1,
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.cot_model = cot_model

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            system_prompt: A system prompt.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """
        assert thinking_tokens is None, "Not implemented for OpenAI"

        # Turn GeneralContentBlock into OpenAI message format
        openai_messages = []
        if system_prompt is not None:
            if self.cot_model:
                raise NotImplementedError("System prompt not supported for cot model")
            system_message = {"role": "system", "content": system_prompt}
            openai_messages.append(system_message)
        for idx, message_list in enumerate(messages):
            if len(message_list) > 1:
                raise ValueError("Only one entry per message supported for openai")
            augment_message = message_list[0]
            if str(type(augment_message)) == str(TextPrompt):
                augment_message = cast(TextPrompt, augment_message)
                message_content = {"type": "text", "text": augment_message.text}
                openai_message = {"role": "user", "content": [message_content]}
            elif str(type(augment_message)) == str(TextResult):
                augment_message = cast(TextResult, augment_message)
                message_content = {"type": "text", "text": augment_message.text}
                openai_message = {"role": "assistant", "content": [message_content]}
            elif str(type(augment_message)) == str(ToolCall):
                augment_message = cast(ToolCall, augment_message)
                tool_call = {
                    "type": "function",
                    "id": augment_message.tool_call_id,
                    "function": {
                        "name": augment_message.tool_name,
                        "arguments": augment_message.tool_input,
                    },
                }
                openai_message = {
                    "role": "assistant",
                    "tool_calls": [tool_call],
                }
            elif str(type(augment_message)) == str(ToolFormattedResult):
                augment_message = cast(ToolFormattedResult, augment_message)
                openai_message = {
                    "role": "tool",
                    "tool_call_id": augment_message.tool_call_id,
                    "content": augment_message.tool_output,
                }
            else:
                print(
                    f"Unknown message type: {type(augment_message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                )
                raise ValueError(f"Unknown message type: {type(augment_message)}")
            openai_messages.append(openai_message)

        # Turn tool_choice into OpenAI tool_choice format
        if tool_choice is None:
            tool_choice_param = OpenAI_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = "required"
        elif tool_choice["type"] == "auto":
            tool_choice_param = "auto"
        elif tool_choice["type"] == "tool":
            tool_choice_param = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice['type']}")

        # Turn tools into OpenAI tool format
        openai_tools = []
        for tool in tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            tool_def["parameters"]["strict"] = True
            openai_tool_object = {
                "type": "function",
                "function": tool_def,
            }
            openai_tools.append(openai_tool_object)

        response = None
        for retry in range(self.max_retries):
            try:
                extra_body = {}
                openai_max_tokens = max_tokens
                openai_temperature = temperature
                if self.cot_model:
                    extra_body["max_completion_tokens"] = max_tokens
                    openai_max_tokens = OpenAI_NOT_GIVEN
                    openai_temperature = OpenAI_NOT_GIVEN

                response = self.client.chat.completions.create(  # type: ignore
                    model=self.model_name,
                    messages=openai_messages,
                    temperature=openai_temperature,
                    tools=openai_tools if len(openai_tools) > 0 else OpenAI_NOT_GIVEN,
                    tool_choice=tool_choice_param,  # type: ignore
                    max_tokens=openai_max_tokens,
                    extra_body=extra_body,
                )
                break
            except (
                OpenAI_APIConnectionError,
                OpenAI_InternalServerError,
                OpenAI_RateLimitError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed OpenAI request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying OpenAI request: {retry + 1}/{self.max_retries}")
                    # Sleep 8-12 seconds with jitter to avoid thundering herd.
                    time.sleep(10 * random.uniform(0.8, 1.2))

        # Convert messages back to Augment format
        augment_messages = []
        assert response is not None
        openai_response_messages = response.choices
        if len(openai_response_messages) > 1:
            raise ValueError("Only one message supported for OpenAI")
        openai_response_message = openai_response_messages[0].message
        tool_calls = openai_response_message.tool_calls
        content = openai_response_message.content

        # Exactly one of tool_calls or content should be present
        if tool_calls and content:
            raise ValueError("Only one of tool_calls or content should be present")
        elif not tool_calls and not content:
            raise ValueError("Either tool_calls or content should be present")

        if tool_calls:
            if len(tool_calls) > 1:
                raise ValueError("Only one tool call supported for OpenAI")
            tool_call = tool_calls[0]
            try:
                # Parse the JSON string into a dictionary
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool arguments: {tool_call.function.arguments}")
                print(f"JSON parse error: {str(e)}")
                raise ValueError(f"Invalid JSON in tool arguments: {str(e)}") from e

            augment_messages.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    tool_input=tool_input,
                    tool_call_id=tool_call.id,
                )
            )
        elif content:
            augment_messages.append(TextResult(text=content))
        else:
            raise ValueError(f"Unknown message type: {openai_response_message}")

        assert response.usage is not None
        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        return augment_messages, message_metadata


class OpenRouterClient(LLMClient):
    """Use models via OpenRouter directly using the OpenAI client."""

    def __init__(
        self,
        model_name: str,
        max_retries: int = 2,
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://openrouter.ai/api/v1",
    ):
        """Initialize the OpenRouter client.

        Args:
            model_name: The model name to use.
            max_retries: The maximum number of retries.
            api_key: The API key to use. If None, will be read from environment.
            api_base: The API base URL. If None, will use the default.
        """
        # Set up API key from environment if not provided
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        # Initialize OpenAI client with OpenRouter base URL
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            max_retries=1,
        )

        # Store the model name
        self.model_name = model_name
        self.max_retries = max_retries

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses using OpenRouter.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.
            thinking_tokens: Number of tokens for thinking (not used).

        Returns:
            A generated response.
        """
        assert thinking_tokens is None, "Not implemented for OpenRouter"

        # Convert messages to OpenAI format
        openai_messages = []
        if system_prompt is not None:
            openai_messages.append({"role": "system", "content": system_prompt})

        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "assistant"

            # Handle multiple content blocks in a message
            if len(message_list) == 1:
                # Simple case: single content block
                message = message_list[0]

                if str(type(message)) == str(TextPrompt):
                    message = cast(TextPrompt, message)
                    openai_messages.append({"role": role, "content": message.text})
                elif str(type(message)) == str(TextResult):
                    message = cast(TextResult, message)
                    openai_messages.append({"role": role, "content": message.text})
                elif str(type(message)) == str(ToolCall):
                    message = cast(ToolCall, message)
                    tool_call = {
                        "id": message.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": message.tool_name,
                            "arguments": json.dumps(message.tool_input),
                        },
                    }
                    openai_messages.append({"role": role, "tool_calls": [tool_call]})
                elif str(type(message)) == str(ToolFormattedResult):
                    message = cast(ToolFormattedResult, message)
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.tool_output,
                    })
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
            else:
                # Complex case: multiple content blocks
                # For now, concatenate text blocks and ignore tool calls
                content = ""
                for message in message_list:
                    if str(type(message)) == str(TextPrompt) or str(type(message)) == str(TextResult):
                        if str(type(message)) == str(TextPrompt):
                            message = cast(TextPrompt, message)
                        else:
                            message = cast(TextResult, message)
                        content += message.text + "\n"

                if content:
                    openai_messages.append({"role": role, "content": content.strip()})

        # Convert tools to OpenAI format
        openai_tools = []
        if tools:
            for tool in tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                }
                openai_tools.append(tool_def)

        # Convert tool_choice to OpenAI format
        openai_tool_choice = None
        if tool_choice:
            if tool_choice["type"] == "any":
                openai_tool_choice = "required"
            elif tool_choice["type"] == "auto":
                openai_tool_choice = "auto"
            elif tool_choice["type"] == "tool":
                openai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }

        # Make the API call with retries
        response = None
        for retry in range(self.max_retries):
            try:
                # Add OpenRouter specific headers
                extra_headers = {
                    "HTTP-Referer": "https://augment.dev",  # Optional
                    "X-Title": "Augment SWE-bench Agent",   # Optional
                }

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=openai_tools if openai_tools else None,
                    tool_choice=openai_tool_choice,
                    extra_headers=extra_headers,
                )
                break
            except Exception as e:
                if retry == self.max_retries - 1:
                    print(f"Failed OpenRouter request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying OpenRouter request: {retry + 1}/{self.max_retries}")
                    # Sleep with jitter to avoid thundering herd
                    time.sleep(5 * random.uniform(0.8, 1.2))

        # Convert response back to Augment format
        augment_messages = []
        assert response is not None

        # Extract the message content
        message = response.choices[0].message

        # Handle tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    # Parse the JSON string into a dictionary
                    tool_input = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse tool arguments: {tool_call.function.arguments}")
                    raise ValueError(f"Invalid JSON in tool arguments: {str(e)}") from e

                augment_messages.append(
                    ToolCall(
                        tool_name=tool_call.function.name,
                        tool_input=tool_input,
                        tool_call_id=tool_call.id,
                    )
                )
        # Handle text content
        elif message.content:
            augment_messages.append(TextResult(text=message.content))
        else:
            raise ValueError(f"Unknown message format: {message}")

        # Prepare metadata
        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        return augment_messages, message_metadata


def load_model_config():
    """Load model configuration from the config file."""
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    if not config_path.exists():
        return None

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_client(client_name: str, **kwargs) -> LLMClient:
    """Get a client for a given client name.

    Args:
        client_name: The client name to use. Can be one of:
            - "anthropic-direct": Use Anthropic models directly
            - "openai-direct": Use OpenAI models directly
            - "litellm": Use LiteLLM for any provider
            - "openrouter": Use OpenRouter via LiteLLM (shorthand for litellm with provider=openrouter)
            - A model name from the config file
        **kwargs: Additional arguments to pass to the client.

    Returns:
        An LLM client instance.
    """
    # Load model configuration
    model_config = load_model_config()

    # Handle direct client requests
    if client_name == "anthropic-direct":
        return AnthropicDirectClient(**kwargs)
    elif client_name == "openai-direct":
        return OpenAIDirectClient(**kwargs)
    elif client_name == "openrouter":
        try:
            # Extract only the parameters that OpenRouterClient accepts
            valid_params = {
                'model_name': kwargs.get('model_name', 'deepseek/deepseek-chat-v3-0324'),
                'max_retries': kwargs.get('max_retries', 2),
            }

            # Add any remaining kwargs that don't conflict
            for k, v in kwargs.items():
                if k not in ['max_tokens', 'thinking_tokens', 'use_caching'] and k not in valid_params:
                    valid_params[k] = v

            return OpenRouterClient(**valid_params)
        except Exception as e:
            print(f"Warning: Failed to initialize OpenRouter client: {str(e)}. Falling back to Anthropic direct client.")
            return AnthropicDirectClient(**kwargs)

    # Check if we have a model config and try to find the model
    if model_config:
        # Check if this is a purpose-specific model request
        if client_name.startswith("purpose:"):
            purpose = client_name.split(":", 1)[1]
            if "purpose_models" in model_config and purpose in model_config["purpose_models"]:
                model_name = model_config["purpose_models"][purpose]
                return get_client(model_name, **kwargs)

        # Check direct providers
        if "providers" in model_config and "direct" in model_config["providers"]:
            # Check Anthropic models
            if "anthropic" in model_config["providers"]["direct"] and model_config["providers"]["direct"]["anthropic"]["enabled"]:
                for model in model_config["providers"]["direct"]["anthropic"]["models"]:
                    if model["name"] == client_name:
                        # Extract parameters that might be duplicated
                        use_caching = kwargs.pop('use_caching', True) if 'use_caching' in kwargs else True
                        model_params = {k: v for k, v in model.items() if k not in ["name", "model_name", "description"]}

                        # Remove parameters that AnthropicDirectClient doesn't accept
                        for param in ['max_tokens', 'use_caching']:
                            if param in model_params:
                                model_params.pop(param)

                        # Only pass parameters that AnthropicDirectClient accepts
                        valid_params = {
                            'model_name': model["model_name"],
                            'max_retries': model_params.get('max_retries', 2),
                            'use_caching': use_caching,
                            'thinking_tokens': model_params.get('thinking_tokens', 0)
                        }

                        # Add any remaining kwargs that don't conflict
                        for k, v in kwargs.items():
                            if k not in valid_params:
                                valid_params[k] = v

                        return AnthropicDirectClient(**valid_params)

            # Check OpenAI models
            if "openai" in model_config["providers"]["direct"] and model_config["providers"]["direct"]["openai"]["enabled"]:
                for model in model_config["providers"]["direct"]["openai"]["models"]:
                    if model["name"] == client_name:
                        # Extract parameters that might be duplicated
                        model_params = {k: v for k, v in model.items() if k not in ["name", "model_name", "description"]}

                        # Handle potential duplicate parameters
                        for param in ['cot_model']:
                            if param in model_params and param in kwargs:
                                model_params.pop(param)

                        return OpenAIDirectClient(
                            model_name=model["model_name"],
                            **model_params,
                            **kwargs
                        )

        # Check OpenRouter models
        if "providers" in model_config and "openrouter" in model_config["providers"] and model_config["providers"]["openrouter"]["enabled"]:
            for model in model_config["providers"]["openrouter"]["models"]:
                if model["name"] == client_name:
                    try:
                        # Extract only the parameters that OpenRouterClient accepts
                        valid_params = {
                            'model_name': model["model_name"],
                            'max_retries': kwargs.get('max_retries', 2),
                        }

                        # Add any remaining kwargs that don't conflict
                        for k, v in kwargs.items():
                            if k not in ['max_tokens', 'thinking_tokens', 'use_caching'] and k not in valid_params:
                                valid_params[k] = v

                        return OpenRouterClient(**valid_params)
                    except Exception as e:
                        print(f"Warning: Failed to initialize OpenRouter client for model '{client_name}': {str(e)}.")
                        print(f"Falling back to Anthropic direct client.")
                        return AnthropicDirectClient(**kwargs)

    # If we get here, we couldn't find the client
    print(f"Warning: Unknown client name: {client_name}. Falling back to Anthropic direct client.")

    # Extract parameters that AnthropicDirectClient accepts
    use_caching = kwargs.pop('use_caching', True) if 'use_caching' in kwargs else True

    # Only pass parameters that AnthropicDirectClient accepts
    valid_params = {
        'use_caching': use_caching,
    }

    # Add model_name if provided
    if 'model_name' in kwargs:
        valid_params['model_name'] = kwargs.pop('model_name')

    # Add any remaining kwargs that don't conflict with AnthropicDirectClient parameters
    for k, v in kwargs.items():
        if k not in ['max_tokens']:  # Skip parameters that AnthropicDirectClient doesn't accept
            valid_params[k] = v

    return AnthropicDirectClient(**valid_params)

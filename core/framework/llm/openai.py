"""OpenAI LLM provider - backward compatible wrapper around LiteLLM."""

import os
from collections.abc import Callable
from typing import Any

from framework.llm.litellm import LiteLLMProvider
from framework.llm.provider import LLMProvider, LLMResponse, Tool, ToolResult, ToolUse


def _get_api_key_from_credential_store() -> str | None:
    """Get API key from CredentialStoreAdapter or environment.

    Priority:
    1. CredentialStoreAdapter (supports encrypted storage + env vars)
    2. os.environ fallback
    """
    try:
        from aden_tools.credentials import CredentialStoreAdapter

        creds = CredentialStoreAdapter.default()
        if creds.is_available("openai"):
            return creds.get("openai")
    except ImportError:
        pass
    return os.environ.get("OPENAI_API_KEY")


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider.

    This is a backward-compatible wrapper that internally uses LiteLLMProvider.
    Existing code using OpenAIProvider will continue to work unchanged,
    while benefiting from LiteLLM's unified interface and features.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, uses CredentialStoreAdapter
                     or OPENAI_API_KEY env var.
            model: Model to use (default: gpt-4o-mini)
        """
        # Delegate to LiteLLMProvider internally.
        self.api_key = api_key or _get_api_key_from_credential_store()
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.model = model

        self._provider = LiteLLMProvider(
            model=model,
            api_key=self.api_key,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[Tool] | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        json_mode: bool = False,
        max_retries: int | None = None,
    ) -> LLMResponse:
        """Generate a completion from OpenAI (via LiteLLM)."""
        return self._provider.complete(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            response_format=response_format,
            json_mode=json_mode,
            max_retries=max_retries,
        )

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[Tool],
        tool_executor: Callable[[ToolUse], ToolResult],
        max_iterations: int = 10,
    ) -> LLMResponse:
        """Run a tool-use loop until OpenAI produces a final response (via LiteLLM)."""
        return self._provider.complete_with_tools(
            messages=messages,
            system=system,
            tools=tools,
            tool_executor=tool_executor,
            max_iterations=max_iterations,
        )

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[Tool] | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        json_mode: bool = False,
        max_retries: int | None = None,
    ) -> LLMResponse:
        """Async completion via LiteLLM."""
        return await self._provider.acomplete(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            response_format=response_format,
            json_mode=json_mode,
            max_retries=max_retries,
        )

    async def acomplete_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[Tool],
        tool_executor: Callable[[ToolUse], ToolResult],
        max_iterations: int = 10,
    ) -> LLMResponse:
        """Async tool-use loop via LiteLLM."""
        return await self._provider.acomplete_with_tools(
            messages=messages,
            system=system,
            tools=tools,
            tool_executor=tool_executor,
            max_iterations=max_iterations,
        )

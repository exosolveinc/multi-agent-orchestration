"""
LLM Provider abstraction supporting Claude (Anthropic) and OpenAI.

Usage:
    from core.llm import create_llm_client, LLMProvider

    # Use Claude (default/recommended)
    client = create_llm_client(LLMProvider.ANTHROPIC, api_key="...")
    
    # Or use OpenAI
    client = create_llm_client(LLMProvider.OPENAI, api_key="...")
"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"  # Default to Claude
    max_tokens: int = 4096
    temperature: float = 0.7


class AnthropicLLMClient:
    """Wrapper for Anthropic's Claude API with OpenAI-compatible interface."""
    
    def __init__(self, api_key: str, default_model: str = "claude-sonnet-4-20250514", base_url: str = None):
        from anthropic import AsyncAnthropic
        
        # Use base_url if provided, otherwise let Anthropic use its default or env var
        if base_url:
            self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)
            print(f"🔗 Anthropic client using base URL: {base_url}")
        else:
            self.client = AsyncAnthropic(api_key=api_key)
            # Log what base URL is being used
            actual_base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
            print(f"🔗 Anthropic client using base URL: {actual_base}")
        
        self.default_model = default_model
        self.chat = self  # For compatibility with OpenAI interface
        self.completions = self
    
    async def create(
        self,
        model: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,  # Ignored for Claude, but accepted for compatibility
        **kwargs
    ) -> Any:
        """Create a chat completion using Claude."""
        import json
        model = model or self.default_model
        # Note: tool_choice is ignored - Claude handles this differently
        
        # Extract system message and convert messages to Claude format
        system_content = ""
        chat_messages = []
        
        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_content += content + "\n"
            elif role == "tool":
                # Convert OpenAI tool response to Claude format
                # Claude expects tool results as user messages with tool_result content blocks
                tool_call_id = msg.get("tool_call_id", "unknown")
                chat_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content if isinstance(content, str) else json.dumps(content)
                        }
                    ]
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert OpenAI assistant message with tool_calls to Claude format
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in msg["tool_calls"]:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                    })
                chat_messages.append({
                    "role": "assistant",
                    "content": content_blocks
                })
            else:
                chat_messages.append({
                    "role": role,
                    "content": content
                })
        
        # Build request
        request_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
            "temperature": temperature,
        }
        
        if system_content:
            request_kwargs["system"] = system_content.strip()
        
        # Convert tools to Claude format
        if tools:
            claude_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    claude_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                    })
            if claude_tools:
                request_kwargs["tools"] = claude_tools
        
        # Make request
        response = await self.client.messages.create(**request_kwargs)
        
        # Convert to OpenAI-compatible format
        return self._convert_response(response)
    
    def _convert_response(self, response: Any) -> Any:
        """Convert Anthropic response to OpenAI-compatible format."""
        # Extract text content
        text_content = ""
        tool_calls = []
        
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text
            elif hasattr(block, 'type') and block.type == 'tool_use':
                tool_calls.append(
                    type('ToolCall', (), {
                        'id': block.id,
                        'function': type('Function', (), {
                            'name': block.name,
                            'arguments': __import__('json').dumps(block.input)
                        })()
                    })()
                )
        
        # Create OpenAI-compatible response structure
        message = type('Message', (), {
            'content': text_content,
            'tool_calls': tool_calls if tool_calls else None,
            'role': 'assistant'
        })()
        
        choice = type('Choice', (), {
            'message': message,
            'finish_reason': response.stop_reason
        })()
        
        return type('Response', (), {
            'choices': [choice],
            'model': response.model,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
        })()


class OpenAILLMClient:
    """Wrapper for OpenAI API."""
    
    def __init__(self, api_key: str, default_model: str = "gpt-4-turbo-preview"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model
        self.chat = self.client.chat
        self.completions = self.client.chat.completions


def create_llm_client(
    provider: LLMProvider = None,
    api_key: str = None,
    model: str = None
) -> Any:
    """
    Create an LLM client based on provider.
    
    Args:
        provider: LLM provider (anthropic or openai). Auto-detects if not specified.
        api_key: API key. Uses environment variable if not specified.
        model: Model to use. Uses provider default if not specified.
    
    Returns:
        LLM client with OpenAI-compatible interface
    """
    # Auto-detect provider based on available API keys
    if provider is None:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = LLMProvider.ANTHROPIC
        elif os.getenv("OPENAI_API_KEY"):
            provider = LLMProvider.OPENAI
        else:
            raise ValueError("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    if provider == LLMProvider.ANTHROPIC:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        default_model = model or "claude-sonnet-4-20250514"
        # Always use official Anthropic API (ignore global env that might point to proxies)
        base_url = "https://api.anthropic.com"
        return AnthropicLLMClient(api_key=api_key, default_model=default_model, base_url=base_url)
    
    elif provider == LLMProvider.OPENAI:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        default_model = model or "gpt-4-turbo-preview"
        return OpenAILLMClient(api_key=api_key, default_model=default_model)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_default_model(provider: LLMProvider) -> str:
    """Get the default model for a provider."""
    defaults = {
        LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
        LLMProvider.OPENAI: "gpt-4-turbo-preview"
    }
    return defaults.get(provider, "claude-sonnet-4-20250514")


"""
Configuration for the Research Orchestrator.

Environment Variables:
    ANTHROPIC_API_KEY   - Primary: Your Anthropic/Claude API key
    OPENAI_API_KEY      - Fallback: Your OpenAI API key (if no Anthropic key)
    TAVILY_API_KEY      - Optional: Tavily API key for web search
    LLM_MODEL           - Optional: LLM model (default: claude-sonnet-4-20250514)
    LLM_PROVIDER        - Optional: LLM provider (default: anthropic)

Create a .env file in this directory with:

    ANTHROPIC_API_KEY=sk-ant-your-key-here
    TAVILY_API_KEY=tvly-your-key-here
    LLM_MODEL=claude-sonnet-4-20250514
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""
    
    # LLM Settings (Claude/Anthropic is primary)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None  # Fallback
    tavily_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-20250514"
    llm_provider: str = "anthropic"
    
    # Agent Settings
    max_retries: int = 3
    timeout_seconds: int = 120
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        # Auto-detect provider based on available keys
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if anthropic_key:
            provider = "anthropic"
            default_model = "claude-sonnet-4-20250514"
        elif openai_key:
            provider = "openai"
            default_model = "gpt-4-turbo-preview"
        else:
            provider = "anthropic"
            default_model = "claude-sonnet-4-20250514"
        
        return cls(
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", default_model),
            llm_provider=os.getenv("LLM_PROVIDER", provider),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "120")),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
        )
    
    def validate(self) -> bool:
        """Check if required configuration is present."""
        return bool(self.anthropic_api_key or self.openai_api_key)
    
    def get_api_key(self) -> Optional[str]:
        """Get the appropriate API key based on provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key


# Global config instance
config = Config.from_env()


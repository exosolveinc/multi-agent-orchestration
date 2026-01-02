from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import uuid

from .types import AgentRole, AgentResponse, ConversationMessage


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str
    role: AgentRole
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
    retry_attempts: int = 3
    timeout_seconds: int = 60


class Agent(ABC):
    """Base class for all agents in the research orchestration system."""
    
    def __init__(self, config: AgentConfig, llm_client: Any):
        self.id = f"{config.role.value}-{str(uuid.uuid4())[:8]}"
        self.config = config
        self.llm_client = llm_client
        self.conversation_history: List[ConversationMessage] = []
        
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def role(self) -> AgentRole:
        return self.config.role
    
    @property
    def description(self) -> str:
        return self.config.description
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        if self.config.system_prompt:
            return self.config.system_prompt
        return self._default_system_prompt()
    
    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return the default system prompt for this agent type."""
        pass
    
    @abstractmethod
    async def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process input and return a response."""
        pass
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Make a call to the LLM."""
        start_time = time.time()
        
        try:
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await self.llm_client.chat.completions.create(**kwargs)
            
            # Handle tool calls if present
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return await self._handle_tool_calls(message, messages, tools)
            
            return message.content
            
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}")
    
    async def _handle_tool_calls(
        self,
        message: Any,
        messages: List[Dict[str, str]],
        tools: List[Dict]
    ) -> str:
        """Handle tool calls from the LLM. Override in subclasses."""
        # Default implementation just returns the content
        return message.content or ""
    
    def add_to_history(self, message: ConversationMessage) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append(message)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """Convert conversation history to LLM message format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]


from typing import Any, Optional, Callable, Dict, List
from dataclasses import dataclass
import inspect
import json


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }


class Tool:
    """A tool that can be used by agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or self._extract_parameters(func)
    
    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameters from function signature."""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Get type hint
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                param_type = type_map.get(param.annotation, "string")
            
            properties[param_name] = {
                "type": param_type,
                "description": f"The {param_name} parameter"
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        try:
            result = self.func(**kwargs)
            if inspect.iscoroutine(result):
                result = await result
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e)
            )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """Registry for managing tools available to agents."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        return await tool.execute(**kwargs)


import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from core.base_agent import Agent, AgentConfig
from core.types import AgentRole, AgentResponse, Citation
from core.tools import Tool, ToolRegistry


@dataclass
class ResearchAgentConfig(AgentConfig):
    """Configuration specific to the Research Agent."""
    tavily_api_key: Optional[str] = None
    max_search_results: int = 5
    search_depth: str = "advanced"  # "basic" or "advanced"


class ResearchAgent(Agent):
    """
    Research Agent: Performs web searches, gathers sources, and validates information.
    
    Capabilities:
    - Web search via Tavily API
    - Source gathering and validation
    - Document retrieval
    - Citation extraction
    """
    
    def __init__(self, config: ResearchAgentConfig, llm_client: Any, tavily_client: Optional[Any] = None):
        # Ensure role is set correctly
        config.role = AgentRole.RESEARCH
        super().__init__(config, llm_client)
        
        self.tavily_client = tavily_client
        self.max_search_results = config.max_search_results
        self.search_depth = config.search_depth
        
        # Set up tools
        self.tool_registry = ToolRegistry()
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register available tools for this agent."""
        self.tool_registry.register(Tool(
            name="web_search",
            description="Search the web for information on a topic. Returns relevant results with titles, URLs, and content snippets.",
            func=self._web_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        ))
    
    def _default_system_prompt(self) -> str:
        return """You are a Research Agent that synthesizes information EXCLUSIVELY from provided web search results.

CRITICAL RULES:
1. You MUST ONLY use information from the search results provided to you
2. DO NOT use your training data or prior knowledge
3. If the search results don't contain relevant information, say so explicitly
4. Every claim MUST be attributable to a specific source from the results
5. Quote or closely paraphrase the actual content from the sources

Your task:
- Extract key facts, data, and insights ONLY from the provided search results
- Identify which sources support which claims
- Note any conflicting information between sources
- Be transparent when search results are limited or unclear

Output Format:
Return a JSON object with:
{
    "sources": [{"title": "exact title from results", "url": "url", "snippet": "relevant quote", "relevance": 0.0-1.0}],
    "key_findings": ["finding from source X", "finding from source Y", ...],
    "raw_content": "Synthesized content citing specific sources...",
    "confidence": 0.0-1.0,
    "notes": "Observations about the quality/limitations of results"
}

Remember: If information is not in the search results, DO NOT include it."""
    
    async def _web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform a web search using Tavily API."""
        if not self.tavily_client:
            # Return mock results for demo/testing
            return self._mock_search_results(query)
        
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=max_results
            )
            
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0)
                })
            
            return {
                "query": query,
                "results": results,
                "answer": response.get("answer", "")
            }
            
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "error": str(e)
            }
    
    def _mock_search_results(self, query: str) -> Dict[str, Any]:
        """Generate mock search results for demo purposes."""
        return {
            "query": query,
            "results": [
                {
                    "title": f"Research on: {query}",
                    "url": "https://example.com/research",
                    "content": f"Comprehensive analysis of {query}. This source provides detailed information and data points relevant to the research topic.",
                    "score": 0.95
                },
                {
                    "title": f"Expert Analysis: {query}",
                    "url": "https://academic.example.edu/paper",
                    "content": f"Academic perspective on {query}. Peer-reviewed research with statistical analysis and methodology.",
                    "score": 0.88
                },
                {
                    "title": f"Latest News: {query}",
                    "url": "https://news.example.com/article",
                    "content": f"Recent developments regarding {query}. Updated information from reliable news sources.",
                    "score": 0.82
                }
            ],
            "answer": f"Based on current research, {query} is a topic with multiple perspectives and ongoing developments."
        }
    
    async def _handle_tool_calls(
        self,
        message: Any,
        messages: List[Dict[str, str]],
        tools: List[Dict]
    ) -> str:
        """Handle tool calls from the LLM."""
        tool_calls = message.tool_calls
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            result = await self.tool_registry.execute(tool_name, **tool_args)
            tool_results.append({
                "tool_call_id": tool_call.id,
                "result": result.result if result.success else result.error
            })
        
        # Add tool results to messages and get final response
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]
        })
        
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": json.dumps(tr["result"])
            })
        
        # Get final response
        response = await self.llm_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a research request.
        
        Args:
            input_data: Should contain 'query' key with the research topic
            context: Optional context from other agents or previous steps
        
        Returns:
            AgentResponse with research findings
        """
        start_time = time.time()
        query = input_data.get("query", "")
        
        if not query:
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content="",
                success=False,
                error="No query provided for research"
            )
        
        try:
            # Step 1: Perform web search directly
            search_results = await self._web_search(query, self.max_search_results)
            
            # Step 2: Build messages with search results for LLM to synthesize
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
            ]
            
            # Add context if available
            if context:
                context_str = f"Previous context:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "system", "content": context_str})
            
            # Add conversation history
            messages.extend(self.get_history_for_llm())
            
            # Format search results for the LLM
            search_context = json.dumps(search_results, indent=2)
            
            # Add the research request with search results - VERY EXPLICIT about using only these results
            messages.append({
                "role": "user",
                "content": f"""Research topic: {query}

IMPORTANT: You must ONLY use the information from these search results. Do NOT use your training data.

=== WEB SEARCH RESULTS (use ONLY these) ===
{search_context}
=== END OF SEARCH RESULTS ===

Instructions:
1. Read through each search result carefully
2. Extract relevant facts, quotes, and data points from the actual content
3. Cite which source each piece of information comes from
4. If the results don't adequately answer the query, state that limitation
5. Return your synthesis as JSON following the output format

DO NOT include any information that is not present in the search results above."""
            })
            
            # Call LLM to synthesize (no tools needed)
            response_content = await self._call_llm(messages)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Parse response to extract citations
            metadata = self._extract_metadata(response_content)
            
            # Also add sources from search results to metadata
            if "citations" not in metadata or not metadata["citations"]:
                metadata["citations"] = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", r.get("snippet", "")),
                        "relevance_score": r.get("score", r.get("relevance", 0.8))
                    }
                    for r in search_results.get("results", [])
                ]
            
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content=response_content,
                success=True,
                metadata=metadata,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content="",
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract structured metadata from the response."""
        try:
            # Try to parse as JSON
            data = json.loads(content)
            citations = []
            for source in data.get("sources", []):
                citations.append(Citation(
                    title=source.get("title", ""),
                    url=source.get("url", ""),
                    snippet=source.get("snippet", ""),
                    relevance_score=source.get("relevance", 0.0)
                ))
            
            return {
                "citations": [c.to_dict() for c in citations],
                "key_findings": data.get("key_findings", []),
                "confidence": data.get("confidence", 0.0),
                "search_queries": data.get("search_queries_used", [])
            }
        except json.JSONDecodeError:
            return {"raw_response": True}


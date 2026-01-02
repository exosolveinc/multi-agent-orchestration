import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from core.base_agent import Agent, AgentConfig
from core.types import AgentRole, AgentResponse


@dataclass 
class AnalysisAgentConfig(AgentConfig):
    """Configuration specific to the Analysis Agent."""
    summarization_style: str = "comprehensive"  # "brief", "comprehensive", "bullet_points"
    extract_patterns: bool = True
    max_key_points: int = 10


class AnalysisAgent(Agent):
    """
    Analysis Agent: Summarizes content, extracts key points, and identifies patterns.
    
    Capabilities:
    - Content summarization
    - Key point extraction
    - Pattern identification across sources
    - Theme and trend analysis
    - Structured insight generation
    """
    
    def __init__(self, config: AnalysisAgentConfig, llm_client: Any):
        config.role = AgentRole.ANALYSIS
        super().__init__(config, llm_client)
        
        self.summarization_style = config.summarization_style
        self.extract_patterns = config.extract_patterns
        self.max_key_points = config.max_key_points
    
    def _default_system_prompt(self) -> str:
        return """You are an Analysis Agent that synthesizes ONLY the research content provided to you.

CRITICAL RULES:
1. Analyze ONLY the research content given to you - do NOT add external knowledge
2. Every insight must be derived from the provided research findings
3. If patterns exist in the data, identify them from the actual content
4. Do NOT supplement with your training data

Your responsibilities:
1. SUMMARIZE: Create summaries based ONLY on the provided research
2. EXTRACT: Pull out key points that are ACTUALLY in the research
3. PATTERN: Find patterns WITHIN the provided sources
4. STRUCTURE: Organize the provided findings logically

Guidelines:
- Quote or closely paraphrase from the research content
- Attribute insights to their sources when possible
- If the research is limited, note that explicitly
- Do NOT fill gaps with your own knowledge

Output Format:
Return a JSON object with:
{
    "summary": "Summary based ONLY on the provided research...",
    "key_points": [
        "Point directly from the research with source attribution",
        ...
    ],
    "patterns": [
        {
            "pattern": "Pattern observed in the sources",
            "evidence": ["Quote/fact from source 1", "Quote/fact from source 2"],
            "significance": "Why this matters based on the research"
        }
    ],
    "themes": ["Theme found in sources", ...],
    "data_points": [
        {"metric": "...", "value": "from source", "source": "source name"}
    ],
    "gaps": ["What the research doesn't cover"],
    "recommendations": ["Based on the research findings..."]
}

Remember: If it's not in the research content, don't include it."""
    
    async def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process research data and generate analysis.
        
        Args:
            input_data: Should contain 'research_content' from Research Agent
            context: Optional additional context
        
        Returns:
            AgentResponse with analysis results
        """
        start_time = time.time()
        
        research_content = input_data.get("research_content", "")
        original_query = input_data.get("query", "")
        
        if not research_content:
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content="",
                success=False,
                error="No research content provided for analysis"
            )
        
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
            ]
            
            # Add context about analysis style
            style_instructions = self._get_style_instructions()
            messages.append({
                "role": "system", 
                "content": f"Analysis style preferences:\n{style_instructions}"
            })
            
            # Add context if available
            if context:
                context_str = f"Additional context:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "system", "content": context_str})
            
            # Add conversation history
            messages.extend(self.get_history_for_llm())
            
            # Create the analysis request - EXPLICIT about using only provided content
            request = f"""Analyze the following research findings.

IMPORTANT: Base your analysis ONLY on the content below. Do NOT add external knowledge.

Original Query: {original_query}

=== RESEARCH CONTENT (analyze ONLY this) ===
{research_content}
=== END OF RESEARCH CONTENT ===

Your analysis must:
1. Summarize ONLY what's in the research content above
2. Extract key points that are ACTUALLY stated in the sources
3. Identify patterns WITHIN the provided sources
4. Note data points WITH their source attribution
5. Identify what the research does NOT cover (gaps)
6. Make recommendations based ONLY on the research findings

Maximum key points: {self.max_key_points}

DO NOT include any information that is not present in the research content above."""

            messages.append({"role": "user", "content": request})
            
            # Call LLM
            response_content = await self._call_llm(messages)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Parse and validate response
            metadata = self._extract_metadata(response_content)
            
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
    
    def _get_style_instructions(self) -> str:
        """Get instructions based on summarization style."""
        styles = {
            "brief": "Keep summaries concise and to the point. Focus on the most critical information only.",
            "comprehensive": "Provide detailed summaries that cover all important aspects. Include nuance and context.",
            "bullet_points": "Use bullet points and numbered lists extensively. Make information scannable."
        }
        return styles.get(self.summarization_style, styles["comprehensive"])
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract structured metadata from the analysis response."""
        try:
            from core.utils import clean_json_response
            data = json.loads(clean_json_response(content))
            return {
                "summary_length": len(data.get("summary", "")),
                "num_key_points": len(data.get("key_points", [])),
                "num_patterns": len(data.get("patterns", [])),
                "num_themes": len(data.get("themes", [])),
                "num_gaps": len(data.get("gaps", [])),
                "num_recommendations": len(data.get("recommendations", []))
            }
        except json.JSONDecodeError:
            return {"raw_response": True}
    
    async def summarize(self, content: str, max_length: int = 500) -> str:
        """Quick summarization utility method."""
        messages = [
            {
                "role": "system", 
                "content": f"Summarize the following content in {max_length} characters or less. Be concise but comprehensive."
            },
            {"role": "user", "content": content}
        ]
        return await self._call_llm(messages)
    
    async def extract_key_points(self, content: str, num_points: int = 5) -> List[str]:
        """Extract key points from content."""
        messages = [
            {
                "role": "system",
                "content": f"Extract exactly {num_points} key points from the following content. Return as a JSON array of strings."
            },
            {"role": "user", "content": content}
        ]
        response = await self._call_llm(messages)
        try:
            from core.utils import clean_json_response
            return json.loads(clean_json_response(response))
        except json.JSONDecodeError:
            return [response]


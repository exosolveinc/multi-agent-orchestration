import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from core.base_agent import Agent, AgentConfig
from core.types import AgentRole, AgentResponse


@dataclass
class CriticAgentConfig(AgentConfig):
    """Configuration specific to the Critic Agent."""
    strictness_level: str = "balanced"  # "lenient", "balanced", "strict"
    check_citations: bool = True
    flag_speculation: bool = True


class CriticAgent(Agent):
    """
    Critic Agent: Fact-checks claims, flags contradictions, and assigns confidence scores.
    
    Capabilities:
    - Fact verification against sources
    - Contradiction detection
    - Confidence scoring
    - Bias identification
    - Source reliability assessment
    - Speculation flagging
    """
    
    def __init__(self, config: CriticAgentConfig, llm_client: Any):
        config.role = AgentRole.CRITIC
        super().__init__(config, llm_client)
        
        self.strictness_level = config.strictness_level
        self.check_citations = config.check_citations
        self.flag_speculation = config.flag_speculation
    
    def _default_system_prompt(self) -> str:
        return """You are a Critic Agent specialized in fact-checking and quality assessment.

Your responsibilities:
1. VERIFY: Check claims against provided sources and general knowledge
2. CONTRADICT: Identify contradictions within the content or with known facts
3. SCORE: Assign confidence scores based on evidence quality
4. FLAG: Highlight unsupported claims, speculation, and potential biases
5. ASSESS: Evaluate source reliability and citation quality

Guidelines:
- Be thorough but fair in your criticism
- Distinguish between facts, opinions, and speculation
- Consider the strength of evidence for each claim
- Note when claims cannot be verified
- Provide constructive feedback for improvement
- Rate confidence on a 0.0 to 1.0 scale

Confidence Score Guidelines:
- 0.9-1.0: Well-documented, multiple reliable sources confirm
- 0.7-0.8: Good evidence, minor gaps or single source
- 0.5-0.6: Mixed evidence, some contradictions
- 0.3-0.4: Limited evidence, significant gaps
- 0.0-0.2: Unverified, speculative, or contradicted

Output Format:
Return your critique as a JSON object with:
{
    "overall_confidence": 0.0-1.0,
    "verified_claims": [
        {"claim": "...", "confidence": 0.0-1.0, "evidence": "..."}
    ],
    "unverified_claims": [
        {"claim": "...", "reason": "Why it couldn't be verified"}
    ],
    "contradictions": [
        {
            "statement1": "...",
            "statement2": "...",
            "nature": "Description of the contradiction"
        }
    ],
    "potential_biases": [
        {"type": "...", "description": "...", "severity": "low/medium/high"}
    ],
    "source_assessment": {
        "reliability": 0.0-1.0,
        "diversity": 0.0-1.0,
        "recency": 0.0-1.0,
        "notes": "..."
    },
    "speculation_flags": ["List of speculative statements"],
    "improvement_suggestions": ["Suggestion 1", "Suggestion 2"],
    "final_verdict": "Summary assessment of the content quality"
}"""
    
    async def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Critique research and analysis content.
        
        Args:
            input_data: Should contain 'content' to critique and optionally 'sources'
            context: Previous agent outputs and original query
        
        Returns:
            AgentResponse with critique results
        """
        start_time = time.time()
        
        content = input_data.get("content", "")
        sources = input_data.get("sources", [])
        original_query = input_data.get("query", "")
        
        if not content:
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content="",
                success=False,
                error="No content provided for critique"
            )
        
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
            ]
            
            # Add strictness configuration
            strictness_instructions = self._get_strictness_instructions()
            messages.append({
                "role": "system",
                "content": f"Critique strictness: {self.strictness_level}\n{strictness_instructions}"
            })
            
            # Add context if available
            if context:
                context_str = f"Context from previous agents:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "system", "content": context_str})
            
            # Add conversation history
            messages.extend(self.get_history_for_llm())
            
            # Create the critique request
            source_info = ""
            if sources:
                source_info = f"\n\nAvailable Sources:\n{json.dumps(sources, indent=2)}"
            
            request = f"""Critically evaluate the following content:

Original Query: {original_query}

Content to Critique:
{content}
{source_info}

Provide a thorough critique including:
1. Overall confidence score
2. List of verified and unverified claims
3. Any contradictions found
4. Potential biases
5. Source assessment
6. Flagged speculation
7. Suggestions for improvement
8. Final verdict"""

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
    
    def _get_strictness_instructions(self) -> str:
        """Get instructions based on strictness level."""
        levels = {
            "lenient": "Be understanding of minor issues. Focus on major problems only. Give benefit of the doubt.",
            "balanced": "Apply fair criticism. Note both strengths and weaknesses. Be constructive.",
            "strict": "Apply rigorous standards. Flag all issues regardless of severity. Require strong evidence."
        }
        return levels.get(self.strictness_level, levels["balanced"])
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract structured metadata from the critique response."""
        try:
            from core.utils import clean_json_response
            data = json.loads(clean_json_response(content))
            return {
                "overall_confidence": data.get("overall_confidence", 0.0),
                "num_verified_claims": len(data.get("verified_claims", [])),
                "num_unverified_claims": len(data.get("unverified_claims", [])),
                "num_contradictions": len(data.get("contradictions", [])),
                "num_biases": len(data.get("potential_biases", [])),
                "source_reliability": data.get("source_assessment", {}).get("reliability", 0.0),
                "num_speculation_flags": len(data.get("speculation_flags", []))
            }
        except json.JSONDecodeError:
            return {"raw_response": True}
    
    async def quick_confidence_check(self, claim: str, evidence: str) -> float:
        """Quick confidence assessment for a single claim."""
        messages = [
            {
                "role": "system",
                "content": "You are a fact-checker. Assess the confidence level (0.0-1.0) for the given claim based on the evidence. Return only a number."
            },
            {
                "role": "user",
                "content": f"Claim: {claim}\n\nEvidence: {evidence}"
            }
        ]
        response = await self._call_llm(messages)
        try:
            return float(response.strip())
        except ValueError:
            return 0.5
    
    async def find_contradictions(self, statements: List[str]) -> List[Dict[str, str]]:
        """Find contradictions in a list of statements."""
        messages = [
            {
                "role": "system",
                "content": "Analyze the following statements and identify any contradictions. Return as JSON array with objects containing 'statement1', 'statement2', and 'nature'."
            },
            {
                "role": "user",
                "content": "\n".join([f"- {s}" for s in statements])
            }
        ]
        response = await self._call_llm(messages)
        try:
            from core.utils import clean_json_response
            return json.loads(clean_json_response(response))
        except json.JSONDecodeError:
            return []


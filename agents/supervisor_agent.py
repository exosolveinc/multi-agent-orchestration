import json
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.base_agent import Agent, AgentConfig
from core.types import (
    AgentRole, 
    AgentResponse, 
    ExecutionMode, 
    TaskStatus,
    ResearchTask,
    ResearchResult,
    HumanCheckpoint,
    Citation
)


class CheckpointType(Enum):
    RESEARCH_COMPLETE = "research_complete"
    ANALYSIS_COMPLETE = "analysis_complete"  
    BEFORE_CRITIQUE = "before_critique"
    FINAL_APPROVAL = "final_approval"


@dataclass
class SupervisorConfig(AgentConfig):
    """Configuration for the Supervisor Agent."""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_human_checkpoints: bool = True
    checkpoint_callback: Optional[Callable] = None  # Called when checkpoint is needed
    timeout_per_agent: int = 120  # seconds


class SupervisorAgent(Agent):
    """
    Supervisor Agent: Orchestrates the research workflow.
    
    Responsibilities:
    - Route tasks to appropriate agents
    - Manage execution flow (sequential or parallel)
    - Handle failures and retries
    - Coordinate human-in-the-loop checkpoints
    - Aggregate results into final output
    """
    
    def __init__(
        self,
        config: SupervisorConfig,
        llm_client: Any,
        research_agent: Agent,
        analysis_agent: Agent,
        critic_agent: Agent
    ):
        config.role = AgentRole.SUPERVISOR
        super().__init__(config, llm_client)
        
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
        self.critic_agent = critic_agent
        
        self.execution_mode = config.execution_mode
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay_seconds
        self.enable_checkpoints = config.enable_human_checkpoints
        self.checkpoint_callback = config.checkpoint_callback
        self.timeout = config.timeout_per_agent
        
        # State management
        self.pending_checkpoints: Dict[str, HumanCheckpoint] = {}
        self.task_history: List[ResearchTask] = []
    
    def _default_system_prompt(self) -> str:
        return """You are a Supervisor Agent coordinating a team of specialized AI agents.

Your team:
1. Research Agent: Gathers information from web searches
2. Analysis Agent: Summarizes and extracts insights
3. Critic Agent: Fact-checks and assigns confidence scores

Your responsibilities:
- Coordinate the workflow between agents
- Ensure quality at each step
- Handle errors gracefully
- Produce a final, polished report

You help synthesize the outputs from all agents into a coherent final report."""
    
    async def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a research request through the full pipeline.
        
        Args:
            input_data: Should contain 'query' and optional configuration
            context: Optional additional context
        
        Returns:
            AgentResponse with the final research result
        """
        start_time = time.time()
        
        query = input_data.get("query", "")
        execution_mode = input_data.get("execution_mode", self.execution_mode)
        require_approval = input_data.get("require_human_approval", self.enable_checkpoints)
        
        if not query:
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content="",
                success=False,
                error="No query provided"
            )
        
        # Create task
        task = ResearchTask(
            query=query,
            execution_mode=execution_mode if isinstance(execution_mode, ExecutionMode) else ExecutionMode(execution_mode),
            require_human_approval=require_approval,
            max_retries=self.max_retries
        )
        
        try:
            # Execute the research pipeline
            result = await self._execute_pipeline(task)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content=json.dumps(result.to_dict(), indent=2),
                success=True,
                metadata={
                    "task_id": task.id,
                    "execution_mode": task.execution_mode.value,
                    "agents_used": result.agents_used,
                    "confidence_score": result.confidence_score
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            task.status = TaskStatus.FAILED
            task.errors.append({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self.task_history.append(task)
            
            return AgentResponse(
                agent_id=self.id,
                agent_role=self.role,
                content="",
                success=False,
                error=str(e),
                metadata={"task_id": task.id},
                execution_time_ms=execution_time
            )
    
    async def _execute_pipeline(self, task: ResearchTask) -> ResearchResult:
        """Execute the full research pipeline."""
        task.status = TaskStatus.IN_PROGRESS
        agents_used = []
        
        # Step 1: Research
        research_response = await self._execute_with_retry(
            self.research_agent,
            {"query": task.query},
            task
        )
        
        if not research_response.success:
            raise RuntimeError(f"Research failed: {research_response.error}")
        
        agents_used.append(self.research_agent.id)
        task.research_results = {
            "content": research_response.content,
            "metadata": research_response.metadata
        }
        
        # Checkpoint after research
        if task.require_human_approval:
            await self._create_checkpoint(
                task,
                CheckpointType.RESEARCH_COMPLETE,
                "Research phase complete. Review findings before analysis.",
                {"research_summary": research_response.content[:500]}
            )
        
        # Step 2: Analysis
        analysis_response = await self._execute_with_retry(
            self.analysis_agent,
            {
                "query": task.query,
                "research_content": research_response.content
            },
            task
        )
        
        if not analysis_response.success:
            raise RuntimeError(f"Analysis failed: {analysis_response.error}")
        
        agents_used.append(self.analysis_agent.id)
        task.analysis_results = {
            "content": analysis_response.content,
            "metadata": analysis_response.metadata
        }
        
        # Step 3: Critique
        critic_response = await self._execute_with_retry(
            self.critic_agent,
            {
                "query": task.query,
                "content": analysis_response.content,
                "sources": research_response.metadata.get("citations", [])
            },
            task
        )
        
        if not critic_response.success:
            raise RuntimeError(f"Critique failed: {critic_response.error}")
        
        agents_used.append(self.critic_agent.id)
        task.critic_results = {
            "content": critic_response.content,
            "metadata": critic_response.metadata
        }
        
        # Parse research for citations
        citations = self._extract_citations(research_response)
        
        # Parse analysis for key points and summary
        try:
            from core.utils import clean_json_response
            analysis_data = json.loads(clean_json_response(analysis_response.content))
            key_points = analysis_data.get("key_points", [])
            analysis_summary = analysis_data.get("summary", "")
        except json.JSONDecodeError:
            key_points = []
            analysis_summary = analysis_response.content
        
        # Parse critic for confidence and contradictions
        try:
            from core.utils import clean_json_response
            critic_data = json.loads(clean_json_response(critic_response.content))
            confidence = critic_data.get("overall_confidence", 0.5)
            contradictions = [c.get("nature", str(c)) for c in critic_data.get("contradictions", [])]
            limitations = critic_data.get("speculation_flags", [])
            final_verdict = critic_data.get("final_verdict", "")
        except json.JSONDecodeError:
            confidence = critic_response.metadata.get("overall_confidence", 0.5)
            contradictions = []
            limitations = []
            final_verdict = ""
        
        # Generate final report (synthesized summary)
        final_report = await self._generate_final_report(
            task.query,
            research_response,
            analysis_response,
            critic_response
        )
        
        # Checkpoint before final
        if task.require_human_approval:
            await self._create_checkpoint(
                task,
                CheckpointType.FINAL_APPROVAL,
                "Final report ready for approval.",
                {"preview": final_report[:1000]}
            )
        
        task.status = TaskStatus.COMPLETED
        task.final_report = final_report
        task.citations = citations
        task.confidence_score = confidence
        self.task_history.append(task)
        
        return ResearchResult(
            task_id=task.id,
            query=task.query,
            summary=analysis_summary if analysis_summary else final_report,
            key_points=key_points,
            confidence_score=confidence,
            contradictions=contradictions,
            limitations=limitations,
            citations=citations,
            agents_used=agents_used,
            execution_time_ms=int((datetime.now() - task.created_at).total_seconds() * 1000),
            execution_mode=task.execution_mode.value
        )
    
    async def _execute_with_retry(
        self,
        agent: Agent,
        input_data: Dict[str, Any],
        task: ResearchTask
    ) -> AgentResponse:
        """Execute an agent with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    task.status = TaskStatus.RETRYING
                    task.retry_count += 1
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                
                response = await asyncio.wait_for(
                    agent.process(input_data),
                    timeout=self.timeout
                )
                
                if response.success:
                    return response
                    
                last_error = response.error
                task.errors.append({
                    "agent": agent.id,
                    "attempt": attempt + 1,
                    "error": response.error,
                    "timestamp": datetime.now().isoformat()
                })
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                task.errors.append({
                    "agent": agent.id,
                    "attempt": attempt + 1,
                    "error": "Timeout",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                last_error = str(e)
                task.errors.append({
                    "agent": agent.id,
                    "attempt": attempt + 1,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return AgentResponse(
            agent_id=agent.id,
            agent_role=agent.role,
            content="",
            success=False,
            error=f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        )
    
    async def _create_checkpoint(
        self,
        task: ResearchTask,
        checkpoint_type: CheckpointType,
        message: str,
        context: Dict[str, Any]
    ) -> None:
        """Create a human-in-the-loop checkpoint."""
        checkpoint = HumanCheckpoint(
            task_id=task.id,
            checkpoint_type=checkpoint_type.value,
            message=message,
            context=context
        )
        
        task.status = TaskStatus.AWAITING_HUMAN
        self.pending_checkpoints[checkpoint.id] = checkpoint
        
        if self.checkpoint_callback:
            # In a real system, this would notify the user and wait
            # For demo purposes, we auto-approve after callback
            await self.checkpoint_callback(checkpoint)
            checkpoint.resolved = True
            checkpoint.resolution = "auto_approved"
            checkpoint.resolved_at = datetime.now()
        else:
            # Auto-approve if no callback
            checkpoint.resolved = True
            checkpoint.resolution = "auto_approved"
            checkpoint.resolved_at = datetime.now()
        
        task.status = TaskStatus.IN_PROGRESS
    
    async def resolve_checkpoint(
        self,
        checkpoint_id: str,
        resolution: str,
        feedback: Optional[str] = None
    ) -> bool:
        """Resolve a pending checkpoint."""
        checkpoint = self.pending_checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False
        
        checkpoint.resolved = True
        checkpoint.resolution = resolution
        checkpoint.resolved_at = datetime.now()
        if feedback:
            checkpoint.context["human_feedback"] = feedback
        
        return True
    
    async def _generate_final_report(
        self,
        query: str,
        research: AgentResponse,
        analysis: AgentResponse,
        critique: AgentResponse
    ) -> str:
        """Generate the final synthesized report."""
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": f"""Synthesize the following into a polished final report:

ORIGINAL QUERY: {query}

RESEARCH FINDINGS:
{research.content}

ANALYSIS:
{analysis.content}

CRITIQUE & CONFIDENCE:
{critique.content}

Create a well-structured report that:
1. Directly answers the original query
2. Presents key findings clearly
3. Includes important caveats and limitations
4. Provides a confidence assessment
5. Lists sources used

Format the report professionally with clear sections."""
            }
        ]
        
        return await self._call_llm(messages)
    
    def _extract_citations(self, research_response: AgentResponse) -> List[Citation]:
        """Extract citations from research response."""
        citations = []
        
        # First try metadata
        citation_data = research_response.metadata.get("citations", [])
        
        # If empty, try parsing the content
        if not citation_data:
            try:
                from core.utils import clean_json_response
                content_data = json.loads(clean_json_response(research_response.content))
                citation_data = content_data.get("sources", [])
            except json.JSONDecodeError:
                pass
        
        for c in citation_data:
            citations.append(Citation(
                title=c.get("title", ""),
                url=c.get("url", ""),
                snippet=c.get("snippet", c.get("content", "")),
                relevance_score=c.get("relevance_score", c.get("relevance", c.get("score", 0.0)))
            ))
        
        return citations
    
    def get_pending_checkpoints(self) -> List[HumanCheckpoint]:
        """Get all pending checkpoints."""
        return [cp for cp in self.pending_checkpoints.values() if not cp.resolved]
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get history of processed tasks."""
        return [task.to_dict() for task in self.task_history]


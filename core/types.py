from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class AgentRole(Enum):
    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CRITIC = "critic"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"  # Agents run one after another
    PARALLEL = "parallel"      # Independent agents run simultaneously


@dataclass
class ConversationMessage:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A source citation for research findings."""
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0
    retrieved_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass
class ResearchTask:
    """A research task to be processed by the agent system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Execution configuration
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    require_human_approval: bool = False
    max_retries: int = 3
    
    # Results from each agent
    research_results: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    critic_results: Optional[Dict[str, Any]] = None
    
    # Final output
    final_report: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "execution_mode": self.execution_mode.value,
            "require_human_approval": self.require_human_approval,
            "research_results": self.research_results,
            "analysis_results": self.analysis_results,
            "critic_results": self.critic_results,
            "final_report": self.final_report,
            "citations": [c.to_dict() for c in self.citations],
            "confidence_score": self.confidence_score,
            "errors": self.errors,
            "retry_count": self.retry_count,
        }


@dataclass
class ResearchResult:
    """Structured output from the research system."""
    task_id: str
    query: str
    
    # Main findings
    summary: str
    key_points: List[str]
    
    # Quality metrics
    confidence_score: float  # 0.0 to 1.0
    contradictions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Sources
    citations: List[Citation] = field(default_factory=list)
    
    # Metadata
    agents_used: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    execution_mode: str = "sequential"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "summary": self.summary,
            "key_points": self.key_points,
            "confidence_score": self.confidence_score,
            "contradictions": self.contradictions,
            "limitations": self.limitations,
            "citations": [c.to_dict() for c in self.citations],
            "agents_used": self.agents_used,
            "execution_time_ms": self.execution_time_ms,
            "execution_mode": self.execution_mode,
        }


@dataclass
class AgentResponse:
    """Response from an individual agent."""
    agent_id: str
    agent_role: AgentRole
    content: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role.value,
            "content": self.content,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class HumanCheckpoint:
    """A checkpoint requiring human approval."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    checkpoint_type: str = ""  # "approval", "review", "clarification"
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=lambda: ["approve", "reject", "modify"])
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "checkpoint_type": self.checkpoint_type,
            "message": self.message,
            "context": self.context,
            "options": self.options,
            "created_at": self.created_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


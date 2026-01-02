import os
import sys
import uuid
import asyncio
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize clients conditionally
llm_client = None
tavily_client = None


def get_llm_client():
    """Get LLM client (Claude/Anthropic preferred, OpenAI fallback)."""
    global llm_client
    if llm_client is None:
        from core.llm import create_llm_client
        llm_client = create_llm_client()  # Auto-detects provider
    return llm_client


def get_tavily_client():
    global tavily_client
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_client is None and tavily_key:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=tavily_key)
    return tavily_client


# Request/Response Models
class ResearchRequest(BaseModel):
    query: str = Field(..., description="The research query to investigate")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    execution_mode: str = Field(default="sequential", description="'sequential' or 'parallel'")
    require_human_approval: bool = Field(default=False, description="Enable human-in-the-loop checkpoints")


class CheckpointResolution(BaseModel):
    checkpoint_id: str
    resolution: str = Field(..., description="'approve', 'reject', or 'modify'")
    feedback: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: dict
    result: Optional[dict] = None


# Initialize FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Research Orchestrator API starting...")
    yield
    # Shutdown
    print("👋 Research Orchestrator API shutting down...")


app = FastAPI(
    title="Multi-Agent Research Orchestrator",
    description="""
    A sophisticated multi-agent system for research tasks featuring:
    - **Supervisor Agent**: Coordinates workflow, handles failures, retries
    - **Research Agent**: Web search and source gathering via Tavily
    - **Analysis Agent**: Summarization and pattern extraction
    - **Critic Agent**: Fact-checking and confidence scoring
    
    Supports human-in-the-loop checkpoints for sensitive decisions.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# In-memory task storage (in production, use Redis)
active_tasks = {}
task_results = {}


def create_agents():
    """Create and configure all agents using Claude (or OpenAI fallback)."""
    from agents.research_agent import ResearchAgent, ResearchAgentConfig
    from agents.analysis_agent import AnalysisAgent, AnalysisAgentConfig
    from agents.critic_agent import CriticAgent, CriticAgentConfig
    from agents.supervisor_agent import SupervisorAgent, SupervisorConfig
    from core.types import AgentRole, ExecutionMode
    from config import config
    
    client = get_llm_client()
    tavily = get_tavily_client()
    model = config.llm_model
    
    research_agent = ResearchAgent(
        config=ResearchAgentConfig(
            name="Research Agent",
            description="Gathers information from web searches and validates sources",
            role=AgentRole.RESEARCH,
            model=model,
            tavily_api_key=config.tavily_api_key,
            max_search_results=5
        ),
        llm_client=client,
        tavily_client=tavily
    )
    
    analysis_agent = AnalysisAgent(
        config=AnalysisAgentConfig(
            name="Analysis Agent",
            description="Summarizes content and extracts key insights",
            role=AgentRole.ANALYSIS,
            model=model,
            summarization_style="comprehensive"
        ),
        llm_client=client
    )
    
    critic_agent = CriticAgent(
        config=CriticAgentConfig(
            name="Critic Agent",
            description="Fact-checks claims and assigns confidence scores",
            role=AgentRole.CRITIC,
            model=model,
            strictness_level="balanced"
        ),
        llm_client=client
    )
    
    supervisor = SupervisorAgent(
        config=SupervisorConfig(
            name="Supervisor Agent",
            description="Orchestrates research workflow and coordinates agents",
            role=AgentRole.SUPERVISOR,
            model=model,
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_retries=3,
            enable_human_checkpoints=False
        ),
        llm_client=client,
        research_agent=research_agent,
        analysis_agent=analysis_agent,
        critic_agent=critic_agent
    )
    
    return supervisor


# API Endpoints
@app.get("/")
async def root():
    """Serve the UI."""
    html_file = Path(__file__).parent.parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return {
        "name": "Multi-Agent Research Orchestrator",
        "version": "1.0.0",
        "status": "running",
        "agents": ["supervisor", "research", "analysis", "critic"]
    }


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "name": "Multi-Agent Research Orchestrator",
        "version": "1.0.0",
        "status": "running",
        "agents": ["supervisor", "research", "analysis", "critic"],
        "features": [
            "Web search via Tavily",
            "Multi-agent coordination",
            "Fact-checking & confidence scoring",
            "Human-in-the-loop checkpoints",
            "Retry with exponential backoff"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from config import config
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": config.llm_provider,
        "anthropic_configured": bool(config.anthropic_api_key),
        "openai_configured": bool(config.openai_api_key),
        "tavily_configured": bool(config.tavily_api_key)
    }


@app.post("/research")
async def create_research_task(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Create a new research task.
    
    The task runs asynchronously. Use the returned task_id to check status.
    """
    task_id = str(uuid.uuid4())
    user_id = request.user_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    # Store task info
    active_tasks[task_id] = {
        "status": "pending",
        "query": request.query,
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "progress": {
            "research": "pending",
            "analysis": "pending",
            "critique": "pending",
            "final": "pending"
        }
    }
    
    # Run task in background
    background_tasks.add_task(
        run_research_task,
        task_id,
        request.query,
        request.execution_mode,
        request.require_human_approval
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Research task created. Use /research/{task_id}/status to check progress."
    }


async def run_research_task(
    task_id: str,
    query: str,
    execution_mode: str,
    require_approval: bool
):
    """Background task to run the research pipeline."""
    try:
        active_tasks[task_id]["status"] = "running"
        active_tasks[task_id]["progress"]["research"] = "running"
        
        supervisor = create_agents()
        
        # Update progress as we go
        response = await supervisor.process({
            "query": query,
            "execution_mode": execution_mode,
            "require_human_approval": require_approval
        })
        
        if response.success:
            active_tasks[task_id]["status"] = "completed"
            active_tasks[task_id]["progress"] = {
                "research": "completed",
                "analysis": "completed",
                "critique": "completed",
                "final": "completed"
            }
            task_results[task_id] = {
                "success": True,
                "result": response.content,
                "metadata": response.metadata,
                "execution_time_ms": response.execution_time_ms
            }
        else:
            active_tasks[task_id]["status"] = "failed"
            task_results[task_id] = {
                "success": False,
                "error": response.error
            }
            
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        task_results[task_id] = {
            "success": False,
            "error": str(e)
        }


@app.get("/research/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the status of a research task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    result = task_results.get(task_id)
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "query": task["query"],
        "progress": task["progress"],
        "created_at": task["created_at"],
        "result": result
    }


@app.get("/research/{task_id}/result")
async def get_task_result(task_id: str):
    """Get the result of a completed research task."""
    if task_id not in task_results:
        if task_id in active_tasks:
            return {
                "task_id": task_id,
                "status": active_tasks[task_id]["status"],
                "message": "Task not yet completed"
            }
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_results[task_id]


@app.post("/research/sync")
async def create_research_task_sync(request: ResearchRequest):
    """
    Run a research task synchronously and return the result.
    
    Warning: This may take 30-60 seconds for complex queries.
    """
    try:
        supervisor = create_agents()
        
        response = await supervisor.process({
            "query": request.query,
            "execution_mode": request.execution_mode,
            "require_human_approval": request.require_human_approval
        })
        
        if response.success:
            return {
                "success": True,
                "query": request.query,
                "result": response.content,
                "metadata": response.metadata,
                "execution_time_ms": response.execution_time_ms
            }
        else:
            raise HTTPException(status_code=500, detail=response.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/checkpoints")
async def list_checkpoints():
    """List all pending human-in-the-loop checkpoints."""
    # In a real implementation, this would query the storage
    return {
        "pending_checkpoints": [],
        "message": "No pending checkpoints"
    }


@app.post("/checkpoints/resolve")
async def resolve_checkpoint(resolution: CheckpointResolution):
    """Resolve a human-in-the-loop checkpoint."""
    # In a real implementation, this would update the checkpoint
    return {
        "checkpoint_id": resolution.checkpoint_id,
        "status": "resolved",
        "resolution": resolution.resolution
    }


@app.get("/tasks")
async def list_tasks():
    """List all tasks."""
    return {
        "tasks": [
            {
                "task_id": tid,
                **task
            }
            for tid, task in active_tasks.items()
        ]
    }


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    if task_id in active_tasks:
        del active_tasks[task_id]
    if task_id in task_results:
        del task_results[task_id]
    return {"status": "deleted", "task_id": task_id}


# Run with: uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
Multi-Agent Research Orchestrator

A sophisticated multi-agent system for comprehensive research tasks featuring:
- Supervisor Agent: Coordinates workflow, handles failures, retries
- Research Agent: Web search and source gathering via Tavily
- Analysis Agent: Summarization and pattern extraction  
- Critic Agent: Fact-checking and confidence scoring

Key Features:
- Configurable agent topology (parallel vs sequential execution)
- Human-in-the-loop checkpoints for sensitive decisions
- Conversation memory that persists across sessions
- Structured output with citations and confidence scores
- Graceful degradation when an agent fails
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent Research Team"


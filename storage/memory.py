from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from core.types import ConversationMessage, ResearchTask, HumanCheckpoint


@dataclass
class ConversationMemory:
    """Memory for a single conversation/session."""
    session_id: str
    user_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)  # Task IDs
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages(self, limit: Optional[int] = None) -> List[ConversationMessage]:
        """Get messages, optionally limited to most recent."""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "agent_id": m.agent_id
                }
                for m in self.messages
            ],
            "tasks": self.tasks,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class MemoryStore:
    """
    In-memory storage for conversations, tasks, and checkpoints.
    
    In production, this would be backed by Redis or a database.
    """
    
    def __init__(self):
        self._conversations: Dict[str, ConversationMemory] = {}
        self._tasks: Dict[str, ResearchTask] = {}
        self._checkpoints: Dict[str, HumanCheckpoint] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
    
    # Conversation methods
    def get_or_create_conversation(
        self, 
        session_id: str, 
        user_id: str
    ) -> ConversationMemory:
        """Get existing conversation or create new one."""
        if session_id not in self._conversations:
            conversation = ConversationMemory(
                session_id=session_id,
                user_id=user_id
            )
            self._conversations[session_id] = conversation
            
            # Track user sessions
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
        
        return self._conversations[session_id]
    
    def get_conversation(self, session_id: str) -> Optional[ConversationMemory]:
        """Get a conversation by session ID."""
        return self._conversations.get(session_id)
    
    def add_message(
        self,
        session_id: str,
        message: ConversationMessage
    ) -> None:
        """Add a message to a conversation."""
        if session_id in self._conversations:
            self._conversations[session_id].add_message(message)
    
    def get_user_conversations(self, user_id: str) -> List[ConversationMemory]:
        """Get all conversations for a user."""
        session_ids = self._user_sessions.get(user_id, [])
        return [
            self._conversations[sid] 
            for sid in session_ids 
            if sid in self._conversations
        ]
    
    # Task methods
    def save_task(self, task: ResearchTask) -> None:
        """Save or update a task."""
        self._tasks[task.id] = task
        task.updated_at = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[ResearchTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ResearchTask]:
        """Get all tasks."""
        return list(self._tasks.values())
    
    def get_tasks_by_status(self, status: str) -> List[ResearchTask]:
        """Get tasks by status."""
        return [t for t in self._tasks.values() if t.status.value == status]
    
    # Checkpoint methods
    def save_checkpoint(self, checkpoint: HumanCheckpoint) -> None:
        """Save a checkpoint."""
        self._checkpoints[checkpoint.id] = checkpoint
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[HumanCheckpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
    
    def get_pending_checkpoints(self) -> List[HumanCheckpoint]:
        """Get all pending (unresolved) checkpoints."""
        return [cp for cp in self._checkpoints.values() if not cp.resolved]
    
    def get_checkpoints_for_task(self, task_id: str) -> List[HumanCheckpoint]:
        """Get all checkpoints for a task."""
        return [cp for cp in self._checkpoints.values() if cp.task_id == task_id]
    
    def resolve_checkpoint(
        self,
        checkpoint_id: str,
        resolution: str,
        feedback: Optional[str] = None
    ) -> bool:
        """Resolve a checkpoint."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False
        
        checkpoint.resolved = True
        checkpoint.resolution = resolution
        checkpoint.resolved_at = datetime.now()
        if feedback:
            checkpoint.context["human_feedback"] = feedback
        
        return True
    
    # Utility methods
    def clear(self) -> None:
        """Clear all stored data."""
        self._conversations.clear()
        self._tasks.clear()
        self._checkpoints.clear()
        self._user_sessions.clear()
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state as JSON-serializable dict."""
        return {
            "conversations": {
                sid: conv.to_dict() 
                for sid, conv in self._conversations.items()
            },
            "tasks": {
                tid: task.to_dict() 
                for tid, task in self._tasks.items()
            },
            "checkpoints": {
                cid: cp.to_dict() 
                for cid, cp in self._checkpoints.items()
            },
            "user_sessions": self._user_sessions
        }


# Global memory store instance
memory_store = MemoryStore()


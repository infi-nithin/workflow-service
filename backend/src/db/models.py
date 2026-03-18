from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import uuid4
from sqlalchemy import (
    String,
    DateTime,
    Index,
    Text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    trace_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    graph_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model_versions_used: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    total_tokens: Mapped[int] = mapped_column(default=0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="running"
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(default=0, nullable=True)
    nodes: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    error: Mapped[Optional[str]] = mapped_column(String(4000), nullable=True)
    # HITL (Human In The Loop) fields
    pending_hitl: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True, default=None
    )
    current_state: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False
    )
    # Indexes
    __table_args__ = (
        Index("idx_workflow_executions_trace_id", "trace_id"),
        Index("idx_workflow_executions_workflow_id", "workflow_id"),
        Index("idx_workflow_executions_intent", "intent"),
        Index("idx_workflow_executions_started_at", "started_at"),
        Index("idx_workflow_executions_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<WorkflowExecution(trace_id={self.trace_id}, workflow_id={self.workflow_id}, status={self.status})>"


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    prompt_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    prompt_content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_prompts_prompt_name", "prompt_name"),
    )

    def __repr__(self) -> str:
        return f"<Prompt(prompt_name={self.prompt_name}, prompt_type={self.prompt_type})>"


class ToolRegistry(Base):
    __tablename__ = "tool_registries"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    sys_id: Mapped[str] = mapped_column(String(3), unique=True, nullable=False)
    tool_registry_url: Mapped[str] = mapped_column(String(500), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ToolRegistry(sys_id={self.sys_id}, tool_registry_url={self.tool_registry_url}, is_active={self.is_active})>"
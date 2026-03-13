"""Initial migration - create Workflow Execution tables

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create workflow_executions table if not exists
    op.execute("""
        CREATE TABLE IF NOT EXISTS workflow_executions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            trace_id VARCHAR(36) UNIQUE NOT NULL,
            workflow_id VARCHAR(255) NOT NULL,
            graph_version VARCHAR(50),
            intent VARCHAR(255),
            model_versions_used JSONB NOT NULL DEFAULT '[]',
            total_tokens INTEGER NOT NULL DEFAULT 0,
            started_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            status VARCHAR(50) NOT NULL DEFAULT 'running',
            duration_ms INTEGER,
            nodes JSONB NOT NULL DEFAULT '[]',
            error VARCHAR(4000),
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    
    # Create indexes if not exists
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_trace_id ON workflow_executions(trace_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_intent ON workflow_executions(intent)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_status")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_started_at")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_intent")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_workflow_id")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_trace_id")
    op.execute("DROP TABLE IF EXISTS workflow_executions")

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "001_initial"
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
            pending_hitl JSONB,
            current_state JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    # Create indexes if not exists
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_workflow_executions_trace_id ON workflow_executions(trace_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_workflow_executions_intent ON workflow_executions(intent)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status)"
    )

    # Create prompts table if not exists
    op.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            prompt_name VARCHAR(255) UNIQUE NOT NULL,
            prompt_content TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    # Create indexes for prompts table
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_prompts_prompt_name ON prompts(prompt_name)"
    )

    # Create tool_registries table if not exists
    op.execute("""
        CREATE TABLE IF NOT EXISTS tool_registries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            sys_id VARCHAR(3) UNIQUE NOT NULL,
            tool_registry_url VARCHAR(500) NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            description VARCHAR(500),
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    # Insert HITL confirmation prompt
    op.execute("""
        INSERT INTO prompts (prompt_name, prompt_content)
        VALUES (
            'hitl_confirmation',
            $prompt$Human confirmation required to proceed with the workflow.
            === Current Context ===
            {context}
            === Node Information ===
            {node_info}
            === Next Steps ===
            The workflow can proceed to: {next_steps}
            === Decision Required ===
            {decision_message}
            Please confirm if you want to proceed, or provide input to modify the workflow.
            $prompt$
        )
        ON CONFLICT (prompt_name) DO NOTHING
    """)

    # Insert HITL system prompt (for supervisor to decide when to use HITL)
    op.execute("""
        INSERT INTO prompts (prompt_name, prompt_content)
        VALUES (
            'hitl_system',
            $prompt$You are a workflow supervisor that decides when human confirmation is needed.
            You should request human confirmation (HITL) when:
            1. The action is potentially dangerous or irreversible (e.g., deleting data, sending emails)
            2. The user needs to verify or modify information before proceeding
            3. The workflow reaches a decision point that requires human judgment
            4. External confirmation is required (e.g., API calls, payments)
            5. The user explicitly requested human review at certain steps

            When deciding on HITL, consider:
            - The potential impact of the action
            - Whether the user wants to review before execution
            - The current context and previous node outputs
            - Available tools and their purposes

            If human confirmation is needed, respond with action=HITL and provide a clear message explaining what needs confirmation.
            $prompt$
        )
        ON CONFLICT (prompt_name) DO NOTHING
    """)

    # Insert supervisor system prompt
    op.execute("""
        INSERT INTO prompts (prompt_name, prompt_content)
        VALUES (
            'supervisor_system',
            $prompt$You are a supervisor agent that orchestrates graph-based agent execution.
            - Intent: {intent}
            - Version: {version}
            - Available Nodes:
            {nodes_text}
            You are responsible for deciding what action to take next in the execution flow.
            You have access to the graph structure and must decide whether to:
            1. Call the LLM to process information or generate responses
            2. Execute a tool from the tool registry  
            3. Continue to the next node in the graph
            4. End the execution
            The graph provides a workflow, but YOU decide:
            - Whether to execute a node now or skip to another
            - What specific parameters to pass to tools (based on context)
            - What prompt to give the LLM when processing
            - Whether to branch to different paths based on results
            1. Start from the entry point of the graph (first node with no incoming edges)
            2. When a node requires LLM processing, use the 'llm' action and provide a specific prompt
            3. When a node requires tool execution, use the 'tool' action with:
            - tool_name: The exact name of the tool
            - tool_arguments: A dictionary of parameters extracted from context
            4. After tool execution, analyze results and decide next action
            5. When all required work is complete, use 'end' action
            You must respond with a JSON object containing:
            - action: One of 'llm', 'tool', 'end', or 'continue'
            - reasoning: Why you made this decision based on current state
            - tool_name: (optional) Name of tool to execute
            - tool_arguments: (optional) Arguments for the tool - extract from context
            - llm_prompt: (optional) Specific prompt to give LLM when action is 'llm'
            - next_node: (optional) Next node to execute (for 'continue')
            - response: (optional) Final response to return
            When calling tools, extract parameter values from:
            - The original user input
            - Previous tool execution results
            - LLM responses in the conversation history
            Provide specific values, not placeholders.
            You may choose different paths based on:
            - Results from previous executions
            - Analysis of intermediate results
            - User input context
            The graph is a guide, you make the final decision.
            Always respond with valid JSON.$prompt$
        )
        ON CONFLICT (prompt_name) DO NOTHING
    """)

    # Insert supervisor decision prompt
    op.execute("""
        INSERT INTO prompts (prompt_name, prompt_content)
        VALUES (
            'supervisor_decision',
            $prompt$Current state:
            {node_info}
            Recent conversation:
            {messages_text}
            {outputs_text}
            {tools_text}
            {history_text}
            Based on the graph workflow above and current state, decide what action to take next.
            Think about:
            1. What does the current node require?
            2. What information do you have from previous executions?
            3. What parameters can you extract from the context to pass to tools?
            4. What should the LLM process next?
            Remember: You make the final decision - the graph is a guide, you decide the execution path.
            Respond with a JSON object containing your decision.$prompt$
        )
        ON CONFLICT (prompt_name) DO NOTHING
    """)

    # Insert LLM context prompt
    op.execute("""
        INSERT INTO prompts (prompt_name, prompt_content)
        VALUES (
            'llm_context',
            $prompt$You are a corporate actions expert assistant.$prompt$
        )
        ON CONFLICT (prompt_name) DO NOTHING
    """)

    # Insert classify intent system prompt
    op.execute("""
        INSERT INTO prompts (prompt_name, prompt_content)
        VALUES (
            'classify_intent_system',
            $prompt$You are an intent classifier. Given a user message and a list of available intents,
            you must select the single best matching intent. If no intent matches well, return 'general_query'.
            Available intents: {intents}
            Respond ONLY with the intent name, nothing else.$prompt$
        )
        ON CONFLICT (prompt_name) DO NOTHING
    """)

    # Insert classify intent user prompt
    op.execute("""
        INSERT INTO prompts (prompt_name, prompt_content)
        VALUES (
            'classify_intent_user',
            $prompt$User message: {user_input}
            Select the best matching intent from the available intents. 
            Consider the semantic meaning of the user message and match it to the most appropriate intent.
            If the message is a general greeting or doesn't match any specific intent, return 'general_query'.$prompt$
        )
        ON CONFLICT (prompt_name) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_status")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_started_at")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_intent")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_workflow_id")
    op.execute("DROP INDEX IF EXISTS idx_workflow_executions_trace_id")
    op.execute("DROP TABLE IF EXISTS workflow_executions")
    op.execute("DROP TABLE IF EXISTS tool_registries")
    op.execute("DROP INDEX IF EXISTS idx_prompts_prompt_type")
    op.execute("DROP INDEX IF EXISTS idx_prompts_prompt_name")
    op.execute("DROP TABLE IF EXISTS prompts")
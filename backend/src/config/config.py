import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables at module import
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration values."""

    host: str = "localhost"
    port: int = 5432
    name: str = "workflow_db"
    user: str = "postgres"
    password: str = ""

    @property
    def url(self) -> str:
        """Generate the async database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class ServerConfig:
    """Server configuration values."""

    host: str = "0.0.0.0"
    port: int = 8000
    app_env: str = "development"


@dataclass
class AWSConfig:
    """AWS and Bedrock configuration values."""

    region: str = "us-east-1"
    secret_access_key: str = ""
    access_key_id: str = ""
    session_token: str = ""
    bedrock_model_id: str = ""


@dataclass
class RegistryConfig:
    """Registry and MCP service URLs."""

    graph_registry_url: str = "http://localhost:8002"
    tool_registry_url: str = "http://localhost:8001"
    mcp_server_url: str = ""


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration values."""

    threshold: int = 3
    cooldown: int = 30


@dataclass
class Config:
    """
    Main configuration class that aggregates all configuration sub-classes.
    
    All values are read from environment variables at initialization.
    """

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    def __post_init__(self):
        """Load all environment variables into the configuration objects."""
        # Database configuration
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", str(self.database.port)))
        self.database.name = os.getenv("DB_NAME", self.database.name)
        self.database.user = os.getenv("DB_USER", self.database.user)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)

        # Server configuration
        self.server.host = os.getenv("HOST", self.server.host)
        self.server.port = int(os.getenv("PORT", str(self.server.port)))
        self.server.app_env = os.getenv("APP_ENV", self.server.app_env)

        # AWS configuration
        # Note: Both AWS_REGION and AWS-REGION are used in the codebase
        # Using AWS_REGION as the canonical name
        self.aws.region = os.getenv("AWS_REGION", os.getenv("AWS-REGION", self.aws.region))
        self.aws.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", self.aws.secret_access_key)
        self.aws.access_key_id = os.getenv("AWS_ACCESS_KEY_ID", self.aws.access_key_id)
        self.aws.session_token = os.getenv("AWS_SESSION_TOKEN", self.aws.session_token)
        self.aws.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", self.aws.bedrock_model_id)

        # Registry configuration
        self.registry.graph_registry_url = os.getenv("GRAPH_REGISTRY_URL", self.registry.graph_registry_url)
        self.registry.tool_registry_url = os.getenv("TOOL_REGISTRY_URL", self.registry.tool_registry_url)
        self.registry.mcp_server_url = os.getenv("MCP_SERVER_URL", self.registry.mcp_server_url)

        # Circuit breaker configuration
        self.circuit_breaker.threshold = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", str(self.circuit_breaker.threshold)))
        self.circuit_breaker.cooldown = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN", str(self.circuit_breaker.cooldown)))


# Global config instance - import this in other modules
config = Config()

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables at module import
load_dotenv()


@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    app_env: str = "development"


@dataclass
class AWSConfig:
    region: str
    secret_access_key: str
    access_key_id: str
    session_token: str
    bedrock_model_id: str


@dataclass
class RegistryConfig:
    graph_registry_url: str


@dataclass
class CircuitBreakerConfig:
    threshold: int = 3
    cooldown: int = 30


@dataclass
class Config:
    database: DatabaseConfig
    server: ServerConfig
    aws: AWSConfig
    registry: RegistryConfig
    circuit_breaker: CircuitBreakerConfig

    def __init__(self):
        # Database configuration
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            name=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )

        # Server configuration
        self.server = ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            app_env=os.getenv("APP_ENV", "development"),
        )

        # AWS configuration
        self.aws = AWSConfig(
            region=os.getenv("AWS_REGION"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID"),
        )

        # Registry configuration
        self.registry = RegistryConfig(
            graph_registry_url=os.getenv("GRAPH_REGISTRY_URL"),
        )

        # Circuit breaker configuration
        self.circuit_breaker = CircuitBreakerConfig(
            threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3")),
            cooldown=int(os.getenv("CIRCUIT_BREAKER_COOLDOWN", "30")),
        )


# Global config instance - initialised once at startup
config = Config()

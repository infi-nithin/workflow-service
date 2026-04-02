"""Core logging module for AOP (Aspect-Oriented Programming) logging."""

from aop_logging.logger import (
    AOPLogger,
    get_aop_logger,
    set_aop_logger,
    log_method,
    correlation_id_var,
)
from aop_logging.models import (
    LogEntry,
    APILogEntry,
    ServiceMethodLogEntry,
    OperationStatus,
    OperationType,
    LogLevel,
    LogSummary,
)
from aop_logging.middleware import (
    AOPLoggingMiddleware,
    RequestTimingMiddleware,
)

from aop_logging.config import LoggingConfig, default_config

__all__ = [
    # Logger
    "AOPLogger",
    "get_aop_logger",
    "set_aop_logger",
    "log_method",
    "correlation_id_var",
    
    # Models
    "LogEntry",
    "APILogEntry",
    "ServiceMethodLogEntry",
    "OperationStatus",
    "OperationType",
    "LogLevel",
    "LogSummary",
    
    # Middleware
    "AOPLoggingMiddleware",
    "RequestTimingMiddleware",

    # Config
    "LoggingConfig",
    "default_config",
]
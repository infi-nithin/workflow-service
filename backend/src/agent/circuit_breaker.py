"""Circuit Breaker implementation for tool execution."""
import time
import threading
from enum import Enum
from typing import Dict
from dataclasses import dataclass, field


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests allowed
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    threshold: int = 3       # Number of failures before opening
    cooldown: int = 30       # Seconds to wait before half-open


@dataclass
class CircuitBreaker:
    """Circuit breaker for individual tools.
    
    Tracks failures per tool and controls whether requests
    are allowed based on the circuit state.
    
    Attributes:
        tool_name: Name of the tool this breaker protects
        threshold: Number of failures before opening the circuit
        cooldown: Seconds to wait before attempting recovery
    """
    tool_name: str
    threshold: int = 3
    cooldown: int = 30
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state with cooldown check."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if cooldown has passed
                if time.time() - self._last_failure_time >= self.cooldown:
                    self._state = CircuitState.HALF_OPEN
            return self._state
    
    def can_execute(self) -> bool:
        """Check if a request can be executed.
        
        Returns:
            True if circuit is closed or half-open, False if open
        """
        state = self.state
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
    
    def record_success(self) -> None:
        """Record a successful execution.
        
        Closes the circuit on success if it was half-open.
        Resets failure count.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
            self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed execution.
        
        Opens the circuit if threshold is exceeded.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.threshold:
                self._state = CircuitState.OPEN
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers per tool."""
    
    def __init__(self, threshold: int = 3, cooldown: int = 30):
        """Initialize the registry.
        
        Args:
            threshold: Number of failures before opening circuit
            cooldown: Seconds to wait before attempting recovery
        """
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._threshold = threshold
        self._cooldown = cooldown
        self._lock = threading.Lock()
    
    def get_breaker(self, tool_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            CircuitBreaker instance for the tool
        """
        with self._lock:
            if tool_name not in self._breakers:
                self._breakers[tool_name] = CircuitBreaker(
                    tool_name=tool_name,
                    threshold=self._threshold,
                    cooldown=self._cooldown
                )
            return self._breakers[tool_name]
    
    def can_execute(self, tool_name: str) -> bool:
        """Check if tool can be executed.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if circuit allows execution
        """
        return self.get_breaker(tool_name).can_execute()
    
    def record_success(self, tool_name: str) -> None:
        """Record successful execution for a tool.
        
        Args:
            tool_name: Name of the tool
        """
        self.get_breaker(tool_name).record_success()
    
    def record_failure(self, tool_name: str) -> None:
        """Record failed execution for a tool.
        
        Args:
            tool_name: Name of the tool
        """
        self.get_breaker(tool_name).record_failure()
    
    def get_status(self) -> Dict[str, Dict[str, any]]:
        """Get status of all circuit breakers.
        
        Returns:
            Dictionary mapping tool names to their status
        """
        with self._lock:
            return {
                name: {
                    "state": breaker.state.value,
                    "failure_count": breaker._failure_count,
                }
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

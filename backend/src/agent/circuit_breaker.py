import time
import threading
from enum import Enum
from typing import Dict
from dataclasses import dataclass, field


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    threshold: int = 3
    cooldown: int = 30


@dataclass
class CircuitBreaker:
    tool_name: str
    threshold: int = 3
    cooldown: int = 30
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.cooldown:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def can_execute(self) -> bool:
        state = self.state
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0


class CircuitBreakerRegistry:
    def __init__(self, threshold: int = 3, cooldown: int = 30):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._threshold = threshold
        self._cooldown = cooldown
        self._lock = threading.Lock()

    def get_breaker(self, tool_name: str) -> CircuitBreaker:
        with self._lock:
            if tool_name not in self._breakers:
                self._breakers[tool_name] = CircuitBreaker(
                    tool_name=tool_name,
                    threshold=self._threshold,
                    cooldown=self._cooldown,
                )
            return self._breakers[tool_name]

    def can_execute(self, tool_name: str) -> bool:
        return self.get_breaker(tool_name).can_execute()

    def record_success(self, tool_name: str) -> None:
        self.get_breaker(tool_name).record_success()

    def record_failure(self, tool_name: str) -> None:
        self.get_breaker(tool_name).record_failure()

    def get_status(self) -> Dict[str, Dict[str, any]]:
        with self._lock:
            return {
                name: {
                    "state": breaker.state.value,
                    "failure_count": breaker._failure_count,
                }
                for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> None:
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

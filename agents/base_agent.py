"""
Base Agent class — all fraud detection agents inherit from this.
Provides structured logging, error handling, and a standard run() interface.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent:
    """
    Abstract base for all agents in the fraud detection system.
    Subclasses must implement _execute(input_data) -> Any
    """

    def __init__(self, name: str):
        self.name = name
        self._logs: list[str] = []

    def log(self, msg: str):
        entry = f"[{self.name}] {msg}"
        self._logs.append(entry)
        print(entry)

    def get_logs(self) -> list[str]:
        return self._logs.copy()

    def run(self, input_data: Any) -> AgentResult:
        """
        Public entry point. Wraps _execute with timing + error handling.
        """
        start = time.time()
        try:
            self.log(f"Starting with input type: {type(input_data).__name__}")
            output = self._execute(input_data)
            duration = (time.time() - start) * 1000
            self.log(f"Completed in {duration:.1f}ms")
            return AgentResult(
                agent_name=self.name,
                success=True,
                output=output,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"ERROR: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                output=None,
                error=str(e),
                duration_ms=duration,
            )

    def _execute(self, input_data: Any) -> Any:
        raise NotImplementedError(f"{self.name} must implement _execute()")

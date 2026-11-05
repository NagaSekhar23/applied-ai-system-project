"""Controlled agent orchestration.

An agent plans, calls registered tools, inspects results, and answers — under
guardrails that guarantee termination. The registry is closed: the agent
cannot reach code that is not registered as a tool.
"""

from agents.controller import AgentConfig, AgentController, AgentRun, build_agent
from agents.planner import AgentDecision, Planner, PlanOutcome, rule_based_plan
from agents.tools import (
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
    UnknownToolError,
    default_registry,
)
from agents.trace import AgentTrace, Span

__all__ = [
    "AgentConfig",
    "AgentController",
    "AgentDecision",
    "AgentRun",
    "AgentTrace",
    "PlanOutcome",
    "Planner",
    "Span",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolValidationError",
    "UnknownToolError",
    "build_agent",
    "default_registry",
    "rule_based_plan",
]

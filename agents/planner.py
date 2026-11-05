"""Planning: deciding the next action.

The planner chooses between calling one registered tool and answering. It is
the only place the model influences control flow, and it does so through a
constrained schema: a tool *name* from the registry plus arguments that are
validated before anything runs.

As with the parser, there is a deterministic planner underneath, so the agent
works offline and degrades rather than breaking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from agents.tools import ToolRegistry
from game import GameState
from natural_language.parser import find_difficulty, find_number
from prompts import get_prompt
from providers import CallStats, LLMRequest, ReliableLLMClient
from pydantic import BaseModel, ConfigDict, Field

Action = Literal["call_tool", "respond", "clarify"]


class AgentDecision(BaseModel):
    """One planning decision.

    ``tool`` is checked against the registry before use; a name outside it is
    rejected and reported back to the planner as a recoverable error.
    """

    model_config = ConfigDict(extra="ignore")

    action: Action
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    response: str | None = Field(default=None, max_length=800)
    reasoning: str = Field(default="", max_length=400)

    @staticmethod
    def json_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["call_tool", "respond", "clarify"]},
                "tool": {"type": ["string", "null"]},
                "arguments": {"type": "object"},
                "response": {"type": ["string", "null"]},
                "reasoning": {"type": "string"},
            },
            "required": ["action", "tool", "arguments", "response", "reasoning"],
            "additionalProperties": False,
        }


@dataclass
class PlanOutcome:
    """A decision plus how it was reached."""

    decision: AgentDecision
    source: str  # "llm" | "rules"
    stats: CallStats | None = None
    fallback_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


WANTS_RULES = re.compile(r"\brules?\b|\bhow (does|do) (it|this|the game) work\b|\bexplain\b")
WANTS_HINT = re.compile(r"\bhints?\b|\bclue\b|\bstuck\b|\bwhat should i\b|\bhelp\b")
WANTS_STATUS = re.compile(r"\bstatus\b|\bscore\b|\bhow many\b|\battempts?\b|\btries\b|\bdoing\b")
WANTS_START = re.compile(r"\bstart\b|\bnew game\b|\bbegin\b|\bplay\b")
WANTS_RESET = re.compile(r"\breset\b|\bstart over\b|\bgive up\b")
WANTS_DIFFICULTY = re.compile(r"\bdifficulty\b|\bswitch to\b|\bchange to\b|\bset to\b")
WANTS_GUESS = re.compile(r"\bguess\b|\btry\b|\bis it\b|\bnumber is\b")


def rule_based_plan(
    message: str,
    state: GameState | None,
    completed: list[str],
    registry: ToolRegistry,
) -> AgentDecision:
    """Deterministic planner.

    Multi-step requests are handled by consulting ``completed``: each pass
    picks the first still-outstanding sub-task, so "start a hard game and
    explain the rules" becomes start_game, then explain_rules, then respond.
    """
    text = (message or "").strip().lower()
    if not text:
        return AgentDecision(
            action="clarify",
            response="What would you like to do? You can start a game, make a guess, or ask for a hint.",
            reasoning="empty request",
        )

    difficulty = find_difficulty(text)
    wants_start = bool(WANTS_START.search(text))
    wants_reset = bool(WANTS_RESET.search(text))
    wants_rules = bool(WANTS_RULES.search(text))
    wants_hint = bool(WANTS_HINT.search(text)) and not wants_rules
    wants_status = bool(WANTS_STATUS.search(text))
    wants_difficulty_change = bool(difficulty and WANTS_DIFFICULTY.search(text))

    number = find_number(text)
    wants_guess = number is not None and bool(WANTS_GUESS.search(text) or not any(
        [wants_start, wants_reset, wants_rules, wants_hint, wants_status, wants_difficulty_change]
    ))

    # Each branch fires only once per run, which is what makes multi-step
    # requests terminate instead of looping on the same tool.
    if wants_reset and "reset_game" not in completed:
        return AgentDecision(action="call_tool", tool="reset_game", reasoning="user asked to reset")

    if wants_difficulty_change and "change_difficulty" not in completed:
        return AgentDecision(
            action="call_tool",
            tool="change_difficulty",
            arguments={"difficulty": difficulty},
            reasoning="user asked to change difficulty",
        )

    if wants_start and "start_game" not in completed:
        return AgentDecision(
            action="call_tool",
            tool="start_game",
            arguments={"difficulty": difficulty} if difficulty else {},
            reasoning="user asked to start a game",
        )

    if wants_rules and "explain_rules" not in completed:
        return AgentDecision(action="call_tool", tool="explain_rules", reasoning="user asked about rules")

    if wants_guess and "submit_guess" not in completed:
        return AgentDecision(
            action="call_tool",
            tool="submit_guess",
            arguments={"guess": number},
            reasoning="user made a guess",
        )

    if wants_hint and "request_hint" not in completed:
        return AgentDecision(action="call_tool", tool="request_hint", reasoning="user asked for a hint")

    if wants_status and "get_game_status" not in completed:
        return AgentDecision(
            action="call_tool", tool="get_game_status", reasoning="user asked about progress"
        )

    if completed:
        return AgentDecision(action="respond", reasoning="all requested steps are done")

    return AgentDecision(
        action="clarify",
        response=(
            "I can start a game, submit a guess, give you a hint, change difficulty, "
            "explain the rules, or tell you your status. Which would you like?"
        ),
        reasoning="no recognised intent",
    )


class Planner:
    """Chooses the next action, via the LLM when available."""

    def __init__(
        self,
        registry: ToolRegistry,
        client: ReliableLLMClient | None = None,
        max_tokens: int = 512,
        effort: str = "low",
    ):
        self.registry = registry
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort

    def plan(
        self,
        message: str,
        state: GameState | None,
        completed: list[str],
        observations: list[str],
    ) -> PlanOutcome:
        """Decide the next action.

        Falls back to the rule-based planner whenever the model is absent,
        failing, or names a tool that is not registered.
        """
        if self.client is None:
            return PlanOutcome(
                rule_based_plan(message, state, completed, self.registry),
                source="rules",
                fallback_reason="no LLM client configured",
            )

        prompt = get_prompt("agent_plan")
        facts = {
            "available_tools": self.registry.schemas(),
            "game_in_progress": state is not None,
            "game": {
                "difficulty": state.difficulty,
                "low": state.low,
                "high": state.high,
                "attempts_left": state.attempts_left,
                "status": state.status,
                "score": state.score,
            }
            if state
            else None,
            "tools_already_called": completed,
            "observations_so_far": observations,
        }

        result = self.client.complete_structured(
            LLMRequest(
                system=prompt.system,
                user=prompt.render_user(facts, message),
                json_schema=AgentDecision.json_schema(),
                max_tokens=self.max_tokens,
                effort=self.effort,
                purpose="agent_plan",
            ),
            AgentDecision,
        )

        if not result.ok or result.value is None:
            reason = result.stats.errors[-1] if result.stats.errors else "no response"
            return PlanOutcome(
                rule_based_plan(message, state, completed, self.registry),
                source="rules",
                stats=result.stats,
                fallback_reason=f"planner fell back: {reason}",
            )

        decision = result.value
        warnings: list[str] = []

        # A tool name outside the registry is never executed.
        if decision.action == "call_tool" and decision.tool not in self.registry:
            warnings.append(
                f"planner named unregistered tool {decision.tool!r}; falling back to rules"
            )
            return PlanOutcome(
                rule_based_plan(message, state, completed, self.registry),
                source="rules",
                stats=result.stats,
                fallback_reason=f"unregistered tool {decision.tool!r}",
                warnings=warnings,
            )

        return PlanOutcome(decision, source="llm", stats=result.stats, warnings=warnings)

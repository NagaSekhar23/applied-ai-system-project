"""Tool registry.

A tool is an explicitly registered capability with a typed input schema, a
typed output schema and an execution function. The agent can call registered
tools and nothing else: there is no dynamic lookup by string into module
globals, no ``eval``, and no filesystem or shell access anywhere in this file.

Every execution function delegates to an existing deterministic service. Tools
are an interface, not a second implementation of the game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant import Assistant, deterministic_hint
from game import GameState, new_game, submit_guess
from logic_utils import DIFFICULTY_NAMES, get_difficulty_config


class ToolError(Exception):
    """A tool could not run. Recoverable: reported back to the planner."""


class UnknownToolError(ToolError):
    """The agent asked for a tool that is not registered."""


class ToolValidationError(ToolError):
    """The arguments did not match the tool's input schema."""


@dataclass
class ToolContext:
    """Everything a tool may touch.

    Tools receive this instead of global state, so a tool cannot reach
    anything the agent was not given.
    """

    state: GameState | None = None
    last_outcome: str | None = None
    assistant: Assistant | None = None


@dataclass
class ToolResult:
    """The outcome of one tool call."""

    tool: str
    ok: bool
    output: dict[str, Any]
    message: str
    new_state: GameState | None = None
    new_last_outcome: str | None = None
    error: str | None = None


# --- Input schemas -----------------------------------------------------------


class NoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DifficultyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    difficulty: str = Field(description="Easy, Normal or Hard")

    def normalised(self) -> str:
        cleaned = self.difficulty.strip().capitalize()
        if cleaned not in DIFFICULTY_NAMES:
            raise ValueError(f"difficulty must be one of {DIFFICULTY_NAMES}")
        return cleaned


class OptionalDifficultyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    difficulty: str | None = Field(default=None, description="Easy, Normal or Hard")

    def normalised(self) -> str | None:
        if self.difficulty is None:
            return None
        cleaned = self.difficulty.strip().capitalize()
        if cleaned not in DIFFICULTY_NAMES:
            raise ValueError(f"difficulty must be one of {DIFFICULTY_NAMES}")
        return cleaned


class GuessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    guess: int = Field(description="The number to guess")


class HintInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(default=None, max_length=300)


# --- Output schemas ----------------------------------------------------------


class GameSnapshot(BaseModel):
    """Public game state. Never carries the secret while a game is live."""

    model_config = ConfigDict(extra="forbid")

    difficulty: str
    low: int
    high: int
    attempts_used: int
    attempts_left: int
    max_attempts: int
    score: int
    status: str
    secret: int | None = None


class GuessOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: str
    hint: str
    snapshot: GameSnapshot


class HintOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hint: str
    source: str


def snapshot_of(state: GameState) -> GameSnapshot:
    """Build a public snapshot, redacting the secret until the game ends."""
    return GameSnapshot(
        difficulty=state.difficulty,
        low=state.low,
        high=state.high,
        attempts_used=state.attempts_used,
        attempts_left=state.attempts_left,
        max_attempts=state.max_attempts,
        score=state.score,
        status=state.status,
        secret=state.secret if state.status != "playing" else None,
    )


@dataclass
class Tool:
    """One registered capability."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    run: Callable[[BaseModel, ToolContext], ToolResult]
    requires_active_game: bool = False
    mutates_state: bool = False

    def schema(self) -> dict[str, Any]:
        """Description the planner sees. No implementation detail leaks."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "mutates_state": self.mutates_state,
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> BaseModel:
        """Parse arguments against the input schema.

        Raises:
            ToolValidationError: with a readable message the planner can act on.
        """
        try:
            return self.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            raise ToolValidationError(
                f"Invalid arguments for {self.name}: {exc.error_count()} problem(s): "
                + "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            ) from exc

    def validate_output(self, result: ToolResult) -> None:
        """Check the tool's own output against its declared schema.

        Guards against a tool drifting from its contract — a bug this would
        otherwise surface only as confusing agent behavior.
        """
        if not result.ok:
            return
        try:
            self.output_model.model_validate(result.output)
        except ValidationError as exc:
            raise ToolError(f"{self.name} returned output that violates its schema: {exc}") from exc


# --- Tool implementations ----------------------------------------------------


def _start_game(args: OptionalDifficultyInput, ctx: ToolContext) -> ToolResult:
    difficulty = args.normalised() or (ctx.state.difficulty if ctx.state else "Normal")
    fresh = new_game(difficulty)
    return ToolResult(
        tool="start_game",
        ok=True,
        output=snapshot_of(fresh).model_dump(),
        message=(
            f"Started a new {difficulty} game: guess between {fresh.low} and "
            f"{fresh.high}, {fresh.max_attempts} attempts."
        ),
        new_state=fresh,
        new_last_outcome=None,
    )


def _reset_game(args: NoInput, ctx: ToolContext) -> ToolResult:
    difficulty = ctx.state.difficulty if ctx.state else "Normal"
    fresh = new_game(difficulty)
    return ToolResult(
        tool="reset_game",
        ok=True,
        output=snapshot_of(fresh).model_dump(),
        message=f"Reset. A fresh {difficulty} game is ready.",
        new_state=fresh,
        new_last_outcome=None,
    )


def _change_difficulty(args: DifficultyInput, ctx: ToolContext) -> ToolResult:
    difficulty = args.normalised()
    fresh = new_game(difficulty)
    return ToolResult(
        tool="change_difficulty",
        ok=True,
        output=snapshot_of(fresh).model_dump(),
        message=(
            f"Difficulty is now {difficulty}, which starts a fresh game: "
            f"{fresh.low} to {fresh.high}, {fresh.max_attempts} attempts."
        ),
        new_state=fresh,
        new_last_outcome=None,
    )


def _submit_guess(args: GuessInput, ctx: ToolContext) -> ToolResult:
    if ctx.state is None:
        return ToolResult(
            tool="submit_guess",
            ok=False,
            output={},
            message="There is no game in progress.",
            error="no active game",
        )

    result = submit_guess(ctx.state, str(args.guess))
    if result.error:
        return ToolResult(
            tool="submit_guess",
            ok=False,
            output={},
            message=result.error,
            error=result.error,
            new_state=ctx.state,
        )

    state = result.state
    return ToolResult(
        tool="submit_guess",
        ok=True,
        output=GuessOutput(
            outcome=result.outcome or "",
            hint=result.hint or "",
            snapshot=snapshot_of(state),
        ).model_dump(),
        message=f"{result.hint} Attempts left: {state.attempts_left}. Score: {state.score}.",
        new_state=state,
        new_last_outcome=result.outcome,
    )


def _get_game_status(args: NoInput, ctx: ToolContext) -> ToolResult:
    if ctx.state is None:
        return ToolResult(
            tool="get_game_status",
            ok=False,
            output={},
            message="There is no game in progress.",
            error="no active game",
        )

    state = ctx.state
    snapshot = snapshot_of(state)
    if state.status == "playing":
        message = (
            f"{state.difficulty} game, range {state.low}-{state.high}. "
            f"Attempts left: {state.attempts_left} of {state.max_attempts}. "
            f"Score: {state.score}."
        )
    else:
        message = (
            f"Game over ({state.status}). The secret was {state.secret}. "
            f"Final score: {state.score}."
        )

    return ToolResult(
        tool="get_game_status",
        ok=True,
        output=snapshot.model_dump(),
        message=message,
        new_state=state,
    )


def _request_hint(args: HintInput, ctx: ToolContext) -> ToolResult:
    if ctx.state is None:
        return ToolResult(
            tool="request_hint",
            ok=False,
            output={},
            message="There is no game in progress.",
            error="no active game",
        )

    if ctx.assistant is not None:
        reply = ctx.assistant.hint(
            ctx.state, last_outcome=ctx.last_outcome, question=args.question or ""
        )
        hint_text, source = reply.message, reply.source
    else:
        hint_text, _ = deterministic_hint(ctx.state, ctx.last_outcome)
        source = "fallback"

    return ToolResult(
        tool="request_hint",
        ok=True,
        output=HintOutput(hint=hint_text, source=source).model_dump(),
        message=hint_text,
        new_state=ctx.state,
    )


def _explain_rules(args: NoInput, ctx: ToolContext) -> ToolResult:
    difficulty = ctx.state.difficulty if ctx.state else "Normal"
    config = get_difficulty_config(difficulty)
    message = (
        f"On {difficulty} you guess a number between {config.low} and {config.high}, "
        f"with {config.max_attempts} attempts. Each wrong guess tells you whether to go "
        "higher or lower and costs 5 points, though your score never drops below zero. "
        "Winning scores max(10, 100 - 10 x guess number). Invalid or out-of-range input "
        "does not use up an attempt."
    )
    return ToolResult(
        tool="explain_rules",
        ok=True,
        output={"rules": message, "difficulty": difficulty},
        message=message,
        new_state=ctx.state,
    )


class RulesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: str
    difficulty: str


# --- Registry ----------------------------------------------------------------


class ToolRegistry:
    """The set of tools an agent may call. Nothing outside it is reachable."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(
                f"No tool named {name!r}. Available tools: {sorted(self._tools)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool descriptions for the planner's prompt."""
        return [self._tools[name].schema() for name in self.names]


def default_registry() -> ToolRegistry:
    """The six game tools, plus a rules explainer."""
    return ToolRegistry(
        [
            Tool(
                name="start_game",
                description="Start a brand new game, optionally at a given difficulty.",
                input_model=OptionalDifficultyInput,
                output_model=GameSnapshot,
                run=_start_game,
                mutates_state=True,
            ),
            Tool(
                name="submit_guess",
                description="Submit a numeric guess for the current game.",
                input_model=GuessInput,
                output_model=GuessOutput,
                run=_submit_guess,
                requires_active_game=True,
                mutates_state=True,
            ),
            Tool(
                name="request_hint",
                description="Get a hint about what to guess next. Does not use an attempt.",
                input_model=HintInput,
                output_model=HintOutput,
                run=_request_hint,
                requires_active_game=True,
            ),
            Tool(
                name="get_game_status",
                description="Read the current score, attempts left, range and status.",
                input_model=NoInput,
                output_model=GameSnapshot,
                run=_get_game_status,
                requires_active_game=True,
            ),
            Tool(
                name="change_difficulty",
                description="Change difficulty. This starts a fresh game at the new level.",
                input_model=DifficultyInput,
                output_model=GameSnapshot,
                run=_change_difficulty,
                mutates_state=True,
            ),
            Tool(
                name="reset_game",
                description="Abandon the current game and start over at the same difficulty.",
                input_model=NoInput,
                output_model=GameSnapshot,
                run=_reset_game,
                mutates_state=True,
            ),
            Tool(
                name="explain_rules",
                description="Explain the rules and scoring of the game.",
                input_model=NoInput,
                output_model=RulesOutput,
                run=_explain_rules,
            ),
        ]
    )

"""Command execution.

The executor is a dispatch table over a closed enum. Every branch calls an
existing deterministic service — ``new_game``, ``submit_guess``, the assistant —
and nothing here re-implements a rule.

There is no path from model output to arbitrary code: the model can only
produce a ``CommandType``, and a ``CommandType`` can only reach one of the
handlers registered below.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from assistant import Assistant, AssistantReply
from game import GameState, new_game, submit_guess
from natural_language.commands import Command, CommandType, ValidatedCommand
from natural_language.validator import validate


@dataclass
class ExecutionResult:
    """What executing a command produced."""

    message: str
    command: Command
    state: GameState | None
    executed: bool
    rejection_reason: str | None = None
    assistant_reply: AssistantReply | None = None
    last_outcome: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.executed


class CommandExecutor:
    """Runs validated commands against deterministic game services."""

    def __init__(self, assistant: Assistant | None = None):
        self.assistant = assistant

    def execute(
        self,
        validated: ValidatedCommand,
        state: GameState | None,
        last_outcome: str | None = None,
    ) -> ExecutionResult:
        """Execute a command that has already passed validation.

        A command that failed validation is never dispatched; its reason is
        returned instead.
        """
        command = validated.command

        if not validated.ok:
            return ExecutionResult(
                message=validated.summary,
                command=command,
                state=state,
                executed=False,
                rejection_reason=validated.summary,
                last_outcome=last_outcome,
            )

        handler = self._handlers().get(command.command)
        if handler is None:  # pragma: no cover - the enum is closed
            return ExecutionResult(
                message=f"No handler for {command.command.value}.",
                command=command,
                state=state,
                executed=False,
                rejection_reason="unhandled command",
            )
        return handler(command, state, last_outcome)

    def run(
        self,
        command: Command,
        state: GameState | None,
        last_outcome: str | None = None,
    ) -> ExecutionResult:
        """Validate, then execute. The normal entry point."""
        return self.execute(validate(command, state), state, last_outcome)

    def _handlers(self):
        return {
            CommandType.START_GAME: self._start_game,
            CommandType.RESET_GAME: self._reset_game,
            CommandType.SUBMIT_GUESS: self._submit_guess,
            CommandType.CHANGE_DIFFICULTY: self._change_difficulty,
            CommandType.GET_GAME_STATUS: self._status,
            CommandType.REQUEST_HINT: self._hint,
        }

    # --- Handlers ------------------------------------------------------------

    def _start_game(self, command, state, last_outcome):
        difficulty = command.difficulty or (state.difficulty if state else "Normal")
        fresh = new_game(difficulty)
        return ExecutionResult(
            message=(
                f"Started a new {difficulty} game. Guess a number between "
                f"{fresh.low} and {fresh.high}. You have {fresh.max_attempts} attempts."
            ),
            command=command,
            state=fresh,
            executed=True,
            last_outcome=None,
        )

    def _reset_game(self, command, state, last_outcome):
        difficulty = command.difficulty or (state.difficulty if state else "Normal")
        fresh = new_game(difficulty)
        return ExecutionResult(
            message=f"Game reset. A fresh {difficulty} game is ready.",
            command=command,
            state=fresh,
            executed=True,
            last_outcome=None,
        )

    def _change_difficulty(self, command, state, last_outcome):
        fresh = new_game(command.difficulty)
        return ExecutionResult(
            message=(
                f"Difficulty is now {command.difficulty}. That starts a fresh game: "
                f"guess between {fresh.low} and {fresh.high}, {fresh.max_attempts} attempts."
            ),
            command=command,
            state=fresh,
            executed=True,
            last_outcome=None,
        )

    def _submit_guess(self, command, state, last_outcome):
        result = submit_guess(state, str(command.guess))

        if result.error:
            # Validation already ran, so this is a defensive second line:
            # the game service remains the final authority on a guess.
            return ExecutionResult(
                message=result.error,
                command=command,
                state=result.state,
                executed=False,
                rejection_reason=result.error,
                last_outcome=last_outcome,
            )

        new_state = result.state
        if new_state.status == "won":
            message = (
                f"{result.hint} You won. The secret was {new_state.secret}. "
                f"Final score: {new_state.score}."
            )
        elif new_state.status == "lost":
            message = (
                f"{result.hint} Out of attempts. The secret was {new_state.secret}. "
                f"Score: {new_state.score}."
            )
        else:
            message = (
                f"{result.hint} Attempts left: {new_state.attempts_left}. "
                f"Score: {new_state.score}."
            )

        return ExecutionResult(
            message=message,
            command=command,
            state=new_state,
            executed=True,
            last_outcome=result.outcome,
            details={"outcome": result.outcome, "guess": command.guess},
        )

    def _status(self, command, state, last_outcome):
        if state is None:
            return ExecutionResult(
                message="No game in progress. Say 'start a game' to begin.",
                command=command,
                state=None,
                executed=True,
                last_outcome=last_outcome,
            )

        if state.status == "playing":
            message = (
                f"{state.difficulty} game, range {state.low}-{state.high}. "
                f"Attempts used: {state.attempts_used} of {state.max_attempts}. "
                f"Attempts left: {state.attempts_left}. Score: {state.score}."
            )
        else:
            message = (
                f"This game is over ({state.status}). The secret was {state.secret}. "
                f"Final score: {state.score}."
            )
        if state.history:
            message += f" Guesses so far: {', '.join(str(g) for g in state.history)}."

        return ExecutionResult(
            message=message,
            command=command,
            state=state,
            executed=True,
            last_outcome=last_outcome,
            details={"attempts_left": state.attempts_left, "score": state.score},
        )

    def _hint(self, command, state, last_outcome):
        if state is None:
            return ExecutionResult(
                message="No game in progress. Say 'start a game' and I will help you play.",
                command=command,
                state=None,
                executed=True,
                last_outcome=last_outcome,
            )

        if self.assistant is None:
            from assistant import deterministic_hint

            message, _ = deterministic_hint(state, last_outcome)
            return ExecutionResult(
                message=message,
                command=command,
                state=state,
                executed=True,
                last_outcome=last_outcome,
            )

        reply = self.assistant.hint(
            state, last_outcome=last_outcome, question=command.question or ""
        )
        return ExecutionResult(
            message=reply.message,
            command=command,
            state=state,
            executed=True,
            assistant_reply=reply,
            last_outcome=last_outcome,
            details={"hint_source": reply.source},
        )


@dataclass
class ConversationTurn:
    """One complete pass through the pipeline, for display and logging."""

    user_message: str
    parse: Any  # ParseOutcome
    validated: ValidatedCommand
    result: ExecutionResult

    @property
    def command_type(self) -> str:
        return self.result.command.command.value


def handle_message(
    message: str,
    state: GameState | None,
    parser,
    executor: CommandExecutor,
    last_outcome: str | None = None,
) -> ConversationTurn:
    """Run the full pipeline: parse → validate → execute.

    This is the only function the UI needs. Each stage is separately testable,
    and the stages cannot be reordered or skipped.
    """
    parse_outcome = parser.parse(message, context=_context_for(state))
    validated = validate(parse_outcome.command, state)
    result = executor.execute(validated, state, last_outcome)
    return ConversationTurn(
        user_message=message,
        parse=parse_outcome,
        validated=validated,
        result=result,
    )


def _context_for(state: GameState | None) -> dict[str, Any]:
    """Redacted state summary for the parser's prompt. Never includes the secret."""
    if state is None:
        return {"game_in_progress": False}
    return {
        "game_in_progress": True,
        "difficulty": state.difficulty,
        "low": state.low,
        "high": state.high,
        "attempts_left": state.attempts_left,
        "status": state.status,
    }

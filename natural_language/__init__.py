"""Natural language interface.

Pipeline, in strict order:

    message -> CommandParser -> Command -> validate() -> CommandExecutor -> result

The parser interprets, the validator checks, the executor calls deterministic
game services. No stage may be skipped, and the executor dispatches only over a
closed enum of command types.
"""

from natural_language.commands import (
    Command,
    CommandType,
    ValidatedCommand,
    ValidationIssue,
)
from natural_language.executor import (
    CommandExecutor,
    ConversationTurn,
    ExecutionResult,
    handle_message,
)
from natural_language.parser import CommandParser, ParseOutcome, rule_based_parse
from natural_language.validator import validate

__all__ = [
    "Command",
    "CommandExecutor",
    "CommandParser",
    "CommandType",
    "ConversationTurn",
    "ExecutionResult",
    "ParseOutcome",
    "ValidatedCommand",
    "ValidationIssue",
    "handle_message",
    "rule_based_parse",
    "validate",
]

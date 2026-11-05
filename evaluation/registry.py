"""The registry of every executor the CLI can run.

Kept in its own module so ``evaluation.executors`` stays free of imports from
the LLM, agent and retrieval layers. Each phase adds one entry here.
"""

from __future__ import annotations

from evaluation.executors import Executor, default_registry


def full_registry() -> dict[str, Executor]:
    """Every registered executor across all implemented phases.

    Optional layers are added defensively: if a phase's dependencies are not
    installed, its executors are simply absent and a case that targets one
    fails with a clear "no executor registered" error rather than preventing
    every other dataset from running.
    """
    registry = default_registry()

    try:
        from evaluation.llm_executors import LLM_EXECUTORS
    except ImportError:  # pragma: no cover - only when pydantic is missing
        pass
    else:
        registry.update(LLM_EXECUTORS)

    try:
        from evaluation.nl_executors import NL_EXECUTORS
    except ImportError:  # pragma: no cover
        pass
    else:
        registry.update(NL_EXECUTORS)

    try:
        from evaluation.agent_executors import AGENT_EXECUTORS
    except ImportError:  # pragma: no cover
        pass
    else:
        registry.update(AGENT_EXECUTORS)

    return registry

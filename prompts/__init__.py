"""Versioned prompt library.

Prompts are data, kept out of business logic so they can be reviewed, diffed
and versioned independently. Each prompt carries an explicit version string;
when a prompt changes meaningfully the version changes with it, so an
evaluation report can record which prompt produced a result.
"""

from prompts.library import (
    PROMPTS,
    Prompt,
    get_prompt,
    render_facts,
)

__all__ = ["PROMPTS", "Prompt", "get_prompt", "render_facts"]

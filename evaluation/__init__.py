"""Reusable evaluation framework.

Phase 2 evaluates deterministic game behavior. Later phases register additional
executors (LLM, natural language, agent, retrieval) against the same runner,
dataset format and metrics.

Nothing in this package imports Streamlit.
"""

from evaluation.cases import Dataset, EvalCase, load_dataset
from evaluation.metrics import EvaluationMetrics, compute_metrics
from evaluation.runner import CaseResult, EvaluationRun, run_dataset

__all__ = [
    "CaseResult",
    "Dataset",
    "EvalCase",
    "EvaluationMetrics",
    "EvaluationRun",
    "compute_metrics",
    "load_dataset",
    "run_dataset",
]

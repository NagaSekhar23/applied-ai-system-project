"""Grounded answer generation.

The rule: an answer is only shown if it is supported by retrieved evidence and
every citation checks out. Three gates, in order:

1. **Retrieval gate.** If nothing clears the evidence threshold, the system
   abstains without calling the model at all — cheaper, and it removes any
   chance of an unsupported answer.
2. **Schema gate.** The model's reply is validated against a Pydantic model.
3. **Citation gate.** Every ``[n]`` must map to a chunk that was retrieved. An
   answer with a fabricated citation is rejected, not footnoted.

Abstention is a first-class outcome, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompts import get_prompt
from providers import CallStats, LLMRequest, ReliableLLMClient
from pydantic import BaseModel, ConfigDict, Field
from retrieval.citations import CitationCheck, format_evidence, validate_citations
from retrieval.reranking import RerankedChunk

# Below this reranker score, the best passage is treated as too weak to answer
# from. Tuned against the evaluation set; raising it trades recall for safety.
MIN_EVIDENCE_SCORE = 0.25

ABSTAIN_MESSAGE = (
    "I could not find enough in the project documentation to answer that "
    "confidently. Try rephrasing, or ask about the architecture, evaluation, "
    "the agent, observability, or the game rules."
)


class GroundedAnswer(BaseModel):
    """The validated shape of a model answer."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1, max_length=1500)
    sufficient_evidence: bool = Field(
        description="The model's own claim that the evidence supported an answer."
    )

    @staticmethod
    def json_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "sufficient_evidence": {"type": "boolean"},
            },
            "required": ["answer", "sufficient_evidence"],
            "additionalProperties": False,
        }


@dataclass
class RagAnswer:
    """The full result of one question."""

    question: str
    answer: str
    abstained: bool
    reason: str | None = None
    evidence: list[RerankedChunk] = field(default_factory=list)
    citation_check: CitationCheck | None = None
    stats: CallStats | None = None
    prompt_id: str | None = None
    source: str = "llm"
    retrieval_ms: float = 0.0

    @property
    def grounded(self) -> bool:
        """True when an answer was given and its citations all verified."""
        return (
            not self.abstained
            and self.citation_check is not None
            and self.citation_check.valid
        )

    @property
    def citations(self):
        return self.citation_check.citations if self.citation_check else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "abstained": self.abstained,
            "grounded": self.grounded,
            "reason": self.reason,
            "evidence_count": len(self.evidence),
            "citations": [c.to_dict() for c in self.citations],
            "citation_check": self.citation_check.to_dict() if self.citation_check else None,
            "source": self.source,
            "retrieval_ms": round(self.retrieval_ms, 2),
        }


class AnswerGenerator:
    """Turns retrieved evidence into a grounded, cited answer."""

    def __init__(
        self,
        client: ReliableLLMClient | None = None,
        min_evidence_score: float = MIN_EVIDENCE_SCORE,
        max_tokens: int = 1024,
        effort: str = "low",
    ):
        self.client = client
        self.min_evidence_score = min_evidence_score
        self.max_tokens = max_tokens
        self.effort = effort

    def answer(self, question: str, evidence: list[RerankedChunk]) -> RagAnswer:
        """Generate an answer, or abstain."""
        # Gate 1: is there anything worth answering from?
        if not evidence:
            return RagAnswer(
                question=question,
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                reason="no documents matched the question",
                source="abstain",
            )

        best = max(item.score for item in evidence)
        if best < self.min_evidence_score:
            return RagAnswer(
                question=question,
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                reason=(
                    f"best evidence scored {best:.2f}, below the "
                    f"{self.min_evidence_score:.2f} threshold"
                ),
                evidence=evidence,
                source="abstain",
            )

        if self.client is None:
            return RagAnswer(
                question=question,
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                reason="no LLM client configured for answer generation",
                evidence=evidence,
                source="abstain",
            )

        prompt = get_prompt("rag_answer")
        facts = {
            "question": question,
            "evidence_count": len(evidence),
            "sources": sorted({item.chunk.source_path for item in evidence}),
        }
        user = (
            f"{prompt.render_user(facts, question)}\n\n"
            f"Evidence:\n{format_evidence(evidence)}"
        )

        result = self.client.complete_structured(
            LLMRequest(
                system=prompt.system,
                user=user,
                json_schema=GroundedAnswer.json_schema(),
                max_tokens=self.max_tokens,
                effort=self.effort,
                purpose="rag_answer",
            ),
            GroundedAnswer,
        )

        # Gate 2: schema.
        if not result.ok or result.value is None:
            reason = result.stats.errors[-1] if result.stats.errors else "no response"
            return RagAnswer(
                question=question,
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                reason=f"answer generation failed: {reason}",
                evidence=evidence,
                stats=result.stats,
                source="abstain",
            )

        generated: GroundedAnswer = result.value

        # The model may decide the evidence was insufficient. Honoured.
        if not generated.sufficient_evidence:
            return RagAnswer(
                question=question,
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                reason="the model judged the retrieved evidence insufficient",
                evidence=evidence,
                stats=result.stats,
                prompt_id=prompt.id,
                source="abstain",
            )

        # Gate 3: citations.
        check = validate_citations(generated.answer, evidence)
        if not check.valid:
            return RagAnswer(
                question=question,
                answer=ABSTAIN_MESSAGE,
                abstained=True,
                reason=f"citation validation failed — {check.summary}",
                evidence=evidence,
                citation_check=check,
                stats=result.stats,
                prompt_id=prompt.id,
                source="abstain",
            )

        return RagAnswer(
            question=question,
            answer=generated.answer,
            abstained=False,
            evidence=evidence,
            citation_check=check,
            stats=result.stats,
            prompt_id=prompt.id,
            source="llm",
        )

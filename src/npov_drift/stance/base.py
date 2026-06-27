"""Core stance types and the classifier interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class StanceLabel(str, Enum):
    FAVORABLE = "favorable"
    CRITICAL = "critical"
    NEUTRAL = "neutral"


@dataclass
class StanceResult:
    """The stance of one sentence toward the article topic.

    ``scores`` holds the raw per-label signal from the underlying model (e.g.
    NLI entailment probabilities for the favorable/critical hypotheses), so the
    decision is auditable rather than a black-box label.
    """

    sentence: str
    label: StanceLabel
    scores: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class StanceClassifier(Protocol):
    """Anything that maps sentences -> stance toward a topic.

    Implementations: ``nli.NLIStanceClassifier`` (real, CPU NLI model) and
    ``stub.KeywordStanceStub`` (deterministic, for tests / offline fallback).
    """

    def classify(self, sentences: list[str], topic: str) -> list[StanceResult]:
        ...

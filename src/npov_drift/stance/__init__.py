"""Stance-toward-topic classification and self-gating.

This package classifies each sentence's stance TOWARD THE ARTICLE'S TOPIC into
{favorable, critical, neutral} and decides whether the stance signal is even
active for an article (the self-gating inactivity test).

Critical design point (WMF caution): stance is toward the TOPIC, not raw
sentiment. "The drug can cause nausea in some patients" is a *neutral
description of a negative fact*, not a *critical stance*; only language that
actually argues for/against the topic counts as favorable/critical.

The heavy NLI model lives in ``nli`` and is imported explicitly so that this
package (and the test suite) loads with no torch/transformers dependency.
"""

from .aggregate import (
    StanceDistribution,
    is_stance_active,
    stance_distribution,
)
from .base import StanceClassifier, StanceLabel, StanceResult
from .sentences import split_sentences
from .stub import KeywordStanceStub
from .topic import topic_from_title

__all__ = [
    "StanceLabel",
    "StanceResult",
    "StanceClassifier",
    "split_sentences",
    "topic_from_title",
    "KeywordStanceStub",
    "StanceDistribution",
    "stance_distribution",
    "is_stance_active",
]

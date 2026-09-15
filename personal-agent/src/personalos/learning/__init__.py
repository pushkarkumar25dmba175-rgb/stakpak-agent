"""Learning: pattern detection, workflow discovery, feedback and skill generation."""

from personalos.learning.feedback_learner import FeedbackLearner, LearnedItem
from personalos.learning.pattern_detector import (
    DetectedPattern,
    cluster_signatures,
    find_repeated,
)
from personalos.learning.skill_generator import generate_skill, suggest_skill_name
from personalos.learning.suggestions import SuggestionDraft, SuggestionStore
from personalos.learning.workflow_detector import WorkflowDetector, WorkflowObservation

__all__ = [
    "DetectedPattern",
    "FeedbackLearner",
    "LearnedItem",
    "SuggestionDraft",
    "SuggestionStore",
    "WorkflowDetector",
    "WorkflowObservation",
    "cluster_signatures",
    "find_repeated",
    "generate_skill",
    "suggest_skill_name",
]

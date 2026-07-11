"""Domain-specific benchmark scoring."""

from small_models_society.scoring.knowledge import KnowledgeScore, score_knowledge
from small_models_society.scoring.logic import score_choice
from small_models_society.scoring.math import score_math

__all__ = ["KnowledgeScore", "score_choice", "score_knowledge", "score_math"]

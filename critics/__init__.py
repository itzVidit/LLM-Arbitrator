"""
critics/__init__.py
-------------------
Convenience imports for all critic agents.
"""

from critics.base import BaseCritic
from critics.accuracy import AccuracyCritic
from critics.logic import LogicCritic
from critics.completeness import CompletenessCritic
from critics.safety import SafetyCritic
from critics.style import StyleCritic

# Registry maps CriticDimension → critic class, useful for the orchestrator
from models.critique import CriticDimension

CRITIC_REGISTRY: dict[CriticDimension, type[BaseCritic]] = {
    CriticDimension.ACCURACY:     AccuracyCritic,
    CriticDimension.LOGIC:        LogicCritic,
    CriticDimension.COMPLETENESS: CompletenessCritic,
    CriticDimension.SAFETY:       SafetyCritic,
    CriticDimension.STYLE:        StyleCritic,
}


def get_all_critics() -> list[BaseCritic]:
    """Instantiate and return one of each critic."""
    return [cls() for cls in CRITIC_REGISTRY.values()]


__all__ = [
    "BaseCritic",
    "AccuracyCritic",
    "LogicCritic",
    "CompletenessCritic",
    "SafetyCritic",
    "StyleCritic",
    "CRITIC_REGISTRY",
    "get_all_critics",
]

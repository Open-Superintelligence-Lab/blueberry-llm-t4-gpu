"""
Configuration classes for the blueberry-llm framework.

This module provides configuration dataclasses for different model types
and training scenarios.
"""

from .t4_moe_config import T4MoEModelConfig, get_rtx4090_config, get_rtx5090_config, get_development_config

__all__ = [
    'T4MoEModelConfig',
    'get_rtx4090_config',
    'get_rtx5090_config',
    'get_development_config',
]

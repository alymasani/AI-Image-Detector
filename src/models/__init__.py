"""Model definitions: custom CNN (original) and ResNet-18 (reference integration)."""

from .factory import build_model, display_name, initialization_source, parameter_counts

__all__ = ["build_model", "display_name", "initialization_source", "parameter_counts"]

"""Cross-cutting utilities (settings, logging) shared by all layers."""

from core.logging import setup_logging
from core.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "setup_logging"]

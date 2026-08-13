"""strandr — find which line of code stranded your GPU memory."""

from .core import start_recording, watch, report, collect

__version__ = "0.2.0"
__all__ = ["start_recording", "watch", "report", "collect"]

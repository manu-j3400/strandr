"""strandr — find which line of code stranded your GPU memory."""

from .core import start_recording, report, collect

__version__ = "0.1.0"
__all__ = ["start_recording", "report", "collect"]

"""strandr — find which line of code stranded your GPU memory."""

from .core import start_recording, watch, report, collect, advise, prove_snippet

__version__ = "0.3.0"
__all__ = ["start_recording", "watch", "report", "collect", "advise", "prove_snippet"]

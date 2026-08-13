"""
strandr — find out which line of code stranded your GPU memory.

A GPU can crash with "out of memory" while gigabytes sit free, because the
free space is scattered into holes too small to reuse (fragmentation). strandr
does the autopsy: it measures the trapped memory, counts the holes, and names
the exact allocation stranding each one.

Usage:
    import strandr
    strandr.start_recording()          # call BEFORE loading your model
    model = load_your_model().cuda()
    run_your_inference()
    strandr.report()                   # autopsy the current GPU memory
"""

import contextlib

import torch
from collections import defaultdict

MB, GB = 1024 ** 2, 1024 ** 3

# frames from these files are torch/framework plumbing, not the real caller
_SKIP = (
    "torch/nn/modules/module.py",
    "torch/cuda/memory",
    "torch/_ops",
    "torch/autograd",
    "_dynamo",
    "_inductor",
)


def _best_frame(frames):
    """Walk an allocation's call stack and return the first meaningful line."""
    for f in frames or []:
        if any(s in f.get("filename", "") for s in _SKIP):
            continue
        short = f.get("filename", "?").split("/")[-1]
        return f'{short}:{f.get("line", "?")} in {f.get("name", "?")}'
    if frames:
        f = frames[0]
        short = f.get("filename", "?").split("/")[-1]
        return f'{short}:{f.get("line", "?")} in {f.get("name", "?")}'
    return "no-stack"


def start_recording(max_entries=200000):
    """Turn on allocation stack recording.

    Call BEFORE any GPU memory is allocated (before loading the model), so
    every block carries the stack that created it. Safe to call as the very
    first CUDA operation — it initializes CUDA first.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "strandr needs a CUDA GPU, but none is available. "
            "On Colab: Runtime > Change runtime type > T4 GPU."
        )
    torch.cuda.init()  # ensure CUDA is up before touching the recording API
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=max_entries,
    )
    print("strandr: recording on — load your model and run inference, then call report()")


@contextlib.contextmanager
def watch(top=6):
    """Record, run your code, and auto-print the autopsy. The safe one-liner.

        import strandr
        with strandr.watch():
            model = load_your_model().cuda()
            run_your_inference()
        # report prints automatically on exit

    This wraps start_recording() and report() so they can't be called in the
    wrong order — the #1 cause of blank ("no-stack") attribution.
    """
    start_recording()
    try:
        yield
    finally:
        report(top=top)


def collect():
    """Return the raw autopsy data as a dict, without printing.

    Useful if you want to build your own view or assert on the numbers.
    """
    hist = torch.cuda.memory._snapshot()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    trapped = reserved - allocated

    holes = [
        b["size"]
        for seg in hist["segments"]
        for b in seg["blocks"]
        if b["state"] == "inactive"
    ]

    # who STRANDS memory: for each hole, blame the live block pinning it open
    stranded = defaultdict(lambda: [0, 0])  # culprit -> [hole_count, bytes]
    for seg in hist["segments"]:
        blocks = seg["blocks"]
        for i, b in enumerate(blocks):
            if b["state"] == "inactive":
                who = "edge of segment"
                for j in range(i - 1, -1, -1):
                    if blocks[j]["state"] == "active_allocated":
                        who = _best_frame(blocks[j].get("frames"))
                        break
                stranded[who][0] += 1
                stranded[who][1] += b["size"]

    ranked = sorted(stranded.items(), key=lambda kv: kv[1][1], reverse=True)
    return {
        "reserved": reserved,
        "allocated": allocated,
        "trapped": trapped,
        "holes": holes,
        "stranded": ranked,
    }


def report(top=6):
    """Print a human-readable fragmentation autopsy of the current GPU memory."""
    data = collect()
    reserved = data["reserved"]
    allocated = data["allocated"]
    trapped = data["trapped"]
    holes = data["holes"]
    ranked = data["stranded"]

    print("=" * 64)
    print("  strandr report")
    print("=" * 64)
    print(f"  reserved (held):    {reserved / GB:6.2f} GB")
    print(f"  allocated (in use): {allocated / GB:6.2f} GB")
    if reserved:
        print(f"  trapped (wasted):   {trapped / GB:6.2f} GB   ({100 * trapped / reserved:.0f}% of held)")
    hole_line = f"  fragmentation holes: {len(holes)}"
    if holes:
        hole_line += f"  (biggest {max(holes) / MB:.0f} MB)"
    print(hole_line)
    print("-" * 64)
    print("  who is stranding memory into holes:")
    if ranked:
        for who, (n, total) in ranked[:top]:
            print(f"    {total / MB:8.1f} MB  in {n:3d} holes  <-  {who}")
    else:
        print("    (no fragmentation detected)")
    print("=" * 64)

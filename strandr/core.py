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


def advise():
    """Read the current fragmentation shape and prescribe the allocator fix.

    Returns the recommended PYTORCH_CUDA_ALLOC_CONF value and prints the
    reasoning. The setting must be applied in a FRESH process (before CUDA
    initializes), so this advises — see prove_snippet() for a before/after
    harness you can run.
    """
    data = collect()
    reserved = data["reserved"]
    allocated = data["allocated"]
    trapped = data["trapped"]
    holes = data["holes"]

    print("=" * 64)
    print("  strandr advice")
    print("=" * 64)

    if not holes or trapped < 50 * MB:
        print("  fragmentation is minimal — no allocator change needed.")
        print("=" * 64)
        return None

    frac = trapped / reserved if reserved else 0
    big_holes = [h for h in holes if h > 50 * MB]
    tiny_holes = [h for h in holes if h < 2 * MB]

    print(f"  trapped: {trapped/GB:.2f} GB ({100*frac:.0f}% of held)"
          f" across {len(holes)} holes")

    recommendation = None
    reason = None

    if frac > 0.20:
        # a lot of memory is stranded relative to what's held — segments aren't
        # being reused. expandable_segments lets the allocator grow/reclaim.
        recommendation = "expandable_segments:True"
        reason = ("a large share of held memory is trapped, which means whole "
                  "segments aren't being reused. expandable_segments lets the "
                  "allocator reclaim them.")
    elif big_holes:
        # large uniform holes -> cap the split size near the hole size so the
        # allocator stops leaving big unusable remainders.
        import statistics
        target = int(statistics.median(big_holes) / MB)
        # round to a sensible boundary
        target = max(128, (target // 128) * 128)
        recommendation = f"max_split_size_mb:{target}"
        reason = (f"you have {len(big_holes)} large holes (median "
                  f"{statistics.median(big_holes)/MB:.0f} MB). capping the split "
                  f"size stops the allocator from leaving big unusable remainders.")
    elif len(tiny_holes) > 20:
        recommendation = "roundup_power2_divisions:8"
        reason = (f"you have {len(tiny_holes)} tiny holes — rounding allocation "
                  "sizes to power-of-two divisions reduces small-block churn.")
    else:
        recommendation = "expandable_segments:True"
        reason = "expandable_segments is the safest general fragmentation fix."

    print("-" * 64)
    print(f"  recommended:  PYTORCH_CUDA_ALLOC_CONF={recommendation}")
    print(f"  why: {reason}")
    print("-" * 64)
    print("  apply it BEFORE importing torch / touching CUDA:")
    print(f"      import os")
    print(f'      os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "{recommendation}"')
    print("  (must be a fresh process — the setting is read at CUDA init)")
    print("=" * 64)
    return recommendation


def prove_snippet(recommendation=None):
    """Print a copy-paste harness that runs a workload twice — once baseline,
    once with the recommended fix — and reports the recovered memory.

    Because the allocator setting is read at CUDA init, proving it requires
    two fresh processes. This prints a script you run once with, once without.
    """
    rec = recommendation or "expandable_segments:True"
    print("# Save as prove.py and run twice:")
    print("#   python prove.py            # baseline")
    print(f'#   FIX=1 python prove.py      # with {rec}')
    print("import os, sys")
    print('if os.environ.get("FIX"):')
    print(f'    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "{rec}"')
    print("import torch, strandr")
    print("strandr.start_recording()")
    print("# --- your workload here ---")
    print("# model = load_your_model().cuda(); run_your_inference()")
    print("# --------------------------")
    print("strandr.report()")


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

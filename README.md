# strandr

**The tool you need to find which line of code stranded your GPU memory.**

Your GPU can crash with "out of memory" while gigabytes sit free. Not because
the memory is full, but because it's in the wrong shape. strandr does the
autopsy: it finds the wasted memory, counts the holes, and names the exact
allocation that stranded each one.

## The parking lot dilemma

A GPU has a chunk of memory. A model grabs space when it needs it and hands it
back when it's done. Do that over and over in different sizes and the free space
gets chopped into scattered slivers. Added up, it looks like plenty. But it's in
useless pieces, so when the model asks for one normal-sized block, no single gap
is big enough, and it crashes.

It's a parking lot that's half empty, but every open spot has a car parked over
the line. A bus pulls up. Nowhere to fit. Not because the lot is full, because
the space is scattered.

Today people debug this by hand: they stare at "out of memory" on a half-empty
GPU and flip allocator settings hoping one sticks. strandr tells you *which
allocation* is parked over the line.

## What it does

- **Measures the waste.** Reserved vs allocated: the gap is trapped memory.
- **Counts the holes.** How many fragments, and how big the largest is.
- **Names the culprit.** For each hole, it finds the live block pinning it open
  and reads the stack that created it — down to the file and line.

## Install

```bash
git clone https://github.com/manu-j3400/strandr
cd strandr
pip install -e .
```

Requires PyTorch with CUDA and `nvidia-ml-py`.

## Usage

The safe one-liner. Wrap your code and the autopsy prints automatically:

```python
import strandr

with strandr.watch():
    model = load_your_model().cuda()
    run_your_inference()
# strandr report prints here
```

`watch()` handles recording and reporting in the right order, so attribution
never comes back blank. If you need manual control:

```python
strandr.start_recording()          # BEFORE loading your model
model = load_your_model().cuda()
run_your_inference()
strandr.report()                   # autopsy the current GPU memory
```

Recording must start before any GPU memory is allocated, so every block carries
the stack that created it. `watch()` and `start_recording()` both enforce this.

## Example output

```
================================================================
  strandr report
================================================================
  reserved (held):     13.14 GB
  allocated (in use):   7.68 GB
  trapped (wasted):     5.46 GB   (42% of held)
  fragmentation holes: 28  (biggest 755 MB)
----------------------------------------------------------------
  who is stranding memory into holes:
      5288.2 MB  in   7 holes  <-  cache_utils.py:166 in update
       273.6 MB  in   4 holes  <-  modeling_utils.py:3700 in cuda
        32.0 MB  in  16 holes  <-  edge of segment
================================================================
```

## Try the demo

```bash
pip install torch transformers
python examples/demo.py
```

It feeds GPT-2 prompts of very different lengths, which fragments the allocator,
then runs the autopsy.

## How it works

PyTorch's CUDA caching allocator grabs memory from the GPU in large *segments*,
then carves *blocks* out of each. When a block frees, it becomes a hole inside
its segment, but the allocator can't always hand that hole back to the GPU. If
the only free space is scattered holes smaller than the next request, you OOM
with memory to spare.

strandr walks the allocator's real block layout, finds every inactive (free but
trapped) block, and for each one identifies the live block physically pinning it
open. It reads that block's allocation stack — captured via
`torch.cuda.memory._record_memory_history` — to name the file and line that
stranded the memory.

## Status

Early. Works on real models today. Attribution is by the block physically
neighboring each hole; larger-model and serving-stack coverage is next.

## License

MIT

"""
strandr demo: make a real model fragment its own GPU memory, then autopsy it.

Run on any CUDA machine or a free Colab T4:
    pip install torch transformers
    python examples/demo.py

Feeds GPT-2 prompts of wildly different lengths. The mismatched sizes fragment
the CUDA allocator's memory, and strandr names what stranded it.
"""

import random
import torch

import strandr


def main():
    assert torch.cuda.is_available(), "Needs a CUDA GPU (a free Colab T4 works)."
    random.seed(0)

    # 1) recording ON before any allocation, so every block gets a birth stack
    strandr.start_recording()

    # 2) load the model AFTER recording is on
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

    # 3) real inference with mismatched lengths -> fragmentation
    for i in range(40):
        length = random.choice([16, 256, 32, 512, 64, 900])
        ids = torch.randint(0, 50000, (1, length), device="cuda")
        with torch.no_grad():
            model.generate(
                ids,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

    # 4) autopsy
    strandr.report()


if __name__ == "__main__":
    main()

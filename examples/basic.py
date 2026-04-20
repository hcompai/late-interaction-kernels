"""Minimal end-to-end examples for late-interaction-kernels.

Run with::

    python examples/basic.py rerank      # inference / reranking
    python examples/basic.py train       # tiny training loop
    python examples/basic.py pylate      # PyLate monkey-patch demo (needs pylate installed)
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from late_interaction_kernels import maxsim, maxsim_inference


def rerank_example():
    """Score 1000 docs against one 32-token query in fp16."""
    Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)
    scores = maxsim_inference(Q, D)  # [1000], fp32
    top10 = scores.topk(10)
    print("top-10 doc indices:", top10.indices.tolist())
    print("top-10 scores:", [f"{s:.3f}" for s in top10.values.tolist()])


def train_example():
    """Contrastive-style training step with in-batch negatives."""
    B, Lq, Ld, d = 16, 32, 200, 128
    q = torch.nn.Parameter(torch.randn(B, Lq, d, device="cuda"))
    pos = torch.nn.Parameter(torch.randn(B, Ld, d, device="cuda"))
    neg = torch.nn.Parameter(torch.randn(B, Ld, d, device="cuda"))
    opt = torch.optim.AdamW([q, pos, neg], lr=1e-3)

    for step in range(20):
        opt.zero_grad()
        qn, pn, nn_ = F.normalize(q, dim=-1), F.normalize(pos, dim=-1), F.normalize(neg, dim=-1)
        s_pos = maxsim(qn, pn)  # [B, B]
        s_neg = maxsim(qn, nn_)
        logits = torch.cat([s_pos, s_neg], dim=1)
        labels = torch.arange(B, device="cuda")
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")


def pylate_example():
    """Monkey-patch PyLate's `colbert_scores` so its trainers use our kernel."""
    from late_interaction_kernels import patch_pylate

    patch_pylate()
    print("PyLate patched — every `colbert_scores` / `Contrastive` / `rerank` call now")
    print("runs through the Triton kernel. Your PyLate code needs no other changes.")
    # (Left to the user to plug into their pipeline:
    #    from pylate import models, losses
    #    model = models.ColBERT(...); loss = losses.Contrastive(model=model); trainer.train()
    # )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rerank"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for late-interaction-kernels kernels.")
    {"rerank": rerank_example, "train": train_example, "pylate": pylate_example}[cmd]()

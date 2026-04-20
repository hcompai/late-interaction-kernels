"""Minimal training-loop example showing gradients flowing through flash-colbert."""

import torch
import torch.nn.functional as F

from flash_colbert import maxsim

B, Lq, Ld, d = 16, 32, 200, 128

q = torch.nn.Parameter(torch.randn(B, Lq, d, device="cuda", dtype=torch.float32))
pos = torch.nn.Parameter(torch.randn(B, Ld, d, device="cuda", dtype=torch.float32))
neg = torch.nn.Parameter(torch.randn(B, Ld, d, device="cuda", dtype=torch.float32))

opt = torch.optim.AdamW([q, pos, neg], lr=1e-3)

for step in range(20):
    opt.zero_grad()
    q_n = F.normalize(q, dim=-1)
    p_n = F.normalize(pos, dim=-1)
    n_n = F.normalize(neg, dim=-1)
    s_pos = maxsim(q_n, p_n)  # [B, B]
    s_neg = maxsim(q_n, n_n)  # [B, B]
    scores = torch.cat([s_pos, s_neg], dim=1)
    labels = torch.arange(B, device="cuda")
    loss = F.cross_entropy(scores, labels)
    loss.backward()
    opt.step()
    if step % 5 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")

"""Minimal reranking example."""

import torch

from flash_colbert import maxsim_inference

Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)
D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)

scores = maxsim_inference(Q, D)  # shape: [1000]
top10 = scores.topk(10)
print("top-10 doc indices:", top10.indices.tolist())
print("top-10 scores:", top10.values.tolist())

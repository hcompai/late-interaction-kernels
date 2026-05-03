# Head-to-head vs flash-maxsim — NVIDIA_H100_80GB_HBM3 (bf16)

flash-maxsim version: `0.2.0`.  50-iter median, CUDA events.

| shape | impl | fwd (no norm) ms | fwd (norm) ms | peak MB | speedup vs FM |
| --- | --- | --- | --- | --- | --- |
| rerank-short | late-interaction-kernels | 0.096 | 0.105 | 0.0 | 1.13× |
| rerank-short | flash-maxsim | 0.109 | 0.109 | 0.0 | - |
| rerank-short | naive-einsum | 0.284 | 0.281 | 183.1 | - |
| rerank-long | late-interaction-kernels | 0.153 | 0.162 | 0.0 | 1.10× |
| rerank-long | flash-maxsim | 0.168 | 0.167 | 0.0 | - |
| rerank-long | naive-einsum | 0.809 | 0.811 | 626.0 | - |
| rerank-very-long | late-interaction-kernels | 0.237 | 0.248 | 0.0 | 1.06× |
| rerank-very-long | flash-maxsim | 0.252 | 0.251 | 0.0 | - |
| rerank-very-long | naive-einsum | 1.564 | 1.564 | 1250.0 | - |
| rerank-colpali | late-interaction-kernels | 0.409 | 0.600 | 0.0 | 1.21× |
| rerank-colpali | flash-maxsim | 0.495 | 0.486 | 0.0 | - |
| rerank-colpali | naive-einsum | 2.022 | 2.020 | 2250.5 | - |
| rerank-10k | late-interaction-kernels | 0.310 | 0.370 | 0.0 | 1.05× |
| rerank-10k | flash-maxsim | 0.324 | 0.324 | 0.1 | - |
| rerank-10k | naive-einsum | 2.362 | 2.373 | 1831.1 | - |
| train-in-batch-32 | late-interaction-kernels | 0.087 | 0.090 | 0.0 | 1.15× |
| train-in-batch-32 | flash-maxsim | 0.100 | 0.100 | 0.0 | - |
| train-in-batch-32 | naive-einsum | 0.169 | 0.169 | 28.6 | - |
| train-in-batch-128 | late-interaction-kernels | 0.233 | 0.345 | 0.1 | 1.26× |
| train-in-batch-128 | flash-maxsim | 0.294 | 0.296 | 0.1 | - |
| train-in-batch-128 | naive-einsum | 0.629 | 0.630 | 414.5 | - |
| train-long-doc | late-interaction-kernels | 0.088 | 0.113 | 0.0 | 1.01× |
| train-long-doc | flash-maxsim | 0.088 | 0.088 | 0.0 | - |
| train-long-doc | naive-einsum | 0.168 | 0.168 | 80.2 | - |
| edge-d48 | late-interaction-kernels | 0.320 | 0.366 | 0.0 | 1.07× |
| edge-d48 | flash-maxsim | 0.341 | 0.343 | 0.0 | - |
| edge-d48 | naive-einsum | 2.831 | 2.827 | 2500.0 | - |
| edge-d64 | late-interaction-kernels | 0.190 | 0.208 | 0.0 | 1.14× |
| edge-d64 | flash-maxsim | 0.218 | 0.221 | 0.1 | - |
| edge-d64 | naive-einsum | 1.401 | 1.404 | 1098.6 | - |
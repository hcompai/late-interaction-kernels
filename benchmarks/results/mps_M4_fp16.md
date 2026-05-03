# MPS forward benchmark — M4 (fp16)

30-iter median, `torch.mps.synchronize` between calls.

| shape | metal ms | compile ms | eager ms | metal vs compile | compile vs eager |
| --- | --- | --- | --- | --- | --- |
| rerank-short | 8.419 | 10.869 | 18.676 | 1.29x | 1.72x |
| rerank-mid | 16.433 | 18.279 | 31.129 | 1.11x | 1.70x |
| rerank-10k | 55.535 | 93.873 | 170.502 | 1.69x | 1.82x |
| colpali | 3.108 | 3.373 | 6.513 | 1.09x | 1.93x |
| colpali-big | 9.840 | 17.272 | 28.893 | 1.76x | 1.67x |
| train-batch | 5.732 | 1.639 | 2.326 | 0.29x | 1.42x |
| edge-d48 | 33.327 | 67.215 | 107.206 | 2.02x | 1.59x |
| edge-d64 | 4.822 | 5.397 | 13.060 | 1.12x | 2.42x |
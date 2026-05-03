# MPS forward benchmark — M4 (fp16)

30-iter median, `torch.mps.synchronize` between calls.

| shape | metal ms | compile ms | eager ms | metal vs compile | compile vs eager |
| --- | --- | --- | --- | --- | --- |
| rerank-short | 8.317 | 10.737 | 17.529 | 1.29x | 1.63x |
| rerank-mid | 13.290 | 17.578 | 28.453 | 1.32x | 1.62x |
| rerank-10k | 78.976 | 93.381 | 166.711 | 1.18x | 1.79x |
| colpali | 3.036 | 3.288 | 7.039 | 1.08x | 2.14x |
| colpali-big | 13.295 | 17.925 | 28.407 | 1.35x | 1.58x |
| train-batch | 6.420 | 1.656 | 2.434 | 0.26x | 1.47x |
| edge-d48 | 52.969 | 64.453 | 105.699 | 1.22x | 1.64x |
| edge-d64 | 5.602 | 6.720 | 12.541 | 1.20x | 1.87x |
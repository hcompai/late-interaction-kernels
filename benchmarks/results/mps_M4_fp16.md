# MPS forward benchmark — M4 (fp16)

30-iter median, ``torch.mps.synchronize`` between calls.

| shape | compile ms | eager ms | speedup | compile peak MB | eager peak MB |
| --- | --- | --- | --- | --- | --- |
| rerank-short | 10.361 | 18.823 | 1.82x | 0.0 | 8.0 |
| rerank-mid | 17.537 | 29.958 | 1.71x | 0.0 | 8.0 |
| rerank-10k | 101.590 | 178.216 | 1.75x | 2490.0 | 3964.0 |
| colpali | 3.589 | 6.507 | 1.81x | 0.0 | 8.0 |
| train-batch | 1.788 | 2.452 | 1.37x | 0.0 | 8.0 |
| edge-d48 | 66.513 | 112.120 | 1.69x | 750.0 | 1508.0 |
| edge-d64 | 5.279 | 11.449 | 2.17x | 0.0 | 8.0 |
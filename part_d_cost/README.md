# Part D — Scaling reality check

**File to submit:** `training_cost_estimator_v2.py`  ·  **Original:** `training_cost_estimator.py` (Assignment 1)

Same model, GPU and utilisation as Assignment 1: **Llama 3 8B** (P = 8×10⁹, T = 15×10¹² tokens) on an **A100** (312 TFLOP/s BF16) at **40% MFU**. Assignment 1 assumed perfect linear scaling. Version 2 adds the gradient all-reduce.

## The model

| Term | Formula | Input and source |
|---|---|---|
| Global batch | G = 4M tokens/step | Dubey et al. 2024, *The Llama 3 Herd of Models*, §3.4.1 |
| Compute per step | T_compute = 6·P·(G/N) / (peak × util) | Assignment 1 numbers |
| Gradient size | n = 2·P bytes = 16 GB (bf16) | — |
| All-reduce per step | T_comm = 2(N−1)α + 2(N−1)/N · n · β | Part B ring formula |
| Interconnect | 1/β = 200 Gb/s = 25 GB/s per GPU, α = 2 µs | NVIDIA DGX A100 datasheet (HDR InfiniBand, one per GPU) |
| Contrast | your own α, β | read automatically from `../results/partB_alpha_beta.json` |
| Efficiency, no overlap | E = T_compute / (T_compute + T_comm) | |
| Efficiency, perfect overlap | E = T_compute / max(T_compute, T_comm) | |

## Run

```bash
cd part_d_cost
python3 training_cost_estimator.py        # Assignment 1: 65.2 days on 1,024 GPUs (perfect scaling)
python3 training_cost_estimator_v2.py     # Assignment 2: adds communication
```

Run Part B `bench` first, so that the second table uses your own measured α and β.

## Expected output (published interconnect)

| N | T_compute (s) | T_comm (s) | E no-overlap | E overlap |
|---|---|---|---|---|
| 8 | 192.3 | 1.12 | 99.4% | 100% |
| 64 | 24.0 | 1.26 | 95.0% | 100% |
| 1024 | 1.50 | 1.28 | **53.9%** | 100% |

At 1,024 GPUs, compute per step shrinks 128× compared with N = 8, but the ring all-reduce time stays about 2·n·β ≈ 1.3 s whatever N is. The corrected time to train is 65.2 / 0.539 ≈ **121 days**, compared with 65 days in Assignment 1.

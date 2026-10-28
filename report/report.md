# Assignment 2 — From "It Runs in Parallel" to "I Know Why It Doesn't Scale"

> **Machine used for the numbers below:** cloud VM, Intel Xeon @ 2.80 GHz, **2 logical cores**, L1d 32 KiB/core, L2 1 MiB/core, L3 33 MiB, Ubuntu 24.04, gcc 13.3, Open MPI 4.1.6, Python 3.13, mpi4py 4.1.
> **Re-run `./run_all.sh` on your own machine and replace the numbers** (all raw output is in `results/`).
> Compile: `gcc -O2 -fopenmp -o omp_kernels omp_kernels.c -lm` · Run: see the "Exact commands" section at the end.

## 1. Part A — OpenMP

**A1 — dot product, N = 2²⁴, `OMP_NUM_THREADS=2`, exact answer 8,388,608**

| Variant | Run 1 | Run 2 | Run 3 | Time (s) |
|---|---|---|---|---|
| racy | 5,739,206 | 4,556,296 | 4,462,889 | 0.03 |
| critical | 8,388,608 | 8,388,608 | 8,388,608 | 1.19–1.63 |
| atomic | 8,388,608 | 8,388,608 | 8,388,608 | 0.58–0.67 |
| reduction | 8,388,608 | 8,388,608 | 8,388,608 | **0.015–0.020** |

The racy answer is wrong and different every run (lost updates from unsynchronised read-modify-write). `sum` is declared `volatile` in the racy version only; at -O2 GCC otherwise keeps it in a register, which hides the race behind a deterministic "last writer wins" result of exactly N/4. **critical** is slowest, due to *serialisation*: one lock makes every iteration run one at a time, plus lock overhead. **atomic** is about 2× faster but still about 40× slower than reduction, due to *contention*: both cores fight for the same cache line. Reduction is correct and fastest because each thread keeps a private partial sum, and these are combined once at the end.

**A2 — GEMM, N = 1024, `gcc -O2 -fopenmp`, 1 warm-up + mean of 3**

| Version | GFLOP/s | max abs diff vs naive |
|---|---|---|
| naive serial (i-j-k) | 0.35 | — |
| naive + OpenMP (outer `i` loop: each thread owns whole rows of C, so no race) | 1.12 | 0 (PASS) |
| tiled + OpenMP (T = 64, `collapse(2)` over C tiles, i-k-j inside) | 3.10 | 0 (PASS) |

**A3 — scaling (S = T₁/Tₚ, E = S/p)**

| p | dot time (s) | S | E | | tiled GEMM time (s) | S | E |
|---|---|---|---|---|---|---|---|
| 1 | 0.0271 | 1.00 | 1.00 | | 1.819 | 1.00 | 1.00 |
| 2 | 0.0161 | 1.68 | 0.84 | | 0.622 | 2.93 | 1.46* |

\*E > 1 is VM noise: 2 vCPUs share a host and the frequency changes between runs. The 1-thread GEMM number is the least stable.

**Classify.** The dot product stops scaling first. It does 2 FLOPs per 16 bytes loaded, an arithmetic intensity of **0.125 FLOP/byte**, far left of the roofline ridge point. So it is memory-bound: at 2 threads it already streams about 16.6 GB/s, and more threads only queue on the same DRAM bandwidth. Tiled GEMM reuses each loaded element about T times (AI ≈ T/8 FLOP/byte), so it stays compute-bound and keeps scaling.

**Change one thing: tile size T = 64 → 32.** *Prediction (before running):* one tile step's working set is 3T²·8 B = 96 KiB at T=64, which overflows the 32 KiB L1d. At T=32 it is 24 KiB, which fits L1, so **faster**. *Result:* 3.10 → **4.43 GFLOP/s** (+43%), so the prediction held. *Sanity check:* run the course `examples/week3_openmp/gemm` and `scaling.sh` on your machine and note whether the numbers are in the same ballpark. If they differ, the likely causes are flags (-O2 vs -O3), tile size, loop order, thread placement, or P- vs E-cores on Apple Silicon.

## 2. Part B — Ring all-reduce (mpi4py, `Sendrecv` only)

**Correctness:** for P = 2, 3, 4 and n ∈ {1, P−1, 7, 1000, 1001, 1003, 4099, 100003, 2²⁰+1} (most not divisible by P), the max |ring − MPI_Allreduce| ranges from 0 to **1.8×10⁻¹⁵ (< 10⁻¹⁰)**. It is not exactly 0 because floating-point addition is not associative and the library adds in a different order.

**Deadlock:** with a plain `MPI_Send` then `MPI_Recv` on every rank, small messages are buffered *eagerly*, so `Send` returns and it appears to work. Above the eager limit, MPI switches to the *rendezvous* protocol, where `Send` blocks until the matching `Recv` is posted. Every rank is then stuck in `Send` waiting on its neighbour (a circular wait), so it deadlocks for large n.

**B2 — P = 4, max across ranks (µs)**

| n | 2⁴ | 2⁸ | 2¹² | 2¹⁶ | 2²⁰ | 2²⁴ |
|---|---|---|---|---|---|---|
| ring | 85 | 95 | 138 | 603 | 8,206 | 129,865 |
| MPI_Allreduce | 13 | 18 | 86 | 609 | 11,357 | 244,408 |

**Fit:** α = **14.2 µs**, 1/β = **1.55 GB/s**, cross-over n* = Pα/(8β) ≈ **11,000 doubles (~86 KiB)**. Below n*, the latency term 2(P−1)α dominates, so sending many small gradient tensors pays α per tensor; *bucket* them into fewer, larger messages. Above n*, the bandwidth term dominates and only bytes matter.

**Change one thing: P = 2 → 4 → 8.** *Prediction from the model:* small-message time grows like 2(P−1)α, i.e. ×3 then ×2.3. Large-message time grows only like 2(P−1)/P (1 → 1.5 → 1.75), nearly flat. *Measured (2⁴ / 2²⁴):* P=2: 12 µs / 51 ms; P=4: 85 µs / 130 ms; P=8: 369 µs / 342 ms. Both grow **much faster** than predicted. On one 2-core machine the "network" is shared memory: P=4 and P=8 are oversubscribed (`--oversubscribe`), so ranks time-share cores and compete for the same memory bandwidth. The model assumes independent links, so β is not constant here.

**Sanity check:** `MPI_Allreduce` beats the ring below ~2¹⁶ doubles. Libraries switch algorithms by size and topology (recursive doubling or shared-memory reductions with log P latency for small messages), and they have no Python-level per-step overhead. For large messages, our bandwidth-optimal ring is competitive or faster in this oversubscribed setup.

## 3. Part C — Data-parallel MLP (`mpirun -np 4 python3 dp_mlp.py`)

Rank 0 initialises and `Bcast`s the weights. G = 32, so each rank takes a disjoint slice of 8. Every gradient goes through the Part B ring, is divided by P, then plain SGD runs for 200 epochs.
- **(i) Replicas identical:** max |W1ᵣ − W1₀| = **0.0** (identical updates on identical data).
- **(ii) DP = full batch:** max per-epoch loss difference vs single-process training (same G, init and order) = **3.6×10⁻¹⁶**; final W1 differs by 4.4×10⁻¹⁵. The same holds with `comm.Allreduce` (5.6×10⁻¹⁶). Validation accuracy is 0.988.
- **Unequal shards:** each rank's gradient is a mean over its *own* shard, so dividing the sum by P gives every shard equal weight. Samples in smaller shards would be over-weighted, so the result is no longer the global-batch mean gradient and the curves would diverge. The fix is to sum the per-sample gradients and divide by G.

## 4. Part D — Llama 3 8B on A100 (40% MFU), G = 4M tokens, bf16 gradients = 16 GB/step

| N | T_compute (s) | T_comm, IB 25 GB/s, α = 2 µs (s) | E no-overlap | E overlap | T_comm, our α/β (s) | E no-overlap |
|---|---|---|---|---|---|---|
| 8 | 192.3 | 1.12 | 99.4% | 100% | 21.6 | 89.9% |
| 64 | 24.0 | 1.26 | 95.0% | 100% | 24.3 | 49.8% |
| 1024 | 1.50 | 1.28 | **53.9%** | 100% | 24.7 | 5.7% |

**1,024-GPU training time:** Assignment 1 (perfect scaling) gave **65.2 days**; the no-overlap corrected estimate is **120.9 days (+85%)**. Sources: G from Dubey et al. 2024 (Llama 3 paper, §3.4.1); 200 Gb/s HDR InfiniBand per GPU from the NVIDIA DGX A100 datasheet.

## 5. Synthesis — what to try first at N = 1024

I would first **overlap the gradient all-reduce with the backward pass using gradient bucketing** (DDP-style). Three pieces of evidence point to it. (1) Part D shows T_comm (1.28 s) is *smaller* than T_compute (1.50 s), so communication is not too large; it is just *exposed*. Perfect overlap raises E from 54% to about 100% with no new hardware. (2) Part B says 16 GB is about 10⁵× past the cross-over, so the cost is pure bandwidth (the 2(N−1)α term is only about 4 ms). Bucketing keeps every message above n* so it doesn't add latency, and adding more GPUs won't shrink the 2(N−1)/N·bytes·β term. (3) Part C proves the math is unchanged: DP equals large-batch SGD whenever the reduced gradients are correct, so overlap changes *when* the communication happens, not *what* is computed.

A larger per-GPU batch would also help: G/N ≈ 3,900 tokens is less than one 4,096-token sequence. But that changes the optimiser's batch size, not just the systems setup. **The confirming measurement** on the real cluster is a profiler trace of one step: measure the *exposed* (non-overlapped) NCCL all-reduce time against backward time, and check that nccl-tests bus bandwidth is close to 25 GB/s.

The link to one node: the A3 dot-product roll-off and the N = 1024 all-reduce are the **same kind of limit**. Both are bound by data movement, not FLOPs; adding workers (threads or GPUs) only splits the compute, while the bytes still have to cross one shared bottleneck (DRAM bus or network link). The A1 critical/atomic slowdowns are the other class, synchronisation, which matches the latency (α) side of the model.

## Exact commands

```bash
bash setup/install_ubuntu_wsl.sh && source .venv/bin/activate     # one-time setup
./run_all.sh                                                      # everything; output in results/
make -C part_a_openmp && ./part_a_openmp/omp_kernels dot|gemm|scale
./part_a_openmp/omp_kernels gemm 1024 32                          # change one thing: T=32
mpirun -np 4 python3 part_b_mpi/my_allreduce.py test|bench        # P=2,3,4 for test; add --oversubscribe if P > cores
mpirun -np 4 python3 part_c_dp_mlp/dp_mlp.py
python3 part_d_cost/training_cost_estimator_v2.py
```

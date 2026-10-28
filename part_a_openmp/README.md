# Part A — OpenMP: first correct, then fast

**File to submit:** `omp_kernels.c`  ·  **Helpers:** `Makefile`, `scaling.sh`

One C program with four modes:

| Command | Assignment step | What runs |
|---|---|---|
| `./omp_kernels dot [N] [N_SYNC]` | A1 | dot product: racy, critical, atomic, reduction |
| `./omp_kernels gemm [N] [T] [ijk\|ikj]` | A2 | naive serial, naive + OpenMP, tiled + OpenMP GEMM |
| `./omp_kernels scale [N_DOT] [N_GEMM] [T] [ORDER]` | A3 | thread sweep 1, 2, 4, … up to your logical cores |
| `./omp_kernels all` | A1–A3 | all of the above with defaults |

## Step 1 — Build

```bash
cd part_a_openmp
make                       # Linux/WSL: gcc -O2 -fopenmp ...   macOS: clang + libomp
```

The program prints the compiler and the exact flags at start-up, plus `omp_get_max_threads()`. Check that the thread count equals your core count, not 1.

```bash
make clean && make OPT=-O3    # optional: -O3 build. The report must state which flags you used.
```

## Step 2 — A1: correctness, the same dot product four ways

```bash
./omp_kernels dot      # run it 3 times
./omp_kernels dot
./omp_kernels dot
```

`a[i] = 1.0`, `b[i] = 0.5`, N = 2²⁴, so the exact answer is N/2 = 8,388,608.

| Variant | Code | What you should see | Why |
|---|---|---|---|
| racy | `sum += a[i]*b[i]` with shared `sum` | **wrong, different each run** | unsynchronised read-modify-write → lost updates |
| critical | `#pragma omp critical` | correct, **slowest** | *serialisation*: one lock, every iteration runs one at a time |
| atomic | `#pragma omp atomic` | correct, slow | *contention*: all cores hammer the same cache line |
| reduction | `reduction(+:sum)` | correct, **fastest** | private partial sums, combined once at the end |

> The racy variant declares `sum` as `volatile`. Without it, `gcc -O2` keeps `sum` in a register and stores it once per thread, which gives a *deterministic* wrong answer (exactly N/4 with 2 threads). `volatile` forces the load-add-store to memory on every iteration, so you see the classic non-deterministic race.
>
> If `critical` takes more than ~10 s: `./omp_kernels dot 16777216 1048576` uses 2²⁰ for critical/atomic only. Mention it in the report.

## Step 3 — A2: naive vs parallel vs tiled GEMM

```bash
./omp_kernels gemm                # N=1024, T=64, in-tile order ikj
```

- **Naive serial**: textbook i-j-k triple loop.
- **Naive + OpenMP**: `#pragma omp parallel for` on the **outer `i` loop**. Each thread owns whole rows of C, so no two threads write the same `C[i][j]`, and there is no race.
- **Tiled + OpenMP**: blocks of T×T, `collapse(2)` over the (ii, jj) tiles of C. Each C tile belongs to exactly one thread.
- Verification: max |C_version − C_naive| < 10⁻⁹ → `PASS`.
- Timing: 1 untimed warm-up + mean of 3 timed calls (the Assignment 1 recipe). GFLOP/s = 2N³/t.

## Step 4 — A3: thread-scaling study

```bash
./omp_kernels scale       # sweep done INSIDE the program with omp_set_num_threads()
./scaling.sh              # the same sweep as a SHELL LOOP over OMP_NUM_THREADS
```

The tables show time, speedup S(p) = T₁/Tₚ, and efficiency E(p) = S(p)/p for the reduction dot product and the tiled GEMM.

## Step 5 — The optimisation loop

1. **Measure**: the numbers from Steps 2–4.
2. **Classify**:
   - The dot product does 2 FLOPs per 16 bytes, an arithmetic intensity of **0.125 FLOP/byte**, far below the roofline ridge. It is **memory-bound**, so it stops scaling first: extra threads just queue on the same DRAM bandwidth.
   - Tiled GEMM reuses each loaded value about T times, so it is compute-bound and keeps scaling.
   - The slowest A1 variant is `critical` (serialisation). `atomic` is cache-line contention.
3. **Change one thing**. Find your cache sizes first:
   ```bash
   lscpu | grep -i cache                              # Linux / WSL
   sysctl hw.l1dcachesize hw.l2cachesize              # macOS
   ```
   One tile step's working set is about 3·T²·8 bytes: T=64 is 96 KiB (larger than a 32–48 KiB L1), and T=32 is 24 KiB (fits L1). **Write your prediction down, then run** one of:
   ```bash
   ./omp_kernels gemm 1024 32          # option 1: tile size 64 -> 32
   ./omp_kernels gemm 1024 64 ijk      # option 2: loop order inside the tile ikj -> ijk (NOT both)
   ```
4. **Re-measure**: compare GFLOP/s and say whether the prediction held.
5. **Sanity check**: build and run the course's `examples/week3_openmp/gemm` and `scaling.sh` and compare ballparks. Only do this *after* writing your own code.

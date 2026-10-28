# Part B — Build your own ring all-reduce

**File to submit:** `my_allreduce.py` (mpi4py)

`ring_allreduce(buf, n, comm)` sums a length-n float64 vector across all ranks **in place**, using only `comm.Sendrecv`. There is no `Allreduce`, `Reduce` or `Reduce_scatter` inside it, and it works when **n is not divisible by P**.

## How the algorithm works

```
chunk c covers indices [c*n//P, (c+1)*n//P)        <- integer division handles uneven n

Phase 1  reduce-scatter (P-1 steps)
  step s: send chunk (rank - s) to   rank+1
          recv chunk (rank - s - 1) from rank-1, ADD it into my copy
  -> after P-1 steps, rank r holds the fully summed chunk (r+1) % P

Phase 2  all-gather (P-1 steps)
  step s: send chunk (rank + 1 - s) to rank+1
          recv chunk (rank - s) from rank-1, OVERWRITE my copy
  -> every rank holds the whole result
```

Each rank sends 2(P−1) messages of about n/P doubles, which gives the cost model
**T_ring(n) ≈ 2(P−1)·α + 2(P−1)/P · (8n bytes) · β**.

## Step 1 — B1: correctness (P = 2, 3, 4)

```bash
cd part_b_mpi
mpirun -np 2 python3 my_allreduce.py test
mpirun -np 3 python3 my_allreduce.py test
mpirun -np 4 python3 my_allreduce.py test    # add --oversubscribe if you have < 4 cores
```

Each rank fills `buf` with random numbers seeded by its rank, runs the ring, and compares against `MPI_Allreduce` on a copy. Sizes include 1, P−1, 7, 1001, 1003, 4099, 100003 and 2²⁰+1, most of them not multiples of P. Expected: every row `PASS` with a max difference around 10⁻¹⁶–10⁻¹⁵. It is **not exactly 0** because floating-point addition is not associative and the library adds in a different order.

**Deadlock sentence for the report:** if every rank did a plain `MPI_Send` and then `MPI_Recv`, small messages would be buffered *eagerly*, so `Send` returns and it seems to work. Above the eager limit, MPI uses the *rendezvous* protocol, where `Send` blocks until the receiver posts `Recv`. Every rank is then stuck in `Send` (a circular wait), so it deadlocks. `Sendrecv` posts both halves together, so it cannot deadlock.

## Step 2 — B2: timing and the α–β fit (P = 4)

```bash
mpirun -np 4 python3 my_allreduce.py bench              # n = 2^4 ... 2^24 doubles
mpirun -np 4 python3 my_allreduce.py bench --max-exp 20 # if you are short on RAM
```

Methodology built into the code:
- one warm-up call;
- `MPI_Barrier` before each timed batch;
- many iterations for small n (up to 2000) and few for large n;
- the reported time is the **maximum across ranks** (`allreduce(..., op=MPI.MAX)`), because the slowest rank sets the step time.

It prints:
- **α** = T(2⁴) / (2(P−1)), from the smallest message (latency-dominated);
- **β** from the largest message, and the effective bandwidth **1/β in GB/s**;
- the **cross-over** n* = P·α/(8β) doubles, where the two terms are equal;
- a weighted least-squares fit over all sizes as a cross-check.

The results are saved to `../results/partB_alpha_beta.json`, which Part D reads.

## Step 3 — Change one thing: the rank count P

**Predict first**, from the model:
- small messages: time ∝ 2(P−1)·α, so it grows about linearly with P;
- large messages: time ∝ 2(P−1)/P, so it is nearly flat (1 → 1.5 → 1.75 for P = 2 → 4 → 8).

```bash
mpirun -np 2 python3 my_allreduce.py bench --out ../results/partB_alpha_beta_P2.json
mpirun -np 8 --oversubscribe python3 my_allreduce.py bench --out ../results/partB_alpha_beta_P8.json
```

On one machine the "network" is shared memory. Once P exceeds your cores, ranks time-share CPUs and compete for the same memory bandwidth, so the measurements grow faster than the model predicts. Explain this in the report.

## Step 4 — Sanity check

After your own code works, run the course's `examples/week4_mpi/ring_allreduce.c`. Usually `MPI_Allreduce` wins for **small** messages, because libraries switch algorithms by message size and topology (recursive doubling or shared-memory reductions with log P latency) and have no Python overhead per step.

## Fallback (no MPI available)

```bash
python3 my_allreduce.py simulate --procs 4
```

This runs the same ring with a Python list of 4 arrays, checks correctness against `np.sum`, and prints an analytic sweep with α = 2 µs and 1/β = 25 GB/s. State in the report that you used the fallback.

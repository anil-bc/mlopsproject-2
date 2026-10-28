# Part C — Make your own MLP data-parallel

**File to submit:** `dp_mlp.py`  ·  **Reused:** `mlp_from_scratch.py` (Assignment 1, unchanged)

`dp_mlp.py` imports `forward`, `backward`, `sgd_update`, … from the Assignment 1 MLP and `ring_allreduce` from Part B. It then trains the same 2 → 16 → 1 MLP on XOR blobs across P ranks.

## What each step of the code does

| Assignment requirement | Where in `dp_mlp.py` |
|---|---|
| 1. **Same start**: rank 0 initialises, `comm.Bcast` to everyone (no shared seed) | `if rank == 0: init_params(...)` / `else: np.empty(...)` → `comm.Bcast(p, root=0)` |
| 2. **Shard the batch**: disjoint 1/P slice of every global batch | `mine = idx[rank*local_bs:(rank+1)*local_bs]`, with G = 32 divisible by P |
| 3. **Sync gradients**: all-reduce *every* gradient, divide by P, then SGD | `allreduce_mean()` → `ring_allreduce(...)` / P → `sgd_update` |
| 4(i) **Replicas identical** | `comm.gather(W1)` to rank 0 → max difference |
| 4(ii) **DP = full batch** | rank 0 reruns single-process training with the same G, initial weights and data order → max per-epoch loss difference |

## Step 1 — Run with MPI

```bash
cd part_c_dp_mlp
mpirun -np 4 python3 dp_mlp.py                  # gradients through YOUR Part B ring
mpirun -np 4 python3 dp_mlp.py --use-library    # same, with comm.Allreduce
mpirun -np 2 python3 dp_mlp.py                  # any P that divides G=32
mpirun -np 4 python3 dp_mlp.py --epochs 500 --global-batch 64 --lr 0.3   # options
```

Expected end of the output:

```
 Claim (i)  replicas identical : max |W1_rank - W1_rank0|      = 0.000e+00
 Claim (ii) DP == full batch   : max |loss_DP - loss_single|   = ~1e-16  (per-epoch curve)
 RESULT: both claims hold (differences are at floating-point round-off level)
```

- (i) is exactly 0 because every rank applies the identical averaged gradient to identical weights.
- (ii) is about 10⁻¹⁶, not 0, because summing 4 shard means and dividing by 4 adds the numbers in a different order than one mean over 32 samples.

## Step 2 — Fallback without MPI

```bash
python3 dp_mlp.py --simulate 4     # 4 simulated ranks in one process, same checks
```

## Unequal-shards sentence (for the report)

Each rank's gradient is the **mean over its own shard**, so dividing the sum by P gives every shard equal weight. With unequal shards, samples in the smaller shards would be over-weighted, the result would no longer equal the global-batch mean gradient, and claim (ii) breaks. The fix is to weight each shard by its size (sum the per-sample gradients and divide by G).

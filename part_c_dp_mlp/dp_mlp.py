"""
dp_mlp.py  --  Assignment 2, Part C: "Make your own MLP data-parallel"

Takes the Assignment 1 MLP (mlp_from_scratch.py: forward, hand-derived backward,
plain SGD) and trains it data-parallel across P ranks:

  1. Same start   : rank 0 initialises the weights and comm.Bcast's them
                    (we do NOT rely on every rank using the same seed).
  2. Shard batch  : each rank takes a disjoint 1/P slice of every global mini-batch.
  3. Sync grads   : local backward -> ring all-reduce (Part B) of EVERY gradient
                    -> divide by P -> identical SGD update on every rank.
  4. Prove claims : (i)  replicas stay identical  (gather W1 to rank 0, max diff)
                    (ii) DP == single-process full-batch training with the same
                         global batch, same initial weights, same data order.

Run:
    mpirun -np 4 python3 dp_mlp.py                 # real MPI (uses Part B ring)
    mpirun -np 4 python3 dp_mlp.py --use-library   # same, but comm.Allreduce
    python3 dp_mlp.py --simulate 4                 # fallback: 4 simulated ranks, no MPI
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # mlp_from_scratch.py (Assignment 1)
sys.path.insert(0, os.path.join(HERE, "..", "part_b_mpi")) # my_allreduce.py (Part B)

from mlp_from_scratch import (make_xor_blobs, init_params, forward, backward,  # noqa: E402
                              bce_loss, sgd_update, train_val_split, accuracy)
from my_allreduce import ring_allreduce, ring_allreduce_simulated              # noqa: E402

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def epoch_order(n_samples, epoch, seed=123):
    """Same shuffled data order for every rank AND for the single-process run."""
    return np.random.default_rng(seed + epoch).permutation(n_samples)


def global_batches(order, G):
    """Yield index arrays of exactly G samples (the remainder is dropped)."""
    for start in range(0, len(order) - G + 1, G):
        yield order[start:start + G]


def reference_full_batch(X, y, params, G, epochs, lr):
    """Single process, same global batch G, same init, same data order."""
    W1, b1, W2, b2 = (p.copy() for p in params)
    curve = []
    for ep in range(epochs):
        losses = []
        for idx in global_batches(epoch_order(len(X), ep), G):
            a2, cache = forward(X[idx], W1, b1, W2, b2)
            losses.append(bce_loss(y[idx], a2))
            grads = backward(y[idx], cache, W1, b1, W2, b2)
            W1, b1, W2, b2 = sgd_update(W1, b1, W2, b2, *grads, lr)
        curve.append(float(np.mean(losses)))
    return (W1, b1, W2, b2), curve


# ---------------------------------------------------------------------------
# Real MPI data parallelism
# ---------------------------------------------------------------------------
def train_dp_mpi(args):
    comm = MPI.COMM_WORLD
    P, rank = comm.Get_size(), comm.Get_rank()
    G = args.global_batch
    if G % P != 0:
        if rank == 0:
            print(f"global batch {G} is not divisible by P={P}; choose e.g. {G - G % P or P}")
        sys.exit(1)
    local_bs = G // P

    # every rank can generate the same dataset (data generation is deterministic);
    # the WEIGHTS are what must come from rank 0
    X, y = make_xor_blobs(n=400, noise=0.25, seed=0)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=0.2, seed=0)

    # 1. Same start: rank 0 initialises, everyone receives via Bcast
    shapes = [(2, args.hidden), (1, args.hidden), (args.hidden, 1), (1, 1)]
    if rank == 0:
        params = list(init_params(hidden_dim=args.hidden, seed=0))
    else:
        params = [np.empty(s) for s in shapes]     # NOT initialised with any seed
    for p in params:
        comm.Bcast(p, root=0)
    init_copy = [p.copy() for p in params]
    W1, b1, W2, b2 = params

    def allreduce_mean(arr):
        flat = np.ascontiguousarray(arr, dtype=np.float64).ravel().copy()
        if args.use_library:
            out = np.empty_like(flat)
            comm.Allreduce(flat, out, op=MPI.SUM)
            flat = out
        else:
            ring_allreduce(flat, flat.size, comm)          # Part B ring
        return (flat / P).reshape(arr.shape)

    dp_curve = []
    t0 = MPI.Wtime()
    for ep in range(args.epochs):
        losses = []
        for idx in global_batches(epoch_order(len(X_train), ep), G):
            # 2. Shard: disjoint 1/P slice of this global batch
            mine = idx[rank * local_bs:(rank + 1) * local_bs]
            a2, cache = forward(X_train[mine], W1, b1, W2, b2)
            local_loss = bce_loss(y_train[mine], a2)
            dW1, db1, dW2, db2 = backward(y_train[mine], cache, W1, b1, W2, b2)

            # 3. Sync gradients: all-reduce every gradient, divide by P
            dW1, db1, dW2, db2 = (allreduce_mean(g) for g in (dW1, db1, dW2, db2))
            W1, b1, W2, b2 = sgd_update(W1, b1, W2, b2, dW1, db1, dW2, db2, args.lr)

            # mean of the shard losses == loss of the whole global batch
            losses.append(comm.allreduce(local_loss, op=MPI.SUM) / P)
        dp_curve.append(float(np.mean(losses)))
        if rank == 0 and (ep % max(1, args.epochs // 5) == 0 or ep == args.epochs - 1):
            print(f"[dp P={P}] epoch {ep:4d}  train_loss={dp_curve[-1]:.6f}")
    train_time = MPI.Wtime() - t0

    # 4(i). Replicas identical: gather every rank's W1 to rank 0
    all_W1 = comm.gather(W1, root=0)

    if rank == 0:
        diff_replicas = max(float(np.max(np.abs(w - all_W1[0]))) for w in all_W1)
        # 4(ii). DP == full batch: single-process run from the SAME initial weights
        ref_params, ref_curve = reference_full_batch(X_train, y_train, init_copy, G,
                                                     args.epochs, args.lr)
        diff_curve = float(np.max(np.abs(np.array(dp_curve) - np.array(ref_curve))))
        diff_w1 = float(np.max(np.abs(W1 - ref_params[0])))
        val_acc = accuracy(y_val, forward(X_val, W1, b1, W2, b2)[0])
        report(P, G, local_bs, args, diff_replicas, diff_curve, diff_w1, val_acc,
               train_time, "comm.Allreduce" if args.use_library else "Part B ring_allreduce")


# ---------------------------------------------------------------------------
# Fallback: P simulated ranks in one process (if MPI is not installed)
# ---------------------------------------------------------------------------
def train_dp_simulated(args):
    P, G = args.simulate, args.global_batch
    if G % P != 0:
        sys.exit(f"global batch {G} is not divisible by P={P}")
    local_bs = G // P
    X, y = make_xor_blobs(n=400, noise=0.25, seed=0)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=0.2, seed=0)

    root_params = init_params(hidden_dim=args.hidden, seed=0)            # "rank 0"
    replicas = [[p.copy() for p in root_params] for _ in range(P)]      # "Bcast"
    init_copy = [p.copy() for p in root_params]

    dp_curve = []
    for ep in range(args.epochs):
        losses = []
        for idx in global_batches(epoch_order(len(X_train), ep), G):
            per_rank_grads, local_losses = [], []
            for r in range(P):
                mine = idx[r * local_bs:(r + 1) * local_bs]
                W1, b1, W2, b2 = replicas[r]
                a2, cache = forward(X_train[mine], W1, b1, W2, b2)
                local_losses.append(bce_loss(y_train[mine], a2))
                per_rank_grads.append(backward(y_train[mine], cache, W1, b1, W2, b2))
            synced = []
            for g in range(4):                       # all-reduce each gradient
                bufs = [per_rank_grads[r][g].ravel().copy() for r in range(P)]
                ring_allreduce_simulated(bufs)
                synced.append([b.reshape(per_rank_grads[0][g].shape) / P for b in bufs])
            for r in range(P):
                replicas[r] = list(sgd_update(*replicas[r], *(synced[g][r] for g in range(4)),
                                              args.lr))
            losses.append(float(np.mean(local_losses)))
        dp_curve.append(float(np.mean(losses)))

    diff_replicas = max(float(np.max(np.abs(rep[0] - replicas[0][0]))) for rep in replicas)
    ref_params, ref_curve = reference_full_batch(X_train, y_train, init_copy, G, args.epochs, args.lr)
    diff_curve = float(np.max(np.abs(np.array(dp_curve) - np.array(ref_curve))))
    diff_w1 = float(np.max(np.abs(replicas[0][0] - ref_params[0])))
    val_acc = accuracy(y_val, forward(X_val, *replicas[0])[0])
    report(P, G, local_bs, args, diff_replicas, diff_curve, diff_w1, val_acc, float("nan"),
           "simulated ring (no MPI)")


def report(P, G, local_bs, args, diff_replicas, diff_curve, diff_w1, val_acc, t, how):
    print("\n" + "=" * 68)
    print(f" Data-parallel MLP: P = {P}, global batch G = {G}, per-rank batch = {local_bs}")
    print(f" epochs = {args.epochs}, lr = {args.lr}, hidden = {args.hidden}, all-reduce = {how}")
    print("=" * 68)
    print(f" Claim (i)  replicas identical : max |W1_rank - W1_rank0|      = {diff_replicas:.3e}")
    print(f" Claim (ii) DP == full batch   : max |loss_DP - loss_single|   = {diff_curve:.3e}"
          f"  (per-epoch curve)")
    print(f"                                 max |W1_DP - W1_single| (end) = {diff_w1:.3e}")
    print(f" final validation accuracy = {val_acc:.3f}"
          + (f"   training wall time = {t:.2f} s" if t == t else ""))
    ok = diff_replicas < 1e-12 and diff_curve < 1e-10
    print(" RESULT:", "both claims hold (differences are at floating-point round-off level)"
          if ok else "CHECK FAILED")


def main():
    ap = argparse.ArgumentParser(description="Assignment 2 Part C: data-parallel MLP")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--global-batch", type=int, default=32, help="must be divisible by P")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--use-library", action="store_true", help="use comm.Allreduce instead of ring")
    ap.add_argument("--simulate", type=int, default=0, help="P simulated ranks, no MPI needed")
    args = ap.parse_args()

    if args.simulate:
        train_dp_simulated(args)
    elif MPI is None:
        sys.exit("mpi4py not found. Install it, or run: python3 dp_mlp.py --simulate 4")
    else:
        train_dp_mpi(args)


if __name__ == "__main__":
    main()

"""
my_allreduce.py  --  Assignment 2, Part B: "Build your own ring all-reduce"

ring_allreduce(buf, n, comm) sums a length-n float64 vector across all ranks
IN PLACE using only point-to-point MPI_Sendrecv (no Allreduce / Reduce /
Reduce_scatter inside it). It handles n that is NOT divisible by P.

Run modes
---------
  B1 correctness (compare against MPI_Allreduce):
      mpirun -np 2 python3 my_allreduce.py test
      mpirun -np 3 python3 my_allreduce.py test
      mpirun -np 4 python3 my_allreduce.py test

  B2 timing + alpha-beta fit (P = 4):
      mpirun -np 4 python3 my_allreduce.py bench
      mpirun -np 4 python3 my_allreduce.py bench --max-exp 20   # smaller, if RAM is tight

  Fallback when MPI is NOT installed (pure NumPy simulation, P "ranks" in one process):
      python3 my_allreduce.py simulate --procs 4
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    from mpi4py import MPI
    HAVE_MPI = True
except ImportError:          # fallback path for machines without MPI
    MPI = None
    HAVE_MPI = False


# ---------------------------------------------------------------------------
# Chunk boundaries: chunk c covers [c*n//P, (c+1)*n//P)  -> works for any n
# ---------------------------------------------------------------------------
def chunk_bounds(c, n, P):
    return (c * n) // P, ((c + 1) * n) // P


# ---------------------------------------------------------------------------
# B1. Ring all-reduce from point-to-point calls only
# ---------------------------------------------------------------------------
def ring_allreduce(buf, n, comm):
    """In-place sum of buf[0:n] (float64) across all ranks of `comm`."""
    P = comm.Get_size()
    rank = comm.Get_rank()
    if P == 1:
        return buf

    right = (rank + 1) % P            # we always SEND to the right neighbour
    left = (rank - 1 + P) % P         # and RECEIVE from the left neighbour
    max_chunk = (n + P - 1) // P + 1
    tmp = np.empty(max_chunk, dtype=np.float64)

    # ---- Phase 1: reduce-scatter, P-1 steps ----------------------------------
    # At step s, rank r sends chunk (r - s) and receives chunk (r - s - 1),
    # adding the received data into its own copy. After P-1 steps rank r owns
    # the FULLY reduced chunk (r + 1) % P.
    for s in range(P - 1):
        send_c = (rank - s) % P
        recv_c = (rank - s - 1) % P
        s_lo, s_hi = chunk_bounds(send_c, n, P)
        r_lo, r_hi = chunk_bounds(recv_c, n, P)
        r_len = r_hi - r_lo
        comm.Sendrecv(sendbuf=buf[s_lo:s_hi], dest=right, sendtag=s,
                      recvbuf=tmp[:r_len], source=left, recvtag=s)
        buf[r_lo:r_hi] += tmp[:r_len]

    # ---- Phase 2: all-gather, P-1 steps ---------------------------------------
    # Circulate the fully reduced chunks: at step s rank r sends chunk (r + 1 - s)
    # and receives chunk (r - s), overwriting (not adding) its copy.
    for s in range(P - 1):
        send_c = (rank + 1 - s) % P
        recv_c = (rank - s) % P
        s_lo, s_hi = chunk_bounds(send_c, n, P)
        r_lo, r_hi = chunk_bounds(recv_c, n, P)
        comm.Sendrecv(sendbuf=buf[s_lo:s_hi], dest=right, sendtag=100 + s,
                      recvbuf=buf[r_lo:r_hi], source=left, recvtag=100 + s)
    return buf


# ---------------------------------------------------------------------------
# Fallback: the same ring algorithm simulated with a Python list of P arrays
# ---------------------------------------------------------------------------
def ring_allreduce_simulated(bufs):
    """bufs: list of P equal-length arrays, one per simulated rank. In place."""
    P, n = len(bufs), len(bufs[0])
    if P == 1:
        return bufs
    for s in range(P - 1):                       # reduce-scatter
        msgs = []
        for r in range(P):                       # every rank sends at the same step
            lo, hi = chunk_bounds((r - s) % P, n, P)
            msgs.append(bufs[r][lo:hi].copy())
        for r in range(P):                       # receive from left neighbour
            lo, hi = chunk_bounds((r - s - 1) % P, n, P)
            bufs[r][lo:hi] += msgs[(r - 1) % P]
    for s in range(P - 1):                       # all-gather
        msgs = []
        for r in range(P):
            lo, hi = chunk_bounds((r + 1 - s) % P, n, P)
            msgs.append(bufs[r][lo:hi].copy())
        for r in range(P):
            lo, hi = chunk_bounds((r - s) % P, n, P)
            bufs[r][lo:hi] = msgs[(r - 1) % P]
    return bufs


# ---------------------------------------------------------------------------
# B1 correctness test
# ---------------------------------------------------------------------------
def run_test(comm):
    P, rank = comm.Get_size(), comm.Get_rank()
    # include sizes that are NOT multiples of P (and one smaller than P)
    sizes = sorted({1, max(P - 1, 1), 7, 1000, 1001, 1003, 4099, 100003, 2 ** 20 + 1})
    if rank == 0:
        print(f"[B1] ring_allreduce correctness vs MPI_Allreduce, P = {P}")
        print(f"  {'n':>10} {'n % P':>6} {'max|ring - MPI_Allreduce|':>28} {'result':>8}")
    all_ok = True
    for n in sizes:
        rng = np.random.default_rng(seed=1234 + rank)     # seed by rank
        buf = rng.standard_normal(n)
        ref = np.empty_like(buf)
        comm.Allreduce(buf.copy(), ref, op=MPI.SUM)       # library result on a copy
        ring_allreduce(buf, n, comm)                      # our result, in place
        local = float(np.max(np.abs(buf - ref))) if n else 0.0
        diff = comm.allreduce(local, op=MPI.MAX)
        ok = diff < 1e-10
        all_ok &= ok
        if rank == 0:
            print(f"  {n:>10} {n % P:>6} {diff:>28.3e} {'PASS' if ok else 'FAIL':>8}")
    if rank == 0:
        print("  ALL PASS" if all_ok else "  SOME FAILED")
        print("  Note: the difference is ~1e-16..1e-15, not exactly 0, because floating-point")
        print("  addition is not associative and the library may sum in a different order.")
    return all_ok


# ---------------------------------------------------------------------------
# B2 timing: ring vs MPI_Allreduce, max across ranks, alpha-beta fit
# ---------------------------------------------------------------------------
def time_op(fn, comm, iters):
    fn()                                    # warm-up
    comm.Barrier()                          # line everyone up before the batch
    t0 = MPI.Wtime()
    for _ in range(iters):
        fn()
    local = (MPI.Wtime() - t0) / iters
    return comm.allreduce(local, op=MPI.MAX)   # slowest rank sets the step time


def run_bench(comm, max_exp, out_json):
    P, rank = comm.Get_size(), comm.Get_rank()
    exps = [e for e in (4, 8, 12, 16, 20, 24) if e <= max_exp]
    rows = []
    if rank == 0:
        print(f"[B2] Ring vs MPI_Allreduce, P = {P} ranks (times = max across ranks)")
        print(f"  {'n (doubles)':>12} {'bytes':>12} {'iters':>6} {'ring (us)':>12} "
              f"{'MPI_Allreduce (us)':>19} {'ring/MPI':>9}")
    for e in exps:
        n = 2 ** e
        iters = max(3, min(2000, (2 ** 22) // n))      # many iterations for small n
        # zeros: an in-place all-reduce repeated `iters` times multiplies the data by P
        # every call, so random data would overflow to inf. Summing zeros costs exactly
        # the same messages and additions, so the timing is unaffected.
        buf = np.zeros(n)
        out = np.empty_like(buf)
        t_ring = time_op(lambda: ring_allreduce(buf, n, comm), comm, iters)
        t_lib = time_op(lambda: comm.Allreduce(buf, out, op=MPI.SUM), comm, iters)
        rows.append((n, t_ring, t_lib))
        if rank == 0:
            print(f"  {'2^%d' % e:>12} {8 * n:>12} {iters:>6} {t_ring * 1e6:>12.2f} "
                  f"{t_lib * 1e6:>19.2f} {t_ring / t_lib:>9.2f}")
        del buf, out

    if rank == 0 and len(rows) >= 2:
        # T_ring(n) ~ 2(P-1) alpha + 2(P-1)/P * (8n) * beta
        n_small, t_small, _ = rows[0]
        n_large, t_large, _ = rows[-1]
        alpha = t_small / (2 * (P - 1))                                  # latency-dominated end
        beta = (t_large - 2 * (P - 1) * alpha) / (2 * (P - 1) / P * 8 * n_large)  # s/byte
        bw = 1.0 / beta / 1e9                                            # GB/s
        n_cross = P * alpha / (8 * beta)                                 # doubles where terms are equal
        # least-squares fit over all sizes as a cross-check, weighted by 1/T so that
        # small (latency) and large (bandwidth) messages count equally
        A = np.array([[2 * (P - 1), 2 * (P - 1) / P * 8 * n] for n, _, _ in rows])
        y = np.array([t for _, t, _ in rows])
        (a_ls, b_ls), *_ = np.linalg.lstsq(A / y[:, None], np.ones_like(y), rcond=None)
        print("\n  alpha-beta model:  T_ring(n) = 2(P-1)*alpha + 2(P-1)/P * (8n bytes) * beta")
        print(f"  alpha (from n=2^{int(np.log2(n_small))})      = {alpha * 1e6:8.3f} us")
        print(f"  beta  (from n=2^{int(np.log2(n_large))})     = {beta * 1e12:8.3f} ps/byte "
              f"-> effective bandwidth 1/beta = {bw:.2f} GB/s")
        print(f"  cross-over n* = P*alpha/(8*beta) = {n_cross:,.0f} doubles "
              f"(~{n_cross * 8 / 1024:,.1f} KiB): below it latency dominates, above it bandwidth")
        print(f"  least-squares fit (all sizes): alpha = {a_ls * 1e6:.3f} us, "
              f"1/beta = {(1 / b_ls / 1e9) if b_ls > 0 else float('nan'):.2f} GB/s")
        if out_json:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json, "w") as f:
                json.dump({"P": P, "alpha_s": alpha, "beta_s_per_byte": beta,
                           "bandwidth_GBps": bw, "crossover_doubles": n_cross,
                           "rows": [{"n": n, "ring_s": tr, "mpi_s": tl} for n, tr, tl in rows]},
                          f, indent=2)
            print(f"  saved alpha/beta to {out_json} (Part D reads this file)")


# ---------------------------------------------------------------------------
# Fallback mode: simulation + analytic sweep (alpha = 2 us, 1/beta = 25 GB/s)
# ---------------------------------------------------------------------------
def run_simulate(P):
    print(f"[FALLBACK] MPI not used -- simulating P = {P} ranks in one process")
    print(f"  {'n':>10} {'n % P':>6} {'max|ring - np.sum|':>20} {'result':>8}")
    for n in (1, 7, 1000, 1001, 4099, 100003):
        bufs = [np.random.default_rng(1234 + r).standard_normal(n) for r in range(P)]
        ref = np.sum(np.stack(bufs), axis=0)
        ring_allreduce_simulated(bufs)
        diff = max(float(np.max(np.abs(b - ref))) for b in bufs)
        print(f"  {n:>10} {n % P:>6} {diff:>20.3e} {'PASS' if diff < 1e-10 else 'FAIL':>8}")
    alpha, beta = 2e-6, 1 / 25e9
    print("\n  Analytic sweep, alpha = 2 us, 1/beta = 25 GB/s (~200 Gb/s InfiniBand)")
    for e in (4, 8, 12, 16, 20, 24):
        n = 2 ** e
        t = 2 * (P - 1) * alpha + 2 * (P - 1) / P * 8 * n * beta
        print(f"  n = 2^{e:<3} T_ring = {t * 1e6:12.2f} us")
    print(f"  cross-over n* = P*alpha/(8*beta) = {P * alpha / (8 * beta):,.0f} doubles")


def main():
    ap = argparse.ArgumentParser(description="Assignment 2 Part B: ring all-reduce")
    ap.add_argument("mode", choices=["test", "bench", "simulate"], nargs="?", default="test")
    ap.add_argument("--max-exp", type=int, default=24, help="largest message = 2^max_exp doubles")
    ap.add_argument("--procs", type=int, default=4, help="simulated ranks (simulate mode)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "results", "partB_alpha_beta.json"))
    args = ap.parse_args()

    if args.mode == "simulate":
        run_simulate(args.procs)
        return
    if not HAVE_MPI:
        sys.exit("mpi4py not found. Install MPI + mpi4py, or run: python3 my_allreduce.py simulate")
    comm = MPI.COMM_WORLD
    if args.mode == "test":
        ok = run_test(comm)
        sys.exit(0 if ok else 1)
    run_bench(comm, args.max_exp, args.out)


if __name__ == "__main__":
    main()

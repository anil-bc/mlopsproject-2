#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- build and run every part of Assignment 2, saving all output
#               to ./results/ so you can copy the numbers into report/report.md
#
#   ./run_all.sh          run everything (A, B, C, D)
#   ./run_all.sh a        only Part A (OpenMP)
#   ./run_all.sh b        only Part B (ring all-reduce)
#   ./run_all.sh c        only Part C (data-parallel MLP)
#   ./run_all.sh d        only Part D (cost estimator)
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RES="$ROOT/results"
mkdir -p "$RES"
WHAT="${1:-all}"

# Use the project's virtual environment if setup created one
if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
PY=python3

# ---- MPI launcher flags -------------------------------------------------------
# Open MPI refuses to start more ranks than cores unless --oversubscribe is given,
# and refuses to run as root (Docker) unless these two variables are set.
if [ "$(id -u)" = "0" ]; then
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
CORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
mpi_np() {   # usage: mpi_np <P> <command...>
  local p=$1; shift
  local extra=""
  if mpirun --version 2>&1 | grep -q "Open MPI" && [ "$p" -gt "$CORES" ]; then
    extra="--oversubscribe"
    echo "  (note: P=$p > $CORES cores -> using --oversubscribe; ranks share cores)"
  fi
  # shellcheck disable=SC2086
  mpirun $extra -np "$p" "$@"
}

banner() { echo; echo "=================================================================="; echo " $1"; echo "=================================================================="; }

part_a() {
  banner "PART A -- OpenMP kernels (results/partA_*.txt)"
  make -C "$ROOT/part_a_openmp" --no-print-directory
  cd "$ROOT/part_a_openmp"
  for i in 1 2 3; do
    echo "--- A1 dot product, run $i of 3 (watch the racy answer change) ---"
    ./omp_kernels dot | tee "$RES/partA_dot_run$i.txt"
  done
  echo "--- A2 GEMM (default T=64, order ikj) ---"
  ./omp_kernels gemm | tee "$RES/partA_gemm.txt"
  echo "--- A3 thread scaling ---"
  ./omp_kernels scale | tee "$RES/partA_scale.txt"
  echo "--- A3 change ONE thing: tile size T=32 (prediction first -- see README) ---"
  ./omp_kernels gemm 1024 32 | tee "$RES/partA_gemm_T32.txt"
  { lscpu 2>/dev/null | grep -iE "model name|cache" || sysctl hw.l1dcachesize hw.l2cachesize; } \
    | tee "$RES/partA_cpu_caches.txt"
}

part_b() {
  banner "PART B -- ring all-reduce (results/partB_*.txt)"
  cd "$ROOT/part_b_mpi"
  if ! "$PY" -c "import mpi4py" 2>/dev/null || ! command -v mpirun >/dev/null; then
    echo "MPI / mpi4py not found -> running the documented FALLBACK simulation"
    "$PY" my_allreduce.py simulate --procs 4 | tee "$RES/partB_fallback.txt"
    return
  fi
  for p in 2 3 4; do
    echo "--- B1 correctness, P=$p ---"
    mpi_np "$p" "$PY" my_allreduce.py test | tee "$RES/partB_test_P$p.txt"
  done
  echo "--- B2 timing + alpha/beta fit, P=4 (writes results/partB_alpha_beta.json) ---"
  mpi_np 4 "$PY" my_allreduce.py bench | tee "$RES/partB_bench_P4.txt"
  for p in 2 8; do
    echo "--- B2 change ONE thing: P=$p ---"
    mpi_np "$p" "$PY" my_allreduce.py bench --out "$RES/partB_alpha_beta_P$p.json" \
      | tee "$RES/partB_bench_P$p.txt"
  done
}

part_c() {
  banner "PART C -- data-parallel MLP (results/partC_*.txt)"
  cd "$ROOT/part_c_dp_mlp"
  if "$PY" -c "import mpi4py" 2>/dev/null && command -v mpirun >/dev/null; then
    mpi_np 4 "$PY" dp_mlp.py | tee "$RES/partC_dp_P4_ring.txt"
    mpi_np 4 "$PY" dp_mlp.py --use-library | tee "$RES/partC_dp_P4_library.txt"
  else
    echo "MPI not found -> simulated ranks"
    "$PY" dp_mlp.py --simulate 4 | tee "$RES/partC_dp_P4_simulated.txt"
  fi
}

part_d() {
  banner "PART D -- training cost estimator v2 (results/partD_estimate.txt)"
  cd "$ROOT/part_d_cost"
  "$PY" training_cost_estimator.py | tee "$RES/partD_assignment1_original.txt"
  "$PY" training_cost_estimator_v2.py | tee "$RES/partD_estimate.txt"
}

case "$WHAT" in
  a) part_a ;;
  b) part_b ;;
  c) part_c ;;
  d) part_d ;;
  all) part_a; part_b; part_c; part_d ;;
  *) echo "usage: $0 [all|a|b|c|d]"; exit 1 ;;
esac
banner "DONE -- all output saved in $RES"

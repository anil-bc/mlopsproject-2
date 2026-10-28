#!/usr/bin/env bash
# A3 thread-scaling sweep written as a SHELL LOOP (alternative to `./omp_kernels scale`,
# which does the same sweep inside the program with omp_set_num_threads()).
#
# Usage:  ./scaling.sh            (sweeps 1,2,4,... up to your logical core count)
#         ./scaling.sh 64 ijk     (same sweep, tiled GEMM with T=64 and in-tile order ijk)
set -euo pipefail
cd "$(dirname "$0")"
[ -x ./omp_kernels ] || make

T=${1:-64}
ORDER=${2:-ikj}

if [[ "$(uname -s)" == "Darwin" ]]; then
  MAXP=$(sysctl -n hw.logicalcpu)
else
  MAXP=$(nproc)
fi

THREADS=()
p=1
while [ "$p" -lt "$MAXP" ]; do THREADS+=("$p"); p=$((p * 2)); done
THREADS+=("$MAXP")

echo "Logical cores: $MAXP   thread counts: ${THREADS[*]}   T=$T order=$ORDER"
echo
printf "%-9s %-14s %-14s\n" "threads" "dot time (s)" "GEMM GFLOP/s"
for t in "${THREADS[@]}"; do
  dot=$(OMP_NUM_THREADS=$t ./omp_kernels dot $((1<<24)) 1024 | awk '/4. reduction/ {print $NF}')
  gf=$(OMP_NUM_THREADS=$t ./omp_kernels gemm 1024 "$T" "$ORDER" | awk '/3. tiled/ {print $(NF-2)}')
  printf "%-9s %-14s %-14s\n" "$t" "$dot" "$gf"
done
echo
echo "Speedup S(p) = T1/Tp and efficiency E(p) = S(p)/p are printed directly by: ./omp_kernels scale"

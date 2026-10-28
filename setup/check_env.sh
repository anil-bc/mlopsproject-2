#!/usr/bin/env bash
# Checks that every tool the assignment needs is installed and that OpenMP
# really spawns threads (a build without OpenMP runs serially and gives
# meaningless numbers).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"

ok=1
check() { if eval "$2" >/dev/null 2>&1; then echo "  [OK]   $1"; else echo "  [MISS] $1"; ok=0; fi; }

echo "Toolchain check:"
check "C compiler (gcc/clang)"    "command -v gcc || command -v clang"
check "make"                      "command -v make"
check "mpirun (Open MPI / MPICH)" "command -v mpirun"
check "python3 >= 3.8"            "python3 -c 'import sys; assert sys.version_info >= (3, 8)'"
check "numpy"                     "python3 -c 'import numpy'"
check "mpi4py"                    "python3 -c 'import mpi4py'"

echo
echo "OpenMP thread check (should print your logical core count, not 1):"
TMP=$(mktemp -d)
cat > "$TMP/t.c" <<'EOF'
#include <stdio.h>
#include <omp.h>
int main(void){ printf("  omp_get_max_threads() = %d\n", omp_get_max_threads()); return 0; }
EOF
if [[ "$(uname -s)" == "Darwin" ]]; then
  LIBOMP=$(brew --prefix libomp)
  clang -Xpreprocessor -fopenmp -I"$LIBOMP/include" -L"$LIBOMP/lib" -lomp "$TMP/t.c" -o "$TMP/t" && "$TMP/t" || ok=0
else
  gcc -fopenmp "$TMP/t.c" -o "$TMP/t" && "$TMP/t" || ok=0
fi
echo "  logical cores = $(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"
rm -rf "$TMP"

echo
echo "MPI check:"
if [ "$(id -u)" = "0" ]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
mpirun --version 2>&1 | head -1 | sed 's/^/  /'
mpirun -np 2 python3 -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print(f'  hello from rank {c.Get_rank()} of {c.Get_size()}')" || ok=0

echo
[ $ok = 1 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED -- see README 'Troubleshooting'"; exit 1; }

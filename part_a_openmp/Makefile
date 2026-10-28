# Part A build file. Usage:
#   make            -> builds ./omp_kernels with -O2 (the flags are printed by the program)
#   make OPT=-O3    -> rebuild with -O3 to compare
#   make run        -> runs dot + gemm + scale and saves output to ../results/partA_*.txt
#   make clean
UNAME_S := $(shell uname -s)
OPT     ?= -O2

ifeq ($(UNAME_S),Darwin)
  LIBOMP  := $(shell brew --prefix libomp 2>/dev/null)
  CC      := clang
  OMPFLAG := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  LDFLAGS := -L$(LIBOMP)/lib -lomp
else
  ifeq ($(origin CC),default)
    CC    := gcc
  endif
  OMPFLAG := -fopenmp
  LDFLAGS :=
endif

FLAGS := $(OPT) $(OMPFLAG)

omp_kernels: omp_kernels.c
	$(CC) $(FLAGS) -DBUILD_FLAGS='"$(CC) $(FLAGS)"' -o $@ $< $(LDFLAGS) -lm

run: omp_kernels
	mkdir -p ../results
	./omp_kernels dot   | tee ../results/partA_dot_run1.txt
	./omp_kernels dot   | tee ../results/partA_dot_run2.txt
	./omp_kernels dot   | tee ../results/partA_dot_run3.txt
	./omp_kernels gemm  | tee ../results/partA_gemm.txt
	./omp_kernels scale | tee ../results/partA_scale.txt

clean:
	rm -f omp_kernels

.PHONY: run clean

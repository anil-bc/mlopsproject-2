/*
 * omp_kernels.c  --  Assignment 2, Part A: "OpenMP: first correct, then fast"
 *
 * One program, four modes (choose with the first command-line argument):
 *
 *   ./omp_kernels dot   [N] [N_SYNC]        A1: dot product four ways (racy/critical/atomic/reduction)
 *   ./omp_kernels gemm  [N] [T] [ORDER]     A2: naive serial vs naive+OpenMP vs tiled+OpenMP GEMM
 *   ./omp_kernels scale [N_DOT] [N_GEMM] [T] [ORDER]
 *                                           A3: thread sweep 1,2,4,.. up to logical cores
 *   ./omp_kernels all                       runs dot, gemm and scale with defaults
 *
 *   ORDER = ikj (default) or ijk  -> loop order *inside* one tile (for the
 *                                     "change one thing" experiment)
 *
 * Build (Linux / WSL):   gcc -O2 -fopenmp -o omp_kernels omp_kernels.c
 * Build (macOS):         clang -O2 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include \
 *                              -L$(brew --prefix libomp)/lib -lomp -o omp_kernels omp_kernels.c
 * (or simply run `make` in this folder -- the Makefile picks the right command)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown (compile with -DBUILD_FLAGS=\"...\")"
#endif

#define REPEATS 3   /* timed calls, after 1 untimed warm-up (Assignment 1 recipe) */

/* ------------------------------------------------------------------------- */
/* small helpers                                                             */
/* ------------------------------------------------------------------------- */
static double *alloc_vec(size_t n) {
    double *p = (double *)malloc(n * sizeof(double));
    if (!p) { fprintf(stderr, "malloc failed for %zu doubles\n", n); exit(1); }
    return p;
}

static void print_header(void) {
    printf("============================================================\n");
    printf(" omp_kernels  |  omp_get_max_threads() = %d  |  omp_get_num_procs() = %d\n",
           omp_get_max_threads(), omp_get_num_procs());
    printf(" compiler     : %s\n", __VERSION__);
    printf(" build flags  : %s\n", BUILD_FLAGS);
    printf("============================================================\n");
}

/* ------------------------------------------------------------------------- */
/* A1. Dot product four ways                                                 */
/* ------------------------------------------------------------------------- */
static double dot_racy(const double *a, const double *b, long n) {
    /* `volatile` stops GCC/Clang at -O2 from keeping `sum` in a register and
       writing it back only once per thread (which would hide the race behind a
       deterministic "last writer wins" result). With volatile, every iteration
       really does load -> add -> store on the SHARED variable, like the slides. */
    volatile double sum = 0.0;
    /* DELIBERATE BUG: every thread does an unsynchronised read-modify-write
       on the shared variable `sum` -> lost updates, non-deterministic result. */
    #pragma omp parallel for
    for (long i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static double dot_critical(const double *a, const double *b, long n) {
    double sum = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        #pragma omp critical
        sum += a[i] * b[i];          /* correct, but fully serialised (one lock) */
    }
    return sum;
}

static double dot_atomic(const double *a, const double *b, long n) {
    double sum = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        double prod = a[i] * b[i];
        #pragma omp atomic
        sum += prod;                 /* correct, hardware atomic -> cache-line contention */
    }
    return sum;
}

static double dot_reduction(const double *a, const double *b, long n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++)
        sum += a[i] * b[i];          /* private partial sums, combined once at the end */
    return sum;
}

typedef double (*dot_fn)(const double *, const double *, long);

static void run_dot(long n, long n_sync) {
    double *a = alloc_vec(n), *b = alloc_vec(n);
    #pragma omp parallel for
    for (long i = 0; i < n; i++) { a[i] = 1.0; b[i] = 0.5; }

    struct { const char *name; dot_fn fn; long len; } v[4] = {
        {"1. racy      ", dot_racy,      n},
        {"2. critical  ", dot_critical,  n_sync},
        {"3. atomic    ", dot_atomic,    n_sync},
        {"4. reduction ", dot_reduction, n},
    };

    printf("\n[A1] Dot product, N = %ld (2^%d), threads = %d\n",
           n, (int)round(log2((double)n)), omp_get_max_threads());
    if (n_sync != n)
        printf("     NOTE: critical/atomic use smaller N_SYNC = %ld (2^%d) to keep runtime < ~10 s\n",
               n_sync, (int)round(log2((double)n_sync)));
    printf("     a[i]=1.0, b[i]=0.5  ->  exact answer = N/2\n\n");
    printf("  %-14s %18s %18s %14s %12s\n", "variant", "result", "expected", "abs error", "time (s)");

    for (int k = 0; k < 4; k++) {
        double expected = (double)v[k].len / 2.0;
        double t0 = omp_get_wtime();
        double r  = v[k].fn(a, b, v[k].len);
        double t  = omp_get_wtime() - t0;
        printf("  %-14s %18.1f %18.1f %14.1f %12.6f\n",
               v[k].name, r, expected, fabs(r - expected), t);
    }
    free(a); free(b);
}

/* ------------------------------------------------------------------------- */
/* A2. GEMM: C = A * B, N x N row-major doubles                              */
/* ------------------------------------------------------------------------- */
static void gemm_naive_serial(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
}

static void gemm_naive_omp(const double *A, const double *B, double *C, int n) {
    /* Parallelised the OUTER i loop: each thread owns whole rows of C, so no two
       threads ever write the same C[i][j] -> no race on C (s is loop-private). */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
}

/* Tiled + OpenMP. Parallel over (ii, jj) tiles of C: every tile of C is owned by
   exactly one thread, so updates to C are race-free. The kk loop walks the
   shared dimension; inside a tile we use loop order `order` (ikj or ijk). */
static void gemm_tiled_omp(const double *A, const double *B, double *C, int n, int T, int ikj) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += T)
        for (int jj = 0; jj < n; jj += T)
            for (int kk = 0; kk < n; kk += T) {
                int imax = ii + T < n ? ii + T : n;
                int jmax = jj + T < n ? jj + T : n;
                int kmax = kk + T < n ? kk + T : n;
                if (ikj) {
                    /* i-k-j: innermost loop streams contiguously along rows of B and C */
                    for (int i = ii; i < imax; i++)
                        for (int k = kk; k < kmax; k++) {
                            double aik = A[(size_t)i * n + k];
                            for (int j = jj; j < jmax; j++)
                                C[(size_t)i * n + j] += aik * B[(size_t)k * n + j];
                        }
                } else {
                    /* i-j-k: innermost loop strides down a column of B (stride n) */
                    for (int i = ii; i < imax; i++)
                        for (int j = jj; j < jmax; j++) {
                            double s = C[(size_t)i * n + j];
                            for (int k = kk; k < kmax; k++)
                                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                            C[(size_t)i * n + j] = s;
                        }
                }
            }
}

static double max_abs_diff(const double *x, const double *y, size_t len) {
    double m = 0.0;
    for (size_t i = 0; i < len; i++) {
        double d = fabs(x[i] - y[i]);
        if (d > m) m = d;
    }
    return m;
}

/* 1 untimed warm-up + average of REPEATS timed calls */
typedef enum { NAIVE_SERIAL, NAIVE_OMP, TILED_OMP } gemm_kind;

static double time_gemm(gemm_kind kind, const double *A, const double *B, double *C,
                        int n, int T, int ikj) {
    double total = 0.0;
    for (int r = 0; r <= REPEATS; r++) {
        double t0 = omp_get_wtime();
        if (kind == NAIVE_SERIAL)   gemm_naive_serial(A, B, C, n);
        else if (kind == NAIVE_OMP) gemm_naive_omp(A, B, C, n);
        else                        gemm_tiled_omp(A, B, C, n, T, ikj);
        double t = omp_get_wtime() - t0;
        if (r > 0) total += t;      /* r == 0 is the warm-up */
    }
    return total / REPEATS;
}

static void fill_matrices(double *A, double *B, int n) {
    srand(42);
    for (size_t i = 0; i < (size_t)n * n; i++) {
        A[i] = (double)rand() / RAND_MAX - 0.5;
        B[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static void run_gemm(int n, int T, int ikj) {
    size_t sz = (size_t)n * n;
    double *A = alloc_vec(sz), *B = alloc_vec(sz);
    double *C1 = alloc_vec(sz), *C2 = alloc_vec(sz), *C3 = alloc_vec(sz);
    fill_matrices(A, B, n);
    double flops = 2.0 * (double)n * n * n;

    printf("\n[A2] GEMM C = A*B, N = %d, tile T = %d, in-tile order = %s, threads = %d\n",
           n, T, ikj ? "ikj" : "ijk", omp_get_max_threads());
    printf("     timing: 1 warm-up + mean of %d timed calls; GFLOP/s = 2N^3 / t\n", REPEATS);
    printf("     tile working set ~ 3*T^2*8 bytes = %.1f KiB\n\n", 3.0 * T * T * 8 / 1024.0);

    double t1 = time_gemm(NAIVE_SERIAL, A, B, C1, n, T, ikj);
    double t2 = time_gemm(NAIVE_OMP,    A, B, C2, n, T, ikj);
    double t3 = time_gemm(TILED_OMP,    A, B, C3, n, T, ikj);
    double d2 = max_abs_diff(C1, C2, sz), d3 = max_abs_diff(C1, C3, sz);

    printf("  %-22s %12s %12s %16s %8s\n", "version", "time (s)", "GFLOP/s", "max|diff| vs 1", "check");
    printf("  %-22s %12.4f %12.2f %16s %8s\n", "1. naive serial", t1, flops / t1 / 1e9, "-", "ref");
    printf("  %-22s %12.4f %12.2f %16.3e %8s\n", "2. naive + OpenMP", t2, flops / t2 / 1e9, d2, d2 < 1e-9 ? "PASS" : "FAIL");
    printf("  %-22s %12.4f %12.2f %16.3e %8s\n", "3. tiled + OpenMP", t3, flops / t3 / 1e9, d3, d3 < 1e-9 ? "PASS" : "FAIL");

    free(A); free(B); free(C1); free(C2); free(C3);
}

/* ------------------------------------------------------------------------- */
/* A3. Thread-scaling study (sweep written inside the program)               */
/* ------------------------------------------------------------------------- */
static void run_scale(long n_dot, int n_gemm, int T, int ikj) {
    int maxp = omp_get_num_procs();      /* logical cores */
    int counts[64], nc = 0;
    for (int p = 1; p < maxp; p *= 2) counts[nc++] = p;
    counts[nc++] = maxp;                 /* always include the full core count */

    /* ---- dot product (reduction) ---- */
    double *a = alloc_vec(n_dot), *b = alloc_vec(n_dot);
    for (long i = 0; i < n_dot; i++) { a[i] = 1.0; b[i] = 0.5; }

    printf("\n[A3] Thread scaling -- reduction dot product, N = %ld (2^%d)\n",
           n_dot, (int)round(log2((double)n_dot)));
    printf("  %8s %14s %10s %12s %12s\n", "threads", "time (s)", "speedup", "efficiency", "GB/s");
    double T1 = 0.0;
    for (int c = 0; c < nc; c++) {
        omp_set_num_threads(counts[c]);
        double total = 0.0;
        for (int r = 0; r <= 10; r++) {        /* 1 warm-up + 10 timed (kernel is fast) */
            double t0 = omp_get_wtime();
            volatile double s = dot_reduction(a, b, n_dot); (void)s;
            double t = omp_get_wtime() - t0;
            if (r > 0) total += t;
        }
        double t = total / 10.0;
        if (c == 0) T1 = t;
        double S = T1 / t;
        printf("  %8d %14.6f %10.2f %12.2f %12.2f\n",
               counts[c], t, S, S / counts[c], 16.0 * n_dot / t / 1e9);
    }
    free(a); free(b);

    /* ---- tiled GEMM ---- */
    size_t sz = (size_t)n_gemm * n_gemm;
    double *A = alloc_vec(sz), *B = alloc_vec(sz), *C = alloc_vec(sz);
    fill_matrices(A, B, n_gemm);
    double flops = 2.0 * (double)n_gemm * n_gemm * n_gemm;

    printf("\n[A3] Thread scaling -- tiled GEMM, N = %d, T = %d, order = %s\n",
           n_gemm, T, ikj ? "ikj" : "ijk");
    printf("  %8s %14s %10s %12s %12s\n", "threads", "time (s)", "speedup", "efficiency", "GFLOP/s");
    for (int c = 0; c < nc; c++) {
        omp_set_num_threads(counts[c]);
        double t = time_gemm(TILED_OMP, A, B, C, n_gemm, T, ikj);
        if (c == 0) T1 = t;
        double S = T1 / t;
        printf("  %8d %14.4f %10.2f %12.2f %12.2f\n",
               counts[c], t, S, S / counts[c], flops / t / 1e9);
    }
    free(A); free(B); free(C);
    omp_set_num_threads(maxp);
}

/* ------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "all";
    print_header();

    if (strcmp(mode, "dot") == 0) {
        long n      = argc > 2 ? atol(argv[2]) : (1L << 24);
        long n_sync = argc > 3 ? atol(argv[3]) : n;
        run_dot(n, n_sync);
    } else if (strcmp(mode, "gemm") == 0) {
        int n   = argc > 2 ? atoi(argv[2]) : 1024;
        int T   = argc > 3 ? atoi(argv[3]) : 64;
        int ikj = argc > 4 ? strcmp(argv[4], "ijk") != 0 : 1;
        run_gemm(n, T, ikj);
    } else if (strcmp(mode, "scale") == 0) {
        long nd = argc > 2 ? atol(argv[2]) : (1L << 24);
        int  ng = argc > 3 ? atoi(argv[3]) : 1024;
        int  T  = argc > 4 ? atoi(argv[4]) : 64;
        int ikj = argc > 5 ? strcmp(argv[5], "ijk") != 0 : 1;
        run_scale(nd, ng, T, ikj);
    } else if (strcmp(mode, "all") == 0) {
        run_dot(1L << 24, 1L << 24);
        run_gemm(1024, 64, 1);
        run_scale(1L << 24, 1024, 64, 1);
    } else {
        fprintf(stderr, "usage: %s [dot|gemm|scale|all] [args...]\n", argv[0]);
        return 1;
    }
    return 0;
}

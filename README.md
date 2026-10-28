# MLOps Assignment 2 — From "It Runs in Parallel" to "I Know Why It Doesn't Scale"

Week 3 (OpenMP and parallelising ML kernels) + Week 4 (MPI for distributed ML systems).
Every part is runnable, and all of it runs from one command: `./run_all.sh`.

| Part | Folder | File(s) to submit | What it does |
|---|---|---|---|
| A | [`part_a_openmp/`](part_a_openmp/README.md) | `omp_kernels.c` | Dot product 4 ways (race → reduction), naive/OpenMP/tiled GEMM, thread-scaling study |
| B | [`part_b_mpi/`](part_b_mpi/README.md) | `my_allreduce.py` | Ring all-reduce built from `Sendrecv` only, correctness vs `MPI_Allreduce`, α–β fit |
| C | [`part_c_dp_mlp/`](part_c_dp_mlp/README.md) | `dp_mlp.py` | Assignment 1 MLP made data-parallel (Bcast, shard, all-reduce), proves 2 claims |
| D | [`part_d_cost/`](part_d_cost/README.md) | `training_cost_estimator_v2.py` | Assignment 1 estimator + communication term, 8/64/1024 GPUs |
| E | [`report/`](report/report.md) | `report.md` | 2-page report with all numbers + synthesis |

```
mlopsproject-2/
├── README.md                  ← you are here (step-by-step guide)
├── run_all.sh                 ← builds + runs Parts A–D, saves output to results/
├── requirements.txt           ← numpy, mpi4py, matplotlib
├── setup/
│   ├── install_ubuntu_wsl.sh  ← one-time setup: Ubuntu / Debian / WSL2 on Windows
│   ├── install_macos.sh       ← one-time setup: macOS (Homebrew)
│   └── check_env.sh           ← verifies compiler, OpenMP threads, MPI, mpi4py
├── part_a_openmp/   omp_kernels.c, Makefile, scaling.sh, README.md
├── part_b_mpi/      my_allreduce.py, README.md
├── part_c_dp_mlp/   dp_mlp.py, mlp_from_scratch.py (Assignment 1), README.md
├── part_d_cost/     training_cost_estimator_v2.py, training_cost_estimator.py (Assignment 1), README.md
├── report/report.md
├── results/                   ← created by run_all.sh (raw output of every run)
├── Dockerfile                 ← optional: run everything in a container
└── .github/workflows/ci.yml   ← GitHub Actions: builds + runs everything on every push
```

---

## Step 0 — Pick your environment

| Your computer | Use | Why |
|---|---|---|
| **Windows 10/11** | **WSL2 + Ubuntu** (Step 1A) | OpenMP with `gcc -fopenmp` and Open MPI work out of the box in Linux. Native Windows needs MSVC + MS-MPI, which is much harder. |
| Ubuntu / Debian | Step 1B | Native |
| macOS | Step 1C | Apple clang needs `libomp` for OpenMP |
| Any OS with Docker | Step 1D | No local installs at all |

---

## Step 1A — Windows: install WSL2 + Ubuntu (one time)

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-24.04
```

**What it does:** enables the Windows Subsystem for Linux, downloads Ubuntu 24.04 and sets it up. **Restart** when asked, then open **"Ubuntu"** from the Start menu and create a Linux username and password.

Check it worked:

```powershell
wsl -l -v          # should show Ubuntu-24.04 with VERSION 2
```

From now on, **type every command in the Ubuntu terminal**, not PowerShell. Your Windows drives are visible under `/mnt/c`, `/mnt/f`, and so on. For speed, keep the repo in the Linux home folder (`~`), not on `/mnt/...`.

Then continue with **Step 1B**.

## Step 1B — Ubuntu / WSL2: clone and install dependencies

```bash
# 1. Get git (usually already installed) and clone the repo
sudo apt-get update && sudo apt-get install -y git
cd ~
git clone https://github.com/anil-bc/mlopsproject-2.git
cd mlopsproject-2

# 2. One-time setup script
bash setup/install_ubuntu_wsl.sh
```

**What `install_ubuntu_wsl.sh` does, step by step:**

| Step | Command inside the script | Why |
|---|---|---|
| 1 | `sudo apt-get install build-essential` | gcc, make; gcc includes OpenMP (`-fopenmp`) |
| 2 | `sudo apt-get install openmpi-bin libopenmpi-dev` | `mpirun` launcher + MPI headers/libraries (needed to build mpi4py) |
| 3 | `sudo apt-get install python3-venv python3-dev` | Python virtual environment + headers to compile mpi4py |
| 4 | `python3 -m venv .venv` | Isolated Python environment for this project |
| 5 | `pip install -r requirements.txt` | numpy, **mpi4py** (compiled against Open MPI), matplotlib |
| 6 | `bash setup/check_env.sh` | Verifies everything (see Step 2) |

Activate the environment **every time you open a new terminal**:

```bash
source .venv/bin/activate        # your prompt now starts with (.venv)
```

## Step 1C — macOS

```bash
git clone https://github.com/anil-bc/mlopsproject-2.git && cd mlopsproject-2
bash setup/install_macos.sh      # brew install libomp open-mpi + venv + pip install
source .venv/bin/activate
```

## Step 1D — Docker (any OS)

```bash
git clone https://github.com/anil-bc/mlopsproject-2.git && cd mlopsproject-2
docker build -t mlops-a2 .
docker run --rm -v "$PWD/results:/app/results" mlops-a2              # runs everything
docker run --rm -v "$PWD/results:/app/results" mlops-a2 ./run_all.sh a   # just Part A
```

---

## Step 2 — Verify the toolchain

```bash
bash setup/check_env.sh
```

Expected output (the core count is your machine's):

```
Toolchain check:
  [OK]   C compiler (gcc/clang)
  [OK]   make
  [OK]   mpirun (Open MPI / MPICH)
  [OK]   python3 >= 3.8
  [OK]   numpy
  [OK]   mpi4py
OpenMP thread check (should print your logical core count, not 1):
  omp_get_max_threads() = 8
MPI check:
  hello from rank 0 of 2
  hello from rank 1 of 2
ALL CHECKS PASSED
```

**Why this matters:** a program compiled *without* OpenMP still runs, serially, and prints numbers that are meaningless for this assignment. `omp_get_max_threads()` must equal your core count.

---

## Step 3 — Run everything with one command

```bash
./run_all.sh            # all parts (~2–5 minutes)
./run_all.sh a          # only Part A   (also: b, c, d)
```

Every run's output is printed **and** saved in `results/`:

| File | Contents |
|---|---|
| `partA_dot_run1..3.txt` | A1: four dot-product variants, 3 runs (racy answer changes each run) |
| `partA_gemm.txt`, `partA_gemm_T32.txt` | A2: GEMM GFLOP/s at T=64, and the "change one thing" run at T=32 |
| `partA_scale.txt` | A3: speedup and efficiency tables |
| `partA_cpu_caches.txt` | Your L1/L2 sizes (for the tile-size prediction) |
| `partB_test_P2/3/4.txt` | B1: correctness vs `MPI_Allreduce` |
| `partB_bench_P2/4/8.txt`, `partB_alpha_beta.json` | B2: timing tables, α, β, cross-over |
| `partC_dp_P4_ring.txt`, `partC_dp_P4_library.txt` | C: proof of claims (i) and (ii) |
| `partD_estimate.txt` | D: T_compute / T_comm / efficiency for N = 8, 64, 1024 |

Then copy your numbers into `report/report.md`. It currently holds results from a 2-core cloud VM.

---

## Step 4 — Run each part by hand (step by step)

Each folder's README explains the theory and expected output in detail. Here are the commands.

### Part A — OpenMP (`part_a_openmp/`)

```bash
cd part_a_openmp
make                              # gcc -O2 -fopenmp -o omp_kernels omp_kernels.c
./omp_kernels dot                 # A1: racy / critical / atomic / reduction (N = 2^24)
./omp_kernels dot                 # run it 2 more times -> racy answer changes
./omp_kernels dot 16777216 1048576   # if critical takes > 10 s: smaller N for critical/atomic only
./omp_kernels gemm                # A2: naive vs naive+OpenMP vs tiled (N=1024, T=64)
./omp_kernels scale               # A3: thread sweep 1,2,4,... with speedup + efficiency
./scaling.sh                      # A3 again, as a shell loop over OMP_NUM_THREADS
lscpu | grep -i cache             # L1/L2 sizes -> predict, THEN:
./omp_kernels gemm 1024 32        # change ONE thing: tile size 64 -> 32
# (or instead: ./omp_kernels gemm 1024 64 ijk  -> change loop order inside a tile)
make clean && make OPT=-O3        # optional: compare -O2 vs -O3 (state flags in the report)
cd ..
```

### Part B — Ring all-reduce (`part_b_mpi/`)

```bash
cd part_b_mpi
mpirun -np 2 python3 my_allreduce.py test     # B1 correctness, P = 2
mpirun -np 3 python3 my_allreduce.py test     # P = 3 (n not divisible by P)
mpirun -np 4 python3 my_allreduce.py test     # P = 4
mpirun -np 4 python3 my_allreduce.py bench    # B2 timing + alpha/beta fit -> results/partB_alpha_beta.json
mpirun -np 8 --oversubscribe python3 my_allreduce.py bench --out ../results/partB_alpha_beta_P8.json
python3 my_allreduce.py simulate --procs 4    # FALLBACK if MPI cannot be installed
cd ..
```

> If you have fewer cores than `-np`, Open MPI says "not enough slots". Add `--oversubscribe` and mention it in the report.

### Part C — Data-parallel MLP (`part_c_dp_mlp/`)

```bash
cd part_c_dp_mlp
mpirun -np 4 python3 dp_mlp.py                 # uses the Part B ring all-reduce
mpirun -np 4 python3 dp_mlp.py --use-library   # same with comm.Allreduce
python3 dp_mlp.py --simulate 4                 # fallback, no MPI
python3 mlp_from_scratch.py                    # the original Assignment 1 single-process MLP
cd ..
```

### Part D — Cost estimator (`part_d_cost/`)

```bash
cd part_d_cost
python3 training_cost_estimator.py       # Assignment 1 original (perfect scaling)
python3 training_cost_estimator_v2.py    # + communication term (reads your Part B alpha/beta)
cd ..
```

---

## Step 5 — Push your results to GitHub

```bash
git add results/ report/report.md
git commit -m "Assignment 2: my measured results"
git push
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) rebuilds and reruns every part on a clean Ubuntu runner on every push. Download the outputs from **Actions → latest run → Artifacts → results**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `omp_get_max_threads() = 1` | Compiled without `-fopenmp`, or `OMP_NUM_THREADS=1` is set. Run `unset OMP_NUM_THREADS; make clean; make`. |
| `There are not enough slots available` | More ranks than cores: add `--oversubscribe` (Open MPI). `run_all.sh` does this automatically. |
| `mpirun has detected an attempt to run as root` | Docker/root only: `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` |
| `pip install mpi4py` fails with `mpicc not found` | Install MPI first: `sudo apt-get install openmpi-bin libopenmpi-dev`, then reinstall. |
| `ModuleNotFoundError: numpy` | You forgot `source .venv/bin/activate`. |
| macOS: `'omp.h' file not found` | `brew install libomp`, then use the Makefile (it adds the libomp include/lib paths). |
| `./run_all.sh: Permission denied` | `chmod +x run_all.sh part_a_openmp/scaling.sh setup/*.sh` |
| WSL very slow | Keep the repo in `~/` (Linux filesystem), not `/mnt/c/...` |
| Part B bench runs out of memory | `mpirun -np 4 python3 my_allreduce.py bench --max-exp 20` (largest message 2^20 instead of 2^24) |

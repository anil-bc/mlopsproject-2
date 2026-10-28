"""
training_cost_estimator.py

Back-of-envelope training cost estimator, per the Week 1 rules of thumb:
  - Compute per training step:  6 * P * T   FLOPs
  - Optimizer memory:           16 * P      bytes
    (FP16 weights [2B] + FP16 grads [2B] + FP32 Adam m [4B] + FP32 Adam v [4B]
     + FP32 master weights [4B] = 16 bytes/param -- the standard mixed-precision
     Adam accounting.)

Model chosen: Llama 3 8B (Meta, released April 2024)
  P = 8e9 parameters
  T = 15e12 training tokens
  Source: Meta AI blog, "Introducing Meta Llama 3" (ai.meta.com/blog/meta-llama-3/)
          -- "Llama 3 is pretrained on over 15T tokens... 8B and 70B parameters."

GPU chosen: NVIDIA A100 (80GB SXM)
  Peak throughput: 312 TFLOP/s BF16 (dense Tensor Core, no sparsity)
  Source: NVIDIA A100 datasheet (nvidia.com/content/dam/en-zz/Solutions/
          Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)

Assumed utilization: 40% (typical realistic MFU for large dense-model training)
"""

# ---------------------------------------------------------------------------
# 1. Inputs
# ---------------------------------------------------------------------------
P = 8e9          # parameter count -- Llama 3 8B
T = 15e12        # training tokens -- Meta AI blog, "over 15T tokens"

GPU_PEAK_FLOPS = 312e12   # A100 BF16 dense peak, NVIDIA datasheet (FLOP/s)
UTILIZATION = 0.40        # assumed achieved fraction of peak (MFU)

N_GPUS_SINGLE = 1
N_GPUS_CLUSTER = 1024

# ---------------------------------------------------------------------------
# 2. Compute total training FLOPs:  6 * P * T
# ---------------------------------------------------------------------------
total_flops = 6 * P * T

# ---------------------------------------------------------------------------
# 3. Optimizer memory:  16 * P bytes (FP16 weights+grads, FP32 Adam state)
# ---------------------------------------------------------------------------
optimizer_memory_bytes = 16 * P
optimizer_memory_gb = optimizer_memory_bytes / 1e9

# ---------------------------------------------------------------------------
# 4. Wall-clock training time on 1 GPU and on 1,024 GPUs
#    (assumes perfect linear scaling across GPUs -- optimistic, see Week 4)
# ---------------------------------------------------------------------------
effective_flops_per_gpu = GPU_PEAK_FLOPS * UTILIZATION

time_seconds_1_gpu = total_flops / (effective_flops_per_gpu * N_GPUS_SINGLE)
time_seconds_1024_gpus = total_flops / (effective_flops_per_gpu * N_GPUS_CLUSTER)

time_days_1_gpu = time_seconds_1_gpu / 86400
time_days_1024_gpus = time_seconds_1024_gpus / 86400

# ---------------------------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------------------------
print(f"Model: Llama 3 8B (P={P:.2e}, T={T:.2e})")
print(f"Total training FLOPs (6*P*T): {total_flops:.3e} FLOPs")
print(f"Optimizer memory (16*P): {optimizer_memory_gb:.2f} GB")
print(f"A100 peak={GPU_PEAK_FLOPS/1e12:.0f} TFLOP/s, utilization={UTILIZATION*100:.0f}%")
print(f"Time on 1 GPU: {time_seconds_1_gpu:.3e} s (~{time_days_1_gpu:,.1f} days)")
print(f"Time on {N_GPUS_CLUSTER} GPUs: {time_seconds_1024_gpus:.3e} s (~{time_days_1024_gpus:,.2f} days)")

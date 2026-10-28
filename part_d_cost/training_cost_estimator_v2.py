"""
training_cost_estimator_v2.py  --  Assignment 2, Part D: "Scaling reality check"

Extends Assignment 1's training_cost_estimator.py (same model, GPU, utilisation)
with a COMMUNICATION term: the per-step gradient all-reduce, costed with the
ring formula from Part B.

Same inputs as Assignment 1:
  Model : Llama 3 8B, P = 8e9 params, T = 15e12 tokens
          (Meta AI blog, "Introducing Meta Llama 3", ai.meta.com/blog/meta-llama-3/)
  GPU   : NVIDIA A100 80GB SXM, 312 TFLOP/s BF16 dense (NVIDIA A100 datasheet)
  MFU   : 40 %

New inputs (Assignment 2):
  Global batch G = 4M tokens per step.
      Source: Dubey et al., "The Llama 3 Herd of Models" (arXiv:2407.21783, 2024),
      Sec. 3.4.1 -- pre-training starts with a batch of 4M tokens (sequences of 4,096).
  Interconnect : 200 Gb/s HDR InfiniBand per GPU = 25 GB/s.
      Source: NVIDIA DGX A100 datasheet -- 8x single-port ConnectX-6 VPI
      200 Gb/s HDR InfiniBand (one per GPU) for the cluster network.
  Latency alpha = 2 us per message step (typical InfiniBand end-to-end latency; assumption).
  Contrast: our own alpha, beta measured in Part B (read from ../results/partB_alpha_beta.json).

Run:   python3 training_cost_estimator_v2.py
"""
import json
import os

# ---------------------------------------------------------------------------
# 1. Inputs -- identical to Assignment 1
# ---------------------------------------------------------------------------
P_PARAMS = 8e9            # Llama 3 8B
T_TOKENS = 15e12          # training tokens
GPU_PEAK_FLOPS = 312e12   # A100 BF16 dense
UTILIZATION = 0.40        # MFU

# ---------------------------------------------------------------------------
# 2. New inputs -- communication (Assignment 2)
# ---------------------------------------------------------------------------
G_TOKENS = 4e6                     # global batch, tokens per step (Llama 3 paper)
GRAD_BYTES = 2 * P_PARAMS          # bf16 gradients: n = 2 * P bytes
IB_ALPHA = 2e-6                    # s per message step (assumed IB latency)
IB_BETA = 1 / 25e9                 # s per byte  (200 Gb/s = 25 GB/s per GPU)
GPU_COUNTS = (8, 64, 1024)

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED_JSON = os.path.join(HERE, "..", "results", "partB_alpha_beta.json")


def t_compute(n_gpus):
    """6 * P * (G/N) / (peak * util): forward+backward FLOPs for this GPU's share."""
    return 6 * P_PARAMS * (G_TOKENS / n_gpus) / (GPU_PEAK_FLOPS * UTILIZATION)


def t_ring_allreduce(n_gpus, nbytes, alpha, beta):
    """Ring all-reduce from Part B: 2(N-1) alpha + 2(N-1)/N * bytes * beta."""
    return 2 * (n_gpus - 1) * alpha + 2 * (n_gpus - 1) / n_gpus * nbytes * beta


def table(label, alpha, beta):
    print(f"\n{label}:  alpha = {alpha * 1e6:.2f} us, 1/beta = {1 / beta / 1e9:.2f} GB/s, "
          f"gradient = {GRAD_BYTES / 1e9:.0f} GB per step")
    print(f"  {'N GPUs':>7} {'T_compute (s)':>14} {'T_comm (s)':>11} "
          f"{'E no-overlap':>13} {'E overlap':>10}")
    rows = {}
    for n in GPU_COUNTS:
        tc = t_compute(n)
        tm = t_ring_allreduce(n, GRAD_BYTES, alpha, beta)
        e_no = tc / (tc + tm)
        e_ov = tc / max(tc, tm)
        rows[n] = (tc, tm, e_no, e_ov)
        print(f"  {n:>7} {tc:>14.3f} {tm:>11.3f} {e_no:>13.1%} {e_ov:>10.1%}")
    return rows


def main():
    total_flops = 6 * P_PARAMS * T_TOKENS
    steps = T_TOKENS / G_TOKENS
    print(f"Model: Llama 3 8B (P={P_PARAMS:.1e}), T={T_TOKENS:.1e} tokens, "
          f"G={G_TOKENS:.0e} tokens/step -> {steps:,.0f} steps")
    print(f"GPU: A100 {GPU_PEAK_FLOPS / 1e12:.0f} TFLOP/s BF16 x {UTILIZATION:.0%} MFU")

    ib = table("Published interconnect (200 Gb/s HDR InfiniBand per GPU)", IB_ALPHA, IB_BETA)

    if os.path.exists(MEASURED_JSON):
        with open(MEASURED_JSON) as f:
            m = json.load(f)
        table(f"Our measured Part B numbers (P={m['P']} ranks, one machine)",
              m["alpha_s"], m["beta_s_per_byte"])
    else:
        print(f"\n(run Part B bench first to also see your measured alpha/beta: {MEASURED_JSON})")

    # Re-compute the Assignment 1 "time to train on 1,024 GPUs"
    t_a1 = total_flops / (GPU_PEAK_FLOPS * UTILIZATION * 1024)       # perfect scaling
    e_no = ib[1024][2]
    t_a2 = t_a1 / e_no
    print("\nTime to train on 1,024 GPUs")
    print(f"  Assignment 1 (perfect linear scaling) : {t_a1 / 86400:6.2f} days")
    print(f"  Assignment 2 (no-overlap E = {e_no:.1%})  : {t_a2 / 86400:6.2f} days  "
          f"(+{t_a2 / t_a1 - 1:.0%})")
    print(f"  with perfect overlap (E = {ib[1024][3]:.1%})      : {t_a1 / ib[1024][3] / 86400:6.2f} days")


if __name__ == "__main__":
    main()

"""
mlp_from_scratch.py

A 2-layer MLP (2 -> h -> 1) implemented in pure NumPy.
No autograd, no PyTorch/TensorFlow, no sklearn.
Trained with plain SGD (no momentum, no Adam) on a toy XOR-blobs
binary classification problem that is NOT linearly separable.

Run:
    python mlp_from_scratch.py

Produces:
    plateau_loss_curve.png       (train/val loss for the "normal" run)
    overfit_loss_curve.png       (train/val loss for the overfitting run)
    Console output with final train/val loss + accuracy for both runs.
"""

import numpy as np

# Matplotlib is optional -- if it isn't available we fall back to printing
# a loss table every N epochs so the trend is still visible.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 1. Data generator (given verbatim -- data generation isn't the objective)
# ---------------------------------------------------------------------------
def make_xor_blobs(n=400, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    # XOR pattern -> not linearly separable
    idx = rng.integers(0, 4, size=n)
    X = centers[idx] + rng.normal(scale=noise, size=(n, 2))
    y = labels[idx].reshape(-1, 1).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# 2. Parameter initialization
# ---------------------------------------------------------------------------
def init_params(input_dim=2, hidden_dim=8, output_dim=1, seed=0):
    """
    He-style small random init. Scaled by sqrt(1/fan_in) to keep early
    activations from saturating tanh/sigmoid or exploding through ReLU.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(scale=np.sqrt(1.0 / input_dim), size=(input_dim, hidden_dim))
    b1 = np.zeros((1, hidden_dim))
    W2 = rng.normal(scale=np.sqrt(1.0 / hidden_dim), size=(hidden_dim, output_dim))
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2


# ---------------------------------------------------------------------------
# 3. Activations
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    # derivative of ReLU: 1 where z > 0, else 0
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    # clip to avoid overflow in exp for very negative z
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 4. Forward pass: two GEMMs + ReLU hidden nonlinearity + sigmoid output
# ---------------------------------------------------------------------------
def forward(X, W1, b1, W2, b2):
    """
    X:  (N, 2)
    W1: (2, h)   b1: (1, h)
    W2: (h, 1)   b2: (1, 1)

    Returns the output prediction plus the cached intermediates needed
    for backprop (z1, a1, z2, a2).
    """
    z1 = X @ W1 + b1          # (N, h)  -- GEMM 1
    a1 = relu(z1)              # (N, h)  -- hidden nonlinearity
    z2 = a1 @ W2 + b2          # (N, 1)  -- GEMM 2
    a2 = sigmoid(z2)           # (N, 1)  -- output nonlinearity

    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, cache


# ---------------------------------------------------------------------------
# 5. Loss: binary cross-entropy
# ---------------------------------------------------------------------------
def bce_loss(y_true, y_pred, eps=1e-8):
    """
    Mean binary cross-entropy over the batch.
    y_true, y_pred: (N, 1)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # avoid log(0)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


# ---------------------------------------------------------------------------
# 6. Backward pass -- gradients derived by hand via the chain rule.
#
#    Forward graph:
#      z1 = X @ W1 + b1
#      a1 = relu(z1)
#      z2 = a1 @ W2 + b2
#      a2 = sigmoid(z2)
#      L  = BCE(y, a2)
#
#    Because we use sigmoid output + BCE loss, dL/dz2 simplifies to
#    (a2 - y) / N -- this is the standard combined-gradient identity
#    (the sigmoid' and 1/(a2(1-a2)) terms from BCE cancel out).
# ---------------------------------------------------------------------------
def backward(y_true, cache, W1, b1, W2, b2):
    X, z1, a1, z2, a2 = cache["X"], cache["z1"], cache["a1"], cache["z2"], cache["a2"]
    N = X.shape[0]

    # dL/dz2 : combined gradient of BCE + sigmoid (standard simplification)
    dz2 = (a2 - y_true) / N                      # (N, 1)

    # dL/dW2 = a1^T @ dz2  -- chain rule: dL/dz2 * dz2/dW2 (dz2/dW2 = a1)
    dW2 = a1.T @ dz2                               # (h, 1)

    # dL/db2 = sum over batch of dz2 -- chain rule: dz2/db2 = 1
    db2 = np.sum(dz2, axis=0, keepdims=True)       # (1, 1)

    # dL/da1 = dz2 @ W2^T  -- chain rule: dL/dz2 * dz2/da1 (dz2/da1 = W2)
    da1 = dz2 @ W2.T                               # (N, h)

    # dL/dz1 = dL/da1 * relu'(z1)  -- chain rule through the ReLU nonlinearity
    dz1 = da1 * relu_grad(z1)                      # (N, h)

    # dL/dW1 = X^T @ dz1  -- chain rule: dL/dz1 * dz1/dW1 (dz1/dW1 = X)
    dW1 = X.T @ dz1                                # (2, h)

    # dL/db1 = sum over batch of dz1 -- chain rule: dz1/db1 = 1
    db1 = np.sum(dz1, axis=0, keepdims=True)       # (1, h)

    return dW1, db1, dW2, db2


# ---------------------------------------------------------------------------
# 7. Plain SGD update loop (no momentum, no Adam)
# ---------------------------------------------------------------------------
def sgd_update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W1, b1, W2, b2


def accuracy(y_true, y_pred):
    preds = (y_pred >= 0.5).astype(float)
    return float(np.mean(preds == y_true))


# ---------------------------------------------------------------------------
# 8. Train/val split helper (80/20, no sklearn)
# ---------------------------------------------------------------------------
def train_val_split(X, y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# 9. Training loop
# ---------------------------------------------------------------------------
def train(
    X_train, y_train, X_val, y_val,
    hidden_dim=8, lr=0.5, epochs=2000, seed=0,
    print_every=200, tag="run",
):
    W1, b1, W2, b2 = init_params(hidden_dim=hidden_dim, seed=seed)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # ---- forward + backward + SGD step (full-batch, plain SGD) ----
        y_pred_train, cache = forward(X_train, W1, b1, W2, b2)
        loss_train = bce_loss(y_train, y_pred_train)

        dW1, db1, dW2, db2 = backward(y_train, cache, W1, b1, W2, b2)
        W1, b1, W2, b2 = sgd_update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        # ---- validation loss (no update) ----
        y_pred_val, _ = forward(X_val, W1, b1, W2, b2)
        loss_val = bce_loss(y_val, y_pred_val)

        train_losses.append(loss_train)
        val_losses.append(loss_val)

        if epoch % print_every == 0 or epoch == 1:
            print(f"[{tag}] epoch {epoch:5d}  train_loss={loss_train:.4f}  val_loss={loss_val:.4f}")

    params = (W1, b1, W2, b2)
    return params, train_losses, val_losses


def plot_or_print(train_losses, val_losses, filename, tag, n_print=15):
    if HAS_MPL:
        plt.figure(figsize=(7, 5))
        plt.plot(train_losses, label="train loss")
        plt.plot(val_losses, label="val loss")
        plt.xlabel("epoch")
        plt.ylabel("BCE loss")
        plt.title(f"Loss curve -- {tag}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")
    else:
        print(f"\n[{tag}] matplotlib not available -- printing loss table instead:")
        epochs = len(train_losses)
        step = max(1, epochs // n_print)
        print(f"{'epoch':>8} {'train_loss':>12} {'val_loss':>12}")
        for e in range(0, epochs, step):
            print(f"{e:8d} {train_losses[e]:12.4f} {val_losses[e]:12.4f}")
        print(f"{epochs-1:8d} {train_losses[-1]:12.4f} {val_losses[-1]:12.4f}")


# ---------------------------------------------------------------------------
# 10. Main: run both experiments
# ---------------------------------------------------------------------------
def main():
    # ----------------------------------------------------------------
    # Run 1: "plateau" run -- enough data to train accuracy > 90%
    # ----------------------------------------------------------------
    print("=" * 70)
    print("RUN 1: plateau run (n=400, 80/20 split)")
    print("=" * 70)

    X, y = make_xor_blobs(n=400, noise=0.25, seed=0)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=0.2, seed=0)

    params, train_losses, val_losses = train(
        X_train, y_train, X_val, y_val,
        hidden_dim=16, lr=1.0, epochs=4000, seed=0,
        print_every=400, tag="plateau",
    )
    W1, b1, W2, b2 = params

    y_pred_train, _ = forward(X_train, W1, b1, W2, b2)
    y_pred_val, _ = forward(X_val, W1, b1, W2, b2)

    plateau_final_train_loss = bce_loss(y_train, y_pred_train)
    plateau_final_val_loss = bce_loss(y_val, y_pred_val)
    plateau_train_acc = accuracy(y_train, y_pred_train)
    plateau_val_acc = accuracy(y_val, y_pred_val)

    print(f"\n[plateau] FINAL  train_loss={plateau_final_train_loss:.4f}  "
          f"train_acc={plateau_train_acc:.3f}  "
          f"val_loss={plateau_final_val_loss:.4f}  val_acc={plateau_val_acc:.3f}")

    plot_or_print(train_losses, val_losses, "plateau_loss_curve.png", "plateau run (n=400)")

    # ----------------------------------------------------------------
    # Run 2: overfitting run -- much smaller training set, many epochs
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUN 2: overfitting run (n_train=40, many epochs)")
    print("=" * 70)

    # Generate a small training set and a separate, larger validation set
    # so validation loss is a meaningful, stable signal.
    X_small_train, y_small_train = make_xor_blobs(n=40, noise=0.25, seed=1)
    X_big_val, y_big_val = make_xor_blobs(n=200, noise=0.25, seed=2)

    params2, train_losses2, val_losses2 = train(
        X_small_train, y_small_train, X_big_val, y_big_val,
        hidden_dim=16, lr=0.5, epochs=6000, seed=0,
        print_every=600, tag="overfit",
    )
    W1o, b1o, W2o, b2o = params2

    y_pred_train2, _ = forward(X_small_train, W1o, b1o, W2o, b2o)
    y_pred_val2, _ = forward(X_big_val, W1o, b1o, W2o, b2o)

    overfit_final_train_loss = bce_loss(y_small_train, y_pred_train2)
    overfit_final_val_loss = bce_loss(y_big_val, y_pred_val2)
    overfit_train_acc = accuracy(y_small_train, y_pred_train2)
    overfit_val_acc = accuracy(y_big_val, y_pred_val2)

    print(f"\n[overfit] FINAL  train_loss={overfit_final_train_loss:.4f}  "
          f"train_acc={overfit_train_acc:.3f}  "
          f"val_loss={overfit_final_val_loss:.4f}  val_acc={overfit_val_acc:.3f}")

    # Report where val loss hit its minimum, to make the "turns back up" point concrete
    min_val_epoch = int(np.argmin(val_losses2))
    print(f"[overfit] val_loss minimum = {val_losses2[min_val_epoch]:.4f} at epoch {min_val_epoch}; "
          f"final val_loss = {val_losses2[-1]:.4f} (rose by "
          f"{val_losses2[-1] - val_losses2[min_val_epoch]:.4f} after that point)")

    plot_or_print(train_losses2, val_losses2, "overfit_loss_curve.png", "overfitting run (n_train=40)")

    # ----------------------------------------------------------------
    # Summary block for the report
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY FOR REPORT")
    print("=" * 70)
    print("Plateau run (n=400, 80/20 split):")
    print(f"  final train_loss = {plateau_final_train_loss:.4f}, train_acc = {plateau_train_acc:.3f}")
    print(f"  final val_loss   = {plateau_final_val_loss:.4f}, val_acc   = {plateau_val_acc:.3f}")
    print("\nOverfitting run (n_train=40, 6000 epochs):")
    print(f"  final train_loss = {overfit_final_train_loss:.4f}, train_acc = {overfit_train_acc:.3f}")
    print(f"  final val_loss   = {overfit_final_val_loss:.4f}, val_acc   = {overfit_val_acc:.3f}")
    print(f"  val_loss minimum = {val_losses2[min_val_epoch]:.4f} at epoch {min_val_epoch} "
          f"(then rose to {val_losses2[-1]:.4f} by the final epoch -> classic overfitting signature)")


if __name__ == "__main__":
    main()

"""
Dictionary-amortized concentration encoder (backprop primary; GPU-ready).

  X_t  --[1D-conv encoder over r]-->  concentrations over the fixed real-PDF
  dictionary D (frozen decoder).  X_hat = conc @ D.T.

Trained supervised on the versatile labeled mixtures from make_dataset.py.
Because D is a fixed library of REAL phases, there is no sign/permutation
ambiguity: the network outputs interpretable phase fractions directly.

Run:  micromamba run -n pdf python train_amortized.py
GPU:  install CUDA torch, then the same command (auto-detects cuda).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

B = os.environ.get("SEMINMF_DIR", os.path.dirname(os.path.abspath(__file__))) + "/"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# Some torch cu130 wheels ship mismatched cuDNN sublibraries -> conv1d raises
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. This model is tiny, so fall back to
# the native (non-cuDNN) conv kernel; negligible speed cost, full robustness.
torch.backends.cudnn.enabled = False
EPOCHS = int(os.environ.get("EPOCHS", 60))
BATCH = int(os.environ.get("BATCH", 256))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", 0))   # 0 = use all; >0 subsamples (fast CPU check)
LR = 2e-3
# recon-against-clean-D is GAMED under heavy augmentation (perfect labels give
# si_recon 0.76 yet the model reaches 0.73 by mis-assigning), so default it OFF:
# with known labels, pure supervised concentration loss is well-posed.
LAMBDA_RECON = float(os.environ.get("LAMBDA_RECON", 0.0))
SEED = 0


class ConcEncoder(nn.Module):
    """1D-conv over r (narrow kernels so it reads local PDF features, not global
    'cheating'), then dense -> P concentrations via softmax (closure, >=0)."""
    def __init__(self, n_r, P, ch=(16, 32, 64), k=7):
        super().__init__()
        layers, c_in = [], 1
        for c in ch:
            layers += [nn.Conv1d(c_in, c, k, stride=2, padding=k // 2), nn.ReLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, n_r)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(),
                                  nn.Linear(128, P))

    def forward(self, x):                 # x: (B, n_r)
        h = self.conv(x.unsqueeze(1))
        return torch.softmax(self.head(h), dim=-1)   # (B, P), >=0, sums to 1


def si_recon(pred, X, D):
    """Scale-invariant reconstruction error: fit a per-sample amplitude alpha so
    the term doesn't penalize correct concentrations for un-modelled amplitude."""
    Dc = pred @ D.T
    alpha = (X * Dc).sum(-1, keepdim=True) / ((Dc * Dc).sum(-1, keepdim=True) + 1e-8)
    return torch.linalg.norm(alpha * Dc - X) / (torch.linalg.norm(X) + 1e-8)


def metrics(pred, Y, X, D):
    """Concentration MAE, scale-invariant reconstruction err, phase-F1@0.05."""
    rec = si_recon(pred, X, D)
    mae = (pred - Y).abs().mean()
    p, y = pred > 0.05, Y > 0.05
    tp = (p & y).sum().float(); fp = (p & ~y).sum().float(); fn = (~p & y).sum().float()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-9))
    return float(mae), float(rec), float(f1)


def main():
    torch.manual_seed(SEED)
    d = np.load(B + "dataset_pc.npz", allow_pickle=True)
    D = torch.tensor(d["D"], dtype=torch.float32, device=DEV)        # (n_r, P)
    names = list(d["phase_names"])
    def norm(x):                              # per-sample scale-invariance (L2; robust to spikes)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    Xtr = norm(torch.tensor(d["X_train"], device=DEV)); Ytr = torch.tensor(d["Y_train"], device=DEV)
    if MAX_TRAIN and MAX_TRAIN < len(Xtr):
        Xtr, Ytr = Xtr[:MAX_TRAIN], Ytr[:MAX_TRAIN]
    Xva = norm(torch.tensor(d["X_val"], device=DEV));  Yva = torch.tensor(d["Y_val"], device=DEV)
    n_r, P = D.shape
    print(f"device={DEV}  n_r={n_r}  P={P}  train={len(Xtr)}  val={len(Xva)}")

    net = ConcEncoder(n_r, P).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    nb = (len(Xtr) + BATCH - 1) // BATCH

    for ep in range(1, EPOCHS + 1):
        net.train(); perm = torch.randperm(len(Xtr), device=DEV)
        for b in range(nb):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            pred = net(xb)
            loss = Fnn.mse_loss(pred, yb) + LAMBDA_RECON * si_recon(pred, xb, D)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                mae, rec, f1 = metrics(net(Xva), Yva, Xva, D)
            # recon is a passive monitor (NOT in the loss); ~constant augmentation gap.
            print(f"  epoch {ep:3d}  val conc-MAE={mae:.4f}  phaseF1={f1:.3f}  [recon~{rec:.2f}]")

    torch.save({"state": net.state_dict(), "names": [str(n) for n in names]},
               B + "amortized_encoder.pt")
    print("saved amortized_encoder.pt")


if __name__ == "__main__":
    main()

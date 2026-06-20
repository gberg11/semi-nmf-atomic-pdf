"""
Run the trained dictionary-amortized encoder on REAL PDF data.

Usage:
  python predict_on_real.py mystery_phase1.gr          # single spectrum
  python predict_on_real.py "data/*.gr"                # a glob = time series
  python predict_on_real.py data/                      # a directory of .gr = series

Outputs predicted phase fractions over the 12 dictionary phases + the "other"
sink (high "other" => a phase OUTSIDE the dictionary, i.e. don't trust the rest).
For a series it also saves predicted_kinetics.png.

Needs: amortized_encoder.pt (shipped; or from train_amortized.py) and the small
dict_cache.npz (for the r-grid) in the same directory. Runs fine on CPU.
"""
import os, sys, glob
import numpy as np
import torch
from train_amortized import ConcEncoder

B = os.path.dirname(os.path.abspath(__file__)) + "/"


def load_gr(path):
    """Robustly read a 2-column .gr (skips xPDFsuite/PDFgui headers)."""
    r, g = [], []
    for line in open(path):
        p = line.split()
        if len(p) >= 2:
            try:
                a, b = float(p[0]), float(p[1])
            except ValueError:
                continue
            r.append(a); g.append(b)
    return np.asarray(r), np.asarray(g)


def main(target):
    # r-grid comes from the small shipped dict_cache.npz (no need for the big dataset_pc.npz)
    d = np.load(B + "dict_cache.npz", allow_pickle=True)
    r = d["r"].astype("float32")
    ckpt = torch.load(B + "amortized_encoder.pt", map_location="cpu", weights_only=False)
    names = list(ckpt["names"]); P = len(names); n_r = len(r)
    net = ConcEncoder(n_r, P); net.load_state_dict(ckpt["state"]); net.eval()
    oidx = names.index("other") if "other" in names else -1

    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.gr")))
    elif any(c in target for c in "*?["):
        files = sorted(glob.glob(target))
    else:
        files = [target]
    if not files:
        print("no files matched:", target); return

    rows = []
    for f in files:
        rr, gg = load_gr(f)
        gi = np.interp(r, rr, gg, left=0.0, right=0.0).astype("float32")  # onto dict grid
        x = torch.tensor(gi)[None]
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)                     # same norm as training
        with torch.no_grad():
            pred = net(x)[0].numpy()
        rows.append(pred)
        print(f"\n{os.path.basename(f)}:")
        for i in np.argsort(-pred):
            if pred[i] > 0.02:
                tag = "  <- unknown/out-of-dictionary" if i == oidx else ""
                print(f"   {names[i]:14s} {pred[i] * 100:5.1f}%{tag}")
        if oidx >= 0 and pred[oidx] > 0.2:
            print("   ** high 'other': a phase OUTSIDE the 12-entry dictionary is present;")
            print("      the dictionary fractions above are unreliable for this spectrum. **")

    rows = np.asarray(rows)
    if len(files) > 1:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for i in range(P):
            if rows[:, i].max() > 0.05:
                plt.plot(rows[:, i], marker="o", ms=3, label=names[i])
        plt.xlabel("sample (time order)"); plt.ylabel("predicted fraction")
        plt.title("Predicted phase kinetics"); plt.legend(fontsize=8); plt.tight_layout()
        out = B + "predicted_kinetics.png"; plt.savefig(out, dpi=130)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else B + "mystery_phase1.gr")

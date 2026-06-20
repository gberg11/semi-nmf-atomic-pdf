"""
Build (1) a dictionary of REAL phase PDFs from Materials Project + local CIFs,
and (2) a large, versatile, *labeled* synthetic dataset of reaction-evolution
mixtures for training the dictionary-amortized encoder.

Each training sample = (X_t : a PDF over r,  y_t : concentration over the full
dictionary). Versatility: random active-phase subsets, multiple kinetic schemes,
random rates, random overall scale, random noise level, small peak broadening.

Output: dataset_pc.npz  with  D (n_r x P), r, phase_names,
        X_train,Y_train, X_val,Y_val.
"""
import os, sys, itertools
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from scipy.ndimage import gaussian_filter1d
from diffpy.structure import load_structure
from diffpy.srreal.pdfcalculator import PDFCalculator
# NOTE: Materials Project + StructureFingerprintingCode are imported lazily inside
# build_dictionary() only when dict_cache.npz is absent — so regenerating the dataset
# from the shipped cache needs no MP API key or extra deps.

B = os.environ.get("SEMINMF_DIR", os.path.dirname(os.path.abspath(__file__))) + "/"

RMIN, RMAX, RSTEP = 1.5, 30.0, 0.05      # n_r ~ 570
QMAX, QDAMP = 18.0, 0.032
N_RUNS = 1500                            # synthetic experiments
N_T = 60                                 # time points per run
SEED = 0

# --- augmentation (sim-to-real): make the network invariant to the distortions
#     real PDFs show vs the canonical dictionary ---
AUGMENT = True
MAX_STRAIN = 0.04        # +/- lattice strain (peaks shift up to ~4% along r)
MAX_BROADEN_A = 0.12     # max extra Gaussian peak broadening (Angstrom)
MAX_QDAMP_EXTRA = 0.03   # max extra high-r exponential damping (1/A)
OTHER_PROB = 0.35        # fraction of runs containing an unknown (non-dictionary) phase
                         # -> trains the "other" sink channel so real spectra with a
                         #    phase outside the dictionary aren't force-fit onto the 12.

# dictionary: (label -> ('mp', mp_id) | ('cif', path))
DICT = {
    "CsMnBr3.2H2O": ("cif", B + "hydrated_phases/CsMnBr3_2(H2O)_1701939.cif"),
    "CsBr":         ("mp", "mp-22906"),
    "MnBr2":        ("mp", "mp-28306"),
    "CsMnBr3":      ("mp", "mp-23048"),
    "Cs2MnBr4":     ("cif", B + "structs/Cs2MnBr4.cif"),
    "Cs3MnBr5":     ("mp", "mp-27332"),
    "CsMnCl3":      ("mp", "mp-23336"),
    "Cs3MnCl5":     ("mp", "mp-504451"),
    "Cs2MnCl6":     ("mp", "mp-1205772"),
    "CsCl":         ("mp", "mp-22865"),
    "Cs2MnO4":      ("mp", "mp-505300"),
    "Cs4Mn3O6":     ("mp", "mp-1226457"),
}


def pdf_of(struct):
    for a in struct:
        a.anisotropy = False
        a.U = np.diag([0.04, 0.04, 0.04])
    pc = PDFCalculator(rmin=RMIN, rmax=RMAX + 1e-6, rstep=RSTEP, qmax=QMAX, qdamp=QDAMP)
    r, g = pc(struct)
    return r, g


def build_dictionary():
    cache = B + "dict_cache.npz"
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        print("  (loaded dictionary from cache)")
        return c["r"], c["D"], list(c["phase_names"])
    # cache miss -> need Materials Project + the pymatgen->diffpy helper
    from dotenv import load_dotenv
    from mp_api.client import MPRester
    sys.path.insert(0, B.rstrip("/"))
    from StructureFingerprintingCode import pymatgen_structure_to_diffpy_structure
    load_dotenv(B + ".env"); API = os.environ["MP_API_KEY"]
    names, cols, r_ref = [], [], None
    with MPRester(API) as mpr:
        for name, (src, ref) in DICT.items():
            try:
                if src == "cif":
                    s = load_structure(ref)
                else:
                    p = mpr.get_structure_by_material_id(ref); p.remove_oxidation_states()
                    s = pymatgen_structure_to_diffpy_structure(p, {"symprec": 0.1})
                r, g = pdf_of(s)
                r_ref = r if r_ref is None else r_ref
                g = np.interp(r_ref, r, g)
                g = g / (np.max(np.abs(g)) + 1e-9)        # normalize each phase
                names.append(name); cols.append(g)
                print(f"  dict + {name:14s} ({src} {ref})")
            except Exception as e:
                print(f"  dict ! {name}: {str(e)[:50]}")
    D = np.column_stack(cols)                              # (n_r, P)
    np.savez_compressed(B + "dict_cache.npz", D=D, r=r_ref, phase_names=np.array(names))
    return r_ref, D, names


def kinetic_profiles(n_t, n_active, rng):
    """Return (n_t, n_active) nonneg smooth concentration curves; mixes schemes."""
    t = np.linspace(0, 1, n_t)
    scheme = rng.integers(0, 3)
    if scheme == 0:                                        # first-order consecutive A->B->C->...
        # proper kinetics via explicit Euler on dC_j/dt = k_{j-1}C_{j-1} - k_j C_j,
        # mass-conserving; arbitrary n_active; start as pure reactant A.
        ks = rng.uniform(1.0, 8.0, max(n_active - 1, 1))
        C = np.zeros((n_t, n_active)); C[0, 0] = 1.0
        dt = t[1] - t[0]
        for i in range(1, n_t):
            c = C[i - 1]
            dc = np.zeros(n_active)
            for j in range(n_active):
                inflow = ks[j - 1] * c[j - 1] if j > 0 else 0.0
                outflow = ks[j] * c[j] if j < n_active - 1 else 0.0
                dc[j] = inflow - outflow
            C[i] = np.clip(c + dt * dc, 0, None)
    elif scheme == 1:                                      # time-shifted Gaussian bumps
        C = np.zeros((n_t, n_active))
        for j in range(n_active):
            c0 = rng.uniform(0, 1); w = rng.uniform(0.1, 0.3)
            C[:, j] = np.exp(-0.5 * ((t - c0) / w) ** 2)
    else:                                                  # monotone rise/fall mix
        C = np.zeros((n_t, n_active))
        for j in range(n_active):
            k = rng.uniform(1, 6)
            C[:, j] = np.exp(-k * t) if rng.random() < 0.5 else 1 - np.exp(-k * t)
    C = np.clip(C, 0, None) + 1e-4
    C = C / C.sum(axis=1, keepdims=True)                   # closure
    return C


def augment_phase(g, r, rng):
    """Distort one phase PDF the way real data differs from the dictionary:
    a random lattice strain (stretch/compress along r) + extra peak broadening
    + extra high-r damping. Labels (concentrations) are unchanged."""
    out = g
    if MAX_STRAIN > 0:                                     # lattice strain: peak at d -> d*s
        s = 1.0 + rng.uniform(-MAX_STRAIN, MAX_STRAIN)
        out = np.interp(r / s, r, out, left=0.0, right=0.0)
    if MAX_BROADEN_A > 0:                                  # extra Gaussian broadening
        sigma_pts = rng.uniform(0.0, MAX_BROADEN_A) / RSTEP
        if sigma_pts > 1e-3:
            out = gaussian_filter1d(out, sigma_pts, mode="nearest")
    if MAX_QDAMP_EXTRA > 0:                                # extra high-r damping (qdamp-like)
        qb = rng.uniform(0.0, MAX_QDAMP_EXTRA)
        out = out * np.exp(-0.5 * (r * qb) ** 2)
    return out.astype("float32")


# chemically-plausible co-occurrence: phases mostly mix within an anion family,
# with occasional shared precursors crossing over.
FAMILIES = [
    ["CsMnBr3.2H2O", "CsBr", "MnBr2", "CsMnBr3", "Cs2MnBr4", "Cs3MnBr5"],   # bromides
    ["CsMnCl3", "Cs3MnCl5", "Cs2MnCl6", "CsCl"],                            # chlorides
    ["Cs2MnO4", "Cs4Mn3O6"],                                               # oxides
]
PRECURSORS = ["CsBr", "CsCl", "MnBr2"]


def sample_active(names, rng):
    fams = [[names.index(p) for p in f if p in names] for f in FAMILIES]
    fams = [f for f in fams if f]
    fam = fams[rng.integers(len(fams))]
    n = int(rng.integers(1, min(4, len(fam)) + 1))
    active = list(rng.choice(fam, n, replace=False))
    if rng.random() < 0.25:                                # cross-family precursor contamination
        prec = [names.index(p) for p in PRECURSORS if p in names and names.index(p) not in active]
        if prec:
            active.append(int(prec[rng.integers(len(prec))]))
    return np.array(active)


def make_background(r, rng):
    rr = (r - r.min()) / (r.max() - r.min() + 1e-9)
    a = rng.normal(0, 1, 3)
    return a[0] * rr + a[1] * rr ** 2 + a[2] * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * rr)


def make_unknown_pdf(r, rng):
    """A synthetic non-dictionary 'phase' (random signed Gaussians) — stands in for
    a real phase outside the 12-entry dictionary; its weight trains the 'other' sink."""
    g = np.zeros_like(r)
    for _ in range(int(rng.integers(3, 7))):
        c = rng.uniform(r.min(), r.max()); w = rng.uniform(0.3, 1.5); a = rng.uniform(-1, 1)
        g += a * np.exp(-0.5 * ((r - c) / w) ** 2)
    return (g / (np.max(np.abs(g)) + 1e-9)).astype("float32")


def make_data(D, r, names, n_runs, n_t, rng):
    Preal = D.shape[1]; P = Preal + 1                      # last column = "other" sink
    Xs, Ys, runs = [], [], []
    for run in range(n_runs):
        active = sample_active(names, rng)
        has_other = rng.random() < OTHER_PROB
        n_sp = len(active) + (1 if has_other else 0)
        C = kinetic_profiles(n_t, n_sp, rng)               # closure over all species
        Yfull = np.zeros((n_t, P))
        Yfull[:, active] = C[:, :len(active)]
        Dr = D.copy()                                      # per-run distortion (constant within run)
        if AUGMENT:
            for j in active:
                Dr[:, j] = augment_phase(D[:, j], r, rng)
        scale = rng.uniform(0.3, 2.0)
        X = scale * (Yfull[:, :Preal] @ Dr.T)              # (n_t, n_r) from real phases
        if has_other:                                      # add an unknown phase -> 'other' label
            u = make_unknown_pdf(r, rng)
            Yfull[:, Preal] = C[:, -1]
            X = X + scale * np.outer(C[:, -1], u)
        if AUGMENT:                                        # smooth experimental background
            X = X + make_background(r, rng) * rng.uniform(0.0, 0.15) * np.max(np.abs(X))
        noise = rng.uniform(0.0, 0.15) * np.std(X)
        X = X + noise * rng.standard_normal(X.shape)
        Xs.append(X.astype("float32")); Ys.append(Yfull.astype("float32"))
        runs.append(np.full(n_t, run, dtype=np.int32))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(runs)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print("Building dictionary from Materials Project + local CIFs ...")
    r, D, names = build_dictionary()
    print(f"dictionary: {D.shape[1]} phases, n_r={D.shape[0]}")
    print(f"Generating versatile labeled mixtures (augment={AUGMENT}: "
          f"strain±{MAX_STRAIN}, broaden≤{MAX_BROADEN_A}Å, qdamp≤{MAX_QDAMP_EXTRA}) ...")
    X, Y, runs = make_data(D, r, names, N_RUNS, N_T, rng)
    # append the 'other' sink: a generic broad basis column + its name (Y already P+1 wide)
    other_col = np.exp(-0.5 * ((r - r.mean()) / ((r.max() - r.min()) / 4)) ** 2)
    D = np.column_stack([D, (other_col / other_col.max()).astype("float32")])
    names = list(names) + ["other"]
    print(f"  + 'other' sink channel -> P={D.shape[1]} (12 real phases + unknown)")
    # split by RUN, not by sample, so val mixtures are independent of train
    uniq = np.unique(runs); rng.shuffle(uniq)
    train_runs = set(uniq[:int(0.9 * len(uniq))].tolist())
    mask_tr = np.array([rr in train_runs for rr in runs])
    tr = np.where(mask_tr)[0]; va = np.where(~mask_tr)[0]
    out = B + "dataset_pc.npz"
    np.savez_compressed(out, D=D, r=r, phase_names=np.array(names),
                        X_train=X[tr], Y_train=Y[tr], X_val=X[va], Y_val=Y[va])
    print(f"saved {out}: train {len(tr)}, val {len(va)}, n_r {X.shape[1]}, P {D.shape[1]}")

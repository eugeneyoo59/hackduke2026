"""
train_tremor_model.py
=====================
Train a GBT tremor-detection model from the
jiehu01/Parkinson-s-Disease-Tremor-Dataset repo and export tremor_model.json
for the PD Glove Monitor dashboard.

Usage
-----
    # Pass the repo path explicitly (most reliable):
    python train_tremor_model.py --repo /path/to/Parkinson-s-Disease-Tremor-Dataset

    # Or from the directory that CONTAINS the cloned repo:
    python train_tremor_model.py

    # Diagnose path/file issues without training:
    python train_tremor_model.py --probe

Dependencies
------------
    pip install numpy scikit-learn scipy matplotlib
"""

import os, sys, json, argparse, warnings
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, precision_recall_curve)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
FS          = 50      # Hz — all datasets pre-resampled to 50 Hz
WINDOW_SIZE = 128     # samples = 2.56 s

# PD hand tremor: resting 3-6 Hz, postural/action up to 8 Hz.
# Wider band than generic 3-7 to catch the full clinical range from a glove.
TREMOR_LO = 3.0
TREMOR_HI = 8.0

N_ESTIMATORS     = 400
LEARNING_RATE    = 0.05
MAX_DEPTH        = 4
SUBSAMPLE        = 0.8
MIN_SAMPLES_LEAF = 8

# Upweight the tremor class — penalises missed tremors more than false alarms.
# Increase to 4-5 if sensitivity is still too low after training.
TREMOR_CLASS_WEIGHT = 3.0

# Set > 0 to skip automatic threshold tuning (e.g. FORCE_THRESHOLD = 0.35)
FORCE_THRESHOLD = 0.0

# Known sub-dataset folder names inside the repo
DATASET_SUBDIRS = {
    "TIM-Tremor":  "Tim-Tremor",
    "PdAssist":    "PdAssist",
    "IMU-Wild":    "IMU-Wild",
    "PD-BioStamp": "PD-BioStamp",
}

REPO_NAMES = [
    "Parkinson-s-Disease-Tremor-Dataset",
    "Parkinson's-Disease-Tremor-Dataset",
    "pd-tremor-dataset",
]

# ─────────────────────────────────────────────────────────────────────────────
# Repo auto-detection
# ─────────────────────────────────────────────────────────────────────────────
def find_repo(hint=None):
    candidates = []
    if hint:
        candidates.append(Path(hint))

    cwd = Path.cwd()
    for name in REPO_NAMES:
        candidates += [cwd / name, cwd.parent / name, Path.home() / name]
    candidates.append(cwd)           # maybe user cd'd into the repo itself

    for p in candidates:
        if not p.is_dir():
            continue
        # Recognise it if at least one known sub-dataset folder exists
        if any((p / sub).is_dir() for sub in DATASET_SUBDIRS.values()):
            return p.resolve()
    return None


def probe_repo(repo):
    print(f"\nRepo root: {repo}\n")
    npy = sorted(p for p in repo.rglob("*.npy")
                 if not any(part.startswith('.venv') or part == 'site-packages'
                            for part in p.parts))
    if not npy:
        print("No .npy files found anywhere under this path.\n")
        print("All files (first 60):")
        for p in sorted(repo.rglob("*"))[:60]:
            if p.is_file():
                print(f"  {p.relative_to(repo)}")
    else:
        print(f"Found {len(npy)} .npy file(s):")
        for p in npy[:50]:
            print(f"  {p.relative_to(repo)}")
        if len(npy) > 50:
            print(f"  … and {len(npy)-50} more")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────
def remove_gravity(raw, alpha=0.98):
    """IIR high-pass — identical to dashboard JS removeGravity()."""
    gravity = raw[0].copy().astype(np.float64)
    out = np.empty_like(raw, dtype=np.float64)
    for i in range(len(raw)):
        gravity = alpha * gravity + (1.0 - alpha) * raw[i]
        out[i] = raw[i] - gravity
    return out


def hann_rfft_mag_sq(x):
    n = len(x)
    nfft = 1
    while nfft < n:
        nfft <<= 1
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    buf = np.zeros(nfft)
    buf[:n] = x * win
    spec = np.fft.rfft(buf)
    return spec.real**2 + spec.imag**2, FS / nfft


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — 32 features
# ─────────────────────────────────────────────────────────────────────────────
# Per-axis (×3 axes = 27 features):
#   tremor_energy, low_energy, high_energy, tremor_ratio, dominant_freq,
#   rms, zcr, jerk_rms, autocorr_tremor
# Global (5):
#   mean_tremor_energy, magnitude_spectral_entropy,
#   mean_inter_axis_correlation, total_rms_magnitude, dominant_freq_magnitude
# ─────────────────────────────────────────────────────────────────────────────
EPS = 1e-10
LOLO, LOHI = 0.5, TREMOR_LO
HILO, HIHI = TREMOR_HI, 15.0
F_MID      = (TREMOR_LO + TREMOR_HI) / 2
AUTOCORR_LAG = max(1, round(FS / F_MID))   # sample lag ≈ 1 period of mid-band

FEATURE_NAMES = []
for _ax in ["x", "y", "z"]:
    for _f in ["tremor_energy","low_energy","high_energy","tremor_ratio",
               "dominant_freq","rms","zcr","jerk_rms","autocorr_tremor"]:
        FEATURE_NAMES.append(f"{_ax}_{_f}")
FEATURE_NAMES += ["mean_tremor_energy","magnitude_spectral_entropy",
                  "mean_inter_axis_corr","total_rms_magnitude",
                  "dominant_freq_magnitude"]
assert len(FEATURE_NAMES) == 32


def extract_features(window):
    """window : [WINDOW_SIZE, 3] float64, gravity already removed."""
    T = len(window)
    feats = []
    trem_energies = []

    for axis in range(3):
        x = window[:, axis]
        mag_sq, fres = hann_rfft_mag_sq(x)
        freqs = np.arange(len(mag_sq)) * fres

        m_tot  = (freqs >= LOLO) & (freqs <= HIHI)
        m_tr   = (freqs >= TREMOR_LO) & (freqs <= TREMOR_HI)
        m_low  = (freqs >= LOLO) & (freqs < LOHI)
        m_hi   = (freqs >= HILO) & (freqs <= HIHI)

        tot_e  = float(mag_sq[m_tot].sum())
        tr_e   = float(mag_sq[m_tr].sum())
        low_e  = float(mag_sq[m_low].sum())
        hi_e   = float(mag_sq[m_hi].sum())
        tr_r   = tr_e / (tot_e + EPS)

        valid  = np.where(m_tot)[0]
        dom_f  = float(freqs[valid[np.argmax(mag_sq[valid])]])

        rms    = float(np.sqrt(np.mean(x**2)))
        signs  = np.sign(x)
        zcr    = float(np.sum(signs[1:] != signs[:-1])) / (T - 1)
        jerk   = np.diff(x) * FS
        j_rms  = float(np.sqrt(np.mean(jerk**2)))

        xc = x - x.mean()
        var = float(np.dot(xc, xc))
        if var > EPS and AUTOCORR_LAG < T:
            ac = float(np.dot(xc[:-AUTOCORR_LAG], xc[AUTOCORR_LAG:])) / var
        else:
            ac = 0.0

        feats.extend([tr_e, low_e, hi_e, tr_r, dom_f, rms, zcr, j_rms, ac])
        trem_energies.append(tr_e)

    # Global spectral features
    feats.append(float(np.mean(trem_energies)))

    mag_sig = np.linalg.norm(window, axis=1)
    mq, mfr = hann_rfft_mag_sq(mag_sig)
    mfreqs  = np.arange(len(mq)) * mfr
    vm      = (mfreqs >= LOLO) & (mfreqs <= HIHI)
    psd     = mq[vm] + EPS
    psd    /= psd.sum()
    feats.append(float(-np.sum(psd * np.log(psd))))   # entropy

    # Mean inter-axis Pearson correlation (X-Y, X-Z, Y-Z)
    corrs = []
    for i, j in [(0,1),(0,2),(1,2)]:
        a, b = window[:,i], window[:,j]
        sa, sb = a.std(), b.std()
        corrs.append(float(np.corrcoef(a, b)[0,1]) if sa > EPS and sb > EPS else 0.0)
    feats.append(float(np.mean(corrs)))

    feats.append(float(np.sqrt(np.mean(np.sum(window**2, axis=1)))))

    vm_idx = np.where(vm)[0]
    feats.append(float(mfreqs[vm_idx[np.argmax(mq[vm_idx])]]))

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(root, name):
    if not root.exists():
        print(f"  [SKIP] {name}: folder not found at {root}")
        return None, None, None

    # Find all *-X.npy files (pattern: {id}-X.npy / {id}-Y.npy)
    # Also handle plain X.npy / Y.npy as fallback.
    # Exclude any .venv / site-packages directories.
    def is_venv(p):
        return any(part.startswith('.venv') or part == 'site-packages'
                   for part in p.parts)

    all_x = sorted(
        p for p in root.rglob("*-X.npy") if not is_venv(p)
    )
    if not all_x:
        all_x = sorted(
            p for p in root.rglob("*.npy")
            if p.name.lower() == "x.npy" and not is_venv(p)
        )
    if not all_x:
        print(f"  [SKIP] {name}: no *-X.npy files found under {root}")
        print(f"         (run --probe to list what is there)")
        return None, None, None

    print(f"  {name}: {len(all_x)} segment file(s) …", end=" ", flush=True)

    parent_to_gid = {}
    X_all, y_all, g_all = [], [], []

    for xf in all_x:
        # Match {id}-X.npy -> {id}-Y.npy, or X.npy -> Y.npy
        stem = xf.name  # e.g. "42-X.npy"
        y_name = stem.replace("-X.npy", "-Y.npy").replace("X.npy", "Y.npy")
        yf = xf.parent / y_name
        if not yf.exists():
            continue

        try:
            Xs = np.load(xf, allow_pickle=False)
            Ys = np.load(yf, allow_pickle=False)
        except Exception as e:
            print(f"\n    Warning loading {xf.name}: {e}")
            continue

        # Normalise to [N, 128, 3]
        if Xs.ndim == 2 and Xs.shape[1] == 3:
            Xs = Xs[np.newaxis]; Ys = np.atleast_1d(Ys)[:1]
        if Xs.ndim != 3 or Xs.shape[2] != 3 or Xs.shape[1] != WINDOW_SIZE:
            continue

        # Use numeric prefix as subject ID if present (e.g. "42" from "42-X.npy")
        prefix = xf.name.split("-")[0]
        gid_key = prefix if prefix.isdigit() else str(xf.parent)
        gid = parent_to_gid.setdefault(gid_key, len(parent_to_gid))
        for i in range(len(Xs)):
            linear = remove_gravity(Xs[i].astype(np.float64))
            X_all.append(extract_features(linear))
            y_all.append(0 if int(Ys[i]) == 0 else 1)
            g_all.append(gid)

    if not X_all:
        print("0 usable windows")
        return None, None, None

    pos = sum(y_all)
    print(f"{len(X_all)} windows  [tremor={pos}, no-tremor={len(X_all)-pos}]")
    return (np.array(X_all, dtype=np.float32),
            np.array(y_all, dtype=np.int32),
            np.array(g_all, dtype=np.int32))


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
def export_model(clf, scaler, threshold, out="tremor_model.json"):
    trees = []
    for stage in clf.estimators_:
        t = stage[0].tree_
        trees.append({
            "children_left":  t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature":        t.feature.tolist(),
            "threshold":      [float(v) for v in t.threshold],
            "value":          [float(v) for v in t.value[:, 0, 0]],
        })
    blob = {
        "sample_rate":   FS,
        "window_size":   WINDOW_SIZE,
        "tremor_lo":     TREMOR_LO,
        "tremor_hi":     TREMOR_HI,
        "threshold":     threshold,
        "feature_names": FEATURE_NAMES,
        "n_estimators":  clf.n_estimators_,
        "learning_rate": clf.learning_rate,
        "init_score":    float(clf.init_.class_prior_[1]),
        "scaler_mean":   scaler.mean_.tolist(),
        "scaler_std":    scaler.scale_.tolist(),
        "trees":         trees,
    }
    with open(out, "w") as f:
        json.dump(blob, f, separators=(",", ":"))
    print(f"\n✓  Saved {out}  ({os.path.getsize(out)/1024:.0f} KB, {len(trees)} trees)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",  default=None,
        help="Full path to the Parkinson-s-Disease-Tremor-Dataset folder")
    ap.add_argument("--probe", action="store_true",
        help="Print the file tree and exit (diagnose missing data)")
    ap.add_argument("--out",   default="tremor_model.json",
        help="Output JSON filename (default: tremor_model.json)")
    args = ap.parse_args()

    repo = find_repo(args.repo)
    if repo is None:
        print("ERROR: dataset repo not found.  Use:")
        print("  python train_tremor_model.py --repo /path/to/Parkinson-s-Disease-Tremor-Dataset")
        sys.exit(1)

    print(f"Repo: {repo}")

    if args.probe:
        probe_repo(repo)
        return

    # 1. Load ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading …")
    Xp, yp, gp = [], [], []
    goff = 0
    for name, sub in DATASET_SUBDIRS.items():
        X, y, g = load_dataset(repo / sub, name)
        if X is None:
            continue
        Xp.append(X); yp.append(y); gp.append(g + goff)
        goff += int(g.max()) + 1

    if not Xp:
        print("\nNo data loaded.  Run with --probe to diagnose the file layout.")
        sys.exit(1)

    X = np.concatenate(Xp)
    y = np.concatenate(yp)
    groups = np.concatenate(gp)
    pos = int(y.sum()); neg = len(y) - pos
    print(f"\n  Total={len(X):,}  tremor={pos:,}({100*pos/len(y):.1f}%)  "
          f"no-tremor={neg:,}({100*neg/len(y):.1f}%)")

    # 2. Split (group-aware) ─────────────────────────────────────────────
    print("\n[2/5] Splitting …")
    rng = np.random.default_rng(42)
    ugroups = np.unique(groups); rng.shuffle(ugroups)
    n = len(ugroups)
    test_g = set(ugroups[:max(1, n//7)])
    val_g  = set(ugroups[max(1, n//7):max(2, n*2//7)])

    te = np.isin(groups, list(test_g))
    va = np.isin(groups, list(val_g))
    tr = ~(te | va)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X[tr])
    Xva_s = scaler.transform(X[va])
    Xte_s = scaler.transform(X[te])
    print(f"  Train={tr.sum():,}  Val={va.sum():,}  Test={te.sum():,}")

    # 3. Train ───────────────────────────────────────────────────────────
    print(f"\n[3/5] Training GBT ({N_ESTIMATORS} trees, depth {MAX_DEPTH}, "
          f"tremor_weight={TREMOR_CLASS_WEIGHT}×) …\n")
    sw = np.where(y[tr] == 1, TREMOR_CLASS_WEIGHT, 1.0)
    clf = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH, subsample=SUBSAMPLE,
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=42, verbose=1)
    clf.fit(Xtr_s, y[tr], sample_weight=sw)

    # 4. Tune threshold on val set (F2 = recall-biased F-score) ─────────
    print("\n[4/5] Tuning threshold (F2) on val set …")
    probs_va = clf.predict_proba(Xva_s)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y[va], probs_va)

    thresh = FORCE_THRESHOLD
    best_f2 = -1.0
    if thresh == 0.0:
        for p, r, t in zip(prec[:-1], rec[:-1], thresholds):
            denom = 4*p + r
            f2 = (5*p*r / denom) if denom > 0 else 0
            if f2 > best_f2:
                best_f2 = f2; thresh = float(t)
    print(f"  threshold={thresh:.4f}  F2={best_f2:.4f}")

    # 5. Report ──────────────────────────────────────────────────────────
    print("\n[5/5] Final evaluation …")
    probs_te = clf.predict_proba(Xte_s)[:, 1]
    preds_te = (probs_te >= thresh).astype(int)
    auc = roc_auc_score(y[te], probs_te)
    print(f"\n  ROC-AUC = {auc:.4f}\n")
    print(classification_report(y[te], preds_te, target_names=["no_tremor","tremor"]))
    cm = confusion_matrix(y[te], preds_te)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Sensitivity : {tp/(tp+fn+EPS):.3f}")
    print(f"  Specificity : {tn/(tn+fp+EPS):.3f}")
    if tp/(tp+fn+EPS) < 0.80:
        print("  ⚠  Sensitivity < 0.80 — increase TREMOR_CLASS_WEIGHT and retrain")

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ConfusionMatrixDisplay(cm, display_labels=["no tremor","tremor"]).plot(
            ax=ax1, cmap="Blues", colorbar=False)
        ax1.set_title(f"Confusion  AUC={auc:.3f}")
        RocCurveDisplay.from_predictions(y[te], probs_te, ax=ax2)
        plt.tight_layout(); plt.savefig("training_report.png", dpi=150)
        print("  Saved training_report.png")
    except ImportError:
        pass

    export_model(clf, scaler, thresh, args.out)
    print(f"Copy {args.out} next to index.html and refresh the dashboard.\n")


if __name__ == "__main__":
    main()
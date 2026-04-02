import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    f1_score, roc_curve, brier_score_loss,
    balanced_accuracy_score,
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET  (must match extract_features.py FEATURE_COLS)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "tracking_ratio",
    "fixation_count",
    "avg_fixation_duration_ms",
    "std_fixation_duration_ms",
    "total_fixation_time_ms",
    "fixation_rate",
    "pct_time_fixating",
    "avg_fixation_dispersion",
    "saccade_count",
    "avg_saccade_amplitude",
    "std_saccade_amplitude",
    "avg_saccade_velocity",
    "max_saccade_velocity",
    "saccade_rate",
    "gaze_std_x",
    "gaze_std_y",
    "gaze_mean_y",
    "avg_dist_from_center",
    "prop_center_200px",
    "prop_left_visual_field",
    "gaze_path_length",
    "scanpath_entropy_2d",
    "convex_hull_area_px2",
    "avg_inter_fixation_dist",
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS  (fixed per project spec)
# ─────────────────────────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators  = 200,
    max_depth     = 5,
    criterion     = "gini",         # Gini impurity splitting
    bootstrap     = True,           # bagging — sample with replacement
    max_features  = "sqrt",         # √p features per node
    min_samples_split = 4,          # guard against overfitting on tiny LOOCV folds
    min_samples_leaf  = 2,
    class_weight  = "balanced",     # handles ASD/TD imbalance
    n_jobs        = -1,
    random_state  = 42,
    oob_score     = True,           # out-of-bag estimate (free sanity check)
)

N_BOOTSTRAP   = 10_000   # resamples for BCa CI
N_PERMUTATIONS = 200   # permutation test resamples


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP BCa CONFIDENCE INTERVAL
# ─────────────────────────────────────────────────────────────────────────────

def bca_ci(stat_func, data, n_boot=10_000, alpha=0.05, rng=None):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Parameters
    ----------
    stat_func : callable → scalar
    data      : array-like passed to stat_func
    n_boot    : number of bootstrap resamples
    alpha     : significance level (0.05 → 95% CI)
    rng       : numpy RandomState / Generator

    Returns
    -------
    (lower, upper) floats
    """
    if rng is None:
        rng = np.random.default_rng(0)

    data = np.asarray(data)
    n    = len(data)
    obs  = stat_func(data)

    # Bootstrap distribution
    boot_stats = np.array([
        stat_func(data[rng.integers(0, n, n)])
        for _ in range(n_boot)
    ])

    # Bias correction z0
    z0 = _ppf_normal(np.mean(boot_stats < obs))

    # Acceleration a (jackknife)
    jack = np.array([stat_func(np.delete(data, i)) for i in range(n)])
    jack_mean = np.mean(jack)
    num  = np.sum((jack_mean - jack) ** 3)
    den  = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a    = num / den if den != 0 else 0.0

    # Adjusted quantiles
    z_alpha  = _ppf_normal(alpha / 2)
    z_1alpha = _ppf_normal(1 - alpha / 2)

    def adj(z):
        inner = z0 + (z0 + z) / (1 - a * (z0 + z))
        return _cdf_normal(inner)

    lo_q = np.clip(adj(z_alpha),  0.001, 0.999)
    hi_q = np.clip(adj(z_1alpha), 0.001, 0.999)

    lo = float(np.percentile(boot_stats, lo_q * 100))
    hi = float(np.percentile(boot_stats, hi_q * 100))
    return lo, hi


def _ppf_normal(p):
    """Inverse normal CDF (probit) via rational approximation."""
    from scipy.special import ndtri
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(ndtri(p))


def _cdf_normal(x):
    """Standard normal CDF."""
    from scipy.special import ndtr
    return float(ndtr(x))


# ─────────────────────────────────────────────────────────────────────────────
# LOOCV
# ─────────────────────────────────────────────────────────────────────────────

def run_loocv(X: np.ndarray, y: np.ndarray, participant_ids: list):
    """
    Leave-One-Out Cross-Validation.

    For each fold i:
      - Train RF on all participants except i
      - Predict probability for participant i
      - Accumulate OOB importance (averaged across folds)

    Returns
    -------
    probs      : ndarray (N,)  predicted P(ASD) for each participant
    importances: ndarray (N_feat, N_folds)  Gini importance per fold
    perm_imps  : ndarray (N_feat, N_folds)  permutation importance per fold
    """
    N = len(y)
    probs       = np.zeros(N)
    importances = np.zeros((X.shape[1], N))
    perm_imps   = np.zeros((X.shape[1], N))

    print(f"\n  Running LOOCV ({N} folds)…")
    for i in range(N):
        # Train / test split
        train_idx = np.array([j for j in range(N) if j != i])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test            = X[[i]]
        y_test            = y[[i]]

        # Scale features  (fit on training set only — no leakage)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Train RF
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)

        # Predict
        prob = rf.predict_proba(X_test)[0, 1]   # P(ASD)
        probs[i] = prob

        # Gini feature importance for this fold
        importances[:, i] = rf.feature_importances_

        # Permutation importance (on training set — no test data leakage)
        perm = permutation_importance(
            rf, X_train, y_train,
            n_repeats=5, random_state=42, scoring="roc_auc"
        )
        perm_imps[:, i] = perm.importances_mean

        pid_str = participant_ids[i] if participant_ids else str(i)
        label   = "ASD" if y[i] == 1 else "TD "
        flag    = "✓" if (prob > 0.5) == (y[i] == 1) else "✗"
        print(f"    fold {i+1:3d}/{N}  [{label}] {pid_str:30s}"
              f"  p(ASD)={prob:.3f}  {flag}")

    return probs, importances, perm_imps


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, probs, threshold=0.5):
    """Compute a full classification metric dict."""
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "auc":          roc_auc_score(y_true, probs),
        "accuracy":     accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "sensitivity":  sensitivity,      # recall / TPR
        "specificity":  specificity,      # TNR
        "ppv":          ppv,              # precision
        "npv":          npv,
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "brier":        brier_score_loss(y_true, probs),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def optimal_threshold(y_true, probs):
    """Youden's J statistic: argmax(sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(X, y, participant_ids, observed_auc, n_perm=N_PERMUTATIONS):
    """
    Permute labels and re-run full LOOCV to get empirical null distribution.

    Returns
    -------
    null_aucs : ndarray (n_perm,)
    p_value   : float
    """
    print(f"\n  Permutation test ({n_perm} permutations)…")
    rng = np.random.default_rng(99)
    null_aucs = np.zeros(n_perm)

    for p in range(n_perm):
        y_perm = rng.permutation(y)
        probs_perm, _, _ = run_loocv_quiet(X, y_perm)
        null_aucs[p] = roc_auc_score(y_perm, probs_perm)
        if (p + 1) % 100 == 0:
            print(f"    {p+1}/{n_perm}  running null AUC mean: {null_aucs[:p+1].mean():.4f}")

    p_value = float(np.mean(null_aucs >= observed_auc))
    return null_aucs, p_value


def run_loocv_quiet(X, y):
    """Silent LOOCV — used inside permutation test loop."""
    N     = len(y)
    probs = np.zeros(N)
    imps  = np.zeros((X.shape[1], N))
    pimps = np.zeros((X.shape[1], N))
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test            = X[[i]]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        probs[i] = rf.predict_proba(X_test)[0, 1]
        imps[:, i]  = rf.feature_importances_
        pimps[:, i] = 0.0  # skip permutation importance in null loop (speed)
    return probs, imps, pimps


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(y_true, probs, auc_val, ci_lo, ci_hi, out_path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2,
            label=f"LOOCV AUC = {auc_val:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#2563EB")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — LOOCV")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_permutation(null_aucs, obs_auc, p_val, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_aucs, bins=40, color="#94A3B8", edgecolor="white", alpha=0.85,
            label=f"Null AUC (n={len(null_aucs)})")
    ax.axvline(obs_auc, color="#DC2626", lw=2.5, label=f"Observed AUC = {obs_auc:.3f}")
    ax.set(xlabel="AUC under label permutation", ylabel="Count",
           title=f"Permutation Test  (p = {p_val:.4f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_feature_importance(feat_names, gini_means, gini_stds,
                            perm_means, perm_stds, out_path):
    idx = np.argsort(gini_means)[::-1]
    top = min(20, len(idx))
    idx = idx[:top]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gini importance
    ax = axes[0]
    colors = ["#2563EB" if gini_means[i] > np.median(gini_means) else "#93C5FD"
              for i in idx]
    ax.barh([feat_names[i] for i in idx[::-1]],
            gini_means[idx[::-1]],
            xerr=gini_stds[idx[::-1]],
            color=colors[::-1], ecolor="#475569", capsize=3, height=0.6)
    ax.set(xlabel="Mean Gini Importance (± std across folds)",
           title="Feature Importance — Gini (averaged LOOCV)")
    ax.grid(axis="x", alpha=0.3)

    # Permutation importance
    ax = axes[1]
    pidx = np.argsort(perm_means)[::-1][:top]
    ax.barh([feat_names[i] for i in pidx[::-1]],
            perm_means[pidx[::-1]],
            xerr=perm_stds[pidx[::-1]],
            color="#10B981", ecolor="#475569", capsize=3, height=0.6)
    ax.set(xlabel="Mean Permutation Importance (± std across folds)",
           title="Feature Importance — Permutation")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_participant_probs(participant_ids, y_true, probs, threshold, out_path):
    sort_idx = np.argsort(probs)
    pids_s   = [participant_ids[i] for i in sort_idx]
    probs_s  = probs[sort_idx]
    colors   = []
    for i in sort_idx:
        if y_true[i] == 1 and probs[i] >= threshold:
            colors.append("#16A34A")    # ASD correct
        elif y_true[i] == 1 and probs[i] < threshold:
            colors.append("#DC2626")    # ASD missed
        elif y_true[i] == 0 and probs[i] < threshold:
            colors.append("#2563EB")    # TD correct
        else:
            colors.append("#F97316")    # TD false alarm

    fig, ax = plt.subplots(figsize=(12, max(5, len(pids_s) * 0.22)))
    ax.barh(range(len(pids_s)), probs_s, color=colors, height=0.7)
    ax.axvline(threshold, color="black", lw=1.5, ls="--", label=f"Threshold={threshold:.2f}")
    ax.set_yticks(range(len(pids_s)))
    ax.set_yticklabels(pids_s, fontsize=7)
    ax.set(xlabel="P(ASD)", xlim=(0, 1),
           title="Per-participant ASD probability (LOOCV)")

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#16A34A", label="ASD — correctly classified"),
        Patch(facecolor="#DC2626", label="ASD — missed"),
        Patch(facecolor="#2563EB", label="TD  — correctly classified"),
        Patch(facecolor="#F97316", label="TD  — false alarm"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_calibration(y_true, probs, out_path):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, "o-", color="#2563EB", lw=2, label="RF LOOCV")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    brier = brier_score_loss(y_true, probs)
    ax.set(xlabel="Mean predicted P(ASD)", ylabel="Fraction true ASD",
           title=f"Calibration Curve  (Brier={brier:.3f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train & evaluate ASD eye-tracking RF via LOOCV."
    )
    ap.add_argument("--features",  "-f", required=True,
                    help="Path to features CSV (from extract_features.py).")
    ap.add_argument("--out_dir",   "-o", default="results",
                    help="Directory for plots and reports (default: results/).")
    ap.add_argument("--model_dir", "-m", default="models",
                    help="Directory to save the final model (default: models/).")
    ap.add_argument("--no_permtest", action="store_true",
                    help="Skip permutation test (speeds up debugging runs).")
    args = ap.parse_args()

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ASD Eye-Tracking Classifier — LOOCV Training")
    print("=" * 65)

    df = pd.read_csv(args.features)
    print(f"\n  Loaded {len(df)} participants from {args.features}")

    # Keep only columns that exist in both the spec and the CSV
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\n  [WARN] Missing feature columns (will be ignored): {missing}")
    if len(available) < 5:
        print("[ERROR] Too few features available. Check extract_features.py output.")
        sys.exit(1)

    feature_names = available
    X = df[feature_names].values.astype(float)
    y = df["label"].values.astype(int)
    participant_ids = df["participant_id"].tolist() if "participant_id" in df.columns \
                      else [str(i) for i in range(len(df))]

    # Replace any NaN with column median (rare edge case)
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = np.isnan(X[:, j])
        X[bad, j] = col_medians[j]

    n_asd = int(y.sum())
    n_td  = int(len(y) - n_asd)
    print(f"  ASD={n_asd}  TD={n_td}  Features={len(feature_names)}")
    print(f"\n  RF params: {RF_PARAMS}")

    # ── LOOCV ──────────────────────────────────────────────────────────────
    probs, gini_imps, perm_imps = run_loocv(X, y, participant_ids)

    # ── Metrics ────────────────────────────────────────────────────────────
    thresh_default = 0.5
    thresh_optimal = optimal_threshold(y, probs)
    metrics_50  = compute_metrics(y, probs, threshold=0.5)
    metrics_opt = compute_metrics(y, probs, threshold=thresh_optimal)

    obs_auc = metrics_50["auc"]

    # BCa bootstrap CI on AUC
    print(f"\n  Computing BCa bootstrap CI ({N_BOOTSTRAP} resamples)…")
    rng = np.random.default_rng(42)

    def _auc_func(data):
        idx  = data.astype(int)
        y_s  = y[idx]
        p_s  = probs[idx]
        if len(np.unique(y_s)) < 2:
            return 0.5
        return roc_auc_score(y_s, p_s)

    ci_lo, ci_hi = bca_ci(
        _auc_func,
        np.arange(len(y)),
        n_boot=N_BOOTSTRAP,
        rng=rng,
    )

    # ── Permutation test ────────────────────────────────────────────────────
    if not args.no_permtest:
        null_aucs, p_value = permutation_test(X, y, participant_ids, obs_auc,
                                               n_perm=N_PERMUTATIONS)
    else:
        null_aucs = np.array([0.5])
        p_value   = float("nan")
        print("\n  [Permutation test skipped]")

    # ── Feature importance summary ──────────────────────────────────────────
    gini_mean = gini_imps.mean(axis=1)
    gini_std  = gini_imps.std(axis=1)
    perm_mean = perm_imps.mean(axis=1)
    perm_std  = perm_imps.std(axis=1)

    imp_df = pd.DataFrame({
        "feature":        feature_names,
        "gini_importance": gini_mean,
        "gini_std":        gini_std,
        "perm_importance": perm_mean,
        "perm_std":        perm_std,
    }).sort_values("gini_importance", ascending=False)

    # ── Print results ───────────────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print("  LOOCV RESULTS")
    print(sep)
    print(f"  AUC        : {obs_auc:.4f}  (95% BCa CI: {ci_lo:.3f}–{ci_hi:.3f})")
    print(f"  Accuracy   : {metrics_50['accuracy']:.4f}")
    print(f"  Balanced ↑ : {metrics_50['balanced_acc']:.4f}")
    print(f"  Sensitivity: {metrics_50['sensitivity']:.4f}  (TPR / recall)")
    print(f"  Specificity: {metrics_50['specificity']:.4f}  (TNR)")
    print(f"  PPV        : {metrics_50['ppv']:.4f}  (precision)")
    print(f"  NPV        : {metrics_50['npv']:.4f}")
    print(f"  F1         : {metrics_50['f1']:.4f}")
    print(f"  Brier      : {metrics_50['brier']:.4f}  (0=perfect, 0.25=chance)")
    print(f"  TP={metrics_50['tp']}  TN={metrics_50['tn']}"
          f"  FP={metrics_50['fp']}  FN={metrics_50['fn']}")
    print(f"\n  Optimal threshold (Youden's J): {thresh_optimal:.3f}")
    print(f"  At optimal threshold →  Sens={metrics_opt['sensitivity']:.4f}"
          f"  Spec={metrics_opt['specificity']:.4f}  Acc={metrics_opt['accuracy']:.4f}")
    if not np.isnan(p_value):
        print(f"\n  Permutation test p-value : {p_value:.4f}"
              f"  ({'SIGNIFICANT' if p_value < 0.05 else 'not significant'} at α=0.05)")
        print(f"  Null AUC mean / std      : {null_aucs.mean():.4f} / {null_aucs.std():.4f}")

    print(f"\n  Top 10 features (Gini importance):")
    for _, row in imp_df.head(10).iterrows():
        bar = "█" * int(row["gini_importance"] * 200)
        print(f"    {row['feature']:<35} {row['gini_importance']:.4f} ±{row['gini_std']:.4f}  {bar}")

    # ── Save artefacts ──────────────────────────────────────────────────────
    # Per-participant probabilities CSV
    prob_df = pd.DataFrame({
        "participant_id": participant_ids,
        "true_label":     y,
        "p_asd":          probs,
        "predicted_50":   (probs >= 0.5).astype(int),
        "predicted_opt":  (probs >= thresh_optimal).astype(int),
        "correct_50":     ((probs >= 0.5) == y).astype(int),
    })
    prob_path = os.path.join(args.out_dir, "loocv_participant_probs.csv")
    prob_df.to_csv(prob_path, index=False)
    print(f"\n  Saved: {prob_path}")

    # Feature importances CSV
    imp_path = os.path.join(args.model_dir, "feature_importances.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")

    # Summary metrics CSV
    summary = {**metrics_50,
               "auc_ci_lo": ci_lo, "auc_ci_hi": ci_hi,
               "permutation_p": p_value,
               "optimal_threshold": thresh_optimal,
               **{f"opt_{k}": v for k, v in metrics_opt.items()}}
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.out_dir, "loocv_summary.csv"), index=False)

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\n  Generating plots…")
    plot_roc(y, probs, obs_auc, ci_lo, ci_hi,
             os.path.join(args.out_dir, "roc_curve.png"))
    if not np.isnan(p_value):
        plot_permutation(null_aucs, obs_auc, p_value,
                         os.path.join(args.out_dir, "permutation_test.png"))
    plot_feature_importance(
        feature_names, gini_mean, gini_std, perm_mean, perm_std,
        os.path.join(args.out_dir, "feature_importance.png"))
    plot_participant_probs(
        participant_ids, y, probs, thresh_optimal,
        os.path.join(args.out_dir, "participant_probs.png"))
    plot_calibration(y, probs,
                     os.path.join(args.out_dir, "calibration_curve.png"))

    # ── Final model — trained on ALL participants ───────────────────────────
    print("\n  Training final model on all participants…")
    scaler_final = StandardScaler()
    X_all        = scaler_final.fit_transform(X)
    rf_final     = RandomForestClassifier(**RF_PARAMS)
    rf_final.fit(X_all, y)

    oob_auc_text = f"{rf_final.oob_score_:.4f}" if hasattr(rf_final, "oob_score_") else "N/A"
    print(f"  OOB score (sanity check, not substitute for LOOCV): {oob_auc_text}")

    model_bundle = {
        "model":          rf_final,
        "scaler":         scaler_final,
        "feature_names":  feature_names,
        "rf_params":      RF_PARAMS,
        "loocv_auc":      obs_auc,
        "loocv_auc_ci":   (ci_lo, ci_hi),
        "permutation_p":  p_value,
        "optimal_threshold": thresh_optimal,
        "metrics":        metrics_50,
    }
    model_path = os.path.join(args.model_dir, "final_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(model_bundle, fh)
    print(f"  Saved: {model_path}")

    feat_path = os.path.join(args.model_dir, "feature_names.pkl")
    with open(feat_path, "wb") as fh:
        pickle.dump(feature_names, fh)
    print(f"  Saved: {feat_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  LOOCV AUC  = {obs_auc:.4f}  (95% CI: {ci_lo:.3f}–{ci_hi:.3f})")
    print(f"  Accuracy   = {metrics_50['accuracy']:.4f}")
    print(f"  Sensitivity= {metrics_50['sensitivity']:.4f}")
    print(f"  Specificity= {metrics_50['specificity']:.4f}")
    if not np.isnan(p_value):
        print(f"  Perm p     = {p_value:.4f}  {'✓ significant' if p_value < 0.05 else '✗ not significant'}")
    print(f"\n  Model saved → {model_path}")
    print(f"  Plots saved → {args.out_dir}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

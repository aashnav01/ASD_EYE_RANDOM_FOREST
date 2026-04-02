import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match degrade_data_webcam.py and the webcam tracker)
# ─────────────────────────────────────────────────────────────────────────────
SCREEN_W        = 1280          # px — degraded webcam output
SCREEN_H        = 1024
CENTER_X        = SCREEN_W / 2  # 640 px
CENTER_Y        = SCREEN_H / 2  # 512 px
CENTER_RADIUS   = 200           # px — keep same radius (or adjust if desired)
IVT_VEL_THRESH  = 300           # px/s  velocity threshold for I-VT
MIN_FIX_MS      = 80            # ms    minimum valid fixation
MIN_SACC_AMP_PX = 5             # px    minimum saccade amplitude (noise guard)
# Columns expected in each CSV
COL_T   = "RecordingTime [ms]"
COL_X   = "Point of Regard Right X [px]"
COL_Y   = "Point of Regard Right Y [px]"
COL_TR  = "Tracking Ratio [%]"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    """Load a Cilia-format CSV (degraded or live webcam)."""
    df = pd.read_csv(path, low_memory=False)
    for col in [COL_T, COL_X, COL_Y, COL_TR]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {path}")
    df[COL_T]  = pd.to_numeric(df[COL_T],  errors="coerce")
    df[COL_X]  = pd.to_numeric(df[COL_X],  errors="coerce")
    df[COL_Y]  = pd.to_numeric(df[COL_Y],  errors="coerce")
    df[COL_TR] = pd.to_numeric(df[COL_TR], errors="coerce")
    return df


def _clean_gaze(df: pd.DataFrame):
    """Return (x, y, t) arrays for valid tracked frames, sorted by time."""
    mask = (
        df[COL_X].notna() & df[COL_Y].notna() & df[COL_T].notna() &
        (df[COL_X] > 0) & (df[COL_Y] > 0)
    )
    sub = df[mask].sort_values(COL_T)
    return sub[COL_X].values, sub[COL_Y].values, sub[COL_T].values

def _run_ivt(x, y, t):
    """
    I-VT fixation classification.

    Returns
    -------
    fixations : list of dict  {x_mean, y_mean, duration_ms, start_ms}
    saccades  : list of dict  {amplitude_px, peak_vel_px_s, avg_vel_px_s}
    velocities: ndarray       per-frame velocity (px/s), len = len(x)-1
    """
    if len(t) < 4:
        return [], [], np.array([])

    dt = np.diff(t) / 1000.0          # seconds between frames
    dt = np.clip(dt, 1e-3, 1.0)       # guard against duplicates / huge gaps

    vx  = np.diff(x) / dt
    vy  = np.diff(y) / dt
    vel = np.sqrt(vx ** 2 + vy ** 2)  # px/s

    is_fix = vel < IVT_VEL_THRESH      # True = fixation frame

    fixations, saccades = [], []
    n = len(is_fix)
    i = 0
    while i < n:
        j = i + 1
        while j < n and is_fix[j] == is_fix[i]:
            j += 1
        # segment [i, j) shares the same classification
        seg_x   = x[i:j + 1]
        seg_y   = y[i:j + 1]
        seg_t   = t[i:j + 1]
        dur_ms  = seg_t[-1] - seg_t[0] if len(seg_t) > 1 else 0

        if is_fix[i]:
            if dur_ms >= MIN_FIX_MS:
                fixations.append({
                    "x_mean":     float(np.mean(seg_x)),
                    "y_mean":     float(np.mean(seg_y)),
                    "duration_ms": float(dur_ms),
                    "start_ms":   float(seg_t[0]),
                    # intra-fixation dispersion (RMS deviation from centroid)
                    "dispersion": float(np.sqrt(
                        np.var(seg_x) + np.var(seg_y)
                    )),
                })
        else:
            amp = float(np.hypot(seg_x[-1] - seg_x[0], seg_y[-1] - seg_y[0]))
            if amp >= MIN_SACC_AMP_PX and dur_ms > 0:
                seg_vel = vel[i:j]
                saccades.append({
                    "amplitude_px":   amp,
                    "peak_vel_px_s":  float(np.max(seg_vel)),
                    "avg_vel_px_s":   float(np.mean(seg_vel)),
                })
        i = j

    return fixations, saccades, vel


def _shannon_entropy_2d(x, y, bins=10):
    """
    2D Shannon entropy of gaze distribution (scanpath irregularity).
    Higher = more uniform / exploratory; lower = more restricted / repetitive.
    Literature: Zhou et al. 2024, Frontiers in Neuroscience.
    """
    if len(x) < 4:
        return 0.0
    H, _, _ = np.histogram2d(
        np.clip(x, 0, SCREEN_W),
        np.clip(y, 0, SCREEN_H),
        bins=bins,
        range=[[0, SCREEN_W], [0, SCREEN_H]],
    )
    H = H.ravel()
    H = H[H > 0]
    p = H / H.sum()
    return float(-np.sum(p * np.log2(p)))


def _convex_hull_area(x, y):
    """
    Area of the convex hull of gaze points (px²).
    Measures spatial coverage / exploration breadth.
    Literature: Zhou et al. 2024.
    """
    if len(x) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([x, y])
        hull = ConvexHull(pts)
        return float(hull.volume)          # .volume = area in 2D
    except Exception:
        # Fallback: bounding-box area (if scipy not available or collinear pts)
        return float((x.max() - x.min()) * (y.max() - y.min()))


def _fixation_dispersion_mean(fixations):
    """Mean intra-fixation RMS dispersion across all fixations (px)."""
    if not fixations:
        return 0.0
    return float(np.mean([f["dispersion"] for f in fixations]))


def _inter_fixation_distances(fixations):
    """
    Distances between consecutive fixation centroids (px).
    Returns array; used for mean saccadic amplitude cross-check.
    """
    if len(fixations) < 2:
        return np.array([0.0])
    xs = np.array([f["x_mean"] for f in fixations])
    ys = np.array([f["y_mean"] for f in fixations])
    return np.hypot(np.diff(xs), np.diff(ys))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(csv_path: str) -> dict:
    """
    Extract all 25 features from a single session CSV.

    Returns a flat dict.  Returns None if the session is too short / bad.
    """
    df = _load_csv(csv_path)
    t_all = df[COL_T].dropna()
    if t_all.empty:
        return None

    total_rows    = len(df)
    total_dur_ms  = t_all.max() - t_all.min()
    if total_dur_ms < 5000:
        return None

    # --- Tracking ratio (Ahmed et al. 2023 — feature #1) ---
    tr = df[COL_TR]
    tracking_ratio = float((tr == 100).sum() / max(len(tr), 1))

    # --- Clean gaze data ---
    x, y, t = _clean_gaze(df)
    if len(x) < 10:
        return None

    # Observed session duration from valid frames
    session_dur_ms = float(t[-1] - t[0])
    if session_dur_ms < 2000:
        return None

    # --- FPS from valid frames ---
    dts = np.diff(t)
    dts_valid = dts[(dts > 0) & (dts < 500)]
    session_fps = float(1000.0 / np.median(dts_valid)) if len(dts_valid) else 0.0

    # --- I-VT fixation & saccade detection ---
    fixations, saccades, vel_all = _run_ivt(x, y, t)

    n_fix  = len(fixations)
    n_sacc = len(saccades)

    fix_durs  = [f["duration_ms"]  for f in fixations]
    sacc_amps = [s["amplitude_px"] for s in saccades]
    sacc_vels = [s["avg_vel_px_s"] for s in saccades]
    sacc_peak = [s["peak_vel_px_s"] for s in saccades]

    # ── Fixation features ──
    avg_fix_dur   = float(np.mean(fix_durs))   if fix_durs  else 0.0
    std_fix_dur   = float(np.std(fix_durs))    if fix_durs  else 0.0
    total_fix_ms  = float(np.sum(fix_durs))    if fix_durs  else 0.0
    fix_rate      = float(n_fix / (session_dur_ms / 1000.0))   # fix/s
    pct_time_fix  = total_fix_ms / session_dur_ms if session_dur_ms > 0 else 0.0

    # Intra-fixation dispersion (Meng et al. 2023)
    avg_fix_disp  = _fixation_dispersion_mean(fixations)

    # ── Saccade features ──
    avg_sacc_amp  = float(np.mean(sacc_amps))  if sacc_amps else 0.0
    avg_sacc_vel  = float(np.mean(sacc_vels))  if sacc_vels else 0.0
    max_sacc_vel  = float(np.max(sacc_peak))   if sacc_peak else 0.0  # Meng et al. 2023
    sacc_rate     = float(n_sacc / (session_dur_ms / 1000.0))
    std_sacc_amp  = float(np.std(sacc_amps))   if sacc_amps else 0.0

    # ── Gaze position features ──
    gaze_std_x    = float(np.std(x))
    gaze_std_y    = float(np.std(y))           # Ahmed et al. 2023 — #3 most important
    gaze_mean_y   = float(np.mean(y))

    # Distance from screen center (Franco et al. 2023 — #1 most important)
    dist_from_ctr = np.hypot(x - CENTER_X, y - CENTER_Y)
    avg_dist_ctr  = float(np.mean(dist_from_ctr))
    prop_center   = float(np.mean(dist_from_ctr < CENTER_RADIUS))  # Franco et al. 2023

    # Lateral bias (left vs right visual field)
    prop_left_vf  = float(np.mean(x < CENTER_X))

    # Scanpath total path length (px)
    path_length   = float(np.sum(np.hypot(np.diff(x), np.diff(y))))

    # ── Advanced scanpath features (Zhou et al. 2024) ──
    entropy_2d    = _shannon_entropy_2d(x, y, bins=10)
    hull_area     = _convex_hull_area(x, y)

    # ── Inter-fixation distance (saccadic amplitude from fixation centres) ──
    ifd = _inter_fixation_distances(fixations)
    avg_ifd = float(np.mean(ifd)) if len(ifd) > 0 else 0.0

    return {
        # ── Tracking ──
        "tracking_ratio":           tracking_ratio,         # Ahmed #1

        # ── Fixation ──
        "fixation_count":           float(n_fix),
        "avg_fixation_duration_ms": avg_fix_dur,
        "std_fixation_duration_ms": std_fix_dur,
        "total_fixation_time_ms":   total_fix_ms,
        "fixation_rate":            fix_rate,
        "pct_time_fixating":        pct_time_fix,
        "avg_fixation_dispersion":  avg_fix_disp,           # Meng 2023

        # ── Saccade ──
        "saccade_count":            float(n_sacc),
        "avg_saccade_amplitude":    avg_sacc_amp,
        "std_saccade_amplitude":    std_sacc_amp,
        "avg_saccade_velocity":     avg_sacc_vel,
        "max_saccade_velocity":     max_sacc_vel,           # Meng #1
        "saccade_rate":             sacc_rate,

        # ── Gaze position ──
        "gaze_std_x":               gaze_std_x,
        "gaze_std_y":               gaze_std_y,             # Ahmed #3
        "gaze_mean_y":              gaze_mean_y,
        "avg_dist_from_center":     avg_dist_ctr,           # Franco #1
        "prop_center_200px":        prop_center,            # Franco
        "prop_left_visual_field":   prop_left_vf,

        # ── Scanpath ──
        "gaze_path_length":         path_length,
        "scanpath_entropy_2d":      entropy_2d,             # Zhou 2024
        "convex_hull_area_px2":     hull_area,              # Zhou 2024
        "avg_inter_fixation_dist":  avg_ifd,

        # ── Session meta (not used as features but kept for QC) ──
        "_session_dur_ms":          session_dur_ms,
        "_session_fps":             session_fps,
        "_n_valid_frames":          float(len(x)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
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

META_COLS = ["_session_dur_ms", "_session_fps", "_n_valid_frames"]


def process_folder(degraded_root: str, output_csv: str):
    """
    Walk degraded_root/{asd,td}/*.csv, extract features, write output_csv.
    """
    records = []
    skipped = []

    for label_name, label_int in [("asd", 1), ("td", 0)]:
        folder = os.path.join(degraded_root, label_name)
        if not os.path.isdir(folder):
            print(f"  [WARN] Folder not found: {folder} — skipping")
            continue

        csv_files = sorted(glob(os.path.join(folder, "*.csv")))
        print(f"\n  Processing {label_name.upper()} ({len(csv_files)} files)…")

        for csv_path in csv_files:
            pid = Path(csv_path).stem
            try:
                feats = extract_features(csv_path)
                if feats is None:
                    print(f"    [SKIP] {pid} — too short or no valid gaze data")
                    skipped.append(pid)
                    continue
                feats["participant_id"] = pid
                feats["label"] = label_int
                records.append(feats)
                _dur = feats["_session_dur_ms"] / 1000
                _fps = feats["_session_fps"]
                print(f"    [OK]   {pid:30s}  dur={_dur:.0f}s  fps={_fps:.1f}"
                      f"  fix={int(feats['fixation_count'])}  sacc={int(feats['saccade_count'])}")
            except Exception as exc:
                print(f"    [ERR]  {pid} — {exc}")
                skipped.append(pid)

    if not records:
        print("\n[ERROR] No valid sessions extracted. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # Reorder: id, label, features, meta
    col_order = (
        ["participant_id", "label"]
        + FEATURE_COLS
        + META_COLS
    )
    df = df[[c for c in col_order if c in df.columns]]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n{'─'*60}")
    print(f"  Sessions extracted : {len(records)}")
    print(f"  Sessions skipped   : {len(skipped)}")
    print(f"  ASD / TD           : {(df['label']==1).sum()} / {(df['label']==0).sum()}")
    print(f"  Features           : {len(FEATURE_COLS)}")
    print(f"  Output             : {output_csv}")

    # Quick sanity check — print feature means by group
    print(f"\n{'─'*60}")
    print("  Feature means by group (ASD vs TD):")
    print(f"  {'Feature':<30} {'ASD':>10} {'TD':>10}  {'p-val':>8}")
    asd_df = df[df["label"] == 1]
    td_df  = df[df["label"] == 0]
    from scipy import stats as sp_stats
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            continue
        a = asd_df[feat].dropna()
        t = td_df[feat].dropna()
        if len(a) < 2 or len(t) < 2:
            continue
        _, pval = sp_stats.mannwhitneyu(a, t, alternative="two-sided")
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  {feat:<30} {a.mean():>10.3f} {t.mean():>10.3f}  {pval:>8.4f} {sig}")

    return df


def process_single(csv_path: str, output_csv: str):
    """Extract features from one CSV (live webcam session)."""
    feats = extract_features(csv_path)
    if feats is None:
        print("[ERROR] Could not extract features — session too short or missing data.")
        sys.exit(1)
    df = pd.DataFrame([feats])
    col_order = ["participant_id", "label"] + FEATURE_COLS + META_COLS
    df = df[[c for c in col_order if c in df.columns]]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Features written to {output_csv}")
    for feat in FEATURE_COLS:
        if feat in feats:
            print(f"  {feat:<35} {feats[feat]:.4f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract ASD eye-tracking features from webcam-format CSVs."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--degraded", metavar="DIR",
        help="Root folder containing asd/ and td/ sub-folders of degraded CSVs."
    )
    group.add_argument(
        "--csv", metavar="FILE",
        help="Single session CSV (e.g., live webcam output)."
    )
    ap.add_argument(
        "--output", "-o", metavar="FILE", required=True,
        help="Output CSV path for the features table."
    )
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("  ASD Eye-Tracking Feature Extraction")
    print("=" * 60)

    if args.degraded:
        process_folder(args.degraded, args.output)
    else:
        process_single(args.csv, args.output)


if __name__ == "__main__":
    main()

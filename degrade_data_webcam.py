import os
import argparse
import warnings
import numpy as np
import pandas as pd
from glob import glob

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        items = list(iterable)
        n = len(items)
        for i, item in enumerate(items):
            if n <= 10 or i % max(1, n // 5) == 0:
                print(f"  {desc}: {i}/{n}")
            yield item


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — tuned to match child webcam targets
# ─────────────────────────────────────────────────────────────────────────────

# Session window to extract from each Cilia file
SESSION_DURATION_MS        = 105_000   # target window (ms) — matches webcam ~108s
SESSION_DURATION_JITTER_MS =  10_000   # +/- jitter so sessions are not identical
MIN_BLOCK_MS               =  60_000   # minimum contiguous block to be usable
BLOCK_GAP_MS               =   5_000   # inter-frame gap that marks a new block

# Target FPS — must match your webcam median (16.1 Hz measured)
TARGET_FPS = 16

# Spatial noise (webcam gaze estimation error ~50-120 px)
NOISE_STD_X = 30   # px horizontal (reduced)
NOISE_STD_Y = 40   # px vertical

# Drift — slow head-sway shared across both eyes
DRIFT_STEP_X =  0.2   # px/frame std of random walk (reduced)
DRIFT_STEP_Y =  0.4
DRIFT_CLIP   = 40     # max absolute drift (px)

# Bursty dropout — child looks away / blinks
BURST_PROB        = 0.006  # probability per frame a burst starts (reduced)
BURST_MEAN_FRAMES = 14     # mean length at 16 fps ~0.9 s
BURST_MAX_FRAMES  = 35     # max length ~2.2 s
FORWARD_FILL_LIMIT = 3     # frames to ffill (short blinks)

# X distribution — compress horizontal spread (Cilia ~400px → target ~150px)
X_SCALE = 0.6   # trial value, adjust if needed

# Y distribution
Y_SCALE            = 0.25   # compression factor
Y_CENTER_TARGET    = 380    # px — child looks slightly below screen centre
Y_STD_MIN_TARGET   = 40     # px — if result is below this, boost with noise
Y_STD_NOISE_BOOST  = 30     # px extra Gaussian noise added if Y_std too low

# Screen bounds (Cilia stimulus screen)
SCREEN_W = 1280
SCREEN_H = 1024

# Column names
TIME_COL  = "RecordingTime [ms]"
TRACK_COL = "Tracking Ratio [%]"

GAZE_X_COLS = [
    "Point of Regard Right X [px]",
    "Point of Regard Left X [px]",
]
GAZE_Y_COLS = [
    "Point of Regard Right Y [px]",
    "Point of Regard Left Y [px]",
]

# Columns unavailable from webcam — drop so model never trains on them
DROP_COLS = [
    "Pupil Size Right X [px]",   "Pupil Size Right Y [px]",
    "Pupil Size Left X [px]",    "Pupil Size Left Y [px]",
    "Eye Position Right X [mm]", "Eye Position Right Y [mm]", "Eye Position Right Z [mm]",
    "Eye Position Left X [mm]",  "Eye Position Left Y [mm]",  "Eye Position Left Z [mm]",
    "Pupil Position Right X [px]","Pupil Position Right Y [px]",
    "Pupil Position Left X [px]", "Pupil Position Left Y [px]",
    "Eye Movement Type", "Eye Movement Type Index",
    "Fixation Index", "Saccade Index",
    "Category Right", "Category Left",
    "Export Start Trial Time [ms]", "Export End Trial Time [ms]",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: detect contiguous trial blocks
# ─────────────────────────────────────────────────────────────────────────────

def get_blocks(t_series: pd.Series):
    """
    Split a timestamp series into contiguous blocks separated by gaps.
    Returns list of (start_ms, end_ms) sorted longest-first.
    """
    t = t_series.sort_values().reset_index(drop=True)
    diffs = t.diff().fillna(0)
    gap_positions = diffs[diffs > BLOCK_GAP_MS].index.tolist()

    starts = [0] + gap_positions
    ends   = gap_positions + [len(t)]

    blocks = []
    for s, e in zip(starts, ends):
        t_start = float(t.iloc[s])
        t_end   = float(t.iloc[e - 1])
        if (t_end - t_start) >= MIN_BLOCK_MS:
            blocks.append((t_start, t_end))

    blocks.sort(key=lambda b: b[1] - b[0], reverse=True)
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# CORE DEGRADATION
# ─────────────────────────────────────────────────────────────────────────────

def degrade(df: pd.DataFrame, rng: np.random.Generator):
    """
    Transform one participant's premium eye-tracker DataFrame into one
    that statistically matches webcam output from the same child.
    Returns degraded DataFrame or None if the file is unusable.
    """
    df = df.copy()

    # Sort and clean timestamps
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    # Deduplicate timestamps — Cilia records both eyes per timestamp
    df = df.drop_duplicates(subset=[TIME_COL], keep='first').reset_index(drop=True)

    # Find trial blocks and use the longest one
    blocks = get_blocks(df[TIME_COL])
    if not blocks:
        return None

    block_start, block_end = blocks[0]
    block_df = df[
        (df[TIME_COL] >= block_start) & (df[TIME_COL] <= block_end)
    ].copy().reset_index(drop=True)

    # Extract a SESSION_DURATION_MS window from within the block
    win_len    = SESSION_DURATION_MS + int(rng.integers(
        -SESSION_DURATION_JITTER_MS, SESSION_DURATION_JITTER_MS
    ))
    block_span = block_end - block_start

    if block_span <= win_len:
        seg = block_df
    else:
        max_offset  = block_span - win_len
        offset      = float(rng.uniform(0, max_offset))
        t_win_start = block_start + offset
        t_win_end   = t_win_start + win_len
        seg = block_df[
            (block_df[TIME_COL] >= t_win_start) &
            (block_df[TIME_COL] <= t_win_end)
        ].copy().reset_index(drop=True)

    if len(seg) < 30:
        return None

    # Resample to TARGET_FPS
    diffs = seg[TIME_COL].diff().dropna()
    diffs = diffs[(diffs > 0) & (diffs < 500)]
    if len(diffs) == 0:
        return None
    current_fps = 1000.0 / diffs.median()
    keep_every  = max(1, round(current_fps / TARGET_FPS))
    seg = seg.iloc[::keep_every].reset_index(drop=True)
    if len(seg) < 20:
        return None

    # Drop webcam-unavailable columns
    seg = seg.drop(
        columns=[c for c in DROP_COLS if c in seg.columns], errors="ignore"
    )

    n = len(seg)

    # Gaussian spatial noise
    for col in GAZE_X_COLS:
        if col in seg.columns:
            seg[col] = pd.to_numeric(seg[col], errors="coerce")
            mask = seg[col].notna() & (seg[col] > 0)
            seg.loc[mask, col] += rng.normal(0, NOISE_STD_X, int(mask.sum()))

    for col in GAZE_Y_COLS:
        if col in seg.columns:
            seg[col] = pd.to_numeric(seg[col], errors="coerce")
            mask = seg[col].notna() & (seg[col] > 0)
            seg.loc[mask, col] += rng.normal(0, NOISE_STD_Y, int(mask.sum()))

    # Bounded shared drift (both eyes move with head)
    drift_x = np.clip(
        np.cumsum(rng.normal(0, DRIFT_STEP_X, n)), -DRIFT_CLIP, DRIFT_CLIP
    )
    drift_y = np.clip(
        np.cumsum(rng.normal(0, DRIFT_STEP_Y, n)), -DRIFT_CLIP, DRIFT_CLIP
    )
    for col in GAZE_X_COLS:
        if col in seg.columns:
            seg[col] = seg[col] + drift_x
    for col in GAZE_Y_COLS:
        if col in seg.columns:
            seg[col] = seg[col] + drift_y

    # Bursty dropout
    burst_active    = False
    burst_remaining = 0
    dropout_mask    = np.zeros(n, dtype=bool)

    for i in range(n):
        if burst_active:
            dropout_mask[i] = True
            burst_remaining -= 1
            if burst_remaining <= 0:
                burst_active = False
        elif rng.random() < BURST_PROB:
            burst_active    = True
            burst_remaining = min(
                int(rng.exponential(BURST_MEAN_FRAMES)), BURST_MAX_FRAMES
            )
            dropout_mask[i] = True

    all_gaze_cols = [c for c in GAZE_X_COLS + GAZE_Y_COLS if c in seg.columns]
    seg.loc[dropout_mask, all_gaze_cols] = np.nan
    if TRACK_COL in seg.columns:
        seg[TRACK_COL] = pd.to_numeric(seg[TRACK_COL], errors="coerce")
        seg.loc[dropout_mask, TRACK_COL] = 0

    # Forward-fill short gaps
    seg[all_gaze_cols] = seg[all_gaze_cols].ffill(limit=FORWARD_FILL_LIMIT)

    # EMA smoothing — webcam gaze estimators output smoothed coordinates,
    # not raw noisy frame values. Without this, 55px noise at 16fps gives
    # 880 px/s spurious velocities on every frame. Alpha=0.25 matches
    # the smoothing level measured in the Child1 webcam output.
    EMA_ALPHA = 0.25
    for col in all_gaze_cols:
        if col in seg.columns:
            vals = seg[col].copy().astype(float)
            smoothed = vals.ewm(alpha=EMA_ALPHA, adjust=False).mean()
            seg[col] = np.where(vals.notna(), smoothed, np.nan)

    # Compress and shift X distribution (reduce horizontal spread)
    for col in GAZE_X_COLS:
        if col not in seg.columns:
            continue
        vals  = pd.to_numeric(seg[col], errors="coerce").copy()
        valid = vals.notna() & (vals > 0)
        if valid.sum() < 5:
            continue

        # Compress spread around screen centre
        vals[valid] = SCREEN_W/2 + (vals[valid] - SCREEN_W/2) * X_SCALE
        seg[col] = vals

    # Compress and shift Y distribution
    for col in GAZE_Y_COLS:
        if col not in seg.columns:
            continue
        vals  = pd.to_numeric(seg[col], errors="coerce").copy()
        valid = vals.notna() & (vals > 0)
        if valid.sum() < 5:
            continue

        col_mean = float(vals[valid].mean())
        # Compress spread
        vals[valid] = col_mean + (vals[valid] - col_mean) * Y_SCALE
        # Shift mean to target
        shift = Y_CENTER_TARGET - float(vals[valid].mean())
        vals[valid] = vals[valid] + shift
        # Boost if Y std still too low (input was already compressed)
        current_std = float(vals[valid].std())
        if current_std < Y_STD_MIN_TARGET:
            boost = float(np.sqrt(
                max(0, Y_STD_MIN_TARGET**2 - current_std**2)
            )) + Y_STD_NOISE_BOOST * 0.4
            vals[valid] = vals[valid] + rng.normal(0, boost, int(valid.sum()))

        seg[col] = vals

    # Clamp to screen bounds
    for col in GAZE_X_COLS:
        if col in seg.columns:
            seg[col] = seg[col].clip(0, SCREEN_W)
    for col in GAZE_Y_COLS:
        if col in seg.columns:
            seg[col] = seg[col].clip(0, SCREEN_H)

    return seg


# ─────────────────────────────────────────────────────────────────────────────
# FINGERPRINT
# ─────────────────────────────────────────────────────────────────────────────

def fingerprint(df: pd.DataFrame) -> dict:
    t  = pd.to_numeric(df[TIME_COL], errors="coerce")
    xcol = "Point of Regard Right X [px]"
    ycol = "Point of Regard Right Y [px]"
    x = pd.to_numeric(df.get(xcol, pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(df.get(ycol, pd.Series(dtype=float)), errors="coerce")

    # Valid samples: gaze present and within screen (ignore tracking ratio)
    valid = x.notna() & (x > 0) & y.notna() & (y > 0)
    xv = x[valid].values
    yv = y[valid].values
    tv = t[valid].values

    t_dedup = t.dropna().drop_duplicates().sort_values()
    diffs = np.diff(t_dedup.values)
    diffs = diffs[(diffs > 0) & (diffs < 500)]
    fps   = float(1000.0 / np.median(diffs)) if len(diffs) > 0 else 0

    sacc_vel = 0.0
    if len(tv) > 10:
        dt  = np.clip(np.diff(tv) / 1000.0, 0.001, 0.5)
        vel = np.sqrt(np.diff(xv)**2 + np.diff(yv)**2) / dt
        s   = vel > 300
        sacc_vel = float(np.mean(vel[s])) if s.any() else 0.0

    dur_s = float((t.max() - t.min()) / 1000) if len(t) > 1 else 0

    return {
        "tracking_%":    float(valid.mean() * 100),
        "fps":           fps,
        "duration_s":    dur_s,
        "X_mean":        float(np.mean(xv)) if len(xv) > 0 else 0,
        "X_std":         float(np.std(xv))  if len(xv) > 0 else 0,
        "Y_mean":        float(np.mean(yv)) if len(yv) > 0 else 0,
        "Y_std":         float(np.std(yv))  if len(yv) > 0 else 0,
        "sacc_vel_px_s": sacc_vel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Degrade Cilia CSVs to webcam quality (LOOCV-ready, no split)"
    )
    parser.add_argument("--raw",  default="./raw",
                        help="Root folder with asd/ and td/ subfolders of raw Cilia CSVs")
    parser.add_argument("--out",  default="./degraded_webcam",
                        help="Output root folder")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    all_stats        = []
    participant_list = []

    for label_name, label_val in [("asd", 1), ("td", 0)]:
        in_dir  = os.path.join(args.raw, label_name)
        out_dir = os.path.join(args.out, label_name)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob(os.path.join(in_dir, "*.csv")))
        if not files:
            print(f"  Warning: no CSV files found in {in_dir}")
            continue

        print(f"\nProcessing {label_name.upper()} — {len(files)} files ...")
        skipped = 0

        for fpath in tqdm(files, desc=f"  {label_name}"):
            pid = os.path.splitext(os.path.basename(fpath))[0]
            try:
                df_raw = pd.read_csv(fpath, low_memory=False)
            except Exception as e:
                print(f"  Cannot read {pid}: {e}")
                skipped += 1
                continue

            df_deg = degrade(df_raw, rng)
            if df_deg is None or len(df_deg) < 20:
                print(f"  Skipped {pid} (unusable after processing)")
                skipped += 1
                continue

            out_path = os.path.join(out_dir, f"{pid}.csv")
            df_deg.to_csv(out_path, index=False)

            fp = fingerprint(df_deg)
            fp["participant_id"] = pid
            fp["label"]          = label_val
            fp["label_name"]     = label_name
            all_stats.append(fp)
            participant_list.append({
                "participant_id": pid,
                "label":          label_val,
                "label_name":     label_name,
            })

        written = len(files) - skipped
        print(f"  Done: {written} written to {out_dir}  ({skipped} skipped)")

    if not all_stats:
        print("No files processed. Check your --raw folder structure.")
        return

    # Save LOOCV manifest
    manifest = os.path.join(args.out, "participants.csv")
    pd.DataFrame(participant_list).to_csv(manifest, index=False)
    n_asd = sum(p["label"] == 1 for p in participant_list)
    n_td  = sum(p["label"] == 0 for p in participant_list)
    print(f"\nParticipant manifest saved -> {manifest}")
    print(f"Total: {len(participant_list)}  (ASD={n_asd}, TD={n_td})")

    # Fingerprint verification
    stats_df = pd.DataFrame(all_stats)

    targets = {
        "tracking_%":    (82,  99,   "child attention — look-aways reduce this"),
        "fps":           (13,  20,   "webcam ~16 Hz"),
        "duration_s":    (55,  130,  "webcam session length"),
        "X_std":         (90,  200,  "horizontal gaze spread"),
        "Y_std":         (30,  80,   "vertical gaze (webcam-limited range)"),
        "Y_mean":        (240, 420,  "child looks below screen centre"),
        "sacc_vel_px_s": (300, 4500, "webcam saccade velocity"),
    }

    print(f"\n{'='*70}")
    print(f"  FINGERPRINT VERIFICATION  --  child webcam targets")
    print(f"{'='*70}")
    print(f"  {'Metric':<22} {'Target':>14} {'Got':>10} {'std':>8}   {'OK?':>5}")
    print(f"  {'-'*66}")

    all_ok = True
    for metric, (lo, hi, note) in targets.items():
        if metric not in stats_df.columns:
            continue
        vals = stats_df[metric].dropna()
        got  = float(vals.mean())
        std  = float(vals.std())
        ok   = lo <= got <= hi
        flag = "OK" if ok else "OFF"
        if not ok:
            all_ok = False
        print(f"  {metric:<22} {f'{lo}-{hi}':>14} {got:>10.1f} {std:>8.1f}   {flag:>5}")
        if not ok:
            print(f"    -> {note}")

    print(f"{'='*70}")

    # ASD vs TD spatial difference
    asd_s = stats_df[stats_df["label"] == 1]
    td_s  = stats_df[stats_df["label"] == 0]
    if len(asd_s) > 0 and len(td_s) > 0:
        dx = asd_s["X_std"].mean() - td_s["X_std"].mean()
        dy = asd_s["Y_std"].mean() - td_s["Y_std"].mean()
        print(f"\n  ASD X std = {asd_s['X_std'].mean():.1f}px  "
              f"TD X std = {td_s['X_std'].mean():.1f}px  delta = {dx:+.1f}px")
        print(f"  ASD Y std = {asd_s['Y_std'].mean():.1f}px  "
              f"TD Y std = {td_s['Y_std'].mean():.1f}px  delta = {dy:+.1f}px")
        if abs(dx) > 5 or abs(dy) > 3:
            print("  ASD/TD spatial difference preserved")
        else:
            print("  Warning: ASD/TD spatial difference is small")

    print(f"\n  Next steps:")
    print(f"    python extract_features.py --degraded {args.out} "
          f"--output data/features/webcam_features.csv")
    print(f"    python train_loocv.py --features data/features/webcam_features.csv\n")

    if all_ok:
        print("  All targets met -- data ready for retraining\n")
    else:
        print("  Some targets missed -- adjust constants at top of script\n")


if __name__ == "__main__":
    main()
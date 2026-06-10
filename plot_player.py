"""
DFL Player Speed & Acceleration Plotter
========================================
Plots speed and acceleration over the duration of a match for one player.

Usage:
    python plot_player.py --output_dir <path> --player <name_or_id> [options]

Arguments:
    --output_dir   Path to the folder containing the TSV files
    --player       Player name (partial match OK, e.g. "Kimmich") or DFL person ID
    --match        Match ID (e.g. DFL-MAT-J03WMX). Defaults to first available match.
    --save         If set, saves the figure to <output_dir>/<player>_<match>.png
                   instead of opening an interactive window

Smoothing strategy:
    Speed        — 2s rolling mean (50 frames). Speed changes slowly enough that
                   a short mean cleans sensor noise without hiding real events.
    Acceleration — 3-frame median (removes single-frame sensor spikes) followed by
                   a Savitzky-Golay filter (11 frames, poly order 2). SavGol fits a
                   polynomial to each local window, which preserves peak height far
                   better than a mean filter — so real 4–7 m/s² bursts stay visible.

Examples:
    python plot_player.py --output_dir ./output --player Kimmich
    python plot_player.py --output_dir ./output --player "Thomas Müller"
    python plot_player.py --output_dir ./output --player DFL-OBJ-0002F5 --save
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
from scipy.signal import savgol_filter, medfilt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_INTERVAL_S = 0.04          # 40ms = 25 fps
SECTION_COLOURS  = {
    "firstHalf":  "#2563EB",      # blue
    "secondHalf": "#DC2626",      # red
}
RAW_ALPHA  = 0.15
SMOOTH_ALPHA = 0.9
HALFTIME_COLOUR = "#6B7280"
FIGURE_SIZE = (16, 8)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(output_dir):
    """Load all TSV files and return a dict of DataFrames."""
    files = {
        "positions": "positions.tsv",
        "players":   "players.tsv",
        "rosters":   "rosters.tsv",
        "matches":   "matches.tsv",
        "teams":     "teams.tsv",
    }
    dfs = {}
    for key, fname in files.items():
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found. Run process_dfl.py first.")
        dfs[key] = pd.read_csv(path, sep="\t")
    return dfs


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def resolve_player(players_df, query):
    """
    Return (player_id, display_name) for a name fragment or exact DFL ID.
    Exits with a helpful message on ambiguity or no match.
    """
    q = query.strip()

    # Exact ID?
    if q.startswith("DFL-OBJ-"):
        row = players_df[players_df["player_id"] == q]
        if row.empty:
            sys.exit(f"ERROR: player_id '{q}' not found in players.tsv.")
        r = row.iloc[0]
        return r["player_id"], f"{r['first_name']} {r['last_name']}"

    # Name search (case-insensitive, first+last combined)
    players_df = players_df.copy()
    players_df["full_name"] = (
        players_df["first_name"].fillna("") + " " + players_df["last_name"].fillna("")
    )
    mask = players_df["full_name"].str.contains(q, case=False, regex=False)
    matches = players_df[mask]

    if matches.empty:
        # Try last name only
        mask2 = players_df["last_name"].str.contains(q, case=False, regex=False)
        matches = players_df[mask2]

    if matches.empty:
        sys.exit(
            f"ERROR: No player found matching '{q}'.\n"
            f"Available: {players_df['full_name'].tolist()}"
        )
    if len(matches) > 1:
        names = matches["full_name"].tolist()
        sys.exit(
            f"ERROR: '{q}' is ambiguous — matched: {names}\n"
            f"Use a more specific name or the DFL player ID."
        )

    r = matches.iloc[0]
    return r["player_id"], r["full_name"].strip()


# ---------------------------------------------------------------------------
# Time axis construction
# ---------------------------------------------------------------------------

def build_time_axis(df):
    """
    Add a 'match_minute' column.
    - firstHalf  starts at 0
    - secondHalf starts at 45 (half-time gap is collapsed)
    Raw timestamps have an 18+ min gap between halves; we collapse it so
    the x-axis shows football time, not wall-clock time.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Reference: first frame of the first half
    t0 = df[df["game_section"] == "firstHalf"]["timestamp"].iloc[0]

    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()

    # Shift second half so it follows immediately after first-half ends
    fh_end_s = df[df["game_section"] == "firstHalf"]["elapsed_s"].max()
    sh_mask  = df["game_section"] == "secondHalf"
    sh_start = df.loc[sh_mask, "elapsed_s"].min()

    df.loc[sh_mask, "elapsed_s"] = (
        df.loc[sh_mask, "elapsed_s"] - sh_start + fh_end_s + FRAME_INTERVAL_S
    )

    df["match_minute"] = df["elapsed_s"] / 60.0
    return df, fh_end_s / 60.0   # also return the half-time marker position


# ---------------------------------------------------------------------------
# Smoothing — separate strategies for speed vs acceleration
# ---------------------------------------------------------------------------

SPEED_SMOOTH_S   = 2.0   # rolling mean window for speed
ACCEL_MEDFILT_N  = 3     # median filter kernel (frames) — kills single-frame spikes
ACCEL_SAVGOL_WIN = 11    # Savitzky-Golay window (frames) — preserves peak height
ACCEL_SAVGOL_ORD = 2     # polynomial order for SavGol


def smooth_speed(series):
    """Short rolling mean. Speed evolves slowly, so a 2s average is fine."""
    win = max(1, int(round(SPEED_SMOOTH_S / FRAME_INTERVAL_S)))
    return series.rolling(window=win, center=True, min_periods=1).mean()


def smooth_accel(series):
    """
    Two-stage for acceleration:
      1. Median filter (3 frames) — removes isolated sensor-noise spikes without
         pulling down neighbouring real values (unlike a mean would).
      2. Savitzky-Golay (11 frames, poly 2) — fits a local polynomial, so peaks
         keep most of their height; a plain mean would cut them in half.
    Net effect: 4–7 m/s² real bursts stay visible; single-frame glitches vanish.
    """
    arr = series.to_numpy(dtype=float)
    # Stage 1: median denoise
    arr = medfilt(arr, kernel_size=ACCEL_MEDFILT_N)
    # Stage 2: SavGol peak-preserving smooth
    arr = savgol_filter(arr, window_length=ACCEL_SAVGOL_WIN, polyorder=ACCEL_SAVGOL_ORD)
    return pd.Series(arr, index=series.index)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_player(player_id, display_name, match_id, dfs, smooth_s=None, save_path=None):
    pos = dfs["positions"]
    roster = dfs["rosters"]
    teams  = dfs["teams"]

    # Filter to this player + match
    df = pos[(pos["player_id"] == player_id) & (pos["match_id"] == match_id)].copy()
    if df.empty:
        sys.exit(f"ERROR: No position data found for {display_name} in match {match_id}.")

    df["speed"]        = pd.to_numeric(df["speed"],        errors="coerce")
    df["acceleration"] = pd.to_numeric(df["acceleration"], errors="coerce")

    df, halftime_min = build_time_axis(df)

    # Player's team for subtitle
    roster_row = roster[
        (roster["player_id"] == player_id) & (roster["match_id"] == match_id)
    ]
    team_id = roster_row["team_id"].iloc[0] if not roster_row.empty else ""
    team_name = (
        teams[teams["team_id"] == team_id]["team_name"].iloc[0]
        if team_id in teams["team_id"].values else team_id
    )
    shirt_no  = roster_row["shirt_number"].iloc[0] if not roster_row.empty else ""
    position  = roster_row["playing_position"].iloc[0] if not roster_row.empty else ""

    match_info = dfs["matches"][dfs["matches"]["match_id"] == match_id].iloc[0]
    match_title = f"{match_info['home_team_name']} {match_info['result']} {match_info['guest_team_name']}"
    match_date  = pd.to_datetime(match_info["kickoff_time"]).strftime("%d %b %Y")

    # ---- figure layout ------------------------------------------------
    fig, (ax_speed, ax_accel) = plt.subplots(
        2, 1, figsize=FIGURE_SIZE, sharex=True,
        gridspec_kw={"hspace": 0.08}
    )
    fig.patch.set_facecolor("#F9FAFB")

    title_line1 = f"#{shirt_no}  {display_name}"
    title_line2 = f"{team_name}  ·  {position}  ·  {match_title}  ·  {match_date}"
    fig.suptitle(title_line1, fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.93, title_line2, ha="center", fontsize=10, color="#6B7280")

    for section, colour in SECTION_COLOURS.items():
        part = df[df["game_section"] == section]
        if part.empty:
            continue
        t = part["match_minute"]

        for ax, col, smoother, label in [
            (ax_speed,  "speed",        smooth_speed,  "Speed"),
            (ax_accel,  "acceleration", smooth_accel,  "Acceleration"),
        ]:
            raw    = part[col]
            smooth = smoother(raw)

            ax.plot(t, raw,    color=colour, alpha=RAW_ALPHA,   linewidth=0.5)
            ax.plot(t, smooth, color=colour, alpha=SMOOTH_ALPHA, linewidth=1.2,
                    label=section.replace("Half", " Half"))

    # Half-time marker
    for ax in (ax_speed, ax_accel):
        ax.axvline(halftime_min, color=HALFTIME_COLOUR, linewidth=1.2,
                   linestyle="--", alpha=0.7, label="Half time")
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.grid(axis="x", color="#F3F4F6", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    # Y labels
    ax_speed.set_ylabel("Speed (m/s)", fontsize=10)
    ax_accel.set_ylabel("Acceleration (m/s²)", fontsize=10)
    ax_accel.set_xlabel("Match time (minutes)", fontsize=10)

    # Acceleration has negative values — add a zero line
    ax_accel.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="-")

    # X-axis: every 5 minutes
    ax_accel.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_accel.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Legend (deduplicate entries)
    handles, labels = [], []
    seen = set()
    for ax in (ax_speed, ax_accel):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); labels.append(l); seen.add(l)

    # Add raw-signal swatch to legend
    raw_patch    = mpatches.Patch(color="#9CA3AF", alpha=0.4,
                                   label="Raw signal (25 Hz)")
    smooth_patch = mpatches.Patch(color="#374151",
                                   label="Speed: 2s mean  |  Accel: median + Savitzky-Golay")
    handles += [raw_patch, smooth_patch]
    labels  += [raw_patch.get_label(), smooth_patch.get_label()]

    ax_speed.legend(handles, labels, fontsize=9, loc="upper right",
                    framealpha=0.9, edgecolor="#E5E7EB")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot speed & acceleration for a DFL player over a match."
    )
    p.add_argument("--output_dir", required=True,
                   help="Folder containing the TSV output files")
    p.add_argument("--player", required=True,
                   help="Player name (partial) or DFL-OBJ-... ID")
    p.add_argument("--match", default=None,
                   help="Match ID (defaults to first available match)")
    p.add_argument("--save", action="store_true",
                   help="Save PNG instead of showing interactive window")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading TSV files…")
    dfs = load_data(args.output_dir)

    player_id, display_name = resolve_player(dfs["players"], args.player)
    print(f"Player resolved: {display_name}  ({player_id})")

    if args.match:
        match_id = args.match
    else:
        match_id = (
            dfs["positions"][dfs["positions"]["player_id"] == player_id]
            ["match_id"].iloc[0]
        )
    print(f"Match: {match_id}")

    save_path = None
    if args.save:
        safe_name = display_name.replace(" ", "_")
        fname = f"{safe_name}_{match_id}.png"
        save_path = os.path.join(args.output_dir, fname)

    plot_player(player_id, display_name, match_id, dfs, save_path=save_path)


if __name__ == "__main__":
    main()
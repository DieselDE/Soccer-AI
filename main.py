"""
DFL Match Data Processor
========================
Reads DFL XML files (match info + position tracking) and writes to TSV databases.

Output files:
  players.tsv       — one row per unique player across all matches
  teams.tsv         — one row per unique team
  matches.tsv       — one row per match
  rosters.tsv       — player ↔ match ↔ team junction (replaces match_players)
  positions.tsv     — frame-level position data (large!)

Usage:
  python process_dfl.py <data_dir> [output_dir]

  <data_dir>   folder containing DFL XML files (or a single match's folder).
               The script auto-discovers files by their DFL naming prefix.
  [output_dir] where to write TSVs (default: ./output)

File naming conventions expected:
  DFL_02_01_matchinformation_*.xml   → match info + rosters
  DFL_04_03_positions_raw_*.xml      → position tracking frames
"""

import os
import sys
import csv
import glob
import xml.etree.ElementTree as ET
from itertools import groupby


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_file_pairs(data_dir):
    """
    Discover (match_info_path, positions_path) pairs inside data_dir.
    Walks subdirectories too, so you can point at a root folder containing
    many per-match sub-folders.
    """
    info_pattern = os.path.join(data_dir, "**", "DFL_02_01_matchinformation_*.xml")
    pos_pattern  = os.path.join(data_dir, "**", "DFL_04_03_positions_raw_*.xml")

    info_files = {extract_match_id(p): p for p in glob.glob(info_pattern, recursive=True)}
    pos_files  = {extract_match_id(p): p for p in glob.glob(pos_pattern,  recursive=True)}

    pairs = []
    for match_id in sorted(info_files):
        if match_id in pos_files:
            pairs.append((match_id, info_files[match_id], pos_files[match_id]))
        else:
            print(f"  [WARN] No positions file for match {match_id}, skipping positions.")
            pairs.append((match_id, info_files[match_id], None))

    for match_id in sorted(pos_files):
        if match_id not in info_files:
            print(f"  [WARN] No match info file for match {match_id}, skipping.")

    return pairs


def extract_match_id(path):
    """Pull the DFL-MAT-XXXXXXX token from a filename."""
    base = os.path.basename(path)
    for part in base.replace(".xml", "").split("_"):
        if part.startswith("DFL-MAT-"):
            return part
    return base  # fallback: use full basename


# ---------------------------------------------------------------------------
# Match info parser
# ---------------------------------------------------------------------------

def parse_match_info(xml_path):
    """
    Returns:
        match_row   dict  — one row for matches.tsv
        team_rows   list  — rows for teams.tsv  (may duplicate across matches)
        player_rows list  — rows for players.tsv (may duplicate across matches)
        roster_rows list  — rows for rosters.tsv
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mi = root.find("MatchInformation")
    gen = mi.find("General")
    env = mi.find("Environment")

    match_row = {
        "match_id":        gen.get("MatchId"),
        "competition_id":  gen.get("CompetitionId"),
        "competition_name":gen.get("CompetitionName"),
        "season":          gen.get("Season"),
        "match_day":       gen.get("MatchDay"),
        "home_team_id":    gen.get("HomeTeamId"),
        "guest_team_id":   gen.get("GuestTeamId"),
        "home_team_name":  gen.get("HomeTeamName"),
        "guest_team_name": gen.get("GuestTeamName"),
        "kickoff_time":    gen.get("KickoffTime"),
        "result":          gen.get("Result"),
        "stadium_name":    env.get("StadiumName"),
        "pitch_x":         env.get("PitchX"),
        "pitch_y":         env.get("PitchY"),
    }

    team_rows   = []
    player_rows = []
    roster_rows = []

    for team_el in mi.find("Teams").findall("Team"):
        team_id   = team_el.get("TeamId")
        team_name = team_el.get("TeamName")
        team_rows.append({"team_id": team_id, "team_name": team_name})

        for p in team_el.find("Players").findall("Player"):
            person_id = p.get("PersonId")
            player_rows.append({
                "player_id":       person_id,
                "first_name":      p.get("FirstName"),
                "last_name":       p.get("LastName"),
                "short_name":      p.get("Shortname"),
            })
            roster_rows.append({
                "match_id":         match_row["match_id"],
                "player_id":        person_id,
                "team_id":          team_id,
                "shirt_number":     p.get("ShirtNumber"),
                "starting":         p.get("Starting"),
                "playing_position": p.get("PlayingPosition", ""),
                "team_leader":      p.get("TeamLeader"),
            })

    return match_row, team_rows, player_rows, roster_rows


# ---------------------------------------------------------------------------
# Positions parser  (streaming — file can be 100s of MB)
# ---------------------------------------------------------------------------

def stream_positions(xml_path, match_id):
    """
    Generator that yields one dict per Frame row.
    Uses iterparse so the whole file is never in memory at once.
    Each FrameSet supplies context (game_section, team_id, person_id)
    for all its child Frame elements.
    """
    current = {}  # context from most-recent FrameSet open tag

    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current = {
                "game_section": elem.get("GameSection"),
                "team_id":      elem.get("TeamId"),
                "player_id":    elem.get("PersonId"),
            }

        elif event == "end" and elem.tag == "Frame":
            yield {
                "match_id":     match_id,
                "player_id":    current.get("player_id", ""),
                "team_id":      current.get("team_id", ""),
                "game_section": current.get("game_section", ""),
                "frame_n":      elem.get("N"),
                "timestamp":    elem.get("T"),
                "x":            elem.get("X"),
                "y":            elem.get("Y"),
                "speed":        elem.get("S"),
                "distance":     elem.get("D"),
                "acceleration": elem.get("A"),
                "status":       elem.get("M"),
            }
            elem.clear()  # free memory — critical for large files

        elif event == "end" and elem.tag == "FrameSet":
            elem.clear()


# ---------------------------------------------------------------------------
# TSV writers
# ---------------------------------------------------------------------------

PLAYERS_COLS  = ["player_id", "first_name", "last_name", "short_name"]
TEAMS_COLS    = ["team_id", "team_name"]
MATCHES_COLS  = ["match_id", "competition_id", "competition_name", "season",
                 "match_day", "home_team_id", "guest_team_id",
                 "home_team_name", "guest_team_name",
                 "kickoff_time", "result", "stadium_name", "pitch_x", "pitch_y"]
ROSTERS_COLS  = ["match_id", "player_id", "team_id", "shirt_number",
                 "starting", "playing_position", "team_leader"]
POSITIONS_COLS = ["match_id", "player_id", "team_id", "game_section",
                  "frame_n", "timestamp", "x", "y",
                  "speed", "distance", "acceleration", "status"]


def open_tsv(path, columns):
    """Open a TSV file for writing, write the header, return (file, writer)."""
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=columns, delimiter="\t",
                       extrasaction="ignore")
    w.writeheader()
    return f, w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_dfl.py <data_dir> [output_dir]")
        sys.exit(1)

    data_dir   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning for DFL match files in: {data_dir}")
    pairs = find_file_pairs(data_dir)
    if not pairs:
        print("No matching file pairs found. Check that your folder contains "
              "DFL_02_01_matchinformation_*.xml files.")
        sys.exit(1)
    print(f"Found {len(pairs)} match(es).\n")

    # Open all output TSVs
    pf,  pw  = open_tsv(os.path.join(output_dir, "players.tsv"),   PLAYERS_COLS)
    tf,  tw  = open_tsv(os.path.join(output_dir, "teams.tsv"),     TEAMS_COLS)
    mf,  mw  = open_tsv(os.path.join(output_dir, "matches.tsv"),   MATCHES_COLS)
    rf,  rw  = open_tsv(os.path.join(output_dir, "rosters.tsv"),   ROSTERS_COLS)
    xf,  xw  = open_tsv(os.path.join(output_dir, "positions.tsv"), POSITIONS_COLS)

    seen_players = set()
    seen_teams   = set()

    try:
        for match_id, info_path, pos_path in pairs:

            # --- match info ---
            print(f"[{match_id}] Parsing match info…")
            match_row, team_rows, player_rows, roster_rows = parse_match_info(info_path)

            mw.writerow(match_row)

            for t in team_rows:
                if t["team_id"] not in seen_teams:
                    tw.writerow(t)
                    seen_teams.add(t["team_id"])

            for p in player_rows:
                if p["player_id"] not in seen_players:
                    pw.writerow(p)
                    seen_players.add(p["player_id"])

            rw.writerows(roster_rows)

            # --- positions ---
            if pos_path:
                print(f"[{match_id}] Streaming positions (this may take a moment)…")
                count = 0
                for row in stream_positions(pos_path, match_id):
                    xw.writerow(row)
                    count += 1
                    if count % 500_000 == 0:
                        print(f"           … {count:,} frames written")
                print(f"[{match_id}] Done — {count:,} position frames written.")

    finally:
        for f in (pf, tf, mf, rf, xf):
            f.close()

    print(f"\nAll done. TSV files written to: {output_dir}")
    print("  players.tsv   — unique players")
    print("  teams.tsv     — unique teams")
    print("  matches.tsv   — match metadata")
    print("  rosters.tsv   — player/match/team membership")
    print("  positions.tsv — frame-level position tracking")


if __name__ == "__main__":
    main()
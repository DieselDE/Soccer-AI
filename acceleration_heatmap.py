"""
DFL Acceleration Heatmap
========================
Reads the positions.tsv produced by main.py and generates an HTML heatmap
showing where a specific player had high acceleration during a match.

Usage:
    python acceleration_heatmap.py <positions_tsv> <player_id> [output_html]

    <positions_tsv>  path to positions.tsv
    <player_id>      the PersonId of the player to visualise
    [output_html]    output file (default: heatmap_<player_id>.html)

The pitch dimensions are taken from the data itself (max X/Y values found
for the match). If you want explicit dimensions, pass --pitch-x and --pitch-y.

Requirements:
    pip install pandas numpy
    No other dependencies — the heatmap is rendered entirely in the browser
    using a self-contained HTML/JS file (Canvas 2D API, no external libs).

How it works:
    1. Load only the rows for the requested player_id from positions.tsv.
    2. Convert X/Y coordinates (DFL uses metres, origin at pitch centre) to
       pixel coordinates on a canvas.
    3. Bin acceleration values into a 2-D grid.
    4. Render each cell as a filled rectangle with an opacity proportional to
       the 95th-percentile-normalised mean acceleration in that cell.
    5. Overlay a pitch outline (centre circle, penalty areas, halfway line).
"""

import sys
import argparse
import csv
import json
import math
import os


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_player_frames(tsv_path: str, player_id: str) -> list[dict]:
    """Stream only the rows belonging to player_id. Returns list of dicts."""
    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["player_id"] != player_id:
                continue
            try:
                x   = float(row["x"])
                y   = float(row["y"])
                acc = float(row["acceleration"]) if row["acceleration"] else 0.0
            except ValueError:
                continue
            rows.append({"x": x, "y": y, "acc": acc, "frame": row["frame_n"]})
    return rows


# ---------------------------------------------------------------------------
# Grid aggregation
# ---------------------------------------------------------------------------

def build_grid(frames: list[dict],
               pitch_x: float, pitch_y: float,
               n_cols: int = 60, n_rows: int = 40) -> dict:
    """
    Aggregate acceleration into an (n_rows × n_cols) grid.

    DFL coordinates: origin at pitch centre, X runs along length, Y along width.
    Both axes range roughly from -pitch_x/2 to +pitch_x/2 etc.

    Returns a dict ready to JSON-serialise and hand off to the HTML renderer.
    """
    half_x = pitch_x / 2.0
    half_y = pitch_y / 2.0

    cell_w = pitch_x / n_cols
    cell_h = pitch_y / n_rows

    # Sum and count per cell
    total  = [[0.0] * n_cols for _ in range(n_rows)]
    counts = [[0]   * n_cols for _ in range(n_rows)]

    for f in frames:
        col = int((f["x"] + half_x) / cell_w)
        row = int((f["y"] + half_y) / cell_h)
        col = max(0, min(n_cols - 1, col))
        row = max(0, min(n_rows - 1, row))
        total[row][col]  += f["acc"]
        counts[row][col] += 1

    # Mean acceleration per cell (0 where never visited)
    mean_grid = [
        [total[r][c] / counts[r][c] if counts[r][c] > 0 else 0.0
         for c in range(n_cols)]
        for r in range(n_rows)
    ]

    # 95th-percentile normalisation so outlier sprints don't wash everything out
    flat = [v for row in mean_grid for v in row if v > 0]
    if flat:
        flat_sorted = sorted(flat)
        p95_idx = max(0, int(len(flat_sorted) * 0.95) - 1)
        p95 = flat_sorted[p95_idx]
    else:
        p95 = 1.0

    norm_grid = [
        [min(mean_grid[r][c] / p95, 1.0) for c in range(n_cols)]
        for r in range(n_rows)
    ]

    return {
        "grid":    norm_grid,
        "counts":  counts,
        "n_rows":  n_rows,
        "n_cols":  n_cols,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "p95_acc": round(p95, 4),
    }


# ---------------------------------------------------------------------------
# Infer pitch dimensions from data
# ---------------------------------------------------------------------------

def infer_pitch_dims(frames: list[dict]) -> tuple[float, float]:
    xs = [f["x"] for f in frames]
    ys = [f["y"] for f in frames]
    # Add small buffer so border cells aren't clipped
    pitch_x = (max(xs) - min(xs)) * 1.02
    pitch_y = (max(ys) - min(ys)) * 1.02
    return round(pitch_x, 1), round(pitch_y, 1)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Acceleration Heatmap — {player_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 32px 16px 48px;
    min-height: 100vh;
  }}

  header {{
    text-align: center;
    margin-bottom: 28px;
  }}
  header h1 {{
    font-size: clamp(1.2rem, 3vw, 1.7rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #e6edf3;
  }}
  header p {{
    margin-top: 6px;
    font-size: 0.82rem;
    color: #8b949e;
  }}

  #canvas-wrap {{
    position: relative;
    width: 100%;
    max-width: 820px;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 0 40px rgba(0,0,0,0.6);
  }}

  /* colour legend */
  #legend {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 16px;
    font-size: 0.75rem;
    color: #8b949e;
  }}
  #legend-bar {{
    width: 180px;
    height: 10px;
    border-radius: 3px;
    background: linear-gradient(to right,
      rgba(0,120,255,0)  0%,
      rgba(0,180,255,.4) 30%,
      rgba(80,220,140,.7) 60%,
      rgba(255,220,0,.85) 80%,
      rgba(255,60,0,1)   100%);
  }}

  #stats {{
    display: flex;
    gap: 28px;
    margin-top: 20px;
    flex-wrap: wrap;
    justify-content: center;
  }}
  .stat {{
    text-align: center;
  }}
  .stat .val {{
    font-size: 1.4rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: -0.02em;
  }}
  .stat .lbl {{
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  #tooltip {{
    position: absolute;
    background: rgba(13,17,23,0.92);
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 7px 11px;
    font-size: 0.75rem;
    pointer-events: none;
    display: none;
    line-height: 1.6;
    color: #e6edf3;
    white-space: nowrap;
  }}
</style>
</head>
<body>

<header>
  <h1>Acceleration Heatmap</h1>
  <p>Player <strong style="color:#58a6ff">{player_id}</strong>
     &nbsp;·&nbsp; {n_frames} tracking frames
     &nbsp;·&nbsp; pitch {pitch_x} × {pitch_y} m</p>
</header>

<div id="canvas-wrap">
  <canvas id="heatmap"></canvas>
  <div id="tooltip"></div>
</div>

<div id="legend">
  <span>Low</span>
  <div id="legend-bar"></div>
  <span>High (&ge; {p95_acc} m/s²)</span>
</div>

<div id="stats">
  <div class="stat"><div class="val" id="s-frames">—</div><div class="lbl">Frames</div></div>
  <div class="stat"><div class="val" id="s-max">—</div><div class="lbl">Peak accel (m/s²)</div></div>
  <div class="stat"><div class="val" id="s-mean">—</div><div class="lbl">Mean accel (m/s²)</div></div>
  <div class="stat"><div class="val" id="s-cells">—</div><div class="lbl">Cells visited</div></div>
</div>

<script>
const DATA = {json_data};

const canvas  = document.getElementById("heatmap");
const tooltip = document.getElementById("tooltip");
const wrap    = document.getElementById("canvas-wrap");

// Canvas intrinsic resolution: keep aspect ratio of pitch
const ASPECT  = DATA.pitch_x / DATA.pitch_y;
const C_H     = 480;
const C_W     = Math.round(C_H * ASPECT);
canvas.width  = C_W;
canvas.height = C_H;

const ctx   = canvas.getContext("2d");
const nRows = DATA.n_rows;
const nCols = DATA.n_cols;
const cw    = C_W / nCols;
const ch    = C_H / nRows;

// ---- pitch background ----
ctx.fillStyle = "#1a4731";
ctx.fillRect(0, 0, C_W, C_H);

// ---- draw grass stripes ----
const stripeW = C_W / 10;
for (let i = 0; i < 10; i++) {{
  ctx.fillStyle = i % 2 === 0 ? "rgba(0,0,0,0.07)" : "rgba(255,255,255,0.04)";
  ctx.fillRect(i * stripeW, 0, stripeW, C_H);
}}

// ---- heatmap cells ----
// colour stops: transparent → blue → cyan → green → yellow → red
function heatColour(t) {{
  // t in [0,1]
  const stops = [
    [0,   [0,   0,   0,   0  ]],
    [0.1, [0,   100, 255, 0.2]],
    [0.35,[0,   200, 255, 0.45]],
    [0.6, [80,  220, 120, 0.65]],
    [0.8, [255, 210, 0,   0.82]],
    [1.0, [255,  50, 0,   0.95]],
  ];
  for (let i = 1; i < stops.length; i++) {{
    const [t0, c0] = stops[i-1];
    const [t1, c1] = stops[i];
    if (t <= t1) {{
      const f = (t - t0) / (t1 - t0);
      const r = Math.round(c0[0] + f*(c1[0]-c0[0]));
      const g = Math.round(c0[1] + f*(c1[1]-c0[1]));
      const b = Math.round(c0[2] + f*(c1[2]-c0[2]));
      const a =            c0[3] + f*(c1[3]-c0[3]);
      return `rgba(${{r}},${{g}},${{b}},${{a.toFixed(3)}})`;
    }}
  }}
  return "rgba(255,50,0,0.95)";
}}

for (let r = 0; r < nRows; r++) {{
  for (let c = 0; c < nCols; c++) {{
    const v = DATA.grid[r][c];
    if (v <= 0) continue;
    // DFL: Y increases downward from centre in canvas coords when we flip
    const px = c * cw;
    const py = (nRows - 1 - r) * ch;   // flip Y so "up pitch" = top of canvas
    ctx.fillStyle = heatColour(v);
    ctx.fillRect(px, py, cw + 0.5, ch + 0.5);  // +0.5 avoids sub-pixel gaps
  }}
}}

// ---- pitch markings ----
ctx.strokeStyle = "rgba(255,255,255,0.55)";
ctx.lineWidth   = 1.5;

function pitchX(mX) {{ return (mX + DATA.pitch_x/2) / DATA.pitch_x * C_W; }}
function pitchY(mY) {{ return C_H - (mY + DATA.pitch_y/2) / DATA.pitch_y * C_H; }}

function rect(x, y, w, h) {{
  ctx.strokeRect(pitchX(x), pitchY(y+h), pitchX(x+w)-pitchX(x), pitchY(y)-pitchY(y+h));
}}
function circle(cx, cy, r) {{
  ctx.beginPath();
  const px = pitchX(cx), py = pitchY(cy);
  const rPx = r / DATA.pitch_x * C_W;
  ctx.arc(px, py, rPx, 0, 2*Math.PI);
  ctx.stroke();
}}

// outer boundary
rect(-DATA.pitch_x/2, -DATA.pitch_y/2, DATA.pitch_x, DATA.pitch_y);

// halfway line
ctx.beginPath();
ctx.moveTo(C_W/2, 0); ctx.lineTo(C_W/2, C_H); ctx.stroke();

// centre circle (r = 9.15 m standard)
circle(0, 0, 9.15);

// penalty areas (standard: 40.32 × 16.5 m)
const paW = 40.32, paD = 16.5;
rect(-DATA.pitch_x/2,          -paW/2, paD,         paW);
rect( DATA.pitch_x/2 - paD,    -paW/2, paD,         paW);

// goal areas (18.32 × 5.5 m)
const gaW = 18.32, gaD = 5.5;
rect(-DATA.pitch_x/2,          -gaW/2, gaD,         gaW);
rect( DATA.pitch_x/2 - gaD,    -gaW/2, gaD,         gaW);

// ---- stats ----
const frames = DATA.raw_frames;
let sumAcc = 0, maxAcc = 0, cellsVisited = 0;
const counts = DATA.counts;
for (let r = 0; r < nRows; r++)
  for (let c = 0; c < nCols; c++)
    if (counts[r][c] > 0) cellsVisited++;

frames.forEach(f => {{ sumAcc += f.acc; if (f.acc > maxAcc) maxAcc = f.acc; }});
document.getElementById("s-frames").textContent = frames.length.toLocaleString();
document.getElementById("s-max").textContent    = maxAcc.toFixed(2);
document.getElementById("s-mean").textContent   = (sumAcc / frames.length).toFixed(2);
document.getElementById("s-cells").textContent  = cellsVisited;

// ---- tooltip on hover ----
canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const scaleX = C_W / rect.width;
  const scaleY = C_H / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top)  * scaleY;

  const col = Math.floor(px / cw);
  const row = nRows - 1 - Math.floor(py / ch);
  if (col < 0 || col >= nCols || row < 0 || row >= nRows) {{ tooltip.style.display="none"; return; }}

  const val     = DATA.grid[row][col];
  const cnt     = counts[row][col];
  const mX      = ((col + 0.5) / nCols - 0.5) * DATA.pitch_x;
  const mY      = ((row + 0.5) / nRows - 0.5) * DATA.pitch_y;

  tooltip.style.display = "block";
  tooltip.style.left    = (e.clientX - wrap.getBoundingClientRect().left + 12) + "px";
  tooltip.style.top     = (e.clientY - wrap.getBoundingClientRect().top  - 10) + "px";
  tooltip.innerHTML =
    `<b>x</b> ${{mX.toFixed(1)}} m &nbsp; <b>y</b> ${{mY.toFixed(1)}} m<br>` +
    `Rel. accel: ${{(val*100).toFixed(0)}}%<br>` +
    `Frames: ${{cnt}}`;
}});
canvas.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DFL Acceleration Heatmap")
    parser.add_argument("positions_tsv", help="Path to positions.tsv")
    parser.add_argument("player_id",     help="DFL PersonId of the player")
    parser.add_argument("output_html",   nargs="?", help="Output HTML file")
    parser.add_argument("--pitch-x", type=float, default=None,
                        help="Pitch length in metres (auto-detected if omitted)")
    parser.add_argument("--pitch-y", type=float, default=None,
                        help="Pitch width in metres (auto-detected if omitted)")
    parser.add_argument("--cols", type=int, default=60,
                        help="Heatmap grid columns (default 60)")
    parser.add_argument("--rows", type=int, default=40,
                        help="Heatmap grid rows (default 40)")
    args = parser.parse_args()

    out_path = args.output_html or f"heatmap_{args.player_id}.html"

    print(f"Loading frames for player {args.player_id} …")
    frames = load_player_frames(args.positions_tsv, args.player_id)
    if not frames:
        print(f"ERROR: No frames found for player_id '{args.player_id}'.")
        print("Check that the player_id matches exactly what's in the TSV.")
        sys.exit(1)
    print(f"  {len(frames):,} frames loaded.")

    pitch_x = args.pitch_x
    pitch_y = args.pitch_y
    if pitch_x is None or pitch_y is None:
        px, py = infer_pitch_dims(frames)
        pitch_x = pitch_x or px
        pitch_y = pitch_y or py
        print(f"  Inferred pitch dimensions: {pitch_x} × {pitch_y} m")

    print("Building acceleration grid …")
    grid_data = build_grid(frames, pitch_x, pitch_y,
                           n_cols=args.cols, n_rows=args.rows)

    # Include raw frames in the JSON so the browser can compute stats.
    # For very large datasets, cap at 200k frames to keep HTML size sane.
    MAX_RAW = 200_000
    raw_sample = frames if len(frames) <= MAX_RAW else frames[::len(frames)//MAX_RAW + 1]
    grid_data["raw_frames"] = raw_sample

    json_str = json.dumps(grid_data, separators=(",", ":"))

    html = HTML_TEMPLATE.format(
        player_id = args.player_id,
        n_frames  = f"{len(frames):,}",
        pitch_x   = pitch_x,
        pitch_y   = pitch_y,
        p95_acc   = grid_data["p95_acc"],
        json_str  = "",   # unused placeholder
        json_data = json_str,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHeatmap written to: {out_path}")
    print("Open it in any browser — no server needed.")


if __name__ == "__main__":
    main()

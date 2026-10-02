#!/usr/bin/env python3
"""Cross-machine comparison chart for bench/gpu/results/*.csv.

Reads every committed result file (one per machine/backend-build, see
bench_gpu.sh's "one per machine" convention -- e.g.
``NVIDIA-H100-NVL__INTEL-XEON-PLATINUM-8558.csv``,
``none__Apple-M4.csv``) and renders a single interactive Plotly HTML page
for the libKriging backends only (GPyTorch rows are dropped -- this chart
is about comparing libKriging's own backends across machines, not
libKriging vs. GPyTorch, which already has its own comparison in
`docs/comparisons/`). Machine is the x-axis -- every backend that ran on a
machine sits at that machine's x position, e.g. `iter-omp` and `chol` on
the same CPU-only host both plot at that host's tick -- with color
carrying the backend/engine (a categorical identity: fixed hue order, see
color-formula.md) and marker shape carrying the training-set size `n` (an
ORDERED quantity). Two legends show both encodings: a static backend
color key, and an `n` shape key whose clicks isolate one `n` at a time
across every machine and backend -- the "groupable" behavior asked for. A
dropdown switches the timing column (fit/logLik/predict), log-scale
y-axis (these span 3+ orders of magnitude).

Standalone: no dependency on the rest of libKriging beyond `plotly`. Run by
hand (`python bench/gpu/plot_comparison.py`) or from CI
(.github/workflows/bench-gpu-report.yml) after any push touching
bench/gpu/results/*.csv.
"""

import argparse
import csv
import glob
import os

import plotly.graph_objects as go

# --- dataviz skill's reference palette (references/palette.md) -----------
# backend/engine is a CATEGORY (identity, no order), so it gets the fixed
# 8-hue categorical order -- assigned once per key below, never re-ranked
# by what happens to be present in a given results/ snapshot (anti-
# patterns.md: "color follows the entity, never its rank").
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
# n IS an ORDERED quantity (250 < 500 < ... < 32000), so it carries marker
# SHAPE only here (color is spoken for by backend) -- color-formula.md's
# categorical-vs-ordinal rule.
MARKER_SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "star", "hexagon", "cross", "x", "pentagon",
]

METRICS = [
    ("loglik_s", "logLik (s)"),
    ("fit_s", "fit (s)"),
    ("predict_s", "predict (s)"),
]

BACKEND_LABELS = {
    "chol": "libKriging-Cholesky",
    "iter-cuda": "libKriging-Iterative-CUDA",
    "iter-hip": "libKriging-Iterative-HIP",
    "iter-metal": "libKriging-Iterative-Metal",
    "iter-omp": "libKriging-Iterative-OpenMP",
}
# Fixed categorical-color assignment, in priority order -- the same
# backend always gets the same color regardless of which subset of
# machines/backends a given results/ snapshot happens to contain.
BACKEND_PRIORITY = ["iter-cuda", "iter-hip", "iter-metal", "iter-omp", "chol"]
BACKEND_COLOR = {bk: CATEGORICAL[i % len(CATEGORICAL)] for i, bk in enumerate(BACKEND_PRIORITY)}
GPYTORCH_PREFIX = "gpt"


def machine_label(csv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    # "<gpu>__<cpu>[__variant]" -> "<gpu> / <cpu> [(variant)]"; "none__X"
    # (no GPU, e.g. a CPU-only or Metal build) -> just "X".
    parts = stem.split("__")
    parts = [p for p in parts if p and p != "none"]
    if len(parts) >= 3:
        return f"{parts[0]} / {parts[1]} ({', '.join(parts[2:])})"
    if len(parts) == 2:
        return f"{parts[0]} / {parts[1]}"
    return parts[0] if parts else stem


def load_rows(results_dir: str):
    """One dict per (machine, backend_key, n) row with status=='ok'."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        machine = machine_label(path)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                if row.get("backend_key", "").startswith(GPYTORCH_PREFIX):
                    continue
                try:
                    n = int(row["n"])
                    metrics = {key: float(row[key]) for key, _ in METRICS}
                except (KeyError, ValueError):
                    continue
                rows.append(dict(machine=machine, backend_key=row["backend_key"], n=n, **metrics))
    return rows


def build_figure(rows):
    ns = sorted({r["n"] for r in rows})

    def backend_rank(bk):
        return (BACKEND_PRIORITY.index(bk) if bk in BACKEND_PRIORITY else 99, bk)

    machines = sorted({r["machine"] for r in rows})
    backend_keys = sorted({r["backend_key"] for r in rows}, key=backend_rank)
    symbol_of = {n: MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)] for i, n in enumerate(ns)}
    color_of = {bk: BACKEND_COLOR.get(bk, "#898781") for bk in backend_keys}

    # One data trace per (backend, n), spanning every machine on the
    # x-axis -- color carries backend (categorical identity), shape carries
    # n (ordered). Both are real, groupable encodings, but only one trace
    # per pair can own a legend entry without flooding it (backends x n
    # could be 40+ entries) -- so these stay out of the legend
    # (showlegend=False) and share a legendgroup with the *shape* proxy
    # below, which is what makes "isolate one n" toggle them all at once.
    traces = []
    for bk in backend_keys:
        for n in ns:
            xs, ys_by_metric = [], {key: [] for key, _ in METRICS}
            for m in machines:
                match = next((r for r in rows if r["machine"] == m and r["backend_key"] == bk and r["n"] == n), None)
                if match is None:
                    continue
                xs.append(m)
                for key, _ in METRICS:
                    ys_by_metric[key].append(match[key])
            if not xs:
                continue
            traces.append(go.Scatter(
                x=xs,
                y=ys_by_metric[METRICS[0][0]],
                mode="markers",
                name=f"{BACKEND_LABELS.get(bk, bk)} / n={n}",
                legendgroup=f"n={n}",
                showlegend=False,
                marker=dict(color=color_of[bk], symbol=symbol_of[n], size=11, line=dict(width=1, color="#ffffff")),
                hovertemplate="%{x}<br>" + BACKEND_LABELS.get(bk, bk) + "<br>n=" + str(n)
                              + "<br>%{y:.3g}s<extra></extra>",
                meta=dict(ys=ys_by_metric),
            ))

    # --- legend 1: backend color key (a static color identity, not an
    # interactive filter -- the one groupable/isolatable axis is n, below;
    # see the module docstring) ------------------------------------------
    for bk in backend_keys:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers", name=BACKEND_LABELS.get(bk, bk),
            marker=dict(color=color_of[bk], symbol="circle", size=11, line=dict(width=1, color="#ffffff")),
            showlegend=True, hoverinfo="skip",
        ))

    # --- legend 2: n shape key (click isolates one n, across every
    # machine and backend) ----------------------------------------------
    for n in ns:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers", name=f"n={n}",
            legendgroup=f"n={n}",
            marker=dict(color="#898781", symbol=symbol_of[n], size=11, line=dict(width=1, color="#ffffff")),
            showlegend=True, hoverinfo="skip", legend="legend2",
        ))

    fig = go.Figure(data=traces)

    # --- metric dropdown: restyles every trace's `y` (proxy legend
    # entries have no `meta` / real y, so leave them untouched) ----------
    metric_buttons = []
    for key, label in METRICS:
        ys = [t.meta["ys"][key] if t.meta else t.y for t in fig.data]
        metric_buttons.append(dict(
            label=label,
            method="restyle",
            args=[{"y": ys}, {"yaxis.title.text": label}],
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color="#0b0b0b"),
        title=dict(text="libKriging iterative backend -- cross-machine comparison", x=0.02, xanchor="left"),
        xaxis=dict(title="machine", categoryorder="array", categoryarray=machines,
                   tickangle=-20, gridcolor="#e1e0d9", linecolor="#c3c2b7"),
        yaxis=dict(title=METRICS[0][1], type="log", gridcolor="#e1e0d9", linecolor="#c3c2b7"),
        legend=dict(title="backend (click to toggle)", bgcolor="rgba(0,0,0,0)", x=1.02, y=1.0, yanchor="top"),
        legend2=dict(title="n (click to isolate)", bgcolor="rgba(0,0,0,0)", x=1.02, y=0.55, yanchor="top"),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        margin=dict(t=90, b=120, r=180),
        updatemenus=[
            dict(buttons=metric_buttons, direction="down", x=0.0, xanchor="left", y=1.15, yanchor="top",
                showactive=True, pad=dict(r=8, t=4)),
        ],
        annotations=[
            dict(text="metric:", x=0.0, xanchor="left", y=1.20, yanchor="bottom", yref="paper", xref="paper",
                showarrow=False, font=dict(size=12, color="#52514e")),
        ],
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--out", default=None, help="output HTML path (default: <results-dir>/comparison.html)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.results_dir, "comparison.html")

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"no 'ok' rows found under {args.results_dir}/*.csv")
    fig = build_figure(rows)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out} ({len(rows)} rows from "
          f"{len(set(r['machine'] for r in rows))} machine(s), "
          f"{len(set(r['backend_key'] for r in rows))} backend(s))")


if __name__ == "__main__":
    main()

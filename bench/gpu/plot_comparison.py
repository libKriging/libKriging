#!/usr/bin/env python3
"""Cross-machine comparison chart for bench/gpu/results/*.csv.

Reads every committed result file (one per machine/backend-build, see
bench_gpu.sh's "one per machine" convention -- e.g.
``NVIDIA-H100-NVL__INTEL-XEON-PLATINUM-8558.csv``,
``none__Apple-M4.csv``) and renders a single interactive Plotly HTML page:
every (host, backend/engine) combination on the x-axis -- machine and
backend are NOT split across a dropdown, they are both part of the same
axis category, so all host-method-engine combos are visible together -- a
chosen timing column on the y-axis (log scale -- these span 3+ orders of
magnitude), one trace per training-set size `n` (color + marker shape both
carry `n`, since it is an ORDERED quantity, not an arbitrary category --
see the dataviz skill's color-formula.md). A dropdown switches the timing
column (fit/logLik/predict); legend clicks (single = toggle, double =
isolate) group/isolate one `n` at a time across every combo, which is the
"groupable" behavior asked for.

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
# n is an ORDERED quantity (250 < 500 < ... < 32000), not an arbitrary
# category, so it takes the sequential single-hue ramp (blue, light->dark),
# not eight categorical hues -- see color-formula.md's categorical-vs-
# ordinal rule. Steps 250..700 (skipping the too-light 100/150/200 steps,
# which the ordinal-ramp rule reserves for the near-surface "zero" end of a
# true sequential/heatmap encoding).
SEQUENTIAL_BLUE = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
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
    "gpt-cuda": "GPyTorch-BBMM-CUDA",
    "gpt-cpu": "GPyTorch-BBMM-CPU",
    "gpt-chol-cuda": "GPyTorch-Cholesky-CUDA",
    "gpt-chol-cpu": "GPyTorch-Cholesky-CPU",
}
# Ordering of the backend/engine part of each x-axis combo -- the backends
# this session's work (and most bench/gpu runs) actually cares about
# comparing come first.
BACKEND_PRIORITY = ["iter-cuda", "iter-hip", "iter-metal", "iter-omp", "chol"]


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

    # One x-axis category per (machine, backend_key) combo -- host and
    # engine are NOT split across a dropdown, both live on the same axis so
    # every host-method-engine combination is visible at once. Grouped by
    # machine first (so a machine's backends sit together), backend priority
    # second.
    combos = sorted(
        {(r["machine"], r["backend_key"]) for r in rows},
        key=lambda mb: (mb[0], backend_rank(mb[1])),
    )
    combo_label = {
        (m, bk): f"{m} · {BACKEND_LABELS.get(bk, bk)}" for m, bk in combos
    }
    xs_all = [combo_label[c] for c in combos]

    # One trace per `n`, spanning every combo on the x-axis. Color/symbol
    # depend only on `n` (color-formula.md: "color follows the entity, never
    # its rank"), so there is nothing left to repaint when the metric changes.
    traces = []
    for i, n in enumerate(ns):
        xs, ys_by_metric = [], {key: [] for key, _ in METRICS}
        for m, bk in combos:
            match = next((r for r in rows if r["machine"] == m and r["backend_key"] == bk and r["n"] == n), None)
            if match is None:
                continue
            xs.append(combo_label[(m, bk)])
            for key, _ in METRICS:
                ys_by_metric[key].append(match[key])
        if not xs:
            continue
        color = SEQUENTIAL_BLUE[i % len(SEQUENTIAL_BLUE)]
        symbol = MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)]
        traces.append(go.Scatter(
            x=xs,
            y=ys_by_metric[METRICS[0][0]],
            mode="markers",
            name=f"n={n}",
            legendgroup=f"n={n}",
            marker=dict(color=color, symbol=symbol, size=11, line=dict(width=1, color="#ffffff")),
            hovertemplate="%{x}<br>n=" + str(n) + "<br>%{y:.3g}s<extra></extra>",
            meta=dict(n=n, ys=ys_by_metric),
        ))

    fig = go.Figure(data=traces)

    # --- metric dropdown: restyles every trace's `y` -----------------
    metric_buttons = []
    for key, label in METRICS:
        metric_buttons.append(dict(
            label=label,
            method="restyle",
            args=[{"y": [t.meta["ys"][key] for t in traces]},
                  {"yaxis.title.text": label}],
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color="#0b0b0b"),
        title=dict(text="libKriging iterative backend -- cross-machine comparison", x=0.02, xanchor="left"),
        xaxis=dict(title="host · method/engine", categoryorder="array", categoryarray=xs_all,
                   tickangle=-20, gridcolor="#e1e0d9", linecolor="#c3c2b7"),
        yaxis=dict(title=METRICS[0][1], type="log", gridcolor="#e1e0d9", linecolor="#c3c2b7"),
        legend=dict(title="n (click to isolate)", bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        margin=dict(t=90, b=160),
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

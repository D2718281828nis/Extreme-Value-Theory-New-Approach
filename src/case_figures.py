"""Six-step figures for the A (aviation), B (brain), C (finance) cases — identical panel types.

    python src/case_figures.py                 # all cases  -> figures/
    python src/case_figures.py --only A B      # a subset

Outputs (figures/):
    stepN_<name>.png          one step, three subplots A | B | C (visual comparison across domains)
    <case>_six_steps.png      one case, all six steps as 2x3 subplots
    all_steps_grid.png        6 rows (steps) x 3 columns (cases)
    <case>_summary.json       the numbers quoted in examples/<case>/README.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

import case_data as D  # noqa: E402

OUT = D.ROOT / "figures"
RED, ORANGE, INK, GREY = "#c0392b", "#e08a00", "#222222", "#9aa0a6"
EVID = ["#4c78a8", "#59a14f", "#b07aa1"]            # evidence lines: cascade, footprint, model
STEPS = [
    ("1", "task", "1 · Task: system signal and the extreme event"),
    ("2", "data", "2 · Data: node responses around the event"),
    ("3", "graph", "3 · Graph-EVT-agent: system graph"),
    ("4", "when", "4 · WHEN: EVT threshold and event point"),
    ("5", "where", "5 · WHERE: source evidence by group"),
    ("6", "result", "6 · Result: onset cascade by group"),
]
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 160,
                     "axes.titleweight": "bold", "legend.frameon": False})


def _dates(c, ax, ticks, view=None):
    """Replace trading-day indices by calendar dates for the finance case."""
    if c.key == "C":
        idx = c._dates
        ticks = [int(t) for t in ticks if 0 <= t < len(idx)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(idx[t].date())[2:] for t in ticks], rotation=0, fontsize=7)


def _xlabel(c):
    return {"cycle": "engine cycle", "s": "time, s", "trading day": "date (yy-mm-dd)"}[c.time_unit]


def _ref(ax, c, ymax=None, label=True):
    ax.axvline(c.event_t, color=INK, ls="--", lw=1)
    ax.text(c.event_t, 1.01, c.event_label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7, color=INK)


# ----------------------------------------------------------------------------- panels
def p_task(ax, c):
    ax.plot(c.sys_t, c.sys_y, color="#4c78a8", lw=0.7)
    ax.axvspan(c.view[0], c.view[1], color=ORANGE, alpha=0.12, lw=0)
    _ref(ax, c)
    ax.set_ylabel(c.sys_label)
    ax.set_xlabel(_xlabel(c))
    if c.key == "C":
        _dates(c, ax, np.linspace(0, len(c._dates) - 1, 5))
    ax.set_xlim(c.sys_t[0], c.sys_t[-1])
    ax.text(0.01, 0.97, "task: when does the extreme event start,\nand which node/group is its source?",
            transform=ax.transAxes, va="top", fontsize=7.5, color=INK)


def p_data(ax, c):
    im = ax.imshow(c.heat, aspect="auto", cmap="magma", vmin=0, vmax=np.nanpercentile(c.heat, 99),
                   extent=[c.heat_t[0], c.heat_t[-1], c.heat.shape[0], 0], interpolation="nearest")
    ax.axvline(c.event_t, color="white", ls="--", lw=1)
    ax.set_ylabel(f"{c.heat.shape[0]} nodes (grouped)")
    ax.set_xlabel(_xlabel(c))
    ax.set_xlim(c.heat_t[0], c.heat_t[-1])
    if c.key == "C":
        _dates(c, ax, np.linspace(c.heat_t[0], c.heat_t[-1], 4))
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("z vs baseline", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.spines["left"].set_visible(False)


def p_graph(ax, c):
    G = nx.Graph()
    G.add_nodes_from(range(len(c.nodes)))
    G.add_edges_from(c.edges)
    pos = nx.spring_layout(G, seed=3, k=1.6 / np.sqrt(len(c.nodes)), iterations=150)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.4, alpha=0.25, edge_color=GREY)
    # small disconnected components (e.g. airlines in C): emphasise edges and name the members
    for comp in nx.connected_components(G):
        if len(comp) < 10 < len(c.nodes):
            nx.draw_networkx_edges(G.subgraph(comp), pos, ax=ax, width=1.0, alpha=0.8, edge_color=GREY)
            pts = np.array([pos[i] for i in comp])
            grp = ", ".join(sorted({str(c.groups[i]) for i in comp}))
            ax.annotate(f"closed cluster, no links to the rest:\n{', '.join(c.nodes[i] for i in sorted(comp))} ({grp})",
                        pts.mean(0), xytext=(14, -4), textcoords="offset points", fontsize=6.5, va="top", color=INK)
    resp = np.log1p(np.nan_to_num(c.node_peak))
    src = np.asarray(c.groups) == c.source
    sz = 14 + 6 * resp ** 2 if len(c.nodes) < 50 else 6 + 3 * resp ** 2
    sc = ax.scatter(*np.array([pos[i] for i in G.nodes]).T, c=resp, s=sz, cmap="viridis", zorder=3, lw=0)
    ax.scatter(*np.array([pos[i] for i in np.where(src)[0]]).T, s=sz[src] * 1.9 if hasattr(sz, "__len__") else sz * 1.9,
               facecolors="none", edgecolors=RED, linewidths=1.2, zorder=4)
    if len(c.nodes) <= 20:
        for i in G.nodes:
            ax.annotate(c.nodes[i], pos[i], fontsize=6.5, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    ax.set_title(ax.get_title(), pad=4)
    ax.text(0.01, 0.01, f"{len(c.nodes)} nodes · {len(c.edges)} edges\n{c.graph_label}\nred ring = source group «{c.source}»",
            transform=ax.transAxes, fontsize=7, va="bottom")
    ax.axis("off")


def p_when(ax, c):
    lo, hi = c.view if c.key != "A" else (c.ind_t[0], c.ind_t[-1])
    if c.key == "B":
        lo, hi = 10100, 10500
    m = (c.ind_t >= lo) & (c.ind_t <= hi)
    ax.plot(c.ind_t[m], c.ind[m], color="#4c78a8", lw=0.8)
    ax.axhline(c.thr, color=ORANGE, ls="--", lw=1.1)
    ax.text(hi, c.thr, f"threshold ({c.tail}) ", color=ORANGE, va="bottom", ha="right", fontsize=7)
    ax.axvline(c.det_t, color=RED, lw=1.4)
    _ref(ax, c)
    ax.scatter([c.det_t], [c.ind[int(np.where(c.ind_t == c.det_t)[0][0])]], color=RED, zorder=5, s=18)
    d = c.det_t - c.event_t
    unit = {"cycle": "cycles", "s": "s", "trading day": "trading days"}[c.time_unit]
    rel = f"{abs(d):.0f}" if c.key != "B" else f"{d:+.1f}"
    txt = {"A": f"detected at cycle {c.det_t:.0f}:\n{abs(d):.0f} {unit} before failure",
           "B": f"detected at {c.det_t:.0f} s:\n{rel} {unit} vs clinical annotation",
           "C": f"detected on day t_det:\n{abs(d):.0f} {unit.rstrip('s')}(s) before crash peak"}[c.key]
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7.5, color=RED)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("EVT indicator")
    ax.set_xlabel(_xlabel(c))
    if c.key == "C":
        _dates(c, ax, np.linspace(lo, hi, 4))
    top = np.nanmax(c.ind[m])
    ax.set_ylim(top=max(top, c.thr) * 1.1)


def p_where(ax, c):
    ev = c.evidence.copy()
    cols = list(ev.columns)
    agree = sum(ev[k].idxmax() == c.source for k in cols)
    ev["_m"] = ev[cols[-1]].fillna(-1) + 1e-3 * ev[cols[:-1]].mean(axis=1, skipna=True)
    ev = ev.sort_values("_m", ascending=False).head(7).iloc[::-1]
    h = 0.8 / len(cols)
    y = np.arange(len(ev))
    for i, k in enumerate(cols):
        ax.barh(y + (i - (len(cols) - 1) / 2) * h, ev[k].fillna(0), height=h * 0.92, color=EVID[i], label=k.replace(" (", "\n("),
                alpha=0.95)
        for yy, v in zip(y + (i - (len(cols) - 1) / 2) * h, ev[k]):
            if np.isnan(v):
                ax.text(0.01, yy, "n/a", fontsize=6, va="center", color=GREY)
    for yy, row in zip(y, ev[cols].values):  # min-max floor: lowest group on the line, not missing data
        if (row == 0).all():
            ax.text(0.01, yy, "lowest on all lines (0)", fontsize=7, va="center", color=GREY)
        else:
            for k, v in enumerate(row):
                if v == 0:
                    ax.text(0.01, yy + (k - (len(cols) - 1) / 2) * h, "0", fontsize=6, va="center", color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  ◀ mean rule" if g == c.source_mean else str(g) for g in ev.index], fontsize=7.5)
    for lab, g in zip(ax.get_yticklabels(), ev.index):
        if g == c.source:
            lab.set_color(RED)
            lab.set_fontweight("bold")
        elif g == c.source_mean:
            lab.set_color(ORANGE)
            lab.set_fontweight("bold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"evidence (min-max over groups) · top-1 agreement {agree}/{len(cols)} lines")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.2), handlelength=1, ncol=1)


def p_result(ax, c):
    on = np.asarray(c.onset, float)
    grp = np.asarray(c.groups)
    names = [g for g in dict.fromkeys(grp) if np.isfinite(on[grp == g]).any()]
    med = {g: np.nanmedian(on[grp == g]) for g in names}
    names = sorted(names, key=lambda g: np.nanmin(on[grp == g]))[:8][::-1]
    rng = np.random.default_rng(0)
    for k, g in enumerate(names):
        v = on[grp == g]
        v = v[np.isfinite(v)]
        col = RED if g == c.source else "#4c78a8"
        ax.scatter(v, k + rng.uniform(-0.22, 0.22, len(v)), s=10, color=col, alpha=0.75, lw=0)
        ax.plot([v.min(), v.min()], [k - 0.35, k + 0.35], color=col, lw=2)
    ax.axvline(0, color=RED, lw=1.2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{g} (n={int(np.isfinite(on[grp == g]).sum())})" for g in names], fontsize=7.5)
    for lab in ax.get_yticklabels():
        if lab.get_text().startswith(c.source + " ("):
            lab.set_color(RED)
            lab.set_fontweight("bold")
    unit = {"cycle": "cycles", "s": "s", "trading day": "days"}[c.time_unit]
    verdict = "same" if c.source_mean == c.source else f"DIFFERENT («{c.source_mean}»)"
    ax.set_xlabel(f"node onset − detected time, {unit}\nGraph-EVT-agent source: «{c.source}» · plain mean rule: {verdict}", color=INK)


PANELS = {"task": p_task, "data": p_data, "graph": p_graph, "when": p_when, "where": p_where, "result": p_result}


# ----------------------------------------------------------------------------- figure assembly
def step_comparison(cases, no, name, title):
    fig, axs = plt.subplots(1, len(cases), figsize=(5.0 * len(cases), 3.9))
    for ax, c in zip(np.atleast_1d(axs), cases):
        PANELS[name](ax, c)
        ax.set_title(c.title if name != "graph" else c.title, fontsize=8.5, loc="left", pad=14 if name in ("task", "when") else 6)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / f"step{no}_{name}.png", bbox_inches="tight")
    plt.close(fig)


def case_six_steps(c):
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for ax, (no, name, title) in zip(axs.ravel(), STEPS):
        PANELS[name](ax, c)
        ax.set_title(title, fontsize=9, loc="left", pad=14 if name in ("task", "when") else 6)
    fig.suptitle(c.title, fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / f"{c.key}_six_steps.png", bbox_inches="tight")
    plt.close(fig)


def grid(cases):
    fig, axs = plt.subplots(len(STEPS), len(cases), figsize=(5.0 * len(cases), 3.3 * len(STEPS)))
    for r, (no, name, title) in enumerate(STEPS):
        for k, c in enumerate(cases):
            ax = axs[r, k]
            PANELS[name](ax, c)
            ax.set_title((c.title.split(" — ")[0] + "   |   " if r == 0 else "") + title if k == 0 else
                         (c.title.split(" — ")[0] if r == 0 else ""), fontsize=8.5, loc="left", pad=14 if name in ("task", "when") else 6)
    fig.tight_layout()
    fig.savefig(OUT / "all_steps_grid.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(D.BUILDERS), default=list(D.BUILDERS))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cases = []
    for key in a.only:
        c = D.BUILDERS[key]()
        cases.append(c)
        case_six_steps(c)
        s = dict(c.summary)
        s["evidence"] = c.evidence.round(3).to_dict()
        (OUT / f"{key}_summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False, default=str))
        print(key, "ok", c.source)
    for no, name, title in STEPS:
        step_comparison(cases, no, name, title)
    if len(cases) > 1:
        grid(cases)


if __name__ == "__main__":
    main()

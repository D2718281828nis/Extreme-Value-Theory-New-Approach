"""One illustrative case per application, in a common structure.

Every builder returns a `Case`: the same fields for aviation (A), brain (B), finance (C) and ECG (D),
so that `case_figures.py` can draw exactly the same six panel types for each of them.

    system   : system-level signal over the whole record (step 1, the task)
    heat     : node x time standardised response around the event (step 2, the data)
    graph    : nodes, groups, edges (step 3, Graph-EVT-agent application)
    indicator: EVT indicator, tail threshold, detected event time (step 4, when)
    evidence : per-group source evidence, normalised to [0, 1] (step 5, where)
    onset    : per-node onset times relative to the detected event (step 6, the cascade)

The heavy domain pipelines (src/*_pipeline.py) stay the source of the research numbers; builders here
re-use their functions on a single engine / episode and read the frozen tables in reference_results/.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import common as C  # noqa: E402

REF = ROOT / "reference_results"
DATA = ROOT / "data"


@dataclass
class Case:
    key: str                      # "A" | "B" | "C" | "D"
    title: str
    time_unit: str                # "cycle" | "s" | "trading day"
    # step 1
    sys_t: np.ndarray = None
    sys_y: np.ndarray = None
    sys_label: str = ""
    event_t: float = None         # reference event time (same axis as sys_t)
    event_label: str = ""
    view: tuple = (None, None)    # window shown in steps 2, 4, 6
    # step 2
    heat: np.ndarray = None       # nodes x time (display window)
    heat_t: np.ndarray = None
    # step 3
    nodes: list = field(default_factory=list)
    groups: np.ndarray = None
    edges: list = field(default_factory=list)
    node_peak: np.ndarray = None  # peak standardised response per node (node size)
    graph_label: str = ""
    graph_variants: dict = None  # ECG: physiological / functional / kNN edge lists
    graph_corr: np.ndarray = None  # ECG: absolute baseline correlation between four derived series
    # step 4
    ind_t: np.ndarray = None
    ind: np.ndarray = None
    thr: float = None
    det_t: float = None
    tail: str = ""
    # step 5
    evidence: pd.DataFrame = None  # index = group, columns = evidence lines in [0, 1]; NaN if not available
    source: str = ""            # Graph-EVT-agent: top of the model-based (graph) evidence line
    source_mean: str = ""       # ordinary-statistics baseline: top of the plain mean across all lines
    # step 6
    onset: np.ndarray = None      # per node or per episode-node point, relative to det_t
    node_onset: np.ndarray = None  # ECG: representative-record onsets used only for graph rings
    summary: dict = field(default_factory=dict)


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def _consensus(ev: pd.DataFrame) -> str:
    """Source = top group of the model-based evidence line (last column); the EVT lines are consistency checks."""
    return str(ev.iloc[:, -1].idxmax())


def _mean_rule(ev: pd.DataFrame) -> str:
    """Ordinary-statistics baseline: plain mean of the evidence lines (missing lines skipped)."""
    return str(ev.mean(axis=1, skipna=True).idxmax())


# ============================================================================ A. aviation
def aviation(unit: int = 7) -> Case:
    import aviation_cmapss_pipeline as A

    d3, _ = A.load(str(DATA / "CMAPSSData"), "FD003")
    g = d3[d3.unit == unit]
    X = g[A.S].values.astype(float)
    n = len(X)
    base_mu, base_sd = X[:A.B].mean(0), X[:A.B].std(0) + 1e-9
    Z = (X - base_mu) / base_sd

    det, hi = A.detect_engine(g)
    thr_gev, _ = C.gev_threshold(hi[: n // A.BL * A.BL].reshape(-1, A.BL).max(1)[: A.B // A.BL], 0.99)
    det_t = det["det_gev"]

    # per-sensor onset: first persistent GEV exceedance of |z| (block maxima), as in the pipeline
    onset_abs = np.full(len(A.S), np.nan)
    for j in range(len(A.S)):
        zz = np.abs(Z[:, j])
        bm = zz[: n // A.BL * A.BL].reshape(-1, A.BL).max(1)
        th, _ = C.gev_threshold(bm[: A.B // A.BL], 0.99)
        k = C.first_persistent(bm > th, A.B // A.BL, 3)
        if k is not None:
            onset_abs[j] = k * A.BL

    mods = np.array([A.MOD[s] for s in A.S])
    names = [A.NAME[s] for s in A.S]
    ev = pd.DataFrame(index=sorted(set(mods)))
    first = pd.Series(onset_abs, index=mods).groupby(level=0).min()
    ev["EVT cascade (earliest onset)"] = _minmax(-first.reindex(ev.index).fillna(first.max() + 1))
    end_z = pd.Series(np.abs(Z[-20:].mean(0)), index=mods).groupby(level=0).mean()
    ev["Event footprint (mean |z| at end)"] = _minmax(end_z.reindex(ev.index))
    L = pd.read_csv(REF / "aviation" / "localization_FD003.csv").set_index("unit").loc[unit]
    gnn = pd.Series({m: L[f"occl_{m.lower()}_total"] for m in ev.index})
    ev["GNN occlusion (all modules)"] = _minmax(gnn)

    lo = max(0, det_t - 3 * A.BL * 10)
    c = Case("A", f"A · Aviation — turbofan engine (C-MAPSS FD003, unit {unit})", "cycle")
    c.sys_t, c.sys_y = np.arange(1, n + 1), np.abs(Z).mean(1)
    c.sys_label = "mean |z| over 14 sensors"
    c.event_t, c.event_label = n, "failure"
    c.view = (1, n)
    c.heat, c.heat_t = Z.T, np.arange(1, n + 1)
    c.nodes, c.groups = names, mods
    c.edges = A.physical_edges()
    c.node_peak = np.abs(Z).max(0)
    c.graph_label = "physical gas-path graph"
    c.ind_t, c.ind, c.thr, c.det_t = np.arange(1, n + 1), hi, thr_gev, det_t
    c.tail = "GEV on block maxima, q=0.99"
    c.evidence, c.source, c.source_mean = ev, _consensus(ev), _mean_rule(ev)
    c.onset = onset_abs - det_t
    c.summary = {"unit": unit, "life_cycles": n, "detected_cycle": int(det_t), "lead_cycles": int(n - det_t),
                 "gev_threshold": float(thr_gev),
                 "signature_source": str(L.source_sig), "cascade_source": str(L.cascade_source),
                 "gnn_source": str(L.gnn_source), "agent_source": c.source, "mean_rule_source": c.source_mean,
                 "earliest_sensors": [names[i] for i in np.argsort(np.where(np.isnan(onset_abs), 1e9, onset_abs))[:3]]}
    return c


# ============================================================================ B. brain
def brain() -> Case:
    ONSET = 10396.445
    casc = np.load(REF / "brain" / "cascade.npz")
    ind = np.load(REF / "brain" / "indicators.npz")
    loc = pd.read_csv(REF / "brain" / "localization_contacts.csv")
    shafts = pd.read_csv(REF / "brain" / "localization_shafts.csv").set_index("shaft")
    feat = pd.read_csv(REF / "brain" / "node_features.csv")
    z, tw, lat = casc["z"], casc["tw"], casc["lat"]
    hi, thr = ind["hi"], float(ind["thr_pot"])
    B = 3600
    k = C.first_persistent(hi > thr, B, 3)
    det_t = float(k)

    names, shaft = loc.name.tolist(), loc.shaft.values
    num = np.array([int("".join(ch for ch in nm if ch.isdigit())) for nm in names])
    E = []
    for s in np.unique(shaft):
        ids = sorted(np.where(shaft == s)[0], key=lambda i: num[i])
        E += [tuple(sorted((ids[a], ids[a + 1]))) for a in range(len(ids) - 1)]
    cols = [c_ for c_ in feat.columns if c_ not in ("name", "shaft", "involved")]
    E = sorted(set(E) | set(C.knn_edges(feat[cols].values.astype(float), 3)))

    ev = pd.DataFrame(index=shafts.index)
    ev["EVT cascade (earliest onset)"] = _minmax(-shafts.latency_min.fillna(shafts.latency_min.max() + 5))
    ev["Event footprint (share > EVT thr)"] = _minmax(shafts.footprint)
    ev["GAT node score (grouped CV)"] = _minmax(shafts.gnn)

    lo, hi_ = 10300, 10500
    sel = (tw >= lo) & (tw <= hi_)
    order = np.argsort(shaft, kind="stable")
    c = Case("B", "B · Brain — sEEG seizure (Zenodo sEEG-HFOs-8, 100 contacts)", "s")
    c.sys_t = tw
    c.sys_y = np.clip(z, 0, 30).mean(0)
    c.sys_label = "mean clipped z, 13–67 Hz power"
    c.event_t, c.event_label = ONSET, "clinical onset"
    c.view = (lo, hi_)
    c.heat, c.heat_t = np.clip(z[order][:, sel], -3, 30), tw[sel]
    c.nodes, c.groups = names, shaft
    c.edges = E
    c.node_peak = np.nanmax(z, axis=1)
    c.graph_label = "shaft chains + feature-kNN"
    tt = np.arange(len(hi))
    c.ind_t, c.ind, c.thr, c.det_t = tt, hi, thr, det_t
    c.tail = "POT/GPD, q=0.999"
    c.evidence, c.source, c.source_mean = ev, _consensus(ev), _mean_rule(ev)
    c.onset = ONSET + lat - det_t
    c.summary = {"detected_s": det_t, "annotation_s": ONSET, "dt_detect_minus_annotation_s": det_t - ONSET,
                 "pot_threshold": thr, "n_contacts": len(names), "n_involved": int(np.isfinite(lat).sum()),
                 "agent_source": c.source, "mean_rule_source": c.source_mean, "earliest_contacts": loc.dropna(subset=["latency"]).nsmallest(5, "latency")
                 .apply(lambda r: f"{r['name']} ({r['latency']:+.1f}s)", axis=1).tolist(),
                 "leading_shaft_by_evidence": {"cascade": str(shafts.n_early10.idxmax()), "footprint": str(shafts.footprint.idxmax()),
                                               "gat": str(shafts.gnn.idxmax()), "residual_z": str(shafts.residual_z.idxmax())}}
    return c


# ============================================================================ C. finance
def finance(episode: str = "2015-08") -> Case:
    import finance_sp500_pipeline as F

    R, sec, _ = F.load(str(DATA))
    det, m = F.detect_market(R)
    EPIS = {"2015-08": ("2015-08-20", "2015-08-24"), "2016-06": ("2016-06-23", "2016-06-24"),
            "2016-09": ("2016-09-08", "2016-09-09"), "2018-02": ("2018-02-01", "2018-02-05")}
    pre_end, peak = EPIS[episode]
    idx = R.index
    L = -m
    thr_m = det["thr_pot"]
    ip = idx.get_loc(pd.Timestamp(peak))
    days = [d for d in det["extreme_days"] if abs(idx.get_loc(pd.Timestamp(d)) - ip) <= 5]
    det_day = pd.Timestamp(min(days))
    idet = idx.get_loc(det_day)

    thr = np.array([C.pot_threshold(-R.values[:F.B, j], 0.9, 0.99)[0] for j in range(R.shape[1])])
    X = R.values
    win = slice(idet - 10, idet + 4)               # onset search window, trading days
    loss = -X[win]
    first = np.full(R.shape[1], np.nan)
    for j in range(R.shape[1]):
        hit = np.where(loss[:, j] > thr[j])[0]
        if len(hit):
            first[j] = hit[0] - 10                 # relative to the detected day
    order = np.argsort(sec.values, kind="stable")
    v0, v1 = idet - 25, idet + 15
    sd = X[:F.B].std(0) + 1e-9
    heat = (-X[v0:v1] / sd).T[order]

    fe = C.functional_edges(X[idet - F.WPRE - 5: idet - 5], k=3)
    LT = pd.read_csv(REF / "finance" / "localization_sectors.csv")
    LT = LT[LT.episode == episode].set_index("sector")
    ev = pd.DataFrame(index=LT.index)
    ev["EVT cascade (pre-peak exceedances)"] = _minmax(LT.pre_onset)
    ev["Event footprint (share > EVT thr)"] = _minmax(LT.footprint)
    ev["Market-model residual z"] = _minmax(LT.residual_z)

    c = Case("C", f"C · Finance — S&P 500 stress, August {episode[:4]} ({R.shape[1]} stocks)", "trading day")
    c.sys_t, c.sys_y = np.arange(len(idx)), L.values
    c.sys_label = "equal-weight market loss"
    c.event_t, c.event_label = float(ip), "crash peak"
    c.view = (v0, v1)
    c.heat, c.heat_t = heat, np.arange(v0, v1)
    c.nodes, c.groups = list(R.columns), sec.values
    c.edges = fe
    c.node_peak = (np.abs(X[win]) / sd).max(0)
    c.graph_label = "functional graph (top-3 |corr|)"
    c.ind_t, c.ind, c.thr, c.det_t = np.arange(len(idx)), L.values, thr_m, float(idet)
    c.tail = "POT/GPD, q=0.998"
    c.evidence, c.source, c.source_mean = ev, _consensus(ev), _mean_rule(ev)
    c.onset = first
    c.summary = {"episode": episode, "detected_day": str(det_day.date()), "peak_day": peak,
                 "pot_threshold": float(thr_m), "n_stocks": int(R.shape[1]),
                 "stocks_with_onset": int(np.isfinite(first).sum()), "agent_source": c.source, "mean_rule_source": c.source_mean,
                 "top_by_evidence": {k: str(ev[k].idxmax()) for k in ev.columns},
                 "date_axis": {"0": str(idx[0].date()), "detected_index": int(idet), "dates_view": [str(idx[v0].date()), str(idx[v1 - 1].date())]}}
    c._dates = idx  # for axis labels
    return c


# ============================================================================ D. ECG
def ecg() -> Case:
    import json
    import ecg_incart_pipeline as E

    R = json.loads((REF / "ecg" / "results.json").read_text(encoding="utf-8"))
    event = R["event"]; onset = float(event["onset"]); det_t = float(R["detection"]["blind_pot"]["event_alarm"])
    ind = np.load(REF / "ecg" / "indicators.npz")
    z, tw, lat = ind["z"], ind["tw"], ind["lat"]
    index = {name: i for i, name in enumerate(E.PARAMETERS)}
    graphs = {name: [tuple(index[p] for p in edge) for edge in edges]
              for name, edges in R["graphs"]["edges"].items()}
    ef = graphs["functional"]
    corr = ind["graph_corr"]

    groups = pd.read_csv(REF / "ecg" / "localization_groups.csv").set_index("parameter")
    ev = pd.DataFrame(index=groups.index)
    latency = groups.latency_median
    fallback = latency.max() + 5 if latency.notna().any() else 5.0
    ev["EVT cascade (median onset)"] = _minmax(-latency.fillna(fallback))
    ev["Event footprint (share > EVT thr)"] = _minmax(groups.footprint)
    ev["GAT node score (record CV)"] = _minmax(groups.gnn)
    points = pd.read_csv(REF / "ecg" / "episode_latencies.csv")
    points = points[points.involved == 1]

    lo, hi = 350, 580
    sel = (tw >= lo) & (tw <= hi)
    c = Case("D", f"D · ECG — {R['n_records']} INCART recordings, {R['n_vt_episodes']} VT episodes", "s")
    series_t = np.arange(len(ind["rr"]), dtype=float)
    c.sys_t, c.sys_y = series_t, ind["rr"]
    c.sys_label = "RR interval, s (representative I42)"
    c.event_t, c.event_label = onset, "VT onset in I42"
    c.view = (lo, hi)
    c.heat, c.heat_t = np.clip(z[sel].T, 0, 15), tw[sel]
    c.nodes = E.PARAMETERS; c.groups = points.parameter.to_numpy()
    c.edges = ef
    c.graph_variants = graphs
    c.graph_corr = corr
    c.node_peak = np.nanmax(z, axis=0)
    c.graph_label = "global physiological + functional + feature-kNN; episode graphs in model CV"
    c.ind_t, c.ind = series_t, ind["hi"]
    c.thr, c.det_t = float(R["detection"]["blind_pot"]["thr"]), det_t
    c.tail = "POT/GPD, q=0.999"
    c.evidence = ev
    leading = R["localization"]["leading_parameter"]
    gnn_auroc = [row["auroc_mean"] for row in R.get("gnn_summary", [])
                 if row["split"] == "record" and row["model"] == "gat" and row["graph"] == "physiological"]
    use_gnn = leading["gnn_oof"] is not None and gnn_auroc and gnn_auroc[0] > .5
    c.source = leading["gnn_oof"] if use_gnn else (leading["cascade"] or leading["footprint"])
    c.source_mean = _mean_rule(ev)
    c.onset = points.latency.to_numpy()
    c.node_onset = onset + lat - det_t
    c.summary = {"record": "I42", "detected_s": det_t, "annotation_s": onset,
                 "dt_detect_minus_annotation_s": det_t - onset, "pot_threshold": c.thr,
                 "n_records": R["n_records"], "n_records_with_vt": R["n_records_with_eligible_vt"],
                 "n_vt_episodes": R["n_vt_episodes"], "n_result_points": len(points),
                 "n_parameters": len(E.PARAMETERS), "n_involved": R["labels"]["involved"],
                 "graph_edges": {k: len(v) for k, v in c.graph_variants.items()}, "graph_jaccard": R["graphs"]["jaccard"],
                 "agent_source": c.source, "mean_rule_source": c.source_mean,
                 "involved_by_parameter": R["labels"]["by_parameter"],
                 "qrs_reference_coverage_80ms": R["signal_qrs"]["reference_position_coverage_80ms"]}
    return c


BUILDERS = {"A": aviation, "B": brain, "C": finance, "D": ecg}

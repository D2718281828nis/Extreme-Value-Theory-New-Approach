"""Simplest Graph-EVT-agent demo: one synthetic episode, three lines of method, one figure.

    python demo/quickstart.py          # ~2 s, writes demo/quickstart.png

A pulse starts at a known channel and spreads over a ring graph. Graph-EVT-agent must find
WHEN (EVT threshold on a baseline-only fit) and WHERE (source ranking on the graph).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from graph_evt_agent import EVTConfig, GraphEVTPipeline
from graph_evt_agent.synthetic import make_propagation_episode

ep = make_propagation_episode(n_channels=6, source=2, seed=1)          # ground truth: t=420, source=ch2
result = GraphEVTPipeline(EVTConfig(baseline_size=400)).run_detailed(ep.values)
det, rank = result.detection, result.ranking
print(f"true event {ep.event_time}, detected {det.time_index}; true source ch{ep.source}, "
      f"found ch{rank.source}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(ep.values, lw=0.6); ax[0].axvline(ep.event_time, c="k", ls="--"); ax[0].set_title("1 data: 6 channels")
ax[1].plot(det.indicator, c="#4c78a8", lw=0.8); ax[1].axhline(det.alarm_threshold, c="orange", ls="--")
ax[1].axvline(det.time_index, c="crimson"); ax[1].set_title(f"2 WHEN: detected t={det.time_index}")
ax[2].bar([f"ch{i}" for i in rank.node_ids], rank.probabilities, color=["crimson"] + ["#4c78a8"] * 5)
ax[2].set_title(f"3 WHERE: source ch{rank.source} (true ch{ep.source})")
fig.tight_layout(); fig.savefig(Path(__file__).with_name("quickstart.png"), dpi=130)

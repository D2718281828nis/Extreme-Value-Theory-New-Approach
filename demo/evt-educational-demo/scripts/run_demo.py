#!/usr/bin/env python3
"""Запустить полный автономный EVT demo."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from evt_demo import AutoregressiveModel, EVTAnalyzer, generate_time_series_with_extreme_events, save_to_csv
from evt_demo.visualization import (compare_evt_vs_autoregression, plot_diagnostic_qq, plot_gev_distribution,
                                    plot_gpd_exceedances, plot_mean_excess, plot_return_levels,
                                    plot_time_series_with_extremes)
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=42); p.add_argument("--bootstrap",type=int,default=100); p.add_argument("--output-dir",type=Path,default=ROOT); a=p.parse_args()
    output=a.output_dir.resolve(); series,truth=generate_time_series_with_extreme_events(seed=a.seed); save_to_csv(series,truth,output/"data/synthetic_data.csv")
    evt=EVTAnalyzer(); gev=evt.fit_gev_block_maxima(series,100); gpd=evt.fit_gpd_peaks_over_threshold(series,threshold_percentile=95)
    detected=evt.detect_extremes_via_pot(series,gpd,tail_probability=.05); returns=evt.calculate_gev_return_levels(gev,n_bootstrap=a.bootstrap,seed=a.seed)
    ar=AutoregressiveModel(1).fit(series); ar_detected=ar.detect_anomalies_via_residuals(series)
    figdir=output/"figures"; plots=[plot_time_series_with_extremes(series,truth,detected,save_path=figdir/"extreme_event_detection.png"),plot_gev_distribution(gev,figdir/"gev_distribution.png"),plot_gpd_exceedances(gpd,figdir/"gpd_fit.png"),compare_evt_vs_autoregression(series,truth,detected,ar_detected,figdir/"evt_vs_ar_comparison.png"),plot_return_levels(returns,figdir/"return_levels.png"),plot_mean_excess(series,np.percentile(series,np.linspace(80,98,30)),figdir/"mean_excess.png")]
    diag=evt.diagnostics(gpd); plots.append(plot_diagnostic_qq(diag["empirical_quantiles"],diag["theoretical_quantiles"],figdir/"gpd_qq.png"))
    import matplotlib.pyplot as plt
    for fig,_ in plots: plt.close(fig)
    rows=[]
    for name,mask in [("EVT/POT",detected),("AR residual",ar_detected)]: rows.append({"method":name,"precision":precision_score(truth,mask,zero_division=0),"recall":recall_score(truth,mask,zero_division=0),"f1":f1_score(truth,mask,zero_division=0)})
    print(pd.DataFrame(rows).round(3).to_string(index=False)); print("\nВозвратные уровни:\n",returns.round(3).to_string(index=False)); print(f"\nФайлы: {figdir}")
if __name__=="__main__": main()

"""Визуализации без изменения глобального стиля Matplotlib."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme, genpareto
from sklearn.metrics import precision_score, recall_score, f1_score
from .evt_analysis import GEVFit, GPDFit

def _save(fig, path):
    if path is not None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig

def plot_time_series_with_extremes(series, true_mask, detected=None, title="Временной ряд с экстремумами", save_path=None):
    x=np.asarray(series); truth=np.asarray(true_mask,bool); fig,ax=plt.subplots(figsize=(12,4)); ax.plot(x,lw=.8,label="Ряд")
    ax.scatter(np.flatnonzero(truth),x[truth],c="tab:red",label="Внедрённые",zorder=3)
    if detected is not None:
        found=np.asarray(detected,bool); ax.scatter(np.flatnonzero(found),x[found],facecolors="none",edgecolors="tab:green",label="POT",zorder=4)
    ax.set(title=title,xlabel="Время",ylabel="Значение"); ax.legend(); return _save(fig,save_path),ax

def plot_gev_distribution(fit: GEVFit, save_path=None):
    fig,ax=plt.subplots(); ax.hist(fit.block_maxima,bins="auto",density=True,alpha=.5,label="Максимумы")
    grid=np.linspace(fit.block_maxima.min(),fit.block_maxima.max(),300); ax.plot(grid,genextreme.pdf(grid,-fit.xi,loc=fit.mu,scale=fit.sigma),label="GEV")
    ax.set(xlabel="Максимум блока",ylabel="Плотность",title="Подгонка GEV"); ax.legend(); return _save(fig,save_path),ax

def plot_gpd_exceedances(fit: GPDFit, save_path=None):
    fig,ax=plt.subplots(); ax.hist(fit.exceedances,bins="auto",density=True,alpha=.5,label="Превышения")
    grid=np.linspace(0,fit.exceedances.max(),300); ax.plot(grid,genpareto.pdf(grid,fit.xi,loc=0,scale=fit.sigma_u),label="GPD")
    ax.set(xlabel="Превышение",ylabel="Плотность",title="Подгонка GPD"); ax.legend(); return _save(fig,save_path),ax

def compare_evt_vs_autoregression(series,true_mask,evt_detected,ar_detected,save_path=None):
    x=np.asarray(series); truth=np.asarray(true_mask,bool); masks=[truth,np.asarray(evt_detected,bool),np.asarray(ar_detected,bool)]
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for ax,mask,color,title in zip(axes,masks,["tab:red","tab:green","tab:orange"],["Истина","POT EVT","AR: порог остатков"]):
        ax.plot(x,lw=.7,color=".45"); ax.scatter(np.flatnonzero(mask),x[mask],c=color,s=18); ax.set_title(title)
    text=[]
    for name,mask in [("EVT",masks[1]),("AR",masks[2])]:
        text.append(f"{name}: P={precision_score(truth,mask,zero_division=0):.2f}, R={recall_score(truth,mask,zero_division=0):.2f}, F1={f1_score(truth,mask,zero_division=0):.2f}")
    axes[1].text(.01,.95,"\n".join(text),transform=axes[1].transAxes,va="top",bbox={"facecolor":"white","alpha":.8}); axes[-1].set_xlabel("Время")
    return _save(fig,save_path),axes

def plot_diagnostic_qq(empirical,theoretical,save_path=None):
    fig,ax=plt.subplots(); ax.scatter(theoretical,empirical,s=16); lo=min(np.min(empirical),np.min(theoretical)); hi=max(np.max(empirical),np.max(theoretical)); ax.plot([lo,hi],[lo,hi],"--",color=".4"); ax.set(xlabel="Теоретические квантили",ylabel="Эмпирические квантили",title="QQ-диагностика"); return _save(fig,save_path),ax

def plot_return_levels(table,save_path=None):
    fig,ax=plt.subplots(); x=np.asarray(table["return_period"]); y=np.asarray(table["return_level"]); low=np.asarray(table["ci_lower"]); high=np.asarray(table["ci_upper"]); ax.plot(x,y,marker="o"); ax.fill_between(x,low,high,alpha=.25); ax.set_xscale("log"); ax.set(xlabel="Период возврата (блоки)",ylabel="Уровень",title="Возвратные уровни GEV"); return _save(fig,save_path),ax

def plot_mean_excess(data,thresholds,save_path=None):
    x=np.asarray(data); u=np.asarray(thresholds); means=np.array([np.mean(x[x>v]-v) if np.any(x>v) else np.nan for v in u]); fig,ax=plt.subplots(); ax.plot(u,means,marker="."); ax.set(xlabel="Порог",ylabel="Среднее превышение",title="Mean Excess Plot"); return _save(fig,save_path),ax

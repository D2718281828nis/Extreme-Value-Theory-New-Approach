"""Пример 2, двигатель: откуда ложные тревоги на исправном участке (циклы 31-60).

Параметрический бутстреп той же проверки, что в aviation_cmapss_pipeline.healthy_false_alarms. Для каждого двигателя
FD001 датчики циклов 1-60 заменяются гауссовым рядом со средним и ковариацией (Ледуа–Вульф) этого двигателя — то есть
заведомо стационарным фоном без деградации, и считается доля двигателей с ложной тревогой. Три нулевые модели:

  A  независимые циклы, индикатор с истинными параметрами   — только ошибка порога на коротком фоне (30 циклов);
  B  независимые циклы, индикатор подогнан по циклам 1-30    — плюс ошибка подгонки индикатора (как в пайплайне);
  C  как B, но циклы AR(1) с наблюдаемой автокорреляцией     — плюс кластеры превышений.

Если B воспроизводит наблюдаемые числа, их объясняет короткий фон; если только C — зависимость превышений; если ни одна,
сам участок 31-60 нестационарен (дрейф). Дополнительно — доля двигателей со значимым трендом индикатора на циклах 1-60.

Запуск: python aviation_false_alarm_bootstrap.py --data ../data/CMAPSSData --out ../results/aviation
"""
from __future__ import annotations
import argparse, json, os, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
import aviation_cmapss_pipeline as A

warnings.filterwarnings("ignore")
KEYS = [f"fa_{m}_healthy_r{r}" for m in ["gev", "pot"] for r in [2, 3]]


def lag1_detrended(X):
    """Медиана по датчикам автокорреляции лага 1 после удаления линейного тренда."""
    t = np.arange(len(X)); slope, icpt = np.polyfit(t, X, 1)
    res = X - (np.outer(t, slope) + icpt)
    return float(np.median([np.corrcoef(res[:-1, j], res[1:, j])[0, 1] for j in range(X.shape[1])]))


def simulate(mu, cov, phi, n, rng):
    """Гауссов ряд n x p с ковариацией cov и общей автокорреляцией лага 1 phi (phi = 0 — независимые циклы)."""
    E = rng.multivariate_normal(np.zeros(len(mu)), cov, size=n)
    Z = np.empty_like(E); Z[0] = E[0]
    for t in range(1, n):
        Z[t] = phi * Z[t - 1] + np.sqrt(1 - phi ** 2) * E[t]
    return mu + Z


def true_indicator(X, mu, cov):
    P = np.linalg.inv(cov); Z = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", Z, P, Z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/CMAPSSData"); ap.add_argument("--out", default="../results/aviation")
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    d1, md5 = A.load(a.data, "FD001")
    obs, sim, trend, phis = [], {k: [] for k in "ABC"}, [], []
    for u, g in d1.groupby("unit"):
        X = g[A.S].values.astype(float)[: A.B]
        obs.append(A.healthy_false_alarms(A.health_indicator(X, 30)))
        mu = X.mean(0); cov = LedoitWolf().fit(X).covariance_; phi = float(np.clip(lag1_detrended(X), 0, 0.95)); phis.append(phi)
        hi = A.health_indicator(X, 30)
        trend.append(stats.spearmanr(np.arange(A.B), hi).pvalue)
        rng = np.random.default_rng([a.seed, int(u)])
        for _ in range(a.reps):
            Xa = simulate(mu, cov, 0.0, A.B, rng)
            sim["A"].append(A.healthy_false_alarms(true_indicator(Xa, mu, cov)))
            sim["B"].append(A.healthy_false_alarms(A.health_indicator(Xa, 30)))
            sim["C"].append(A.healthy_false_alarms(A.health_indicator(simulate(mu, cov, phi, A.B, rng), 30)))
    O = pd.DataFrame(obs)
    def per100(df):
        return {k: round(100 * float(df[k].mean()), 1) for k in KEYS}
    def ci(df):  # 95 % интервал числа двигателей из 100: биномиальный по доле в 100 * reps прогонах
        return {k: [round(100 * float(x), 1) for x in stats.binomtest(int(df[k].sum()), len(df)).proportion_ci()] for k in KEYS}
    R = {"dataset": "NASA C-MAPSS FD001", "md5": md5, "reps_per_engine": a.reps, "seed": a.seed,
         "setup": "фон датчиков циклов 1-60 каждого двигателя заменён гауссовым рядом (среднее и ковариация Ледуа–Вульфа "
                  "этого двигателя); проверка — aviation_cmapss_pipeline.healthy_false_alarms; числа — двигателей из 100",
         "observed": {k: int(O[k].sum()) for k in KEYS},
         "null_A_threshold_error_only": per100(pd.DataFrame(sim["A"])),
         "null_B_plus_indicator_fit": per100(pd.DataFrame(sim["B"])),
         "null_C_plus_ar1_dependence": per100(pd.DataFrame(sim["C"])),
         "ci95": {n: ci(pd.DataFrame(sim[k])) for n, k in [("A", "A"), ("B", "B"), ("C", "C")]},
         "ar1_phi_median": float(np.median(phis)), "ar1_phi_iqr": [float(np.quantile(phis, .25)), float(np.quantile(phis, .75))],
         "trend_1_60": {"frac_engines_spearman_p_lt_0_05": float(np.mean(np.array(trend) < 0.05)), "expected_under_no_trend": 0.05},
         "runtime_s": round(time.time() - t0, 1)}
    json.dump(R, open(f"{a.out}/false_alarm_bootstrap.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

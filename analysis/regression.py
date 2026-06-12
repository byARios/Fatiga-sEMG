"""
analysis/regression.py
-----------------------
Regresión lineal sobre los valores de MPF por bloque.

La pendiente de la recta ajustada (MPFs) es el indicador
cuantitativo de la tasa de fatiga muscular:
  - Pendiente más negativa → mayor tasa de fatiga.
  - R² > 0.5 indica que la tendencia lineal es confiable.

Uso:
    from analysis.regression import fit_mpf_regression, summarize_results
    resultado = fit_mpf_regression(mpf_values)
"""

import numpy as np
from scipy.stats import linregress, ttest_rel


def fit_mpf_regression(mpf_values):
    # ajusta una recta a los 5 valores de MPF
    # pendiente negativa = fatiga, R2 > 0.5 = tendencia confiable
    t = np.arange(1, len(mpf_values) + 1, dtype=float)
    slope, intercept, r, p, _ = linregress(t, mpf_values)
    mpf_pred = slope * t + intercept
    return {
        'slope'    : slope,
        'intercept': intercept,
        'r2'       : r ** 2,
        'p_value'  : p,
        'mpf_pred' : mpf_pred,
        'n_blocks' : len(mpf_values),
    }


def compare_conditions(results_c0, results_c5):
    # prueba t pareada para ver si C5 se fatiga mas que C0
    slopes_c0 = np.array([r['slope'] for r in results_c0])
    slopes_c5 = np.array([r['slope'] for r in results_c5])
    t_stat, p_value = ttest_rel(slopes_c0, slopes_c5)
    return {
        'slopes_c0'  : slopes_c0,
        'slopes_c5'  : slopes_c5,
        'mean_c0'    : float(np.mean(slopes_c0)),
        'mean_c5'    : float(np.mean(slopes_c5)),
        'std_c0'     : float(np.std(slopes_c0)),
        'std_c5'     : float(np.std(slopes_c5)),
        't_stat'     : float(t_stat),
        'p_value'    : float(p_value),
        'significant': bool(p_value < 0.05),
    }


def summarize_results(subject_id, res_c0, res_c1):
    # imprime los resultados de un sujeto en la terminal
    print(f"\n--- Sujeto {subject_id} ---")
    print(f"  C0 ->  MPFs = {res_c0['slope']:+.3f} Hz/bloque  |  "
          f"R2 = {res_c0['r2']:.3f}  |  p = {res_c0['p_value']:.3f}")
    print(f"  C5 ->  MPFs = {res_c1['slope']:+.3f} Hz/bloque  |  "
          f"R2 = {res_c1['r2']:.3f}  |  p = {res_c1['p_value']:.3f}")
    delta = res_c1['slope'] - res_c0['slope']
    print(f"  Delta MPFs (C5 - C0) = {delta:+.3f} Hz/bloque")
    if res_c1['slope'] < res_c0['slope']:
        print("  -> C5 muestra mayor tasa de fatiga que C0 (esperado).")
    else:
        print("  -> C0 muestra mayor o igual tasa de fatiga que C5 (revisar).")
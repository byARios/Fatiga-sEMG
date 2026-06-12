"""
pipeline.py
------------
Script principal del pipeline de análisis de fatiga del bíceps braquial.

Ejecuta el análisis completo sobre todos los sujetos disponibles en la
carpeta de datos y genera las figuras y la tabla de resultados.

Estructura esperada de datos:
    data/
        S01_C0.txt
        S01_C5.txt
        S02_C0.txt
        S02_C5.txt
        ...

Uso desde terminal:
    python pipeline.py

    Opcional — para usar datos de muestra (señal sintética):
    python pipeline.py --demo
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.filter_butterworth import apply_filter
from analysis.mpf_calculator import load_signal, compute_mpf_all_blocks
from analysis.regression import fit_mpf_regression, compare_conditions, summarize_results
from visualization.plot_mpf import (plot_raw_vs_filtered, plot_spectrum,
                                     plot_mpf_vs_block, plot_slope_comparison,
                                     plot_boxplot_ci)

# parametros del pipeline
FS       = 1000
N_BLOCKS = 5
WINDOW   = 1000
OVERLAP  = 500
REST_S   = 10
LOWCUT   = 20
HIGHCUT  = 500
ORDER    = 4
OUT_DIR  = 'figuras'


def process_subject(subject_id, path_c0, path_c5, save_figs=True):
    # carga, filtra, calcula MPF y regresion para un sujeto
    raw_c0, _ = load_signal(path_c0, fs=FS)
    raw_c5, _ = load_signal(path_c5, fs=FS)

    filt_c0 = apply_filter(raw_c0, fs=FS, lowcut=LOWCUT,
                            highcut=HIGHCUT, order=ORDER)
    filt_c5 = apply_filter(raw_c5, fs=FS, lowcut=LOWCUT,
                            highcut=HIGHCUT, order=ORDER)

    mpf_c0, _ = compute_mpf_all_blocks(filt_c0, fs=FS, n_blocks=N_BLOCKS,
                                        window=WINDOW, overlap=OVERLAP,
                                        rest_s=REST_S)
    mpf_c5, _ = compute_mpf_all_blocks(filt_c5, fs=FS, n_blocks=N_BLOCKS,
                                        window=WINDOW, overlap=OVERLAP,
                                        rest_s=REST_S)

    res_c0 = fit_mpf_regression(mpf_c0)
    res_c5 = fit_mpf_regression(mpf_c5)

    summarize_results(subject_id, res_c0, res_c5)

    if save_figs:
        os.makedirs(OUT_DIR, exist_ok=True)

        # señal y espectro para todos los sujetos
        if True:
            plot_raw_vs_filtered(
                raw_c5, filt_c5, fs=FS,
                subject=subject_id, condition='C5',
                n_blocks=N_BLOCKS, rest_s=REST_S,
                save_path=f'{OUT_DIR}/fig1_senal_{subject_id}.png'
            )
            plt.close('all')
            plot_spectrum(
                raw_c5, filt_c5, fs=FS,
                subject=subject_id, condition='C5',
                save_path=f'{OUT_DIR}/fig2_espectro_{subject_id}.png'
            )
            plt.close('all')

        plot_mpf_vs_block(
            mpf_c0, mpf_c5, res_c0, res_c5,
            subject=subject_id,
            save_path=f'{OUT_DIR}/fig3_mpf_{subject_id}.png'
        )
        plt.close('all')

    return res_c0, res_c5, mpf_c0, mpf_c5


def print_summary_table(subject_ids, all_res_c0, all_res_c5):
    # imprime la tabla final de resultados
    header = (f"{'Sujeto':>8}  {'MPFs C0':>10}  {'R2 C0':>7}  "
              f"{'MPFs C5':>10}  {'R2 C5':>7}  {'dMPFs':>10}")
    print("\n" + "=" * len(header))
    print("TABLA DE RESULTADOS — Pendiente de fatiga por sujeto y condicion")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for sid, r0, r5 in zip(subject_ids, all_res_c0, all_res_c5):
        delta = r5['slope'] - r0['slope']
        print(f"{sid:>8}  {r0['slope']:>+10.3f}  {r0['r2']:>7.3f}  "
              f"{r5['slope']:>+10.3f}  {r5['r2']:>7.3f}  {delta:>+10.3f}")
    m0  = np.mean([r['slope'] for r in all_res_c0])
    m5  = np.mean([r['slope'] for r in all_res_c5])
    r0m = np.mean([r['r2'] for r in all_res_c0])
    r5m = np.mean([r['r2'] for r in all_res_c5])
    print("-" * len(header))
    print(f"{'Promedio':>8}  {m0:>+10.3f}  {r0m:>7.3f}  "
          f"{m5:>+10.3f}  {r5m:>7.3f}  {m5-m0:>+10.3f}")
    print("=" * len(header))


def main(demo=False):
    if demo:
        print("\n[DEMO] usando señales sinteticas de data/sample/")
        pairs = [
            ('S01_demo',
             'data/sample/synthetic_emg_C0.txt',
             'data/sample/synthetic_emg_C5.txt')
        ]
    else:
        files_c0 = sorted(glob.glob('data/*_C0.txt'))
        if not files_c0:
            print("\n[ERROR] no se encontraron archivos en data/*_C0.txt")
            print("        usa --demo para correr con señales sinteticas")
            print("        o coloca tus archivos como: data/S01_C0.txt")
            sys.exit(1)
        pairs = []
        for path_c0 in files_c0:
            sid     = os.path.basename(path_c0).replace('_C0.txt', '')
            path_c5 = path_c0.replace('_C0.txt', '_C5.txt')
            if not os.path.exists(path_c5):
                print(f"[aviso] no se encontro {path_c5}, saltando {sid}")
                continue
            pairs.append((sid, path_c0, path_c5))

    print(f"\nSujetos a procesar: {len(pairs)}")
    print(f"fs={FS} Hz | bloques={N_BLOCKS} | ventana={WINDOW} | "
          f"filtro {LOWCUT}-{HIGHCUT} Hz\n")

    subject_ids = []
    all_res_c0  = []
    all_res_c5  = []

    for sid, path_c0, path_c5 in pairs:
        res_c0, res_c5, _, _ = process_subject(sid, path_c0, path_c5,
                                                save_figs=True)
        subject_ids.append(sid)
        all_res_c0.append(res_c0)
        all_res_c5.append(res_c5)

    print_summary_table(subject_ids, all_res_c0, all_res_c5)

    if len(subject_ids) > 1:
        comp = compare_conditions(all_res_c0, all_res_c5)
        print(f"\nprueba t pareada (C0 vs C5):")
        print(f"  media C0 = {comp['mean_c0']:+.3f} +/- {comp['std_c0']:.3f} Hz/bloque")
        print(f"  media C5 = {comp['mean_c5']:+.3f} +/- {comp['std_c5']:.3f} Hz/bloque")
        print(f"  t = {comp['t_stat']:.3f}  |  p = {comp['p_value']:.4f}")
        if comp['significant']:
            print("  -> diferencia significativa (p < 0.05): C5 se fatiga mas")
        else:
            print("  -> diferencia no significativa todavia")

        plot_slope_comparison(
            comp,
            save_path=f'{OUT_DIR}/fig4_comparacion_pendientes.png'
        )
        plt.close('all')

        plot_boxplot_ci(
            comp,
            save_path=f'{OUT_DIR}/fig5_boxplot_pendientes.png'
        )
        plt.close('all')

    print(f"\nfiguras guardadas en: {OUT_DIR}/")
    print("pipeline completado\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fatiga sEMG - Equipo 4')
    parser.add_argument('--demo', action='store_true',
                        help='usar señales sinteticas')
    args = parser.parse_args()
    main(demo=args.demo)
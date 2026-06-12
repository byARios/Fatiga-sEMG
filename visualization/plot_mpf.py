"""
visualization/plot_mpf.py
--------------------------
Funciones de visualización del pipeline de fatiga sEMG.

Genera cuatro tipos de figura:
  1. Señal cruda vs filtrada con sombreado de bloques.
  2. Espectro de potencia antes y después del filtrado.
  3. MPF vs bloque para C0 y C5 con rectas de regresión.
  4. Comparación de pendientes: barras por sujeto + panel media ± DE.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'legend.framealpha': 0.9,
    'axes.grid'        : True,
    'grid.color'       : '#E0E0E0',
    'grid.linewidth'   : 0.6,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
})

COLOR_C0      = '#1B4F72'
COLOR_C5      = '#922B21'
BLOCK_PALETTE = ['#D6EAF8', '#D5F5E3', '#FDEBD0', '#F9EBEA', '#EAF2FF']


def _clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_raw_vs_filtered(raw, filtered, fs=1000,
                          subject='S01', condition='C5',
                          n_blocks=5, rest_s=10,
                          save_path=None):
    time = np.arange(len(raw)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for ax in axes:
        _clean_ax(ax)

    axes[0].plot(time, raw, color='#555555', linewidth=0.4, alpha=0.85)
    axes[0].set_ylabel('Amplitud (mV)')
    axes[0].set_title(f'Señal sEMG cruda — {subject}, condición {condition}')

    axes[1].plot(time, filtered, color=COLOR_C5, linewidth=0.5)
    axes[1].set_ylabel('Amplitud (mV)')
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_title('Señal filtrada (Butterworth 20–500 Hz)')

    # sombreado de bloques en ambos paneles
    block_s = 240 / n_blocks
    for i in range(n_blocks):
        t0 = rest_s + i * block_s
        t1 = rest_s + (i + 1) * block_s
        for ax in axes:
            ax.axvspan(t0, t1, alpha=0.15, color=BLOCK_PALETTE[i], zorder=0)
        # etiqueta de bloque en el panel inferior
        axes[1].text(
            (t0 + t1) / 2, 0.04, f'B{i+1}',
            ha='center', va='bottom', fontsize=8,
            color='#333333', fontweight='bold',
            transform=axes[1].get_xaxis_transform()
        )

    fig.tight_layout(rect=[0, 0, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"guardada: {save_path}")
    return fig


def plot_spectrum(raw, filtered, fs=1000,
                  subject='S01', condition='C5',
                  save_path=None):
    def psd(sig):
        # elimina el offset DC antes de calcular el espectro
        sig_ac = sig - np.mean(sig)
        f  = np.fft.rfftfreq(len(sig_ac), d=1.0 / fs)
        p  = np.abs(np.fft.rfft(sig_ac)) ** 2
        pn = p / np.max(p)
        return f[1:], pn[1:]   # omite el bin DC (índice 0)

    f_raw, p_raw = psd(raw)
    f_fil, p_fil = psd(filtered)

    fig, ax = plt.subplots(figsize=(9, 4))
    _clean_ax(ax)

    # banda de paso sombreada
    ax.axvspan(20, 500, alpha=0.07, color=COLOR_C0, label='Banda de paso (20–500 Hz)')

    ax.semilogy(f_raw, p_raw + 1e-10, color='#AAAAAA', linewidth=0.8, label='Cruda')
    ax.semilogy(f_fil, p_fil + 1e-10, color=COLOR_C5, linewidth=1.3,
                label='Filtrada (20–500 Hz)')
    ax.axvline(20,  color='#B46A00', linestyle='--', linewidth=1.2, alpha=0.85,
               label='20 Hz')
    ax.axvline(500, color='#7B241C', linestyle='--', linewidth=1.2, alpha=0.85,
               label='500 Hz')

    ax.set_xlim(0, 550)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Potencia normalizada (log)')
    ax.set_title(f'Espectro de potencia — {subject}, condición {condition}')
    ax.legend(loc='upper right')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"guardada: {save_path}")
    return fig


def plot_mpf_vs_block(mpf_c0, mpf_c5, res_c0, res_c5,
                       subject='S01', save_path=None):
    n = len(mpf_c0)
    t = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _clean_ax(ax)

    # relleno entre las dos curvas
    ax.fill_between(t, mpf_c0, mpf_c5, alpha=0.08, color='#555555')

    # datos con marcadores huecos
    ax.plot(t, mpf_c0, 'o-', color=COLOR_C0, linewidth=2,
            markersize=9, markerfacecolor='white', markeredgewidth=2.2,
            label='C0 — sin carga')
    ax.plot(t, mpf_c5, 's-', color=COLOR_C5, linewidth=2,
            markersize=9, markerfacecolor='white', markeredgewidth=2.2,
            label='C5 — 5 kg')

    # regresión como línea punteada
    ax.plot(t, res_c0['mpf_pred'], '--', color=COLOR_C0, linewidth=1.3, alpha=0.55)
    ax.plot(t, res_c5['mpf_pred'], '--', color=COLOR_C5, linewidth=1.3, alpha=0.55)

    # cuadro de estadísticas fuera de la leyenda
    stats = (
        f"C0:  pendiente = {res_c0['slope']:+.2f} Hz/bloque   R² = {res_c0['r2']:.2f}\n"
        f"C5:  pendiente = {res_c5['slope']:+.2f} Hz/bloque   R² = {res_c5['r2']:.2f}"
    )
    ax.text(0.98, 0.97, stats,
            transform=ax.transAxes, fontsize=8.5,
            va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.95))

    ax.set_xticks(t)
    ax.set_xticklabels([f'B{i}' for i in t])
    ax.set_xlabel('Bloque temporal')
    ax.set_ylabel('MPF (Hz)')
    ax.set_title(f'MPF vs bloque — {subject}: C0 vs C5')
    ax.legend(loc='lower left')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"guardada: {save_path}")
    return fig


def plot_boxplot_ci(comparison, save_path=None):
    """Boxplot de pendientes C0 vs C5 con IC 95% y puntos individuales."""
    slopes_c0 = np.array(comparison['slopes_c0'])
    slopes_c5 = np.array(comparison['slopes_c5'])
    n     = len(slopes_c0)
    pval  = comparison['p_value']
    sig   = comparison['significant']

    fig, ax = plt.subplots(figsize=(6, 5.5))
    _clean_ax(ax)

    data   = [slopes_c0, slopes_c5]
    labels = ['C0\n(sin carga)', 'C5\n(5 kg)']
    colors = [COLOR_C0, COLOR_C5]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='x', color='#888888', markersize=5),
                    widths=0.4)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # puntos individuales con jitter leve
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.09, 0.09, size=len(vals))
        ax.scatter(i + jitter, vals, color=color, s=42, zorder=5,
                   edgecolors='white', linewidths=0.7, alpha=0.9)

    # IC 95% de la media (línea roja horizontal sobre cada caja)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        mean  = np.mean(vals)
        se    = stats.sem(vals)
        ci    = stats.t.ppf(0.975, df=n - 1) * se
        ax.errorbar(i, mean, yerr=ci, fmt='D', color='#1C1C1C',
                    markersize=6, capsize=6, capthick=1.8,
                    elinewidth=1.8, zorder=6, label='Media ± IC 95%' if i == 1 else '')

    # línea cero y anotación p
    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--')
    p_color = '#1A7A1A' if sig else '#C0392B'
    p_text  = f'p = {pval:.4f}{"  ✓" if sig else ""}  (t = {comparison["t_stat"]:.2f})'
    ax.text(0.98, 0.03, p_text, transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom', color=p_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=p_color, alpha=0.9))

    ax.set_ylabel('Pendiente MPFs (Hz/bloque)')
    ax.set_title('Distribución de pendientes de fatiga\nC0 vs C5 — n = 11', pad=10)
    ax.legend(loc='upper right', fontsize=8.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"guardada: {save_path}")
    return fig


def plot_slope_comparison(comparison, save_path=None):
    slopes_c0 = comparison['slopes_c0']
    slopes_c5 = comparison['slopes_c5']
    n     = len(slopes_c0)
    subj  = [f'S{i+1:02d}' for i in range(n)]
    x     = np.arange(n)
    w     = 0.32

    mean_c0 = comparison['mean_c0']
    mean_c5 = comparison['mean_c5']
    std_c0  = comparison['std_c0']
    std_c5  = comparison['std_c5']
    pval    = comparison['p_value']
    sig     = comparison['significant']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                              gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle('Comparación de pendientes de fatiga: C0 vs C5',
                 fontsize=12, fontweight='bold')

    for ax in axes:
        _clean_ax(ax)

    # --- panel izquierdo: barras individuales por sujeto ---
    ax = axes[0]
    ax.bar(x - w/2, slopes_c0, w, color=COLOR_C0, alpha=0.82,
           label='C0 — sin carga')
    ax.bar(x + w/2, slopes_c5, w, color=COLOR_C5, alpha=0.82,
           label='C5 — 5 kg')

    # líneas de media punteadas
    ax.axhline(mean_c0, color=COLOR_C0, linestyle=':', linewidth=2.0,
               alpha=0.75, label=f'Media C0 = {mean_c0:+.2f}')
    ax.axhline(mean_c5, color=COLOR_C5, linestyle=':', linewidth=2.0,
               alpha=0.75, label=f'Media C5 = {mean_c5:+.2f}')
    ax.axhline(0, color='#222222', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(subj)
    ax.set_xlabel('Sujeto')
    ax.set_ylabel('Pendiente MPFs (Hz/bloque)')
    ax.set_title('Pendientes individuales')
    ax.legend(loc='upper right', fontsize=8.5)

    # anotación p-value con color según significancia
    p_color = '#1A7A1A' if sig else '#C0392B'
    p_label = (f'p = {pval:.3f}  (significativo)'
               if sig else f'p = {pval:.3f}  (no significativo)')
    ax.text(0.02, 0.04, p_label,
            transform=ax.transAxes, fontsize=9,
            color=p_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=p_color, alpha=0.9))

    # --- panel derecho: media ± desviación estándar ---
    ax2 = axes[1]
    ax2.bar([0, 1], [mean_c0, mean_c5], 0.45,
            color=[COLOR_C0, COLOR_C5], alpha=0.85,
            yerr=[std_c0, std_c5], capsize=8,
            error_kw=dict(elinewidth=2.0, ecolor='#222222', capthick=2.0))
    ax2.axhline(0, color='#222222', linewidth=0.8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['C0', 'C5'])
    ax2.set_ylabel('Pendiente media (Hz/bloque)')
    ax2.set_title('Media ± DE')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"guardada: {save_path}")
    return fig

"""
preprocessing/filter_butterworth.py
------------------------------------
Filtrado digital de señales sEMG.

Aplica un filtro Butterworth pasa-banda de cuarto orden
con corte inferior en 20 Hz y superior en 500 Hz, usando
representación en secciones de segundo orden (SOS) para
garantizar estabilidad numérica.

Uso:
    from preprocessing.filter_butterworth import apply_filter
    signal_filtrada = apply_filter(signal_cruda, fs=1000)
"""


import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz
import matplotlib.pyplot as plt


def design_filter(lowcut=20, highcut=500, fs=1000, order=4):
    # diseña el filtro con los cortes que necesitamos
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.999)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def apply_filter(signal, fs=1000, lowcut=20, highcut=500, order=4):
    # aplica el filtro a la señal cruda
    sos = design_filter(lowcut, highcut, fs, order)
    return sosfilt(sos, signal)


def plot_filter_response(fs=1000, lowcut=20, highcut=500, order=4,
                          save_path=None):
    # grafica la respuesta en frecuencia para verificar que el filtro quedo bien
    sos = design_filter(lowcut, highcut, fs, order)
    w, h = sosfreqz(sos, worN=8000, fs=fs)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12),
            color='#003850', linewidth=1.8)
    ax.axvline(lowcut, color='#B46A00', linestyle='--',
               linewidth=1.2, label=f'Corte inferior: {lowcut} Hz')
    ax.axvline(highcut, color='#963319', linestyle='--',
               linewidth=1.2, label=f'Corte superior: {highcut} Hz')
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-80, 5)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Magnitud (dB)')
    ax.set_title(f'Filtro Butterworth orden {order} ({lowcut}–{highcut} Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"guardada: {save_path}")
    plt.show()
    return fig
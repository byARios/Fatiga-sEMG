"""
analysis/mpf_calculator.py
---------------------------
Segmentación de la señal sEMG en bloques y cálculo de la
Frecuencia Media de Potencia (MPF) por bloque mediante FFT
con ventana deslizante.

La MPF es el centroide espectral de la densidad de potencia:

    MPF = Σ(f · PSD(f)) / Σ(PSD(f))

Un descenso progresivo de la MPF a lo largo de los bloques
indica fatiga muscular acumulada.

Uso:
    from analysis.mpf_calculator import segment_signal, compute_mpf_block
    bloques = segment_signal(signal_filtrada, n_blocks=5)
    mpf_values = [compute_mpf_block(b) for b in bloques]
"""
import numpy as np


def load_signal(filepath, fs=1000):
    # carga el archivo txt del BIOPAC, una columna sin encabezado
    signal = np.loadtxt(filepath)
    time = np.arange(len(signal)) / fs
    return signal, time


def remove_rest(signal, fs=1000, rest_s=10):
    # quitamos los primeros 10 seg de reposo y tomamos solo 240 seg de tarea
    rest_samples = int(rest_s * fs)
    task_samples = int(240 * fs)
    return signal[rest_samples:rest_samples + task_samples]


def segment_signal(signal, n_blocks=5):
    # dividimos en 5 bloques iguales
    return np.array_split(signal, n_blocks)


def compute_mpf_block(segment, fs=1000, window=1000, overlap=500):
    # calcula la MPF de un bloque usando FFT con ventana deslizante
    step = window - overlap
    mpf_vals = []
    i = 0
    while i + window <= len(segment):
        win = segment[i:i + window]
        # ventana de hanning para reducir ruido espectral
        win_hann = win * np.hanning(window)
        psd = np.abs(np.fft.rfft(win_hann)) ** 2
        freqs = np.fft.rfftfreq(window, d=1.0 / fs)
        total_power = np.sum(psd)
        if total_power > 0:
            # MPF = promedio ponderado de frecuencias
            mpf = np.sum(freqs * psd) / total_power
            mpf_vals.append(mpf)
        i += step
    if len(mpf_vals) == 0:
        return np.nan
    return float(np.mean(mpf_vals))


def compute_mpf_all_blocks(signal, fs=1000, n_blocks=5,
                            window=1000, overlap=500, rest_s=10):
    # pipeline completo: quita reposo, segmenta y calcula MPF
    signal_task = remove_rest(signal, fs=fs, rest_s=rest_s)
    blocks = segment_signal(signal_task, n_blocks=n_blocks)
    mpf_values = np.array([
        compute_mpf_block(b, fs=fs, window=window, overlap=overlap)
        for b in blocks
    ])
    return mpf_values, blocks

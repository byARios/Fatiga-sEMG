"""
Genera una señal sEMG sintética con fatiga progresiva para demo.
Simula 40 repeticiones de curl de bíceps con frecuencia media decreciente.
Exporta en el mismo formato .txt del BIOPAC: una columna, sin encabezado.
"""
import numpy as np

def generate_synthetic_emg(fs=1000, n_reps=40, cycle_s=6,
                            rest_s=10, seed=42):
    np.random.seed(seed)
    rest_samples = int(rest_s * fs)
    cycle_samples = int(cycle_s * fs)
    total_samples = rest_samples + n_reps * cycle_samples

    signal = np.zeros(total_samples)

    # Reposo inicial: ruido de fondo
    signal[:rest_samples] = np.random.normal(0, 0.005, rest_samples)

    # Frecuencia media inicial y final (simulando fatiga)
    freq_start = 120  # Hz
    freq_end   = 75   # Hz

    for rep in range(n_reps):
        t_rep = rep / n_reps  # 0 → 1
        freq_center = freq_start - t_rep * (freq_start - freq_end)
        amplitude = 0.8 + 0.3 * np.random.randn()
        amplitude = max(0.4, amplitude)

        start = rest_samples + rep * cycle_samples
        # Activación: clicks 1 y 2 (primeros 2 segundos)
        active_samples = int(2 * fs)
        t_active = np.linspace(0, 2, active_samples)

        burst = amplitude * np.random.randn(active_samples)
        # Modular con una envolvente y frecuencia central simulada
        carrier = np.sin(2 * np.pi * freq_center * t_active)
        envelope = np.exp(-t_active * 0.5) + 0.3
        burst = burst * envelope * 0.3

        signal[start:start + active_samples] = burst

        # Descanso: ruido de fondo (clicks 3 y 4)
        rest_rep = cycle_samples - active_samples
        signal[start + active_samples:start + cycle_samples] = \
            np.random.normal(0, 0.005, rest_rep)

    return signal

if __name__ == "__main__":
    fs = 1000
    for cond, seed in [("C0", 42), ("C5", 99)]:
        sig = generate_synthetic_emg(fs=fs, seed=seed)
        # Más fatiga en C5
        if cond == "C5":
            sig = generate_synthetic_emg(fs=fs, n_reps=40,
                                          freq_start=120,
                                          freq_end=55, seed=seed)
        fname = f"synthetic_emg_{cond}.txt"
        np.savetxt(f"/home/claude/emg-biceps-fatigue/data/sample/{fname}",
                   sig, fmt="%.6f")
        print(f"Generado: {fname} ({len(sig)} muestras)")

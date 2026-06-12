"""
ver_senal.py
-------------
Script rápido para visualizar una señal sEMG del BIOPAC.
Úsado para verificar que la señal se ve bien antes de correr el pipeline completo.

Uso:
    python ver_senal.py S01_C0.txt
    python ver_senal.py data/S01_C0.txt
"""

# visor rapido de señal sEMG del BIOPAC
# uso: python ver_senal.py data/S01_C0.txt
# Equipo 4 - Biopotenciales Neuromusculares

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import linregress

# parametros
FS       = 1000
LOWCUT   = 20
HIGHCUT  = 499
ORDER    = 4
REST_S   = 10
N_BLOCKS = 5
WINDOW   = 1000
OVERLAP  = 500

if len(sys.argv) < 2:
    print("uso: python ver_senal.py data/S01_C0.txt")
    sys.exit(1)

filepath = sys.argv[1]

try:
    signal = np.loadtxt(filepath)
except Exception as e:
    print(f"error al cargar el archivo: {e}")
    sys.exit(1)

time = np.arange(len(signal)) / FS
print(f"\nArchivo: {filepath}")
print(f"Muestras: {len(signal)}")
print(f"Duracion: {len(signal)/FS:.1f} segundos")
print(f"Amplitud maxima: {np.max(np.abs(signal)):.4f} mV")

# filtrado
nyq  = 0.5 * FS
low  = LOWCUT / nyq
high = min(HIGHCUT / nyq, 0.999)
sos  = butter(ORDER, [low, high], btype='band', output='sos')
filtered = sosfilt(sos, signal)

# quitamos reposo inicial y tomamos 240 seg de tarea
rest_samples = int(REST_S * FS)
task_samples = int(240 * FS)
signal_task  = filtered[rest_samples:rest_samples + task_samples]
time_task    = np.arange(len(signal_task)) / FS

# espectro de la señal completa filtrada y cruda
freqs     = np.fft.rfftfreq(len(filtered), d=1.0/FS)
psd       = np.abs(np.fft.rfft(filtered)) ** 2
psd_n     = psd / np.max(psd)
freqs_raw = np.fft.rfftfreq(len(signal), d=1.0/FS)
psd_raw   = np.abs(np.fft.rfft(signal)) ** 2
psd_raw_n = psd_raw / np.max(psd_raw)

# MPF por bloque
blocks = np.array_split(signal_task, N_BLOCKS)

def mpf_block(seg):
    step = WINDOW - OVERLAP
    vals = []
    i = 0
    while i + WINDOW <= len(seg):
        w   = seg[i:i+WINDOW] * np.hanning(WINDOW)
        p   = np.abs(np.fft.rfft(w)) ** 2
        f   = np.fft.rfftfreq(WINDOW, d=1.0/FS)
        tot = np.sum(p)
        if tot > 0:
            vals.append(np.sum(f * p) / tot)
        i += step
    return float(np.mean(vals)) if vals else np.nan

mpf_values = np.array([mpf_block(b) for b in blocks])
block_nums = np.arange(1, N_BLOCKS + 1)

slope, intercept, r, p, _ = linregress(block_nums, mpf_values)
mpf_pred = slope * block_nums + intercept

print(f"\nMPF por bloque (Hz): {np.round(mpf_values, 2)}")
print(f"Pendiente: {slope:+.3f} Hz/bloque")
print(f"R2: {r**2:.3f}")

# figura principal con 4 paneles
fig = plt.figure(figsize=(12, 9))
fig.suptitle(f"{filepath}", fontsize=11, fontweight='bold')
gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time, signal, color='#888888', linewidth=0.5)
ax1.set_title('señal cruda completa', fontsize=10, fontweight='bold')
ax1.set_xlabel('tiempo (s)')
ax1.set_ylabel('amplitud (mV)')
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time_task, signal_task, color='#003850', linewidth=0.6)
ax2.set_title('señal filtrada - 240 seg de tarea', fontsize=10, fontweight='bold')
ax2.set_xlabel('tiempo (s)')
ax2.set_ylabel('amplitud (mV)')
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax3 = fig.add_subplot(gs[2, 0])
ax3.semilogy(freqs_raw, psd_raw_n + 1e-10, color='#AAAAAA',
             linewidth=0.8, label='cruda')
ax3.semilogy(freqs, psd_n + 1e-10, color='#963319',
             linewidth=1.2, label='filtrada')
ax3.axvline(LOWCUT,  color='#B46A00', linestyle='--', linewidth=1, alpha=0.8)
ax3.axvline(HIGHCUT, color='#963319', linestyle='--', linewidth=1, alpha=0.8)
ax3.set_xlim(0, 550)
ax3.set_title('espectro de potencia', fontsize=10, fontweight='bold')
ax3.set_xlabel('frecuencia (Hz)')
ax3.set_ylabel('potencia (log)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(block_nums, mpf_values, 'o-', color='#003850',
         linewidth=1.8, markersize=8, label='MPF por bloque')
ax4.plot(block_nums, mpf_pred, '--', color='#963319',
         linewidth=1.2, alpha=0.8,
         label=f'{slope:+.2f} Hz/bloque  R2={r**2:.2f}')
ax4.set_xticks(block_nums)
ax4.set_xticklabels([f'B{i}' for i in block_nums])
ax4.set_title('MPF vs bloque', fontsize=10, fontweight='bold')
ax4.set_xlabel('bloque temporal')
ax4.set_ylabel('MPF (Hz)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.savefig('ver_senal_output.png', dpi=150, bbox_inches='tight')
print("\nfigura guardada: ver_senal_output.png")

# figura separada con los bloques coloreados sobre la señal
fig2, ax5 = plt.subplots(figsize=(12, 3))
colores    = ['#003850', '#2E7D9A', '#5BA3C0', '#8EC8DE', '#C2E4F0']
block_size = len(signal_task) // N_BLOCKS
time_task2 = np.arange(len(signal_task)) / FS

for i in range(N_BLOCKS):
    inicio = i * block_size
    fin    = inicio + block_size if i < N_BLOCKS - 1 else len(signal_task)
    ax5.axvspan(time_task2[inicio], time_task2[fin - 1],
                alpha=0.2, color=colores[i])
    ax5.text((time_task2[inicio] + time_task2[fin - 1]) / 2,
             np.max(np.abs(signal_task)) * 0.85,
             f'B{i+1}\n{mpf_values[i]:.1f} Hz',
             ha='center', fontsize=8, color=colores[i], fontweight='bold')

ax5.plot(time_task2, signal_task, color='#333333', linewidth=0.5)
ax5.set_title('señal filtrada con bloques y MPF por bloque', fontsize=10, fontweight='bold')
ax5.set_xlabel('tiempo (s)')
ax5.set_ylabel('amplitud (mV)')
ax5.grid(True, alpha=0.3)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
fig2.tight_layout()
fig2.savefig('ver_senal_bloques.png', dpi=150, bbox_inches='tight')
print("figura de bloques guardada: ver_senal_bloques.png")

plt.show()
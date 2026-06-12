# Análisis de Fatiga Muscular del Bíceps Braquial mediante sEMG

**Equipo 4 · Análisis de Biopotenciales Neuro-Musculares · Universidad Veracruzana · Ingeniería Biomédica**

Detección de fatiga muscular del **bíceps braquial** a partir de señales de **electromiografía de superficie (sEMG)**, comparando el ejercicio de curl **sin carga (C0)** contra **5 kg (C5)**. El indicador de fatiga es la pendiente de la **Frecuencia Media de Potencia (MPF)**, que disminuye conforme el músculo se fatiga.

> **Artículo base:** Fang, N., Zhang, C., & Lv, J. (2021). *Effects of Vertical Lifting Distance on Upper-Body Muscle Fatigue.* Int. J. Environ. Res. Public Health, 18(10), 5468. https://doi.org/10.3390/ijerph18105468

---

## Tabla de contenido

- [Descripción del proyecto](#descripción-del-proyecto)
- [Resultado principal](#resultado-principal)
- [Protocolo experimental](#protocolo-experimental)
- [Pipeline de procesamiento](#pipeline-de-procesamiento)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos y dependencias](#requisitos-y-dependencias)
- [Instalación](#instalación)
- [Cómo ejecutarlo](#cómo-ejecutarlo)
- [Ejemplo de uso con datos](#ejemplo-de-uso-con-datos)
- [Formato de los datos](#formato-de-los-datos)
- [Salidas que genera](#salidas-que-genera)
- [Solución de problemas](#solución-de-problemas)
- [Equipo](#equipo)

---

## Descripción del proyecto

Cuando un músculo se fatiga, la velocidad de conducción de sus fibras disminuye y el espectro de potencia de la señal sEMG se desplaza hacia frecuencias más bajas. Ese desplazamiento se cuantifica con la **MPF**: registramos cómo cae la MPF a lo largo de la tarea y ajustamos una recta; **la pendiente (MPF_s) es la tasa de fatiga** — cuanto más negativa, mayor la fatiga.

La hipótesis es que el bíceps, al actuar como **motor principal** del curl, mostrará una caída de MPF mucho más marcada con carga (C5) que sin carga (C0). El proyecto:

1. Adquiere señales sEMG de 11 sujetos en ambas condiciones (BIOPAC, 1000 Hz).
2. Filtra, segmenta y calcula la MPF por bloque.
3. Estima la pendiente de fatiga por sujeto y condición mediante regresión lineal.
4. Compara C0 vs C5 con una **prueba *t* pareada** y visualizaciones agregadas.

## Resultado principal

| Métrica | C0 (sin carga) | C5 (5 kg) |
|---|---|---|
| Pendiente media MPF_s (Hz/bloque) | −0.04 ± 5.29 | **−4.18 ± 4.40** |
| Sujetos con ΔMPF_s < 0 | — | **11 / 11 (100 %)** |
| R² > 0.5 en C5 | — | **8 / 11** |

Prueba *t* pareada: **t(10) = 3.523, p = 0.0055** (diferencia significativa, *d* de Cohen = 1.06). Con 5 kg el bíceps se fatiga significativamente más rápido. Detalle completo en el reporte (`Reporte_Final_Equipo4.pdf`).

---

## Protocolo experimental

- Curl de bíceps sentado, controlado con **metrónomo a 40 bpm**.
- Ciclo de 6 s: 3 s de flexión + 3 s de extensión.
- **40 repeticiones** por condición (≈ 240 s de tarea efectiva) + 10 s de reposo al inicio y al final.
- Dos condiciones: **C0** (sin carga) y **C5** (mancuerna de 5 kg), en sesiones separadas ≥ 48 h y orden aleatorizado.
- Electrodos bipolares Ag/AgCl según **SENIAM** sobre el vientre del bíceps (1/3 de la línea fosa cubital → acromion), separación inter-electrodo 20 mm; referencia en el cóndilo radial.

## Pipeline de procesamiento

```
Señal .txt (BIOPAC, 1000 Hz)
   └─ 1. Recorte de reposo  ......... se descartan los primeros y últimos 10 s
   └─ 2. Filtro Butterworth ......... paso-banda 20–500 Hz, orden 4 (sosfilt)
   └─ 3. Segmentación ............... 5 bloques de 48 s (~8 repeticiones c/u)
   └─ 4. MPF por bloque ............. FFT con ventana Hann de 1000 muestras, 50 % traslape
   └─ 5. Regresión lineal ........... pendiente MPF_s, R² y p (por sujeto y condición)
   └─ 6. Comparación grupal ......... prueba t pareada C0 vs C5 + boxplots con IC 95 %
```

---

## Estructura del repositorio

```
Fatiga-sEMG/
├── README.md                       ← este archivo
├── requirements.txt                ← dependencias de Python
├── pipeline.py                     ← script principal: análisis completo de todos los sujetos
├── ver_senal.py                    ← visor rápido para inspeccionar una señal
├── Reporte_Final_Equipo4.tex       ← reporte técnico final (LaTeX)
├── Reporte_Final_Equipo4.pdf       ← reporte compilado
│
├── preprocessing/
│   └── filter_butterworth.py       ← filtro Butterworth 20–500 Hz
├── analysis/
│   ├── mpf_calculator.py           ← carga, segmentación y cálculo de MPF
│   └── regression.py               ← regresión lineal y prueba t pareada
├── visualization/
│   └── plot_mpf.py                 ← generación de todas las figuras
├── notebooks/
│   └── pipeline_completo.ipynb     ← versión interactiva paso a paso
│
├── data/                           ← señales reales (S01_C0.txt, S01_C5.txt, ...)
│   └── sample/                     ← señales sintéticas para el modo demo
│       ├── synthetic_emg_C0.txt
│       ├── synthetic_emg_C5.txt
│       └── generate_sample.py      ← script que regenera las señales sintéticas
└── figuras/                        ← figuras generadas por el pipeline (salida)
```

> Los módulos `preprocessing/`, `analysis/` y `visualization/` constituyen el código fuente (`src`) del proyecto; `data/` contiene los datos y el reporte cumple el rol de `docs`.

---

## Requisitos y dependencias

- **Python 3.9 o superior** (probado en 3.12 / 3.14).
- Paquetes (en `requirements.txt`):

| Paquete | Uso |
|---|---|
| `numpy` | FFT, segmentación, operaciones vectoriales |
| `scipy` | filtro Butterworth, regresión lineal, prueba *t* |
| `matplotlib` | generación de figuras |
| `jupyter` | notebook interactivo (opcional) |

Para compilar el reporte `.tex` se necesita además una distribución de **LaTeX** (MiKTeX o TeX Live) con `pdflatex`.

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/byARios/Fatiga-sEMG.git
cd Fatiga-sEMG

# 2. (Recomendado) crear un entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Cómo ejecutarlo

### 1. Modo demo (sin datos reales)

La forma más rápida de comprobar que todo funciona. Usa las señales sintéticas de `data/sample/`:

```bash
python pipeline.py --demo
```

### 2. Análisis completo (datos reales)

Procesa todos los pares `SXX_C0.txt` / `SXX_C5.txt` que encuentre en `data/`:

```bash
python pipeline.py
```

### 3. Inspeccionar una sola señal

Útil antes del análisis completo para verificar la calidad de un registro:

```bash
python ver_senal.py data/S01_C0.txt
```

### 4. Notebook interactivo

```bash
jupyter notebook notebooks/pipeline_completo.ipynb
```

Las figuras se guardan automáticamente en `figuras/`.

---

## Ejemplo de uso con datos

Ejecutando el modo demo se obtiene una salida como esta:

```text
$ python pipeline.py --demo

[DEMO] usando señales sinteticas de data/sample/
fs=1000 Hz | bloques=5 | ventana=1000 | filtro 20-500 Hz

--- Sujeto S01_demo ---
  C0 ->  MPFs = +0.860 Hz/bloque  |  R2 = 0.075  |  p = 0.657
  C5 ->  MPFs = -2.511 Hz/bloque  |  R2 = 0.820  |  p = 0.034
  Delta MPFs (C5 - C0) = -3.370 Hz/bloque
  -> C5 muestra mayor tasa de fatiga que C0 (esperado).

figuras guardadas en: figuras/
pipeline completado
```

Con los **datos reales completos** (11 sujetos), la tabla final y la prueba estadística son:

```text
  Sujeto     MPFs C0    R2 C0     MPFs C5    R2 C5       dMPFs
--------------------------------------------------------------
     S01      +1.096    0.045      -2.467    0.850      -3.563
     ...
Promedio      -0.037    0.484      -4.176    0.680      -4.139
--------------------------------------------------------------
prueba t pareada (C0 vs C5):
  t = 3.523  |  p = 0.0055
  -> diferencia significativa (p < 0.05): C5 se fatiga mas
```

---

## Formato de los datos

El BIOPAC exporta cada señal como un `.txt` de **una sola columna** (amplitud), muestreada a **1000 Hz, sin encabezado**:

```
 0.000123
-0.000045
 0.000312
 ...
```

**Nomenclatura obligatoria:** `S<##>_C0.txt` y `S<##>_C5.txt` (por ejemplo `S01_C0.txt`, `S01_C5.txt`, `S02_C0.txt`, …). El pipeline empareja automáticamente C0 con C5 por sujeto.

## Salidas que genera

| Archivo | Contenido |
|---|---|
| `figuras/fig1_senal_SXX.png` | señal cruda vs filtrada, con bloques sombreados |
| `figuras/fig2_espectro_SXX.png` | espectro de potencia antes/después del filtrado |
| `figuras/fig3_mpf_SXX.png` | MPF vs bloque (C0 y C5) con rectas de regresión |
| `figuras/fig4_comparacion_pendientes.png` | pendientes de todos los sujetos comparadas |
| `figuras/fig5_boxplot_pendientes.png` | boxplot agregado C0 vs C5 con IC 95 % |

Además imprime en la terminal la tabla de resultados (pendiente y R² por sujeto) y el resultado de la prueba *t* pareada.

---

## Solución de problemas

- **`[ERROR] no se encontraron archivos en data/*_C0.txt`** → no hay datos reales en `data/`. Usa `python pipeline.py --demo` o coloca tus archivos siguiendo la nomenclatura `SXX_C0.txt` / `SXX_C5.txt`.
- **`ModuleNotFoundError: numpy / scipy / matplotlib`** → faltan dependencias: `pip install -r requirements.txt` (con el entorno virtual activado).
- **`[aviso] no se encontro data/SXX_C5.txt, saltando SXX`** → un sujeto tiene solo una de las dos condiciones; se omite del análisis pareado.
- **La carpeta `figuras/` no aparece** → se crea sola al ejecutar el pipeline; revisa permisos de escritura en el directorio.

---

## Equipo

| Integrante |
|---|
| Carlos Andrés Avendaño Díaz |
| Cristóbal Jesús Colina Crisanto |
| Alejandro de Jesús Ríos Ordaz |
| Maximiliano Sánchez Solano |

**Académico:** Dr. Ismael Kelly Pérez · **Institución:** Universidad Veracruzana, Facultad de Ingeniería — Ingeniería Biomédica

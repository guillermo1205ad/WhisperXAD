# WhisperXAD – Pipeline de Transcripción de Audio con WhisperX (GPU + Diarización)

Este proyecto define un **pipeline de transcripción, alineación y diarización** basado en:

- **WhisperX** (ASR + alineación, usando *faster-whisper* como backend optimizado).
- **PyTorch 2.4.x** con GPU (CUDA) cuando está disponible.
- **pyannote.audio 3.x** (diarización moderna, vía `pyannote/speaker-diarization-3.1`).
- Un runner Bash en `audio_transcription/run_transcribe.sh`.
- Un script principal en `audio_transcription/scripts/transcribe.py`.

Está pensado para ser:

- **Replicable** (entorno definido en `pyproject.toml` + `uv.lock`).
- **Eficiente en GPU** (ASR + alineación en GPU; diarización intenta GPU y, si falla, hace *fallback* ordenado).
- **Usable por lote** sobre directorios completos de audios.

---

## 🧩 Descripción general

El pipeline procesa archivos de audio (individuales o directorios completos) y genera:

1. **Transcripción automática (ASR)** con WhisperX (modelo por defecto: `large-v3`).
2. **Alineación a nivel de palabra**, usando el módulo de alineación de WhisperX (modelos `wav2vec2`).
3. **Diarización de hablantes** con `pyannote.audio` 3.x vía `pyannote/speaker-diarization-3.1`, asignando hablantes (p. ej. `SPEAKER_00`, `SPEAKER_01`) a cada palabra o segmento.

La lógica está encapsulada en:

- `audio_transcription/run_transcribe.sh` — runner práctico para procesar los audios del repositorio.
- `audio_transcription/scripts/transcribe.py` — CLI principal para procesar un archivo o una carpeta completa.

---

## ⚙️ Características principales

- **ASR de alta calidad:** usa `large-v3` por defecto, ejecutándose en GPU (`cuda`) si está disponible.
- **Timestamps a nivel de palabra:** gracias al módulo de alineación de WhisperX (`load_align_model`).
- **Diarización de hablantes (pyannote 3.x):**
  - Utiliza el modelo `pyannote/speaker-diarization-3.1`.
  - Intenta correr en **GPU** cuando hay CUDA disponible.
  - Si no se puede (falta de token, problemas de descarga, etc.), el pipeline **continúa sin diarización** en lugar de fallar.
- **Procesamiento por lotes:** recorre recursivamente un directorio y procesa todos los audios compatibles.
- **Configuración en CLI:** se pueden ajustar modelo, idioma, GPU y salida.
- **Salidas en JSON y CSV:** un archivo detallado y otro a nivel de palabra por cada audio.
- **Replicabilidad:** el entorno está definido en `pyproject.toml` y se instala mediante `uv sync`.

---

## 📁 Estructura del proyecto

```text
WhisperXAD/
├── audio_transcription/
│   ├── inputs/
│   │   └── audios/                  # Carpeta raíz donde el usuario crea o copia sus sets
│   ├── outputs/
│   │   └── transcripciones/         # Salida por defecto del runner Bash
│   ├── scripts/
│   │   └── transcribe.py            # Script CLI principal
│   ├── run_transcribe.sh            # Runner Bash que procesa inputs/audios/
│   └── utilities/
│       └── nltk_data/               # Datos de NLTK (ej. punkt, punkt_tab)
├── .env                             # Token de Hugging Face y opcionalmente DROPBOX_URL
├── pyproject.toml                   # Definición del proyecto y dependencias
├── README.md                        # Este archivo
└── uv.lock                          # Archivo de bloqueo de dependencias para uv
```

---

## 🛠️ Instalación

Este proyecto utiliza **`uv`** como gestor de paquetes y entorno virtual.

1. **Clona el repositorio:**

```bash
git clone <url-del-repositorio>
cd WhisperXAD
```

2. **Crea el entorno virtual y sincroniza las dependencias:**

`uv` leerá `pyproject.toml` y `uv.lock` para instalar todo lo necesario.

```bash
uv sync
```

Esto instala, entre otros:

- `whisperx`
- `faster-whisper`
- `torch`
- `pyannote.audio`
- `nltk`
- `python-dotenv`
- `psutil`

---

## 🔧 Configuración previa

### 1. Token de Hugging Face (requerido para diarización)

La diarización (identificación de hablantes) utiliza modelos de `pyannote.audio` que requieren autenticación.

1. Crea un token de acceso (con permisos de **read**) en tu cuenta de Hugging Face.
2. Crea un archivo `.env` en la **raíz del repositorio** (`WhisperXAD/`) con el contenido:

```ini
HUGGING_FACE_TOKEN="hf_TU_TOKEN_AQUI"
```

Si quieres usar la descarga automática del runner Bash cuando no haya audios locales, agrega también:

```ini
DROPBOX_URL="https://www.dropbox.com/.../archivo.zip?dl=0"
```

3. Acepta los términos de uso de los modelos en Hugging Face (una sola vez por cuenta):

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Si no proporcionas un token válido o no aceptas los términos, **la diarización se omitirá**. El resto del pipeline seguirá funcionando.

### 2. Datos de NLTK (requeridos para alineación)

La alineación necesita los tokenizadores `punkt` y `punkt_tab` de NLTK.

- Estos modelos **ya están incluidos** en el repositorio dentro de `audio_transcription/utilities/nltk_data/`.
- El script principal añade automáticamente esta ruta al `nltk.data.path`.
- No necesitas descargarlos manualmente.

### 3. Carpeta de entrada y organización por sets

La carpeta de entrada que debes crear o poblar dentro del repositorio es:

```text
WhisperXAD/audio_transcription/inputs/audios/
```

Esa carpeta es la **raíz de entrada**, no un path fijo para un único set. Dentro de ella puedes:

- Poner archivos de audio sueltos.
- Crear una carpeta por set, por ejemplo `set_01/`, `set_02/`, `entrevistas_mayo/`.
- Crear subcarpetas adicionales si necesitas organizar por lote, fecha o fuente.

El script procesa esa ruta de forma **recursiva**, así que detectará audios en cualquier subcarpeta bajo `audio_transcription/inputs/audios/`.

Los formatos soportados por el código actual son:

- `.mp3`
- `.wav`
- `.m4a`
- `.flac`

---

## 🚀 Uso (línea de comandos)

Esta rama tiene dos formas de uso.

### Opción A. Runner Bash del repositorio

Ejecuta desde la raíz del proyecto:

```bash
bash audio_transcription/run_transcribe.sh
```

Qué hace:

- Busca audios dentro de `audio_transcription/inputs/audios/`.
- Si encuentra audios allí, procesa toda esa carpeta de forma recursiva.
- Si no encuentra audios y existe `DROPBOX_URL` en `.env`, descarga un ZIP y lo extrae dentro de `audio_transcription/inputs/audios/`.
- Escribe la salida por defecto en `audio_transcription/outputs/transcripciones/`.

### Opción B. Script Python con control explícito

Ejecuta el script `transcribe.py` desde la **raíz del repositorio** usando `uv run`:

```bash
uv run python audio_transcription/scripts/transcribe.py --input_path [RUTA_ENTRADA] --output_dir [RUTA_SALIDA] [OPCIONES...]
```

### Argumentos principales

- `--input_path`: Ruta a un **archivo de audio** individual o a un **directorio** con uno o más sets de audio.
- `--output_dir`: Directorio donde se guardarán los resultados.
- `--gpu_index`: GPU visible para el pipeline. El runner Bash usa el valor configurado en el script.
- `--model_size`: Modelo Whisper. Default: `large-v3`.
- `--language`: Idioma del audio. Default: `es`.
- `--no-diarize`: Desactiva diarización si solo quieres transcripción y alineación.

### Formato de entrada recomendado

- Un set de audio puede ser una carpeta completa dentro de `audio_transcription/inputs/audios/`.
- También puedes agrupar múltiples sets como subcarpetas dentro de esa misma raíz.
- El pipeline recorrerá subdirectorios automáticamente.

Ejemplo:

```text
audio_transcription/inputs/audios/
├── set_01/
│   ├── entrevista_001.wav
│   └── entrevista_002.wav
├── set_02/
│   ├── audio_a.mp3
│   └── audio_b.flac
└── lote_mayo/
    └── bloque_1/
        └── testimonio_01.m4a
```

---

## 📌 Ejemplos de uso

### 1. Procesar todos los audios de `inputs/audios/` con el runner Bash

```bash
bash audio_transcription/run_transcribe.sh
```

- Usa `audio_transcription/inputs/audios/` como raíz de entrada.
- Recorre todos los sets y subcarpetas dentro de esa ruta.
- Escribe resultados en `audio_transcription/outputs/transcripciones/`.

### 2. Procesar un set específico con salida dedicada

```bash
uv run python audio_transcription/scripts/transcribe.py \
  --input_path audio_transcription/inputs/audios/set_01 \
  --output_dir audio_transcription/outputs/set_01 \
  --gpu_index 0
```

- Recomendado cuando quieres separar claramente un set de otro.
- Evita mezclar resultados de distintos lotes en una sola carpeta.

### 3. Procesar varios sets de una sola vez

```bash
uv run python audio_transcription/scripts/transcribe.py \
  --input_path audio_transcription/inputs/audios \
  --output_dir audio_transcription/outputs/lote_completo \
  --gpu_index 0
```

- Procesa todos los archivos compatibles encontrados bajo `audio_transcription/inputs/audios/`.
- Útil si quieres correr todos los sets del repositorio en una sola ejecución.

### 4. Procesar un solo archivo desactivando diarización

```bash
uv run python audio_transcription/scripts/transcribe.py \
  --input_path audio_transcription/inputs/audios/set_01/entrevista_001.wav \
  --output_dir audio_transcription/outputs/prueba_sin_diarizacion \
  --gpu_index 0 \
  --no-diarize
```

- Útil para pruebas rápidas o cuando no quieres depender del token de Hugging Face.

---

## 📤 Formato de salida

Por cada archivo de audio procesado (ej. `mi_audio.wav`), se generan dos archivos en el directorio de salida:

### 1. `mi_audio_full.json`

Archivo JSON detallado que contiene la información completa del pipeline. Incluye, entre otros:

- `segments`: lista de segmentos (generalmente frases u oraciones), cada uno con:
  - `text`
  - `start`, `end` (en segundos)
  - `speaker` (si la diarización se ejecutó)
  - `words`: lista de palabras alineadas con:
    - `word`
    - `start`, `end`
    - `alignment_score` (confianza de alineación)
    - `probability` (confianza semántica del Whisper original)
    - `speaker` (si aplica)

### 2. `mi_audio_words.csv`

CSV a nivel de palabra con estas columnas:

```text
start,end,word,alignment_score,probability,speaker
```

Notas operativas:

- Si ya existe `nombre_base_full.json` en la carpeta de salida, ese audio se considera procesado y se salta.
- El nombre de salida usa solo el nombre base del archivo. Si dos audios distintos tienen el mismo nombre en carpetas diferentes y comparten carpeta de salida, colisionarán.

### Recomendación para múltiples sets

- Usa una carpeta de salida distinta por set cuando quieras evitar colisiones y mantener los resultados separados.
- Si vas a procesar todos los sets juntos, asegúrate de que no existan nombres de archivo repetidos.

---

## 📦 Dependencias principales

- **whisperx:** Librería principal que integra Whisper, alineación y diarización.
- **faster-whisper:** Backend optimizado y eficiente de Whisper (utilizado por `whisperx`).
- **pyannote.audio 3.x:** Librería para la diarización de hablantes.
- **torch:** Framework de deep learning requerido por los modelos.
- **nltk:** Utilizado para la tokenización necesaria en alineación.
- **python-dotenv:** Para cargar el token de Hugging Face desde `.env`.
- **psutil:** Para monitorizar el uso de recursos (CPU/RAM).

---

## 🧠 Notas sobre GPU y CPU

- Si **hay GPU (CUDA)** disponible:
  - ASR (`asr_model`) se ejecuta en `cuda`.
  - Alineación (`align_model`) también se ejecuta en `cuda`.
  - La diarización intenta cargar en `cuda` (si hay token y modelos disponibles).
  - cuDNN se desactiva explícitamente para evitar comportamientos no deterministas.

- Si **no hay GPU disponible**:
  - Todo el pipeline se ejecuta en **CPU** (más lento, pero funcional).
  - El `compute_type` por defecto pasa a ser `int8` para ahorrar memoria.

---

## ✅ Resumen

Este pipeline proporciona una solución completa para:

- Transcribir audio con WhisperX.
- Obtener marcas de tiempo a nivel de palabra.
- Identificar y etiquetar hablantes mediante pyannote 3.x (cuando sea posible).
- Procesar archivos individuales o directorios completos en modo batch.
- Exportar resultados detallados (JSON) y salidas tabulares a nivel de palabra (CSV).

La combinación de `uv` + `pyproject.toml` + `uv.lock` garantiza reproducibilidad del entorno, mientras que `audio_transcription/scripts/transcribe.py` y `audio_transcription/run_transcribe.sh` proporcionan una base reutilizable para automatizaciones, APIs o interfaces mayores.

## Autores / Mantenedores

- Autor original del repositorio: Camilo Gutiérrez (GitHub: @cygnusbarrett)
- Refactor del pipeline WhisperX, integración GPU en todas las etapas y documentación extendida:
  Guillermo Peralta (GitHub: @guillermo1205ad)

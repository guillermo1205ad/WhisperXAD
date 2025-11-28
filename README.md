# NuestroMemorIA – Pipeline de Transcripción de Audio con WhisperX (GPU + Diarización pyannote 3.x)

Este proyecto define un **pipeline de transcripción, alineación y diarización** basado en:

- **WhisperX** (ASR + alineación, usando *faster-whisper* como backend optimizado).
- **PyTorch 2.4.x** con GPU (CUDA) cuando está disponible.
- **pyannote.audio 3.x** (diarización moderna, vía `pyannote/speaker-diarization-3.1`).
- Integración mediante la clase `WhisperXPipeline` en `audio_transcription/scripts/pipeline.py`.

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

- `WhisperXPipeline` — clase principal del pipeline (`audio_transcription/scripts/pipeline.py`).
- `transcribe_whisperx.py` — script de entrada (CLI) que invoca el pipeline.

---

## ⚙️ Características principales

- **ASR de alta calidad:** usa `large-v3` por defecto, ejecutándose en GPU (`cuda`) si está disponible.
- **Timestamps a nivel de palabra:** gracias al módulo de alineación de WhisperX (`load_align_model`).
- **Diarización de hablantes (pyannote 3.x):**
  - Utiliza el modelo `pyannote/speaker-diarization-3.1`.
  - Intenta correr en **GPU** cuando hay CUDA disponible.
  - Si no se puede (falta de token, problemas de descarga, etc.), el pipeline **continúa sin diarización** en lugar de fallar.
- **Procesamiento por lotes:** puede recorrer recursivamente un directorio y procesar todos los audios compatibles.
- **Optimización de recursos:**
  - Carga los modelos **una sola vez** para procesar múltiples archivos.
  - Monitoriza uso de **CPU/RAM** con `psutil`.
  - Desactiva **cuDNN** explícitamente para evitar comportamientos no deterministas o conflictos.
- **Configuración en código y/o CLI:** se pueden ajustar modelo ASR, idioma, `batch_size`, `compute_type`, etc.
- **Salidas en JSON y TXT:** un archivo detallado y otro simple por cada audio.
- **Replicabilidad:** el entorno está definido en `pyproject.toml` y se instala mediante `uv sync`.

---

## 📁 Estructura del proyecto

```text
NuestraMemorIA-developments/
├── audio_transcription/
│   ├── inputs/
│   │   └── audios/                  # Carpeta donde se colocan los audios de entrada
│   ├── logs/                        # (Opcional) logs de ejecución
│   ├── outputs/
│   │   └── transcripciones/         # Carpeta destino para los resultados
│   ├── scripts/
│   │   ├── pipeline.py              # Clase principal WhisperXPipeline
│   │   └── transcribe_whisperx.py   # Script CLI de entrada
│   └── utilities/
│       └── nltk_data/               # Datos de NLTK (ej. punkt, punkt_tab)
├── .env                             # Token de Hugging Face (no se versiona)
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
cd NuestraMemorIA-developments
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
2. Crea un archivo `.env` en la **raíz del repositorio** (`NuestraMemorIA-developments/`) con el contenido:

```ini
HUGGING_FACE_TOKEN="hf_TU_TOKEN_AQUI"
```

3. Acepta los términos de uso de los modelos en Hugging Face (una sola vez por cuenta):

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Si no proporcionas un token válido o no aceptas los términos, **la diarización se omitirá**. El resto del pipeline seguirá funcionando.

### 2. Datos de NLTK (requeridos para alineación)

La alineación necesita los tokenizadores `punkt` y `punkt_tab` de NLTK.

- Estos modelos **ya están incluidos** en el repositorio dentro de `audio_transcription/utilities/nltk_data/`.
- `WhisperXPipeline` añade automáticamente esta ruta al `nltk.data.path`.
- No necesitas descargarlos manualmente.

---

## 🚀 Uso (línea de comandos)

Ejecuta el script `transcribe_whisperx.py` desde la **raíz del repositorio** (`NuestraMemorIA-developments/`) usando `uv run`:

```bash
uv run python audio_transcription/scripts/transcribe_whisperx.py [RUTA_ENTRADA] [OPCIONES...]
```

### Argumento obligatorio

- `RUTA_ENTRADA`: Ruta a un **archivo de audio** individual (ej. `audio_transcription/inputs/audios/audio1.wav`) o a un **directorio** que contenga múltiples archivos de audio (ej. `audio_transcription/inputs/audios/`).  
  El script procesará formatos comunes: `.wav`, `.mp3`, `.mp4`, `.m4a`, `.flac`, `.aac`, `.ogg`, `.wma`.

### Argumentos opcionales principales

- `-o, --output_dir`: Directorio donde se guardarán los resultados.  
  - *Default:* `audio_transcription/outputs/transcripciones`
- `-l, --language`: Código de idioma del audio (ISO 639-1).  
  - *Default:* `es`
- `--asr_model`: Modelo ASR de Whisper a utilizar.  
  - *Default:* `large-v3`  
  - *Opciones típicas:* `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` (con posibles sufijos `.en` para modelos solo en inglés).
- `--batch_size`: Número de fragmentos de audio a procesar en paralelo por el ASR. Reduce si tienes poca RAM/VRAM.  
  - *Default:* `16`
- `--compute_type`: Precisión numérica para el modelo ASR.
  - Si **no** se especifica (`None`), el pipeline hace:
    - GPU (`cuda`): `float32`
    - CPU: `int8`
  - Puedes forzar otro valor con este argumento.  
  - *Opciones típicas:* `int8`, `float16`, `float32`.

> **Nota:** `compute_type="float32"` ofrece la **máxima precisión**, pero consume más VRAM y es más lento. `int8` y `float16` son más ligeros y rápidos, a costa de algo de precisión.

---

## 📌 Ejemplos de uso

### 1. Procesar todos los audios de un directorio (configuración por defecto)

```bash
uv run python audio_transcription/scripts/transcribe_whisperx.py   audio_transcription/inputs/audios/
```

- Usará:
  - `asr_model = "large-v3"`
  - `language = "es"`
  - `batch_size = 16`
  - `compute_type`:
    - `float32` si hay GPU (CUDA)
    - `int8` si solo hay CPU
- Guardará los resultados en: `audio_transcription/outputs/transcripciones/`.

---

### 2. Procesar un solo archivo, especificando salida e idioma inglés

```bash
uv run python audio_transcription/scripts/transcribe_whisperx.py   "audio_transcription/inputs/audios/mi_audio_en.mp3"   -o "audio_transcription/outputs/ingles_results"   -l "en"
```

- Cambia el idioma a inglés (`en`).
- Escribe los resultados del archivo en la carpeta `outputs/ingles_results`.

---

### 3. Procesar un directorio con un modelo más pequeño y menor batch size

```bash
uv run python audio_transcription/scripts/transcribe_whisperx.py   audio_transcription/inputs/audios/   --asr_model "medium"   --batch_size 4   --compute_type "float16"
```

- Usa el modelo `medium` (más ligero que `large-v3`).
- Reduce el `batch_size` para ahorrar memoria.
- Usa `float16` para un compromiso razonable entre precisión y VRAM.

---

## 📤 Formato de salida

Por cada archivo de audio procesado (ej. `mi_audio.wav`), se generan dos archivos en el directorio de salida:

### 1. `mi_audio_completo.json`

Archivo JSON detallado que contiene la información completa del pipeline, típicamente con campos como:

- `segments`: lista de segmentos (generalmente frases u oraciones), cada uno con:
  - `text`
  - `start`, `end` (en segundos)
  - `speaker` (si la diarización se ejecutó)
  - `words`: lista de palabras alineadas con:
    - `word`
    - `start`, `end`
    - `score` (confianza de alineación)
    - `speaker` (si aplica)
- Posibles campos adicionales según la versión de WhisperX.

### 2. `mi_audio_simple.txt`

Archivo de texto plano, pensado para lectura rápida, con el formato:

```text
[SPEAKER_00]: Texto del primer segmento hablado por el hablante 0.
[SPEAKER_01]: Texto del segmento hablado por el hablante 1.
[SPEAKER_00]: Continuación del hablante 0.
...
```

- Si la diarización **no** se ejecutó (o falló), se utilizará `[HABLANTE_DESCONOCIDO]` como etiqueta.

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
- Exportar resultados detallados (JSON) y resúmenes legibles (TXT).

La combinación de `uv` + `pyproject.toml` + `uv.lock` garantiza reproducibilidad del entorno, mientras que `WhisperXPipeline` encapsula la lógica para que puedas reutilizarla desde otros scripts o integrarla en aplicaciones mayores (por ejemplo, una API o una interfaz web).

## Autores / Mantenedores

- Autor original del repositorio: Camilo Gutiérrez (GitHub: @cygnusbarrett)
- Refactor del pipeline WhisperX, integración GPU en todas las etapas y documentación extendida:
  Guillermo Peralta (GitHub: @guillermo1205ad)

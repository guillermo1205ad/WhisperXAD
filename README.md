

# NuestroMemorIA - Pipeline de Transcripción de Audio con WhisperX

Este repositorio contiene un pipeline para la transcripción automática de audio, alineación de palabras y diarización de hablantes, utilizando los recursos de `whisperx` como base. Ha sido diseñado para ser flexible, reutilizable y eficiente, especialmente para el procesamiento por lotes de archivos de audio.

## Descripción

El objetivo principal de este proyecto es procesar archivos de audio (individuales o en lotes) para generar transcripciones con marcas de tiempo a nivel de palabra y asignación de hablantes. El pipeline integra tres tecnologías:

1.  **Reconocimiento Automático de Voz (ASR):** Utiliza `faster-whisper` (backend optimizado de Whisper) para la transcripción inicial del audio. Por defecto, emplea el modelo `large-v3`.
2.  **Alineación Forzada:** Emplea modelos `wav2vec2` (a través de `whisperx`) para alinear la transcripción con el audio, obteniendo timestamps precisos para cada palabra.
3.  **Diarización de Hablantes:** Utiliza `pyannote.audio` para identificar los diferentes hablantes en el audio y asignar etiquetas (ej. `SPEAKER_00`, `SPEAKER_01`) a cada palabra.

El pipeline ha sido encapsulado en una clase reutilizable (`WhisperXPipeline`) y expuesto a través de un script de línea de comandos (`transcribe_whisperx.py`) para facilitar su uso.

## Características Principales

  * **Precisión:** Aprovecha `whisperx` y el modelo `large-v3` por defecto.
  * **Timestamps a Nivel de Palabra:** Genera marcas de tiempo precisas gracias a la alineación forzada.
  * **Identificación de Hablantes:** Diferencia y etiqueta quién habló y cuándo.
  * **Procesamiento por Lotes:** Capaz de procesar directorios completos de archivos de audio de forma eficiente.
  * **Optimizado para Recursos:** Carga los modelos una sola vez para procesar lotes, ahorrando tiempo y memoria. Incluye monitorización de uso de CPU/RAM.
  * **Configurable:** Permite ajustar modelos, idioma, batch size y otros parámetros vía CLI.
  * **Manejo de Apple Silicon:** Incluye lógica específica para compatibilidad con GPUs de Apple (MPS), forzando el ASR a CPU para evitar conflictos conocidos.
  * **Salidas Flexibles:** Genera un archivo `.json` detallado y un archivo `.txt` simple por cada audio procesado.
  * **Gestión de Dependencias:** Utiliza `uv` para una gestión rápida y eficiente del entorno virtual y las dependencias.

## Estructura del Proyecto

```
NuestraMemorIA-developments/
├── audio_transcription/
│   ├── inputs/                     # Directorio para colocar los audios de entrada
│   │   └── audios/                 # (Ejemplo de subdirectorio)
│   ├── logs/                       # (Opcional) Directorio para guardar logs detallados
│   ├── outputs/                    # Directorio donde se guardan los resultados
│   │   └── transcripciones/        # (Ejemplo, configurable vía CLI)
│   ├── scripts/
│   │   ├── pipeline.py             # Clase principal WhisperXPipeline 
│   │   └── transcribe_whisperx.py  # Script de entrada para la línea de comandos (CLI)
│   └── utilities/
│       └── nltk_data/              # Datos necesarios para NLTK (punkt, punkt_tab)
├── .env                            # Archivo para el token de Hugging Face
├── pyproject.toml                  # Definición del proyecto y dependencias
├── README.md                       # Este archivo
└── uv.lock                         # Archivo de bloqueo de dependencias para uv
```

## Instalación

Este proyecto utiliza `uv` como gestor de paquetes y entorno virtual.

1.  **Clona el repositorio:**
    ```bash
    git clone <url-del-repositorio>
    cd NuestraMemorIA-developments
    ```
2.  **Crea el entorno virtual y sincroniza las dependencias:**
    `uv` leerá `pyproject.toml` y `uv.lock` para instalar todo lo necesario de forma rápida.
    ```bash
    uv sync
    ```
    Esto instala `whisperx`, `torch`, `pyannote.audio`, `faster-whisper`, `nltk`, `python-dotenv`, `psutil` y sus dependencias.

## Configuración

Antes de ejecutar el pipeline, considera estos puntos:

1.  **Token de Hugging Face (Requerido para Diarización):**

      * La diarización (identificación de hablantes) utiliza modelos de `pyannote.audio` que requieren autenticación.
      * Crea un token de acceso (con permisos de `read`) en tu [configuración de Hugging Face](https://huggingface.co/settings/tokens).
      * Crea un archivo llamado `.env` en la **raíz del repositorio** (`NuestraMemorIA-developments/`) con el siguiente contenido:
        ```ini
        HUGGING_FACE_TOKEN="hf_TU_TOKEN_AQUI"
        ```
      * **Importante:** Asegúrate de **aceptar los términos de uso** en las páginas de los modelos de `pyannote` en Hugging Face (requerido una sola vez por cuenta):
          * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
          * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
      * Si no proporcionas un token válido o no aceptas los términos, el paso de diarización se omitirá.

2.  **Datos de NLTK (Requerido para Alineación):**

      * La alineación necesita los tokenizadores `punkt` y `punkt_tab` de NLTK.
      * Estos modelos **están incluidos** en el repositorio dentro de `audio_transcription/utilities/nltk_data/`. El script está configurado para encontrarlos allí automáticamente. No necesitas descargarlos manualmente.

## Uso (Línea de Comandos)

Ejecuta el script `transcribe_whisperx.py` desde la **raíz del repositorio** (`NuestraMemorIA-developments/`) usando `uv run`.

**Sintaxis Básica:**

```bash
uv run python audio_transcription/scripts/transcribe_whisperx.py [RUTA_ENTRADA] [OPCIONES...]
```

**Argumento Obligatorio:**

  * `RUTA_ENTRADA`: Ruta a un **archivo de audio** individual (ej. `inputs/audios/audio1.wav`) o a un **directorio** que contenga múltiples archivos de audio (ej. `inputs/audios/`). El script procesará todos los formatos comunes (`.wav`, `.mp3`, `.mp4`, `.m4a`, etc.).

**Argumentos Opcionales:**

  * `-o, --output_dir`: Directorio donde se guardarán los resultados.
      * *Default:* `audio_transcription/outputs/transcripciones`
  * `-l, --language`: Código de idioma del audio (ISO 639-1).
      * *Default:* `es`
  * `--asr_model`: Modelo ASR de Whisper a utilizar. `large-v3` es el más preciso.
      * *Default:* `large-v3`
      * *Opciones:* `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` (pueden tener sufijos `.en` para modelos solo en inglés).
  * `--batch_size`: Número de segmentos de audio a procesar en paralelo por el ASR. Reduce si tienes poca RAM/VRAM.
      * *Default:* `8`
  * `--compute_type`: Precisión numérica para el modelo ASR. `int8` es rápido y eficiente en memoria.
      * *Default:* `int8`
      * *Opciones:* `int8`, `float16`, `float32`.

**Ejemplos:**

1.  **Procesar todos los audios en un directorio (configuración por defecto):**

    ```bash
    uv run python audio_transcription/scripts/transcribe_whisperx.py audio_transcription/inputs/audios/
    ```

      * Usará `large-v3`, idioma `es`, batch size 8, compute `int8`.
      * Guardará los resultados en `audio_transcription/outputs/transcripciones/`.

2.  **Procesar un solo archivo, especificando salida e idioma inglés:**

    ```bash
    uv run python audio_transcription/scripts/transcribe_whisperx.py "audio_transcription/inputs/audios/mi_audio_en.mp3" -o "audio_transcription/outputs/ingles_results" -l "en"
    ```

3.  **Procesar un directorio usando un modelo más pequeño y batch size reducido:**

    ```bash
    uv run python audio_transcription/scripts/transcribe_whisperx.py audio_transcription/inputs/audios/ --asr_model "medium" --batch_size 4
    ```

## Formato de Salida

Por cada archivo de audio procesado (ej. `mi_audio.wav`), se generarán dos archivos en el directorio de salida especificado:

1.  **`mi_audio_completo.json`:**

      * Un archivo JSON detallado que contiene toda la información del pipeline:
          * `segments`: Lista de segmentos (generalmente oraciones), cada uno con `text`, `start`, `end`, `speaker` (si la diarización se ejecutó) y una lista `words`.
          * `word_segments`: Lista plana de todas las palabras detectadas, cada una con `word`, `start`, `end`, `score` (confianza de alineación) y `speaker` (si aplica).

2.  **`mi_audio_simple.txt`:**

      * Un archivo de texto plano simple, ideal para lectura rápida, con el formato:
        ```
        [SPEAKER_00]: Texto del primer segmento hablado por el hablante 0.
        [SPEAKER_01]: Texto del segmento hablado por el hablante 1.
        [SPEAKER_00]: Continuación del hablante 0.
        ...
        ```
      * Si la diarización no se ejecutó (o falló), mostrará `[HABLANTE_DESCONOCIDO]` en lugar de `[SPEAKER_XX]`.

## Dependencias 

  * **whisperx:** Librería principal que integra Whisper, alineación y diarización.
  * **faster-whisper:** Backend optimizado para Whisper (usado por `whisperx`).
  * **pyannote.audio:** Librería para la diarización de hablantes.
  * **torch:** Framework de deep learning (requerido por todos los modelos).
  * **nltk:** Utilizado para la tokenización de oraciones en la alineación.
  * **python-dotenv:** Para cargar el token de Hugging Face desde `.env`.
  * **psutil:** Para monitorizar el uso de recursos (CPU/RAM).

## Notas sobre Apple Silicon (Mac M-Series)

El pipeline incluye una lógica específica para detectar si se está ejecutando en un Mac con chip M-series (usando MPS). Debido a incompatibilidades entre `faster-whisper` y `pyannote.audio` respecto al identificador del dispositivo (`"auto"` vs `"mps"`), el script **forzará la ejecución del modelo ASR (Paso 1) en la CPU**. Los pasos de Alineación (Paso 2) y Diarización (Paso 3) sí utilizarán la GPU (MPS) para un rendimiento óptimo. Esto asegura la compatibilidad sin sacrificar demasiado rendimiento, ya que los pasos 2 y 3 son significativamente más rápidos que el ASR.
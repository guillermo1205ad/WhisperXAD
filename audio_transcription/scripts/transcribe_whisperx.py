#!/usr/bin/env python3
"""
Script de entrada (CLI) para ejecutar el WhisperXPipeline.

Ejemplos de uso (desde la raíz del repo):

  # Procesar todos los audios del directorio por defecto
  uv run python audio_transcription/scripts/transcribe_whisperx.py audio_transcription/inputs/audios/

  # Procesar un solo archivo, cambiando salida e idioma
  uv run python audio_transcription/scripts/transcribe_whisperx.py \
      audio_transcription/inputs/audios/mi_audio.mp3 \
      -o audio_transcription/outputs/transcripciones_en \
      -l en
"""

import argparse
import logging
import os
import sys

# Aseguramos que el proyecto raíz esté en sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from audio_transcription.scripts.pipeline import WhisperXPipeline, PipelineConfig
except ImportError as e:
    print("Error: No se pudo importar 'audio_transcription.scripts.pipeline'.")
    print("Asegúrate de ejecutar este script desde la raíz del proyecto, por ejemplo:")
    print("  uv run python audio_transcription/scripts/transcribe_whisperx.py audio_transcription/inputs/audios/")
    print(f"Detalle del error: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de transcripción + alineación + diarización con WhisperX."
    )

    # Argumento posicional: ruta de entrada (archivo o carpeta)
    parser.add_argument(
        "input_path",
        type=str,
        help=(
            "Ruta a un archivo de audio/video o a un directorio que contenga "
            "múltiples archivos (ej: audio_transcription/inputs/audios/)."
        ),
    )

    # Directorio de salida
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="audio_transcription/outputs/transcripciones",
        help="Directorio donde se guardarán las transcripciones (JSON/TXT).",
    )

    # Idioma (código ISO 639-1)
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="es",
        help="Código de idioma del audio (ej: es, en, pt). Default: es.",
    )

    # Modelo ASR
    parser.add_argument(
        "--asr_model",
        type=str,
        default="large-v3",
        help=(
            "Nombre del modelo WhisperX/faster-whisper a utilizar "
            "(ej: tiny, base, small, medium, large-v2, large-v3). "
            "Default: large-v3."
        ),
    )

    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Tamaño de batch para la transcripción ASR. Default: 16.",
    )

    # compute_type
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        choices=[None, "int8", "int8_float16", "float16", "float32"],
        help=(
            "Tipo de precisión para ASR (faster-whisper). Si se deja vacío, "
            "el pipeline decide automáticamente (float32 en GPU, int8 en CPU)."
        ),
    )

    return parser


def main() -> None:
    # --- Parsing de argumentos ---
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- Logging básico ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== Iniciando transcribe_whisperx ===")
    logger.info("input_path   = %s", args.input_path)
    logger.info("output_dir   = %s", args.output_dir)
    logger.info("language     = %s", args.language)
    logger.info("asr_model    = %s", args.asr_model)
    logger.info("batch_size   = %d", args.batch_size)
    logger.info("compute_type = %s", args.compute_type)

    # Nos aseguramos de que el directorio de salida exista
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Construir configuración del pipeline ---
    config_kwargs = {
        "language": args.language,
        "asr_model": args.asr_model,
        "batch_size": args.batch_size,
    }
    if args.compute_type is not None:
        config_kwargs["compute_type"] = args.compute_type

    config = PipelineConfig(**config_kwargs)

    # --- Inicializar pipeline ---
    pipeline = WhisperXPipeline(config)

    # --- Ejecutar transcripción en batch ---
    pipeline.transcribe_batch(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
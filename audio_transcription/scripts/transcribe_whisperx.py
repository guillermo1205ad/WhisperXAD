import os
import logging
import argparse
import sys

# Añade el directorio raíz del proyecto (NuestraMemorIA-developments) al sys.path
# para permitir importaciones relativas como 'from audio_transcription.pipeline...'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

try:
    from audio_transcription.scripts.pipeline import WhisperXPipeline, PipelineConfig
except ImportError:
    print(f"Error: No se pudo importar 'audio_transcription.pipeline'.")
    print(f"Asegúrate de que el script se está ejecutando desde la raíz del proyecto o que '{project_root}' está en tu PYTHONPATH.")
    sys.exit(1)


# --- Configuración del Logging ---
# Configura el logging en el script de entrada para capturar logs
# tanto de este script como del módulo 'pipeline' importado.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    """
    Analiza los argumentos de la línea de comandos para crear un objeto 
    PipelineConfig, inicializa el WhisperXPipeline y comienza el
    procesamiento por lotes.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe, alinea y diariza archivos de audio con WhisperX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Argumentos de Entrada/Salida ---
    parser.add_argument(
        "input_path",
        type=str,
        help="Ruta a un archivo de audio (ej. 'audio.wav') o un directorio que contenga archivos de audio (ej. 'mis_audios/')."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="audio_transcription/outputs/transcripciones",
        help="Directorio donde se guardarán los archivos .json y .txt resultantes."
    )
    
    # --- Argumentos de Configuración de Tarea ---
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="es",
        help="Código de idioma del audio (ej. 'es', 'en', 'fr')."
    )
    parser.add_argument(
        "--asr_model",
        type=str,
        default="large-v3",
        help="Nombre del modelo ASR de Whisper (ej. 'large-v3')."
    )
    
    # --- Argumentos de Configuración de Hardware ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamaño del lote para la transcripción. Reducir si hay poca VRAM/RAM."
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="int8",
        help="Tipo de cómputo para el modelo ASR (ej. 'int8', 'float16', 'float32'). 'int8' es más rápido y ligero."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Inicializando configuración del pipeline...")

    # Crear la configuración a partir de los argumentos de la CLI
    # Los dispositivos (device_asr, device_torch) y hf_token se
    # determinarán automáticamente dentro de PipelineConfig.
    try:
        config = PipelineConfig(
            language_code=args.language,
            asr_model_name=args.asr_model,
            batch_size=args.batch_size,
            compute_type=args.compute_type
        )
    except Exception as e:
        logger.critical(f"Error al inicializar la configuración del pipeline: {e}")
        return

    # Inicializar el pipeline con la configuración
    try:
        pipeline = WhisperXPipeline(config)
    except Exception as e:
        logger.critical(f"No se pudo inicializar el pipeline (¿error al cargar modelos?): {e}")
        return

    # Ejecutar el procesamiento por lotes
    try:
        pipeline.transcribe_batch(
            input_path=args.input_path,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.critical(f"Error fatal durante la ejecución del lote: {e}")
    finally:
        # Asegura la descarga de modelos incluso si el lote falla
        pipeline._unload_models()

if __name__ == "__main__":
    main()


"""
EJEMPLO DE EJECUCIÓN:
-----------------------------------------------------

1. Procesar una carpeta completa (usando valores por defecto):

   $ uv run python audio_transcription/scripts/transcribe_whisperx.py audio_transcription/inputs/audios/

2. Procesar un solo archivo con opciones personalizadas:

   $ uv run python audio_transcription/scripts/transcribe_whisperx.py "ruta/al/audio.mp3" \
       -o "mi/carpeta/salida" \
       -l "en" \
       --asr_model "large-v2" \
       --batch_size 4

ARGUMENTOS OPCIONALES:
---------------------

-o, --output_dir : Directorio donde se guardarán los archivos .json y .txt.
                   (Default: audio_transcription/outputs/transcripciones)

-l, --language : Código de idioma del audio (ej. 'es', 'en', 'fr').
                 (Default: es)

--asr_model : Nombre del modelo ASR de Whisper (ej. 'large-v3', 'base').
              (Default: large-v3)

--batch_size : Tamaño del lote. Reducir si hay poca VRAM/RAM.
               (Default: 8)

--compute_type : Precisión del modelo ASR (ej. 'int8', 'float16').
                 (Default: int8)
"""



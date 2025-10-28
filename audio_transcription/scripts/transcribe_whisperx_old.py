import whisperx
import torch
import gc
import os
import json
import nltk
import logging
import psutil
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# carga las variables de entorno desde un archivo .env en la raíz del proyecto.
load_dotenv(override=True)

# añade la carpeta 'utilities/nltk_data' del proyecto a las rutas de búsqueda de NLTK.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_audio = os.path.dirname(script_dir)
custom_nltk_path = os.path.join(project_root_audio, "utilities", "nltk_data")
logger.info(f"Añadiendo ruta de NLTK: {custom_nltk_path}")
nltk.data.path.append(custom_nltk_path)


def log_resource_usage():
    """
    Registra el uso actual de CPU y memoria RAM en el log.

    Utiliza psutil para obtener métricas de uso y las reporta
    a nivel de INFO.
    """
    logger.info(f"Uso de CPU: {psutil.cpu_percent(interval=None)}% | Uso de Memoria: {psutil.virtual_memory().percent}%")


def main():
    """
    Orquesta el pipeline completo de transcripción, alineación y diarización.

    El pipeline sigue estos pasos:
    1.  Configura los dispositivos (ASR en CPU, Torch en MPS/GPU).
    2.  Carga el token de Hugging Face desde el entorno.
    3.  Inicializa el monitor de recursos (psutil).
    4.  Carga el archivo de audio.
    5.  Paso 1: Transcribe con ASR (large-v3 en CPU).
    6.  Paso 2: Alinea la transcripción (en MPS/GPU).
    7.  Paso 3: Diariza los hablantes (en MPS/GPU), si el token está presente.
    8.  Guarda los resultados en archivos JSON y TXT.
    """

    # --- Configuración de archivos y hardware ---

    AUDIO_FILE = "audio_transcription/inputs/one_audio/D 394 caja 6 cinta 1 Osvaldo Muray lado B-01.5_2.wav"
    OUTPUT_DIR = "audio_transcription/outputs/audio_transcripcion"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configuración específica para Apple Silicon (Mac M-series)
    if torch.backends.mps.is_available():
        # NOTA DE COMPATIBILIDAD:
        # `whisperx.load_model` carga tanto ASR (faster-whisper) como VAD (pyannote/torch).
        # `faster-whisper` requiere "auto" para usar la GPU de Apple (Metal).
        # `torch` (VAD) requiere "mps" y falla con "auto".
        # Dado que `whisperx` usa el mismo parámetro `device` para ambos,
        # forzamos el ASR (Paso 1) a "cpu" para evitar el conflicto.
        # Los Pasos 2 y 3 (Alineación y Diarización) usarán "mps" (GPU).
        device_asr = "cpu"
        device_torch = "mps"
        logger.info(f"MPS detectado. Usando '{device_asr}' para ASR y '{device_torch}' para Alignment/Diarization.")
    else:
        logger.warning("MPS (GPU) no está disponible, usando CPU. La transcripción será MUY lenta.")
        device_asr = "cpu"
        device_torch = "cpu"

    
    compute_type = "int8" # tipo de cómputo cuantizado, más ligero y eficiente para `large-v3` en CPUs o GPUs 
    batch_size = 8 # batch size conservador para 16GB de RAM.

    # --- Configuración para la ejecución ---

    language_code = "es"
    asr_model_name = "large-v3"

    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if hf_token is None:
        logger.warning(
            "¡ADVERTENCIA! No se encontró la variable de entorno HUGGING_FACE_TOKEN.\n"
            "La diarización (paso 3) fallará.\n"
            "Por favor, añade tu token de HF (con permisos de 'read') a tus "
            "variables de entorno o pégalo directamente en el script."
        )

    logger.info("--- Iniciando pipeline de WhisperX ---")
    logger.info(f"Audio: {AUDIO_FILE}")
    logger.info(f"Dispositivo ASR: {device_asr} | Dispositivo Torch: {device_torch} | Compute Type: {compute_type} | Batch Size: {batch_size}")

    psutil.cpu_percent(interval=None) # inicializa psutil para la primera medición de uso de CPU


    # --- Inicio del pipeline ---

    logger.info("Cargando audio...")
    try:
        audio = whisperx.load_audio(AUDIO_FILE)
        logger.info("Audio cargado exitosamente.")
    except Exception as e:
        logger.error(f"Error cargando el audio: {e}")
        exit()

    logger.info(f"[Paso 1/3] Cargando modelo ASR: {asr_model_name}...")
    model = whisperx.load_model(
        asr_model_name,
        device_asr,
        compute_type=compute_type,
        language=language_code
    )
    log_resource_usage()

    logger.info("Transcribiendo audio (ASR)...")
    result = model.transcribe(audio, batch_size=batch_size)
    log_resource_usage()
    logger.info("Transcripción completada.")

    logger.info("Descargando modelo ASR de la memoria...")
    del model
    gc.collect()
    if device_torch == "mps":
        torch.mps.empty_cache()

    logger.info(f"[Paso 2/3] Cargando modelo de alineación para '{language_code}'...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device_torch
    )
    log_resource_usage()

    logger.info("Alineando transcripción (Alignment)...")
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device_torch,
        return_char_alignments=False
    )
    log_resource_usage()
    logger.info("Alineación completada.")

    logger.info("Descargando modelo de alineación de la memoria...")
    del model_a
    del metadata
    gc.collect()
    if device_torch == "mps":
        torch.mps.empty_cache()

    logger.info("[Paso 3/3] Ejecutando diarización...")
    if hf_token:
        try:
            logger.info("Cargando modelo de diarización (pyannote)...")
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=device_torch
            )
            log_resource_usage()

            logger.info("Ejecutando diarización de hablantes...")
            diarize_segments = diarize_model(audio)
            log_resource_usage()

            logger.info("Asignando hablantes a las palabras...")
            result = assign_word_speakers(diarize_segments, result)
            logger.info("Diarización completada.")
        except Exception as e:
            logger.error(f"Error durante la diarización: {e}")
            logger.warning("El resultado final NO incluirá hablantes.")
    else:
        logger.warning("Saltando diarización (no se proveyó token de HF).")

    logger.info("--- Proceso completado. Guardando resultados... ---")

    base_filename = os.path.splitext(os.path.basename(AUDIO_FILE))[0]

    json_output_path = os.path.join(OUTPUT_DIR, base_filename + "_completo.json")
    logger.info(f"Guardando resultado JSON completo en: {json_output_path}")
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error al guardar el JSON: {e}")

    txt_output_path = os.path.join(OUTPUT_DIR, base_filename + "_simple.txt")
    logger.info(f"Guardando transcripción TXT simple en: {txt_output_path}")
    try:
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            if "segments" in result:
                for segment in result["segments"]:
                    speaker = segment.get("speaker", "HABLANTE_DESCONOCIDO")
                    text = segment["text"].strip()
                    f.write(f"[{speaker}]: {text}\n")
            else:
                f.write("No se encontraron segmentos en la transcripción.")
    except Exception as e:
        logger.error(f"Error al guardar el TXT: {e}")

    logger.info("¡Script finalizado!")


if __name__ == "__main__":
    main()

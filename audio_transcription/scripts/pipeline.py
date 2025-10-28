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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

load_dotenv(override=True)

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """
    Configuración para el WhisperXPipeline, agrupando parámetros.
    Esta clase almacena todas las configuraciones de hardware y de tarea
    necesarias para inicializar y ejecutar el pipeline.
    """

    language_code: str = "es"
    asr_model_name: str = "large-v3"
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HUGGING_FACE_TOKEN"))
    
    batch_size: int = 8 # numero de segmentos de audio que se procesan en paralelo en la CPU/GPU
    compute_type: str = "int8"
    device_asr: Optional[str] = None
    device_torch: Optional[str] = None

    def __post_init__(self):
        """
        Determina automáticamente los dispositivos de hardware si no se proporcionan.
        """
        if self.device_asr is None or self.device_torch is None:
            if torch.backends.mps.is_available():
                # Forzamos ASR a CPU por compatibilidad de whisperx/pyannote en MPS
                self.device_asr = "cpu"
                self.device_torch = "mps"
                logger.info(f"MPS detectado. Usando '{self.device_asr}' para ASR y '{self.device_torch}' para Alignment/Diarization.")
            else:
                # Fallback a CPU si no hay GPU de Apple
                self.device_asr = "cpu"
                self.device_torch = "cpu"
                logger.warning("MPS (GPU) no está disponible, usando CPU. La transcripción será MUY lenta.")
        
        if self.hf_token is None:
            logger.warning(
                "¡ADVERTENCIA! No se encontró HUGGING_FACE_TOKEN.\n"
                "La diarización (paso 3) fallará para todos los archivos.\n"
                "Asegúrate de tener un .env o variables de entorno."
            )

class WhisperXPipeline:
    """
    Encapsula el pipeline completo de WhisperX (ASR, Alineación, Diarización).

    Esta clase carga los modelos una sola vez al inicializarse (basado en 
    PipelineConfig) y los reutiliza para procesar múltiples archivos,
    optimizando significativamente el rendimiento en lotes.
    """

    def __init__(self, config: PipelineConfig):
        """
        Inicializa el pipeline y carga todos los modelos necesarios en memoria.

        :param config: Objeto PipelineConfig con todos los parámetros.
        """
        self.config = config
        self.asr_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        
        self._setup_dependencies()
        self._load_models()

    def _setup_dependencies(self):
        """Configura dependencias externas como NLTK y dotenv."""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        custom_nltk_path = os.path.join(os.path.dirname(script_dir), "utilities", "nltk_data")
        
        if os.path.exists(custom_nltk_path):
            logger.info(f"Añadiendo ruta de NLTK personalizada: {custom_nltk_path}")
            nltk.data.path.append(custom_nltk_path)
        else:
            logger.warning(f"No se encontró la ruta de NLTK: {custom_nltk_path}. Se usará la ruta por defecto.")

    def _load_models(self):
        """Carga los modelos ASR, Alineación y Diarización en memoria."""
        try:
            logger.info(f"[Paso 1/3] Cargando modelo ASR: {self.config.asr_model_name}...")
            self.asr_model = whisperx.load_model(
                self.config.asr_model_name,
                self.config.device_asr,
                compute_type=self.config.compute_type,
                language=self.config.language_code
            )
            
            logger.info(f"[Paso 2/3] Cargando modelo de alineación para '{self.config.language_code}'...")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.config.language_code,
                device=self.config.device_torch
            )
            
            if self.config.hf_token:
                logger.info("[Paso 3/3] Cargando modelo de diarización (pyannote)...")
                self.diarize_model = DiarizationPipeline(
                    use_auth_token=self.config.hf_token,
                    device=self.config.device_torch
                )
            else:
                logger.warning("Saltando carga del modelo de diarización (no se proveyó token de HF).")
            
            logger.info("Todos los modelos han sido cargados exitosamente.")
            
        except Exception as e:
            logger.critical(f"Error fatal cargando modelos: {e}")
            raise

    def _log_resource_usage(self):
        """Registra el uso actual de CPU y memoria RAM."""
        logger.info(f"Uso de CPU: {psutil.cpu_percent(interval=None)}% | Uso de Memoria: {psutil.virtual_memory().percent}%")

    def _process_file(self, audio_path: str, output_dir: str):
        """
        Ejecuta el pipeline completo para un solo archivo de audio.
        Asume que todos los modelos ya están cargados en memoria.

        :param audio_path: Ruta al archivo de audio a procesar.
        :param output_dir: Directorio base donde se guardarán los resultados.
        """
        try:
            logger.info("Cargando audio...")
            audio = whisperx.load_audio(audio_path)
            self._log_resource_usage()

            # --- Transcripción (ASR) ---
            logger.info("Transcribiendo audio (ASR)...")
            result = self.asr_model.transcribe(audio, batch_size=self.config.batch_size)
            self._log_resource_usage()
            logger.info("Transcripción completada.")

            # --- Alineación ---
            logger.info("Alineando transcripción (Alignment)...")
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.config.device_torch,
                return_char_alignments=False
            )
            self._log_resource_usage()
            logger.info("Alineación completada.")

            # --- Diarización ---
            if self.diarize_model:
                try:
                    logger.info("Ejecutando diarización de hablantes...")
                    diarize_segments = self.diarize_model(audio)
                    self._log_resource_usage()
                    logger.info("Asignando hablantes a las palabras...")
                    result = assign_word_speakers(diarize_segments, result)
                    logger.info("Diarización completada.")
                except Exception as e:
                    logger.error(f"Error durante la diarización en {audio_path}: {e}")
                    logger.warning("El resultado final NO incluirá hablantes.")
            else:
                logger.warning("Saltando diarización (modelo no cargado).")

            # --- Guardar Resultados ---
            self._save_results(result, audio_path, output_dir)

        except Exception as e:
            logger.error(f"Error fatal procesando el archivo {audio_path}: {e}")
            
    def _save_results(self, result: Dict[str, Any], audio_path: str, output_dir: str):
        """
        Guarda los resultados del pipeline en archivos JSON y TXT.

        :param result: El diccionario de resultados del pipeline.
        :param audio_path: Ruta original del audio (para generar el nombre).
        :param output_dir: Directorio donde se guardarán los archivos.
        """
        logger.info("--- Proceso completado. Guardando resultados... ---")
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]

        json_output_path = os.path.join(output_dir, base_filename + "_completo.json")
        logger.info(f"Guardando resultado JSON completo en: {json_output_path}")
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar el JSON: {e}")

        txt_output_path = os.path.join(output_dir, base_filename + "_simple.txt")
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

    def transcribe_batch(self, input_path: str, output_dir: str):
        """
        Descubre y procesa un lote de archivos de audio desde una ruta de entrada.

        La ruta puede ser un solo archivo o un directorio.

        :param input_path: Ruta a un archivo de audio o un directorio de audios.
        :param output_dir: Directorio base donde se guardarán todos los resultados.
        """
        files_to_process = self._discover_files(input_path)
        if not files_to_process:
            logger.info("No hay archivos para procesar. Saliendo.")
            return

        logger.info("--- Iniciando pipeline de WhisperX ---")
        logger.info(f"Configuración: Modelo={self.config.asr_model_name}, "
                    f"ASR={self.config.device_asr}, Torch={self.config.device_torch}, "
                    f"Compute={self.config.compute_type}, Batch={self.config.batch_size}")

        psutil.cpu_percent(interval=None)  # inicializa psutil

        total_files = len(files_to_process)
        for i, file_path in enumerate(files_to_process, 1):
            logger.info(f"--- Procesando archivo {i}/{total_files}: {file_path} ---")
            self._process_file(file_path, output_dir)
            logger.info(f"--- Finalizado archivo {i}/{total_files}: {file_path} ---")

        logger.info("¡Script finalizado!")

    def _discover_files(self, input_path: str) -> List[str]:
        """
        Genera una lista de archivos de audio válidos a partir de una ruta.

        :param input_path: Ruta a un archivo o directorio.
        :return: Lista de rutas de archivos de audio válidos.
        """
        files_to_process: List[str] = []
        VALID_EXTENSIONS = ('.wav', '.mp3', '.mp4', '.m4a', '.flac', '.aac', '.ogg', '.wma')

        if not os.path.exists(input_path):
            logger.error(f"La ruta de entrada no existe: {input_path}")
            return files_to_process

        if os.path.isfile(input_path):
            if input_path.lower().endswith(VALID_EXTENSIONS):
                files_to_process.append(input_path)
                logger.info("Procesando un solo archivo.")
            else:
                logger.error(f"El archivo {input_path} no es un formato de audio/video válido.")
        elif os.path.isdir(input_path):
            logger.info(f"Escaneando directorio: {input_path}...")
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(VALID_EXTENSIONS):
                        files_to_process.append(os.path.join(root, file))
            if not files_to_process:
                logger.warning(f"No se encontraron archivos de audio/video válidos en {input_path}")
            else:
                logger.info(f"Se encontraron {len(files_to_process)} archivos de audio/video.")
        
        return files_to_process

    def _unload_models(self):
        """
        Descarga todos los modelos de la memoria para liberar recursos.
        Comprueba si el atributo existe antes de intentar eliminarlo.
        """
        logger.info("Descargando modelos de la memoria...")
        
        if hasattr(self, 'asr_model'):
            del self.asr_model
            
        if hasattr(self, 'align_model'):
            del self.align_model
            
        if hasattr(self, 'align_metadata'):
            del self.align_metadata
            
        if hasattr(self, 'diarize_model'):
            del self.diarize_model
            
        gc.collect()
        if self.config.device_torch == "mps":
            torch.mps.empty_cache()
        logger.info("Memoria liberada.")

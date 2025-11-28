import os
import gc
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
import psutil
import nltk
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    try:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        # Opcional: hacer la ejecución determinista
        torch.backends.cudnn.deterministic = True
        logger.info(
            "cuDNN desactivado explícitamente. "
            "El cómputo seguirá en GPU, pero sin usar librerías cuDNN."
        )
    except Exception as e:
        logger.warning(f"No se pudo desactivar cuDNN limpiamente: {e}")

# ---------------------------------------------------------------------
# Cargar de .env desde la raíz del proyecto
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    logger.info(f"Cargado .env desde {ENV_PATH}")
else:
    load_dotenv()
    logger.warning(
        f"No se encontró .env en {ENV_PATH}. "
        "Se intentará cargar variables de entorno globales."
    )

# ---------------------------------------------------------------------
# Configuración del pipeline
# ---------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    Configuración para el WhisperXPipeline.

    Objetivo: TODO lo que se pueda en GPU:

      - ASR: siempre en cuda si hay GPU
      - Alineación: siempre en cuda si hay GPU
      - Diarización: siempre en cuda si hay GPU

    Si no hay GPU, todo se cae a CPU de forma ordenada.
    """

    language: str = "es"
    asr_model: str = "large-v3"
    batch_size: int = 16

    # compute_type: float32 en GPU, int8 en CPU (se ajusta en __post_init__)
    compute_type: Optional[str] = None

    # Dispositivos
    device_asr: Optional[str] = None      # "cuda" o "cpu"
    device_align: Optional[str] = None    # "cuda" o "cpu"
    device_diar: Optional[str] = None     # "cuda" o "cpu"

    # Token HF para diarización pyannote 3.x
    hf_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HUGGING_FACE_TOKEN")
    )

    def __post_init__(self):
        # -----------------------------------------------------------------
        # Detección de GPU
        # -----------------------------------------------------------------
        if torch.cuda.is_available():
            # ASR y alineación en GPU siempre
            self.device_asr = self.device_asr or "cuda"
            self.device_align = self.device_align or "cuda"
            # Diarización en GPU
            self.device_diar = self.device_diar or "cuda"

            logger.info(
                "CUDA detectado. ASR, Alineación y Diarización usarán GPU (cuda), "
                "con cuDNN DESACTIVADO."
            )
        else:
            self.device_asr = self.device_asr or "cpu"
            self.device_align = self.device_align or "cpu"
            self.device_diar = self.device_diar or "cpu"
            logger.warning(
                "CUDA NO disponible. Todo el pipeline se ejecutará en CPU "
                "(más lento)."
            )

        # -----------------------------------------------------------------
        # compute_type por defecto
        # -----------------------------------------------------------------
        if self.compute_type is None:
            if self.device_asr == "cuda":
                # En GPU usamos float32 para aprovechar VRAM
                self.compute_type = "float32"
            else:
                # En CPU, int8 más eficiente
                self.compute_type = "int8"
            logger.info(f"compute_type no especificado. Usando: {self.compute_type}")

        # -----------------------------------------------------------------
        # Token HF
        # -----------------------------------------------------------------
        if self.hf_token is None:
            logger.warning(
                "HUGGING_FACE_TOKEN no está definido. "
                "La diarización (pyannote) NO se podrá cargar."
            )


# ---------------------------------------------------------------------
# Clase principal del pipeline
# ---------------------------------------------------------------------


class WhisperXPipeline:
    """
    Pipeline completo: ASR + alineación + diarización + guardado.

    - ASR + alineación: WhisperX, en GPU si está disponible.
    - Diarización: pyannote 3.x (pyannote/speaker-diarization-3.1), en GPU si existe.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.asr_model = None
        self.align_model = None
        self.align_metadata = None
        self.diar_model: Optional[DiarizationPipeline] = None

        self._setup_nltk()
        self._load_models()

    # ------------------------- SETUP & MODELOS -------------------------

    def _setup_nltk(self) -> None:
        """Configura la ruta de NLTK local (utilities/nltk_data)."""
        custom_nltk_path = os.path.join(
            PROJECT_ROOT, "audio_transcription", "utilities", "nltk_data"
        )
        if os.path.exists(custom_nltk_path):
            nltk.data.path.append(custom_nltk_path)
            logger.info(f"NLTK path añadido: {custom_nltk_path}")
        else:
            logger.warning(
                f"No se encontró NLTK en {custom_nltk_path}. "
                "Se usarán rutas por defecto."
            )

    def _load_models(self) -> None:
        """
        Carga el modelo ASR, el modelo de alineación y el modelo de diarización.
        ASR + align → GPU cuando existe.
        Diarización → GPU cuando existe.
        """
        try:
            # 1) ASR (WhisperX / faster-whisper)
            logger.info(
                f"[1/3] Cargando ASR WhisperX '{self.config.asr_model}' "
                f"en {self.config.device_asr} (compute_type={self.config.compute_type})…"
            )
            self.asr_model = whisperx.load_model(
                self.config.asr_model,
                device=self.config.device_asr,
                compute_type=self.config.compute_type,
                language=self.config.language,
            )

            # 2) Alineación
            logger.info(
                f"[2/3] Cargando modelo de alineación para '{self.config.language}' "
                f"en {self.config.device_align}…"
            )
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.config.language,
                device=self.config.device_align,
            )

            # 3) Diarización (pyannote 3.x) vía WhisperX.DiarizationPipeline
            self._load_diarization_model()

            logger.info("✔ Todos los modelos cargados correctamente.")
        except Exception as e:
            logger.critical(f"Error fatal cargando modelos: {e}")
            raise

    def _load_diarization_model(self) -> None:
        """
        Carga el modelo de diarización usando pyannote/speaker-diarization-3.1.

        - Requiere HUGGING_FACE_TOKEN válido y aceptación de términos en HF.
        - Usa siempre el dispositivo indicado en config.device_diar.
        """

        if not self.config.hf_token:
            logger.warning(
                "No se cargará diarización: falta HUGGING_FACE_TOKEN en el entorno."
            )
            return

        device = self.config.device_diar or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            logger.info(
                f"[3/3] Cargando diarización pyannote "
                f"'pyannote/speaker-diarization-3.1' en {device}…"
            )
            self.diar_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.hf_token,
                device=device,
            )
            logger.info(f"Diarización cargada correctamente en {device}.")
        except Exception as e:
            self.diar_model = None
            logger.error(
                "No se pudo cargar diarización. "
                f"Se ejecutará el pipeline SIN diarización. Error: {e}"
            )

    # ------------------------- UTILIDADES -------------------------

    @staticmethod
    def _log_resource_usage() -> None:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        logger.info(f"CPU={cpu:.1f}% | RAM={mem:.1f}%")

    def _discover_files(self, input_path: str) -> List[str]:
        """
        Lista archivos de audio/vídeo válidos desde un archivo o un directorio.
        """
        VALID_EXT = (".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".wma")
        files: List[str] = []

        if not os.path.exists(input_path):
            logger.error(f"La ruta de entrada no existe: {input_path}")
            return files

        if os.path.isfile(input_path):
            if input_path.lower().endswith(VALID_EXT):
                files.append(input_path)
                logger.info("Procesando un único archivo.")
            else:
                logger.error(f"El archivo no es un formato válido: {input_path}")
        else:
            logger.info(f"Escaneando directorio: {input_path}")
            for root, _, fnames in os.walk(input_path):
                for fn in fnames:
                    if fn.lower().endswith(VALID_EXT):
                        files.append(os.path.join(root, fn))
            logger.info(f"Se encontraron {len(files)} archivos de audio/vídeo.")

        return files

    # ------------------------- GUARDADO -------------------------

    def _save_results(
        self, result: Dict[str, Any], audio_path: str, output_dir: str
    ) -> None:
        """
        Guarda JSON completo y TXT simple (un speaker por línea).
        """
        base = os.path.splitext(os.path.basename(audio_path))[0]

        json_out = os.path.join(output_dir, base + "_completo.json")
        txt_out = os.path.join(output_dir, base + "_simple.txt")

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Guardando JSON en: {json_out}")
        try:
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error guardando JSON: {e}")

        logger.info(f"Guardando TXT en: {txt_out}")
        try:
            with open(txt_out, "w", encoding="utf-8") as f:
                segments = result.get("segments", [])
                if not segments:
                    f.write("No se encontraron segmentos.\n")
                else:
                    for seg in segments:
                        speaker = seg.get("speaker", "HABLANTE_DESCONOCIDO")
                        text = seg.get("text", "").strip()
                        f.write(f"[{speaker}]: {text}\n")
        except Exception as e:
            logger.error(f"Error guardando TXT: {e}")

    # ------------------------- PROCESAMIENTO -------------------------

    def _process_file(self, audio_path: str, output_dir: str) -> None:
        """
        Ejecuta ASR + alineación + (opcional) diarización sobre un archivo.
        """
        try:
            logger.info(f"Cargando audio: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            self._log_resource_usage()

            # 1) ASR
            logger.info("Transcribiendo (ASR)…")
            result = self.asr_model.transcribe(
                audio,
                batch_size=self.config.batch_size,
            )
            self._log_resource_usage()
            logger.info("Transcripción completada.")

            # 2) Alineación
            logger.info("Alineando transcripción…")
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.config.device_align,
                return_char_alignments=False,
            )
            self._log_resource_usage()
            logger.info("Alineación completada.")

            # 3) Diarización (si está disponible)
            if self.diar_model is not None:
                try:
                    logger.info(
                        "Ejecutando diarización (pyannote "
                        "pyannote/speaker-diarization-3.1)…"
                    )
                    diar_segments = self.diar_model(audio)
                    self._log_resource_usage()
                    logger.info("Asignando hablantes a palabras/segmentos…")
                    result = assign_word_speakers(diar_segments, result)
                    logger.info("Diarización completada.")
                except Exception as e:
                    logger.error(
                        f"Error en diarización para {audio_path}: {e}. "
                        "Se continúa sin hablantes."
                    )
            else:
                logger.info("Diarización desactivada (sin modelo cargado).")

            # 4) Guardar
            self._save_results(result, audio_path, output_dir)

        except Exception as e:
            logger.error(f"Error procesando {audio_path}: {e}")

    def transcribe_batch(self, input_path: str, output_dir: str) -> None:
        """
        Orquesta el procesamiento de un archivo o un lote de archivos.
        """
        files = self._discover_files(input_path)
        if not files:
            logger.info("No hay archivos para procesar. Saliendo.")
            return

        logger.info(f"{len(files)} archivos detectados.")
        logger.info("Procesando archivos…")

        for idx, path in enumerate(files, start=1):
            logger.info(f"--- {idx}/{len(files)} --- {path}")
            self._process_file(path, output_dir)

        logger.info("✔ Pipeline finalizado.")

    # ------------------------- LIMPIEZA -------------------------

    def unload_models(self) -> None:
        """
        Libera memoria de los modelos (por si se usa en procesos largos).
        """
        logger.info("Descargando modelos de la memoria…")
        for attr in ["asr_model", "align_model", "align_metadata", "diar_model"]:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Memoria liberada.")
import os
import sys
import gc
import json
import csv
import argparse
import logging
import tempfile
import stat
import warnings
from pathlib import Path

# --- GESTIÓN DE RUTAS RELATIVAS ---
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parent                  
MODULE_DIR = SCRIPTS_DIR.parent                    
PROJECT_ROOT = MODULE_DIR.parent                   

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import torch
import pandas as pd
import imageio_ffmpeg
import nltk  

import whisper 
import whisperx


def setup_local_nltk():
    """
    Configura NLTK para usar los tokenizers locales (punkt/punkt_tab)
    """
    local_nltk_path = MODULE_DIR / "utilities" / "nltk_data"
    
    if local_nltk_path.exists():
        nltk.data.path.insert(0, str(local_nltk_path))
        print(f"📚 NLTK local cargado desde: {local_nltk_path}")
    else:
        print(f"⚠️ ADVERTENCIA CRÍTICA: No se encontró NLTK en {local_nltk_path}.")
        print("   WhisperX intentará descargar datos, lo cual fallará sin internet.")

# ======================================================
# 1. SETUP DEL ENTORNO
# ======================================================

def setup_ffmpeg_wrapper():
    """Shim de ffmpeg usando imageio-ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    wrapper_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_wrapper_hifi")
    os.makedirs(wrapper_dir, exist_ok=True)
    wrapper_path = os.path.join(wrapper_dir, "ffmpeg")
    
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(f"#!/usr/bin/env bash\n\"{ffmpeg_exe}\" \"$@\"\n")
    
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)
    os.environ["PATH"] = wrapper_dir + os.pathsep + os.path.dirname(ffmpeg_exe) + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg_exe

def setup_hardware_precision():
    """Configuración FP32 para PyTorch."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False 
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("high")

# ======================================================
# 2. CLASE PIPELINE (MOTOR HÍBRIDO)
# ======================================================

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.device = f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger("HiFi-Pipeline")
        
        # Aislar la GPU para este proceso
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        
        self.logger.info(f"🧠 Pipeline iniciado en {self.device} (FP32 Force)")
        
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not self.hf_token and args.diarize:
            self.logger.warning("⚠️ Faltan credenciales HF. Diarización será omitida.")

    def _clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- ETAPA 1: ASR (OpenAI Whisper - FP32) ---
    def transcribe_fp32(self, audio_path):
        self.logger.info(f"🎤 [1/3] ASR (FP32)...")
        
        model = whisper.load_model(self.args.model_size, device=self.device)
        
        result = model.transcribe(
            audio_path,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            best_of=5,
            beam_size=5,
            patience=1.0,
            length_penalty=1.0,
            verbose=False,
            fp16=False, 
            language=self.args.language
        )
        
        del model
        self._clear_vram()
        return result

    # --- ETAPA 2: ALINEACIÓN (WhisperX + NLTK Local) ---
    def align_segments(self, asr_result, audio_path):
        self.logger.info(f"⏱️ [2/3] Alineación Forzada (Wav2Vec2)...")
        
        audio = whisperx.load_audio(audio_path)
        
        align_model, align_metadata = whisperx.load_align_model(
            language_code=self.args.language,
            device=self.device
        )
        
        # Aquí WhisperX usa internamente nltk.sent_tokenize con nuestros recursos locales
        aligned_result = whisperx.align(
            asr_result["segments"],
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False
        )
        
        del align_model
        del align_metadata
        self._clear_vram()
        
        return aligned_result, audio

    # --- ETAPA 3: DIARIZACIÓN (Pyannote) ---
    def diarize_speakers(self, aligned_result, audio):
        if not self.args.diarize or not self.hf_token:
            self.logger.info("⏩ Saltando diarización.")
            return aligned_result

        self.logger.info(f"👥 [3/3] Diarización (Pyannote 3.1)...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
            
            diarize_segments = diarize_model(audio)
            final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            
            del diarize_model
            self._clear_vram()
            return final_result
        except Exception as e:
            self.logger.error(f"❌ Error en diarización: {e}")
            return aligned_result

    # --- GUARDADO ---
    def save_results(self, result, audio_path):
        base_name = Path(audio_path).stem
        out_path = Path(self.args.output_dir) / base_name
        
        # 1. JSON FULL
        with open(f"{out_path}_full.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        segments = result["segments"]

        # 2. TXT Legible
        with open(f"{out_path}.txt", "w", encoding="utf-8") as f:
            for seg in segments:
                spk = seg.get("speaker", "UNKNOWN")
                text = seg.get("text", "").strip()
                f.write(f"[{spk}] {text}\n")

        # 3. CSV Words (Confianza detallada)
        csv_data = []
        for seg in segments:
            spk = seg.get("speaker", "UNKNOWN")
            for w in seg.get("words", []):
                conf = w.get("score", w.get("probability", 0.0))
                csv_data.append({
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "word": w.get("word", "").strip(),
                    "confidence": f"{conf:.4f}",
                    "speaker": spk
                })
        
        pd.DataFrame(csv_data).to_csv(f"{out_path}_words.csv", index=False)
        self.logger.info(f"✅ Guardado: {base_name}")

    def run(self):
        input_path = Path(self.args.input_path)
        output_path = Path(self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if input_path.is_file():
            files = [input_path]
        else:
            files = sorted([f for f in input_path.rglob('*') if f.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac']])
        
        total = len(files)
        self.logger.info(f"📂 Archivos a procesar: {total}")

        for idx, fpath in enumerate(files, 1):
            try:
                self.logger.info(f"--- Procesando {idx}/{total}: {fpath.name} ---")
                
                # Check Idempotencia
                if (output_path / f"{fpath.stem}_full.json").exists():
                    self.logger.info("⏭️  Ya procesado. Saltando.")
                    continue

                asr_res = self.transcribe_fp32(str(fpath))
                align_res, audio_obj = self.align_segments(asr_res, str(fpath))
                final_res = self.diarize_speakers(align_res, audio_obj)
                
                self.save_results(final_res, str(fpath))

            except Exception as e:
                self.logger.error(f"❌ Fallo crítico en {fpath.name}: {e}", exc_info=True)
                self._clear_vram()

if __name__ == "__main__":
    setup_local_nltk()     
    setup_ffmpeg_wrapper()
    setup_hardware_precision()

    parser = argparse.ArgumentParser(description="High-Fidelity Hybrid Transcriber")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--model_size", default=os.getenv("DEFAULT_MODEL_SIZE", "large-v3"))
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "es"))
    parser.add_argument("--no-diarize", action="store_false", dest="diarize")
    parser.set_defaults(diarize=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    pipeline = Pipeline(args)
    pipeline.run()
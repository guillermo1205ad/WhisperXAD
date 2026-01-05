import os
import sys
import gc
import json
import csv
import argparse
import logging
import tempfile
import stat
import math
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
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


def setup_local_nltk():
    local_nltk_path = MODULE_DIR / "utilities" / "nltk_data"
    if local_nltk_path.exists():
        nltk.data.path.insert(0, str(local_nltk_path))
    else:
        print(f"⚠️ ADVERTENCIA: No se encontró NLTK local en {local_nltk_path}")

# ======================================================
# 1. SETUP DEL ENTORNO
# ======================================================

def setup_ffmpeg_wrapper():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    wrapper_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_wrapper_hifi")
    os.makedirs(wrapper_dir, exist_ok=True)
    wrapper_path = os.path.join(wrapper_dir, "ffmpeg")
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(f"#!/usr/bin/env bash\n\"{ffmpeg_exe}\" \"$@\"\n")
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)
    os.environ["PATH"] = wrapper_dir + os.pathsep + os.path.dirname(ffmpeg_exe) + os.pathsep + os.environ.get("PATH", "")

def setup_hardware_precision():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False 
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("high")

# ======================================================
# 2. CLASE PIPELINE
# ======================================================

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger("HiFi-Pipeline")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.logger.info(f"🧠 Pipeline iniciado en {self.device} (FP32 Force)")
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")

    def _clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- ETAPA 1: ASR (OpenAI Whisper - FP32) ---
    def transcribe_fp32(self, audio_path):
        self.logger.info(f"🎤 [1/3] ASR (FP32) + Word Timestamps...")
        model = whisper.load_model(self.args.model_size, device=self.device)
        
        result = model.transcribe(
            audio_path,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            best_of=5, beam_size=5, patience=1.0, length_penalty=1.0,
            verbose=False, fp16=False, word_timestamps=True,
            language=self.args.language
        )
        del model
        self._clear_vram()
        return result

    # --- ETAPA 2: ALINEACIÓN ---
    def align_segments(self, asr_result, audio_path):
        self.logger.info(f"⏱️ [2/3] Alineación Forzada (Wav2Vec2)...")
        audio = whisperx.load_audio(audio_path)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=self.args.language, device=self.device
        )
        
        aligned_result = whisperx.align(
            asr_result["segments"],
            align_model, align_metadata, audio, self.device,
            return_char_alignments=False
        )
        del align_model
        del align_metadata
        self._clear_vram()
        return aligned_result, audio

    # --- ETAPA 3: DIARIZACIÓN ---
    def diarize_speakers(self, aligned_result, audio):
        if not self.args.diarize or not self.hf_token:
            return aligned_result
        self.logger.info(f"👥 [3/3] Diarización...")
        try:
            diarize_model = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
            diarize_segments = diarize_model(audio)
            final_result = assign_word_speakers(diarize_segments, aligned_result)
            del diarize_model
            self._clear_vram()
            return final_result
        except Exception as e:
            self.logger.error(f"❌ Error diarización: {e}")
            return aligned_result

    # --- GUARDADO ---
    def save_results(self, final_result, raw_whisper_result, audio_path):
        base_name = Path(audio_path).stem
        out_path = Path(self.args.output_dir) / base_name
        
        # ==============================================================================
        # MERGING Y RENOMBRADO DE MÉTRICAS
        # ==============================================================================
        
        raw_words_flat = [w for s in raw_whisper_result["segments"] for w in s.get("words", [])]
        raw_idx = 0
        total_raw = len(raw_words_flat)
        
        for seg in final_result["segments"]:
            for w in seg.get("words", []):
                # 1. Inyectar 'probability' (Semántica - Whisper Original)
                if raw_idx < total_raw:
                    w["probability"] = raw_words_flat[raw_idx].get("probability", 0.0)
                    raw_idx += 1
                else:
                    w["probability"] = 0.0
                
                # 2. Renombrar 'score' -> 'alignment_score' (Tiempo - WhisperX)
                if "score" in w:
                    w["alignment_score"] = w.pop("score") # pop elimina 'score' y devuelve su valor

        # Ahora el objeto `final_result` tiene claves limpias: 
        # { ... "alignment_score": 0.9, "probability": 0.8 ... }

        # 1. JSON FULL (Keys corregidas)
        with open(f"{out_path}_full.json", "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        # 2. CSV DE PALABRAS
        word_csv_data = []
        for seg in final_result["segments"]:
            spk = seg.get("speaker", "UNKNOWN")
            for w in seg.get("words", []):
                word_csv_data.append({
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "word": w.get("word", "").strip(),
                    # Ahora accedemos directamente a la nueva clave
                    "alignment_score": f"{w.get('alignment_score', 0.0):.4f}", 
                    "probability": f"{w.get('probability', 0.0):.4f}", 
                    "speaker": spk
                })
        
        pd.DataFrame(word_csv_data).to_csv(f"{out_path}_words.csv", index=False)
        self.logger.info(f"✅ Guardado Completo: {base_name}")

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
                
                if (output_path / f"{fpath.stem}_full.json").exists():
                    self.logger.info("⏭️  Ya procesado.")
                    continue

                asr_res_raw = self.transcribe_fp32(str(fpath))
                
                import copy
                asr_for_align = copy.deepcopy(asr_res_raw)
                align_res, audio_obj = self.align_segments(asr_for_align, str(fpath))
                
                final_res = self.diarize_speakers(align_res, audio_obj)
                
                self.save_results(final_res, asr_res_raw, str(fpath))

            except Exception as e:
                self.logger.error(f"❌ Fallo crítico en {fpath.name}: {e}", exc_info=True)
                self._clear_vram()

if __name__ == "__main__":
    setup_local_nltk()     
    setup_ffmpeg_wrapper()
    setup_hardware_precision()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--model_size", default=os.getenv("DEFAULT_MODEL_SIZE", "large-v3"))
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "es"))
    parser.add_argument("--no-diarize", action="store_false", dest="diarize")
    parser.set_defaults(diarize=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    Pipeline(args).run()
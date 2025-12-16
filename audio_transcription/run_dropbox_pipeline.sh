#!/bin/bash

# ==============================================================================
# SMART PIPELINE RUNNER v9 (FIXED EXECUTION PATH)
# Descripción:
#   Orquesta WhisperX usando rutas absolutas para evitar errores de "File not found".
# ==============================================================================

# --- CONFIGURACIÓN DE USUARIO ---

# 1. LINK DE DROPBOX
DROPBOX_LINK="https://www.dropbox.com/scl/fi/03wych8ijyd1k8idw6mwr/seleccion_audios.zip?rlkey=32tp7n934h65msh75c7w5xrab&e=3&st=rrspn3e6&dl=0"

# 2. CONFIGURACIÓN MANUAL (Opcional)
MANUAL_GPU_ID="0"       # Forzamos GPU 0 como pediste
MANUAL_BATCH_SIZE="64" # Forzamos Batch 256 como pediste

# 3. CARPETA DE SALIDA
OUTPUT_SUBFOLDER="dataset"
# --------------------------------

# 1. Rutas y Directorios Absolutos
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

STAGING_ROOT="$PROJECT_ROOT/audio_transcription/inputs/dropbox_staging"
ZIP_FILE="$STAGING_ROOT/audios_descargados.zip"
EXTRACT_DIR="$STAGING_ROOT/extracted"
QUEUE_DIR="$STAGING_ROOT/.processing_queue"
OUTPUT_DIR="$PROJECT_ROOT/audio_transcription/outputs/transcripciones/$OUTPUT_SUBFOLDER"

# Ruta ABSOLUTA al script de python (Aquí estaba el error antes)
PYTHON_SCRIPT="$PROJECT_ROOT/audio_transcription/scripts/transcribe_whisperx.py"

mkdir -p "$STAGING_ROOT"
mkdir -p "$EXTRACT_DIR"
mkdir -p "$OUTPUT_DIR"

# --- BLOQUE CRÍTICO: AUTO-DETECCIÓN DE UV ---
UV_CMD=""
if command -v uv &> /dev/null; then UV_CMD="uv"
elif [ -f "$HOME/.cargo/bin/uv" ]; then UV_CMD="$HOME/.cargo/bin/uv"
elif [ -f "$HOME/.local/bin/uv" ]; then UV_CMD="$HOME/.local/bin/uv"
elif [ -f "/usr/bin/uv" ]; then UV_CMD="/usr/bin/uv"
elif [ -f "/usr/local/bin/uv" ]; then UV_CMD="/usr/local/bin/uv"
fi

if [ -z "$UV_CMD" ]; then
    echo "❌ ERROR CRÍTICO: El script no encuentra 'uv'."
    exit 127
fi

# 2. Lógica de Descarga
echo "--- Paso 1: Verificación de Inputs ---"
if [ -d "$EXTRACT_DIR" ] && [ "$(find "$EXTRACT_DIR" -type f | head -n 1)" ]; then
    echo "✅ Archivos detectados en local. Saltando descarga."
else
    echo "⬇️  Iniciando descarga..."
    DIRECT_LINK="${DROPBOX_LINK%=0}=1"
    DIRECT_LINK="${DIRECT_LINK//dl=0/dl=1}"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$ZIP_FILE" "$DIRECT_LINK"
    else
        curl -L -o "$ZIP_FILE" "$DIRECT_LINK"
    fi
    echo "📦 Descomprimiendo..."
    unzip -o -q "$ZIP_FILE" -d "$EXTRACT_DIR"
fi

# 3. Lógica Incremental
echo "--- Paso 2: Buscando audios y calculando faltantes ---"
rm -rf "$QUEUE_DIR"
mkdir -p "$QUEUE_DIR"
COUNT_PENDING=0
COUNT_SKIPPED=0

echo "🔎 Verificando existencia de archivos con sufijo '_completo.json'..."

while IFS= read -r audio_file; do
    filename=$(basename "$audio_file")
    filename_no_ext="${filename%.*}"
    
    if [[ "$audio_file" == *"__MACOSX"* ]]; then continue; fi

    TARGET_CHECK_1="$OUTPUT_DIR/${filename_no_ext}_completo.json"
    TARGET_CHECK_2="$OUTPUT_DIR/${filename_no_ext}.json"

    if [ -f "$TARGET_CHECK_1" ]; then
        ((COUNT_SKIPPED++))
    elif [ -f "$TARGET_CHECK_2" ]; then
        ((COUNT_SKIPPED++))
    else
        ln -s "$audio_file" "$QUEUE_DIR/$filename"
        ((COUNT_PENDING++))
    fi
done < <(find "$EXTRACT_DIR" -type f \( -iname "*.mp3" -o -iname "*.wav" -o -iname "*.m4a" -o -iname "*.flac" \))

echo "📊 Resumen Final:"
echo "   ✅ Ya procesados (Ignorados): $COUNT_SKIPPED"
echo "   ⏳ Pendientes en cola:        $COUNT_PENDING"

if [ "$COUNT_PENDING" -eq 0 ]; then
    echo "🎉 ¡Todo listo! No hay archivos nuevos para procesar."
    exit 0
fi

INPUT_DIR="$QUEUE_DIR"

# 4. Configuración de Hardware
echo "--- Paso 3: Configuración de Hardware ---"

if [ -n "$MANUAL_GPU_ID" ]; then
    GPU_ID=$MANUAL_GPU_ID
    echo "🔧 MODO MANUAL: GPU ID $GPU_ID"
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$GPU_ID)
else
    echo "🔍 MODO AUTO: Buscando mejor GPU..."
    BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1)
    GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F', ' '{print $1}')
    FREE_MEM=$(echo "$BEST_GPU_INFO" | awk -F', ' '{print $2}')
    echo "✅ GPU Seleccionada: $GPU_ID ($FREE_MEM MiB Libres)"
fi

if [ -z "$FREE_MEM" ]; then echo "❌ Error: No se pudo leer VRAM."; exit 1; fi

COMPUTE_TYPE="float16"
if [ -n "$MANUAL_BATCH_SIZE" ]; then
    BATCH_SIZE=$MANUAL_BATCH_SIZE
    echo "🔧 MODO MANUAL: Batch $BATCH_SIZE"
else
    echo "🧠 Calculando Batch para $FREE_MEM MiB..."
    if [ "$FREE_MEM" -ge 40000 ]; then BATCH_SIZE=256
    elif [ "$FREE_MEM" -ge 24000 ]; then BATCH_SIZE=128
    elif [ "$FREE_MEM" -ge 12000 ]; then BATCH_SIZE=64
    elif [ "$FREE_MEM" -ge 6000 ]; then BATCH_SIZE=16
    else BATCH_SIZE=4; COMPUTE_TYPE="int8"; fi
fi

echo "🚀 Ejecutando -> GPU: $GPU_ID | Batch: $BATCH_SIZE"

# 5. Ejecución (CORREGIDA CON RUTA ABSOLUTA)
export CUDA_VISIBLE_DEVICES=$GPU_ID
CUDNN_PATH=$($UV_CMD run python -c "import os; import nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')" 2>/dev/null)
[ -n "$CUDNN_PATH" ] && export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"

# Verificar que el script python existe antes de lanzar
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error Fatal: No se encuentra el script de python en:"
    echo "   $PYTHON_SCRIPT"
    exit 1
fi

$UV_CMD run python "$PYTHON_SCRIPT" \
    "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --compute_type $COMPUTE_TYPE \
    --asr_model "large-v3"

EXIT_CODE=$?
rm -rf "$QUEUE_DIR"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Completado."
else
    echo "⚠️ Error ($EXIT_CODE)."
fi

exit $EXIT_CODE

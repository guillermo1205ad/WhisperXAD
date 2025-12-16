# --- 1. CONFIGURACIÓN DE USUARIO ---

# ID de la GPU a usar. Déjalo vacío "" para detección automática.
MANUAL_GPU_IDX="0"

INPUT_SUBDIR="inputs/dropbox_staging"
OUTPUT_SUBDIR="outputs/dataset_hifi"

# -------------------------------------------------------

# 2.  Rutas Absolutas 
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # .../audio_transcription
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"                    # .../WhisperXAD (Raíz)

ENV_FILE="$PROJECT_ROOT/.env"
PYTHON_SCRIPT="$SCRIPT_PATH/scripts/transcribe.py"

# Rutas completas 
STAGING_DIR="$PROJECT_ROOT/$INPUT_SUBDIR"
ZIP_FILE="$STAGING_DIR/audios.zip"
EXTRACT_DIR="$STAGING_DIR/extracted"
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_SUBDIR"

# 3. Cargar variables del .env
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "❌ Error: No se encontró .env en $ENV_FILE"
    exit 1
fi

if [ -z "$DROPBOX_URL" ]; then
    echo "❌ Error: DROPBOX_URL no definido en .env"
    exit 1
fi

mkdir -p "$STAGING_DIR" "$EXTRACT_DIR" "$OUTPUT_DIR"

# 4. Descarga 
echo "⬇️  Verificando Dropbox..."
if [ "$(ls -A "$EXTRACT_DIR" 2>/dev/null)" ]; then
    echo "✅ Archivos ya extraídos en '$EXTRACT_DIR'. Saltando descarga."
else
    DIRECT_LINK="${DROPBOX_URL%=0}=1"
    DIRECT_LINK="${DIRECT_LINK//dl=0/dl=1}"
    
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$ZIP_FILE" "$DIRECT_LINK"
    else
        curl -L -o "$ZIP_FILE" "$DIRECT_LINK"
    fi
    
    echo "📦 Descomprimiendo..."
    unzip -o -q "$ZIP_FILE" -d "$EXTRACT_DIR"
fi

# 5. Selector de GPU (Manual con Fallback Automático)
echo "🔍 Configurando GPU..."

if [ -n "$MANUAL_GPU_IDX" ]; then
    BEST_GPU=$MANUAL_GPU_IDX
    echo "🔒 MODO MANUAL: Forzando uso de GPU ID $BEST_GPU (Por seguridad)"
else
    # Lógica automática 
    echo "🤖 MODO AUTO: Buscando GPU con más VRAM libre..."
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F', ' '{print $1}')
    
    if [ -z "$BEST_GPU" ]; then
        echo "⚠️ No se pudo detectar GPU. Usando ID 0 por defecto."
        BEST_GPU=0
    fi
    echo "✅ GPU Detectada: ID $BEST_GPU"
fi

# 6. Ejecución del Pipeline
echo "🚀 Iniciando Transcripción (High Fidelity)..."

cd "$PROJECT_ROOT"

uv run python "$PYTHON_SCRIPT" \
    --input_path "$EXTRACT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --gpu_index $BEST_GPU \
    --model_size "${DEFAULT_MODEL_SIZE:-large-v3}" \
    --language "${DEFAULT_LANGUAGE:-es}"

EXIT_CODE=$?
echo "🏁 Proceso finalizado con código $EXIT_CODE"
exit $EXIT_CODE
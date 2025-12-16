# 1. Definir Rutas Absolutas 
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # .../audio_transcription
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"                    # .../WhisperXAD (Raíz)

ENV_FILE="$PROJECT_ROOT/.env"
PYTHON_SCRIPT="$SCRIPT_PATH/scripts/transcribe.py"

# Rutas de Datos (inputs/outputs)
STAGING_DIR="$PROJECT_ROOT/inputs/dropbox_staging"
ZIP_FILE="$STAGING_DIR/audios.zip"
EXTRACT_DIR="$STAGING_DIR/extracted"
OUTPUT_DIR="$PROJECT_ROOT/outputs/dataset_hifi"

# 2. Cargar variables del .env
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

# 3. Descarga 
echo "⬇️  Verificando Dropbox..."
if [ "$(ls -A "$EXTRACT_DIR" 2>/dev/null)" ]; then
    echo "✅ Archivos ya extraídos. Saltando descarga."
else
    DIRECT_LINK="${DROPBOX_URL%=0}=1"
    DIRECT_LINK="${DIRECT_LINK//dl=0/dl=1}"
    
    # Intento con wget, fallback a curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$ZIP_FILE" "$DIRECT_LINK"
    else
        curl -L -o "$ZIP_FILE" "$DIRECT_LINK"
    fi
    
    echo "📦 Descomprimiendo..."
    unzip -o -q "$ZIP_FILE" -d "$EXTRACT_DIR"
fi

# 4. Selector de GPU 
echo "🔍 Buscando GPU libre..."
# Ordena por memoria libre descendente y toma el ID de la primera
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F', ' '{print $1}')

if [ -z "$BEST_GPU" ]; then
    echo "⚠️ No se pudo detectar GPU automáticamente. Usando ID 0 por defecto."
    BEST_GPU=0
fi

echo "✅ GPU Seleccionada: ID $BEST_GPU"

# 5. Ejecución del Pipeline
echo "🚀 Iniciando Transcripción..."

# Ejecutamos desde PROJECT_ROOT para que uv encuentre pyproject.toml
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
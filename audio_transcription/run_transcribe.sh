# --- CONFIGURACIÓN ---
MANUAL_GPU_IDX="3"

# Definir Rutas Base
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"
PYTHON_SCRIPT="$SCRIPT_PATH/scripts/transcribe.py"
ENV_FILE="$PROJECT_ROOT/.env"

# --- RUTAS  ---
BASE_INPUT_DIR="$PROJECT_ROOT/audio_transcription/inputs/audios"
ZIP_FILE="$BASE_INPUT_DIR/source.zip"
FINAL_INPUT_PATH="$BASE_INPUT_DIR/test_audios"
OUTPUT_DIR="$PROJECT_ROOT/audio_transcription/outputs/dataset_hifi"

# --------------------------------------------------------
# CARGAR .ENV
# --------------------------------------------------------
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "❌ Error: No existe $ENV_FILE"
    exit 1
fi

if [ -z "$DROPBOX_URL" ]; then
    echo "❌ Error: DROPBOX_URL vacío."
    exit 1
fi

mkdir -p "$BASE_INPUT_DIR" "$OUTPUT_DIR"

# --------------------------------------------------------
# DESCARGA
# --------------------------------------------------------
echo "⬇️  Verificando datos..."

if [ -d "$FINAL_INPUT_PATH" ] && [ "$(ls -A "$FINAL_INPUT_PATH")" ]; then
    echo "✅ Datos encontrados en: $FINAL_INPUT_PATH"
else
    echo "📦 Descargando en: $BASE_INPUT_DIR"
    DIRECT_LINK="${DROPBOX_URL%=0}=1"
    DIRECT_LINK="${DIRECT_LINK//dl=0/dl=1}"
    
    wget -q --show-progress -O "$ZIP_FILE" "$DIRECT_LINK"
    
    echo "📂 Descomprimiendo..."
    unzip -o -q "$ZIP_FILE" -d "$BASE_INPUT_DIR"
fi

# --------------------------------------------------------
# EJECUTAR PYTHON
# --------------------------------------------------------
if [ -n "$MANUAL_GPU_IDX" ]; then
    GPU=$MANUAL_GPU_IDX
else
    GPU=0
fi

echo "🚀 Procesando carpeta: $FINAL_INPUT_PATH"
cd "$PROJECT_ROOT"

uv run python "$PYTHON_SCRIPT" \
    --input_path "$FINAL_INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --gpu_index $GPU

exit $?
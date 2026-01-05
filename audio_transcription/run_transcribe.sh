# --- CONFIGURACIÓN ---
MANUAL_GPU_IDX="3"

# Definir Rutas Base
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")"
PYTHON_SCRIPT="$SCRIPT_PATH/scripts/transcribe.py"
ENV_FILE="$PROJECT_ROOT/.env"

# --- RUTAS ---
BASE_INPUT_DIR="$PROJECT_ROOT/audio_transcription/inputs/audios"
ZIP_FILE="$BASE_INPUT_DIR/source.zip"
FINAL_INPUT_PATH="$BASE_INPUT_DIR/test_audios"
OUTPUT_DIR="$PROJECT_ROOT/audio_transcription/outputs/dataset_hifi"

# --------------------------------------------------------
# 1. CARGAR .ENV (Modo Texto Seguro)
# --------------------------------------------------------
# Usamos grep para leer la URL completa ignorando símbolos como '&'
if [ -f "$ENV_FILE" ]; then
    DROPBOX_URL=$(grep "^DROPBOX_URL=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '\r')
else
    echo "❌ Error: No se encontró .env en $ENV_FILE"
    exit 1
fi

# Validación
if [ -z "$DROPBOX_URL" ]; then
    echo "❌ Error: DROPBOX_URL no encontrado o vacío."
    exit 1
fi

mkdir -p "$BASE_INPUT_DIR" "$OUTPUT_DIR"

# --------------------------------------------------------
# 2. DESCARGA
# --------------------------------------------------------
echo "⬇️  Verificando datos..."

if [ -d "$FINAL_INPUT_PATH" ] && [ "$(ls -A "$FINAL_INPUT_PATH")" ]; then
    echo "✅ Datos encontrados en: $FINAL_INPUT_PATH"
else
    rm -f "$ZIP_FILE"

    echo "📦 Descargando en: $BASE_INPUT_DIR"

    DIRECT_LINK=$(echo "$DROPBOX_URL" | sed 's/dl=0/dl=1/g')
    
    wget -q --show-progress -O "$ZIP_FILE" "$DIRECT_LINK"
    
    echo "📂 Descomprimiendo..."
    unzip -o -q "$ZIP_FILE" -d "$BASE_INPUT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ Error Crítico: El archivo descargado no es un ZIP válido."
        echo "   Posible causa: La URL de Dropbox expiró o es incorrecta."
        echo "   Contenido del archivo erróneo:"
        head -n 5 "$ZIP_FILE"
        exit 1
    fi
fi

# --------------------------------------------------------
# 3. EJECUTAR PYTHON
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
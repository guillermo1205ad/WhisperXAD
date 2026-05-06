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
OUTPUT_DIR="$PROJECT_ROOT/audio_transcription/outputs/transcripciones"

mkdir -p "$BASE_INPUT_DIR" "$OUTPUT_DIR"

has_audio_files() {
    find "$BASE_INPUT_DIR" -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.flac' \) | grep -q .
}

# --------------------------------------------------------
# 1. RESOLVER FUENTE DE DATOS
# --------------------------------------------------------
echo "⬇️  Verificando datos..."

if has_audio_files; then
    echo "✅ Datos encontrados bajo: $BASE_INPUT_DIR"
else
    # Usamos grep para leer la URL completa ignorando símbolos como '&'
    if [ -f "$ENV_FILE" ]; then
        DROPBOX_URL=$(grep "^DROPBOX_URL=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '\r')
    else
        echo "❌ Error: No se encontraron audios en $BASE_INPUT_DIR ni .env en $ENV_FILE"
        exit 1
    fi

    if [ -z "$DROPBOX_URL" ]; then
        echo "❌ Error: No se encontraron audios en $BASE_INPUT_DIR y DROPBOX_URL no existe o está vacío."
        exit 1
    fi

    rm -f "$ZIP_FILE"

    echo "📦 Descargando en: $BASE_INPUT_DIR"

    DIRECT_LINK=$(echo "$DROPBOX_URL" | sed 's/dl=0/dl=1/g')
    
    wget -q --show-progress -O "$ZIP_FILE" "$DIRECT_LINK"
    
    echo "📂 Descomprimiendo..."
    uv run python -c "import zipfile; print('📂 Extrayendo con Python...'); zipfile.ZipFile('$ZIP_FILE').extractall('$BASE_INPUT_DIR')"

    if [ $? -ne 0 ]; then
        echo "❌ Error Crítico: El archivo descargado no es un ZIP válido."
        echo "   Posible causa: La URL de Dropbox expiró o es incorrecta."
        echo "   Contenido del archivo erróneo:"
        head -n 5 "$ZIP_FILE"
        exit 1
    fi

    if ! has_audio_files; then
        echo "❌ Error: La descarga/extracción terminó, pero no se encontraron audios válidos en $BASE_INPUT_DIR"
        exit 1
    fi
fi

# --------------------------------------------------------
# 2. EJECUTAR PYTHON
# --------------------------------------------------------
if [ -n "$MANUAL_GPU_IDX" ]; then
    GPU=$MANUAL_GPU_IDX
else
    GPU=0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "🚀 Procesando carpeta raíz: $BASE_INPUT_DIR"
cd "$PROJECT_ROOT"

uv run python "$PYTHON_SCRIPT" \
    --input_path "$BASE_INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --gpu_index $GPU

exit $?

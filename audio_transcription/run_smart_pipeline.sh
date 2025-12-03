# ==============================================================================
# SMART PIPELINE RUNNER
# Descripción:
#   Orquesta la ejecución de WhisperX adaptándose a CUALQUIER entorno GPU.
#   1. Detecta la GPU con más memoria libre.
#   2. Calcula parámetros óptimos (Batch Size) basado en VRAM disponible.
#   3. Aísla el proceso para evitar colisiones.
#   4. Ejecuta el pipeline de WhisperX sobre carpetas predefinidas.
#
#   RUTAS PREDEFINIDAS:
#   - Input:  audio_transcription/inputs/audios/
#   - Output: audio_transcription/outputs/transcripciones/
#
#   LÓGICA DE SEGURIDAD VRAM (Large-v3 + Pyannote):
#   - < 6 GB:    ABORTAR. Riesgo inminente de OOM (Out of Memory).
#   - 6-10 GB:   MODO SEGURO. Forzar 'int8' y batch bajo.
#   - > 10 GB:   MODO ESTÁNDAR. Usar 'float16' para mayor velocidad/precisión.
# ==============================================================================

# 1. Configuración de entorno y rutas
# ------------------------------------------------------------------------------
# Determinar ruta del script y raíz del proyecto dinámicamente
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Nos movemos a la raíz para que uv detecte el entorno virtual
cd "$PROJECT_ROOT" || { echo "Error crítico: No se pudo acceder a $PROJECT_ROOT"; exit 1; }

# --- CONFIGURACIÓN DE RUTAS ---
INPUT_DIR="$PROJECT_ROOT/audio_transcription/inputs/audios/"
OUTPUT_DIR="$PROJECT_ROOT/audio_transcription/outputs/transcripciones/"

# --- CONFIGURACIÓN DE LIBRERÍAS NVIDIA (CUDNN/CUBLAS) ---
# Intenta localizar las librerías de nvidia instaladas por uv para inyectarlas al PATH.

echo "Configurando librerías NVIDIA..."
CUDNN_PATH=""

# Método 1: Preguntar a Python 
CUDNN_PATH=$(uv run python -c "import os; import nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')" 2>/dev/null)

# Método 2: Búsqueda física en .venv (si método 1 falla)
if [ -z "$CUDNN_PATH" ]; then
    echo "   Método Python falló, buscando en sistema de archivos..."
    CUDNN_PATH=$(find "$PROJECT_ROOT/.venv" -path "*/nvidia/cudnn/lib" -type d -print -quit 2>/dev/null)
fi

# Inyección al entorno
if [ -n "$CUDNN_PATH" ] && [ -d "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH actualizado con cuDNN: $CUDNN_PATH"
else
    echo "Advertencia CRÍTICA: No se pudo detectar nvidia-cudnn en .venv/"
    echo "Si la ejecución falla, asegura haber corrido: uv add nvidia-cudnn-cu12 nvidia-cublas-cu12"
fi
# ------------------------------

echo "Directorio de ejecución: $(pwd)"
echo "Input definido:  $INPUT_DIR"
echo "Output definido: $OUTPUT_DIR"

# Validación básica
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: El directorio de entrada no existe: $INPUT_DIR"
    exit 1
fi

# 2. Detección de hardware con nvidia-smi
# ------------------------------------------------------------------------------
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi no detectado. Se ejecutará en CPU (Compute Type: int8)."
    echo "Esto será significativamente más lento."
    
    # Fallback CPU con rutas fijas
    uv run python audio_transcription/scripts/transcribe_whisperx.py \
        "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --compute_type int8
    exit $?
fi

echo "Buscando la GPU más libre..."

# Obtenemos: ID, Memoria Libre (MiB), Nombre. Ordenamos por memoria libre descendente.
BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free,gpu_name --format=csv,noheader,nounits | sort -k2 -nr | head -n 1)

GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F', ' '{print $1}')
FREE_MEM=$(echo "$BEST_GPU_INFO" | awk -F', ' '{print $2}')
GPU_NAME=$(echo "$BEST_GPU_INFO" | awk -F', ' '{print $3}')

# 3. Lógica de asignación de recursos 
# ------------------------------------------------------------------------------

# UMBRAL DE CORTE: 6GB
MIN_REQ_MEM=6000

if [ -z "$FREE_MEM" ] || [ "$FREE_MEM" -lt "$MIN_REQ_MEM" ]; then
    echo "ERROR: Recursos insuficientes."
    echo "   GPU disponible: $GPU_NAME (ID $GPU_ID) tiene $FREE_MEM MiB libres."
    echo "   Requerido mínimo: $MIN_REQ_MEM MiB para garantizar estabilidad."
    exit 1
fi

echo "GPU Asignada: $GPU_NAME (ID $GPU_ID) | VRAM Disponible: $FREE_MEM MiB"

# Definición de perfiles 
if [ "$FREE_MEM" -ge 40000 ]; then
    # Memoria de sobra 
    BATCH_SIZE=24
    COMPUTE_TYPE="float16"
    echo "Perfil seleccionado: ULTRA (Batch 24, FP16)"

elif [ "$FREE_MEM" -ge 22000 ]; then
    # High End (24GB)
    BATCH_SIZE=8
    COMPUTE_TYPE="float16"
    echo "Perfil seleccionado: HIGH (Batch 8, FP16)"

elif [ "$FREE_MEM" -ge 10000 ]; then
    # Standard (11GB)
    BATCH_SIZE=4
    COMPUTE_TYPE="float16"
    echo "Perfil seleccionado: STANDARD (Batch 4, FP16)"

else
    # Low End (6GB-10GB)
    BATCH_SIZE=4
    COMPUTE_TYPE="int8"
    echo "Perfil seleccionado: SAFE MODE (Batch 4, INT8 forced)"
fi

# 4. Aislamiento y ejecución
# ------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "--- Iniciando Pipeline WhisperX ---"
echo "   Modelo: large-v3"
echo "   Batch:  $BATCH_SIZE"
echo "   Type:   $COMPUTE_TYPE"
echo "   Ruta:   $INPUT_DIR -> $OUTPUT_DIR"
echo "-----------------------------------"

# Ejecución con rutas fijas
uv run python audio_transcription/scripts/transcribe_whisperx.py \
    "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --compute_type $COMPUTE_TYPE \
    --asr_model "large-v3"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Ejecución finalizada correctamente en GPU $GPU_ID."
else
    echo "Error en la ejecución (Código $EXIT_CODE). Revisa los logs superiores."
fi

exit $EXIT_CODE

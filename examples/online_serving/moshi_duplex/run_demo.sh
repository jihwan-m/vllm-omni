#!/bin/bash
# =============================================================================
# Moshi Full-Duplex Demo: All-in-One Setup & Run
# =============================================================================
#
# End-to-end script that installs dependencies, starts the server, waits
# for it to be ready, and runs a WebSocket client demo.
#
# Usage:
#   ./run_demo.sh                     # FP16 on H100/A100 (default)
#   ./run_demo.sh --quantized         # INT4 AWQ on RTX 4070 Ti Super / 4090
#   ./run_demo.sh --skip-install      # Skip venv/pip setup (already installed)
#   ./run_demo.sh --client-only       # Only run client (server already running)
#
# Requirements:
#   FP16:      1x H100-80G / A100-80G, CUDA 12.1+, Python 3.10+
#   Quantized: 1x RTX 4070 Ti Super (16 GB+), CUDA 12.1+, Python 3.10+
#
# Reference: https://arxiv.org/abs/2410.00037
# =============================================================================

set -euo pipefail

# ----------------------------- Python Detection ------------------------------
# On Windows (MINGW/MSYS), Python is typically 'python', not 'python3'
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Install Python 3.10+ and ensure it's on PATH."
    exit 1
fi

# ----------------------------- Configuration ---------------------------------

MODEL="kmhf/hf-moshiko"
QUANTIZED=false
QUANTIZATION_METHOD=""
STAGE_CONFIG="moshi_duplex.yaml"
PORT=8000
HOST="0.0.0.0"
SKIP_INSTALL=false
CLIENT_ONLY=false
DURATION=15          # Default client recording duration (seconds)
GPU_MEMORY_UTIL=0.90

# ----------------------------- Parse Arguments -------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --quantized)
            QUANTIZED=true
            shift
            ;;
        --quantization)
            QUANTIZATION_METHOD="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --client-only)
            CLIENT_ONLY=true
            SKIP_INSTALL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quantized              Use INT4 AWQ quantization (for 16 GB GPUs)"
            echo "  --quantization METHOD    Quantization method: awq, gptq (default: awq)"
            echo "  --model MODEL            Model name or path (default: kmhf/hf-moshiko)"
            echo "  --port PORT              Server port (default: 8000)"
            echo "  --host HOST              Server host (default: 0.0.0.0)"
            echo "  --duration SECONDS       Client recording duration (default: 15)"
            echo "  --skip-install           Skip venv creation and pip install"
            echo "  --client-only            Only run the client (server must be running)"
            echo "  --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                       # Full setup + FP16 demo on H100"
            echo "  $0 --quantized           # Full setup + INT4 demo on RTX 4070 Ti"
            echo "  $0 --client-only         # Just run client against running server"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

# Apply quantized defaults
if [ "$QUANTIZED" = true ]; then
    QUANTIZATION_METHOD="${QUANTIZATION_METHOD:-awq}"
    STAGE_CONFIG="moshi_duplex_quantized.yaml"
    GPU_MEMORY_UTIL=0.95
fi

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STAGE_CONFIGS_DIR="$REPO_ROOT/vllm_omni/model_executor/stage_configs"

echo "=========================================="
echo " Moshi Full-Duplex Audio Streaming Demo"
echo "=========================================="
echo ""
echo "  Mode:        $([ "$QUANTIZED" = true ] && echo "INT4 $QUANTIZATION_METHOD (16 GB GPU)" || echo "FP16 (80 GB GPU)")"
echo "  Model:       $MODEL"
echo "  Stage config: $STAGE_CONFIG"
echo "  Server:      http://${HOST}:${PORT}"
echo "  Duration:    ${DURATION}s"
echo ""
echo "=========================================="

# ----------------------------- Step 1: Install -------------------------------

if [ "$SKIP_INSTALL" = false ]; then
    echo ""
    echo "[1/4] Installing dependencies..."
    echo "--------------------------------------"

    # Create venv if not active
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        # Locate the activate script (cross-platform)
        _find_activate() {
            for candidate in \
                "$REPO_ROOT/.venv/Scripts/activate" \
                "$REPO_ROOT/.venv/bin/activate"; do
                if [ -f "$candidate" ]; then
                    echo "$candidate"
                    return 0
                fi
            done
            return 1
        }

        if ! _find_activate &>/dev/null; then
            echo "Creating Python virtual environment..."
            # Remove broken venv if it exists without an activate script
            [ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
            $PYTHON -m venv "$REPO_ROOT/.venv"
        fi

        ACTIVATE_SCRIPT="$(_find_activate)" || {
            echo "ERROR: Could not find venv activate script after creation."
            echo "  Contents of .venv/:"
            ls -R "$REPO_ROOT/.venv/" 2>/dev/null | head -30
            echo ""
            echo "Try creating a venv manually:"
            echo "  $PYTHON -m venv .venv"
            echo "  source .venv/Scripts/activate   # Windows/Git Bash"
            echo "  source .venv/bin/activate        # Linux/Mac"
            echo "Then re-run with: bash $0 --skip-install --quantized"
            exit 1
        }
        echo "Activating virtual environment ($ACTIVATE_SCRIPT)..."
        source "$ACTIVATE_SCRIPT"
    else
        echo "Using active venv: $VIRTUAL_ENV"
    fi

    # Install vLLM
    echo "Installing vLLM..."
    pip install -q vllm 2>&1 | tail -1

    # Install vllm-omni
    echo "Installing vllm-omni..."
    pip install -q -e "$REPO_ROOT" 2>&1 | tail -1

    # Install client dependencies
    echo "Installing client dependencies (websockets, numpy)..."
    pip install -q websockets numpy 2>&1 | tail -1

    echo "Done."
else
    echo ""
    echo "[1/4] Skipping installation (--skip-install)"
fi

# ----------------------------- Step 2: Server --------------------------------

SERVER_PID=""
# Cross-platform temp directory
TMPDIR="${TMPDIR:-${TEMP:-${TMP:-/tmp}}}"
LOG_FILE="${TMPDIR}/moshi_duplex_server_${PORT}.log"
OUTPUT_FILE="${TMPDIR}/moshi_output.pcm"

cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT SIGINT SIGTERM

if [ "$CLIENT_ONLY" = false ]; then
    echo ""
    echo "[2/4] Starting vLLM-Omni server..."
    echo "--------------------------------------"

    # Build server command
    SERVER_CMD=(vllm serve "$MODEL" --omni
        --port "$PORT"
        --host "$HOST"
        --stage-configs-path "$STAGE_CONFIGS_DIR/$STAGE_CONFIG"
        --gpu-memory-utilization "$GPU_MEMORY_UTIL"
        --trust-remote-code
        --enforce-eager
    )
    if [ -n "$QUANTIZATION_METHOD" ]; then
        SERVER_CMD+=(--quantization "$QUANTIZATION_METHOD")
    fi

    echo "Command: ${SERVER_CMD[*]}"
    echo ""

    "${SERVER_CMD[@]}" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to start (this may take a few minutes for model download)..."
    MAX_WAIT=600
    ELAPSED=0

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        # Check if server is still running
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: Server process exited. Last 20 lines of log:"
            tail -20 "$LOG_FILE"
            exit 1
        fi

        # Check health endpoint
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo ""
            echo "Server is ready!"
            break
        fi

        # Progress indicator every 10 seconds
        if [ $((ELAPSED % 10)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
            echo "  ... still waiting (${ELAPSED}s elapsed)"
        fi

        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo ""
        echo "ERROR: Server did not start within ${MAX_WAIT}s. Last 20 lines of log:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
else
    echo ""
    echo "[2/4] Skipping server start (--client-only)"
    echo "Checking server at http://localhost:${PORT}..."
    if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: No server running at http://localhost:${PORT}"
        echo "Start the server first, or remove --client-only"
        exit 1
    fi
    echo "Server is running."
fi

# ----------------------------- Step 3: Client --------------------------------

echo ""
echo "[3/4] Running WebSocket client demo..."
echo "--------------------------------------"
echo ""
echo "Connecting to ws://localhost:${PORT}/v1/audio/duplex"
echo "Sending ${DURATION}s of synthetic audio (sine wave at 440 Hz)"
echo ""

$PYTHON "$SCRIPT_DIR/client_demo.py" \
    --url "ws://localhost:${PORT}/v1/audio/duplex" \
    --duration "$DURATION" \
    --duplex \
    --output "$OUTPUT_FILE"

# ----------------------------- Step 4: Summary -------------------------------

echo ""
echo "[4/4] Demo complete!"
echo "--------------------------------------"
echo ""
echo "  Output audio: $OUTPUT_FILE"
echo "  Server log:   $LOG_FILE"
echo ""
echo "  Play output:  ffplay -f s16le -ar 24000 -ac 1 $OUTPUT_FILE"
echo "  Or with sox:  play -t raw -r 24000 -e signed -b 16 -c 1 $OUTPUT_FILE"
echo ""

if [ "$CLIENT_ONLY" = false ]; then
    echo "Server is still running on http://localhost:${PORT}"
    echo "Press Ctrl+C to stop."
    echo ""
    wait "$SERVER_PID" || true
fi

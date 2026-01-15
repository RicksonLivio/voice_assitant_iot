#!/bin/bash
# Complete Setup Script for Offline Voice Assistant - IMPROVED VERSION
# Includes Kokoro TTS installation for natural human-like voice

set -e  # Exit on error

echo "=========================================="
echo "  Offline Voice Assistant Setup"
echo "  (IMPROVED with Kokoro TTS)"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Step 1: Update system packages
echo -e "${GREEN}[1/9] Updating system packages...${NC}"
sudo apt update
sudo apt install -y build-essential python3-dev portaudio19-dev \
    libportaudio2 ffmpeg espeak espeak-data libespeak-dev git wget curl \
    libsndfile1 cmake

# Step 2: Create/activate virtual environment
echo -e "${GREEN}[2/9] Creating virtual environment with UV...${NC}"
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate

# Step 3: Install PyTorch (CPU version)
echo -e "${GREEN}[3/9] Installing PyTorch (CPU version)...${NC}"
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install Kokoro TTS (natural voice)
echo -e "${GREEN}[4/9] Installing Kokoro TTS (natural human-like voice)...${NC}"
if uv pip install kokoro-tts 2>/dev/null; then
    echo -e "${GREEN}✓ Kokoro TTS installed successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Kokoro TTS installation failed, will use pyttsx3 fallback${NC}"
fi

# Step 5: Install other dependencies
echo -e "${GREEN}[5/9] Installing other dependencies...${NC}"
uv pip install openai-whisper numpy scipy sounddevice soundfile pyaudio pyttsx3 llama-cpp-python pyyaml

# Step 6: Create models directory
echo -e "${GREEN}[6/9] Creating models directory...${NC}"
mkdir -p models

# Step 7: Download Whisper model
echo -e "${GREEN}[7/9] Pre-downloading Whisper model...${NC}"
python3 << 'PYEOF'
import whisper
print("Downloading Whisper 'tiny' model...")
model = whisper.load_model("tiny")
print("✓ Whisper model downloaded successfully!")
PYEOF

# Step 8: Download LLM model
echo -e "${GREEN}[8/9] Downloading LLM model (TinyLlama)...${NC}"
if [ ! -f "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    cd models
    wget -q --show-progress https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    cd ..
    echo -e "${GREEN}✓ LLM model downloaded!${NC}"
else
    echo -e "${YELLOW}Model already exists, skipping download.${NC}"
fi

# Step 9: Test installation
echo -e "${GREEN}[9/9] Testing installation...${NC}"
python3 << 'PYEOF'
print("\nTesting imports...")
passed = []
failed = []

# Test PyTorch
try:
    import torch
    passed.append(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    failed.append(f"✗ PyTorch: {e}")

# Test Whisper
try:
    import whisper
    passed.append("✓ Whisper")
except ImportError as e:
    failed.append(f"✗ Whisper: {e}")

# Test llama-cpp-python
try:
    from llama_cpp import Llama
    passed.append("✓ llama-cpp-python")
except ImportError as e:
    failed.append(f"✗ llama-cpp-python: {e}")

# Test Kokoro TTS
try:
    from kokoro import generate
    passed.append("✓ Kokoro TTS (natural voice)")
except ImportError:
    passed.append("⚠️  Kokoro TTS not available (will use pyttsx3)")

# Test pyttsx3
try:
    import pyttsx3
    passed.append("✓ pyttsx3")
except ImportError as e:
    failed.append(f"✗ pyttsx3: {e}")

# Test sounddevice
try:
    import sounddevice
    passed.append("✓ sounddevice")
except ImportError as e:
    failed.append(f"✗ sounddevice: {e}")

# Test numpy
try:
    import numpy
    passed.append("✓ NumPy")
except ImportError as e:
    failed.append(f"✗ NumPy: {e}")

# Test scipy
try:
    import scipy
    passed.append("✓ SciPy")
except ImportError as e:
    failed.append(f"✗ SciPy: {e}")

# Test yaml
try:
    import yaml
    passed.append("✓ PyYAML")
except ImportError as e:
    failed.append(f"✗ PyYAML: {e}")

# Print results
for item in passed:
    print(item)

if failed:
    print("\n⚠️  Some dependencies failed:")
    for item in failed:
        print(item)
else:
    print("\n✅ All core dependencies installed successfully!")
PYEOF

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "✨ NEW FEATURES:"
echo "  • Natural human-like voice (Kokoro TTS)"
echo "  • Smart voice detection (waits for you to finish)"
echo "  • Noise-robust recording"
echo "  • 3-second pause detection"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run the assistant: python3 voice_assistant.py"
echo "  or"
echo "  3. Use the launcher: ./start_assistant.sh"
echo ""
echo "Configuration:"
echo "  • Edit config.yaml to adjust voice detection sensitivity"
echo "  • Adjust 'energy_threshold' for your environment noise level"
echo ""
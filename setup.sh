#!/bin/bash
# Complete Setup Script for Offline Voice Assistant
# Run this script to set up everything from scratch

set -e  # Exit on error

echo "=========================================="
echo "  Offline Voice Assistant Setup"
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
echo -e "${GREEN}[1/8] Updating system packages...${NC}"
sudo apt update
sudo apt install -y build-essential python3-dev portaudio19-dev \
    libportaudio2 ffmpeg espeak espeak-data libespeak-dev git wget curl

# Step 2: Create/activate virtual environment
echo -e "${GREEN}[2/8] Creating virtual environment with UV...${NC}"
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate

# Step 3: Install PyTorch (CPU version)
echo -e "${GREEN}[3/8] Installing PyTorch (CPU version)...${NC}"
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install other dependencies
echo -e "${GREEN}[4/8] Installing other dependencies...${NC}"
uv pip install openai-whisper numpy scipy sounddevice soundfile pyaudio pyttsx3 llama-cpp-python

# Step 5: Create models directory
echo -e "${GREEN}[5/8] Creating models directory...${NC}"
mkdir -p models

# Step 6: Download Whisper model
echo -e "${GREEN}[6/8] Pre-downloading Whisper model...${NC}"
python3 << 'PYEOF'
import whisper
print("Downloading Whisper 'tiny' model...")
model = whisper.load_model("tiny")
print("✓ Whisper model downloaded successfully!")
PYEOF

# Step 7: Download LLM model
echo -e "${GREEN}[7/8] Downloading LLM model (TinyLlama)...${NC}"
if [ ! -f "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    cd models
    wget -q --show-progress https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    cd ..
    echo -e "${GREEN}✓ LLM model downloaded!${NC}"
else
    echo -e "${YELLOW}Model already exists, skipping download.${NC}"
fi

# Step 8: Test installation
echo -e "${GREEN}[8/8] Testing installation...${NC}"
python3 << 'PYEOF'
print("\nTesting imports...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    
try:
    import whisper
    print("✓ Whisper")
except ImportError as e:
    print(f"✗ Whisper: {e}")

try:
    from llama_cpp import Llama
    print("✓ llama-cpp-python")
except ImportError as e:
    print(f"✗ llama-cpp-python: {e}")

try:
    import pyttsx3
    print("✓ pyttsx3")
except ImportError as e:
    print(f"✗ pyttsx3: {e}")

try:
    import sounddevice
    print("✓ sounddevice")
except ImportError as e:
    print(f"✗ sounddevice: {e}")

print("\nAll core dependencies installed successfully!")
PYEOF

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run the assistant: python3 voice_assistant.py"
echo "  or"
echo "  3. Use the launcher: ./start_assistant.sh"
echo ""




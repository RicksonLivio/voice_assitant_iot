#!/bin/bash
# Setup Script for Voice Assistant with Qwen2.5:1.5B
# Qwen2.5 is a superior model with better reasoning and multilingual support

set -e  # Exit on error

echo "=========================================="
echo "  Voice Assistant Setup"
echo "  With Qwen2.5:1.5B (Superior Model)"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Display model information
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}  QWEN2.5:1.5B MODEL INFORMATION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Why Qwen2.5 instead of TinyLlama?${NC}"
echo "  ✓ Better reasoning and logic"
echo "  ✓ Superior instruction following"
echo "  ✓ Multilingual support (29 languages)"
echo "  ✓ More coherent responses"
echo "  ✓ Better context understanding"
echo ""
echo -e "${YELLOW}Trade-offs:${NC}"
echo "  • Slightly slower (~3-5s vs 2-3s)"
echo "  • Slightly more RAM (~2GB vs 1.5GB)"
echo "  • Larger download (~950MB vs 700MB)"
echo ""
echo -e "${GREEN}Worth it? Absolutely for better conversations!${NC}"
echo ""
read -p "Press ENTER to continue with Qwen2.5 setup..."
echo ""

# Step 1: Update system packages
echo -e "${GREEN}[1/10] Updating system packages...${NC}"
sudo apt update
sudo apt install -y build-essential python3-dev portaudio19-dev \
    libportaudio2 ffmpeg espeak espeak-data libespeak-dev git wget curl \
    libsndfile1 cmake

# Step 2: Create/activate virtual environment
echo -e "${GREEN}[2/10] Creating virtual environment with UV...${NC}"
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate

# Step 3: Install PyTorch (CPU version)
echo -e "${GREEN}[3/10] Installing PyTorch (CPU version)...${NC}"
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install Kokoro TTS (natural voice)
echo -e "${GREEN}[4/10] Installing Kokoro TTS (natural human-like voice)...${NC}"
if uv pip install kokoro-tts 2>/dev/null; then
    echo -e "${GREEN}✓ Kokoro TTS installed successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Kokoro TTS installation failed, will use pyttsx3 fallback${NC}"
fi

# Step 5: Install other dependencies
echo -e "${GREEN}[5/10] Installing other dependencies...${NC}"
uv pip install openai-whisper numpy scipy sounddevice soundfile pyaudio pyttsx3 llama-cpp-python pyyaml

# Step 6: Create models directory
echo -e "${GREEN}[6/10] Creating models directory...${NC}"
mkdir -p models

# Step 7: Download Whisper model
echo -e "${GREEN}[7/10] Pre-downloading Whisper model...${NC}"
python3 << 'PYEOF'
import whisper
print("Downloading Whisper 'tiny' model...")
model = whisper.load_model("tiny")
print("✓ Whisper model downloaded successfully!")
PYEOF

# Step 8: Download Qwen2.5:1.5B model
echo -e "${GREEN}[8/10] Downloading Qwen2.5:1.5B model (~950MB)...${NC}"
echo -e "${BLUE}This is a high-quality model - the download is worth it!${NC}"
if [ ! -f "models/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
    cd models
    
    # Download from HuggingFace
    echo "Downloading from HuggingFace..."
    wget -q --show-progress https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
    
    cd ..
    echo -e "${GREEN}✓ Qwen2.5 model downloaded!${NC}"
else
    echo -e "${YELLOW}Model already exists, skipping download.${NC}"
fi

# Step 9: Copy Qwen config
echo -e "${GREEN}[9/10] Setting up Qwen2.5 configuration...${NC}"
if [ -f "config_qwen.yaml" ]; then
    cp config_qwen.yaml config.yaml
    echo -e "${GREEN}✓ Configuration updated for Qwen2.5${NC}"
else
    echo -e "${YELLOW}⚠️  config_qwen.yaml not found, keeping existing config${NC}"
fi

# Step 10: Test installation
echo -e "${GREEN}[10/10] Testing installation...${NC}"
python3 << 'PYEOF'
print("\nTesting imports and components...")
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
    passed.append("✓ Whisper (Speech Recognition)")
except ImportError as e:
    failed.append(f"✗ Whisper: {e}")

# Test llama-cpp-python
try:
    from llama_cpp import Llama
    passed.append("✓ llama-cpp-python (LLM Engine)")
except ImportError as e:
    failed.append(f"✗ llama-cpp-python: {e}")

# Test Kokoro TTS
try:
    from kokoro import generate
    passed.append("✓ Kokoro TTS (Natural Voice) - EXCELLENT!")
except ImportError:
    passed.append("⚠️  Kokoro TTS not available (will use pyttsx3)")

# Test pyttsx3
try:
    import pyttsx3
    passed.append("✓ pyttsx3 (TTS Fallback)")
except ImportError as e:
    failed.append(f"✗ pyttsx3: {e}")

# Test sounddevice
try:
    import sounddevice
    passed.append("✓ sounddevice (Audio Recording)")
except ImportError as e:
    failed.append(f"✗ sounddevice: {e}")

# Test numpy
try:
    import numpy
    passed.append("✓ NumPy (Audio Processing)")
except ImportError as e:
    failed.append(f"✗ NumPy: {e}")

# Test scipy
try:
    import scipy
    passed.append("✓ SciPy (Signal Processing)")
except ImportError as e:
    failed.append(f"✗ SciPy: {e}")

# Test yaml
try:
    import yaml
    passed.append("✓ PyYAML (Configuration)")
except ImportError as e:
    failed.append(f"✗ PyYAML: {e}")

# Print results
print("\n" + "="*50)
for item in passed:
    print(item)

if failed:
    print("\n⚠️  Some dependencies failed:")
    for item in failed:
        print(item)
else:
    print("\n✅ All core dependencies installed successfully!")

# Check Qwen model
import os
if os.path.exists("models/qwen2.5-1.5b-instruct-q4_k_m.gguf"):
    print("\n✓ Qwen2.5:1.5B model ready!")
    size_mb = os.path.getsize("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") / (1024*1024)
    print(f"  Model size: {size_mb:.1f} MB")
else:
    print("\n✗ Qwen2.5 model not found")
PYEOF

echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}✨ QWEN2.5:1.5B FEATURES:${NC}"
echo "  • Superior reasoning and logic"
echo "  • Multilingual support (29 languages)"
echo "  • Better instruction following"
echo "  • More coherent conversations"
echo "  • Natural human-like voice (Kokoro TTS)"
echo "  • Smart voice detection"
echo "  • Noise-robust recording"
echo ""
echo -e "${GREEN}🚀 NEXT STEPS:${NC}"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run the assistant: python3 voice_assistant.py"
echo "  or"
echo "  3. Use the launcher: ./start_assistant.sh"
echo ""
echo -e "${YELLOW}💡 CONFIGURATION:${NC}"
echo "  • Edit config.yaml to adjust settings"
echo "  • Model path: ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
echo "  • Adjust energy_threshold for your environment"
echo ""
echo -e "${BLUE}📊 PERFORMANCE EXPECTATIONS:${NC}"
echo "  • Startup: 5-10 seconds"
echo "  • Response time: 3-5 seconds"
echo "  • Memory usage: ~2GB RAM"
echo "  • Quality: Excellent!"
echo ""
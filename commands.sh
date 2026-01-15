#!/bin/bash
# Quick reference commands for Voice Assistant

cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║         VOICE ASSISTANT - QUICK REFERENCE COMMANDS          ║
╚══════════════════════════════════════════════════════════════╝

📦 INITIAL SETUP (run once):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  chmod +x setup.sh
  ./setup.sh

🧪 TEST INSTALLATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  source .venv/bin/activate
  python3 test_components.py

🚀 START ASSISTANT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ./start_assistant.sh

  Or manually:
  source .venv/bin/activate
  python3 voice_assistant.py

📦 INSTALL PACKAGES (if needed):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # First, install PyTorch separately
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  
  # Then install everything else
  uv pip install openai-whisper numpy scipy sounddevice pyaudio pyttsx3 llama-cpp-python

📥 DOWNLOAD MODELS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mkdir -p models
  cd models
  
  # TinyLlama (recommended - 700MB)
  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  
  # Or Phi-2 (better quality - 1.6GB)
  wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf

🧪 TEST INDIVIDUAL COMPONENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Test imports
  python3 -c "import torch, whisper, llama_cpp, pyttsx3; print('All imports OK')"
  
  # Test Whisper
  python3 -c "import whisper; m = whisper.load_model('tiny'); print('Whisper OK')"
  
  # Test LLM
  python3 -c "from llama_cpp import Llama; print('LLM OK')"
  
  # Test TTS
  python3 -c "import pyttsx3; e = pyttsx3.init(); e.say('test'); e.runAndWait()"
  
  # Test microphone
  python3 -c "import sounddevice as sd; print(sd.query_devices())"

🔧 TROUBLESHOOTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Fix PyAudio
  sudo apt install portaudio19-dev python3-pyaudio
  uv pip install pyaudio
  
  # Fix permissions
  sudo usermod -aG audio $USER
  
  # Reinstall llama-cpp-python with optimizations
  CMAKE_ARGS="-DLLAMA_BLAS=ON" uv pip install llama-cpp-python --force-reinstall
  
  # Test audio recording
  arecord -d 3 test.wav && aplay test.wav

📊 CHECK RESOURCES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Check RAM usage
  free -h
  
  # Check CPU
  lscpu | grep -E '^CPU\(s\)|Model name'
  
  # Check disk space
  df -h

🎯 USAGE TIPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Press ENTER to start listening
  • Speak clearly into microphone
  • Wait for silence detection (auto-stops)
  • Say "exit" or "quit" to stop
  • Press Ctrl+C to force quit

🔍 USEFUL FILES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  voice_assistant.py    - Main assistant code
  test_components.py    - Test script
  config.yaml           - Configuration
  setup.sh              - Complete setup
  start_assistant.sh    - Quick launcher
  pyproject.toml        - Project dependencies

📝 PROJECT STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  voice-assistant/
  ├── .venv/                          # Virtual environment
  ├── models/                         # Model files
  │   └── tinyllama-*.gguf
  ├── voice_assistant.py              # Main code
  ├── test_components.py              # Tests
  ├── setup.sh                        # Setup script
  ├── start_assistant.sh              # Launcher
  ├── config.yaml                     # Config
  └── pyproject.toml                  # Dependencies

💡 NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Run setup.sh
  2. Run test_components.py
  3. Run start_assistant.sh
  4. Try asking questions!
  5. Customize config.yaml for your needs

═══════════════════════════════════════════════════════════════

EOF




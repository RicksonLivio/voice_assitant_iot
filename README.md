🎙️ Offline Voice Assistant
A complete CPU-based voice assistant that runs entirely offline on Ubuntu/Linux systems. Perfect for IoT devices, Raspberry Pi, or any system without GPU.
🌟 Features

100% Offline - No internet required after initial setup
CPU Optimized - Runs on any modern CPU
Complete Pipeline:

🎤 Speech Recognition (Whisper)
🤖 Natural Language Processing (LLM via llama.cpp)
🔊 Text-to-Speech (pyttsx3)


Auto Silence Detection - Stops recording automatically
Lightweight - ~3-4GB RAM usage

📋 Prerequisites

Ubuntu 18.04+ (or any Debian-based Linux)
Python 3.11+
At least 8GB RAM (16GB recommended)
20GB free disk space
Microphone and speakers
UV package manager installed

🚀 Quick Start
1. Clone/Setup Project
bashmkdir ~/voice-assistant
cd ~/voice-assistant

# Create these files: pyproject.toml, setup.sh, voice_assistant.py, etc.
2. Run Setup
bashchmod +x setup.sh
./setup.sh
This will:

Install system dependencies
Create virtual environment with UV
Install Python packages
Download Whisper model
Download TinyLlama model (~700MB)

3. Test Components
bashsource .venv/bin/activate
python3 test_components.py
4. Run Assistant
bash./start_assistant.sh
Or manually:
bashsource .venv/bin/activate
python3 voice_assistant.py
📁 Project Structure
voice-assistant/
├── pyproject.toml              # Project configuration
├── setup.sh                    # Complete setup script
├── voice_assistant.py          # Main assistant code
├── test_components.py          # Component testing
├── start_assistant.sh          # Quick launcher
├── config.yaml                 # Configuration file
├── models/                     # Model directory
│   └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
└── .venv/                      # Virtual environment
🎯 Usage

Start the assistant:

bash   ./start_assistant.sh

Press ENTER when you want to speak
Speak your question - The assistant will automatically stop recording after silence
Listen to the response
Say "exit", "quit", or "goodbye" to stop

⚙️ Configuration
Edit config.yaml to customize:

Whisper model size (tiny/base/small/medium/large)
LLM parameters (temperature, tokens, etc.)
Audio settings (silence threshold, sample rate)
TTS settings (voice, speed)

🔧 Troubleshooting
Audio Issues
No microphone detected:
bash# List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
arecord -d 3 test.wav && aplay test.wav
Permission denied:
bashsudo usermod -aG audio $USER
# Logout and login again
Performance Issues
Too slow:

Use smaller Whisper model (tiny)
Reduce LLM max_tokens
Increase n_threads to match CPU cores
Use quantized models (Q4_K_M)

Out of memory:

Close other applications
Use tiny Whisper model
Reduce LLM context size

Installation Issues
PyAudio fails:
bashsudo apt install portaudio19-dev python3-pyaudio
uv pip install pyaudio
llama-cpp-python fails:
bashsudo apt install build-essential
uv pip install llama-cpp-python --force-reinstall
📊 Model Comparison
ModelSizeRAMSpeedQualityWhisper Tiny39M~1GBFastestGoodWhisper Base74M~1.5GBFastBetterWhisper Small244M~2.5GBMediumGreatTinyLlama Q41.1B~1GBFastGoodLlama-2 7B Q47B~4GBSlowerBetter
🎨 Advanced Usage
Custom System Prompt
Edit the system prompt in voice_assistant.py:
pythonsystem_prompt = """
You are a smart home assistant specializing in home automation.
Respond briefly and naturally.
"""
Use Different Models
Larger Whisper model:
pythonassistant = VoiceAssistant(whisper_model="base")
Different LLM:
bash# Download Phi-2 or Llama-2
cd models/
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf
Update in code:
pythonllm_model_path = "./models/phi-2.Q4_K_M.gguf"
Run as System Service
bashsudo cp voice-assistant.service /etc/systemd/system/
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant
🔍 Testing Individual Components
bash# Test all components
python3 test_components.py

# Test Whisper only
python3 -c "import whisper; m = whisper.load_model('tiny'); print('OK')"

# Test LLM only
python3 -c "from llama_cpp import Llama; m = Llama('./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf', n_ctx=512); print('OK')"

# Test TTS only
python3 -c "import pyttsx3; e = pyttsx3.init(); e.say('test'); e.runAndWait()"
📈 Performance Optimization
CPU Optimization
bash# Install OpenBLAS for faster math
sudo apt install libopenblas-dev

# Reinstall llama-cpp-python with BLAS support
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
uv pip install llama-cpp-python --force-reinstall --no-cache-dir
Adjust Thread Count
In voice_assistant.py, set threads to your CPU core count:
pythonn_threads=8  # For 8-core CPU
🎓 Learning Resources

Whisper Documentation
llama.cpp Guide
GGUF Models on HuggingFace

🤝 Contributing
Contributions welcome! Areas for improvement:

Wake word detection
Multi-language support
Context memory
Home automation integration

📝 License
MIT License - Feel free to use and modify
🆘 Support
If you encounter issues:

Run python3 test_components.py to diagnose
Check microphone permissions
Verify models are downloaded
Review logs for error messages

🎉 Next Steps
Once working:

Add wake word detection (e.g., "Hey Assistant")
Integrate with Home Assistant
Add conversation history
Fine-tune prompts for your use case
Deploy on Raspberry Pi


Enjoy your offline voice assistant! 🚀

# 🎙️ Offline Voice Assistant (IMPROVED)

A complete offline voice assistant that runs entirely on CPU. Features natural human-like voice with Kokoro TTS, smart voice detection, and noise-robust recording.

## ✨ Key Improvements

### 🗣️ Natural Human Voice (Kokoro TTS)
- **Human-like speech** instead of robotic voice
- Natural intonation and pronunciation
- Multiple voice options (male/female)
- Fallback to pyttsx3 if Kokoro unavailable

### 🎯 Smart Voice Detection
- **Waits for you to finish speaking** - No more cutting off mid-sentence!
- **3-second pause detection** - Automatically stops after you're done
- **Energy-based detection** - Works in noisy environments
- **Pre-speech buffering** - Captures the start of your speech

### 🔊 Noise-Robust Recording
- Adaptive background noise cancellation
- Automatic volume threshold adjustment
- Works in various environments (quiet/normal/noisy)
- Configurable sensitivity

## 🚀 Quick Start

### 1. One-Command Setup
```bash
chmod +x setup.sh
./setup.sh
```

This installs everything you need:
- Python dependencies
- PyTorch (CPU)
- Whisper for speech recognition
- Kokoro TTS for natural voice
- TinyLlama for conversation
- All audio libraries

### 2. Run the Assistant
```bash
./start_assistant.sh
```

Or manually:
```bash
source .venv/bin/activate
python3 voice_assistant.py
```

## 💬 How to Use

1. **Press ENTER** to start listening
2. **Speak naturally** - the assistant waits for you
3. **Stop talking for 3 seconds** - recording auto-stops
4. **Get natural response** - hear human-like voice
5. Say **"exit"** or **"quit"** to stop

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Adjust for your environment
audio:
  silence_duration: 3.0        # Seconds to wait after speech
  energy_threshold: 2.0        # Voice detection sensitivity
  # Recommended values:
  # - Quiet room: 1.5 - 2.0
  # - Normal room: 2.0 - 3.0  
  # - Noisy environment: 3.0 - 5.0

# Choose your voice
tts:
  kokoro_voice: "af_bella"     # Options: af, am, af_bella, af_sarah, am_adam, am_michael
```

## 🎤 Voice Detection Explained

The improved system uses **energy-based Voice Activity Detection (VAD)**:

1. **Adaptive Background Learning**: Learns your room's noise level
2. **Speech Detection**: Detects when you start speaking
3. **Continuous Recording**: Records while you speak
4. **Smart Stop**: Stops 3 seconds after you finish
5. **Pre-buffering**: Captures 0.5s before you start talking

### Sensitivity Adjustment

If the assistant is:
- **Too sensitive** (triggers on background noise): Increase `energy_threshold` to 3.0-5.0
- **Not sensitive enough** (doesn't hear you): Decrease `energy_threshold` to 1.5-2.0

## 🧪 Testing

Test all components:
```bash
source .venv/bin/activate
python3 test_components.py
```

Tests include:
- ✓ Imports
- ✓ Whisper (speech recognition)
- ✓ LLM (conversation)
- ✓ Kokoro TTS (natural voice)
- ✓ Voice Activity Detection
- ✓ Audio recording
- ✓ Full pipeline

## 📋 Features

### Speech Recognition
- **Whisper AI** - State-of-the-art accuracy
- Supports multiple languages
- Works offline

### Conversation
- **Local LLM** (TinyLlama) - Fast responses
- Conversation history support
- Customizable personality

### Voice Output
- **Kokoro TTS** - Natural human voice
- Multiple voice styles
- Fast generation
- Fallback to pyttsx3

### Audio Processing
- **Smart VAD** - Waits for you to finish
- Noise-robust recording
- Automatic silence detection
- Pre-speech buffering

## 🛠️ System Requirements

- **CPU**: Any modern CPU (2+ cores recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for models
- **OS**: Linux (Ubuntu/Debian)
- **Microphone**: Any USB or built-in mic

## 📦 What Gets Installed

```
Models (~1.5GB):
├── Whisper tiny (~100MB)
└── TinyLlama 1.1B (~700MB)

Python Packages:
├── PyTorch (CPU)
├── OpenAI Whisper
├── llama-cpp-python
├── Kokoro TTS
├── pyttsx3 (fallback)
├── sounddevice
└── Various audio libraries
```

## 🔧 Troubleshooting

### Kokoro TTS Not Available
The assistant will automatically fall back to pyttsx3. To install Kokoro:
```bash
source .venv/bin/activate
uv pip install kokoro-tts
```

### Audio Issues
```bash
# Check microphone
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test recording
arecord -d 3 test.wav && aplay test.wav
```

### Voice Detection Too Sensitive
Edit `config.yaml`:
```yaml
audio:
  energy_threshold: 3.0  # Higher = less sensitive
```

### Voice Detection Not Sensitive Enough
Edit `config.yaml`:
```yaml
audio:
  energy_threshold: 1.5  # Lower = more sensitive
```

## 📁 Project Structure

```
voice-assistant/
├── voice_assistant.py          # Main improved code
├── config.yaml                 # Configuration
├── setup.sh                    # Improved setup script
├── test_components.py          # Updated tests
├── start_assistant.sh          # Quick launcher
├── requirements.txt            # Updated dependencies
├── models/                     # Model files
│   └── tinyllama-*.gguf
└── .venv/                      # Virtual environment
```

## 🎯 Advanced Usage

### Custom Voice Selection
```yaml
tts:
  kokoro_voice: "am_adam"  # Male voice
  # Options: af, am, af_bella, af_sarah, am_adam, am_michael
```

### Conversation History
```yaml
app:
  conversation_history: true  # Remember context
  max_history: 10            # Keep last 10 exchanges
```

### Different Whisper Model
```yaml
whisper:
  model: "base"  # Options: tiny, base, small, medium, large
```

## 🚀 Performance

- **Startup**: 5-10 seconds
- **Voice Detection**: <100ms latency
- **Transcription**: 1-2 seconds (tiny model)
- **LLM Response**: 2-5 seconds
- **TTS Generation**: <1 second
- **Total Response**: 5-8 seconds

## 📝 Tips for Best Results

1. **Speak clearly** but naturally
2. **Wait 3 seconds** after finishing your question
3. **Minimize background noise** when possible
4. **Adjust sensitivity** in config.yaml for your environment
5. **Use conversation history** for contextual responses

## 🤝 Contributing

Improvements welcome! Key areas:
- Additional TTS engines
- Better VAD algorithms
- More language support
- GPU acceleration options

## 📄 License

MIT License - Use freely for any purpose

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Kokoro TTS for natural voice synthesis
- llama.cpp for efficient LLM inference
- TinyLlama for the conversation model

---

**Made with ❤️ for offline, privacy-first AI assistants**
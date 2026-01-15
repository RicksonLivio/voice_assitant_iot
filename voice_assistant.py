#!/usr/bin/env python3
"""
Offline Voice Assistant - Improved Version
Features:
- Kokoro TTS for natural human-like voice
- Advanced Voice Activity Detection (VAD)
- Noise-robust recording with energy-based detection
- Smart pause detection (3s after speech ends)
"""

import whisper
from llama_cpp import Llama
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import sys
import time
from pathlib import Path
from collections import deque
import yaml

# Try to import Kokoro, fallback to pyttsx3 if not available
try:
    from kokoro import generate
    KOKORO_AVAILABLE = True
except ImportError:
    import pyttsx3
    KOKORO_AVAILABLE = False
    print("⚠️  Kokoro not available, falling back to pyttsx3")


class VoiceActivityDetector:
    """
    Advanced Voice Activity Detector using energy-based detection
    Robust to background noise
    """
    def __init__(self, sample_rate=16000, frame_duration=0.03, 
                 energy_threshold=0.6, silence_duration=3.0):
        """
        Args:
            sample_rate: Audio sample rate
            frame_duration: Duration of each frame in seconds
            energy_threshold: Energy threshold multiplier for voice detection
            silence_duration: Seconds of silence before stopping
        """
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_duration)
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.silence_frames = int(silence_duration / frame_duration)
        
        # Adaptive threshold
        self.background_energy = None
        self.adaptation_frames = 30  # Frames to adapt to background
        
    def compute_energy(self, frame):
        """Compute energy of audio frame"""
        return np.sqrt(np.mean(frame ** 2))
    
    def is_speech(self, frame):
        """
        Determine if frame contains speech
        
        Returns:
            (is_speech, energy) tuple
        """
        energy = self.compute_energy(frame)
        
        # Initialize background energy
        if self.background_energy is None:
            self.background_energy = energy
            return False, energy
        
        # Adaptive background energy (slow moving average)
        self.background_energy = 0.95 * self.background_energy + 0.05 * energy
        
        # Speech threshold is relative to background
        threshold = self.background_energy * self.energy_threshold
        
        is_speech = energy > threshold
        return is_speech, energy


class VoiceAssistant:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the Offline Voice Assistant with config file
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.verbose = self.config['app']['verbose']
        
        self._log("🚀 Initializing Offline Voice Assistant (Improved)...")
        
        # Validate model file exists
        llm_model_path = self.config['llm']['model_path']
        if not os.path.exists(llm_model_path):
            raise FileNotFoundError(
                f"LLM model not found at: {llm_model_path}\n"
                f"Please run setup.sh to download the model."
            )
        
        # Load Whisper model
        whisper_model = self.config['whisper']['model']
        self._log(f"📥 Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Load LLM
        self._log(f"📥 Loading LLM: {os.path.basename(llm_model_path)}")
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=self.config['llm']['context_size'],
            n_threads=self.config['llm']['threads'],
            n_gpu_layers=self.config['llm']['gpu_layers'],
            verbose=False
        )
        
        # Initialize TTS (Kokoro or fallback)
        self._init_tts()
        
        # Initialize Voice Activity Detector
        self.vad = VoiceActivityDetector(
            sample_rate=self.config['audio']['sample_rate'],
            energy_threshold=self.config['audio'].get('energy_threshold', 2.0),
            silence_duration=self.config['audio']['silence_duration']
        )
        
        self.sample_rate = self.config['audio']['sample_rate']
        
        # Conversation history
        self.conversation_history = []
        self.keep_history = self.config['app']['conversation_history']
        self.max_history = self.config['app']['max_history']
        
        self._log("✅ Voice Assistant ready!\n")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            # Return default config
            return {
                'whisper': {'model': 'tiny', 'language': 'en', 'fp16': False},
                'llm': {
                    'model_path': './models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
                    'context_size': 2048, 'threads': 4, 'gpu_layers': 0,
                    'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 'max_tokens': 150,
                    'system_prompt': 'You are a helpful voice assistant.'
                },
                'tts': {'rate': 165, 'volume': 1.0, 'voice_index': 0},
                'audio': {
                    'sample_rate': 16000, 'channels': 1, 
                    'silence_duration': 3.0, 'energy_threshold': 2.0
                },
                'app': {'verbose': True, 'conversation_history': False, 'max_history': 10}
            }
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_tts(self):
        """Initialize Text-to-Speech engine"""
        self._log("🔊 Initializing Text-to-Speech...")
        
        if KOKORO_AVAILABLE:
            self._log("   Using Kokoro TTS (natural voice)")
            self.tts_engine = 'kokoro'
            # Kokoro voices: af (female), am (male), af_bella, af_sarah, etc.
            self.kokoro_voice = 'af_bella'  # Natural female voice
        else:
            self._log("   Using pyttsx3 TTS (fallback)")
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.config['tts']['rate'])
            self.tts_engine.setProperty('volume', self.config['tts']['volume'])
            
            # Set voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                voice_idx = self.config['tts']['voice_index']
                if voice_idx < len(voices):
                    self.tts_engine.setProperty('voice', voices[voice_idx].id)
    
    def _log(self, message):
        """Print log message if verbose mode is on"""
        if self.verbose:
            print(message)
    
    def record_with_vad(self):
        """
        Record audio with Voice Activity Detection
        Waits for speech, then stops 3 seconds after speech ends
        Robust to background noise
        
        Returns:
            numpy array of audio data
        """
        self._log("🎤 Listening... (speak when ready)")
        
        audio_chunks = []
        chunk_duration = 0.03  # 30ms chunks
        chunk_size = int(self.sample_rate * chunk_duration)
        
        speech_detected = False
        silence_counter = 0
        silence_frames_needed = int(self.vad.silence_duration / chunk_duration)
        
        # Pre-speech buffer (keep last 0.5s before speech detected)
        pre_buffer = deque(maxlen=int(0.5 / chunk_duration))
        
        def callback(indata, frames, time_info, status):
            nonlocal speech_detected, silence_counter
            
            frame = indata[:, 0] if indata.ndim > 1 else indata
            is_speech, energy = self.vad.is_speech(frame)
            
            if not speech_detected:
                # Still waiting for speech
                pre_buffer.append(frame.copy())
                if is_speech:
                    speech_detected = True
                    # Add pre-buffer to recording
                    audio_chunks.extend(list(pre_buffer))
                    audio_chunks.append(frame.copy())
                    self._log("   🗣️  Speech detected, recording...")
                    silence_counter = 0
            else:
                # Speech was detected, now recording
                audio_chunks.append(frame.copy())
                
                if is_speech:
                    silence_counter = 0
                else:
                    silence_counter += 1
        
        # Start recording
        with sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=chunk_size,
            dtype=np.float32
        ):
            # Wait for speech detection
            while not speech_detected:
                sd.sleep(100)
            
            # Wait for silence after speech
            while silence_counter < silence_frames_needed:
                sd.sleep(100)
        
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0)
            self._log(f"✓ Recording complete ({len(audio_data)/self.sample_rate:.1f}s)")
            return audio_data
        else:
            self._log("⚠️  No audio recorded")
            return np.array([])
    
    def transcribe(self, audio_data):
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_data: numpy array of audio data
            
        Returns:
            Transcribed text string
        """
        if len(audio_data) == 0:
            return ""
        
        self._log("📝 Transcribing audio...")
        
        # Convert to int16 for saving
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            wav.write(temp_path, self.sample_rate, audio_int16)
        
        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_path,
                fp16=self.config['whisper']['fp16'],
                language=self.config['whisper']['language']
            )
            transcription = result["text"].strip()
            
            if transcription:
                self._log(f"📖 You: {transcription}")
            else:
                self._log("⚠️  No speech detected")
            
            return transcription
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def generate_response(self, user_input):
        """
        Generate response using local LLM
        
        Args:
            user_input: User's text input
            
        Returns:
            Generated response text
        """
        if not user_input:
            return "I didn't catch that. Could you please repeat?"
        
        self._log("🤖 Thinking...")
        
        # Build prompt with history if enabled
        if self.keep_history and self.conversation_history:
            history_text = ""
            for role, text in self.conversation_history[-self.max_history:]:
                history_text += f"<|{role}|>\n{text}\n</s>\n"
            
            prompt = f"""<|system|>
{self.config['llm']['system_prompt']}
</s>
{history_text}<|user|>
{user_input}
</s>
<|assistant|>"""
        else:
            prompt = f"""<|system|>
{self.config['llm']['system_prompt']}
</s>
<|user|>
{user_input}
</s>
<|assistant|>"""
        
        # Generate response
        start_time = time.time()
        response = self.llm(
            prompt,
            max_tokens=self.config['llm']['max_tokens'],
            temperature=self.config['llm']['temperature'],
            top_p=self.config['llm']['top_p'],
            top_k=self.config['llm']['top_k'],
            stop=["</s>", "<|user|>", "\n\n"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        elapsed = time.time() - start_time
        
        # Update history
        if self.keep_history:
            self.conversation_history.append(("user", user_input))
            self.conversation_history.append(("assistant", answer))
        
        self._log(f"💭 Assistant ({elapsed:.1f}s): {answer}")
        return answer
    
    def speak(self, text):
        """
        Convert text to speech and play
        
        Args:
            text: Text to speak
        """
        if not text:
            return
        
        self._log("🔊 Speaking...")
        try:
            if self.tts_engine == 'kokoro':
                # Use Kokoro TTS
                audio = generate(text, voice=self.kokoro_voice)
                # Play audio
                sd.play(audio, samplerate=24000)
                sd.wait()
            else:
                # Use pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        except Exception as e:
            self._log(f"⚠️  TTS Error: {e}")
    
    def process_voice_input(self):
        """
        Complete pipeline: Record → Transcribe → Generate → Speak
        
        Returns:
            (transcription, response) tuple
        """
        # Record audio with VAD
        audio_data = self.record_with_vad()
        
        # Transcribe
        transcription = self.transcribe(audio_data)
        
        if not transcription:
            return "", ""
        
        # Generate response
        response = self.generate_response(transcription)
        
        # Speak response
        self.speak(response)
        
        return transcription, response
    
    def run_interactive(self):
        """Run interactive voice assistant loop"""
        print("\n" + "="*60)
        print("  🎙️  OFFLINE VOICE ASSISTANT (IMPROVED)")
        print("="*60)
        print("\n💡 Features:")
        print("   - Natural human-like voice (Kokoro TTS)" if KOKORO_AVAILABLE else "   - Standard TTS voice")
        print("   - Smart voice detection (waits for you to finish)")
        print("   - Noise-robust recording")
        print("   - 3-second pause detection")
        print("\n💡 Commands:")
        print("   - Press ENTER to start listening")
        print("   - Speak naturally, pause for 3 seconds when done")
        print("   - Say 'exit', 'quit', or 'goodbye' to stop")
        print("   - Press Ctrl+C to force quit")
        print("="*60 + "\n")
        
        conversation_count = 0
        
        while True:
            try:
                input("Press ENTER to speak... ")
                conversation_count += 1
                print(f"\n[Conversation #{conversation_count}]")
                
                # Process voice input
                transcription, response = self.process_voice_input()
                
                # Check for exit commands
                if any(word in transcription.lower() for word in 
                       ['exit', 'quit', 'goodbye', 'stop', 'bye']):
                    self.speak("Goodbye! Shutting down the assistant.")
                    break
                
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n✅ Assistant stopped.\n")


def main():
    """Main entry point"""
    try:
        # Create assistant with config file
        assistant = VoiceAssistant(config_path="config.yaml")
        
        # Run interactive mode
        assistant.run_interactive()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Offline Voice Assistant
Complete pipeline: Speech → Text → LLM → Speech
Runs entirely on CPU for IoT devices
"""

import whisper
from llama_cpp import Llama
import pyttsx3
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import sys
import time
from pathlib import Path

class VoiceAssistant:
    def __init__(
        self,
        whisper_model="tiny",
        llm_model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        verbose=True
    ):
        """
        Initialize the Offline Voice Assistant
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            llm_model_path: Path to GGUF model file
            verbose: Print detailed logs
        """
        self.verbose = verbose
        self._log("🚀 Initializing Offline Voice Assistant...")
        
        # Validate model file exists
        if not os.path.exists(llm_model_path):
            raise FileNotFoundError(
                f"LLM model not found at: {llm_model_path}\n"
                f"Please run setup.sh to download the model."
            )
        
        # Load Whisper model
        self._log(f"📥 Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Load LLM
        self._log(f"📥 Loading LLM: {os.path.basename(llm_model_path)}")
        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=2048,          # Context window
            n_threads=4,         # CPU threads (adjust for your CPU)
            n_gpu_layers=0,      # CPU only
            verbose=False
        )
        
        # Initialize TTS
        self._log("🔊 Initializing Text-to-Speech...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 165)  # Speed
        self.tts_engine.setProperty('volume', 1.0)  # Volume
        
        # Set voice (use first available)
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        
        # Audio settings
        self.sample_rate = 16000
        self.silence_threshold = 500  # Adjust based on environment
        self.silence_duration = 2.0   # Seconds of silence to stop recording
        
        self._log("✅ Voice Assistant ready!\n")
        
    def _log(self, message):
        """Print log message if verbose mode is on"""
        if self.verbose:
            print(message)
    
    def record_audio(self, duration=5):
        """
        Record audio for a fixed duration
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            numpy array of audio data
        """
        self._log(f"🎤 Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        self._log("✓ Recording complete")
        return audio_data
    
    def record_until_silence(self):
        """
        Record audio until silence is detected
        
        Returns:
            numpy array of audio data
        """
        self._log("🎤 Listening... (speak now, will stop after silence)")
        
        audio_chunks = []
        chunk_size = 1024
        silence_chunks = int(self.silence_duration * self.sample_rate / chunk_size)
        silent_chunks_count = 0
        
        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks_count
            audio_chunks.append(indata.copy())
            
            # Check if current chunk is silent
            volume = np.abs(indata).mean()
            if volume < self.silence_threshold:
                silent_chunks_count += 1
            else:
                silent_chunks_count = 0
        
        with sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=chunk_size
        ):
            while silent_chunks_count < silence_chunks:
                sd.sleep(100)
        
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0)
            self._log("✓ Recording stopped (silence detected)")
            return audio_data
        else:
            self._log("⚠ No audio recorded")
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
        
        self._log("🔄 Transcribing audio...")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            wav.write(temp_path, self.sample_rate, audio_data)
        
        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_path,
                fp16=False,  # Use FP32 for CPU
                language='en'  # Change as needed
            )
            transcription = result["text"].strip()
            
            if transcription:
                self._log(f"📝 You: {transcription}")
            else:
                self._log("⚠ No speech detected")
            
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
        
        # Create prompt for TinyLlama
        prompt = f"""<|system|>
You are a helpful voice assistant. Give brief, clear, and natural responses suitable for speech. Keep answers under 3 sentences unless more detail is specifically requested.
</s>
<|user|>
{user_input}
</s>
<|assistant|>"""
        
        # Generate response
        start_time = time.time()
        response = self.llm(
            prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop=["</s>", "<|user|>", "\n\n"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        elapsed = time.time() - start_time
        
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
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            self._log(f"⚠ TTS Error: {e}")
    
    def process_voice_input(self, fixed_duration=None):
        """
        Complete pipeline: Record → Transcribe → Generate → Speak
        
        Args:
            fixed_duration: If set, record for fixed duration instead of auto-stop
            
        Returns:
            (transcription, response) tuple
        """
        # Record audio
        if fixed_duration:
            audio_data = self.record_audio(fixed_duration)
        else:
            audio_data = self.record_until_silence()
        
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
        print("  🎙️  OFFLINE VOICE ASSISTANT")
        print("="*60)
        print("\n💡 Commands:")
        print("   - Press ENTER to start listening")
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
                continue
        
        print("\n✅ Assistant stopped.\n")


def main():
    """Main entry point"""
    # Configuration
    WHISPER_MODEL = "tiny"  # Options: tiny, base, small, medium, large
    LLM_MODEL = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Check if model exists
    if not os.path.exists(LLM_MODEL):
        print(f"\n❌ Error: Model not found at {LLM_MODEL}")
        print("Please run: ./setup.sh to download the model\n")
        sys.exit(1)
    
    try:
        # Create assistant
        assistant = VoiceAssistant(
            whisper_model=WHISPER_MODEL,
            llm_model_path=LLM_MODEL,
            verbose=True
        )
        
        # Run interactive mode
        assistant.run_interactive()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()




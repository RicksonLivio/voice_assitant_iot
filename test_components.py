#!/usr/bin/env python3
"""
Test individual components of the voice assistant
Run this to diagnose issues with specific parts
"""

import sys
import os

def test_imports():
    """Test if all required libraries can be imported"""
    print("\n" + "="*60)
    print("  TESTING IMPORTS")
    print("="*60 + "\n")
    
    tests = [
        ("torch", "PyTorch"),
        ("whisper", "OpenAI Whisper"),
        ("llama_cpp", "llama-cpp-python"),
        ("pyttsx3", "pyttsx3 (TTS)"),
        ("sounddevice", "sounddevice"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, display_name in tests:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
            passed += 1
        except ImportError as e:
            print(f"✗ {display_name}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

def test_whisper():
    """Test Whisper model loading"""
    print("\n" + "="*60)
    print("  TESTING WHISPER")
    print("="*60 + "\n")
    
    try:
        import whisper
        print("Loading Whisper 'tiny' model...")
        model = whisper.load_model("tiny")
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_llm():
    """Test LLM loading"""
    print("\n" + "="*60)
    print("  TESTING LLM")
    print("="*60 + "\n")
    
    model_path = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found at: {model_path}")
        print("  Run setup.sh to download the model")
        return False
    
    try:
        from llama_cpp import Llama
        print(f"Loading LLM from: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=2,
            verbose=False
        )
        print("✓ Model loaded successfully")
        
        # Test generation
        print("\nTesting generation...")
        response = llm("Hello, how are you?", max_tokens=20, echo=False)
        output = response['choices'][0]['text'].strip()
        print(f"  Response: {output}")
        print("✓ Generation works")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_tts():
    """Test Text-to-Speech"""
    print("\n" + "="*60)
    print("  TESTING TEXT-TO-SPEECH")
    print("="*60 + "\n")
    
    try:
        import pyttsx3
        print("Initializing TTS engine...")
        engine = pyttsx3.init()
        
        # Get voices
        voices = engine.getProperty('voices')
        print(f"✓ TTS initialized")
        print(f"  Available voices: {len(voices)}")
        
        # Test speaking
        print("\nTesting speech output...")
        engine.say("Testing text to speech")
        engine.runAndWait()
        print("✓ TTS works (you should have heard audio)")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_audio():
    """Test audio recording"""
    print("\n" + "="*60)
    print("  TESTING AUDIO RECORDING")
    print("="*60 + "\n")
    
    try:
        import sounddevice as sd
        
        # List devices
        print("Audio devices:")
        devices = sd.query_devices()
        print(devices)
        
        # Test recording
        print("\nTesting microphone (3 seconds)...")
        print("Speak now!")
        
        duration = 3
        sample_rate = 16000
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )
        sd.wait()
        
        print("✓ Recording complete")
        
        # Check if audio was captured
        import numpy as np
        max_amplitude = np.abs(recording).max()
        print(f"  Max amplitude: {max_amplitude}")
        
        if max_amplitude > 100:
            print("✓ Audio captured successfully")
            return True
        else:
            print("⚠ Audio level very low - check your microphone")
            return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_full_pipeline():
    """Test the complete voice assistant pipeline"""
    print("\n" + "="*60)
    print("  TESTING FULL PIPELINE")
    print("="*60 + "\n")
    
    try:
        from voice_assistant import VoiceAssistant
        
        print("Creating VoiceAssistant instance...")
        assistant = VoiceAssistant(
            whisper_model="tiny",
            llm_model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            verbose=False
        )
        print("✓ Assistant created")
        
        print("\nTesting text-only mode...")
        response = assistant.generate_response("What is 2 plus 2?")
        print(f"  Response: {response}")
        
        print("\nTesting TTS...")
        assistant.speak("Testing one two three")
        
        print("\n✓ Pipeline test complete")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  VOICE ASSISTANT COMPONENT TESTS")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Whisper", test_whisper),
        ("LLM", test_llm),
        ("TTS", test_tts),
        ("Audio", test_audio),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60 + "\n")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Your assistant is ready to use.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())




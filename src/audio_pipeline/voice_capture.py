"""
Voice Capture Module using Vosk for offline speech recognition.
"""
import json
from vosk import Model, KaldiRecognizer
import pyaudio
from typing import Optional

class VoiceCapture:
    def __init__(self, model_path: str):
        """
        Initialize voice capture with Vosk model.
        
        Args:
            model_path: Path to the Vosk model directory
        """
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def start_capture(self) -> None:
        """Start capturing audio from microphone."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )
        self.stream.start_stream()
        
    def capture_command(self) -> Optional[str]:
        """
        Capture and recognize a voice command.
        
        Returns:
            str: Recognized text if successful, None otherwise
        """
        data = self.stream.read(4000, exception_on_overflow=False)
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            return result.get("text", "").strip()
        return None
    
    def stop_capture(self) -> None:
        """Stop capturing audio and clean up resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

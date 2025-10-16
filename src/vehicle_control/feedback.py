"""
Voice Feedback Module using pyttsx3 for Windows compatibility.
"""
import pyttsx3
from typing import Optional
import os
import contextlib

class VoiceFeedback:
    def __init__(self, voice: str = "en"):
        """
        Initialize voice feedback system.
        
        Args:
            voice: Language/voice to use (default: English)
        """
        self.engine = None
        try:
            self.engine = pyttsx3.init()
            # Get the voice that matches the language code
            voices = self.engine.getProperty('voices')
            for v in voices:
                if voice in v.id.lower():
                    self.engine.setProperty('voice', v.id)
                    break
                    
            # Initialize with default values from environment or defaults
            self.speed = int(os.getenv("SPEAKING_RATE", "175"))  # Default WPM
            self.pitch = int(os.getenv("VOICE_PITCH", "50"))     # Default pitch
            
            # Set initial properties
            self.engine.setProperty('rate', self.speed)
            self.engine.setProperty('pitch', self.pitch / 50)  # pyttsx3 uses 0-2 range for pitch
        except Exception as e:
            self.cleanup()
            raise e
            
    def cleanup(self):
        """Clean up text-to-speech resources."""
        if hasattr(self, 'engine') and self.engine is not None:
            with contextlib.suppress(Exception):
                self.engine.stop()
            self.engine = None
            
    def __del__(self):
        """Ensure cleanup on destruction."""
        self.cleanup()
        
    def speak(self, text: str) -> bool:
        """
        Convert text to speech using pyttsx3.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception:
            return False
            
    def adjust_voice(self, speed: Optional[int] = None, pitch: Optional[int] = None) -> None:
        """
        Adjust voice parameters.
        
        Args:
            speed: Speech speed (words per minute)
            pitch: Voice pitch (0-99)
        """
        if speed is not None:
            self.speed = max(80, min(450, speed))
            self.engine.setProperty('rate', self.speed)
        if pitch is not None:
            self.pitch = max(0, min(99, pitch))
            self.engine.setProperty('pitch', self.pitch / 50)  # Convert to 0-2 range
            
class AudioFeedback(VoiceFeedback):
    def __init__(self):
        """Initialize the audio feedback system with default English voice."""
        super().__init__(voice="en")
        # Set initial voice parameters for clear ADAS feedback
        self.adjust_voice(speed=150, pitch=60)  # Slightly slower and higher pitch for clarity
        
    def __str__(self) -> str:
        """Return string representation of the audio feedback system."""
        return f"AudioFeedback(voice={self.voice}, speed={self.speed}, pitch={self.pitch})"

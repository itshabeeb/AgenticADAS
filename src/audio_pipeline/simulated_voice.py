"""
Simulated input mode for voice commands testing.
"""
from typing import Optional, Dict

class SimulatedVoiceCapture:
    def __init__(self, *args, **kwargs):
        pass
        
    def start_capture(self):
        """Simulate starting voice capture."""
        pass
        
    def get_command(self) -> Optional[str]:
        """
        Get command from keyboard input instead of microphone.
        The input will be processed by DistilBERT just like voice commands.
        """
        print("\nType your command (examples):")
        print("- Switch to Normal mode")
        print("- Enable Priority mode")
        print("- Set Critical mode")
        print("- Change to Cautious mode")
        print("- Activate Eco mode")
        print("(Press Enter with no text to skip)")
        
        command = input("\nEnter your command: ").strip()
        return command if command else None
        
    def capture_command(self) -> Optional[str]:
        """
        Alias for get_command to match VoiceCapture interface.
        """
        return self.get_command()
        
    def stop_capture(self):
        """Cleanup simulation."""
        pass
"""
AgenticADAS: Intelligent Driver Assistance System
Integrates voice commands, speed sign detection, and dynamic driving modes.
"""
import os
import logging
from dotenv import load_dotenv
import time
from typing import Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from audio_pipeline.voice_capture import VoiceCapture
from audio_pipeline.intent_classifier import IntentClassifier
from vision_pipeline.speed_detector import SpeedDetector
from reasoning_engine.llm_engine import LLMEngine
from vehicle_control.simulator import VehicleSimulator
from vehicle_control.feedback import AudioFeedback

# Load environment variables
load_dotenv()

class AgenticADAS:
    def __init__(self):
        """Initialize the ADAS system with all components."""
        self.voice_capture = None
        self.intent_classifier = None
        self.speed_detector = None
        self.llm_engine = None
        self.vehicle = None
        self.feedback = None
        
        try:
            # Ask for input mode
            print("\n=== AgenticADAS Input Mode Selection ===")
            print("1: Real Input (Camera + Microphone)")
            print("2: Simulated Input (Keyboard)")
            mode = input("Enter mode (1 or 2): ").strip()
            
            if mode == "2":
                # Simulated mode
                from vision_pipeline.simulated_detector import SimulatedSpeedDetector
                from audio_pipeline.simulated_voice import SimulatedVoiceCapture
                
                self.voice_capture = SimulatedVoiceCapture()
                self.speed_detector = SimulatedSpeedDetector()
                logging.info("Running in SIMULATION mode (keyboard input)")
            else:
                # Real input mode
                self.voice_capture = VoiceCapture(
                    model_path=os.getenv("VOSK_MODEL_PATH")
                )
                self.speed_detector = SpeedDetector(
                    model_path=os.getenv("YOLOV11N_MODEL_PATH")
                )
                logging.info("Running in REAL mode (camera + microphone)")
            
            # Common initialization for both modes
            self.intent_classifier = IntentClassifier(
                model_path=os.getenv("DISTILBERT_MODEL_PATH")
            )
            
            # Initialize reasoning engine
            self.llm_engine = LLMEngine(
                model_path=os.getenv("LLM_MODEL_PATH")
            )
            
            # Initialize vehicle control
            self.vehicle = VehicleSimulator()
            self.feedback = AudioFeedback()
        except Exception as e:
            self.cleanup()
            raise e
    
    def cleanup(self):
        """Clean up resources properly."""
        try:
            if self.voice_capture:
                self.voice_capture.stop_capture()
            if self.speed_detector:
                self.speed_detector.stop_capture()
            if self.feedback:
                self.feedback.cleanup()
            if self.llm_engine:
                self.llm_engine.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()
        
    def start(self):
        """Start the ADAS system."""
        # Start capture devices
        self.voice_capture.start_capture()
        self.speed_detector.start_capture()
        
        print("AgenticADAS started. Listening for commands...")
        
        while True:
            try:
                # 1. Voice command processing
                command = self.voice_capture.capture_command()
                if command:
                    intent_data = self.intent_classifier.get_intent_data(command)
                    print(f"\nVoice Command: {command}")
                    print(f"Intent Classification: {intent_data}")
                else:
                    continue
                    
                # 2. Speed sign detection
                speed_info = self.speed_detector.detect_speed_limit()
                
                # Ensure we have a valid dictionary
                if speed_info is None:
                    speed_info = {"speed_limit": 0, "confidence": 0.0}
                elif isinstance(speed_info, dict):
                    # Extract speed from detection output if available
                    if 'sign_type' in speed_info:
                        sign_type = speed_info['sign_type']
                        if isinstance(sign_type, str) and sign_type.startswith('Speed'):
                            try:
                                speed_value = int(sign_type[5:])  # Extract number from 'Speed20', 'Speed40', etc.
                                speed_info = {
                                    'speed_limit': speed_value,
                                    'confidence': speed_info.get('confidence', 0.0)
                                }
                            except (ValueError, IndexError):
                                speed_info = {"speed_limit": 0, "confidence": 0.0}
                    elif 'speed_limit' not in speed_info:
                        speed_info = {"speed_limit": 0, "confidence": 0.0}
                else:
                    speed_info = {"speed_limit": 0, "confidence": 0.0}
                    
                print(f"Speed Detection: {speed_info}")
                    
                # 3. LLM reasoning
                decision = self.llm_engine.reason(intent_data, speed_info)
                print(f"Selected Mode: {decision['mode']}")
                print(f"Reasoning: {decision['reason']}\n")
                
                # 4. Apply vehicle control
                self.vehicle.set_mode(decision["mode"])
                
                # 5. Provide audio feedback
                self.feedback.speak(decision["reason"])
                
            except KeyboardInterrupt:
                print("\nShutting down AgenticADAS...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
                
        # Cleanup
        self.voice_capture.stop_capture()
        self.speed_detector.stop_capture()

if __name__ == "__main__":
    adas = AgenticADAS()
    adas.start()

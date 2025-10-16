"""
Main application entry point for Agentic ADAS.
"""
import os
from dotenv import load_dotenv
from src.audio_pipeline.voice_capture import VoiceCapture
from src.audio_pipeline.intent_classifier import IntentClassifier
from src.vision_pipeline.speed_detector import SpeedDetector
from src.reasoning_engine.llm_engine import LLMEngine
from src.vehicle_control.simulator import VehicleSimulator
from src.vehicle_control.feedback import VoiceFeedback
import logging
from pythonjsonlogger import jsonlogger

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

def main():
    """Initialize and run the Agentic ADAS system."""
    try:
        # Initialize components
        voice_capture = VoiceCapture(os.getenv("VOSK_MODEL_PATH"))
        intent_classifier = IntentClassifier(os.getenv("DISTILBERT_MODEL_PATH"))
        speed_detector = SpeedDetector(os.getenv("YOLO_MODEL_PATH"))
        llm_engine = LLMEngine(os.getenv("PHI3_MODEL_PATH"))
        vehicle_sim = VehicleSimulator()
        feedback = VoiceFeedback()
        
        # Start capture systems
        voice_capture.start_capture()
        speed_detector.start_capture()
        
        logger.info("Agentic ADAS initialized successfully")
        
        try:
            while True:
                # Process voice command
                command = voice_capture.capture_command()
                if command:
                    intent_data = intent_classifier.get_intent_data(command)
                    logger.info("Processed voice command", extra={"intent_data": intent_data})
                    
                    # Get speed limit
                    speed_data = speed_detector.detect_speed_limit()
                    logger.info("Detected speed limit", extra={"speed_data": speed_data})
                    
                    # Generate reasoned response
                    decision = llm_engine.reason(intent_data, speed_data or {})
                    logger.info("Generated decision", extra={"decision": decision})
                    
                    # Apply vehicle control
                    vehicle_state = vehicle_sim.set_mode(decision["mode"])
                    logger.info("Updated vehicle state", extra={"vehicle_state": vehicle_state})
                    
                    # Provide feedback
                    feedback.speak(decision["reason"])
                    
        except KeyboardInterrupt:
            logger.info("Shutting down Agentic ADAS")
            
    except Exception as e:
        logger.error("Error initializing Agentic ADAS", exc_info=True)
        
    finally:
        # Cleanup
        voice_capture.stop_capture()
        speed_detector.stop_capture()

if __name__ == "__main__":
    main()
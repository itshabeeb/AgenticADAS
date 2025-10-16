"""
TinyLLM Reasoning Engine using Microsoft Phi-3 Mini model.
"""
from llama_cpp import Llama
import json
import logging
from typing import Dict, Any

class LLMEngine:
    def __init__(self, model_path: str):
        """
        Initialize the LLM engine with Phi-3 Mini model.
        
        Args:
            model_path: Path to the quantized model file
        """
        self.llm = None
        try:
            # Optimize memory usage
            self.llm = Llama(
                model_path=model_path,
                n_ctx=512,          # Reduced context window
                n_threads=2,        # Reduced thread count
                n_batch=8,         # Smaller batch size
                n_gpu_layers=0,    # Force CPU only
                seed=42,           # Fixed seed for deterministic behavior
                vocab_only=False,  # Load full model
                offload_kqv=True   # Offload key/query/value matrices
            )
            logging.info("LLM initialized with optimized memory settings")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {str(e)}")
            self.cleanup()
            raise e
            
    def cleanup(self):
        """Clean up LLM resources."""
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # Use proper cleanup for llama-cpp-python
                self.llm = None
            except Exception as e:
                print(f"Error cleaning up LLM: {e}")
                
    def __del__(self):
        """Ensure cleanup on destruction."""
        self.cleanup()
        
    def _generate_prompt(self, driver_intent: Dict, speed_limit: Dict) -> str:
        """
        Generate structured prompt for the model.
        
        Args:
            driver_intent: Intent classification results
            speed_limit: Speed limit detection results
            
        Returns:
            str: Formatted prompt
        """
        # Calculate adjusted speed based on intent
        speed = speed_limit.get('speed_limit', 0)
        intent = driver_intent['intent']
        
        if speed > 0:  # Valid speed limit detected
            if intent == "Normal":
                adjusted_speed = speed  # Standard speed
                speed_info = f"Following exact speed limit of {speed} km/h"
            elif intent == "Priority":
                adjusted_speed = speed + 5  # Slightly higher
                speed_info = f"Adjusting speed from {speed} to {adjusted_speed} km/h for priority mode (+5 km/h)"
            elif intent == "Critical":
                adjusted_speed = speed + 10  # Much higher
                speed_info = f"Increasing speed from {speed} to {adjusted_speed} km/h for critical mode (+10 km/h)"
            elif intent == "Cautious":
                adjusted_speed = max(speed - 5, 0)  # Slightly lower
                speed_info = f"Reducing speed from {speed} to {adjusted_speed} km/h for cautious mode (-5 km/h)"
            elif intent == "Eco":
                adjusted_speed = max(speed - 10, 0)  # Much lower
                speed_info = f"Lowering speed from {speed} to {adjusted_speed} km/h for eco mode (-10 km/h)"
        else:
            speed_info = "No valid speed limit detected, maintaining safe operation"
            adjusted_speed = 0

        return f"""Context:
Command: "{driver_intent['text']}"
Intent: {intent} ({driver_intent['confidence']:.2f})
Speed: {speed_info}

Modes: Normal(+0), Priority(+5), Critical(+10), Cautious(-5), Eco(-10)

JSON response:
{{"mode": str, "reason": str, "adjusted_speed": int}}

Response:"""

    def reason(self, driver_intent: Dict, speed_limit: Dict) -> Dict[str, Any]:
        """
        Generate reasoned response based on inputs.
        
        Args:
            driver_intent: Intent classification results
            speed_limit: Speed limit detection results
            
        Returns:
            dict: Decision including mode and explanation
        """
        intent = driver_intent['intent']
        speed = speed_limit.get('speed_limit', 0)
        
        # Calculate adjusted speed based on mode
        if speed > 0:
            if intent == "Normal":
                adjusted_speed = speed
                reason = f"Following exact speed limit of {speed} km/h in Normal mode"
            elif intent == "Priority":
                adjusted_speed = speed + 5
                reason = f"Priority mode: Adjusting speed from {speed} to {adjusted_speed} km/h (+5 km/h)"
            elif intent == "Critical":
                adjusted_speed = speed + 10
                reason = f"Critical mode: Increasing speed from {speed} to {adjusted_speed} km/h (+10 km/h)"
            elif intent == "Cautious":
                adjusted_speed = max(speed - 5, 0)
                reason = f"Cautious mode: Reducing speed from {speed} to {adjusted_speed} km/h (-5 km/h)"
            elif intent == "Eco":
                adjusted_speed = max(speed - 10, 0)
                reason = f"Eco mode: Lowering speed from {speed} to {adjusted_speed} km/h (-10 km/h)"
            else:
                adjusted_speed = speed
                reason = f"Maintaining standard speed of {speed} km/h"
        else:
            adjusted_speed = 0
            reason = "No valid speed limit detected, maintaining safe operation"

        # Return the decision dictionary
        return {
            "mode": intent,
            "reason": reason,
            "adjusted_speed": adjusted_speed
        }

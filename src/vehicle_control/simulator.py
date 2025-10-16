"""
Vehicle Control Simulation Module.
"""
from typing import Dict
import time

class VehicleSimulator:
    def __init__(self):
        """Initialize vehicle simulator with default parameters."""
        self.current_mode = "Normal"
        self.mode_params = {
            "Eco": {
                "max_acceleration": 0.5,
                "throttle_response": 0.3
            },
            "Normal": {
                "max_acceleration": 0.7,
                "throttle_response": 0.6
            },
            "Sport": {
                "max_acceleration": 1.0,
                "throttle_response": 0.9
            }
        }
        # Add other modes to mode_params so set_mode accepts them
        self.mode_params.update({
            "Priority": self.mode_params["Normal"],
            "Critical": self.mode_params["Sport"],
            "Cautious": self.mode_params["Eco"],
        })
        # Current dynamic state (for simulation purposes)
        self.current_speed = 0.0  # km/h
        self.target_speed = 0.0
        # Define supported modes and their speed adjustment multipliers or offsets
        self.mode_speed_policy = {
            # mode: (type, value) where type is 'offset' or 'multiplier'
            "Normal": ("offset", 0.0),      # follow speed limit
            "Priority": ("offset", 5.0),    # +5 km/h
            "Critical": ("offset", 10.0),   # +10 km/h
            "Cautious": ("offset", -5.0),   # -5 km/h
            "Eco": ("offset", -10.0),       # -10 km/h
        }
        
    def set_mode(self, mode: str) -> Dict:
        """
        Set vehicle driving mode and return parameters.
        
        Args:
            mode: Driving mode (Eco, Normal, Sport)
            
        Returns:
            dict: Current vehicle parameters
        """
        if mode in self.mode_params:
            self.current_mode = mode
            return {
                "mode": mode,
                "params": self.mode_params[mode],
                "timestamp": time.time()
            }
        return {
            "mode": self.current_mode,
            "params": self.mode_params[self.current_mode],
            "timestamp": time.time()
        }
        
    def get_current_state(self) -> Dict:
        """
        Get current vehicle state.
        
        Returns:
            dict: Current vehicle state information
        """
        return {
            "mode": self.current_mode,
            "params": self.mode_params[self.current_mode],
            "timestamp": time.time()
        }

    def apply_decision(self, decision: Dict, detected_speed_limit: float = None, reset: bool = False, ticks: int = 1) -> Dict:
        """
        Apply a decision from the reasoning engine and compute a target speed.

        Args:
            decision: JSON-like dict from LLM with at least a 'mode' key.
            detected_speed_limit: current detected speed limit in km/h (optional).

        Returns:
            dict: Updated vehicle state including computed target speed.
        """
        mode = decision.get("mode", self.current_mode)
        # Validate mode
        if mode not in self.mode_speed_policy:
            # default to Normal if unknown
            mode = "Normal"

        self.current_mode = mode

        # Optionally reset current speed for isolated evaluation (useful for tests)
        if reset:
            self.current_speed = 0.0

        # Compute target speed based on policy and detected speed limit
        if detected_speed_limit is None:
            # if no detected limit, keep current speed or apply conservative default
            self.target_speed = max(0.0, self.current_speed)
        else:
            policy_type, value = self.mode_speed_policy[mode]
            if policy_type == "offset":
                self.target_speed = max(0.0, detected_speed_limit + value)
            else:
                # multiplier (not used currently)
                self.target_speed = max(0.0, detected_speed_limit * value)

        # Simulate acceleration/deceleration over `ticks` steps
        params = self.mode_params.get(mode, self.mode_params.get("Normal"))
        max_acc = params.get("max_acceleration", 0.7)
        for _ in range(max(1, int(ticks))):
            speed_diff = self.target_speed - self.current_speed
            step = max(-max_acc, min(max_acc, speed_diff))
            self.current_speed = round(self.current_speed + step, 2)

        state = {
            "mode": self.current_mode,
            "detected_speed_limit": detected_speed_limit,
            "target_speed": self.target_speed,
            "current_speed": self.current_speed,
            "params": params,
            "timestamp": time.time()
        }

        return state

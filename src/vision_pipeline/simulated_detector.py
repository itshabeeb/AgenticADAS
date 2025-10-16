"""
Simulated input mode for vision pipeline testing.
"""
from typing import Optional, Dict

class SimulatedSpeedDetector:
    def __init__(self, *args, **kwargs):
        self.current_sign = None
        
    def start_capture(self, *args, **kwargs):
        """Simulate starting capture - no actual device needed."""
        pass
        
    def detect_speed_limit(self) -> Optional[Dict]:
        """
        Get speed limit from keyboard input.
        Returns same format as real detector.
        """
        print("\nEnter the sign you see (examples):")
        print("- Speed20")
        print("- Speed40")
        print("- Speed60")
        print("- Stop")
        print("- none (for no sign detected)")
        print("(Press Enter to keep previous sign)")
        
        sign = input("\nEnter what you see: ").strip().lower()
        
        if not sign and self.current_sign:
            return self.current_sign
            
        # Process the input
        if sign == 'none':
            self.current_sign = {'speed_limit': 0, 'confidence': 0.0}
        elif sign == 'stop':
            self.current_sign = {
                'speed_limit': 0,
                'confidence': 0.95,
                'bbox': [100, 100, 200, 200],
                'sign_type': 'Stop'
            }
        elif sign.startswith('speed'):
            try:
                speed = int(sign.replace('speed', ''))
                if speed in [20, 40, 60]:
                    self.current_sign = {
                        'speed_limit': speed,
                        'confidence': 0.95,
                        'bbox': [100, 100, 200, 200],
                        'sign_type': f'Speed{speed}'
                    }
                else:
                    print("Invalid speed limit. Using no detection.")
                    self.current_sign = {'speed_limit': 0, 'confidence': 0.0}
            except ValueError:
                print("Invalid input format. Using no detection.")
                self.current_sign = {'speed_limit': 0, 'confidence': 0.0}
        else:
            print("Invalid sign type. Using no detection.")
            self.current_sign = {'speed_limit': 0, 'confidence': 0.0}
            
        return self.current_sign
        
    def stop_capture(self):
        """Cleanup simulation."""
        pass
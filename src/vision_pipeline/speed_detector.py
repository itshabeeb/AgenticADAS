"""
Speed Sign Detection Module using YOLOv11n and OpenCV.
Optimized for Raspberry Pi 5 deployment with real-time detection.
"""
import os
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SpeedDetector:
    # Speed sign classes (matching the trained model's classes)
    CLASSES = {
        0: 'Speed20',
        1: 'Speed40',
        2: 'Speed60',
        3: 'Stop'
    }
    
    # Detection settings
    CONFIDENCE_THRESHOLD = float(os.getenv('SPEED_DETECTION_CONFIDENCE', '0.35'))  # Further lowered for better detection rate
    
    # Base model path (YOLOv11n architecture)
    BASE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'models', 'yolov11n.pt')
    
    # Detection counters for stability
    REQUIRED_DETECTIONS = {
        'Speed20': 2,  # Reduced for faster response
        'Speed40': 2,  # Reduced for faster response
        'Speed60': 2,  # Reduced for faster response
        'Stop': 3      # Still more detections for STOP signs
    }
    
    def __init__(self, model_path: str):
        """
        Initialize speed sign detector with YOLOv11n model.
        
        Args:
            model_path: Path to the trained speed sign model (speed_sign_board.pt)
        """
        try:
            # First check if we have the base YOLOv11n model
            if not os.path.exists(self.BASE_MODEL_PATH):
                logging.error(f"Base YOLOv11n model not found at {self.BASE_MODEL_PATH}")
                raise FileNotFoundError(f"Please ensure yolov11n.pt is in the models directory")
                
            # Load the base model first
            self.model = YOLO(self.BASE_MODEL_PATH)
            
            # Load the trained weights
            self.model = YOLO(model_path)
            self.model.conf = self.CONFIDENCE_THRESHOLD  # Set confidence threshold
            self.cap = None
            
            logging.info("Successfully loaded both base YOLOv11n and trained speed sign models")
            
            # Initialize detection counters
            self.detection_counts = {class_name: 0 for class_name in self.CLASSES.values()}
            self.current_detection = None
            
            logging.info(f"YOLOv11n model loaded successfully with conf={self.CONFIDENCE_THRESHOLD}")
        except Exception as e:
            logging.error(f"Failed to load YOLOv11n model: {str(e)}")
            raise
            
    def start_capture(self, device: str = "0") -> None:
        """
        Start video capture from camera.
        
        Args:
            device: Camera device ("0" for USB camera, "picamera0" for Pi camera)
        """
        try:
            # Use YOLO's standard input size
            self.input_size = (640, 640)  # Standard YOLO input size
            
            # Define camera resolution
            self.camera_width = 1280
            self.camera_height = 720
            
            if device == "picamera0":
                # Raspberry Pi camera setup
                self.cap = cv2.VideoCapture(0)
            else:
                # USB camera setup
                self.cap = cv2.VideoCapture(int(device))
                
            # Set resolution to slightly higher than needed for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera device")
                
            # Verify actual camera resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"Camera initialized: {device} at {actual_width}x{actual_height}")
        except Exception as e:
            logging.error(f"Camera initialization failed: {str(e)}")
            raise
        
    def detect_speed_limit(self) -> Optional[Dict]:
        """
        Detect speed limit sign in current frame.
        Uses YOLOv11n model optimized for Raspberry Pi 5.
        Implements detection stability with counting mechanism.
        
        Returns:
            dict: Detection results including speed limit, confidence, and bbox
        """
        if not self.cap or not self.cap.isOpened():
            logging.warning("Camera not initialized")
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Failed to read frame from camera")
            return None
            
        # Preprocess frame
        try:
            # Store original frame for visualization
            display_frame = frame.copy()
            
            # Resize to YOLO input size while maintaining aspect ratio
            h, w = frame.shape[:2]
            scale = min(self.input_size[0]/w, self.input_size[1]/h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Create black canvas of target size
            frame_padded = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            
            # Center the resized image on black canvas
            x_offset = (self.input_size[0] - new_w) // 2
            y_offset = (self.input_size[1] - new_h) // 2
            frame_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
            
            logging.info("Running detection on preprocessed frame")
            results = self.model(frame_padded, verbose=False)  # Disable verbose YOLO output
            
            # Log all detections, even low confidence ones
            if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    conf = float(box.conf)
                    cls_id = int(box.cls[0])
                    cls_name = self.CLASSES.get(cls_id, "Unknown")
                    logging.info(f"Detection {i+1}: class={cls_name}, confidence={conf:.3f} {'(below threshold)' if conf < self.CONFIDENCE_THRESHOLD else ''}")
            
            # Process detection results
            if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                # Get detection with highest confidence
                boxes = results[0].boxes
                confidences = [float(conf) for conf in boxes.conf]
                max_conf_idx = np.argmax(confidences)
                
                # Get the box with highest confidence
                box = boxes[max_conf_idx]
                confidence = float(box.conf)
                class_id = int(box.cls[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # Debug logging
                logging.info(f"Raw detection: class_id={class_id}, conf={confidence:.2f}")
                
                # Draw detection on display frame
                self._draw_detection(display_frame, bbox, self.CLASSES.get(class_id, "Unknown"), confidence)
                
                # Show the frame
                cv2.imshow('Speed Sign Detection', display_frame)
                cv2.waitKey(1)  # Update window with 1ms delay
                
                # Continue with detection processing
                return self._process_detection(class_id, confidence, bbox)
            else:
                # Show frame even when no detections
                cv2.imshow('Speed Sign Detection', display_frame)
                cv2.waitKey(1)
            
        except Exception as e:
            logging.error(f"Error during detection: {str(e)}")
            return None
            
            # Only process high-confidence detections
            if confidence > self.CONFIDENCE_THRESHOLD:
                detected_class = self.CLASSES.get(class_id)
                
                # Debug logging
                logging.info(f"Detection: class={detected_class}, conf={confidence:.2f}, counts={self.detection_counts}")
                
                # Reset other class counters if we detect a new sign
                if self.current_detection != detected_class:
                    self.detection_counts = {class_name: 0 for class_name in self.CLASSES.values()}
                    self.current_detection = detected_class
                    logging.info(f"New sign detected: {detected_class}, reset counts")
                
                # Increment detection counter for this class
                self.detection_counts[detected_class] += 1
                logging.info(f"Updated counts for {detected_class}: {self.detection_counts[detected_class]}")
                
                # Check if we have enough consistent detections
                if self.detection_counts[detected_class] >= self.REQUIRED_DETECTIONS[detected_class]:
                    # Convert speed limit text to number
                    if detected_class == 'Stop':
                        speed_value = 0
                    else:
                        speed_value = int(detected_class.replace('Speed', ''))
                    
                    # Log detection for monitoring
                    logging.info(f"Confirmed detection: {detected_class} (conf: {confidence:.2f})")
                    
                    return {
                        "speed_limit": speed_value,
                        "confidence": confidence,
                        "bbox": bbox.tolist(),  # For visualization
                        "sign_type": detected_class  # Original class name
                    }
                
        return None
    
    def _process_detection(self, class_id: int, confidence: float, bbox: np.ndarray) -> Optional[Dict]:
        """
        Process a detection and handle the detection counting logic.
        
        Args:
            class_id: The detected class ID
            confidence: The detection confidence
            bbox: The bounding box coordinates
            
        Returns:
            dict: Detection results if valid, None otherwise
        """
        if confidence > self.CONFIDENCE_THRESHOLD:
            detected_class = self.CLASSES.get(class_id)
            if detected_class is None:
                return None
                
            # Debug logging
            logging.info(f"Processing detection: class={detected_class}, conf={confidence:.2f}")
            logging.info(f"Current counts: {self.detection_counts}")
            
            # Reset other class counters if we detect a new sign
            if self.current_detection != detected_class:
                self.detection_counts = {class_name: 0 for class_name in self.CLASSES.values()}
                self.current_detection = detected_class
                logging.info(f"New sign detected: {detected_class}, reset counts")
            
            # Increment detection counter for this class
            self.detection_counts[detected_class] += 1
            logging.info(f"Updated counts for {detected_class}: {self.detection_counts[detected_class]}")
            
            # Check if we have enough consistent detections
            if self.detection_counts[detected_class] >= self.REQUIRED_DETECTIONS[detected_class]:
                # Convert speed limit text to number
                if detected_class == 'Stop':
                    speed_value = 0
                else:
                    speed_value = int(detected_class.replace('Speed', ''))
                
                logging.info(f"Confirmed detection: {detected_class} (speed: {speed_value}, conf: {confidence:.2f})")
                
                return {
                    "speed_limit": speed_value,
                    "confidence": confidence,
                    "bbox": bbox.tolist(),
                    "sign_type": detected_class
                }
                
        return None
        
    def _draw_detection(self, frame: np.ndarray, bbox: np.ndarray, label: str, conf: float) -> None:
        """
        Draw detection box and label on frame.
        
        Args:
            frame: The frame to draw on
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            label: Class label to display
            conf: Detection confidence
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for detections
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label with confidence
        label_text = f"{label} ({conf:.2f})"
        
        # Get label size and set background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1-label_h-baseline-10), (x1+label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1-baseline-5), font, font_scale, (0, 0, 0), font_thickness)
        
    def stop_capture(self) -> None:
        """Stop video capture and clean up resources."""
        try:
            cv2.destroyAllWindows()  # Close any open windows
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                logging.info("Camera released successfully")
        except Exception as e:
            logging.error(f"Error releasing camera: {str(e)}")
            
    def __del__(self):
        """Ensure cleanup on destruction."""
        self.stop_capture()

# AgenticADAS - AI Development Instructions

This document provides guidance for AI coding agents working with the AgenticADAS codebase.

## Project Structure

```
src/
├── audio_pipeline/      # Voice processing pipeline
│   ├── voice_capture.py    # Vosk-based speech recognition
│   └── intent_classifier.py # DistilBERT intent classification
├── vision_pipeline/     # Speed sign detection pipeline
│   └── speed_detector.py   # YOLOv8n + OpenCV implementation
├── reasoning_engine/    # Decision making system
│   └── llm_engine.py      # Phi-3 Mini LLM integration
└── vehicle_control/     # Output handling
    ├── simulator.py       # Vehicle mode simulation
    └── feedback.py        # Voice feedback system
```

## Key Integration Points

1. **Pipeline Integration**
   - Audio and Vision pipelines run independently
   - Results converge at `LLMEngine.reason()` in `reasoning_engine/llm_engine.py`
   - See `main.py` for the orchestration flow

2. **Model Integration**
   - Models are loaded via environment variables (paths in `.env`)
   - All models run offline for edge deployment
   - Model quantization/optimization handled in respective pipeline modules

3. **Data Flow**
   - Each pipeline produces structured dictionary outputs
   - Standard format maintained throughout the system
   - Example: `{"speed_limit": 60, "confidence": 0.95}`

## Development Patterns

1. **Error Handling**
   - All external operations (I/O, model inference) must be try-except wrapped
   - Use Optional types for potentially failing operations
   - Log errors with context using JSON logger

2. **Configuration**
   - All configurable values come from environment variables
   - Use `example.env` as template for required variables
   - Access via `os.getenv()` with appropriate type casting

3. **Testing**
   - Unit tests focus on pipeline integration points
   - Mock heavy I/O operations (camera, microphone)
   - Use pytest fixtures for model loading

## Common Tasks

1. **Adding a New Model**
   - Add model path to `example.env`
   - Create wrapper class in appropriate pipeline
   - Update main loop in `main.py`

2. **Modifying Pipeline Logic**
   - Each pipeline has clear input/output contracts
   - Maintain backward compatibility when possible
   - Update docstrings and type hints

3. **Performance Optimization**
   - Profile before optimizing
   - Consider edge device constraints
   - Optimize inference batch sizes
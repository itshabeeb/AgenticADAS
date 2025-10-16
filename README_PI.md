Raspberry Pi 5 (8GB) Deployment Guide for AgenticADAS

This guide walks through preparing a Raspberry Pi 5 for running AgenticADAS with offline models.

1) OS
- Use Raspberry Pi OS (64-bit) or Debian Bookworm (arm64). Update and upgrade packages:

  sudo apt update && sudo apt upgrade -y

2) System packages
- Install build and runtime dependencies (portaudio, ffmpeg, etc.):

  sudo apt install -y python3-venv python3-pip build-essential libatlas-base-dev libopenblas-dev libsndfile1 ffmpeg git wget unzip libportaudio2

- Install eSpeak for TTS:

  sudo apt install -y espeak

3) Python environment
- Create and activate a virtual environment in the project directory:

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip

- Install Python packages:

  pip install -r requirements.txt

4) Model files and storage
- Place models under a `models/` directory in the project root and update `.env` with paths.
  - Vosk: a small offline model compatible with Raspberry Pi
  - DistilBERT: prefer a quantized or smaller model; consider exporting to ONNX
  - YOLOv8n: use the smallest `n` variant, convert to ONNX if needed
  - Phi-3 Mini: ensure you have a quantized Llama/phi3 model suitable for llama-cpp-python

5) Performance tips
- Use CPU-bound optimizations: set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to limit thread counts.
- Prefer model quantization (4-bit/8-bit) and use `llama-cpp-python` with the quantized model files.
- Lower camera resolution to 640x480 and reduce frame rate.
- Consider swapping heavy NLP inference to a separate edge device if latency is critical.

6) Running
- Activate the venv and run:

  source .venv/bin/activate
  python main.py

7) Troubleshooting
- If `pip install` fails for torch or other large wheels, use the official PyPI wheels or build from source, or install platform-specific wheel files.
- Some packages (e.g., torch) require specific wheels for ARM64; consult PyTorch's instructions for Raspberry Pi.

8) Notes
- This guide assumes offline operation for all models.
- For real deployment, test thermal and power considerations; Raspberry Pi 5 can throttle under sustained load.

from setuptools import setup, find_packages

setup(
    name="agentic_adas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vosk>=0.3.45",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "llama-cpp-python>=0.2.0",
        "espeak>=0.5.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "ultralytics>=8.0.0",
        "python-json-logger>=2.0.7",
        "pytest>=7.4.0",
    ],
    author="itshabeeb",
    description="An Agentic Advanced Driver Assistance System using dual-pipeline architecture",
    python_requires=">=3.8",
)
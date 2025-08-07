# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an intelligent eye disease diagnosis system built on the Ultralytics YOLO framework. The project combines computer vision, deep learning, and AI-powered medical assistance to detect and classify eye diseases from retinal images.

### Core Architecture

**Main Components:**
- **ultralytics-main/**: Contains the core YOLO framework with custom modifications
- **visualization_2.0.py**: Complete PyQt5 GUI application for the diagnosis system
- **eyes_train.py**: Training script for the eye disease classification model
- **Custom Model**: AKConv-based YOLO model located at `ultralytics-main/self_model/AKConv_best_moudle/`

**Key Technologies:**
- Ultralytics YOLO11 for object detection and classification
- PyQt5 for desktop GUI interface
- DeepSeek API integration for AI-powered treatment recommendations
- OpenCV for image processing
- Matplotlib for visualization and reporting

### Dataset Structure

The eye disease dataset is organized with 8 classes:
- **A**: AMD (Age-related Macular Degeneration)
- **N**: Normal
- **D**: Diabetic Retinopathy  
- **G**: Glaucoma
- **C**: Cataract
- **H**: Hypertensive Retinopathy
- **M**: Myopia
- **O**: Other

## Common Development Tasks

### Training the Model

```bash
# Basic training command
python eyes_train.py

# The script automatically checks for dataset existence and runs training with optimized parameters
```

**Training Configuration:**
- Model: `yolo11n-cls.pt` (YOLOv11 classification)
- Epochs: 200
- Batch size: 32
- Image size: 512x512
- Mixed precision training enabled
- Cosine learning rate scheduling

### Running the GUI Application

```bash
python visualization_2.0.py
```

**Key Features:**
- Load and display medical images
- Real-time eye disease detection
- AI-powered treatment recommendations via DeepSeek API
- Batch processing capabilities
- Historical record management
- Statistical reporting with visualizations

### Model Operations

**Using the CLI:**
```bash
# Prediction
yolo predict model=ultralytics-main/self_model/AKConv_best_moudle/best.pt source=path/to/image.jpg

# Validation  
yolo val model=best.pt data=ultralytics-main/datasets/eyes_split/eyes_disease.yaml

# Export model
yolo export model=best.pt format=onnx
```

**Python API:**
```python
from ultralytics import YOLO

# Load model
model = YOLO("ultralytics-main/self_model/AKConv_best_moudle/best.pt")

# Predict
results = model.predict("image.jpg", conf=0.5)
```

## Configuration Management

### Dataset Configuration
- **Location**: `ultralytics-main/datasets/eyes_split/eyes_disease.yaml`
- **Structure**: Standard YOLO classification format with train/val splits
- **Classes**: 8 eye disease categories mapped to single letters

### Dependencies
- **Core**: ultralytics, PyQt5, opencv-python, numpy, matplotlib
- **Optional**: requests (for DeepSeek API), pandas, seaborn
- **Development**: pytest, coverage tools as defined in pyproject.toml

## Important File Paths

**Models:**
- Primary model: `ultralytics-main/self_model/AKConv_best_moudle/best.pt`
- ONNX export: `ultralytics-main/self_model/AKConv_best_moudle/best.onnx`
- Base models: `yolo11n.pt`, `yolo11n-cls.pt`, `yolo11x-cls.pt`

**Data:**
- Dataset config: `ultralytics-main/datasets/eyes_split/eyes_disease.yaml`
- Training images: `ultralytics-main/datasets/eyes_split/train/`
- Validation images: `ultralytics-main/datasets/eyes_split/val/`

**Scripts:**
- GUI application: `visualization_2.0.py`
- Training script: `ultralytics-main/eyes_train.py`
- Jupyter notebook: `ultralytics-main/eyes_train.ipynb`

## Testing and Quality Assurance

### Running Tests
```bash
# From ultralytics-main directory
pytest tests/

# Specific test categories
pytest tests/test_engine.py  # Core engine tests
pytest tests/test_python.py  # Python API tests
pytest tests/test_cli.py     # CLI functionality tests
```

### Code Quality Tools
```bash
# Code formatting (from ultralytics-main/)
yapf --in-place --recursive ultralytics/

# Linting
ruff check ultralytics/

# Type checking would typically use mypy (not configured in this project)
```

## API Integration

### DeepSeek AI Integration
The system integrates with DeepSeek's chat completion API for medical advice:
- Endpoint: `https://api.deepseek.com/v1/chat/completions`
- Model: `deepseek-chat`
- Provides treatment recommendations based on detected diseases
- Falls back to built-in advice if API is unavailable

### History and Data Management
- **History Storage**: `~/EyeDiseaseDetectorHistory/history.json`
- **Format**: JSON records with timestamps, disease classifications, and confidence scores
- **Features**: Exportable reports, batch processing summaries, visualization charts

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Ensure the model path in `visualization_2.0.py` line 377 points to the correct location
2. **Dataset Path Issues**: Update paths in `eyes_train.py` and dataset YAML to match your system
3. **GUI Display Problems**: Verify PyQt5 installation and system compatibility
4. **API Connection Failures**: Check DeepSeek API key validity and network connectivity

### Performance Optimization
- Enable GPU acceleration by setting `device=0` in training/inference
- Use mixed precision training (`amp=True`) for better performance
- Adjust batch size based on available GPU memory
- Consider model pruning/quantization for deployment scenarios
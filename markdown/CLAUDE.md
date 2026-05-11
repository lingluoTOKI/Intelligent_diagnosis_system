# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an intelligent eye disease diagnosis system built on the Ultralytics YOLO framework. The project combines computer vision, deep learning, AI-powered medical assistance, and voice interaction to detect and classify eye diseases from retinal images.

### Core Architecture

**Main Applications:**
- **ultralytics-main/visualization2.0.py**: Primary PyQt5 GUI application with complete feature set
- **visualization_test1.py & visualization_test2.py**: Development/testing versions of the GUI
- **ultralytics-main/**: Modified YOLO framework with custom AKConv modules
- **test/**: Voice recognition and integration testing modules

**Key Technologies:**
- Ultralytics YOLO11 for object detection and classification with custom AKConv enhancements
- PyQt5 for desktop GUI interface with advanced styling
- DeepSeek API integration for AI-powered medical treatment recommendations
- Voice Recognition: Speech-to-text (Google API + Vosk fallback) and Text-to-speech (pyttsx3)
- OpenCV for image processing and computer vision
- Matplotlib for statistical visualization and reporting

### Voice Integration Architecture

**Voice Components:**
- **test/enhanced_voice_recognition.py**: Advanced voice recognition manager with error handling
- **test/voice_integration.py**: Voice system integration layer
- **vosk-model-cn-0.22/**: Local Chinese voice recognition model for offline functionality
- **Google Speech API**: Primary online voice recognition service
- **pyttsx3**: Text-to-speech synthesis for AI response playback

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

### Running the Main Application

```bash
# Primary GUI application (main production version)
python ultralytics-main/visualization2.0.py

# Alternative test versions for development
python visualization_test1.py
python visualization_test2.py
```

### Training the Model

```bash
# From ultralytics-main directory
cd ultralytics-main
python eyes_train.py

# Or using the Jupyter notebook
jupyter notebook eyes_train.ipynb
```

**Training Configuration:**
- Model: `yolo11n-cls.pt` (YOLOv11 classification)
- Epochs: 200
- Batch size: 32
- Image size: 512x512
- Mixed precision training enabled
- Cosine learning rate scheduling

### Voice Feature Testing

```bash
# Test voice recognition components
python test/voice_test_demo.py
python test/smart_voice_demo.py

# Test voice integration with GUI
python test/test_smart_voice_integration.py

# Install voice dependencies
python test/install_voice_deps.py
```

### Application Features

**Core Medical Features:**
- Load and display retinal images (jpg, png, bmp formats)
- Real-time eye disease classification using custom AKConv-enhanced YOLO11
- Confidence score display and threshold adjustment
- AI-powered treatment recommendations via DeepSeek API
- Batch processing capabilities for multiple images
- Historical record management with exportable reports
- Statistical analysis with matplotlib visualizations

**Voice Interaction Features:**
- Speech-to-text: Google API (online) with Vosk (offline backup)
- Text-to-speech: AI response playback in Chinese
- Voice command processing for medical queries
- Real-time voice status indicators
- Continuous voice conversation mode

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
- **Core**: ultralytics, PyQt5, opencv-python, numpy, matplotlib, torch, torchvision
- **Voice**: SpeechRecognition, pyttsx3, vosk, pyaudio, google-api-python-client
- **API**: requests (for DeepSeek API), json
- **Data Processing**: pandas, seaborn, scipy
- **Development**: pytest, coverage tools, yapf, ruff (as defined in ultralytics-main/pyproject.toml)

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
- Primary GUI: `ultralytics-main/visualization2.0.py`
- Test GUIs: `visualization_test1.py`, `visualization_test2.py`
- Training script: `ultralytics-main/eyes_train.py`
- Jupyter notebook: `ultralytics-main/eyes_train.ipynb`

**Voice System:**
- Enhanced voice recognition: `test/enhanced_voice_recognition.py`
- Voice integration layer: `test/voice_integration.py`
- Voice setup utilities: `test/voice_setup.py`
- Local Chinese model: `vosk-model-cn-0.22/`

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
cd ultralytics-main
yapf --in-place --recursive ultralytics/

# Linting and type checking
ruff check ultralytics/
ruff format ultralytics/

# Running formatters on the entire project
yapf --in-place --recursive .
```

### Voice System Testing
```bash
# Test voice recognition functionality
python test/test_voice_features.py
python test/test_voice_status.py

# Test complete voice integration
python test/test_smart_voice_integration.py

# Voice diagnostics and debugging
python test/voice_diagnostic.py
```

## API Integration

### DeepSeek AI Integration
The system integrates with DeepSeek's chat completion API for medical advice:
- Endpoint: `https://api.deepseek.com/v1/chat/completions`
- Model: `deepseek-chat`
- Provides treatment recommendations based on detected diseases
- Falls back to built-in advice if API is unavailable
- Voice integration: AI responses can be automatically converted to speech
- API key storage: Saved to `saved_api_key.txt` for persistence

### Voice API Integration
- **Google Speech Recognition API**: Primary online voice recognition (requires internet)
- **Vosk API**: Local offline Chinese speech recognition using `vosk-model-cn-0.22`
- **pyttsx3**: Cross-platform text-to-speech synthesis
- **Fallback Strategy**: Automatically switches from Google API to local Vosk when offline

### History and Data Management
- **History Storage**: `~/EyeDiseaseDetectorHistory/history.json`
- **Format**: JSON records with timestamps, disease classifications, and confidence scores
- **Features**: Exportable reports, batch processing summaries, visualization charts

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Ensure the model path in `ultralytics-main/visualization2.0.py` points to `ultralytics-main/self_model/AKConv_best_moudle/best.pt`
2. **Dataset Path Issues**: Update paths in `eyes_train.py` and `ultralytics-main/datasets/eyes_split/eyes_disease.yaml` to match your system
3. **GUI Display Problems**: Verify PyQt5 installation and system compatibility
4. **API Connection Failures**: Check DeepSeek API key validity and network connectivity
5. **Voice Recognition Issues**:
   - Install voice dependencies: `python test/install_voice_deps.py`
   - Check microphone permissions and audio device settings
   - Verify Vosk model exists at `vosk-model-cn-0.22/`
   - Test individual voice components with `python test/voice_diagnostic.py`
6. **Path Resolution**: The project has multiple visualization files - use `ultralytics-main/visualization2.0.py` as the primary version

### Performance Optimization
- Enable GPU acceleration by setting `device=0` in training/inference
- Use mixed precision training (`amp=True`) for better performance
- Adjust batch size based on available GPU memory
- Consider model pruning/quantization for deployment scenarios
- Voice processing: Use Google API for better accuracy when internet is available
- For offline deployment, pre-load Vosk model to improve first-time recognition speed

## Development Workflow

### Working with Multiple GUI Versions
- **ultralytics-main/visualization2.0.py**: Production version with all features
- **visualization_test1.py & visualization_test2.py**: Development/testing versions
- Always test changes in test versions before modifying the main application

### Custom YOLO Module Development
The project uses custom AKConv modules located in `ultralytics-main/ultralytics/nn/modules/akconv.py`. When modifying the neural network architecture:
1. Update the AKConv module implementations
2. Rebuild the model using the training script
3. Test inference with the custom model weights in `ultralytics-main/self_model/AKConv_best_moudle/`

### Voice Feature Development
1. Test individual components in the `test/` directory
2. Use `test/voice_diagnostic.py` for debugging voice issues
3. Install dependencies via `test/install_voice_deps.py`
4. Follow the Chinese documentation in `语音对话使用说明.md` for user features
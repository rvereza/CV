# LLaVA Real-Time Object Tracking System

A sophisticated computer vision system that uses **LLaVA (Large Language and Vision Assistant)** to detect and track military vehicles and vessels in video streams. Unlike traditional object detection models, this system leverages vision-language understanding to locate objects through intelligent text-based prompting.

## 🎯 Target Objects

The system is optimized to detect and track:
- **War Tanks** / Military Tanks
- **Airplanes** / Aircraft
- **Tug Boats**
- **Tanker Vessels**
- **Destroyers** / Warships

## ✨ Key Features

- **Vision-Language Detection**: Uses LLaVA's text-based output with intelligent prompting to estimate object locations
- **Real-Time Processing**: Optimized for video processing with configurable frame sampling
- **Metal Acceleration**: Native support for Apple Silicon (M1/M2/M3) via Metal Performance Shaders
- **Adaptive Localization**: Multiple parsing strategies to extract bounding boxes from natural language
- **Live Visualization**: Real-time display with colored bounding boxes and object labels
- **Flexible Output**: Optional video export with annotations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS (for Metal acceleration) or Linux/Windows with CUDA/CPU
- 8GB+ RAM (16GB+ recommended)
- For best performance: Apple Silicon Mac or NVIDIA GPU

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd CV
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **First Run** (downloads model automatically)
```bash
python llava_tracker.py --video path/to/your/video.mp4
```

The first run will download the LLaVA model (~13GB), which may take several minutes.

## 📖 Usage

### Basic Usage

Track objects in a video with live display:
```bash
python llava_tracker.py --video input.mp4
```

### Save Annotated Video

```bash
python llava_tracker.py --video input.mp4 --output tracked_output.mp4
```

### Adjust Processing Speed

Process every N frames (lower = more accurate but slower):
```bash
# Process every frame (slowest, most accurate)
python llava_tracker.py --video input.mp4 --interval 1

# Process every 10 frames (faster, less accurate)
python llava_tracker.py --video input.mp4 --interval 10
```

### Headless Mode

Run without display window (useful for servers):
```bash
python llava_tracker.py --video input.mp4 --output output.mp4 --no-display
```

### Custom Model

Use a specific LLaVA model:
```bash
python llava_tracker.py --video input.mp4 --model path/to/model
```

### Disable Metal Acceleration

For debugging or compatibility:
```bash
python llava_tracker.py --video input.mp4 --no-metal
```

## ⌨️ Keyboard Controls

During playback:
- **`q`** - Quit
- **`p`** - Pause/Resume
- **`s`** - Save screenshot of current frame

## 🏗️ Architecture

### System Components

1. **`llava_tracker.py`** - Main tracking system
   - Video processing pipeline
   - Frame-by-frame analysis
   - Visualization and output

2. **`model_loader.py`** - LLaVA model interface
   - Model initialization with Metal/CUDA/CPU support
   - Image preprocessing
   - Inference optimization

3. **`prompt_engine.py`** - Intelligent prompt generation
   - Detection prompts (what objects are present?)
   - Localization prompts (where are they located?)
   - Multiple prompt strategies for robustness

4. **`location_parser.py`** - Text-to-location extraction
   - Parses structured responses (LOCATION:, BBOX:)
   - Natural language position understanding
   - Confidence estimation

### How It Works

Since LLaVA outputs text rather than bounding boxes, the system uses a clever multi-step approach:

1. **Detection Phase**
   - Ask LLaVA: "What target objects do you see?"
   - Parse response to identify present objects

2. **Localization Phase**
   - For each detected object, ask: "Where is the [object] located?"
   - Use structured prompts requesting percentage-based coordinates
   - Example: "LOCATION: 45, 30, 25, 15" means center at (45%, 30%) with size (25%, 15%)

3. **Parsing Phase**
   - Extract coordinates from text using multiple strategies:
     - Explicit format parsing (LOCATION:, BBOX:)
     - Natural language understanding ("upper left corner, large object")
     - Numeric coordinate extraction

4. **Visualization Phase**
   - Convert percentages to pixel coordinates
   - Draw bounding boxes with object-specific colors
   - Display performance metrics

## 🎨 Color Coding

- **Red** - War Tanks / Tanks
- **Blue** - Airplanes / Aircraft
- **Yellow** - Tug Boats
- **Orange** - Tanker Vessels
- **Purple** - Destroyers / Warships
- **Olive** - Military Vehicles
- **Green** - Other detected objects

## ⚡ Performance Tips

### For Apple Silicon Macs (M1/M2/M3)

Metal acceleration is enabled by default and provides excellent performance:
```bash
# Optimized for M1/M2/M3
python llava_tracker.py --video input.mp4 --interval 3
```

Expected performance:
- M1 Max/Ultra, M2 Max/Ultra, M3 Max/Ultra: ~2-5 FPS
- M1/M2/M3 Base: ~1-3 FPS

### For NVIDIA GPUs

CUDA acceleration is automatic when available:
```bash
# Will use CUDA if available
python llava_tracker.py --video input.mp4
```

### For CPU-Only Systems

Reduce processing frequency for acceptable performance:
```bash
# Process every 15 frames
python llava_tracker.py --video input.mp4 --interval 15
```

## 🧪 Testing Components

Test individual components:

```bash
# Test model loader
python model_loader.py

# Test prompt engine
python prompt_engine.py

# Test location parser
python location_parser.py
```

## 🔧 Advanced Configuration

### Modify Target Objects

Edit `llava_tracker.py` and modify the `TARGET_OBJECTS` list:

```python
TARGET_OBJECTS = [
    "war tank",
    "airplane",
    "custom_object_1",
    "custom_object_2",
]
```

### Adjust Prompt Strategy

Edit `prompt_engine.py` to customize prompts:

```python
def create_detection_prompt(self) -> str:
    # Customize your detection prompt
    prompt = "Your custom prompt here..."
    return prompt
```

### Tune Location Parsing

Modify `location_parser.py` to adjust parsing behavior:

```python
# Adjust default object sizes
SIZE_KEYWORDS = {
    "small": {"scale": 0.15},
    "large": {"scale": 0.60},  # Make "large" bigger
}
```

## 📊 Output Examples

### Console Output
```
Loading LLaVA model: liuhaotian/llava-v1.5-7b
Device: mps
Using Metal Performance Shaders (MPS) acceleration

Video Info:
  Resolution: 1920x1080
  FPS: 30
  Total Frames: 900
  Duration: 30.00s

Processing every 5 frames...
Target Objects: war tank, airplane, tug boat, tanker vessel, destroyer

Frame 0: Found 2 object(s)
  - war tank at (120, 300, 450, 580)
  - airplane at (800, 150, 1100, 400)

Frame 5: Found 2 object(s)
  - war tank at (130, 305, 460, 585)
  - airplane at (820, 145, 1120, 395)

...

Processing Summary:
  Total Frames Processed: 180
  Average FPS: 2.3
  Total Processing Time: 78.26s
  Output saved to: output.mp4
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional object types
- Better localization strategies
- Temporal tracking (linking detections across frames)
- Batch processing optimization
- Alternative vision-language models

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LLaVA**: [Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)
- **Transformers**: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **PyTorch**: [PyTorch Framework](https://pytorch.org/)

## 📞 Support

For issues and questions:
1. Check existing issues in the repository
2. Review this README thoroughly
3. Test individual components (see Testing section)
4. Open a new issue with detailed information

## 🔬 Technical Details

### Why LLaVA for Object Tracking?

Traditional object detection models (YOLO, Faster R-CNN) require:
- Large labeled datasets for each object class
- Retraining for new object types
- Fixed class vocabularies

LLaVA's advantages:
- Zero-shot detection of any describable object
- Natural language understanding of spatial relationships
- No training required for new object types
- Flexible and interpretable

### Limitations

- **Speed**: Slower than specialized detection models (2-5 FPS vs 30-60 FPS)
- **Precision**: Location estimates less precise than regression-based detectors
- **Consistency**: Text-based output requires robust parsing

### Best Use Cases

- Research and prototyping
- Custom/rare object classes without training data
- Scenarios requiring interpretability
- Educational demonstrations
- Systems where flexibility > speed

---

**Built with ❤️ for Computer Vision Research**

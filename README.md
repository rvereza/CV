# YouTube Object Detection

Real-time object detection on YouTube video streams using YOLOv8. This robust Python script allows you to input any YouTube URL and perform live object detection with high performance.

## Features

- Real-time object detection on YouTube videos
- Multiple YOLO model options (nano to extra-large)
- Adjustable confidence thresholds
- Multi-threaded video processing for optimal performance
- FPS counter and object count display
- Keyboard controls (pause, screenshot, quit)
- Resolution selection (360p to 1080p)
- Robust error handling
- Thread-safe frame buffering

## Requirements

- Python 3.8 or higher
- OpenCV
- YOLOv8 (Ultralytics)
- yt-dlp for YouTube streaming
- CUDA-capable GPU (optional, but recommended for better performance)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd CV
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first time you run the script, YOLOv8 will automatically download the model weights (approximately 6-100MB depending on the model).

## Usage

### Basic Usage

```bash
python youtube_object_detector.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Or run without arguments to be prompted for URL:

```bash
python youtube_object_detector.py
```

### Advanced Usage

```bash
# Use a more accurate model
python youtube_object_detector.py "URL" --model yolov8m.pt

# Adjust confidence threshold
python youtube_object_detector.py "URL" --confidence 0.7

# Select video resolution
python youtube_object_detector.py "URL" --resolution 480p

# Hide FPS counter
python youtube_object_detector.py "URL" --no-fps

# Combine options
python youtube_object_detector.py "URL" --model yolov8s.pt --confidence 0.6 --resolution 1080p
```

### Command-line Arguments

- `url`: YouTube video URL (optional, will prompt if not provided)
- `--model`: YOLO model to use
  - `yolov8n.pt` - Nano (fastest, default)
  - `yolov8s.pt` - Small
  - `yolov8m.pt` - Medium
  - `yolov8l.pt` - Large
  - `yolov8x.pt` - Extra Large (most accurate)
- `--confidence`: Detection confidence threshold (0.0-1.0, default: 0.5)
- `--resolution`: Video resolution (360p, 480p, 720p, 1080p, default: 720p)
- `--no-fps`: Hide FPS counter

### Keyboard Controls

While the application is running:

- **Q** or **ESC** - Quit the application
- **P** - Pause/Resume playback
- **S** - Save screenshot (saved as screenshot_001.jpg, screenshot_002.jpg, etc.)

## Performance Tips

1. **Model Selection**:
   - Use `yolov8n.pt` for fastest performance (recommended for real-time)
   - Use `yolov8s.pt` or `yolov8m.pt` for balanced performance/accuracy
   - Use `yolov8l.pt` or `yolov8x.pt` for highest accuracy (requires powerful GPU)

2. **Resolution**: Lower resolutions (480p, 360p) process faster

3. **GPU Acceleration**: Ensure PyTorch is installed with CUDA support for GPU acceleration:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Internet Speed**: Stable high-speed internet connection recommended for HD streams

## Detected Object Classes

YOLOv8 can detect 80 different object classes including:
- People, vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (cat, dog, bird, horse, etc.)
- Common objects (phone, laptop, book, cup, etc.)
- Sports equipment, furniture, and more

Full list available at: https://docs.ultralytics.com/datasets/detect/coco/

## Architecture

The script is organized into several components:

- **YouTubeVideoStream**: Handles YouTube URL extraction and stream URL retrieval
- **ObjectDetector**: Manages YOLO model and performs inference
- **FrameBuffer**: Thread-safe buffer for smooth video processing
- **YouTubeObjectDetectionApp**: Main application orchestrating all components

## Troubleshooting

### "Failed to extract stream URL"
- Check your internet connection
- Verify the YouTube URL is valid and accessible
- Some videos may be region-restricted or require authentication

### Low FPS / Performance Issues
- Try a smaller model (yolov8n.pt)
- Reduce video resolution
- Ensure GPU acceleration is working (check with `torch.cuda.is_available()`)
- Close other resource-intensive applications

### "Failed to open video stream"
- The video URL may have expired (YouTube URLs are temporary)
- Try running the script again
- Check firewall settings

### OpenCV Window Issues
- Ensure you have a display available (X11 on Linux, GUI on Windows/Mac)
- For remote servers, consider using X forwarding or saving frames instead

## Examples

### Detect objects in a nature documentary
```bash
python youtube_object_detector.py "https://www.youtube.com/watch?v=nature_video"
```

### High-accuracy traffic detection
```bash
python youtube_object_detector.py "https://www.youtube.com/watch?v=traffic_cam" \
    --model yolov8l.pt --confidence 0.6 --resolution 1080p
```

### Fast processing for live streams
```bash
python youtube_object_detector.py "https://www.youtube.com/watch?v=live_stream" \
    --model yolov8n.pt --resolution 480p
```

## License

This project uses:
- YOLOv8 by Ultralytics (AGPL-3.0)
- yt-dlp (Unlicense)
- OpenCV (Apache 2.0)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video extraction
- [OpenCV](https://opencv.org/) for computer vision capabilities

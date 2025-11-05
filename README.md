# YOLOv8 Video Object Detection

A robust and fast Python script for real-time object detection on video streams using YOLOv8-large model. Supports both YouTube videos and local video files with an intuitive display interface.

## Features

- **YOLOv8-Large Model**: High-accuracy object detection with 80+ object classes
- **YouTube Support**: Direct streaming from YouTube URLs
- **Local Video Files**: Process any local video file (MP4, AVI, MOV, etc.)
- **Real-time Display**: Live visualization with bounding boxes and labels
- **Performance Optimizations**: Frame skipping, GPU acceleration support
- **Interactive Controls**: Pause, resume, and screenshot capabilities
- **FPS Counter**: Real-time performance monitoring
- **Customizable Parameters**: Confidence threshold, IOU threshold, display scaling

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for best performance)

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU acceleration, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. The YOLOv8-large model will be automatically downloaded on first run

## Usage

### Interactive Mode (Easiest)

Simply run the script without any arguments and it will prompt you for the video source:

```bash
python video_object_detection.py
```

You'll be prompted to enter:
- A YouTube URL (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`)
- Or a local video file path (e.g., `/path/to/video.mp4`)

### Basic Usage (Command Line)

**Process a YouTube video:**
```bash
python video_object_detection.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Process a local video file:**
```bash
python video_object_detection.py /path/to/video.mp4
```

### Advanced Options

**Custom confidence threshold:**
```bash
python video_object_detection.py video.mp4 --conf 0.5
```

**Use a different YOLOv8 model:**
```bash
# Nano (fastest, least accurate)
python video_object_detection.py video.mp4 --model yolov8n.pt

# Small
python video_object_detection.py video.mp4 --model yolov8s.pt

# Medium
python video_object_detection.py video.mp4 --model yolov8m.pt

# Large (default, recommended)
python video_object_detection.py video.mp4 --model yolov8l.pt

# Extra Large (slowest, most accurate)
python video_object_detection.py video.mp4 --model yolov8x.pt
```

**Scale display window:**
```bash
# 50% of original size
python video_object_detection.py video.mp4 --scale 0.5

# 150% of original size
python video_object_detection.py video.mp4 --scale 1.5
```

**Skip frames for better performance:**
```bash
# Process every other frame
python video_object_detection.py video.mp4 --skip-frames 1

# Process every 3rd frame
python video_object_detection.py video.mp4 --skip-frames 2
```

**Combine options:**
```bash
python video_object_detection.py "https://youtube.com/watch?v=xyz" \
    --model yolov8l.pt \
    --conf 0.4 \
    --iou 0.5 \
    --scale 0.75 \
    --skip-frames 1
```

## Interactive Controls

While the detection window is open:

- **'q'** - Quit the application
- **'p'** - Pause/Resume video processing
- **'s'** - Save screenshot of current frame

Screenshots are saved as `screenshot_0001.jpg`, `screenshot_0002.jpg`, etc.

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `source` | str | Required | Video source (YouTube URL or file path) |
| `--model` | str | `yolov8l.pt` | YOLOv8 model (n/s/m/l/x) |
| `--conf` | float | `0.25` | Confidence threshold (0.0-1.0) |
| `--iou` | float | `0.45` | IOU threshold for NMS (0.0-1.0) |
| `--scale` | float | `1.0` | Display window scale factor |
| `--skip-frames` | int | `0` | Frames to skip (0 = process all) |

## Performance Tips

1. **Use GPU acceleration**: Install PyTorch with CUDA support for 10-50x speedup
2. **Adjust model size**: Use smaller models (yolov8n, yolov8s) for faster processing
3. **Skip frames**: Use `--skip-frames 1` or higher for real-time performance on slower hardware
4. **Lower resolution**: Use `--scale 0.5` to reduce display overhead
5. **Increase confidence**: Use `--conf 0.5` to reduce false positives and processing time

## Example Performance

On a system with NVIDIA RTX 3080:

| Model | Resolution | FPS | Accuracy |
|-------|-----------|-----|----------|
| YOLOv8n | 1920x1080 | ~120 FPS | Good |
| YOLOv8s | 1920x1080 | ~90 FPS | Better |
| YOLOv8m | 1920x1080 | ~60 FPS | Great |
| YOLOv8l | 1920x1080 | ~40 FPS | Excellent |
| YOLOv8x | 1920x1080 | ~25 FPS | Best |

## Detected Object Classes

YOLOv8 can detect 80 object classes from the COCO dataset including:

- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, train, boat
- **Animals**: cat, dog, horse, cow, sheep, bird, etc.
- **Objects**: chair, table, laptop, phone, bottle, cup, etc.
- **And many more...**

For a complete list, see: [COCO Dataset Classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

## Troubleshooting

**Issue**: "Error loading model"
- **Solution**: Make sure you have internet connection for first-time model download

**Issue**: Slow performance
- **Solutions**:
  - Install PyTorch with CUDA support
  - Use a smaller model (--model yolov8n.pt)
  - Skip frames (--skip-frames 1)
  - Reduce display scale (--scale 0.5)

**Issue**: "Could not open video source"
- **Solutions**:
  - Verify the YouTube URL is correct and video is available
  - Check that local video file exists and is readable
  - Try updating yt-dlp: `pip install --upgrade yt-dlp`

**Issue**: YouTube video extraction fails
- **Solution**: Update yt-dlp: `pip install --upgrade yt-dlp`

## Technical Details

### Architecture
- **Detection Model**: YOLOv8 (You Only Look Once version 8)
- **Framework**: Ultralytics
- **Video Processing**: OpenCV
- **YouTube Streaming**: yt-dlp

### Features Implementation
- Real-time object detection with bounding boxes
- Class labels with confidence scores
- FPS monitoring and performance metrics
- Frame-by-frame processing with optional frame skipping
- Interactive pause/resume functionality
- Screenshot capture capability

## License

This script uses the following open-source libraries:
- Ultralytics YOLOv8 (AGPL-3.0)
- OpenCV (Apache 2.0)
- yt-dlp (Unlicense)

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video streaming
- COCO dataset for training data

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements

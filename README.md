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

**Process a YouTube video (downloads first for reliability):**
```bash
python video_object_detection.py "https://www.youtube.com/watch?v=VIDEO_ID"
```
Note: YouTube videos are downloaded to your temp directory first for reliable processing. This prevents streaming connection errors.

**Stream YouTube video without downloading:**
```bash
python video_object_detection.py "https://www.youtube.com/watch?v=VIDEO_ID" --stream
```
Warning: Streaming may encounter "Cannot reuse HTTP connection" errors. Downloading is recommended.

**Process a local video file:**
```bash
python video_object_detection.py /path/to/video.mp4
```

### Advanced Options

**Show all detections including overlapping boxes:**
```bash
# Show ALL object detections - even overlapping ones with different classifications
python video_object_detection.py video.mp4 --show-all-detections
```

This flag disables aggressive Non-Maximum Suppression (NMS), allowing you to see ALL objects that YOLO detects, even if they overlap. This is especially useful when:
- YOLO misclassifies objects (e.g., military vehicles detected as both "boat" and "truck")
- You want to see all alternative classifications with their confidence scores
- The same object might be detected as multiple different classes

Example: A tank might show multiple overlapping boxes:
- Green box: `boat: 0.65` (high confidence)
- Yellow box: `truck: 0.42` (medium confidence)
- Orange box: `car: 0.38` (lower confidence)

Color coding:
- **Green** = High confidence (≥60%)
- **Yellow** = Medium confidence (40-60%)
- **Orange** = Lower confidence (<40%)

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

- **'q'** or **ESC** - Quit the application
- **'p'** - Pause/Resume video processing
- **'s'** - Save screenshot of current frame
- **'r'** - Manually reconnect stream (YouTube streaming mode only)

Screenshots are saved as `screenshot_0001.jpg`, `screenshot_0002.jpg`, etc.

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `source` | str | Optional | Video source (YouTube URL or file path). Prompts if not provided. |
| `--model` | str | `yolov8l.pt` | YOLOv8 model (n/s/m/l/x) |
| `--conf` | float | `0.25` | Confidence threshold (0.0-1.0) |
| `--iou` | float | `0.45` | IOU threshold for NMS (0.0-1.0) |
| `--scale` | float | `1.0` | Display window scale factor |
| `--skip-frames` | int | `0` | Frames to skip (0 = process all) |
| `--show-all-detections` | flag | False | Show ALL detections including overlapping boxes |
| `--stream` | flag | False | Stream YouTube videos instead of downloading |

## YouTube Video Processing

### Download Mode (Default - Recommended)

By default, YouTube videos are downloaded to your system's temp directory before processing. This provides:

- **Reliability**: No streaming connection errors or interruptions
- **Performance**: Faster processing without network delays
- **Caching**: Videos are cached and reused if you run the script again
- **Quality**: Consistent quality throughout processing

Downloaded videos are stored in your temp directory (e.g., `/tmp` on Mac/Linux) and can be safely deleted later.

### Streaming Mode (Optional)

Use the `--stream` flag to stream YouTube videos directly without downloading:

```bash
python video_object_detection.py "URL" --stream
```

Note: Streaming mode may encounter connection errors like "Cannot reuse HTTP connection" and is less reliable, especially for longer videos.

## Performance Tips

1. **Use GPU acceleration**: Install PyTorch with CUDA support for 10-50x speedup
2. **Adjust model size**: Use smaller models (yolov8n, yolov8s) for faster processing
3. **Skip frames**: Use `--skip-frames 1` or higher for real-time performance on slower hardware
4. **Lower resolution**: Use `--scale 0.5` to reduce display overhead
5. **Increase confidence**: Use `--conf 0.5` to reduce false positives and processing time
6. **Download YouTube videos**: Use default download mode for best reliability (automatic)

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

### Understanding Misclassifications

YOLO is trained on the COCO dataset which doesn't include many specialized objects like:
- Military vehicles (tanks, armored vehicles)
- Specialized equipment
- Rare vehicles or objects

When YOLO encounters objects not in its training data, it will classify them as the closest match it knows. For example:
- **Tank → Boat**: Tanks have similar metal hull shapes and track patterns that YOLO might interpret as boat features
- **Forklift → Truck**: Similar industrial vehicle characteristics
- **Specialized vehicles → Car/Truck**: Default to common vehicle classes

**Solution**: Use the `--show-all-detections` flag to see ALL detections including alternative classifications. This disables aggressive filtering and shows you every object YOLO detects with its confidence score.

```bash
python video_object_detection.py video.mp4 --show-all-detections
```

You'll see multiple overlapping boxes with different colors:
```
Green box:  boat: 0.65  (highest confidence)
Yellow box: truck: 0.42 (medium confidence)
Orange box: car: 0.38   (lower confidence)
```

This helps you understand that while YOLO is most confident it's a "boat" (65%), it's also detecting it as a "truck" (42%) and "car" (38%) in the same location.

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

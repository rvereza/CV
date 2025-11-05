# Quick Start Guide

Get up and running with YouTube Object Detection in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run the Script

### Option A: With URL argument
```bash
python youtube_object_detector.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Option B: Interactive (will prompt for URL)
```bash
python youtube_object_detector.py
```

## Step 3: Use the Controls

Once the window opens:
- Watch the real-time object detection!
- Press **P** to pause/resume
- Press **S** to save a screenshot
- Press **Q** or **ESC** to quit

## Example Videos to Try

Here are some good YouTube videos for testing object detection:

### Traffic/Urban Scenes
- Search for "traffic camera live" or "city walking tour"
- Good for detecting: cars, buses, people, traffic lights

### Nature/Wildlife
- Search for "wildlife documentary" or "safari live"
- Good for detecting: animals, birds

### Sports
- Search for "football match" or "basketball game"
- Good for detecting: people, sports balls

### Indoor Scenes
- Search for "cooking tutorial" or "home tour"
- Good for detecting: people, kitchen items, furniture

## Performance Modes

### Fast Mode (for lower-end hardware)
```bash
python youtube_object_detector.py "URL" --resolution 480p
```

### Balanced Mode (recommended)
```bash
python youtube_object_detector.py "URL" --model yolov8s.pt --resolution 720p
```

### Accuracy Mode (requires good GPU)
```bash
python youtube_object_detector.py "URL" --model yolov8m.pt --resolution 1080p
```

## Troubleshooting Quick Fixes

### Script runs but no window appears
- Make sure you have a display/GUI available
- On remote servers, use X forwarding or VNC

### Low FPS (< 10)
- Use `--resolution 480p` or `--resolution 360p`
- Stick with default model (yolov8n.pt)
- Check if GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`

### "Failed to extract stream URL"
- Check your internet connection
- Make sure the YouTube URL is valid and public
- Try a different video

### Dependencies won't install
- Make sure you're using Python 3.8 or higher: `python --version`
- Try upgrading pip: `pip install --upgrade pip`
- Install system dependencies (on Ubuntu/Debian):
  ```bash
  sudo apt-get update
  sudo apt-get install python3-opencv
  ```

## Next Steps

- Check out README.md for full documentation
- Experiment with different confidence thresholds: `--confidence 0.3` to `--confidence 0.8`
- Try different YOLO models for speed/accuracy tradeoffs
- Save screenshots of interesting detections with the **S** key

Enjoy detecting objects!

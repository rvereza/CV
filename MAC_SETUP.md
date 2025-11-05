# Running YOLOv8 Object Detection on Mac with Visual Studio Code

This guide will help you set up and run the video object detection script using Visual Studio Code on macOS.

## Prerequisites

### 1. Install Homebrew (if not already installed)
Open Terminal and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.8 or higher
```bash
brew install python@3.11
```

Verify installation:
```bash
python3 --version
```

### 3. Install Visual Studio Code
Download from: https://code.visualstudio.com/download

Or install via Homebrew:
```bash
brew install --cask visual-studio-code
```

## Setup Instructions

### Step 1: Open Project in VS Code

1. Open Terminal and navigate to the project directory:
```bash
cd /path/to/CV
```

2. Open VS Code from Terminal:
```bash
code .
```

Or open VS Code and use `File > Open Folder...` to select the CV directory.

### Step 2: Install Python Extension

1. In VS Code, click the Extensions icon in the left sidebar (or press `Cmd+Shift+X`)
2. Search for "Python"
3. Install the **Python** extension by Microsoft
4. Also install **Pylance** (usually installed automatically with Python extension)

### Step 3: Create Virtual Environment

1. Open Terminal in VS Code (`Terminal > New Terminal` or press `` Ctrl+` ``)

2. Create a virtual environment:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 4: Install Dependencies

With the virtual environment activated, install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For Apple Silicon Macs (M1/M2/M3):**

If you have an Apple Silicon Mac, you can use MPS (Metal Performance Shaders) for GPU acceleration:

```bash
# PyTorch should automatically detect MPS
pip install torch torchvision
```

**Note**: OpenCV on Mac may need additional setup for display windows. If you encounter issues, install:
```bash
brew install opencv
```

### Step 5: Select Python Interpreter in VS Code

1. Press `Cmd+Shift+P` to open Command Palette
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your virtual environment: `./venv/bin/python`

### Step 6: Run the Script

#### Method 1: Using VS Code's Run Button (Easiest)

1. Open `video_object_detection.py` in VS Code
2. Click the "Run and Debug" icon in the left sidebar (or press `Cmd+Shift+D`)
3. Select one of the pre-configured options from the dropdown:
   - **Run with Video File** - Edit `.vscode/launch.json` to specify your video path
   - **Run with YouTube URL** - Edit `.vscode/launch.json` to specify the URL
   - **Run with Custom Settings** - For advanced options
   - **Run with Smaller Model (Faster)** - Uses YOLOv8-nano for speed

4. Edit `.vscode/launch.json` to customize the paths/URLs:
   - Replace `/path/to/your/video.mp4` with your actual video file path
   - Replace `https://www.youtube.com/watch?v=VIDEO_ID` with actual YouTube URL

5. Click the green "Start Debugging" button (or press `F5`)

#### Method 2: Using Integrated Terminal

1. Open Terminal in VS Code (`` Ctrl+` ``)
2. Make sure virtual environment is activated (you should see `(venv)`)
3. Run the script:

**For a local video file:**
```bash
python video_object_detection.py /path/to/your/video.mp4
```

**For a YouTube video:**
```bash
python video_object_detection.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

**With custom options:**
```bash
python video_object_detection.py video.mp4 --model yolov8l.pt --conf 0.4 --scale 0.75
```

#### Method 3: Right-Click Run

1. Open `video_object_detection.py`
2. Right-click anywhere in the editor
3. Select "Run Python File in Terminal"
4. Note: This won't pass arguments, so you'll need to modify the script or use Terminal

## Common Issues on Mac

### Issue 1: OpenCV Window Not Displaying

If the OpenCV window doesn't appear:

1. Make sure you're not running in SSH or remote session
2. Install OpenCV via Homebrew:
```bash
brew install opencv
```

3. Reinstall opencv-python:
```bash
pip uninstall opencv-python
pip install opencv-python
```

### Issue 2: "Python Not Found"

Make sure your virtual environment is activated:
```bash
source venv/bin/activate
```

### Issue 3: Permission Denied for Video/Camera

macOS may require camera permissions:
1. Go to `System Preferences > Security & Privacy > Camera`
2. Grant permission to Terminal and/or VS Code

### Issue 4: Slow Performance

On Mac without NVIDIA GPU:
- Use smaller models: `--model yolov8n.pt` or `yolov8s.pt`
- Skip frames: `--skip-frames 1` or higher
- Reduce display size: `--scale 0.5`

**Apple Silicon Macs (M1/M2/M3):**
- PyTorch will automatically use MPS backend for GPU acceleration
- Check if MPS is available by running:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Issue 5: YouTube Download Fails

Update yt-dlp:
```bash
pip install --upgrade yt-dlp
```

## Quick Test

Test the installation with a sample video:

1. Download a sample video:
```bash
curl -L "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4" -o test_video.mp4
```

2. Run the script:
```bash
python video_object_detection.py test_video.mp4 --model yolov8n.pt
```

## Performance Tips for Mac

### For Intel Macs:
- CPU-only processing will be slower
- Use `yolov8n.pt` (nano) or `yolov8s.pt` (small) models
- Use `--skip-frames 1` to process every other frame
- Expected performance: 5-15 FPS with yolov8n

### For Apple Silicon Macs (M1/M2/M3):
- Metal Performance Shaders (MPS) provides GPU acceleration
- Can handle `yolov8m.pt` or `yolov8l.pt` reasonably well
- Expected performance: 15-30 FPS with yolov8l

### For All Macs:
- Use `--scale 0.5` to reduce display window size
- Close other applications to free up resources
- Use `.mov` or `.mp4` files for best compatibility

## Keyboard Shortcuts in VS Code

- `Cmd+Shift+P` - Command Palette
- `F5` - Start Debugging
- `Cmd+Shift+D` - Open Debug Panel
- `` Ctrl+` `` - Toggle Terminal
- `Cmd+B` - Toggle Sidebar
- `Shift+Cmd+F` - Search across files

## Example Commands

### Basic usage:
```bash
python video_object_detection.py my_video.mp4
```

### YouTube video:
```bash
python video_object_detection.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Fast processing (nano model):
```bash
python video_object_detection.py video.mp4 --model yolov8n.pt
```

### Optimized for Mac:
```bash
python video_object_detection.py video.mp4 --model yolov8s.pt --skip-frames 1 --scale 0.75
```

### High accuracy (slower):
```bash
python video_object_detection.py video.mp4 --model yolov8l.pt --conf 0.5
```

## Getting Help

View all available options:
```bash
python video_object_detection.py --help
```

## Next Steps

1. Try with a sample video first
2. Experiment with different models (n, s, m, l, x)
3. Adjust `--conf` threshold to filter detections
4. Use `--skip-frames` if performance is slow
5. Press 's' during playback to save screenshots

## Troubleshooting Checklist

- [ ] Python 3.8+ installed: `python3 --version`
- [ ] Virtual environment activated: See `(venv)` in terminal
- [ ] Dependencies installed: `pip list | grep ultralytics`
- [ ] VS Code Python extension installed
- [ ] Correct interpreter selected in VS Code
- [ ] Video file exists and path is correct
- [ ] OpenCV can create windows (not in SSH session)

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## Support

If you encounter issues:
1. Check the main README.md for general troubleshooting
2. Verify your Python environment is set up correctly
3. Try with a smaller model first (yolov8n.pt)
4. Check VS Code's Output panel for error messages

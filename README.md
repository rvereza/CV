# YouTube Video Downloader

A simple Python script to download YouTube videos and convert them to MP4 format.

## Features

- Download YouTube videos from URL
- Automatic conversion to MP4 format
- Progress tracking during download
- Custom output directory support
- Command-line and interactive modes

## Requirements

- Python 3.6 or higher
- FFmpeg (for video conversion)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd CV
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (if not already installed):

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Usage

### Method 1: Command-line argument
```bash
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Method 2: Command-line with custom output directory
```bash
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" "my_videos"
```

### Method 3: Interactive mode
```bash
python youtube_downloader.py
```
Then enter the URL and output directory when prompted.

## Examples

Download a video to the default `downloads` folder:
```bash
python youtube_downloader.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Download a video to a custom folder:
```bash
python youtube_downloader.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "my_videos"
```

## Output

Videos are saved in MP4 format with the original video title as the filename.
Default output directory: `downloads/`

## Troubleshooting

**Error: yt-dlp is not installed**
- Run: `pip install yt-dlp`

**Error: FFmpeg not found**
- Install FFmpeg using the instructions above

**Download fails**
- Ensure you have a stable internet connection
- Check that the YouTube URL is valid
- Some videos may be restricted or require authentication

## License

MIT License

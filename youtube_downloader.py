#!/usr/bin/env python3
"""
YouTube Video Downloader
Downloads YouTube videos and converts them to MP4 format.
"""

import sys
import os
from pathlib import Path
try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed.")
    print("Please install it using: pip install yt-dlp")
    sys.exit(1)


def download_youtube_video(url, output_path="downloads"):
    """
    Download a YouTube video and convert it to MP4.

    Args:
        url (str): YouTube video URL
        output_path (str): Directory where the video will be saved
    """
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [progress_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\nFetching video information...")
            info = ydl.extract_info(url, download=False)
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 0) // 60}:{info.get('duration', 0) % 60:02d}")
            print(f"\nDownloading...")

            ydl.download([url])

            print(f"\n✓ Download complete!")
            print(f"Video saved to: {output_path}/")

    except Exception as e:
        print(f"\n✗ Error downloading video: {str(e)}")
        sys.exit(1)


def progress_hook(d):
    """Hook to display download progress."""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A')
        speed = d.get('_speed_str', 'N/A')
        eta = d.get('_eta_str', 'N/A')
        print(f"\rProgress: {percent} | Speed: {speed} | ETA: {eta}", end='', flush=True)
    elif d['status'] == 'finished':
        print(f"\n✓ Download finished, now converting to MP4...")


def main():
    """Main function to handle command-line input."""
    print("=" * 60)
    print("YouTube Video Downloader (MP4)")
    print("=" * 60)

    # Check if URL is provided as command-line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "downloads"
    else:
        # Prompt user for URL
        url = input("\nEnter YouTube URL: ").strip()
        if not url:
            print("Error: No URL provided.")
            sys.exit(1)

        output_path = input("Enter output directory (default: downloads): ").strip()
        output_path = output_path if output_path else "downloads"

    # Validate URL
    if not url.startswith(('http://', 'https://')):
        print("Error: Invalid URL. Please provide a valid YouTube URL.")
        sys.exit(1)

    # Download the video
    download_youtube_video(url, output_path)


if __name__ == "__main__":
    main()

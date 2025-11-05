#!/usr/bin/env python3
"""
Robust YouTube Object Detection Script
Performs real-time object detection on YouTube video streams using YOLO
"""

import cv2
import sys
import argparse
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Tuple
import yt_dlp
from ultralytics import YOLO
import numpy as np


class YouTubeVideoStream:
    """Handle YouTube video streaming with yt-dlp"""

    def __init__(self, url: str, resolution: str = "720p"):
        """
        Initialize YouTube video stream

        Args:
            url: YouTube video URL
            resolution: Preferred resolution (default: 720p)
        """
        self.url = url
        self.resolution = resolution
        self.stream_url: Optional[str] = None
        self.video_info: Optional[dict] = None

    def get_stream_url(self) -> str:
        """
        Extract direct video stream URL from YouTube

        Returns:
            Direct stream URL

        Raises:
            Exception: If unable to extract stream URL
        """
        ydl_opts = {
            'format': f'best[height<={self.resolution[:-1]}]/best',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"Extracting video information from: {self.url}")
                info = ydl.extract_info(self.url, download=False)

                self.video_info = info
                self.stream_url = info['url']

                print(f"Title: {info.get('title', 'Unknown')}")
                print(f"Duration: {info.get('duration', 0)} seconds")
                print(f"Resolution: {info.get('width', '?')}x{info.get('height', '?')}")

                return self.stream_url

        except Exception as e:
            raise Exception(f"Failed to extract stream URL: {e}")


class ObjectDetector:
    """YOLO-based object detection with optimizations"""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize object detector

        Args:
            model_name: YOLO model to use (n=nano, s=small, m=medium, l=large, x=xlarge)
            confidence: Confidence threshold for detections
        """
        self.confidence = confidence
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Perform object detection on a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (annotated_frame, detections_list)
        """
        results = self.model(frame, conf=self.confidence, verbose=False)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Extract detection information
        detections = []
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            class_name = self.model.names[int(class_id)]
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': confidence,
                'class': class_name
            })

        return annotated_frame, detections


class FrameBuffer:
    """Thread-safe frame buffer for video streaming"""

    def __init__(self, maxsize: int = 30):
        """Initialize frame buffer with maximum size"""
        self.queue = queue.Queue(maxsize=maxsize)
        self.stopped = False

    def put(self, frame: np.ndarray):
        """Add frame to buffer"""
        if not self.stopped:
            if self.queue.full():
                # Remove oldest frame if buffer is full
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def get(self) -> Optional[np.ndarray]:
        """Get frame from buffer"""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the buffer"""
        self.stopped = True


class YouTubeObjectDetectionApp:
    """Main application for YouTube object detection"""

    def __init__(self,
                 youtube_url: str,
                 model_name: str = "yolov8n.pt",
                 confidence: float = 0.5,
                 resolution: str = "720p",
                 show_fps: bool = True):
        """
        Initialize the application

        Args:
            youtube_url: YouTube video URL
            model_name: YOLO model to use
            confidence: Detection confidence threshold
            resolution: Video resolution preference
            show_fps: Whether to display FPS counter
        """
        self.youtube_url = youtube_url
        self.model_name = model_name
        self.confidence = confidence
        self.resolution = resolution
        self.show_fps = show_fps

        self.detector: Optional[ObjectDetector] = None
        self.stream: Optional[YouTubeVideoStream] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_buffer: Optional[FrameBuffer] = None

        self.running = False
        self.paused = False

    def initialize(self):
        """Initialize all components"""
        print("=" * 60)
        print("YouTube Object Detection System")
        print("=" * 60)

        # Initialize detector
        self.detector = ObjectDetector(self.model_name, self.confidence)

        # Initialize YouTube stream
        self.stream = YouTubeVideoStream(self.youtube_url, self.resolution)
        stream_url = self.stream.get_stream_url()

        # Open video capture
        print("\nOpening video stream...")
        self.cap = cv2.VideoCapture(stream_url)

        if not self.cap.isOpened():
            raise Exception("Failed to open video stream")

        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Stream opened successfully!")
        print(f"FPS: {fps:.2f}, Resolution: {width}x{height}")

        # Initialize frame buffer
        self.frame_buffer = FrameBuffer(maxsize=30)

        print("\n" + "=" * 60)
        print("Controls:")
        print("  Q or ESC - Quit")
        print("  P - Pause/Resume")
        print("  S - Save screenshot")
        print("=" * 60 + "\n")

    def read_frames_thread(self):
        """Thread function to read frames from video stream"""
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_buffer.put(frame)
                else:
                    print("End of stream or read error")
                    self.running = False
                    break
            else:
                time.sleep(0.1)

    def run(self):
        """Main application loop"""
        try:
            self.initialize()

            self.running = True

            # Start frame reading thread
            read_thread = threading.Thread(target=self.read_frames_thread, daemon=True)
            read_thread.start()

            # FPS calculation
            fps_counter = 0
            fps_start_time = time.time()
            current_fps = 0

            window_name = "YouTube Object Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            screenshot_count = 0

            while self.running:
                if self.paused:
                    cv2.waitKey(100)
                    continue

                # Get frame from buffer
                frame = self.frame_buffer.get()

                if frame is None:
                    continue

                # Perform object detection
                annotated_frame, detections = self.detector.detect(frame)

                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                # Add FPS and detection count to frame
                if self.show_fps:
                    info_text = f"FPS: {current_fps:.1f} | Objects: {len(detections)}"
                    cv2.putText(annotated_frame, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display frame
                cv2.imshow(window_name, annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    print("Quit requested by user")
                    self.running = False
                    break
                elif key == ord('p'):  # Pause
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"Playback {status}")
                elif key == ord('s'):  # Screenshot
                    screenshot_count += 1
                    filename = f"screenshot_{screenshot_count:03d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Screenshot saved: {filename}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.running = False

        if self.frame_buffer:
            self.frame_buffer.stop()

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-time object detection on YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s "https://youtu.be/dQw4w9WgXcQ" --model yolov8s.pt --confidence 0.6
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --resolution 480p
        """
    )

    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube video URL"
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLO model to use (n=fastest, x=most accurate, default: yolov8n.pt)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (0.0-1.0, default: 0.5)"
    )

    parser.add_argument(
        "--resolution",
        default="720p",
        choices=["360p", "480p", "720p", "1080p"],
        help="Preferred video resolution (default: 720p)"
    )

    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Hide FPS counter"
    )

    args = parser.parse_args()

    # Get URL from argument or prompt user
    url = args.url
    if not url:
        url = input("Enter YouTube URL: ").strip()
        if not url:
            print("Error: No URL provided")
            sys.exit(1)

    # Validate URL
    if not ("youtube.com" in url or "youtu.be" in url):
        print("Error: Invalid YouTube URL")
        sys.exit(1)

    # Create and run application
    app = YouTubeObjectDetectionApp(
        youtube_url=url,
        model_name=args.model,
        confidence=args.confidence,
        resolution=args.resolution,
        show_fps=not args.no_fps
    )

    app.run()


if __name__ == "__main__":
    main()

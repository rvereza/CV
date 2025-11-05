#!/usr/bin/env python3
"""
YOLOv8 Video Object Detection Script
Performs real-time object detection on YouTube videos or local video files
using YOLOv8-large model for high accuracy.
"""

import argparse
import cv2
import sys
import time
from pathlib import Path
import yt_dlp
from ultralytics import YOLO
import numpy as np


class VideoObjectDetector:
    """Fast and robust object detection for video streams."""

    def __init__(self, model_name='yolov8l.pt', conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the detector with YOLOv8 model.

        Args:
            model_name: YOLOv8 model to use (default: yolov8l.pt for large model)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        print(f"Loading {model_name} model...")
        try:
            self.model = YOLO(model_name)
            print(f"✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def get_youtube_stream_url(self, youtube_url):
        """
        Extract the best video stream URL from YouTube.

        Args:
            youtube_url: YouTube video URL

        Returns:
            Direct stream URL
        """
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }

        try:
            print(f"Extracting YouTube video stream...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                video_url = info['url']
                print(f"✓ Stream extracted successfully!")
                return video_url
        except Exception as e:
            print(f"✗ Error extracting YouTube stream: {e}")
            sys.exit(1)

    def open_video_source(self, source):
        """
        Open video source (YouTube URL or local file).

        Args:
            source: YouTube URL or path to local video file

        Returns:
            OpenCV VideoCapture object
        """
        # Check if source is a YouTube URL
        if 'youtube.com' in source or 'youtu.be' in source:
            stream_url = self.get_youtube_stream_url(source)
            cap = cv2.VideoCapture(stream_url)
        else:
            # Assume it's a local file
            if not Path(source).exists():
                print(f"✗ Error: Video file '{source}' not found!")
                sys.exit(1)
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"✗ Error: Could not open video source!")
            sys.exit(1)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(f"Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        if total_frames > 0:
            print(f"  Total Frames: {total_frames}")
            print(f"  Duration: {total_frames/fps:.2f}s")
        print(f"{'='*60}\n")

        return cap

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            results: YOLO detection results

        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prepare label
                label = f"{class_name}: {conf:.2f}"

                # Calculate label size and position
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        return annotated_frame

    def draw_info_panel(self, frame, fps, detection_count, frame_num):
        """
        Draw information panel on frame.

        Args:
            frame: Input frame
            fps: Current FPS
            detection_count: Number of detected objects
            frame_num: Current frame number
        """
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw info text
        info_text = [
            f"FPS: {fps:.1f}",
            f"Objects Detected: {detection_count}",
            f"Frame: {frame_num}",
            f"Model: YOLOv8-Large"
        ]

        y_offset = 35
        for text in info_text:
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 25

    def process_video(self, source, display_scale=1.0, skip_frames=0):
        """
        Process video with object detection.

        Args:
            source: Video source (YouTube URL or file path)
            display_scale: Scale factor for display window (default: 1.0)
            skip_frames: Number of frames to skip between detections (0 = process all)
        """
        cap = self.open_video_source(source)

        # Create window
        window_name = "YOLOv8 Object Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Starting object detection...")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save screenshot\n")

        frame_num = 0
        fps = 0
        fps_counter = []
        paused = False
        screenshot_count = 0

        try:
            while True:
                if not paused:
                    start_time = time.time()

                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        print("\n✓ Video ended or stream interrupted")
                        break

                    frame_num += 1

                    # Skip frames if needed (for performance)
                    if skip_frames > 0 and frame_num % (skip_frames + 1) != 0:
                        continue

                    # Run detection
                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )

                    # Draw detections
                    annotated_frame = self.draw_detections(frame, results)

                    # Count detections
                    detection_count = sum(len(result.boxes) for result in results)

                    # Calculate FPS
                    end_time = time.time()
                    fps_counter.append(1 / (end_time - start_time))
                    if len(fps_counter) > 30:
                        fps_counter.pop(0)
                    fps = sum(fps_counter) / len(fps_counter)

                    # Draw info panel
                    self.draw_info_panel(annotated_frame, fps, detection_count, frame_num)

                    # Resize for display if needed
                    if display_scale != 1.0:
                        height, width = annotated_frame.shape[:2]
                        new_width = int(width * display_scale)
                        new_height = int(height * display_scale)
                        annotated_frame = cv2.resize(
                            annotated_frame,
                            (new_width, new_height),
                            interpolation=cv2.INTER_LINEAR
                        )

                    # Display frame
                    cv2.imshow(window_name, annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n✓ Quit requested by user")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n{status}")
                elif key == ord('s'):
                    screenshot_count += 1
                    screenshot_path = f"screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"✓ Screenshot saved: {screenshot_path}")

        except KeyboardInterrupt:
            print("\n\n✓ Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {frame_num}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Video Object Detection - Fast and robust object detection for video streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process YouTube video
  python video_object_detection.py "https://www.youtube.com/watch?v=VIDEO_ID"

  # Process local video file
  python video_object_detection.py /path/to/video.mp4

  # Process with custom confidence threshold
  python video_object_detection.py video.mp4 --conf 0.5

  # Process with smaller display window
  python video_object_detection.py video.mp4 --scale 0.5

  # Process every other frame for better performance
  python video_object_detection.py video.mp4 --skip-frames 1
        """
    )

    parser.add_argument(
        'source',
        type=str,
        help='Video source (YouTube URL or path to local video file)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolov8l.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLOv8 model to use (default: yolov8l.pt)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS (default: 0.45)'
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Display window scale factor (default: 1.0)'
    )

    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='Number of frames to skip between detections (0 = process all frames)'
    )

    args = parser.parse_args()

    # Create detector
    detector = VideoObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Process video
    detector.process_video(
        source=args.source,
        display_scale=args.scale,
        skip_frames=args.skip_frames
    )


if __name__ == '__main__':
    main()

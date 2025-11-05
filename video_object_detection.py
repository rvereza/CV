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
import tempfile
import os


class VideoObjectDetector:
    """Fast and robust object detection for video streams."""

    def __init__(self, model_name='yolov8l.pt', conf_threshold=0.25, iou_threshold=0.45, top_predictions=1):
        """
        Initialize the detector with YOLOv8 model.

        Args:
            model_name: YOLOv8 model to use (default: yolov8l.pt for large model)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            top_predictions: Number of top class predictions to display per object (default: 1)
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
        self.top_predictions = top_predictions
        self.youtube_url = None
        self.stream_info = None
        self.last_stream_refresh = 0
        self.stream_refresh_interval = 300  # Refresh stream URL every 5 minutes

    def get_youtube_stream_url(self, youtube_url, retry_count=3):
        """
        Extract the best video stream URL from YouTube with retry logic.

        Args:
            youtube_url: YouTube video URL
            retry_count: Number of retry attempts

        Returns:
            Tuple of (stream_url, full_info_dict)
        """
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 10,
        }

        for attempt in range(retry_count):
            try:
                if attempt > 0:
                    print(f"  Retry attempt {attempt + 1}/{retry_count}...")
                    time.sleep(2 ** attempt)  # Exponential backoff

                print(f"Extracting YouTube video stream...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    video_url = info['url']
                    print(f"✓ Stream extracted successfully!")
                    return video_url, info

            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"⚠ Attempt {attempt + 1} failed: {e}")
                else:
                    print(f"✗ Error extracting YouTube stream after {retry_count} attempts: {e}")
                    print(f"\n💡 Suggestions:")
                    print(f"  1. Check your internet connection")
                    print(f"  2. Verify the YouTube URL is correct and video is available")
                    print(f"  3. Update yt-dlp: pip install --upgrade yt-dlp")
                    print(f"  4. Try using a local video file instead")
                    sys.exit(1)

        return None, None

    def download_youtube_video(self, youtube_url, output_dir=None):
        """
        Download YouTube video to local file for reliable processing.

        Args:
            youtube_url: YouTube video URL
            output_dir: Directory to save video (default: temp directory)

        Returns:
            Path to downloaded video file
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        # Create a unique filename
        video_id = youtube_url.split('=')[-1].split('&')[0]
        output_path = os.path.join(output_dir, f"youtube_{video_id}.mp4")

        # Check if already downloaded
        if os.path.exists(output_path):
            print(f"✓ Using cached video: {output_path}")
            return output_path

        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
        }

        try:
            print(f"\n{'='*60}")
            print(f"Downloading YouTube video for reliable processing...")
            print(f"This may take a moment depending on video length and internet speed.")
            print(f"{'='*60}\n")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            if os.path.exists(output_path):
                print(f"\n✓ Video downloaded successfully: {output_path}")
                return output_path
            else:
                print(f"✗ Download failed - file not created")
                return None

        except Exception as e:
            print(f"✗ Error downloading video: {e}")
            return None

    def open_video_source(self, source, download_youtube=True):
        """
        Open video source (YouTube URL or local file).

        Args:
            source: YouTube URL or path to local video file
            download_youtube: If True, download YouTube videos instead of streaming (more reliable)

        Returns:
            OpenCV VideoCapture object
        """
        # Check if source is a YouTube URL
        if 'youtube.com' in source or 'youtu.be' in source:
            self.youtube_url = source

            if download_youtube:
                # Download video for reliable processing (recommended)
                downloaded_path = self.download_youtube_video(source)
                if downloaded_path:
                    cap = cv2.VideoCapture(downloaded_path)
                else:
                    print("⚠ Download failed, falling back to streaming...")
                    stream_url, self.stream_info = self.get_youtube_stream_url(source)
                    self.last_stream_refresh = time.time()
                    cap = cv2.VideoCapture(stream_url)
            else:
                # Stream directly (less reliable but no download wait)
                stream_url, self.stream_info = self.get_youtube_stream_url(source)
                self.last_stream_refresh = time.time()
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
        else:
            print(f"  Duration: Live stream/Unknown")
        print(f"{'='*60}\n")

        return cap

    def reconnect_youtube_stream(self, cap):
        """
        Reconnect to YouTube stream when connection is lost.

        Args:
            cap: Current VideoCapture object to release

        Returns:
            New VideoCapture object or None if reconnection failed
        """
        print("\n⚠ Stream connection lost. Attempting to reconnect...")

        if cap is not None:
            cap.release()

        try:
            # Get fresh stream URL
            stream_url, self.stream_info = self.get_youtube_stream_url(self.youtube_url)
            self.last_stream_refresh = time.time()

            # Open new connection
            new_cap = cv2.VideoCapture(stream_url)

            if new_cap.isOpened():
                print("✓ Successfully reconnected to stream!")
                return new_cap
            else:
                print("✗ Failed to open reconnected stream")
                return None

        except Exception as e:
            print(f"✗ Reconnection failed: {e}")
            return None

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IOU) between two boxes.

        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]

        Returns:
            IOU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_multiple_class_predictions(self, frame):
        """
        Get multiple class predictions for same object by running inference
        with lower confidence threshold.

        Args:
            frame: Input frame

        Returns:
            List of grouped detections with multiple class predictions
        """
        # Run prediction with very low confidence to get all possible detections
        low_conf_results = self.model.predict(
            frame,
            conf=0.10,  # Very low threshold to catch alternative predictions
            iou=0.3,  # Lower IOU to keep more overlapping boxes
            verbose=False
        )

        grouped_detections = []

        if len(low_conf_results) == 0 or len(low_conf_results[0].boxes) == 0:
            return grouped_detections

        boxes_data = []
        for box in low_conf_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.model.names[cls]
            boxes_data.append({
                'box': [x1, y1, x2, y2],
                'class': class_name,
                'conf': conf,
                'used': False
            })

        # Group overlapping boxes (likely same object with different class predictions)
        for i, box_i in enumerate(boxes_data):
            if box_i['used']:
                continue

            group = [box_i]
            box_i['used'] = True

            for j, box_j in enumerate(boxes_data):
                if i != j and not box_j['used']:
                    iou = self.calculate_iou(box_i['box'], box_j['box'])
                    if iou > 0.5:  # Overlapping significantly
                        group.append(box_j)
                        box_j['used'] = True

            # Sort group by confidence and take top N
            group_sorted = sorted(group, key=lambda x: x['conf'], reverse=True)
            top_n = group_sorted[:self.top_predictions]

            # Create grouped detection with top N predictions
            main_box = group_sorted[0]['box']
            predictions = [(item['class'], item['conf']) for item in top_n]

            grouped_detections.append({
                'box': main_box,
                'predictions': predictions
            })

        return grouped_detections

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            results: YOLO detection results (can be None if using multi-class mode)

        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()

        # If showing multiple predictions per object, use different approach
        if self.top_predictions > 1:
            detections = self.get_multiple_class_predictions(frame)

            for detection in detections:
                x1, y1, x2, y2 = detection['box']
                predictions = detection['predictions']

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw multiple predictions as stacked labels
                font_scale = 0.45
                y_offset = y1

                for rank, (class_name, conf) in enumerate(predictions, 1):
                    label = f"[{rank}] {class_name}: {conf:.2f}"

                    # Calculate label size
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )

                    # Adjust y position for each label
                    label_y = y_offset - (len(predictions) - rank + 1) * (label_height + baseline + 3)

                    # Draw label background
                    alpha = 1.0 if rank == 1 else 0.8  # First prediction more prominent
                    bg_color = (0, 255, 0) if rank == 1 else (0, 200, 200)

                    cv2.rectangle(
                        annotated_frame,
                        (x1, label_y - 2),
                        (x1 + label_width + 4, label_y + label_height + baseline),
                        bg_color,
                        -1
                    )

                    # Draw label text
                    text_color = (0, 0, 0) if rank == 1 else (50, 50, 50)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 2, label_y + label_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        1
                    )

        else:
            # Original single-prediction mode
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

    def process_video(self, source, display_scale=1.0, skip_frames=0, download_youtube=True):
        """
        Process video with object detection.

        Args:
            source: Video source (YouTube URL or file path)
            display_scale: Scale factor for display window (default: 1.0)
            skip_frames: Number of frames to skip between detections (0 = process all)
            download_youtube: If True, download YouTube videos instead of streaming (default: True)
        """
        cap = self.open_video_source(source, download_youtube=download_youtube)
        is_youtube = self.youtube_url is not None

        # Create window
        window_name = "YOLOv8 Object Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Starting object detection...")
        print(f"\n{'='*60}")
        print("Controls:")
        print("  Q or ESC - Quit")
        print("  P - Pause/Resume")
        print("  S - Save screenshot")
        if is_youtube:
            print("  R - Manually reconnect stream")
        print(f"{'='*60}\n")

        frame_num = 0
        fps = 0
        fps_counter = []
        paused = False
        screenshot_count = 0
        consecutive_read_failures = 0
        max_read_failures = 30  # Try 30 times before giving up
        annotated_frame = None  # Keep last frame for screenshots

        try:
            while True:
                if not paused:
                    start_time = time.time()

                    # Read frame
                    ret, frame = cap.read()

                    # Handle read failures (especially for YouTube streams)
                    if not ret:
                        consecutive_read_failures += 1

                        if is_youtube and consecutive_read_failures < max_read_failures:
                            # Try to reconnect for YouTube streams
                            if consecutive_read_failures % 10 == 1:  # Try reconnecting every 10 failures
                                new_cap = self.reconnect_youtube_stream(cap)
                                if new_cap is not None:
                                    cap = new_cap
                                    consecutive_read_failures = 0
                                    continue
                                else:
                                    time.sleep(0.1)
                                    continue
                            else:
                                time.sleep(0.05)  # Brief wait before retry
                                continue
                        else:
                            # Local file ended or too many YouTube failures
                            if is_youtube:
                                print(f"\n✗ Stream failed after {consecutive_read_failures} attempts")
                                print("💡 Suggestions:")
                                print("  1. Check your internet connection")
                                print("  2. The video might have been removed or made private")
                                print("  3. Try a different video")
                            else:
                                print("\n✓ Video ended")
                            break

                    # Successfully read frame
                    if ret:
                        consecutive_read_failures = 0  # Reset failure counter
                        frame_num += 1

                        # Check if we need to refresh YouTube stream URL
                        if is_youtube and (time.time() - self.last_stream_refresh) > self.stream_refresh_interval:
                            print("\n⚠ Stream URL expired. Refreshing...")
                            new_cap = self.reconnect_youtube_stream(cap)
                            if new_cap is not None:
                                cap = new_cap

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
                            display_frame = cv2.resize(
                                annotated_frame,
                                (new_width, new_height),
                                interpolation=cv2.INTER_LINEAR
                            )
                        else:
                            display_frame = annotated_frame

                        # Display frame
                        cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    print("\n✓ Quit requested by user")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n{status}")
                elif key == ord('s') and annotated_frame is not None:
                    screenshot_count += 1
                    screenshot_path = f"screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"✓ Screenshot saved: {screenshot_path}")
                elif key == ord('r') and is_youtube:
                    print("\n⚠ Manual reconnection requested...")
                    new_cap = self.reconnect_youtube_stream(cap)
                    if new_cap is not None:
                        cap = new_cap
                        consecutive_read_failures = 0

        except KeyboardInterrupt:
            print("\n\n✓ Interrupted by user")

        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            print(f"\n{'='*60}")
            print(f"Processing complete!")
            print(f"Total frames processed: {frame_num}")
            if screenshot_count > 0:
                print(f"Screenshots saved: {screenshot_count}")
            print(f"{'='*60}")


def get_video_source_interactive():
    """
    Prompt user for video source interactively.

    Returns:
        Video source path or URL
    """
    print("\n" + "="*60)
    print("YOLOv8 Video Object Detection")
    print("="*60)
    print("\nPlease provide a video source:")
    print("  1. YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)")
    print("  2. Local video file path (e.g., /path/to/video.mp4)")
    print("="*60)

    while True:
        source = input("\nEnter video source: ").strip()

        if not source:
            print("⚠ Error: Please enter a valid source")
            continue

        # Remove quotes if user included them
        source = source.strip('"').strip("'")

        # Check if it's a YouTube URL
        if 'youtube.com' in source or 'youtu.be' in source:
            print(f"✓ YouTube URL detected: {source}")
            return source

        # Check if it's a local file
        if Path(source).exists():
            print(f"✓ Local video file found: {source}")
            return source
        else:
            # Ask user to confirm if file doesn't exist
            print(f"⚠ Warning: File not found at '{source}'")
            retry = input("Would you like to try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("Using provided path anyway (in case it's a special path)...")
                return source


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Video Object Detection - Fast and robust object detection for video streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with interactive prompt
  python video_object_detection.py

  # Process YouTube video (downloads first - recommended)
  python video_object_detection.py "https://www.youtube.com/watch?v=VIDEO_ID"

  # Show top 3 predictions per object (see alternative classifications)
  python video_object_detection.py video.mp4 --top-predictions 3

  # Stream YouTube video without downloading (less reliable)
  python video_object_detection.py "https://www.youtube.com/watch?v=VIDEO_ID" --stream

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
        nargs='?',
        type=str,
        help='Video source (YouTube URL or path to local video file). If not provided, will prompt interactively.'
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

    parser.add_argument(
        '--top-predictions',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='Number of top class predictions to show per object (default: 1). Useful to see alternative classifications like "tank detected as boat"'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream YouTube videos instead of downloading (less reliable, may have connection issues)'
    )

    args = parser.parse_args()

    # Get video source - either from command line or interactive prompt
    if args.source:
        source = args.source
    else:
        source = get_video_source_interactive()

    # Create detector
    detector = VideoObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        top_predictions=args.top_predictions
    )

    # Process video
    detector.process_video(
        source=source,
        display_scale=args.scale,
        skip_frames=args.skip_frames,
        download_youtube=not args.stream  # Download by default, stream if --stream flag is used
    )


if __name__ == '__main__':
    main()

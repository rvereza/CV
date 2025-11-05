#!/usr/bin/env python3
"""
LLaVA Real-Time Object Tracking System
Tracks military vehicles and vessels in video using LLaVA vision-language model
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import sys

from model_loader import LLaVAModel
from prompt_engine import PromptEngine
from location_parser import LocationParser


@dataclass
class DetectedObject:
    """Represents a detected object with its location and confidence"""
    label: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frame_number: int


class LLaVATracker:
    """Main object tracking system using LLaVA"""

    TARGET_OBJECTS = [
        "war tank",
        "airplane",
        "aircraft",
        "tug boat",
        "tank",
        "tanker vessel",
        "destroyer",
        "military vehicle",
        "warship"
    ]

    def __init__(self, model_path: Optional[str] = None, use_metal: bool = True):
        """
        Initialize the LLaVA tracker

        Args:
            model_path: Path to LLaVA model (if None, downloads default)
            use_metal: Use Metal acceleration on macOS
        """
        print("Initializing LLaVA Object Tracker...")
        self.model = LLaVAModel(model_path=model_path, use_metal=use_metal)
        self.prompt_engine = PromptEngine(target_objects=self.TARGET_OBJECTS)
        self.location_parser = LocationParser()

        # Tracking state
        self.detected_objects: List[DetectedObject] = []
        self.frame_count = 0
        self.total_processing_time = 0.0

        # Colors for different object types (BGR format)
        self.colors = {
            "war tank": (0, 0, 255),      # Red
            "tank": (0, 0, 255),          # Red
            "airplane": (255, 0, 0),       # Blue
            "aircraft": (255, 0, 0),       # Blue
            "tug boat": (0, 255, 255),     # Yellow
            "tanker vessel": (0, 165, 255), # Orange
            "destroyer": (128, 0, 128),    # Purple
            "warship": (128, 0, 128),      # Purple
            "military vehicle": (0, 128, 128), # Olive
            "default": (0, 255, 0)         # Green
        }

    def get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        """Get color for object label"""
        label_lower = label.lower()
        for key, color in self.colors.items():
            if key in label_lower:
                return color
        return self.colors["default"]

    def process_frame(self, frame: np.ndarray, frame_number: int,
                     process_every_n: int = 5) -> Tuple[np.ndarray, List[DetectedObject]]:
        """
        Process a single frame for object detection

        Args:
            frame: Input video frame
            frame_number: Current frame number
            process_every_n: Process every N frames (for performance)

        Returns:
            Annotated frame and list of detected objects
        """
        # Skip frames for performance
        if frame_number % process_every_n != 0:
            # Just draw previous detections
            return self.draw_detections(frame, self.detected_objects), self.detected_objects

        start_time = time.time()

        # Step 1: Initial detection - ask LLaVA what objects are present
        detection_prompt = self.prompt_engine.create_detection_prompt()
        detection_response = self.model.generate(frame, detection_prompt)

        # Step 2: Parse which target objects were detected
        detected_labels = self.location_parser.extract_detected_objects(
            detection_response,
            self.TARGET_OBJECTS
        )

        # Step 3: For each detected object, get its location
        current_detections = []

        if detected_labels:
            # Use grid-based localization for better accuracy
            for label in detected_labels:
                # Ask LLaVA to locate the object
                location_prompt = self.prompt_engine.create_localization_prompt(label)
                location_response = self.model.generate(frame, location_prompt)

                # Parse the location from response
                bbox = self.location_parser.parse_location(
                    location_response,
                    frame.shape[1],  # width
                    frame.shape[0]   # height
                )

                if bbox:
                    detection = DetectedObject(
                        label=label,
                        bbox=bbox,
                        confidence=0.8,  # LLaVA doesn't provide confidence, use default
                        frame_number=frame_number
                    )
                    current_detections.append(detection)

        # Update tracking state
        self.detected_objects = current_detections
        self.frame_count = frame_number

        # Calculate performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time

        # Annotate frame
        annotated_frame = self.draw_detections(frame, current_detections)

        # Add performance info
        fps = 1.0 / processing_time if processing_time > 0 else 0
        avg_fps = self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0

        cv2.putText(annotated_frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_number}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Detections: {len(current_detections)}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_frame, current_detections

    def draw_detections(self, frame: np.ndarray,
                       detections: List[DetectedObject]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.get_color_for_label(det.label)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label_text = f"{det.label}"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(annotated,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)

            # Draw label text
            cv2.putText(annotated, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def track_video(self, video_path: str, output_path: Optional[str] = None,
                   display: bool = True, process_every_n: int = 5):
        """
        Process video for object tracking

        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            display: Show real-time display window
            process_every_n: Process every N frames
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        print(f"\nProcessing every {process_every_n} frames...")
        print(f"Target Objects: {', '.join(self.TARGET_OBJECTS)}")
        print("\nPress 'q' to quit, 'p' to pause/resume, 's' to save screenshot\n")

        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Create display window
        if display:
            cv2.namedWindow('LLaVA Object Tracker', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('LLaVA Object Tracker', 1280, 720)

        frame_number = 0
        paused = False

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\nEnd of video reached.")
                        break

                    # Process frame
                    annotated_frame, detections = self.process_frame(
                        frame, frame_number, process_every_n
                    )

                    # Save to output video
                    if writer:
                        writer.write(annotated_frame)

                    # Display
                    if display:
                        cv2.imshow('LLaVA Object Tracker', annotated_frame)

                    # Print detections
                    if detections and frame_number % process_every_n == 0:
                        print(f"Frame {frame_number}: Found {len(detections)} object(s)")
                        for det in detections:
                            print(f"  - {det.label} at {det.bbox}")

                    frame_number += 1

                # Handle keyboard input
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"\n{'Paused' if paused else 'Resumed'}")
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_number}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        print(f"\nScreenshot saved: {screenshot_path}")
                else:
                    # No display, just check for early termination
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            # Print summary
            print(f"\n{'='*60}")
            print("Processing Summary:")
            print(f"  Total Frames Processed: {frame_number}")
            print(f"  Average FPS: {frame_number/self.total_processing_time:.2f}")
            print(f"  Total Processing Time: {self.total_processing_time:.2f}s")
            if output_path:
                print(f"  Output saved to: {output_path}")
            print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LLaVA Real-Time Object Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track objects in video
  python llava_tracker.py --video path/to/video.mp4

  # Save annotated video
  python llava_tracker.py --video input.mp4 --output output.mp4

  # Process more frames for higher accuracy (slower)
  python llava_tracker.py --video input.mp4 --interval 1

  # Run without display (headless)
  python llava_tracker.py --video input.mp4 --no-display
        """
    )

    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save annotated video (optional)')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Path to LLaVA model (downloads default if not specified)')
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='Process every N frames (default: 5, lower is slower but more accurate)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display window')
    parser.add_argument('--no-metal', action='store_true',
                       help='Disable Metal acceleration on macOS')

    args = parser.parse_args()

    # Validate video path
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Initialize tracker
    try:
        tracker = LLaVATracker(
            model_path=args.model,
            use_metal=not args.no_metal
        )
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        sys.exit(1)

    # Run tracking
    try:
        tracker.track_video(
            video_path=args.video,
            output_path=args.output,
            display=not args.no_display,
            process_every_n=args.interval
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error during tracking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

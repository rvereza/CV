#!/usr/bin/env python3
"""
Example Usage of LLaVA Object Tracking System

This script demonstrates various ways to use the tracking system
"""

from llava_tracker import LLaVATracker
import sys


def example_basic_tracking():
    """Example 1: Basic video tracking"""
    print("=" * 60)
    print("Example 1: Basic Video Tracking")
    print("=" * 60)

    # Initialize tracker
    tracker = LLaVATracker(use_metal=True)

    # Track objects in video
    video_path = "sample_video.mp4"  # Replace with your video path

    try:
        tracker.track_video(
            video_path=video_path,
            display=True,
            process_every_n=5
        )
    except FileNotFoundError:
        print(f"\nError: Video file '{video_path}' not found.")
        print("Please provide a valid video path.")


def example_save_output():
    """Example 2: Track and save annotated video"""
    print("\n" + "=" * 60)
    print("Example 2: Track and Save Output")
    print("=" * 60)

    tracker = LLaVATracker(use_metal=True)

    video_path = "sample_video.mp4"  # Replace with your video path
    output_path = "tracked_output.mp4"

    try:
        tracker.track_video(
            video_path=video_path,
            output_path=output_path,
            display=True,
            process_every_n=5
        )
        print(f"\nAnnotated video saved to: {output_path}")
    except FileNotFoundError:
        print(f"\nError: Video file '{video_path}' not found.")


def example_headless_processing():
    """Example 3: Headless processing (no display)"""
    print("\n" + "=" * 60)
    print("Example 3: Headless Processing")
    print("=" * 60)

    tracker = LLaVATracker(use_metal=True)

    video_path = "sample_video.mp4"
    output_path = "tracked_output.mp4"

    try:
        tracker.track_video(
            video_path=video_path,
            output_path=output_path,
            display=False,  # No display window
            process_every_n=10  # Process fewer frames for speed
        )
    except FileNotFoundError:
        print(f"\nError: Video file '{video_path}' not found.")


def example_test_on_image():
    """Example 4: Test on a single image"""
    print("\n" + "=" * 60)
    print("Example 4: Test on Single Image")
    print("=" * 60)

    import cv2
    import numpy as np

    # Create or load a test image
    # Option 1: Create a blank image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Sample Image", (200, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Option 2: Load from file
    # test_image = cv2.imread("sample_image.jpg")

    # Initialize tracker
    tracker = LLaVATracker(use_metal=True)

    # Process single frame
    print("Processing image...")
    annotated_frame, detections = tracker.process_frame(test_image, frame_number=0)

    # Display results
    print(f"\nDetections: {len(detections)}")
    for det in detections:
        print(f"  - {det.label} at {det.bbox}")

    # Save or display
    cv2.imwrite("test_output.jpg", annotated_frame)
    print("\nAnnotated image saved to: test_output.jpg")


def example_custom_objects():
    """Example 5: Track custom objects"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Object Types")
    print("=" * 60)

    # You can modify TARGET_OBJECTS in llava_tracker.py
    # Or create a custom tracker with different objects

    print("To track custom objects:")
    print("1. Edit llava_tracker.py")
    print("2. Modify the TARGET_OBJECTS list in LLaVATracker class")
    print("3. Add corresponding colors in the colors dictionary")
    print("\nExample:")
    print('TARGET_OBJECTS = ["car", "person", "dog", "cat"]')


def main():
    """Main function to run examples"""
    print("\n" + "=" * 60)
    print("LLaVA Object Tracking System - Examples")
    print("=" * 60)

    print("\nAvailable Examples:")
    print("1. Basic video tracking")
    print("2. Track and save output")
    print("3. Headless processing (no display)")
    print("4. Test on single image")
    print("5. Custom object types (info)")
    print("0. Run all examples")

    try:
        choice = input("\nSelect example (0-5): ").strip()

        if choice == "1":
            example_basic_tracking()
        elif choice == "2":
            example_save_output()
        elif choice == "3":
            example_headless_processing()
        elif choice == "4":
            example_test_on_image()
        elif choice == "5":
            example_custom_objects()
        elif choice == "0":
            # Run all (except headless which doesn't show anything)
            example_test_on_image()
            example_custom_objects()
            print("\n\nFor video examples, run them individually with a video file.")
        else:
            print("Invalid choice. Please select 0-5.")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Location Parser for LLaVA Text Responses
Extracts object locations and bounding boxes from natural language descriptions
"""

import re
from typing import List, Tuple, Optional, Dict
import numpy as np


class LocationParser:
    """
    Parses LLaVA text responses to extract object locations
    """

    # Common position keywords and their approximate coordinates
    POSITION_KEYWORDS = {
        # Horizontal positions
        "left": {"x": 20, "range": (0, 35)},
        "center": {"x": 50, "range": (35, 65)},
        "middle": {"x": 50, "range": (35, 65)},
        "right": {"x": 80, "range": (65, 100)},

        # Vertical positions
        "top": {"y": 20, "range": (0, 35)},
        "upper": {"y": 20, "range": (0, 35)},
        "bottom": {"y": 80, "range": (65, 100)},
        "lower": {"y": 80, "range": (65, 100)},
    }

    # Size keywords
    SIZE_KEYWORDS = {
        "small": {"scale": 0.15},
        "tiny": {"scale": 0.10},
        "medium": {"scale": 0.30},
        "large": {"scale": 0.50},
        "huge": {"scale": 0.70},
        "massive": {"scale": 0.80},
    }

    def __init__(self):
        """Initialize location parser"""
        pass

    def extract_detected_objects(self, response: str, target_objects: List[str]) -> List[str]:
        """
        Extract which objects were detected from response

        Args:
            response: LLaVA response text
            target_objects: List of target object names

        Returns:
            List of detected object names
        """
        detected = []
        response_lower = response.lower()

        # Look for explicit DETECTED: format
        if "detected:" in response_lower:
            detected_section = response_lower.split("detected:")[-1].split("\n")[0]

            # Check if "none" is mentioned
            if "none" in detected_section:
                return []

            # Look for each target object
            for obj in target_objects:
                if obj.lower() in detected_section:
                    detected.append(obj)

        else:
            # Fallback: look for mentions of target objects anywhere in response
            for obj in target_objects:
                obj_lower = obj.lower()

                # Check for various phrasings
                patterns = [
                    f"\\b{obj_lower}\\b",
                    f"\\b{obj_lower}s\\b",  # plural
                    f"see.*{obj_lower}",
                    f"detect.*{obj_lower}",
                    f"visible.*{obj_lower}",
                    f"there.*{obj_lower}",
                ]

                for pattern in patterns:
                    if re.search(pattern, response_lower):
                        if obj not in detected:
                            detected.append(obj)
                        break

        return detected

    def parse_location(self, response: str, image_width: int,
                      image_height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Parse location from LLaVA response

        Args:
            response: LLaVA response text
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Bounding box as (x1, y1, x2, y2) or None
        """
        # Try different parsing strategies in order

        # Strategy 1: Explicit LOCATION: format
        bbox = self._parse_location_format(response, image_width, image_height)
        if bbox:
            return bbox

        # Strategy 2: BBOX: format
        bbox = self._parse_bbox_format(response, image_width, image_height)
        if bbox:
            return bbox

        # Strategy 3: Parse from natural language description
        bbox = self._parse_natural_language(response, image_width, image_height)
        if bbox:
            return bbox

        # Strategy 4: Look for numeric coordinates
        bbox = self._parse_numeric_coordinates(response, image_width, image_height)
        if bbox:
            return bbox

        return None

    def _parse_location_format(self, response: str, width: int,
                               height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Parse LOCATION: center_x, center_y, width, height format

        Returns:
            (x1, y1, x2, y2) or None
        """
        # Look for LOCATION: pattern
        match = re.search(r'LOCATION:\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',
                         response, re.IGNORECASE)

        if match:
            center_x = float(match.group(1))
            center_y = float(match.group(2))
            obj_width = float(match.group(3))
            obj_height = float(match.group(4))

            # Convert from percentages to pixels
            center_x_px = int(center_x * width / 100)
            center_y_px = int(center_y * height / 100)
            obj_width_px = int(obj_width * width / 100)
            obj_height_px = int(obj_height * height / 100)

            # Calculate bounding box
            x1 = max(0, center_x_px - obj_width_px // 2)
            y1 = max(0, center_y_px - obj_height_px // 2)
            x2 = min(width, center_x_px + obj_width_px // 2)
            y2 = min(height, center_y_px + obj_height_px // 2)

            return (x1, y1, x2, y2)

        return None

    def _parse_bbox_format(self, response: str, width: int,
                          height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Parse BBOX: left, top, right, bottom format

        Returns:
            (x1, y1, x2, y2) or None
        """
        # Look for BBOX: pattern
        match = re.search(r'BBOX:\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',
                         response, re.IGNORECASE)

        if match:
            left = float(match.group(1))
            top = float(match.group(2))
            right = float(match.group(3))
            bottom = float(match.group(4))

            # Convert from percentages to pixels
            x1 = int(left * width / 100)
            y1 = int(top * height / 100)
            x2 = int(right * width / 100)
            y2 = int(bottom * height / 100)

            # Validate and clamp
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Ensure x2 > x1 and y2 > y1
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)

        return None

    def _parse_natural_language(self, response: str, width: int,
                                height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Parse location from natural language description

        Returns:
            (x1, y1, x2, y2) or None
        """
        response_lower = response.lower()

        # Extract position information
        x_pos = 50  # default center
        y_pos = 50  # default center
        obj_width = 30  # default size
        obj_height = 30

        # Find horizontal position
        if "left" in response_lower:
            x_pos = self.POSITION_KEYWORDS["left"]["x"]
            if "far left" in response_lower or "very left" in response_lower:
                x_pos = 10
        elif "right" in response_lower:
            x_pos = self.POSITION_KEYWORDS["right"]["x"]
            if "far right" in response_lower or "very right" in response_lower:
                x_pos = 90
        elif "center" in response_lower or "middle" in response_lower:
            x_pos = 50

        # Find vertical position
        if "top" in response_lower or "upper" in response_lower:
            y_pos = self.POSITION_KEYWORDS["top"]["y"]
            if "very top" in response_lower:
                y_pos = 10
        elif "bottom" in response_lower or "lower" in response_lower:
            y_pos = self.POSITION_KEYWORDS["bottom"]["y"]
            if "very bottom" in response_lower:
                y_pos = 90
        elif "center" in response_lower or "middle" in response_lower:
            y_pos = 50

        # Find size information
        for size_key, size_info in self.SIZE_KEYWORDS.items():
            if size_key in response_lower:
                scale = size_info["scale"]
                obj_width = scale * 100
                obj_height = scale * 100
                break

        # Adjust size based on descriptors
        if "occupies" in response_lower or "fills" in response_lower:
            obj_width *= 1.5
            obj_height *= 1.5

        # Convert to pixel coordinates
        center_x = int(x_pos * width / 100)
        center_y = int(y_pos * height / 100)
        w = int(obj_width * width / 100)
        h = int(obj_height * height / 100)

        x1 = max(0, center_x - w // 2)
        y1 = max(0, center_y - h // 2)
        x2 = min(width, center_x + w // 2)
        y2 = min(height, center_y + h // 2)

        return (x1, y1, x2, y2)

    def _parse_numeric_coordinates(self, response: str, width: int,
                                   height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Try to find any numeric coordinates in the response

        Returns:
            (x1, y1, x2, y2) or None
        """
        # Look for patterns like "at 45%, 30%" or "x: 45, y: 30"
        patterns = [
            r'(\d+\.?\d*)%\s*,\s*(\d+\.?\d*)%',
            r'x:\s*(\d+\.?\d*)\s*,\s*y:\s*(\d+\.?\d*)',
            r'horizontal:\s*(\d+\.?\d*)\s*,\s*vertical:\s*(\d+\.?\d*)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Use the first match
                x_pct, y_pct = float(matches[0][0]), float(matches[0][1])

                # Assume default size
                w_pct, h_pct = 25.0, 25.0

                # Convert to pixels
                center_x = int(x_pct * width / 100)
                center_y = int(y_pct * height / 100)
                w = int(w_pct * width / 100)
                h = int(h_pct * height / 100)

                x1 = max(0, center_x - w // 2)
                y1 = max(0, center_y - h // 2)
                x2 = min(width, center_x + w // 2)
                y2 = min(height, center_y + h // 2)

                return (x1, y1, x2, y2)

        return None

    def parse_multiple_locations(self, response: str, width: int,
                                height: int) -> List[Tuple[int, int, int, int]]:
        """
        Parse multiple object locations from response

        Args:
            response: LLaVA response
            width: Image width
            height: Image height

        Returns:
            List of bounding boxes
        """
        bboxes = []

        # Split by numbered items
        sections = re.split(r'#\d+:', response)

        for section in sections[1:]:  # Skip first (before #1)
            bbox = self.parse_location(section, width, height)
            if bbox:
                bboxes.append(bbox)

        return bboxes

    def extract_confidence(self, response: str) -> str:
        """
        Extract confidence level from response

        Args:
            response: LLaVA response

        Returns:
            Confidence level (high/medium/low)
        """
        response_lower = response.lower()

        if "high confidence" in response_lower or "very confident" in response_lower:
            return "high"
        elif "low confidence" in response_lower or "uncertain" in response_lower:
            return "low"
        elif "medium confidence" in response_lower or "somewhat confident" in response_lower:
            return "medium"

        # Default
        return "medium"


# Test the parser
def test_parser():
    """Test the location parser"""
    print("Testing Location Parser...\n")

    parser = LocationParser()

    # Test detection extraction
    test_response1 = "DETECTED: war tank, airplane, destroyer"
    detected = parser.extract_detected_objects(test_response1,
                                               ["war tank", "airplane", "tug boat", "destroyer"])
    print(f"Detected objects: {detected}")

    # Test location parsing
    test_response2 = "LOCATION: 45, 30, 25, 15"
    bbox = parser.parse_location(test_response2, 1920, 1080)
    print(f"Parsed location (explicit): {bbox}")

    # Test natural language parsing
    test_response3 = "The tank is in the upper left corner of the image, it's quite large."
    bbox = parser.parse_natural_language(test_response3, 1920, 1080)
    print(f"Parsed location (natural language): {bbox}")

    # Test BBOX format
    test_response4 = "BBOX: 10, 20, 40, 60"
    bbox = parser.parse_bbox_format(test_response4, 1920, 1080)
    print(f"Parsed location (bbox format): {bbox}")

    print("\nParser test complete!")


if __name__ == "__main__":
    test_parser()

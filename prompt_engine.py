"""
Intelligent Prompt Engineering for Object Detection with LLaVA
Creates structured prompts to extract object locations from text-based responses
"""

from typing import List, Optional, Dict
import random


class PromptEngine:
    """
    Generates optimized prompts for object detection and localization
    """

    def __init__(self, target_objects: List[str]):
        """
        Initialize prompt engine

        Args:
            target_objects: List of objects to detect
        """
        self.target_objects = target_objects
        self.object_list_str = ", ".join(target_objects)

    def create_detection_prompt(self) -> str:
        """
        Create prompt for initial object detection

        Returns:
            Detection prompt string
        """
        prompt = f"""List any: {self.object_list_str}
DETECTED:"""

        return prompt

    def create_region_prompt(self, object_name: str) -> str:
        """
        Create prompt for fast region-based location

        Args:
            object_name: Name of object to locate

        Returns:
            Region prompt string
        """
        prompt = f"""Where is the {object_name}? Answer with ONE word:
top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right
REGION:"""

        return prompt

    def create_localization_prompt(self, object_name: str) -> str:
        """
        Create prompt to locate a specific object

        Args:
            object_name: Name of object to locate

        Returns:
            Localization prompt string
        """
        prompt = f"""Where is {object_name}? LOCATION:"""

        return prompt

    def create_detailed_localization_prompt(self, object_name: str) -> str:
        """
        Create detailed prompt for precise localization

        Args:
            object_name: Name of object to locate

        Returns:
            Detailed localization prompt
        """
        prompt = f"""I need to draw a bounding box around the {object_name} in this image.

Please provide the exact position using percentages:

1. Where is the LEFT edge of the {object_name}? (0% = left edge of image, 100% = right edge)
2. Where is the TOP edge of the {object_name}? (0% = top of image, 100% = bottom)
3. Where is the RIGHT edge of the {object_name}?
4. Where is the BOTTOM edge of the {object_name}?

Respond in this EXACT format:
BBOX: left%, top%, right%, bottom%

Example: BBOX: 10, 20, 40, 60

This means:
- Left edge at 10% from left
- Top edge at 20% from top
- Right edge at 40% from left
- Bottom edge at 60% from top

What are the coordinates for the {object_name}?"""

        return prompt

    def create_region_detection_prompt(self, regions: List[str]) -> str:
        """
        Create prompt to detect objects in specific regions (grid-based approach)

        Args:
            regions: List of region names (e.g., ["top-left", "center", "bottom-right"])

        Returns:
            Region detection prompt
        """
        region_desc = ", ".join(regions)

        prompt = f"""Let's examine this image region by region.

For each of these regions: {region_desc}

Tell me if you see any of these objects: {self.object_list_str}

Respond in this format:
REGION_DETECTIONS:
- top-left: [objects detected]
- top-center: [objects detected]
- top-right: [objects detected]
- center-left: [objects detected]
- center: [objects detected]
- center-right: [objects detected]
- bottom-left: [objects detected]
- bottom-center: [objects detected]
- bottom-right: [objects detected]

If no objects in a region, write "none"

What do you see in each region?"""

        return prompt

    def create_counting_prompt(self) -> str:
        """
        Create prompt to count detected objects

        Returns:
            Counting prompt string
        """
        prompt = f"""Count how many of each of these objects appear in this image:
{self.object_list_str}

Respond in this EXACT format:
COUNT:
- war tanks: [number]
- airplanes: [number]
- tug boats: [number]
- tanker vessels: [number]
- destroyers: [number]

Use 0 if none are present.

How many of each object do you count?"""

        return prompt

    def create_verification_prompt(self, object_name: str, bbox: tuple) -> str:
        """
        Create prompt to verify if an object is at a specific location

        Args:
            object_name: Object to verify
            bbox: Bounding box (x1, y1, x2, y2) in percentages

        Returns:
            Verification prompt
        """
        x1, y1, x2, y2 = bbox

        prompt = f"""I believe there is a {object_name} in this region of the image:
- Left: {x1}% from the left edge
- Top: {y1}% from the top edge
- Right: {x2}% from the left edge
- Bottom: {y2}% from the top edge

Is there a {object_name} in this region?

Respond in this EXACT format:
VERIFICATION: yes/no
CONFIDENCE: high/medium/low
CORRECTION: [if location is wrong, provide correct percentages]

Your response:"""

        return prompt

    def create_description_prompt(self, object_name: str) -> str:
        """
        Create prompt to get detailed description of object

        Args:
            object_name: Object to describe

        Returns:
            Description prompt
        """
        prompt = f"""Describe the {object_name} in this image in detail:

1. What does it look like?
2. What is it doing or its orientation?
3. Approximately where in the image is it located? (left/center/right, top/middle/bottom)
4. How large is it compared to the full image? (small/medium/large)
5. Are there any distinctive features?

Please be specific and accurate."""

        return prompt

    def create_multiple_objects_prompt(self, object_name: str) -> str:
        """
        Create prompt to handle multiple instances of same object

        Args:
            object_name: Object type

        Returns:
            Multi-object prompt
        """
        prompt = f"""I see there might be multiple {object_name}s in this image.

For EACH {object_name} you can see, provide its location:

{object_name} #1:
LOCATION: center_x, center_y, width, height

{object_name} #2:
LOCATION: center_x, center_y, width, height

(continue if there are more)

If you only see one, just provide one location.
If you see none, respond with: NONE

Use the 0-100 percentage coordinate system where (0,0) is top-left.

List all {object_name}s:"""

        return prompt

    def create_contextual_prompt(self) -> str:
        """
        Create prompt with contextual awareness

        Returns:
            Contextual detection prompt
        """
        prompt = f"""You are analyzing a video frame for military vehicle and vessel detection.

Target objects to identify:
{self.object_list_str}

Please analyze this frame and provide:

1. ENVIRONMENT: [describe the scene - is it aerial view, ground level, naval, etc.]
2. DETECTED: [list all target objects you can identify]
3. CONFIDENCE: [high/medium/low for each detected object]

Be thorough but accurate. This is for tracking purposes.

Your analysis:"""

        return prompt

    @staticmethod
    def create_followup_prompt(previous_response: str, object_name: str) -> str:
        """
        Create follow-up prompt based on previous response

        Args:
            previous_response: Previous model response
            object_name: Object to clarify

        Returns:
            Follow-up prompt
        """
        prompt = f"""You mentioned seeing a {object_name}.

Can you be more specific about its location?

Using percentages where 0% is the left/top edge and 100% is the right/bottom edge:

LEFT EDGE of {object_name}: ___%
TOP EDGE of {object_name}: ___%
RIGHT EDGE of {object_name}: ___%
BOTTOM EDGE of {object_name}: ___%

Fill in the percentages:"""

        return prompt


# Test prompts
def test_prompts():
    """Test the prompt engine"""
    print("Testing Prompt Engine...\n")

    target_objects = ["war tank", "airplane", "destroyer"]
    engine = PromptEngine(target_objects)

    print("=== Detection Prompt ===")
    print(engine.create_detection_prompt())

    print("\n=== Localization Prompt ===")
    print(engine.create_localization_prompt("war tank"))

    print("\n=== Detailed Localization Prompt ===")
    print(engine.create_detailed_localization_prompt("airplane"))

    print("\n=== Verification Prompt ===")
    print(engine.create_verification_prompt("destroyer", (10, 20, 40, 60)))

    print("\n=== Multiple Objects Prompt ===")
    print(engine.create_multiple_objects_prompt("war tank"))

    print("\nPrompt engine test complete!")


if __name__ == "__main__":
    test_prompts()

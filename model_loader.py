"""
LLaVA Model Loader with Metal Acceleration Support
Handles loading and inference with LLaVA vision-language model
"""

import torch
import numpy as np
from PIL import Image
import platform
from typing import Optional, Union
import sys
import os


class LLaVAModel:
    """
    Wrapper for LLaVA model with optimizations for macOS Metal
    """

    def __init__(self, model_path: Optional[str] = None, use_metal: bool = True):
        """
        Initialize LLaVA model

        Args:
            model_path: Path to model weights (uses default if None)
            use_metal: Enable Metal acceleration on macOS
        """
        self.device = self._setup_device(use_metal)
        # Use the HuggingFace version which has proper structure
        self.model_path = model_path or "llava-hf/llava-1.5-7b-hf"

        print(f"Loading LLaVA model: {self.model_path}")
        print(f"Device: {self.device}")

        try:
            self._load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying to install required packages...")
            self._install_dependencies()
            self._load_model()

    def _setup_device(self, use_metal: bool) -> str:
        """
        Setup compute device (Metal on macOS, CUDA on others, or CPU)

        Args:
            use_metal: Whether to use Metal on macOS

        Returns:
            Device string
        """
        system = platform.system()

        if system == "Darwin" and use_metal:
            # macOS with Metal Performance Shaders (MPS)
            if torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("Warning: MPS not built, falling back to CPU")
                    return "cpu"
                print("Using Metal Performance Shaders (MPS) acceleration")
                return "mps"
            else:
                print("Warning: MPS not available, falling back to CPU")
                return "cpu"

        elif torch.cuda.is_available():
            # CUDA GPU
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"

        else:
            # CPU fallback
            print("Using CPU (this will be slower)")
            return "cpu"

    def _install_dependencies(self):
        """Install required packages"""
        try:
            import subprocess
            print("Installing transformers and related packages...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate", "sentencepiece", "protobuf"
            ])
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            raise

    def _load_model(self):
        """Load the LLaVA model and processor"""
        try:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration

            print("Loading model (this may take a few minutes on first run)...")
            print("Downloading model files (~13GB)...")

            # Load processor
            self.processor = LlavaProcessor.from_pretrained(
                self.model_path
            )

            # Load model with appropriate settings for device
            load_kwargs = {
                "low_cpu_mem_usage": True,
            }

            if self.device == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            elif self.device == "mps":
                # MPS works better with float16
                load_kwargs["torch_dtype"] = torch.float16
            else:
                # CPU - use float32
                load_kwargs["torch_dtype"] = torch.float32

            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs
            )

            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()

            print("✓ Model loaded successfully!")

        except ImportError:
            print("\nError: LLaVA requires transformers library")
            print("Installing required packages...")
            self._install_dependencies()
            # Try again
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            self._load_model()

        except Exception as e:
            print(f"\nFailed to load LLaVA model: {e}")
            raise RuntimeError(f"Could not load LLaVA model: {e}")

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """
        Preprocess image for model input

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR to RGB
            image = Image.fromarray(image)

        return image

    @torch.no_grad()
    def generate(self, image: Union[np.ndarray, Image.Image],
                 prompt: str,
                 max_new_tokens: int = 200,
                 temperature: float = 0.2) -> str:
        """
        Generate text response for image and prompt

        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)

        Returns:
            Generated text response
        """
        # Preprocess image
        pil_image = self.preprocess_image(image)

        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=prompt_text,
            images=pil_image,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        # Generate
        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

            # Decode output
            output = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Extract just the assistant's response
            if "ASSISTANT:" in output:
                output = output.split("ASSISTANT:")[-1].strip()
            elif "assistant" in output.lower():
                parts = output.lower().split("assistant")
                if len(parts) > 1:
                    output = output[output.lower().rfind("assistant") + len("assistant"):].strip()

            return output

        except Exception as e:
            print(f"Error during generation: {e}")
            return ""

    def generate_batch(self, images: list, prompts: list,
                      max_new_tokens: int = 200,
                      temperature: float = 0.2) -> list:
        """
        Generate responses for multiple images (for better performance)

        Args:
            images: List of input images
            prompts: List of prompts (one per image)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated text responses
        """
        # For now, process sequentially (batching can be added for optimization)
        results = []
        for image, prompt in zip(images, prompts):
            result = self.generate(image, prompt, max_new_tokens, temperature)
            results.append(result)

        return results


# Test function
def test_model():
    """Test the model loader"""
    print("Testing LLaVA Model Loader...")

    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Initialize model
    model = LLaVAModel()

    # Test generation
    response = model.generate(
        test_image,
        "Describe what you see in this image.",
        max_new_tokens=50
    )

    print(f"\nTest Response: {response}")
    print("\nModel loader test complete!")


if __name__ == "__main__":
    test_model()

# image-captioning-module

## Overview (WIP)

This Python module generates concise 5-7 word captions for images based on user prompts. It supports advanced captioning models like CLIP, BLIP-2, LLaVA-7B, LLaVA-16B, and GPT-4Vision.

## Features

- **Caption Generation**: Generate captions for images using various models.
  
- **Model Variety**: Choose from CLIP, BLIP-2, LLaVA-7B, LLaVA-16B, and GPT-4Vision for captioning.

- **Flexible Usage**: Easily switch between models to optimize caption quality.

## Usage

1. **Input**: Provide an image file and a prompt.
2. **Output**: Get a concise 5-7 word caption for the image based on the prompt.

## Installation

Install dependencies listed in `requirements.txt`. Then, import and use the module for caption generation.

## Example

```python
from image_captioning_module import generate_caption

image_path = "path/to/image.jpg"
prompt = "What is happening in this image?"

caption = generate_caption(image_path, prompt)
print("Generated Caption:", caption)

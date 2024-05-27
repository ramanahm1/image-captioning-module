
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# Loads image from path. For now, it downloads from the web (you must provide the link)
# Returns: The image
def get_image(img_path):
	try:
		response = requests.get(img_path, stream=True)
		response.raise_for_status()  
		image = Image.open(response.raw)
		return image
	except requests.exceptions.RequestException as e:
		print(f"Error downloading image: {e}")
		return None
		

def generate_caption(image_url, prompt):
  """
  This function takes an image URL and a sentence prompt, uses a CLIP model to analyze the image,
  and generates a concise caption based on the entire prompt.
  """

  # Load pre-trained CLIP model and processor (consider using GPU if available)
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  
  


  # Prepare input for CLIP model (add padding)
  inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)

  # Generate image-text similarity scores
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image

  # Get probabilities (softmax for normalized scores)
  probs = logits_per_image.softmax(dim=1)

  # No longer selecting a single top word - consider the entire prompt

  # Craft a caption based on the prompt (limit to 5-7 words)
  caption_words = prompt.split()[:7]  # Split prompt into words, limit to 7

  # Reduce verbosity (optional)
  # You can add logic here to identify and remove stop words or redundant information
  # from the caption (e.g., using NLTK)

  caption = " ".join(caption_words)

  return caption


import sys
import argparse

def foo(args):
   print(args)

class ModelTypes:
	clip = 'clip'
	blip2 = 'blip2'
	llm = 'llm'

def invoke_clip(image):
  caption = "This is a CLIP-generated caption for the image."
  return caption

def invoke_blip2(image):
  caption = "This is a BLIP2-generated caption for the image."
  return caption

def invoke_llm(image):
  caption = "This is an LLM-generated caption for the image."
  return caption

def generate_caption(args):
  image = get_image(args.image_path)

  if args.model not in ModelTypes.__dict__.values():
    raise ValueError(f"Invalid model argument: {args.model}. Choose from {', '.join(ModelTypes.__dict__.values())}")

  # Invoke appropriate model based on the --model argument
  model_name = args.model
  if model_name == ModelTypes.clip:
    caption = invoke_clip(image)
  elif model_name == ModelTypes.blip2:
    caption = invoke_blip2(image)
  elif model_name == ModelTypes.llm:
    caption = invoke_llm(image)
  else:  # Double safety check
    raise RuntimeError("Unexpected model name: " + model_name)  

  return caption
		


if __name__ == '__main__':
   
   parser = argparse.ArgumentParser(description='Arguments for Caption Generation')
   parser.add_argument('--image_path', type=str, required=True, help='Path to the image (the image must be stored locally)')
   parser.add_argument('--prompt', type=str, required=True, help='For multiple prompts in sequence, separate each with ;')
   parser.add_argument('--model', type=str, default='clip', help='Choose among: clip, blip2, llm')
   
   args = parser.parse_args()


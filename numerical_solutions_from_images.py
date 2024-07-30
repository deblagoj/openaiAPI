#!/usr/bin/env python
# coding: utf-8

import json
import base64
import os
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import openai
import tiktoken
import logging



# Function to encode the image as base64
def encode_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to resize images
def resize_image(image_path: str, max_width: int = 800):
    with Image.open(image_path) as img:
        if img.width > max_width:
            img = img.resize((max_width, int((max_width / img.width) * img.height)))
            img.save(image_path)

# Sample images for testing
image_dir = "/Users/blagojdelipetrev/Code/EUknowledge/Numerical/2"

# Encode all images within the directory
image_files = os.listdir(image_dir)
image_data = {}
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    resize_image(image_path)  # Resize the image before encoding
    encoded_image = encode_image(image_path)
    image_data[image_file.split('.')[0]] = encoded_image
    logging.debug(f"Encoded image: {image_file}")

def display_images(image_data: dict):
    fig, axs = plt.subplots(1, len(image_data), figsize=(18, 6))
    for i, (key, value) in enumerate(image_data.items()):
        try:
            img = Image.open(BytesIO(base64.b64decode(value)))
            ax = axs[i]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(key)
        except Exception as e:
            logging.error(f"Error decoding image {key}: {e}")
    plt.tight_layout()
    plt.show()

display_images(image_data)

# OpenAI API setup
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv('OPENAI_API_KEY'),
)
# Token counting function
def count_tokens(text: str, model: str = "gpt-4-turbo-2024-04-09"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

INSTRUCTION_PROMPT='Can you describe what is in the image below'
MODEL="gpt-4-turbo-2024-04-09"
def gpt_with_image_input(image_data: dict, test_image: str):
    encoded_image = image_data[test_image]
    image_url = f"data:image/jpeg;base64,{encoded_image}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTION_PROMPT},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
        ],
        "temperature": 0.0,  # for less diversity in responses
    }

    # Count tokens
    total_tokens = sum(count_tokens(json.dumps(msg), MODEL) for msg in payload )
    print(f"Total tokens used in the request: {total_tokens}")


    try:
        response = client.chat.completions.create(**payload)
        print(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error processing image {test_image}: {e}")

# Example usage
gpt_with_image_input(image_data, '3')  # Example test image


#!/usr/bin/env python
# coding: utf-8
'''
## Sources 

THIS CODE IS FOR SOLVING ABSTRACT ASSIGMENT FROM A SCREENSHOT WITH FEW EXAMPLES 


- https://cookbook.openai.com/examples/multimodal/using_gpt4_vision_with_function_calling
- https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
- https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models


## To-do 
Make a cycle to go over all the pictures in the folder and generate the ouput, put the numeber of the picture before the html output.
tell gabi gi make the screenshotes and numerate then by numbers

do the same for abstract 

'''
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

# Sample images for prompt
image_dir = "/Users/blagojdelipetrev/Code/EUknowledge/Numerical/abs_prompt_images"

# Encode all images within the prompt and all_images directory 
image_files = os.listdir(image_dir)
image_data_prompt = {}
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    resize_image(image_path)  # Resize the image before encoding
    encoded_image = encode_image(image_path)
    image_data_prompt[image_file.split('.')[0]] = encoded_image
    logging.debug(f"Encoded image: {image_file}")

# Sample images for all images
image_dir = "/Users/blagojdelipetrev/Code/EUknowledge/Numerical/abs_all_images"

# Encode all images within the all_images directory 
image_files = os.listdir(image_dir)
image_data_all = {}
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    resize_image(image_path)  # Resize the image before encoding
    encoded_image = encode_image(image_path)
    image_data_all[image_file.split('.')[0]] = encoded_image
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

display_images(image_data_all)


def display_single_image(image_data: dict, image_key: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        value = image_data[image_key]
        img = Image.open(BytesIO(base64.b64decode(value)))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(image_key)
    except Exception as e:
        logging.error(f"Error decoding image {image_key}: {e}")
    plt.tight_layout()
    plt.show()

# Example usage
#display_single_image(image_data_prompt, '5')

# OpenAI API setup
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv('OPENAI_API_KEY'),
)

# Token counting function
def count_tokens(text: str, model: str = "gpt-4-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

INSTRUCTION_PROMPT = 'You are a helpful assistant that solves ESPO abstract assignments. You are given the solution of the assigment, please explain it step by step and generate HTML output. '
RESPOSE_1 = '<p>Rule 1: The bottom of the boat alternates between flat and rounded.</p><p>Rule 2: The square flag alternates between being raised to the top and raised to the bottom.</p><p>Rule 3: The arrow in the square flag indicates the position where the next triangular flag should be moved.</p><p> <b> The solutions is E.</b></b></p>'
RESPOSE_2 = '<p>Rule 1: The flask transitions from empty to half full to full then repeats this pattern.<br>Rule 2: The total number of bubbles alternates between two and three.</p><p> <b> The solutions is D.</b></b></p>'
MODEL = "gpt-4o"

#MODEL = "gpt-4-turbo-2024-04-09"
def process_all_images(image_data_prompt: dict, image_data_all: dict):
    results = {}
    for image_key, encoded_image in image_data_all.items():
        image_url1 = f"data:image/jpeg;base64,{image_data_prompt['39_B_E']}"
        image_url2 = f"data:image/jpeg;base64,{image_data_prompt['148_A_D']}"
        image_url = f"data:image/jpeg;base64,{encoded_image}"

        # Extract the solution figure from the image filename (last character after underscore)
        solution_figure = image_key.split('_')[-1][-1]
        print("soluion is ", solution_figure)
        messages = [
                {"role": "system", "content": INSTRUCTION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "The solution is figure E."},
                        {"type": "image_url", "image_url": {"url": image_url1}}
                    ]
                },
                {"role": "assistant", "content": RESPOSE_1},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "The solution is figure D."},
                        {"type": "image_url", "image_url": {"url": image_url2}}
                    ]
                },
                {"role": "assistant", "content": RESPOSE_2},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Finally, please analyse this image. The solution is figure {solution_figure}, can you explain step by step?"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
                ]
        
        # Print the messages
       # print("these are the messages ")
       # print(json.dumps(messages, indent=4))
       # print("______________________________")

        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,  # for less diversity in responses
        }

        # Count tokens
        total_tokens = sum(count_tokens(json.dumps(msg), MODEL) for msg in messages)
        print(f"Total tokens used in the request: {total_tokens}")

        try:
            response = client.chat.completions.create(**payload)
            results[image_key] = response.choices[0].message.content
            print(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error processing image {image_key}: {e}")

         # Save results to JSON file
    with open('abs_output_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)


# Example usage
# display_single_image(image_data, '3')  # Display a single image
# Example usage
process_all_images(image_data_prompt, image_data_all)


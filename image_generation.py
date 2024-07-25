import base64
import io
import json
import os
import sys

# External dependencies
import boto3
from PIL import Image
import botocore

boto3_bedrock = boto3.client("bedrock-runtime")
prompt = "a beautiful lake surrounded by trees with a mountain range at the distance"
negative_prompts = "poorly rendered, poor background details, poorly drawn mountains, disfigured mountain features"

body = json.dumps(
    {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,  # Required
            "negativeText": negative_prompts,  # Optional
        },
        "imageGenerationConfig": {
            "numberOfImages": 2,
            "quality": "standard",  # Options: standard or premium
            "height": 1024,  # Supported height list in the docs
            "width": 1024,  # Supported width list in the docs
            "cfgScale": 7.5,  # Range: 1.0 (exclusive) to 10.0
            "seed": 42,  # Range: 0 to 214783647
        },
    }
)

# Make model request
response = boto3_bedrock.invoke_model(
    body=body,
    modelId="amazon.titan-image-generator-v1",
    accept="application/json",
    contentType="application/json",
)

# Process the image
response_body = json.loads(response.get("body").read())
img1_b64 = response_body["images"][0]

# Debug
print(f"Output: {img1_b64[0:80]}...")


os.makedirs("images", exist_ok=True)

# Decode + save
img1 = Image.open(io.BytesIO(base64.decodebytes(bytes(img1_b64, "utf-8"))))
img1.save(f"images/image_1.png")

change_prompt = "add a house on the lake shore"
negative_prompt = "bad quality, low resolution, cartoon"

body = json.dumps(
    {
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": {
            "text": change_prompt,  # Optional
            "negativeText": negative_prompts,  # Optional
            "images": [img1_b64],  # One image is required
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": 1024,
            "width": 1024,
            "cfgScale": 10,
            "seed": 42,
        },
    }
)

# Model invocation
response = boto3_bedrock.invoke_model(
    body=body,
    modelId="amazon.titan-image-generator-v1",
    accept="application/json",
    contentType="application/json",
)

# Output processing
response_body = json.loads(response.get("body").read())
img2_b64 = response_body["images"][0]

# Debug
print(f"Output: {img2_b64[0:80]}...")
os.makedirs("images", exist_ok=True)

# Decode + save
img2 = Image.open(io.BytesIO(base64.decodebytes(bytes(img2_b64, "utf-8"))))
img2.save("images/image_2.png")

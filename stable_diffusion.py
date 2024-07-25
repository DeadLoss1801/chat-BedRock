import base64
import io
import json
import os
import sys

import boto3
from PIL import Image
import botocore

boto3_bedrock = boto3.client("bedrock-runtime")


prompt = "a beautiful mountain landscape"
negative_prompts = [
    "poorly rendered",
    "poor background details",
    "poorly drawn mountains",
    "disfigured mountain features",
]
style_preset = "photographic"
clip_guidance_preset = "FAST_GREEN"
sampler = "K_DPMPP_2S_ANCESTRAL"
width = 768


request = json.dumps(
    {
        "text_prompts": (
            [{"text": prompt, "weight": 1.0}]
            + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
        ),
        "cfg_scale": 5,
        "seed": 42,
        "steps": 60,
        "style_preset": style_preset,
        "clip_guidance_preset": clip_guidance_preset,
        "sampler": sampler,
        "width": width,
    }
)
modelId = "stability.stable-diffusion-xl-v1"

response = boto3_bedrock.invoke_model(body=request, modelId=modelId)
response_body = json.loads(response.get("body").read())

print(response_body["result"])
base_64_img_str = response_body["artifacts"][0].get("base64")
print(f"{base_64_img_str[0:80]}...")


os.makedirs("images", exist_ok=True)
image_1 = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8"))))
image_1.save("images/image_3_1.png")

import base64
import requests
from src import config_gpt
import openai

# OpenAI API Key
api_key = "oaip_belbKkpnFZLbGPJXzqBiDcWkYJCVPpbs"
openai.api_key  =  config_gpt.openai_api_key
openai.api_base = config_gpt.openai_api_base
openai.api_type = config_gpt.openai_api_type
openai.api_version = config_gpt.openai_api_version
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "data/images/Picture1.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What’s in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}
response = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages = payload["messages"],
            temperature = 0,
            max_tokens = 1024,
            top_p = 0.95,
            n = 1,
            frequency_penalty = 0,
            presence_penalty = 0,
            stop = None)
#response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response)
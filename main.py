import torch

from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


def test_dinov2():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-base')
    model = AutoModel.from_pretrained('facebook/dinov2-with-registers-base')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print("Testing DinoV2")
    print(f"Output Shape: {outputs.pooler_output.shape}")

def test_dinov3():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-base')
    model = AutoModel.from_pretrained('facebook/dinov2-with-registers-base')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print("Testing DinoV3")
    print(f"Output Shape: {outputs.pooler_output.shape}")

def main():
    print("Hello from pxmodel-general-localization!")
    test_dinov3()
    test_dinov2()

if __name__ == "__main__":
    main()

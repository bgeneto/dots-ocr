import json
import io
import base64
import math
from PIL import Image
import requests
from dots_ocr.utils.image_utils import PILimage_to_base64
from openai import OpenAI, AsyncOpenAI
import os
import asyncio
from typing import List


def inference_with_vllm(
    image,
    prompt,
    ip="localhost",
    port=8000,
    temperature=0.1,
    top_p=0.9,
    max_completion_tokens=32768,
    model_name="model",
):

    addr = f"http://{ip}:{port}/v1"
    client = OpenAI(api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr)
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": PILimage_to_base64(image)},
                },
                {
                    "type": "text",
                    "text": f"<|img|><|imgpad|><|endofimg|>{prompt}",
                },  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response = response.choices[0].message.content
        return response
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        # Re-raise connection errors so they can be handled properly
        if any(
            keyword in str(e).lower()
            for keyword in ["connection", "timeout", "refused", "unreachable"]
        ):
            raise e
        return None


async def inference_with_vllm_async(
    image,
    prompt,
    ip="localhost",
    port=8000,
    temperature=0.1,
    top_p=0.9,
    max_completion_tokens=32768,
    model_name="model",
):
    """
    Async version of inference_with_vllm for better performance in batch processing
    """
    addr = f"http://{ip}:{port}/v1"
    client = AsyncOpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"},
            ],
        }
    ]
    try:
        response = await client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"async request error: {e}")
        # Re-raise connection errors so they can be handled properly
        if any(
            keyword in str(e).lower()
            for keyword in ["connection", "timeout", "refused", "unreachable"]
        ):
            raise e
        return None
    finally:
        await client.close()


def inference_with_vllm_batch(
    images: List[Image.Image],
    prompts: List[str],
    ip="localhost",
    port=8000,
    temperature=0.1,
    top_p=0.9,
    max_completion_tokens=32768,
    model_name="model",
) -> List[str]:
    """
    Batch inference using async calls for better performance

    Args:
        images: List of PIL Image objects
        prompts: List of prompt strings (same length as images)

    Returns:
        List of response strings in the same order as input
    """
    if len(images) != len(prompts):
        raise ValueError(
            f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})"
        )

    async def run_batch():
        tasks = []
        for image, prompt in zip(images, prompts):
            task = inference_with_vllm_async(
                image,
                prompt,
                ip,
                port,
                temperature,
                top_p,
                max_completion_tokens,
                model_name,
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    # Run the async batch processing
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    results = loop.run_until_complete(run_batch())

    # Handle any exceptions in the results
    processed_results = []
    connection_errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Check if it's a connection error
            if any(
                keyword in str(result).lower()
                for keyword in ["connection", "timeout", "refused", "unreachable"]
            ):
                connection_errors.append(result)
                print(f"Connection error for image {i}: {result}")
            else:
                print(f"Error processing image {i}: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)

    # If we have connection errors, raise the first one to alert the user
    if connection_errors:
        raise connection_errors[0]

    return processed_results

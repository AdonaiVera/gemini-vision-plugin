"""Gemini Vision plugin.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import io
import tempfile
from datetime import datetime

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo
import fiftyone.core.utils as fou

import base64
import requests
from PIL import Image


def allows_gemini_models(ctx):
    """Returns whether the current environment allows Gemini models."""
    return "GEMINI_API_KEY" in ctx.secrets.keys()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_video(video_path):
    """Encode video file to base64."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def get_video_size_mb(video_path):
    """Get video file size in MB."""
    size_bytes = os.path.getsize(video_path)
    return size_bytes / (1024 * 1024)


def get_closest_aspect_ratio(width, height):
    """Calculate the closest supported aspect ratio from image dimensions."""
    supported_ratios = {
        "1:1": 1.0,
        "2:3": 2/3,
        "3:2": 3/2,
        "3:4": 3/4,
        "4:3": 4/3,
        "4:5": 4/5,
        "5:4": 5/4,
        "9:16": 9/16,
        "16:9": 16/9,
        "21:9": 21/9,
    }

    image_ratio = width / height
    closest_ratio = min(supported_ratios.items(), key=lambda x: abs(x[1] - image_ratio))
    return closest_ratio[0]


def list_gemini_models(api_key):
    """Returns a list of available Gemini model ids that support generateContent.

    Model ids are returned without the leading "models/" prefix for convenient use
    in the v1 models endpoint path.
    """
    manual_models = ["gemini-3-pro-preview"]

    try:
        headers = {"x-goog-api-key": api_key}
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1/models",
            headers=headers,
            timeout=10,
        )
        data = resp.json()
        models = []
        for m in data.get("models", []):
            methods = m.get("supportedGenerationMethods", [])
            name = m.get("name", "")
            if "generateContent" in methods and name.startswith("models/"):
                models.append(name.split("/", 1)[1])

        all_models = set(manual_models + models)

        def sort_key(model):
            if model.startswith("gemini-3"):
                return (0, model)
            elif model.startswith("gemini-2"):
                return (1, model)
            else:
                return (2, model)

        return sorted(all_models, key=sort_key)
    except Exception:
        return manual_models


def save_image_to_dataset(dataset, base64_data, prompt, operation_type="generated"):
    """Save generated image to the dataset."""
    try:
        img_data = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_data))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{operation_type}_{timestamp}.png"

        dataset_dir = os.path.dirname(dataset.first().filepath) if len(dataset) > 0 else tempfile.gettempdir()
        filepath = os.path.join(dataset_dir, filename)

        img.save(filepath, "PNG")

        sample = fo.Sample(filepath=filepath)
        sample["prompt"] = prompt
        sample["generation_type"] = operation_type
        dataset.add_sample(sample)

        return filepath
    except Exception as e:
        raise ValueError(f"Failed to save image: {str(e)}")


def generate_image(prompt, api_key, aspect_ratio="1:1", model="gemini-3-pro-image-preview"):
    """Generate image from text prompt using Gemini."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    # Gemini 3 uses uppercase modalities
    if model.startswith("gemini-3"):
        response_modalities = ["TEXT", "IMAGE"]
    else:
        response_modalities = ["Image"]

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseModalities": response_modalities,
            "imageConfig": {"aspectRatio": aspect_ratio}
        }
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers=headers,
        json=payload,
    )

    content = response.json()
    if "error" in content:
        err = content.get("error", {})
        raise ValueError(err.get("message") or str(err))

    try:
        if "candidates" not in content:
            raise ValueError(f"No candidates in response. Full response: {content}")

        parts = content["candidates"][0]["content"]["parts"]
        for part in parts:
            if "inlineData" in part:
                return part["inlineData"]["data"]
            elif "inline_data" in part:
                return part["inline_data"]["data"]

        raise ValueError(f"No image data found. Response parts: {parts}")
    except KeyError as e:
        raise ValueError(f"Failed to extract image - missing key: {str(e)}. Response: {content}")
    except Exception as e:
        raise ValueError(f"Failed to extract image: {str(e)}. Response: {content}")


def edit_image(image_path, prompt, api_key, aspect_ratio="1:1", model="gemini-3-pro-image-preview"):
    """Edit image using text prompt."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    base64_image = encode_image(image_path)
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    # Gemini 3 uses uppercase modalities
    if model.startswith("gemini-3"):
        response_modalities = ["TEXT", "IMAGE"]
    else:
        response_modalities = ["Image"]

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "responseModalities": response_modalities,
            "imageConfig": {"aspectRatio": aspect_ratio}
        }
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers=headers,
        json=payload,
    )

    content = response.json()
    if "error" in content:
        err = content.get("error", {})
        raise ValueError(err.get("message") or str(err))

    try:
        parts = content["candidates"][0]["content"]["parts"]
        for part in parts:
            if "inlineData" in part:
                return part["inlineData"]["data"]
            elif "inline_data" in part:
                return part["inline_data"]["data"]
        raise ValueError("No image data in response")
    except Exception as e:
        raise ValueError(f"Failed to extract image: {str(e)}")


def compose_images(image_paths, prompt, api_key, aspect_ratio="1:1", model="gemini-3-pro-image-preview"):
    """Compose multiple images with text prompt."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    parts = []
    for image_path in image_paths[:3]:
        base64_image = encode_image(image_path)
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_image
            }
        })

    parts.append({"text": prompt})

    # Gemini 3 uses uppercase modalities
    if model.startswith("gemini-3"):
        response_modalities = ["TEXT", "IMAGE"]
    else:
        response_modalities = ["Image"]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": response_modalities,
            "imageConfig": {"aspectRatio": aspect_ratio}
        }
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers=headers,
        json=payload,
    )

    content = response.json()
    if "error" in content:
        err = content.get("error", {})
        raise ValueError(err.get("message") or str(err))

    try:
        parts = content["candidates"][0]["content"]["parts"]
        for part in parts:
            if "inlineData" in part:
                return part["inlineData"]["data"]
            elif "inline_data" in part:
                return part["inline_data"]["data"]
        raise ValueError("No image data in response")
    except Exception as e:
        raise ValueError(f"Failed to extract image: {str(e)}")


def analyze_video(video_path, prompt, api_key, task_type="describe", model="gemini-3-pro-preview", thinking_level="high", media_resolution="high"):
    """Analyze video using Gemini Vision API.

    Args:
        video_path: Path to video file
        prompt: User prompt for video analysis
        task_type: Type of analysis (describe, segment, extract, question)
        model: Gemini model to use (default: gemini-3-pro-preview)
        thinking_level: Reasoning depth for Gemini 3.0 (low/high)
        media_resolution: Video frame resolution (low/medium/high)
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    # Get video mime type
    video_ext = video_path.lower().split('.')[-1]
    mime_type_map = {
        'mp4': 'video/mp4',
        'mpeg': 'video/mpeg',
        'mov': 'video/mov',
        'avi': 'video/avi',
        'flv': 'video/x-flv',
        'mpg': 'video/mpg',
        'webm': 'video/webm',
        'wmv': 'video/wmv',
        '3gp': 'video/3gpp',
    }
    mime_type = mime_type_map.get(video_ext, 'video/mp4')

    # Encode video
    base64_video = encode_video(video_path)

    # Build payload
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_video
                    }
                },
                {"text": prompt}
            ]
        }],
        "generationConfig": {}
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers=headers,
        json=payload,
    )

    content = response.json()
    if "error" in content:
        err = content.get("error", {})
        raise ValueError(err.get("message") or str(err))

    try:
        candidates = content.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in response")

        parts = candidates[0]["content"]["parts"]
        response_text = ""
        for part in parts:
            if "text" in part:
                response_text += part["text"]

        if not response_text:
            raise ValueError("No text response from model")

        return response_text
    except Exception as e:
        raise ValueError(f"Failed to analyze video: {str(e)}")


def query_gemini_vision(ctx):
    """Queries a Google Gemini Vision model (multimodal)."""
    dataset = ctx.dataset
    sample_ids = ctx.selected
    query_text = ctx.params.get("query_text", None)
    max_tokens = ctx.params.get("max_tokens", 65536)
    model_name = ctx.params.get("model", "gemini-3-pro-preview")
    thinking_level = ctx.params.get("thinking_level", "high")

    parts = []
    if query_text:
        parts.append({"text": query_text})
    for sample_id in sample_ids:
        filepath = dataset[sample_id].filepath
        base64_image = encode_image(filepath)
        mime_type = "image/jpeg"
        if filepath.lower().endswith(".png"):
            mime_type = "image/png"
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_image,
                }
            }
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }

    api_key = ctx.secrets.get("GEMINI_API_KEY")
    headers = {"Content-Type": "application/json"}

    headers["x-goog-api-key"] = api_key

    api_version = "v1beta" if model_name.startswith("gemini-3") else "v1"
    response = requests.post(
        f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent",
        headers=headers,
        json=payload,
    )

    content = response.json()
    if isinstance(content, str):
        return content
    if "error" in content:
        err = content.get("error", {})
        return err.get("message") or str(err)
    try:
        return (
            content["candidates"][0]["content"]["parts"][0].get("text")
        )
    except Exception:
        return str(content)


class QueryGeminiVision(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="query_gemini_vision",
            label="Gemini: Chat with your images!",
            dynamic=True,
        )
        _config.icon = "/assets/icon_dark.svg"
        _config.dark_icon = "/assets/icon_dark.svg"
        _config.light_icon = "/assets/icon_light.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Gemini",
                icon="/assets/icon_dark.svg",
                dark_icon="/assets/icon_dark.svg",
                light_icon="/assets/icon_light.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Gemini",
            description="Ask a question about the selected image!",
        )

        gemini_flag = allows_gemini_models(ctx)
        if not gemini_flag:
            inputs.message(
                "no_gemini_key",
                label=(
                    "No Gemini API Key. Please set GEMINI_API_KEY in your environment."
                ),
            )
            return types.Property(inputs)

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label=f"You must select a sample to use this operator"
                ),
            )
        else:
            if num_selected > 10:
                inputs.str(
                    "many_samples_warning",
                    view=types.Warning(
                        label=(
                            f"You have {num_selected} samples selected. Gemini may charge"
                            " per image. Are you sure you want to continue?"
                        ),
                    ),
                )

            inputs.str(
                "query_text", label="Query about your images", required=True
            )
            # Populate model dropdown from live list; fall back to text if unavailable
            api_key = ctx.secrets.get("GEMINI_API_KEY")
            model_choices = list_gemini_models(api_key) if api_key else []
            default_model = "gemini-3-pro-preview"
            if model_choices:
                inputs.enum(
                    "model",
                    values=model_choices,
                    default=default_model if default_model in model_choices else model_choices[0],
                    label="Model",
                    description="Select a Gemini model",
                )
            else:
                inputs.str(
                    "model",
                    label="Model",
                    default=default_model,
                    description="The Gemini model to use (e.g., gemini-3-pro-preview)",
                )

            inputs.enum(
                "thinking_level",
                values=["low", "high"],
                default="high",
                label="Thinking Level",
                description="Reasoning depth: 'low' minimizes latency/cost, 'high' maximizes reasoning (Gemini 3.0 only)",
            )

        inputs.int(
            "max_tokens",
            label="Max output tokens",
            default=65536,
            description=(
                "The maximum number of output tokens (64K). Gemini 3.0 supports full 64K token output."
            ),
            view=types.FieldView(),
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        question = ctx.params.get("query_text", None)
        answer = query_gemini_vision(ctx)
        return {"question": question, "answer": answer}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("question", label="Question")
        outputs.str("answer", label="Answer")
        header = "Gemini: Chat with your images"
        return types.Property(outputs, view=types.View(label=header))

class TextToImage(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="text_to_image",
            label="Gemini: Generate Image from Text",
            dynamic=True,
        )
        _config.icon = "/assets/text_image.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Text to Image",
                icon="/assets/text_image.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Text to Image",
            description="Generate an image from a text prompt",
        )

        if not allows_gemini_models(ctx):
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs)

        inputs.str("prompt", label="Image prompt", required=True)

        inputs.enum(
            "model",
            values=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
            default="gemini-3-pro-image-preview",
            label="Image Model",
            description="gemini-3-pro-image-preview is Nano Banana Pro (better quality, supports 2K/4K)",
        )

        has_images = len(ctx.dataset) > 0
        if has_images:
            inputs.bool(
                "use_dataset_size",
                default=False,
                label="Use dataset image aspect ratio",
                description="Use the aspect ratio from a random image in the dataset",
            )

        inputs.enum(
            "aspect_ratio",
            values=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            default="1:1",
            label="Aspect Ratio" if not has_images else "Custom Aspect Ratio",
            description=None if not has_images else "Only used if 'Use dataset image aspect ratio' is unchecked",
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        prompt = ctx.params.get("prompt")
        use_dataset = ctx.params.get("use_dataset_size", False)
        model = ctx.params.get("model", "gemini-3-pro-image-preview")

        try:
            if use_dataset and len(ctx.dataset) > 0:
                sample = ctx.dataset.first()
                img = Image.open(sample.filepath)
                width, height = img.size
                aspect_ratio = get_closest_aspect_ratio(width, height)
            else:
                aspect_ratio = ctx.params.get("aspect_ratio", "1:1")

            api_key = ctx.secrets.get("GEMINI_API_KEY")
            image_data = generate_image(prompt, api_key, aspect_ratio, model)
            filepath = save_image_to_dataset(ctx.dataset, image_data, prompt, "text_to_image")
            return {"prompt": prompt, "filepath": filepath, "status": "success", "model": model}
        except Exception as e:
            return {"prompt": prompt, "status": "error", "error": str(e)}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("prompt", label="Prompt")
        outputs.str("status", label="Status")
        outputs.str("filepath", label="Generated Image Path")
        outputs.str("error", label="Error Details")
        return types.Property(outputs, view=types.View(label="Image Generation Result"))


class ImageEditing(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="image_editing",
            label="Gemini: Edit Image with Text",
            dynamic=True,
        )
        _config.icon = "/assets/banana.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Edit Image",
                icon="/assets/banana.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Image Editing",
            description="Edit selected image with text instructions",
        )

        if not allows_gemini_models(ctx):
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs)

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label="You must select exactly one image to edit"
                ),
            )
        elif num_selected > 1:
            inputs.str(
                "multiple_samples_warning",
                view=types.Warning(
                    label=f"Please select only one image. You have {num_selected} selected."
                ),
            )
        else:
            inputs.str("prompt", label="Edit instruction", required=True)

            inputs.enum(
                "model",
                values=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
                default="gemini-3-pro-image-preview",
                label="Image Model",
                description="gemini-3-pro-image-preview is Nano Banana Pro (better quality, supports 2K/4K)",
            )

            inputs.bool(
                "use_original_size",
                default=True,
                label="Use original image aspect ratio",
                description="Keep the same aspect ratio as the input image",
            )
            inputs.enum(
                "aspect_ratio",
                values=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                default="1:1",
                label="Custom Aspect Ratio",
                description="Only used if 'Use original image aspect ratio' is unchecked",
            )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        if len(ctx.selected) != 1:
            return {"status": "error", "error": "Please select exactly one image"}

        sample_id = ctx.selected[0]
        filepath = ctx.dataset[sample_id].filepath
        prompt = ctx.params.get("prompt")
        use_original = ctx.params.get("use_original_size", True)
        model = ctx.params.get("model", "gemini-3-pro-image-preview")

        try:
            if use_original:
                img = Image.open(filepath)
                width, height = img.size
                aspect_ratio = get_closest_aspect_ratio(width, height)
            else:
                aspect_ratio = ctx.params.get("aspect_ratio", "1:1")

            api_key = ctx.secrets.get("GEMINI_API_KEY")
            image_data = edit_image(filepath, prompt, api_key, aspect_ratio, model)
            new_filepath = save_image_to_dataset(ctx.dataset, image_data, prompt, "image_editing")
            return {"prompt": prompt, "filepath": new_filepath, "status": "success", "model": model}
        except Exception as e:
            return {"prompt": prompt, "status": "error", "error": str(e)}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("prompt", label="Edit Instruction")
        outputs.str("status", label="Status")
        outputs.str("filepath", label="Edited Image Path")
        outputs.str("error", label="Error Details")
        return types.Property(outputs, view=types.View(label="Image Editing Result"))


class MultiImageComposition(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="multi_image_composition",
            label="Gemini: Compose Multiple Images",
            dynamic=True,
        )
        _config.icon = "/assets/multiple_image.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Compose Images",
                icon="/assets/multiple_image.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Multi-Image Composition",
            description="Compose a new image from multiple selected images",
        )

        if not allows_gemini_models(ctx):
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs)

        num_selected = len(ctx.selected)
        if num_selected < 2:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label="You must select at least 2 images to compose"
                ),
            )
        else:
            if num_selected > 3:
                inputs.str(
                    "many_samples_warning",
                    view=types.Warning(
                        label=f"You have {num_selected} images selected. Only the first 3 will be used."
                    ),
                )

            inputs.str("prompt", label="Composition instruction", required=True)

            inputs.enum(
                "model",
                values=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
                default="gemini-3-pro-image-preview",
                label="Image Model",
                description="gemini-3-pro-image-preview is Nano Banana Pro (better quality, supports 2K/4K)",
            )

            inputs.bool(
                "use_original_size",
                default=True,
                label="Use first image aspect ratio",
                description="Keep the same aspect ratio as the first selected image",
            )
            inputs.enum(
                "aspect_ratio",
                values=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                default="1:1",
                label="Custom Aspect Ratio",
                description="Only used if 'Use first image aspect ratio' is unchecked",
            )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        if len(ctx.selected) < 2:
            return {"status": "error", "error": "Please select at least 2 images"}

        image_paths = [ctx.dataset[sample_id].filepath for sample_id in ctx.selected]
        prompt = ctx.params.get("prompt")
        use_original = ctx.params.get("use_original_size", True)
        model = ctx.params.get("model", "gemini-3-pro-image-preview")

        try:
            if use_original:
                img = Image.open(image_paths[0])
                width, height = img.size
                aspect_ratio = get_closest_aspect_ratio(width, height)
            else:
                aspect_ratio = ctx.params.get("aspect_ratio", "1:1")

            api_key = ctx.secrets.get("GEMINI_API_KEY")
            image_data = compose_images(image_paths, prompt, api_key, aspect_ratio, model)
            new_filepath = save_image_to_dataset(ctx.dataset, image_data, prompt, "multi_image_composition")
            return {"prompt": prompt, "filepath": new_filepath, "status": "success", "images_used": min(len(image_paths), 3), "model": model}
        except Exception as e:
            return {"prompt": prompt, "status": "error", "error": str(e)}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("prompt", label="Composition Instruction")
        outputs.str("status", label="Status")
        outputs.str("filepath", label="Composed Image Path")
        outputs.int("images_used", label="Images Used")
        outputs.str("error", label="Error Details")
        return types.Property(outputs, view=types.View(label="Image Composition Result"))


class VideoUnderstanding(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="video_understanding",
            label="Gemini: Analyze Video",
            dynamic=True,
        )
        _config.icon = "/assets/icon_video.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Analyze Video",
                icon="/assets/icon_video.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Video Understanding",
            description="Analyze video content using Gemini Vision",
        )

        if not allows_gemini_models(ctx):
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs)

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label="You must select exactly one video to analyze"
                ),
            )
        elif num_selected > 1:
            inputs.str(
                "multiple_samples_warning",
                view=types.Warning(
                    label=f"Please select only one video. You have {num_selected} selected."
                ),
            )
        else:
            # Check if selected sample is a video and within size limit
            sample_id = ctx.selected[0]
            filepath = ctx.dataset[sample_id].filepath

            # Check file size
            try:
                video_size_mb = get_video_size_mb(filepath)
                if video_size_mb > 20:
                    inputs.str(
                        "size_warning",
                        view=types.Error(
                            label=f"Video is too large ({video_size_mb:.1f}MB). Maximum size is 20MB for inline video analysis."
                        ),
                    )
                    return types.Property(inputs, view=form_view)
            except Exception as e:
                inputs.str(
                    "error_warning",
                    view=types.Error(
                        label=f"Error checking video file: {str(e)}"
                    ),
                )
                return types.Property(inputs, view=form_view)

            inputs.enum(
                "task_type",
                values=["describe", "segment", "extract", "question"],
                default="describe",
                label="Analysis Type",
                description="Choose the type of video analysis to perform",
            )

            inputs.str(
                "prompt",
                label="Analysis Prompt",
                required=True,
                description="Describe what you want to analyze in the video. For questions, include specific timestamps if needed (e.g., 'What happens at 0:30?')",
            )

            api_key = ctx.secrets.get("GEMINI_API_KEY")
            model_choices = list_gemini_models(api_key) if api_key else []
            default_model = "gemini-3-pro-preview"
            if model_choices:
                inputs.enum(
                    "model",
                    values=model_choices,
                    default=default_model if default_model in model_choices else model_choices[0],
                    label="Model",
                    description="Select a Gemini model",
                )
            else:
                inputs.str(
                    "model",
                    label="Model",
                    default=default_model,
                    description="The Gemini model to use (e.g., gemini-3-pro-preview)",
                )

            inputs.enum(
                "thinking_level",
                values=["low", "high"],
                default="high",
                label="Thinking Level",
                description="Reasoning depth: 'low' minimizes latency/cost, 'high' maximizes reasoning (Gemini 3.0 only)",
            )

            inputs.enum(
                "media_resolution",
                values=["low", "medium", "high"],
                default="high",
                label="Media Resolution",
                description="Video frame resolution: 'high' (1,120 tokens/frame) for analysis, 'medium' (560) for PDFs, 'low' (70) for efficiency",
            )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        if len(ctx.selected) != 1:
            return {"status": "error", "error": "Please select exactly one video"}

        sample_id = ctx.selected[0]
        filepath = ctx.dataset[sample_id].filepath
        prompt = ctx.params.get("prompt")
        task_type = ctx.params.get("task_type", "describe")
        model = ctx.params.get("model", "gemini-3-pro-preview")
        thinking_level = ctx.params.get("thinking_level", "high")
        media_resolution = ctx.params.get("media_resolution", "high")

        try:
            # Check video size
            video_size_mb = get_video_size_mb(filepath)
            if video_size_mb > 20:
                return {
                    "status": "error",
                    "error": f"Video is too large ({video_size_mb:.1f}MB). Maximum size is 20MB.",
                    "prompt": prompt,
                    "task_type": task_type
                }

            # Analyze video
            api_key = ctx.secrets.get("GEMINI_API_KEY")
            result = analyze_video(filepath, prompt, api_key, task_type, model, thinking_level, media_resolution)

            # Store result in sample metadata
            sample = ctx.dataset[sample_id]

            analysis_entry = {
                "prompt": prompt,
                "task_type": task_type,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

            # Append to existing analysis or create new list
            if sample.has_field("video_analysis") and sample["video_analysis"] is not None:
                current_analysis = sample["video_analysis"]
                if isinstance(current_analysis, list):
                    current_analysis.append(analysis_entry)
                else:
                    current_analysis = [analysis_entry]
                sample["video_analysis"] = current_analysis
            else:
                sample["video_analysis"] = [analysis_entry]

            sample.save()

            return {
                "prompt": prompt,
                "task_type": task_type,
                "result": result,
                "status": "success",
                "video_size_mb": f"{video_size_mb:.2f}"
            }
        except Exception as e:
            return {
                "prompt": prompt,
                "task_type": task_type,
                "status": "error",
                "error": str(e)
            }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("prompt", label="Analysis Prompt")
        outputs.str("task_type", label="Analysis Type")
        outputs.str("status", label="Status")
        outputs.str("result", label="Analysis Result")
        outputs.str("video_size_mb", label="Video Size (MB)")
        outputs.str("error", label="Error Details")
        return types.Property(outputs, view=types.View(label="Video Analysis Result"))


def register(plugin):
    plugin.register(QueryGeminiVision)
    plugin.register(TextToImage)
    plugin.register(ImageEditing)
    plugin.register(MultiImageComposition)
    plugin.register(VideoUnderstanding)

def download_model(model_name, model_path):
    """Prepare remote HTTP model; create a marker file at model_path."""
    return True

def load_model(model_name, model_path, **kwargs):
    from .zoo import GeminiRemoteModel
    return GeminiRemoteModel(config=kwargs)

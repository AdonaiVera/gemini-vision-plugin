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


def allows_gemini_models():
    """Returns whether the current environment allows Gemini models."""
    return "GEMINI_API_KEY" in os.environ


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
        return sorted(set(models))
    except Exception:
        return []


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


def generate_image(prompt, aspect_ratio="1:1"):
    """Generate image from text prompt using Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseModalities": ["Image"],
            "imageConfig": {"aspectRatio": aspect_ratio}
        }
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
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


def edit_image(image_path, prompt, aspect_ratio="1:1"):
    """Edit image using text prompt."""
    api_key = os.environ.get("GEMINI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    base64_image = encode_image(image_path)
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

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
            "responseModalities": ["Image"],
            "imageConfig": {"aspectRatio": aspect_ratio}
        }
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
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


def compose_images(image_paths, prompt, aspect_ratio="1:1"):
    """Compose multiple images with text prompt."""
    api_key = os.environ.get("GEMINI_API_KEY")
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

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["Image"],
            "imageConfig": {"aspectRatio": aspect_ratio}
        }
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
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


def query_gemini_vision(ctx):
    """Queries a Google Gemini Vision model (multimodal)."""
    dataset = ctx.dataset
    sample_ids = ctx.selected
    query_text = ctx.params.get("query_text", None)
    max_tokens = ctx.params.get("max_tokens", 300)
    model_name = ctx.params.get("model", "gemini-2.5-flash")

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

    api_key = os.environ.get("GEMINI_API_KEY")
    headers = {"Content-Type": "application/json"}

    headers["x-goog-api-key"] = api_key
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent",
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

        gemini_flag = allows_gemini_models()
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
            api_key = os.environ.get("GEMINI_API_KEY")
            model_choices = list_gemini_models(api_key) if api_key else []
            default_model = "gemini-2.5-flash"
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
                    description="The Gemini model to use (e.g., gemini-2.5-flash)",
                )

        inputs.int(
            "max_tokens",
            label="Max output tokens",
            default=2048,
            description=(
                "The maximum number of tokens to generate. Higher values may increase cost."
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
        _config.icon = "/assets/banana.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Text to Image",
                icon="/assets/banana.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Text to Image",
            description="Generate an image from a text prompt",
        )

        if not allows_gemini_models():
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs)

        inputs.str("prompt", label="Image prompt", required=True)

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

        try:
            if use_dataset and len(ctx.dataset) > 0:
                sample = ctx.dataset.first()
                img = Image.open(sample.filepath)
                width, height = img.size
                aspect_ratio = get_closest_aspect_ratio(width, height)
            else:
                aspect_ratio = ctx.params.get("aspect_ratio", "1:1")

            image_data = generate_image(prompt, aspect_ratio)
            filepath = save_image_to_dataset(ctx.dataset, image_data, prompt, "text_to_image")
            return {"prompt": prompt, "filepath": filepath, "status": "success"}
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

        if not allows_gemini_models():
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

        try:
            if use_original:
                img = Image.open(filepath)
                width, height = img.size
                aspect_ratio = get_closest_aspect_ratio(width, height)
            else:
                aspect_ratio = ctx.params.get("aspect_ratio", "1:1")

            image_data = edit_image(filepath, prompt, aspect_ratio)
            new_filepath = save_image_to_dataset(ctx.dataset, image_data, prompt, "image_editing")
            return {"prompt": prompt, "filepath": new_filepath, "status": "success"}
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
        _config.icon = "/assets/banana.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Compose Images",
                icon="/assets/banana.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Multi-Image Composition",
            description="Compose a new image from multiple selected images",
        )

        if not allows_gemini_models():
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

        try:
            if use_original:
                img = Image.open(image_paths[0])
                width, height = img.size
                aspect_ratio = get_closest_aspect_ratio(width, height)
            else:
                aspect_ratio = ctx.params.get("aspect_ratio", "1:1")

            image_data = compose_images(image_paths, prompt, aspect_ratio)
            new_filepath = save_image_to_dataset(ctx.dataset, image_data, prompt, "multi_image_composition")
            return {"prompt": prompt, "filepath": new_filepath, "status": "success", "images_used": min(len(image_paths), 3)}
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


def register(plugin):
    plugin.register(QueryGeminiVision)
    plugin.register(TextToImage)
    plugin.register(ImageEditing)
    plugin.register(MultiImageComposition)

def download_model(model_name, model_path):
    """Prepare remote HTTP model; create a marker file at model_path."""
    return True

def load_model(model_name, model_path, **kwargs):
    from .zoo import GeminiRemoteModel
    return GeminiRemoteModel(config=kwargs)

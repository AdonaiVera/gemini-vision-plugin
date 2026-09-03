"""Gemini Vision plugin.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import json
import os
import re
import time

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo

from . import gemini_media as gm
from . import gemini_video as gv
from . import gemini_video_gen as gvg

import base64
import requests
from PIL import Image


def allows_gemini_models(ctx):
    """Returns whether the current environment allows Gemini models."""
    return "GEMINI_API_KEY" in ctx.secrets.keys()


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


REQUEST_TIMEOUT = 300

_MODEL_CACHE = {}
_MODEL_CACHE_TTL = 300


def list_gemini_models(api_key):
    """Returns a list of available Gemini model ids that support generateContent."""
    manual_models = ["gemini-3.1-pro-preview"]

    cached = _MODEL_CACHE.get(api_key)
    if cached is not None and time.time() - cached[0] < _MODEL_CACHE_TTL:
        return cached[1]

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
            match = re.match(r"gemini-(\d+)(?:\.(\d+))?", model)
            if match is None:
                return (1, 0, 0, model)

            return (0, -int(match.group(1)), -int(match.group(2) or 0), model)

        models = sorted(all_models, key=sort_key)
        _MODEL_CACHE[api_key] = (time.time(), models)
        return models
    except Exception:
        return manual_models


VIDEO_TASK_ORDER = [
    (name, gv.VIDEO_TASKS[name])
    for name in (
        "describe",
        "state_change",
        "anomaly",
        "needle",
        "count",
        "frame_extract",
        "chapters",
        "cause_effect",
        "qa",
        "transcript",
    )
]

_DEFAULT_PROMPTS = {
    task["default_prompt"] for _, task in VIDEO_TASK_ORDER if task["default_prompt"]
}


def _is_stale_prompt(prompt, task):
    """Returns whether ``prompt`` is a different task's leftover default."""
    if not task["prompt_required"]:
        return False

    prompt = (prompt or "").strip()
    return bool(prompt) and prompt in _DEFAULT_PROMPTS - {task["default_prompt"]}

_PROMPT_HELP = {
    "describe": "Optional — steer the description toward what you care about",
    "state_change": "Optional — narrow the search, e.g. 'only the robot arm'",
    "anomaly": "Optional — describe what 'normal' looks like for this footage",
    "needle": "Required — describe exactly what to find, e.g. 'a red forklift'",
    "count": "Required — what to count, e.g. 'how many times the door opens'",
    "frame_extract": "Optional — e.g. 'one sharp frame per person who appears'",
    "chapters": "Optional — e.g. 'chapter per surgical step'",
    "cause_effect": "Required — e.g. 'why did the stack of boxes fall over?'",
    "qa": "Required — your question about the video",
    "transcript": "Optional — e.g. 'only transcribe the instructor'",
}


def list_video_models(api_key):
    """Returns the available Gemini models that support agentic video.

    Args:
        api_key: a Gemini API key

    Returns:
        a list of model ids
    """
    available = set(list_gemini_models(api_key))
    models = [m for m in gv.AGENTIC_VIDEO_MODELS if m in available]
    return models or list(gv.AGENTIC_VIDEO_MODELS)


def _resolve_video_targets(ctx):
    """Resolves which videos to analyze from the App's selection.

    Args:
        ctx: the operator :class:`fiftyone.operators.executor.ExecutionContext`

    Returns:
        a ``(targets, error)`` tuple, where ``targets`` is a list of
        ``(sample_id, sample)`` and ``error`` is a message or ``None``
    """
    if ctx.dataset is None:
        return [], "No dataset is loaded"

    selected = list(ctx.selected)
    if not selected:
        return [], "Select one or more videos to analyze."

    max_videos = int(ctx.params.get("max_videos") or 5)
    if len(selected) > max_videos:
        return [], (
            f"{len(selected)} videos selected, above the limit of "
            f"{max_videos}. Select fewer, or raise Max videos — Gemini bills "
            "per video."
        )

    targets = []
    for sample_id in selected:
        sample = ctx.dataset[sample_id]
        if sample.media_type != "video":
            return [], (
                f"'{os.path.basename(sample.filepath)}' is a "
                f"{sample.media_type}, not a video."
            )

        targets.append((sample_id, sample))

    return targets, None


def _render_result(result, filename=None, field=None):
    """Renders one analysis result as Markdown for the operator output.

    Args:
        result: a :func:`gemini_video.run_interaction` result
        filename (None): the video's basename, when more than one was analyzed
        field (None): the sample field the answer was saved to
    """
    _, task = gv.resolve_task(result["task"])

    parts = []
    if filename:
        parts.append(f"**{filename}**")

    data = result.get("data")
    if data is None or not task["events_key"]:
        parts.append(result["text"])
    else:
        if data.get("summary"):
            parts.append(str(data["summary"]))
        if data.get("total") is not None:
            parts.append(f"**Total: {data['total']}**")
        if data.get("counting_rule"):
            parts.append(f"_Counting rule: {data['counting_rule']}_")
        if data.get("found") is False:
            parts.append("_Target not found in this video._")
        parts.append(gv.format_events_markdown(data.get(task["events_key"])))

    usage = result["usage"]
    meta = [
        result["model"],
        f"{result['processing_calls']} agentic video lookups",
        f"{usage.get('total_tokens')} tokens",
    ]
    if field:
        meta.append(f"saved to `{field}`")
    parts.append("_" + " · ".join(meta) + "_")

    return "\n\n".join(p for p in parts if p)


def save_image_to_dataset(dataset, base64_data, prompt, operation_type="generated"):
    """Saves a generated image beside the dataset's media and adds a sample."""
    try:
        img_data = base64.b64decode(base64_data)

        filepath = gm.output_path(
            gm.media_root(dataset), operation_type, "png"
        )
        gm.write_media(img_data, filepath)

        sample = fo.Sample(filepath=filepath)
        sample["prompt"] = prompt
        sample["generation_type"] = operation_type
        dataset.add_sample(sample)

        return filepath
    except Exception as e:
        raise ValueError(f"Failed to save image: {str(e)}")


def generate_image(prompt, api_key, aspect_ratio="1:1", model="gemini-3-pro-image"):
    """Generate image from text prompt using Gemini."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

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
        timeout=REQUEST_TIMEOUT,
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


def edit_image(image_path, prompt, api_key, aspect_ratio="1:1", model="gemini-3-pro-image"):
    """Edit image using text prompt."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    base64_image = encode_image(image_path)
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

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
        timeout=REQUEST_TIMEOUT,
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


def compose_images(image_paths, prompt, api_key, aspect_ratio="1:1", model="gemini-3-pro-image"):
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
        timeout=REQUEST_TIMEOUT,
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


def run_ocr(image_path, api_key, model="gemini-3.1-pro-preview"):
    """Extract text with bounding boxes using Gemini Vision.

    Args:
        image_path: Path to image file
        api_key: Gemini API key
        model: Gemini model to use

    Returns:
        list: List of dicts with 'text' and 'bbox' keys
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    base64_image = encode_image(image_path)
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    prompt = """Extract all text from this image with bounding boxes.
For each text region, return a JSON array with objects containing:
- "text": the extracted text
- "bbox": [ymin, xmin, ymax, xmax] coordinates scaled 0-1000

Return ONLY valid JSON array, no other text. Example:
[{"text": "Hello World", "bbox": [100, 50, 150, 200]}]"""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_image,
                    }
                },
            ]
        }],
        "generationConfig": {"maxOutputTokens": 65536},
    }

    api_version = "v1beta" if model.startswith("gemini-3") else "v1"
    response = requests.post(
        f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    content = response.json()
    if "error" in content:
        err = content.get("error", {})
        raise ValueError(err.get("message") or str(err))

    try:
        text = content["candidates"][0]["content"]["parts"][0].get("text", "")
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except Exception as e:
        raise ValueError(f"Failed to parse OCR response: {str(e)}")


def gemini_bbox_to_fiftyone(bbox):
    """Convert Gemini bbox [ymin, xmin, ymax, xmax] (0-1000) to FiftyOne [x, y, w, h] (0-1).

    Args:
        bbox: List of [ymin, xmin, ymax, xmax] in 0-1000 scale

    Returns:
        list: [x, y, width, height] normalized to 0-1
    """
    ymin, xmin, ymax, xmax = bbox
    x = xmin / 1000.0
    y = ymin / 1000.0
    w = (xmax - xmin) / 1000.0
    h = (ymax - ymin) / 1000.0
    return [x, y, w, h]


def run_spatial(image_path, api_key, prompt, model="gemini-3.1-pro-preview"):
    """Detect points/keypoints using Gemini spatial understanding.

    Args:
        image_path: Path to image file
        api_key: Gemini API key
        prompt: User prompt describing what to detect
        model: Gemini model to use

    Returns:
        list: List of dicts with 'point' and 'label' keys
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    base64_image = encode_image(image_path)
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    full_prompt = f"""{prompt}

Return a JSON array with objects containing:
- "point": [y, x] coordinates scaled 0-1000
- "label": description of the point

Return ONLY valid JSON array. Example:
[{{"point": [250, 400], "label": "left eye"}}, {{"point": [250, 600], "label": "right eye"}}]"""

    payload = {
        "contents": [{
            "parts": [
                {"text": full_prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_image,
                    }
                },
            ]
        }],
        "generationConfig": {"maxOutputTokens": 65536},
    }

    api_version = "v1beta" if model.startswith("gemini-3") else "v1"
    response = requests.post(
        f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    content = response.json()
    if "error" in content:
        err = content.get("error", {})
        raise ValueError(err.get("message") or str(err))

    try:
        text = content["candidates"][0]["content"]["parts"][0].get("text", "")
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except Exception as e:
        raise ValueError(f"Failed to parse spatial response: {str(e)}")


def gemini_points_to_keypoints(points_data):
    """Convert Gemini points to FiftyOne Keypoints.

    Args:
        points_data: List of dicts with 'point' [y, x] in 0-1000 scale

    Returns:
        list: List of fo.Keypoint objects, or empty list if invalid
    """
    if not points_data:
        return []

    keypoints = []
    for item in points_data:
        point = item.get("point", [])
        label = item.get("label", "point")
        if point and len(point) == 2:
            y, x = point
            keypoints.append(
                fo.Keypoint(
                    label=label,
                    points=[(x / 1000.0, y / 1000.0)],
                )
            )

    return keypoints


def query_gemini_vision(ctx):
    """Queries a Google Gemini Vision model (multimodal)."""
    dataset = ctx.dataset
    sample_ids = ctx.selected
    query_text = ctx.params.get("query_text", None)
    max_tokens = ctx.params.get("max_tokens", 65536)
    model_name = ctx.params.get("model", "gemini-3.1-pro-preview")
    thinking_level = ctx.params.get("thinking_level", "high")

    parts = []
    if query_text:
        parts.append({"text": query_text})
    for sample_id in sample_ids:
        filepath = gm.localize(dataset[sample_id])
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
        timeout=REQUEST_TIMEOUT,
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
            label="Gemini: Vision Tasks",
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
            label="Gemini Vision",
            description="Run vision tasks on selected images",
        )

        if not allows_gemini_models(ctx):
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs, view=form_view)

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(label="You must select a sample to use this operator"),
            )
            return types.Property(inputs, view=form_view)

        if num_selected > 10:
            inputs.str(
                "many_samples_warning",
                view=types.Warning(
                    label=f"You have {num_selected} samples selected. Gemini may charge per image.",
                ),
            )

        inputs.enum(
            "task",
            values=["chat", "ocr", "spatial"],
            default="chat",
            label="Task",
            description="Chat: ask questions | OCR: extract text | Spatial: detect points/keypoints",
        )

        task = ctx.params.get("task", "chat")

        api_key = ctx.secrets.get("GEMINI_API_KEY")
        model_choices = list_gemini_models(api_key) if api_key else []
        default_model = "gemini-3.1-pro-preview"

        if model_choices:
            inputs.enum(
                "model",
                values=model_choices,
                default=default_model if default_model in model_choices else model_choices[0],
                label="Model",
            )
        else:
            inputs.str("model", label="Model", default=default_model)

        if task == "chat":
            inputs.str("query_text", label="Query", required=True)
            inputs.int("max_tokens", label="Max Tokens", default=65536)
        elif task == "ocr":
            inputs.str(
                "label_field",
                label="Label Field",
                default="ocr_detections",
                description="Field name to store OCR detections",
            )
        elif task == "spatial":
            inputs.str(
                "query_text",
                label="Prompt",
                required=True,
                description="e.g. 'Point to all human body keypoints' or 'Detect the trajectory path'",
            )
            inputs.str(
                "label_field",
                label="Label Field",
                default="spatial_keypoints",
                description="Field name to store keypoints",
            )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        task = ctx.params.get("task", "chat")

        if task == "chat":
            question = ctx.params.get("query_text", None)
            answer = query_gemini_vision(ctx)
            return {
                "status": "success",
                "task": task,
                "question": question,
                "answer": answer,
            }

        api_key = ctx.secrets.get("GEMINI_API_KEY")
        model = ctx.params.get("model", "gemini-3.1-pro-preview")
        label_field = ctx.params.get("label_field", "ocr_detections" if task == "ocr" else "spatial_keypoints")

        processed = 0
        errors = []

        for sample_id in ctx.selected:
            sample = ctx.dataset[sample_id]
            try:
                if task == "ocr":
                    ocr_results = run_ocr(gm.localize(sample), api_key, model)
                    detections = []
                    for item in ocr_results:
                        text = item.get("text", "")
                        bbox = item.get("bbox", [])
                        if bbox and len(bbox) == 4:
                            detections.append(
                                fo.Detection(
                                    label=text,
                                    bounding_box=gemini_bbox_to_fiftyone(bbox),
                                )
                            )
                    sample[label_field] = fo.Detections(detections=detections)
                elif task == "spatial":
                    prompt = ctx.params.get("query_text", "")
                    spatial_results = run_spatial(
                        gm.localize(sample), api_key, prompt, model
                    )
                    keypoints = gemini_points_to_keypoints(spatial_results)
                    sample[label_field] = fo.Keypoints(keypoints=keypoints)

                sample.save()
                processed += 1
            except Exception as e:
                errors.append(str(e))

        ctx.ops.reload_dataset()
        ctx.ops.reload_samples()

        return {
            "status": "success" if processed else "error",
            "task": task,
            "processed": processed,
            "total": len(ctx.selected),
            "label_field": label_field,
            "errors": "; ".join(errors) if errors else None,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("task", label="Task")
        outputs.str("status", label="Status")

        task = ctx.params.get("task", "chat")
        if task == "chat":
            outputs.str("question", label="Question")
            outputs.str("answer", label="Answer", view=types.MarkdownView())
        else:
            outputs.int("processed", label="Processed")
            outputs.int("total", label="Total")
            outputs.str("label_field", label="Label Field")
            outputs.str("errors", label="Errors")

        return types.Property(outputs, view=types.View(label="Gemini Vision Results"))

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
            values=["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image"],
            default="gemini-3-pro-image",
            label="Image Model",
            description="gemini-3-pro-image is Nano Banana Pro (best quality, supports 2K/4K); gemini-3.1-flash-image is the fast tier",
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
        model = ctx.params.get("model", "gemini-3-pro-image")

        try:
            if use_dataset and len(ctx.dataset) > 0:
                sample = ctx.dataset.first()
                img = Image.open(gm.localize(sample))
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
                values=["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image"],
                default="gemini-3-pro-image",
                label="Image Model",
                description="gemini-3-pro-image is Nano Banana Pro (best quality, supports 2K/4K); gemini-3.1-flash-image is the fast tier",
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
        filepath = gm.localize(ctx.dataset[sample_id])
        prompt = ctx.params.get("prompt")
        use_original = ctx.params.get("use_original_size", True)
        model = ctx.params.get("model", "gemini-3-pro-image")

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
                values=["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image"],
                default="gemini-3-pro-image",
                label="Image Model",
                description="gemini-3-pro-image is Nano Banana Pro (best quality, supports 2K/4K); gemini-3.1-flash-image is the fast tier",
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

        image_paths = [
            gm.localize(ctx.dataset[sample_id]) for sample_id in ctx.selected
        ]
        prompt = ctx.params.get("prompt")
        use_original = ctx.params.get("use_original_size", True)
        model = ctx.params.get("model", "gemini-3-pro-image")

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
            description=(
                "Agentic video understanding — state changes, anomalies, "
                "counting, needle-in-a-haystack search, keyframe extraction "
                "and causal Q&A, written back as temporal detections"
            ),
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=False,
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
            label="Agentic Video Understanding",
            description=(
                "Gemini navigates the timeline itself — searching, scanning "
                "and re-sampling only the segments your prompt needs"
            ),
        )

        if not allows_gemini_models(ctx):
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs, view=form_view)

        source = ctx.params.get("source", "selected")
        inputs.enum(
            "source",
            values=["selected", "youtube"],
            default="selected",
            label="Video source",
            description="Analyze selected samples, or a public YouTube URL",
            view=types.RadioGroup(),
        )

        if source == "youtube":
            inputs.str(
                "youtube_url",
                label="YouTube URL",
                required=True,
                description="A public YouTube video URL",
            )
        else:
            targets, error = _resolve_video_targets(ctx)
            if error is not None:
                prop = inputs.str(
                    "target_error", view=types.Warning(label=error)
                )
                prop.invalid = True
                return types.Property(inputs, view=form_view)

            oversized = [
                (os.path.basename(s.filepath), gv.get_video_size_mb(s.filepath))
                for _, s in targets
                if os.path.isfile(s.filepath)
                and gv.get_video_size_mb(s.filepath) > gv.MAX_UPLOAD_MB
            ]
            if oversized:
                names = ", ".join(f"{n} ({s:.0f}MB)" for n, s in oversized[:3])
                prop = inputs.str(
                    "size_error",
                    view=types.Error(
                        label=(
                            f"{len(oversized)} video(s) exceed the "
                            f"{gv.MAX_UPLOAD_MB}MB limit: {names}"
                        )
                    ),
                )
                prop.invalid = True
                return types.Property(inputs, view=form_view)

            large = sum(
                1
                for _, s in targets
                if os.path.isfile(s.filepath)
                and gv.get_video_size_mb(s.filepath) > gv.INLINE_LIMIT_MB
            )
            remote = sum(1 for _, s in targets if not gm.is_ready(s))
            note = f"{len(targets)} video(s) will be analyzed."
            if large:
                note += (
                    f" {large} exceed {gv.INLINE_LIMIT_MB}MB and will be "
                    "uploaded via the Files API first."
                )
            if remote:
                note += (
                    f" {remote} are not cached locally and will be downloaded "
                    "before analysis."
                )
            inputs.str("target_note", view=types.Notice(label=note))

        inputs.int(
            "max_videos",
            label="Max videos",
            default=5,
            description="A guard against a runaway bill — Gemini charges per video",
        )

        task_choices = types.Dropdown()
        for name, task in VIDEO_TASK_ORDER:
            task_choices.add_choice(
                name,
                label=f"{task['label']} — {task['theme']}",
                description=task["description"],
            )

        alias = ctx.params.get("task_type")
        if alias in gv.LEGACY_TASK_ALIASES:
            ctx.params["task_type"] = gv.LEGACY_TASK_ALIASES[alias]

        inputs.enum(
            "task_type",
            values=task_choices.values(),
            default="describe",
            label="Task",
            view=task_choices,
        )

        task_type = ctx.params.get("task_type", "describe")
        try:
            task_name, task = gv.resolve_task(task_type)
        except gv.GeminiVideoError:
            task_name, task = gv.resolve_task("describe")

        if _is_stale_prompt(ctx.params.get("prompt"), task):
            inputs.str(
                "stale_prompt_warning",
                view=types.Warning(
                    label=(
                        "That prompt is another task's default. Replace it with "
                        f"what you actually want to {task['label'].lower()}."
                    )
                ),
            )

        inputs.str(
            "prompt",
            label="Prompt",
            required=task["prompt_required"],
            default=task["default_prompt"] or None,
            description=_PROMPT_HELP.get(task_name, "What do you want to know?"),
        )

        api_key = ctx.secrets.get("GEMINI_API_KEY")
        model_choices = list_video_models(api_key) if api_key else []
        if model_choices:
            inputs.enum(
                "model",
                values=model_choices,
                default=(
                    gv.DEFAULT_VIDEO_MODEL
                    if gv.DEFAULT_VIDEO_MODEL in model_choices
                    else model_choices[0]
                ),
                label="Model",
                description="Models that support agentic video processing",
            )
        else:
            inputs.str(
                "model",
                label="Model",
                default=gv.DEFAULT_VIDEO_MODEL,
                description="A Gemini model that supports agentic video",
            )

        inputs.enum(
            "processing_mode",
            values=["agentic", "static"],
            default="agentic",
            label="Processing mode",
            description=(
                "Agentic lets the model navigate the timeline itself; static "
                "samples frames at a fixed rate"
            ),
            view=types.RadioGroup(),
        )

        if ctx.params.get("processing_mode") == "static":
            inputs.float(
                "fps",
                label="Frames per second",
                default=1.0,
                description="1.0 samples one frame per second; 0.5 one every two",
            )
            inputs.float(
                "start_offset",
                label="Start offset (seconds)",
                description="Optional — analyze only from this point",
            )
            inputs.float(
                "end_offset",
                label="End offset (seconds)",
                description="Optional — analyze only up to this point",
            )
        else:
            inputs.enum(
                "thinking_level",
                values=["low", "high"],
                default="high",
                label="Thinking level",
                description=(
                    "'low' minimizes latency and cost; 'high' maximizes "
                    "reasoning over the timeline"
                ),
            )

        if task["default_field"]:
            inputs.str(
                "label_field",
                label="Label field",
                default=task["default_field"],
                description=(
                    "Timestamped results are written here as temporal "
                    "detections, seekable in the App"
                ),
            )

        if task_name == "frame_extract":
            inputs.bool(
                "extract_frames",
                label="Cut the selected frames out as images",
                default=True,
                description=(
                    "Writes the chosen frames to disk and adds them to a "
                    "companion image dataset"
                ),
            )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        api_key = ctx.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "error": "GEMINI_API_KEY is not set"}

        task_type = ctx.params.get("task_type", "describe")
        try:
            task_name, task = gv.resolve_task(task_type)
        except gv.GeminiVideoError as e:
            return {"status": "error", "error": str(e)}

        kwargs = dict(
            api_key=api_key,
            task_type=task_name,
            model=ctx.params.get("model", gv.DEFAULT_VIDEO_MODEL),
            processing_mode=ctx.params.get("processing_mode", "agentic"),
            fps=ctx.params.get("fps"),
            start_offset=ctx.params.get("start_offset"),
            end_offset=ctx.params.get("end_offset"),
            thinking_level=ctx.params.get("thinking_level", "high"),
        )
        prompt = ctx.params.get("prompt") or ""
        if _is_stale_prompt(prompt, task):
            return {
                "status": "error",
                "task_type": task_name,
                "error": (
                    f"The prompt is another task's default ({prompt!r}). Replace "
                    f"it with what you want the '{task_name}' task to look for."
                ),
            }

        label_field = ctx.params.get("label_field") or task["default_field"]

        if ctx.params.get("source") == "youtube":
            url = (ctx.params.get("youtube_url") or "").strip()
            if not gv.is_youtube_url(url):
                return {"status": "error", "error": f"Not a YouTube URL: {url}"}

            try:
                result = gv.run_interaction(url, prompt, **kwargs)
            except Exception as e:
                return {"status": "error", "error": str(e), "task_type": task_name}

            return {
                "status": "success",
                "task_type": task_name,
                "prompt": result["prompt"],
                "result": _render_result(result),
                "result_field": None,
                "processed": 1,
                "total": 1,
                "processing_calls": result["processing_calls"],
                "total_tokens": result["usage"].get("total_tokens"),
            }

        targets, error = _resolve_video_targets(ctx)
        if error is not None:
            return {"status": "error", "error": error, "task_type": task_name}

        rendered = []
        errors = []
        processed = 0
        total_events = 0
        total_tokens = 0
        processing_calls = 0
        frames_summary = None
        written = {}

        for sample_id, sample in targets:
            filepath = sample.filepath
            try:
                source = gm.localize(sample)
                result = gv.run_interaction(source, prompt, **kwargs)
                written = gv.save_result(sample, result, label_field=label_field)

                processed += 1
                total_events += written["num_events"]
                processing_calls += result["processing_calls"]
                total_tokens += result["usage"].get("total_tokens") or 0

                if task_name == "frame_extract" and ctx.params.get(
                    "extract_frames", True
                ):
                    events = (result.get("data") or {}).get("events") or []
                    frames_summary = gv.extract_frames(sample, events)

                rendered.append(
                    _render_result(
                        result,
                        filename=(
                            os.path.basename(filepath)
                            if len(targets) > 1
                            else None
                        ),
                        field=written.get("summary_field"),
                    )
                )
            except Exception as e:
                errors.append(f"{os.path.basename(filepath)}: {e}")

        body = "\n\n---\n\n".join(rendered)
        if frames_summary and frames_summary["num_frames"]:
            body += (
                f"\n\n---\n\n**Extracted {frames_summary['num_frames']} frames** "
                f"into dataset `{frames_summary['dataset']}` "
                f"(`{frames_summary['output_dir']}`)."
            )

        return {
            "status": "success" if processed else "error",
            "task_type": task_name,
            "prompt": prompt or task["default_prompt"],
            "result": body,
            "processed": processed,
            "total": len(targets),
            "num_events": total_events,
            "result_field": written.get("summary_field"),
            "label_field": label_field if total_events else None,
            "processing_calls": processing_calls,
            "total_tokens": total_tokens,
            "frames_dataset": (
                frames_summary["dataset"] if frames_summary else None
            ),
            "error": "\n".join(errors) if errors else None,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("task_type", label="Task")
        outputs.str("status", label="Status")
        outputs.str("prompt", label="Prompt")
        outputs.str("result", label="Result", view=types.MarkdownView())
        outputs.int("processed", label="Videos analyzed")
        outputs.int("total", label="Videos requested")
        outputs.int("num_events", label="Timestamped events")
        outputs.str("result_field", label="Saved to field")
        outputs.str("label_field", label="Temporal detections field")
        outputs.int("processing_calls", label="Agentic video lookups")
        outputs.int("total_tokens", label="Total tokens")
        outputs.str("frames_dataset", label="Extracted frames dataset")
        outputs.str("error", label="Error details")
        return types.Property(
            outputs, view=types.View(label="Video Analysis Result")
        )


VIDEO_GEN_TASK_ORDER = [
    (name, gvg.GENERATION_TASKS[name])
    for name in (
        "text_to_video",
        "image_to_video",
        "first_last_frame",
        "reference_to_video",
        "edit",
        "extend",
    )
]

_GEN_SELECTION_RULES = {
    "none": (0, 0),
    "one_image": (1, 1),
    "two_images": (2, 2),
    "images": (1, 3),
    "generated_video": (1, 1),
}

_GEN_SELECTION_ASKS = {
    "one_image": "exactly one image",
    "two_images": "exactly two images",
    "images": "one to three images",
    "generated_video": "exactly one generated clip",
}


def _resolve_generation_sources(ctx, spec):
    """Validates the selection against a generation task's requirements.

    Returns:
        an ``(image_paths, interaction_id, error)`` tuple
    """
    needs = spec["needs"]
    low, high = _GEN_SELECTION_RULES[needs]
    selected = list(ctx.selected)

    if needs == "none":
        return [], None, None

    asks = _GEN_SELECTION_ASKS[needs]

    if not selected:
        return [], None, f"Select {asks} to use with '{spec['label']}'."

    if len(selected) < low or len(selected) > high:
        return [], None, (
            f"'{spec['label']}' takes {asks} — you have {len(selected)} "
            "selected. Narrow the selection to run it."
        )

    try:
        samples = [ctx.dataset[sample_id] for sample_id in selected]
    except KeyError:
        return [], None, (
            "The selected sample is not in this dataset. Generated clips are "
            "added to a companion video dataset — switch to it before editing "
            "or extending one."
        )

    if needs == "generated_video":
        sample = samples[0]
        interaction_id = (
            sample["gemini_interaction_id"]
            if sample.has_field("gemini_interaction_id")
            else None
        )
        if not interaction_id:
            return [], None, (
                "That sample was not generated by this plugin, so there is no "
                "clip to edit. Editing an arbitrary uploaded video is not "
                "supported here."
            )

        return [], interaction_id, None

    paths = []
    for sample in samples:
        if sample.media_type != "image":
            return [], None, (
                f"'{os.path.basename(sample.filepath)}' is a "
                f"{sample.media_type}, and '{spec['label']}' takes {asks}."
            )

        paths.append(gm.localize(sample))

    return paths, None, None


class VideoGeneration(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="video_generation",
            label="Gemini: Generate Video",
            description=(
                "Generate video with audio using Gemini Omni — from text, from "
                "an image, or by editing and extending a clip you generated"
            ),
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=False,
        )
        _config.icon = "/assets/icon_video.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Generate Video",
                icon="/assets/icon_video.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Gemini Video Generation",
            description="Omni generates video with a synthesized audio track",
        )

        if not allows_gemini_models(ctx):
            inputs.message(
                "no_gemini_key",
                label="No Gemini API Key. Please set GEMINI_API_KEY in your environment.",
            )
            return types.Property(inputs, view=form_view)

        task_choices = types.Dropdown()
        for name, spec in VIDEO_GEN_TASK_ORDER:
            task_choices.add_choice(
                name, label=spec["label"], description=spec["description"]
            )

        inputs.enum(
            "task",
            values=task_choices.values(),
            default="text_to_video",
            label="Task",
            view=task_choices,
        )

        task_name = ctx.params.get("task", "text_to_video")
        try:
            task_name, spec = gvg.resolve_task(task_name)
        except gvg.GeminiVideoGenError:
            task_name, spec = gvg.resolve_task("text_to_video")

        image_paths, interaction_id, error = _resolve_generation_sources(ctx, spec)
        if error is not None:
            prop = inputs.str("selection_error", view=types.Warning(label=error))
            prop.invalid = True
        elif spec["needs"] != "none":
            selected = list(ctx.selected)
            names = ", ".join(
                os.path.basename(ctx.dataset[i].filepath) for i in selected
            )
            inputs.str(
                "selection_note",
                view=types.Notice(label=f"Using {len(selected)} selected: {names}"),
            )

        inputs.str(
            "prompt",
            label="Prompt",
            required=True,
            description=spec["prompt_help"],
        )

        inputs.enum(
            "model",
            values=gvg.VIDEO_GEN_MODELS,
            default=gvg.DEFAULT_VIDEO_GEN_MODEL,
            label="Model",
            description="Gemini Omni generates video with audio",
        )

        inputs.enum(
            "resolution",
            values=gvg.RESOLUTIONS,
            default="720p",
            label="Resolution",
            description=(
                "360p renders a draft at roughly a third of 720p's cost; "
                "1080p and 4k are upscaled"
            ),
        )

        inputs.enum(
            "aspect_ratio",
            values=gvg.ASPECT_RATIOS,
            default="16:9",
            label="Aspect ratio",
            view=types.RadioGroup(),
        )

        try:
            default_dir = gm.media_root(ctx.dataset)
            dir_error = None
        except gm.MediaError as e:
            default_dir = ""
            dir_error = str(e)

        if dir_error:
            prop = inputs.str("output_dir_error", view=types.Warning(label=dir_error))
            prop.invalid = True

        inputs.str(
            "output_dir",
            label="Output directory",
            required=bool(dir_error),
            default=default_dir or None,
            description=(
                "Where the clip is written. Defaults beside the dataset's "
                "media so everyone can see it; set a cloud location such as "
                "gs://your-bucket/generated when the media folder is read-only"
            ),
        )

        inputs.str(
            "notice",
            view=types.Notice(
                label=(
                    "Generation takes tens of seconds to a few minutes and "
                    "adds a new video sample to your dataset."
                )
            ),
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        api_key = ctx.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "error": "GEMINI_API_KEY is not set"}

        try:
            task_name, spec = gvg.resolve_task(
                ctx.params.get("task", "text_to_video")
            )
        except gvg.GeminiVideoGenError as e:
            return {"status": "error", "error": str(e)}

        image_paths, interaction_id, error = _resolve_generation_sources(ctx, spec)
        if error is not None:
            return {"status": "error", "task": task_name, "error": error}

        selected = list(ctx.selected)

        try:
            result = gvg.generate_video(
                ctx.params.get("prompt"),
                api_key,
                task=task_name,
                model=ctx.params.get("model", gvg.DEFAULT_VIDEO_GEN_MODEL),
                image_paths=image_paths,
                previous_interaction_id=interaction_id,
                aspect_ratio=ctx.params.get("aspect_ratio", "16:9"),
                resolution=ctx.params.get("resolution", "720p"),
            )
            saved = gvg.save_video(
                ctx.dataset,
                result,
                output_dir=ctx.params.get("output_dir") or None,
                source_ids=selected or None,
            )
        except Exception as e:
            return {"status": "error", "task": task_name, "error": str(e)}

        ctx.trigger("reload_dataset")

        return {
            "status": "success",
            "task": task_name,
            "prompt": result["prompt"],
            "filepath": saved["filepath"],
            "dataset": saved["dataset"],
            "sample_id": saved["sample_id"],
            "duration": (
                f"{saved['duration']:.1f}s" if saved["duration"] else None
            ),
            "resolution": result["resolution"],
            "total_tokens": result["usage"].get("total_tokens"),
            "interaction_id": result["interaction_id"],
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("task", label="Task")
        outputs.str("status", label="Status")
        outputs.str("prompt", label="Prompt")
        outputs.str("filepath", label="Generated video")
        outputs.str("dataset", label="Added to dataset")
        outputs.str("duration", label="Duration")
        outputs.str("resolution", label="Resolution")
        outputs.int("total_tokens", label="Total tokens")
        outputs.str("interaction_id", label="Interaction id (edit and extend)")
        outputs.str("error", label="Error details")
        return types.Property(
            outputs, view=types.View(label="Video Generation Result")
        )


def register(plugin):
    plugin.register(QueryGeminiVision)
    plugin.register(TextToImage)
    plugin.register(ImageEditing)
    plugin.register(MultiImageComposition)
    plugin.register(VideoUnderstanding)
    plugin.register(VideoGeneration)

def download_model(model_name, model_path):
    """Prepare remote HTTP model; create a marker file at model_path."""
    return True

def load_model(model_name, model_path, **kwargs):
    from .zoo import GeminiRemoteModel
    return GeminiRemoteModel(config=kwargs)

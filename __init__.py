"""Gemini Vision plugin.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo
import fiftyone.core.utils as fou

import base64
import requests


def allows_gemini_models():
    """Returns whether the current environment allows Gemini models."""
    return "GEMINI_API_KEY" in os.environ


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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

def register(plugin):
    plugin.register(QueryGeminiVision)


def download_model(model_name, model_path):
    """Prepare remote HTTP model; create a marker file at model_path."""
    return True

def load_model(model_name, model_path, **kwargs):
    from .zoo import GeminiRemoteModel
    return GeminiRemoteModel(config=kwargs)

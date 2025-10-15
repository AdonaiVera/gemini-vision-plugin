import os
import base64
import requests

import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone.core.models import Model

class GeminiRemoteModel(Model):
    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gemini-2.5-flash")
        self.max_tokens = int(config.get("max_tokens", 2048))
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiRemoteModel")
        self.api_key = api_key
        self.config = config

    @property
    def media_type(self):
        """Returns the media type for the model."""
        # TODO: add support for other media types (VIDEO, AUDIO, etc.)
        return "image"

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _infer_sample(self, filepath, prompt):
        mime_type = "image/jpeg"
        if filepath.lower().endswith(".png"):
            mime_type = "image/png"
        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": self._encode_image(filepath)}},
        ]
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent",
            headers=headers,
            json=payload,
            timeout=60,
        )
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message") or str(data["error"]))
        return data["candidates"][0]["content"]["parts"][0].get("text")

    def apply(self, sample_collection, prompt=None, prompt_field=None, label_field="gemini_output", image_field="filepath"):
        view = sample_collection.view()
        for sample in view.iter_samples(autosave=True, progress=True):
            q = prompt if prompt_field is None else getattr(sample, prompt_field, None) or prompt
            if not q:
                q = "What is in this image?"
            answer = self._infer_sample(getattr(sample, image_field), q)
            setattr(sample, label_field, answer)
            sample.save()





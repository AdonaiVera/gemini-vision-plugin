import os
import base64
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone.core.models import Model
from PIL import Image
import numpy as np

class GeminiRemoteModel(Model):
    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gemini-2.5-flash")
        self.max_tokens = int(config.get("max_tokens", 2048))
        self.prompt = config.get("prompt", "What is in this image?")
        self.max_workers = int(config.get("max_workers", 8))
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiRemoteModel")
        self.api_key = api_key
        self.config = config
        self.needs_fields = {}
        self._session = requests.Session()

    @property
    def media_type(self):
        """Returns the media type for the model."""
        # TODO: add support for other media types (VIDEO, AUDIO, etc.)
        return "image"

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_pil(self, image: Image.Image):
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

    def _post(self, parts):
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        resp = self._session.post(
            f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent",
            headers=headers,
            json=payload,
            timeout=60,
        )
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message") or str(data["error"]))
        return data["candidates"][0]["content"]["parts"][0].get("text")

    def predict(self, image, sample=None):
        """Predict text for an image. Accepts path, PIL.Image, or numpy array.

        Returns a plain string VQA response.
        """
        prompt = self.prompt
        if sample is not None and isinstance(self.needs_fields, dict):
            prompt_field = self.needs_fields.get("prompt_field")
            if prompt_field and hasattr(sample, prompt_field):
                value = getattr(sample, prompt_field)
                if value:
                    prompt = str(value)

        if isinstance(image, str):
            try:
                pil = Image.open(image)
                b64, mime_type = self._encode_pil(pil)
            except Exception:
                mime_type = "image/jpeg"
                if image.lower().endswith(".png"):
                    mime_type = "image/png"
                b64 = self._encode_image(image)
            parts = [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64}},
            ]
            return self._post(parts)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            if sample is not None and hasattr(sample, "filepath"):
                mime_type = "image/jpeg"
                if sample.filepath.lower().endswith(".png"):
                    mime_type = "image/png"
                b64 = self._encode_image(sample.filepath)
                parts = [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                ]
                return self._post(parts)
            raise ValueError("Unsupported image type for predict()")

        b64, mime_type = self._encode_pil(image)
        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}},
        ]
        return self._post(parts)

    def apply(self, sample_collection, prompt=None, prompt_field=None, label_field="gemini_output", image_field="filepath"):
        if prompt is not None:
            self.prompt = prompt
        if prompt_field is not None:
            self.needs_fields = {"prompt_field": prompt_field}
        items = []
        view = sample_collection.view()
        for sample in view.iter_samples(autosave=False, progress=False):
            items.append((sample.id, getattr(sample, image_field), sample))

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.predict, path, sample): sid for sid, path, sample in items}
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    results[sid] = fut.result()
                except Exception as e:
                    results[sid] = str(e)

        for sid, answer in results.items():
            s = sample_collection[sid]
            setattr(s, label_field, answer)
            s.save()





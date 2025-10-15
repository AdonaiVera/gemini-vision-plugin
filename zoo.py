import os
import base64
import requests
import io
from fiftyone.core.models import Model
from PIL import Image
import numpy as np

class GeminiRemoteModel(Model):
    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gemini-2.5-flash")
        self.max_tokens = int(config.get("max_tokens", 2048))
        self.prompt = config.get("prompt", "What is in this image?")
        self.prompt_echo_field = config.get("prompt_echo_field", None)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiRemoteModel")
        self.api_key = api_key
        self.config = config
        self._needs_fields = {}
        self._session = requests.Session()

    @property
    def media_type(self):
        """Returns the media type for the model."""
        # TODO: add support for other media types (VIDEO, AUDIO, etc.)
        return "image"

    @property
    def needs_fields(self):
        return self._needs_fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._needs_fields = fields or {}

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
        try:
            candidates = data.get("candidates") or []
            for cand in candidates:
                content = cand.get("content") or {}
                parts_list = content.get("parts") or []
                texts = [p.get("text") for p in parts_list if isinstance(p, dict) and p.get("text")]
                if texts:
                    return "\n".join(texts)
            pf = data.get("promptFeedback") or {}
            block = pf.get("blockReason")
            if block:
                return f"Blocked by safety system ({block})."
            return str(data)
        except Exception:
            return str(data)

    def _resolve_prompt(self, sample):
        field_name = None
        if isinstance(self.needs_fields, dict):
            field_name = self.needs_fields.get("prompt_field") or next(iter(self.needs_fields.values()), None)

        if sample is not None and field_name:
            try:
                value = sample.get_field(field_name)
            except Exception:
                value = None
            if value is not None and str(value).strip():
                return str(value)

        return self.prompt

    def predict(self, image, sample=None):
        """Predict text for an image. Accepts path, PIL.Image, or numpy array.
        Returns a plain string VQA response.
        """
        prompt = self._resolve_prompt(sample)

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
        result = self._post(parts)
        if self.prompt_echo_field and sample is not None:
            try:
                setattr(sample, self.prompt_echo_field, prompt)
                sample.save()
            except Exception:
                pass
        return result


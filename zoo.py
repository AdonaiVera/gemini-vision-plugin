import os
import base64
import requests
import io
from fiftyone.core.models import Model
from fiftyone import SamplesMixin
from PIL import Image
import numpy as np

class GeminiRemoteModel(SamplesMixin, Model):
    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gemini-2.5-flash")
        self.max_tokens = int(config.get("max_tokens", 2048))
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiRemoteModel")
        self.api_key = api_key
        self.config = config
        self._fields = {}
        self._session = requests.Session()

    @property
    def media_type(self):
        """Returns the media type for the model."""
        # TODO: add support for other media types (VIDEO, AUDIO, etc.)
        return "image"

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def _get_field(self):
        """Get the prompt field from needs_fields."""
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        return prompt_field

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
        """Resolve prompt from sample field."""
        prompt = None
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        return prompt

    def _predict(self, image: Image.Image, sample=None):
        """Internal prediction method that does the actual inference."""
        prompt = self._resolve_prompt(sample)
        b64, mime_type = self._encode_pil(image)
        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}},
        ]
        return self._post(parts)

    def predict(self, image, sample=None):
        """Predict text for an image. Accepts path, PIL.Image, or numpy array.
        Returns a plain string VQA response.
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                raise ValueError(f"Could not load image from path: {image}")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type for predict()")
        
        return self._predict(image, sample)


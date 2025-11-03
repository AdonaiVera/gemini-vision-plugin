## Gemini Vision Plugin

![screen-2025-10-15_13 02 13-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/77b4a2f3-8e4b-40dd-921d-3771b257b8d9)

### Plugin Overview

This plugin integrates Google Gemini's multimodal Vision models (e.g., `gemini-2.5-flash`)
into your FiftyOne workflows. Prompt with text and one or more images; receive a
text response grounded in visual inputs.

## Installation

If you haven't already, install FiftyOne:

```shell
pip install fiftyone
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/AdonaiVera/gemini-vision-plugin
```

To use Gemini Vision, set the following environment variable with your API key:

- `GEMINI_API_KEY`

**Getting your API Key:** Follow this step-by-step guide to create your Gemini API key: [Getting Your API Key Guide](https://github.com/google-gemini/nano-banana-hackathon-kit/blob/main/guides/01-getting-your-api-key.ipynb)

**Important:** You need an active Google Cloud account with billing enabled and credits to use the Gemini API. The free tier has limited quotas. If you encounter quota errors like "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests", you'll need to:

1. Enable billing on your Google Cloud project
2. Purchase credits or upgrade to a paid plan
3. Monitor your usage at: https://ai.dev/usage?tab=rate-limit

Refer to the official docs for pricing and quotas: https://ai.google.dev/gemini-api/docs/rate-limits

### Getting your Data into FiftyOne

To use GPT-4 Vision, you will need to have a dataset of images in FiftyOne. If
you don't have a dataset, you can create one from a directory of images:

```python
import fiftyone as fo

# Load BDD unsafe/safe dataset 
dataset = foz.load_zoo_dataset(
    "https://github.com/AdonaiVera/bddoia-fiftyone",
    split="validation",
    max_samples=10
)
dataset.persistent = True

## view the dataset in the App
session = fo.launch_app(dataset)
```

## Operators

### `query_gemini_vision`

**Demo Video:**

<!-- Add video here -->

Chat with your images using Gemini Vision models.

Inputs:
- `query_text`: The text to prompt Gemini with
- `model`: Select from available Gemini models
- `max_tokens`: The maximum number of output tokens to generate

The operator encodes all selected images and sends them along with your text
prompt to the Gemini Vision API. The model's text response is displayed in the
output panel.

### `text_to_image`

**Demo Video:**

<!-- Add video here -->

Generate high-quality images from text descriptions using Gemini's image generation capabilities.

Inputs:
- `prompt`: Text description of the image to generate
- `aspect_ratio`: Choose from multiple aspect ratios (1:1, 16:9, 9:16, etc.)

The generated image is automatically saved to your dataset with metadata including
the prompt and generation type.

### `image_editing`

**Demo Video:**

<!-- Add video here -->

Edit existing images using text instructions. Provide an image and use text prompts
to add, remove, or modify elements, change the style, or adjust the color grading.

Inputs:
- `prompt`: Edit instruction (e.g., "add sunglasses", "change to watercolor style")
- `aspect_ratio`: Choose from multiple aspect ratios

Select exactly one image from your dataset. The edited image is automatically
saved to your dataset with the original prompt preserved.

### `multi_image_composition`

**Demo Video:**

<!-- Add video here -->

Compose a new image from multiple input images. Use multiple images to create
a new scene or transfer the style from one image to another.

Inputs:
- `prompt`: Composition instruction (e.g., "combine these in a collage", "transfer style from first to second")
- `aspect_ratio`: Choose from multiple aspect ratios

Select 2-3 images from your dataset (optimally up to 3 images). The composed
image is automatically saved to your dataset.

### `video_understanding`

**Demo Video:**

<!-- Add video here -->

Analyze and extract information from videos using Gemini's video understanding capabilities.

Inputs:
- `task_type`: Choose analysis type (describe, segment, extract, question)
- `prompt`: Analysis prompt describing what you want to know about the video

Features:
- **Describe**: Get a comprehensive description of video content
- **Segment**: Identify and describe different segments within the video
- **Extract**: Extract specific information from the video
- **Question**: Ask specific questions about video content, including timestamp-based queries (e.g., "What happens at 0:30?")

Select exactly one video from your dataset. The video must be under 20MB for inline analysis. Analysis results are automatically saved to the video sample's metadata under the `video_analysis` field.

Happy exploring!

## Next Steps

If you like this plugin and find it useful, please leave a ⭐ star on the repository!

### Future Enhancements

We're planning to add more exciting features:

- **Batch Image Generation**: Create multiple images from a single query
- **Pipeline Support**: Build workflows to generate multiple images with different variations
- **Dynamic Prompting**: Use dynamic variables per image for automated, customized generation at scale

Stay tuned for updates!

## Remote Zoo Model

You can also use this repo as a remote model source and load a Gemini model via FiftyOne's Model Zoo API

### Register and load

```python
import fiftyone as fo
import fiftyone.zoo as foz
import os

os.environ["GEMINI_API_KEY"] = "<YOUR_KEY>"

# Register the remote model source
foz.register_zoo_model_source(
    "https://github.com/AdonaiVera/gemini-vision-plugin",
    overwrite=True,
)

# Load the model (Gemini remote HTTP model)
model = foz.load_zoo_model(
    "google/Gemini-Vision",
    model="gemini-2.5-flash",
    max_tokens=2048,
    max_workers=16,
)

# Load a small sample dataset
dataset = foz.load_zoo_dataset("quickstart", split="validation", max_samples=10)

# Apply to a dataset (supports prompt_field for dynamic prompts)
dataset.apply_model(
    model,
    prompt_field="dynamic_prompt",  # Use dynamic prompts from dataset
    label_field="gemini_output",
    image_field="filepath",
)
```

See FiftyOne docs on remote models for more details: [Remote models](https://docs.voxel51.com/model_zoo/remote.html).

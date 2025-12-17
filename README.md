## Gemini Vision Plugin

![screen-2025-10-15_13 02 13-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/77b4a2f3-8e4b-40dd-921d-3771b257b8d9)

### Plugin Overview

This plugin integrates Google Gemini's multimodal Vision models (including the latest `gemini-3-pro-preview`)
into your FiftyOne workflows. Prompt with text and one or more images; receive a
text response grounded in visual inputs. Now featuring Gemini 3.0 with advanced reasoning capabilities!

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

Multi-task vision operator with three modes:

#### Chat
![first_video-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/17d0afd7-5aeb-413b-815a-0f8b04628da6)

Ask questions about your images. Supports up to 64K output tokens with Gemini 3.0.

#### OCR
![ocr_gemini](https://cdn.voxel51.com/tutorial_gemini_vision/ocr_gemini.webp)

Extract text with bounding boxes. Results stored as `fo.Detections`.

#### Spatial
![pointing](https://cdn.voxel51.com/tutorial_gemini_vision/pointing.webp)

Detect points/keypoints (e.g., pose estimation, object pointing). Results stored as `fo.Keypoints`.

**Inputs:**
- `task`: Select mode (`chat`, `ocr`, `spatial`)
- `model`: Gemini model (default: `gemini-3-pro-preview`)
- `query_text`: Prompt for chat/spatial tasks
- `label_field`: Field name to store OCR/spatial results

### `text_to_image`
![text_image-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/5eb1ce21-3cfd-4649-ba5e-24e2a035c7c4)

Generate high-quality images from text descriptions using Gemini's image generation capabilities.

Inputs:
- `prompt`: Text description of the image to generate
- `model`: Choose between `gemini-2.5-flash-image` or `gemini-3-pro-image-preview` (Nano Banana Pro - default, supports 2K/4K)
- `aspect_ratio`: Choose from multiple aspect ratios (1:1, 16:9, 9:16, etc.)

The generated image is automatically saved to your dataset with metadata including the prompt and generation type.

### `image_editing`
![edit_image](https://github.com/user-attachments/assets/e0b6a483-fe22-464f-98f2-4d27b5cf21eb)

Edit existing images using text instructions. Provide an image and use text prompts to add, remove, or modify elements, change the style, or adjust the color grading.

Inputs:
- `prompt`: Edit instruction (e.g., "add sunglasses", "change to watercolor style")
- `model`: **NEW** - Choose between `gemini-2.5-flash-image` or `gemini-3-pro-image-preview` (Nano Banana Pro - default)
- `aspect_ratio`: Choose from multiple aspect ratios

Select exactly one image from your dataset. The edited image is automatically saved to your dataset with the original prompt preserved.

### `multi_image_composition`

Compose a new image from multiple input images. Use multiple images to create a new scene or transfer the style from one image to another.

Inputs:
- `prompt`: Composition instruction (e.g., "combine these in a collage", "transfer style from first to second")
- `model`: **NEW** - Choose between `gemini-2.5-flash-image` or `gemini-3-pro-image-preview` (Nano Banana Pro - default)
- `aspect_ratio`: Choose from multiple aspect ratios

Select 2-3 images from your dataset (optimally up to 3 images). The composed image is automatically saved to your dataset.

### `video_understanding`
![video_describing-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/559420bd-2b08-4941-a2db-12d373eff178)

Analyze and extract information from videos using Gemini's video understanding capabilities.

Inputs:
- `task_type`: Choose analysis type (describe, segment, extract, question)
- `prompt`: Analysis prompt describing what you want to know about the video
- `model`: Select from available Gemini models (default: `gemini-3-pro-preview`)
- `thinking_level`: **NEW in Gemini 3.0** - Control reasoning depth (`low` for speed/cost, `high` for complex reasoning)
- `media_resolution`: **NEW in Gemini 3.0** - Video frame resolution control:
  - `high` (1,120 tokens/frame) - Recommended for detailed analysis
  - `medium` (560 tokens/frame) - Optimal for PDFs
  - `low` (70 tokens/frame) - Most efficient, fewer tokens consumed

Features:
- **Describe**: Get a comprehensive description of video content
- **Segment**: Identify and describe different segments within the video
- **Extract**: Extract specific information from the video
- **Question**: Ask specific questions about video content, including timestamp-based queries (e.g., "What happens at 0:30?")

Select exactly one video from your dataset. The video must be under 20MB for inline analysis. Analysis results are automatically saved to the video sample's metadata under the `video_analysis` field. Gemini 3.0 consumes fewer tokens per video while providing better reasoning!

Happy exploring!

## Next Steps

If you like this plugin and find it useful, please leave a ⭐ star on the repository!

### Future Enhancements

We're planning to add more exciting features:

- **Batch Image Generation**: Create multiple images from a single query
- **Pipeline Support**: Build workflows to generate multiple images with different variations
- **Dynamic Prompting**: Use dynamic variables per image for automated, customized generation at scale

Stay tuned for updates!

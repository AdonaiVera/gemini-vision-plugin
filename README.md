## Gemini Vision Plugin

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

Refer to the official docs for pricing and quotas: `https://ai.google.dev/gemini-api/docs`.

### Getting your Data into FiftyOne

To use GPT-4 Vision, you will need to have a dataset of images in FiftyOne. If
you don't have a dataset, you can create one from a directory of images:

```python
import fiftyone as fo

dataset = fo.Dataset.from_images_dir("/path/to/images")

## optionally name the dataset and persist to disk
dataset.name = "my-dataset"
dataset.persistent = True

## view the dataset in the App
session = fo.launch_app(dataset)
```

## Operators

### `query_gemini_vision`

Inputs:

- `query_text`: The text to prompt Gemini with
- `max_tokens`: The maximum number of output tokens to generate

The operator encodes all selected images and sends them along with your text
prompt to the Gemini Vision API. The model's text response is displayed in the
output panel.

Happy exploring!

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

# Apply to a dataset (supports prompt_field or a static prompt)
raw_output_field = "gemini_output"
classification_field = None
dataset.apply_model(
    model,
    prompt_field="dynamic_prompt",
    label_field=raw_output_field,
    image_field="filepath",
)
```

See FiftyOne docs on remote models for more details: [Remote models](https://docs.voxel51.com/model_zoo/remote.html).

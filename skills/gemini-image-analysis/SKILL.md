---
name: gemini-image-analysis
description:
    Use when the user wants to ask questions about IMAGES, read text out of
    them, or point at things in them with Gemini — "what's in these images",
    "caption my dataset", "extract the text / read the labels / OCR these
    receipts", "find the license plates", "point at each person's head",
    "label the keypoints". Writes answers as sample fields, OCR as Detections,
    and points as Keypoints.
category: Curate
---

# Gemini Image Analysis

Runs Gemini over the images in a FiftyOne dataset in three modes — free-form
chat, OCR with bounding boxes, and spatial pointing — and writes the results
back as real FiftyOne labels rather than loose text.

Backed by the `@adonaivera/gemini_vision` plugin. Requires a `GEMINI_API_KEY`.

For video, use `gemini-video-understanding`. For generating or editing images,
use `gemini-image-generation`.

## Key Directives

**ALWAYS follow these rules — no exceptions:**

### 1. Verify the plugin is installed

```
list_operators()
```

Look for `@adonaivera/gemini_vision/query_gemini_vision`. If absent, check
`list_plugins(enabled=True)` for `@adonaivera/gemini_vision` and tell the user
to install or enable it. Do not silently fall back to raw API code.

### 2. This operator runs on the App's selection

It reads `ctx.selected`. If nothing is selected it has nothing to do. Before
running, either confirm the user has a selection or set one:

```
set_selected_samples(sample_ids=["<id1>", "<id2>"])
```

Say how many samples you are about to send — Gemini bills per image.

### 3. Pick the mode from the user's words, not from the field they want

| The user asks | `task` |
| --- | --- |
| "What's in this image?", "caption these", "is there a person here" | `chat` |
| "Read the text", "OCR", "extract the serial number", "what does the sign say" | `ocr` |
| "Point at each X", "mark the keypoints", "where is the handle" | `spatial` |

### 4. Never overwrite an existing label field without asking

```
get_field_schema(dataset_name="<name>")
```

If `label_field` already exists, either pick a new name or confirm the
overwrite with the user first.

### 5. Show the results in the App

```
set_active_fields(["<label_field>"])
```

OCR detections and spatial keypoints only mean something drawn on the image.

---

## Operator Reference

| Action | Operator URI | Requires Prompt |
| --- | --- | --- |
| Chat / OCR / spatial over selected images | `@adonaivera/gemini_vision/query_gemini_vision` | True |

The operator surfaces its parameters before running so the user can adjust the
task, prompt, model and label field. Each call bills per image.

### Parameters

| Parameter | Type | Notes |
| --- | --- | --- |
| `task` | enum | `chat`, `ocr` or `spatial` |
| `query_text` | str | Required for `chat` and `spatial` |
| `model` | str | Default `gemini-3.1-pro-preview` |
| `max_tokens` | int | `chat` only. Default 65536 |
| `label_field` | str | `ocr` → `Detections` (default `ocr_detections`); `spatial` → `Keypoints` (default `spatial_keypoints`) |

---

## Workflow

### Step 1 — Confirm the dataset and the selection

```
dataset_summary(name="<name>")
get_field_schema(dataset_name="<name>")
```

Confirm `media_type` is `image`. On a video dataset route to
`gemini-video-understanding`.

### Step 2 — Run

```
execute_operator(
    operator_uri="@adonaivera/gemini_vision/query_gemini_vision",
    params={"task": "ocr", "label_field": "receipt_text"},
)
```

For `chat`:

```
params={"task": "chat", "query_text": "Is anyone wearing a hard hat?"}
```

For `spatial`:

```
params={
    "task": "spatial",
    "query_text": "Point at the center of each visible face",
    "label_field": "face_points",
}
```

### Step 3 — Report and display

`chat` returns `question` and `answer`. `ocr` and `spatial` return `processed`,
`total`, `label_field` and `errors` — always report `errors` if non-empty
rather than claiming a clean run.

```
set_active_fields(["<label_field>"])
```

### Step 4 — Aggregate

```
count_values("<label_field>.detections.label")
```

Useful for spotting what OCR actually pulled out across the dataset before the
user trusts it.

---

## Common Use Cases

### "Read the text in all these images"

`ocr` with a descriptive `label_field`. Results are `fo.Detections` with the
recognized string as the label, so `count_values` gives a frequency table of
everything read across the dataset — the fastest way to spot systematic OCR
failures.

### "Caption my dataset"

`chat` with a `query_text` like "Write a one-sentence caption." The answer
comes back in the operator output. For a large dataset, work in batches of
selected samples and say what each batch cost.

### "Point at every X"

`spatial`. Results are `fo.Keypoints`, which is the right structure for pose
and pointing tasks — not bounding boxes. If the user actually wants boxes, say
so and use `ocr` (for text) or route them to a detection model via
`fiftyone-dataset-inference`.

---

## Troubleshooting

**"You must select a sample to use this operator"**

Nothing is selected. Use `set_selected_samples(sample_ids=[...])` first.

**OCR returns nothing on clearly legible text**

Check the image resolution. Very small crops lose text at the model's input
resolution — try the full-resolution source image rather than a thumbnail.

**Quota errors**

`generativelanguage.googleapis.com/generate_content_free_tier_requests`
exceeded means the key is on the free tier. The user needs billing enabled on
their Google Cloud project. Point them at
https://ai.google.dev/gemini-api/docs/rate-limits.

---
name: gemini-image-generation
description:
    Use when the user wants to CREATE or MODIFY imagery with Gemini's image
    models (Nano Banana Pro) — "generate an image of...", "add synthetic
    samples", "edit this image", "remove the background", "change it to
    winter", "make a night-time version", "combine these two images",
    "transfer the style". Generated and edited images are added to the dataset
    as new samples.
category: Curate
---

# Gemini Image Generation and Editing

Creates new imagery with Gemini's image models and adds it to the dataset as
real samples — useful for augmenting a thin class, producing counterfactual
variants of an existing sample, or composing reference imagery.

Backed by the `@adonaivera/gemini_vision` plugin. Requires a `GEMINI_API_KEY`.

## Key Directives

**ALWAYS follow these rules — no exceptions:**

### 1. These operators write files and add samples — confirm first

Every operator here creates an image on disk and appends a sample to the
dataset. That is not something to do on a hunch. Restate what you are about to
generate, how many samples it will add, and where they will land, and get the
user's go-ahead before running.

### 2. Know which operator takes which selection

| Operator | Selection required |
| --- | --- |
| `text_to_image` | none |
| `image_editing` | exactly **one** sample |
| `multi_image_composition` | **2 or 3** samples (only the first 3 are used) |

Check `ctx.selected` equivalents before calling:

```
set_selected_samples(sample_ids=["<id>"])
```

Calling `image_editing` with a multi-sample selection fails; it does not pick
one for you.

### 3. Generated samples are not ground truth — mark them

Every generated sample carries `prompt` and `generation_type` fields. After a
run, tag them so they never silently enter a training or evaluation split:

```
tag_samples(tags=["synthetic"])
```

Tell the user you did this and why.

### 4. Pick the model deliberately

| Model | When |
| --- | --- |
| `gemini-3-pro-image` | Default — Nano Banana Pro, best quality, 2K/4K |
| `gemini-3.1-flash-image` | Fast tier — bulk generation where quality is secondary |
| `gemini-2.5-flash-image` | Older fast model; use only if the user asks |

### 5. Match the aspect ratio to the dataset

All three operators accept a "use the source image's aspect ratio" toggle. For
augmentation, keep it on — new samples that differ in shape from the rest of
the dataset skew downstream training.

---

## Operator Reference

| Action | Operator URI | Requires Prompt |
| --- | --- | --- |
| Generate an image from text | `@adonaivera/gemini_vision/text_to_image` | True |
| Edit one selected image | `@adonaivera/gemini_vision/image_editing` | True |
| Compose 2-3 selected images | `@adonaivera/gemini_vision/multi_image_composition` | True |

All three are marked as requiring a parameter prompt: they create files and add
samples, so the user should see the parameters before the run rather than
after.

### Shared parameters

| Parameter | Type | Notes |
| --- | --- | --- |
| `prompt` | str | Required — the instruction |
| `model` | enum | See the table above |
| `aspect_ratio` | enum | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `use_dataset_size` | bool | `text_to_image` only — derive the ratio from the dataset's first image |
| `use_original_size` | bool | `image_editing` and `multi_image_composition` — derive the ratio from the source image. Default `True` |

These operators run on the **App's selection**. Set it before calling:

```
set_selected_samples(sample_ids=["<id>"])
```

---

## Workflow

### Step 1 — Confirm scope with the user

State the prompt, the model, how many samples will be added, and the dataset
they will be added to. Wait for confirmation.

### Step 2 — Set the selection (editing and composition only)

```
set_selected_samples(sample_ids=["<id>"])
```

### Step 3 — Run

```
execute_operator(
    operator_uri="@adonaivera/gemini_vision/text_to_image",
    params={"prompt": "a forklift in a warehouse at night", "aspect_ratio": "16:9"},
)
```

```
execute_operator(
    operator_uri="@adonaivera/gemini_vision/image_editing",
    params={"prompt": "change the weather to heavy rain"},
)
```

```
execute_operator(
    operator_uri="@adonaivera/gemini_vision/multi_image_composition",
    params={"prompt": "put the subject of the first image into the scene of the second"},
)
```

### Step 4 — Tag and show

```
reload_dataset()
tag_samples(tags=["synthetic"])
```

The operator returns `filepath` for the new image. Report it, and open the new
sample so the user can judge the result:

```
open_sample(sample_id="<new id>")
```

---

## Common Use Cases

### "Add more examples of a rare class"

`text_to_image` in a loop, varying the prompt each time. Keep the aspect ratio
matched to the dataset, tag everything `synthetic`, and remind the user that
generated samples need labels before they are useful for training — route to
`fiftyone-auto-labeling` or the annotation workflow next.

### "What would this scene look like at night?"

`image_editing` on one selected sample. This is the counterfactual-augmentation
case: keep the original and the edit side by side so the user can compare, and
tag the edit.

### "Combine these images"

`multi_image_composition` with 2 or 3 selected samples. Be specific in the
prompt about which image contributes what — "the subject of the first, the
background of the second" works far better than "combine these".

---

## Troubleshooting

**"You must select exactly one image to edit"**

`image_editing` had zero or multiple samples selected. Narrow the selection.

**"You must select at least 2 images to compose"**

`multi_image_composition` needs 2 or 3. Only the first 3 are used.

**New sample doesn't appear**

```
reload_dataset()
```

The App caches the sample list; the file was written but the grid is stale.

**Quota errors**

Image generation is billed per image and is not on the free tier for the pro
model. See https://ai.google.dev/gemini-api/docs/rate-limits.

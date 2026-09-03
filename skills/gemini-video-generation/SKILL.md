---
name: gemini-video-generation
description:
    Use when the user wants to CREATE or EDIT video with Gemini Omni —
    "generate a video of...", "make a clip", "animate this image", "turn this
    photo into a video", "make it night time", "extend that clip", "continue
    the shot", "add a synthetic video sample". Generates video with a
    synthesized audio track and adds it to the dataset as a real video sample.
category: Curate
---

# Gemini Video Generation

Generates video with Gemini Omni and adds it to the dataset as a video sample.
Omni produces a clip with a synthesized audio track from text, from a still
image, or by editing and extending a clip it produced earlier.

This is the opposite direction to `gemini-video-understanding`: that skill
reads video, this one writes it. Route to that one for any question *about*
existing footage.

Backed by the `@adonaivera/gemini_vision` plugin. Requires a `GEMINI_API_KEY`.

## Key Directives

**ALWAYS follow these rules — no exceptions:**

### 1. Confirm before generating — this writes files and adds samples

Every run creates an MP4 on disk and appends a sample to a dataset. State the
prompt, the task, the resolution and which dataset the clip will land in, then
wait for the user to agree. Never generate on a hunch.

### 2. Draft at 360p, finish at 720p or above

`360p` costs roughly a third of `720p`. For any prompt the user has not seen
results from yet, generate a 360p draft first, show it, and only re-run at
higher resolution once they are happy with the shot. Say that you are doing
this.

### 3. Match the task to what the user is starting from

| The user has | Task | Selection |
| --- | --- | --- |
| only a description | `text_to_video` | none |
| one image to bring to life | `image_to_video` | exactly 1 image |
| a start and end frame | `first_last_frame` | exactly 2 images |
| subjects to keep consistent | `reference_to_video` | 1-3 images |
| a clip **this plugin generated**, to change | `edit` | exactly 1 generated video |
| a clip **this plugin generated**, to continue | `extend` | exactly 1 generated video |

### 4. `edit` and `extend` only work on clips this plugin generated

They resume the model's own interaction, keyed by the `gemini_interaction_id`
field on the generated sample. An uploaded or imported video has no such id and
cannot be edited here. Check before promising it:

```
get_field_schema(dataset_name="<name>")
```

If the user wants to edit footage they brought, say plainly that Omni cannot do
that through this plugin.

### 5. Tag generated clips as synthetic

```
tag_samples(tags=["synthetic"])
```

A generated clip must never drift into a training or evaluation split
unmarked. Do this after every run and tell the user you did.

### 6. Delegate — generation is slow

20-60 seconds is normal, longer at high resolution. Pass `delegate=True` for
anything beyond a single draft, then poll:

```
list_delegated_operations(dataset_name="<name>")
```

---

## Operator Reference

| Action | Operator URI | Requires Prompt |
| --- | --- | --- |
| Generate, edit or extend a video | `@adonaivera/gemini_vision/video_generation` | True |

Marked as requiring a parameter prompt: it creates files and samples, so the
user sees the parameters before the run, not after.

### Parameters

| Parameter | Type | Notes |
| --- | --- | --- |
| `task` | enum | See the table above. Default `text_to_video` |
| `prompt` | str | Required for every task |
| `model` | enum | `gemini-omni-1.1-flash` (default) or `gemini-omni-flash-preview` |
| `resolution` | enum | `360p`, `720p` (default), `1080p`, `4k`. Above 720p is upscaled |
| `aspect_ratio` | enum | `16:9` (default) or `9:16` |
| `output_dir` | str | Where the clip is written. Prefilled with a folder beside the dataset's media; **required** when that location is read-only |

The operator runs on the **App's selection** — there is no `sample_ids`
parameter. Set the selection before calling:

```
set_selected_samples(sample_ids=["<id>"])
```

A selection that does not match the task blocks execution rather than warning:
`edit`, `extend` and `image_to_video` take exactly one sample,
`first_last_frame` exactly two, `reference_to_video` one to three, and
`text_to_video` none.

### What lands on the sample

| Field | Holds |
| --- | --- |
| `prompt` | the instruction that produced the clip |
| `generation_type` | the task used |
| `gemini_interaction_id` | **what makes the clip editable later** — do not discard it |
| `gemini_model`, `resolution`, `aspect_ratio` | generation settings |
| `source_sample_ids` | the samples the generation drew on |

On a video dataset the clip is added directly. On an image dataset it goes to a
companion `<dataset>-gemini-video` dataset, because a FiftyOne dataset holds a
single media type. Always tell the user which dataset it landed in.

---

## Workflow

### Step 1 — Confirm scope

State the task, prompt, resolution and destination dataset. Wait.

### Step 2 — Set the selection for image- and clip-based tasks

```
set_selected_samples(sample_ids=["<id>"])
```

### Step 3 — Generate a draft

```
execute_operator(
    operator_uri="@adonaivera/gemini_vision/video_generation",
    params={
        "task": "image_to_video",
        "prompt": "slow cinematic push-in, natural ambient sound",
        "resolution": "360p",
    },
)
```

### Step 4 — Show it, then refine

```
reload_dataset()
open_sample(sample_id="<new id>")
tag_samples(tags=["synthetic"])
```

To change the draft, use `edit` against the clip that came back — not a fresh
`text_to_video` with a longer prompt. Editing preserves what the user already
liked:

```
params={"task": "edit", "prompt": "make it night time"}   # on the selected clip
```

### Step 5 — Re-render at full resolution once approved

Generate again at `720p` or above only after the user is happy with the draft.

---

## Common Use Cases

### "Make a video of X"

`text_to_video` at 360p. Describe motion and sound in the prompt, not just the
subject — Omni synthesizes an audio track, and a prompt that says nothing about
sound gets whatever the model picks.

### "Bring this photo to life"

`image_to_video` on one selected image. Keep the prompt about camera and motion
("slow push-in", "handheld drift left"); the content already comes from the
image.

### "Add synthetic video samples to my dataset"

`text_to_video` in a loop with varied prompts, tagged `synthetic`. Remind the
user the clips carry no labels yet, and route to `gemini-video-understanding`
or an annotation workflow to label them.

### "Change something about that clip"

`edit`, against the generated sample. Multiple edits chain — each result is
itself editable, so a session can converge on a shot over several turns.

### "Make it longer"

`extend`, up to 40 seconds cumulative in 10-second increments.

---

## Limits

- A single generation is 3-10 seconds; `extend` reaches 40 seconds cumulative
- `1080p` and `4k` are upscaled, not natively rendered
- System instructions, temperature and negative prompts are unsupported — put
  what you do *not* want into the prompt itself
- Audio cannot be supplied as input, and generated voices cannot be edited
- YouTube URLs cannot be used as a source
- Editing or extending *uploaded* video is unavailable in the EEA, Switzerland
  and the UK; editing the model's own generations is available

---

## Troubleshooting

**"That sample was not generated by this plugin"**

`edit` and `extend` need `gemini_interaction_id` on the sample. The user
selected footage they imported. Offer `image_to_video` from a frame instead.

**"Operator not found"**

```
list_plugins(enabled=True)
```

Install with
`fiftyone plugins download https://github.com/AdonaiVera/gemini-vision-plugin`,
or have an admin enable `@adonaivera/gemini_vision`.

**The clip does not appear in the grid**

```
reload_dataset()
```

On an image dataset, check the companion `<dataset>-gemini-video` dataset — the
clip is not in the dataset the user was looking at.

**Generation times out**

Re-run with `delegate=True` and a lower resolution.

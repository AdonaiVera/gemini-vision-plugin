## Gemini Vision Plugin

![screen-2025-10-15_13 02 13-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/77b4a2f3-8e4b-40dd-921d-3771b257b8d9)

### Plugin Overview

Brings Google Gemini into FiftyOne — **agentic video understanding**, image
Q&A, OCR, spatial pointing, and image generation with Nano Banana Pro.

Version 2.0 rebuilds video on Gemini's [agentic video
understanding](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/),
where the model navigates the timeline itself — searching, scanning and
re-sampling only the segments your prompt actually needs — instead of being
handed a fixed 1 FPS strip of frames. Google reports up to **88% fewer tokens,
66% lower cost and 7% better accuracy** than static sampling — figures for long,
heterogeneous content. They did not hold on our own benchmark, so the plugin
ships both modes and the cost note below shows what we measured.

The part that matters inside FiftyOne: timestamped answers come back
**structured**, so they land on the sample as `TemporalDetections` — seekable
in the App's video timeline and filterable like any other label field. A video
question stops being a paragraph of prose and becomes data.

Version 2.1 adds the other direction: **video generation with Gemini Omni**,
Google's video model. Omni produces a clip with a synthesized audio track from
text, from a still image, or by editing and extending a clip it made earlier —
and the result lands in your dataset as a real video sample.

The plugin also ships four **FiftyOne skills**, so the FiftyOne Agent can
discover these operators and drive them from a natural-language request.

## Installation

If you haven't already, install FiftyOne:

```shell
pip install fiftyone
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/AdonaiVera/gemini-vision-plugin
```

To use it, set your API key:

- `GEMINI_API_KEY`

**Getting your API Key:** follow [this step-by-step
guide](https://github.com/google-gemini/nano-banana-hackathon-kit/blob/main/guides/01-getting-your-api-key.ipynb).

**Important:** you need an active Google Cloud account with billing enabled.
The free tier has limited quotas. If you hit
`Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests`:

1. Enable billing on your Google Cloud project
2. Purchase credits or upgrade to a paid plan
3. Monitor usage at https://ai.dev/usage?tab=rate-limit

See the [pricing and quota docs](https://ai.google.dev/gemini-api/docs/rate-limits).

### Getting your data into FiftyOne

```python
import fiftyone as fo
import fiftyone.zoo as foz

# A video dataset to try the video operator on
dataset = foz.load_zoo_dataset("quickstart-video")
dataset.persistent = True

session = fo.launch_app(dataset)
```

---

## Operators

### `video_understanding` — agentic video

![video_describing-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/559420bd-2b08-4941-a2db-12d373eff178)

Ten tasks, covering the four workflows agentic video was built for:

| Task | You ask | Result |
| --- | --- | --- |
| `describe` | "What happens here?" | `TemporalDetections` — one per beat — + a summary |
| `state_change` | "When does the door open?" | `TemporalDetections` |
| `anomaly` | "Anything unsafe here?" | `TemporalDetections` + `severity` |
| `needle` | "Find every red forklift" | `TemporalDetections` |
| `count` | "How many times does X happen?" | `TemporalDetections` + a total |
| `frame_extract` | "Pull the best frames" | `TemporalDetections` + a frames dataset |
| `chapters` | "Split this into chapters" | `TemporalDetections` |
| `cause_effect` | "Why did the stack fall?" | `TemporalDetections` — one per causal link — + a summary |
| `qa` | "What's at 0:30?" | Markdown |
| `transcript` | "Transcribe this" | Markdown |

Eight of the ten tasks return **timestamped moments**, not prose. That is the
point: a question about a video should leave behind something you can seek to,
filter and count later — not a paragraph you have to re-read. Only `qa` (a
direct answer) and `transcript` (inherently line-based) stay prose.

`needle`, `count`, `cause_effect` and `qa` require a prompt; the rest take one
only to narrow the scope.

**Inputs**

| Parameter | Notes |
| --- | --- |
| `task_type` | See the table above. Default `describe` |
| `prompt` | Required for four of the tasks |
| `max_videos` | Default 5 — a guard against a runaway bill |
| `model` | Default `gemini-3.8-flash`. Also `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` |
| `processing_mode` | `agentic` (default) or `static` |
| `fps`, `start_offset`, `end_offset` | `static` mode only |
| `thinking_level` | `low` or `high` (default) |
| `label_field` | Where temporal detections land |
| `extract_frames` | `frame_extract` only — cuts the frames to disk and into a companion dataset |
| `source` / `youtube_url` | Analyze a public YouTube video directly, no download |

**Outputs** include `result_field` (the sample field the answer was written
to), `num_events`, `label_field`, `total_tokens`, and `processing_calls` — how
many times the model reached back into the video mid-reasoning. That last one is
the agentic loop made visible; static mode always returns 0.

Every task writes to its own field, so running several on one video does not
overwrite anything. Tasks that return moments write `TemporalDetections` to
their field plus the readable answer to `<field>_summary`; `qa` and
`transcript` write `gemini_qa` and `gemini_transcript` directly.

Each detection carries `label`, `support` (the `[first, last]` frame numbers
the App seeks on), `confidence`, `start_time` and `end_time` in seconds, and a
`description`.

**Video sources.** Videos up to 20MB are sent inline; larger ones are uploaded
through the Files API automatically, up to 2GB. Public YouTube URLs are passed
straight through.

**Cost note — measure, don't assume.** On a controlled needle-in-a-haystack
test (one armored truck hidden once in 30 minutes of repeated street footage,
`gemini-3.8-flash`, same prompt):

| Configuration | Tokens | Video lookups | Found it |
| --- | --- | --- | --- |
| Agentic, `thinking_level="high"` | 553,560 | 18 | yes, exactly |
| Agentic, `thinking_level="low"` | 274,183 | 16 | **no** |
| Static, `fps=1` | 119,184 | 0 | yes, exactly |
| Static, `fps=0.5` | 61,053 | 0 | yes, exactly |

The agentic loop has a cost floor it pays before reading any video, so on
footage a uniform pass already handles, static sampling can be an order of
magnitude cheaper for the same answer. And `thinking_level="low"` cost
*recall* here, not just latency — it was the only run that missed. The footage
is deliberately repetitive, which favours a uniform pass, so treat this as a
reason to measure on your own video rather than as a general verdict. Try
`processing_mode="static"` with `fps=0.5` first; reach for agentic when a
uniform pass demonstrably misses things.

```python
import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("quickstart-video")

foo.execute_operator(
    "@adonaivera/gemini_vision/video_understanding",
    ctx=dict(
        dataset=dataset,
        selected=[dataset.first().id],
        params={
            "task_type": "needle",
            "prompt": "a pedestrian crossing the street",
            "label_field": "pedestrian_events",
        },
    ),
)

sample = dataset.first()
for det in sample.pedestrian_events.detections:
    print(det.label, det.support, det.confidence)
```

---

### `video_generation` — video generation with Omni

Generates video **with audio** using Gemini Omni, and adds it to your dataset
as a video sample.

| Task | Starting from | Selection |
| --- | --- | --- |
| `text_to_video` | a description | none |
| `image_to_video` | one image to animate | exactly 1 image |
| `first_last_frame` | a start and an end frame | exactly 2 images |
| `reference_to_video` | subjects to keep consistent | 1-3 images |
| `edit` | a clip this plugin generated | exactly 1 generated video |
| `extend` | a clip this plugin generated | exactly 1 generated video |

**Inputs:** `task`, `prompt` (required), `model`
(`gemini-omni-1.1-flash` default), `resolution` (`360p`, `720p` default,
`1080p`, `4k`), `aspect_ratio` (`16:9` or `9:16`).

A generation is 3-10 seconds; `extend` reaches 40 seconds cumulative in 10
second increments. `360p` costs roughly a third of `720p`, so draft there and
re-render once you like the shot. `1080p` and `4k` are upscaled.

**What lands on the sample**

| Field | Holds |
| --- | --- |
| `prompt` | the instruction that produced the clip |
| `generation_type` | the task used |
| `gemini_interaction_id` | what makes the clip editable and extendable later |
| `gemini_model`, `resolution`, `aspect_ratio` | generation settings |
| `source_sample_ids` | the samples the generation drew on |

On a video dataset the clip is added directly; on an image dataset it goes to a
companion `<dataset>-gemini-video` dataset, since a FiftyOne dataset holds one
media type. Clips are written under the target dataset's own directory, never
beside the source media.

```python
import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("quickstart")

foo.execute_operator(
    "@adonaivera/gemini_vision/video_generation",
    ctx=dict(
        dataset=dataset,
        selected=[dataset.first().id],
        params={
            "task": "image_to_video",
            "prompt": "slow cinematic push-in, natural ambient sound",
            "resolution": "360p",
        },
    ),
)
```

**Limits.** Audio cannot be supplied as input and generated voices cannot be
edited. System instructions, temperature and negative prompts are unsupported —
put what you do not want into the prompt. `edit` and `extend` work only on
clips this plugin generated, because they resume the model's own interaction;
editing uploaded video is additionally unavailable in the EEA, Switzerland and
the UK.

---

### `query_gemini_vision` — image tasks

Three modes over the images you have selected.

#### Chat
![question_answer_gemini-ezgif com-optiwebp](https://github.com/user-attachments/assets/48dce8da-fcca-443a-8398-a51020d4b125)

Ask questions about your images.

#### OCR
![ocr_gemini](https://github.com/user-attachments/assets/c1b3eb6a-6d08-415f-91e1-6ab1dc6b6f0f)

Extract text with bounding boxes. Stored as `fo.Detections`.

#### Spatial
![pointing](https://github.com/user-attachments/assets/8a5bf159-2e90-433e-a13d-cac5bab796d2)

Detect points and keypoints (pose estimation, object pointing). Stored as
`fo.Keypoints`.

**Inputs:** `task` (`chat` | `ocr` | `spatial`), `model` (default
`gemini-3.1-pro-preview`), `query_text`, `label_field`.

---

### `text_to_image`
![text_image-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/5eb1ce21-3cfd-4649-ba5e-24e2a035c7c4)

Generate images from text and add them to your dataset.

**Inputs:** `prompt`, `model` (`gemini-3-pro-image` — Nano Banana Pro, default;
`gemini-3.1-flash-image`; `gemini-2.5-flash-image`), `aspect_ratio`.

### `image_editing`
![edit_image](https://github.com/user-attachments/assets/e0b6a483-fe22-464f-98f2-4d27b5cf21eb)

Edit one selected image with a text instruction — add or remove elements,
change style, adjust color grading.

**Inputs:** `prompt`, `model`, `aspect_ratio`. Select exactly one image.

### `multi_image_composition`

Compose a new image from 2-3 selected images — build a scene, or transfer style
from one to another.

**Inputs:** `prompt`, `model`, `aspect_ratio`.

---

## Media: open-source and Enterprise

The plugin runs unchanged on open-source FiftyOne and on FiftyOne Enterprise,
and adapts to whichever it finds. It never imports an Enterprise module or
checks a version — it uses what the runtime hands it.

**Reading.** Enterprise datasets commonly have cloud filepaths
(`gs://`, `s3://`). Those are resolved through the media cache the sample
itself exposes, so the operators read cloud-backed media the same way the App
renders it. On open-source, where no such cache exists, local files work as
before and a cloud path is refused with a message saying why rather than a
bare "not found".

**Writing.** Generated clips, extracted frames and generated images land in a
`gemini_generated` or `gemini_frames` folder **beside the media they came
from** — the same bucket on a cloud-backed dataset, the same directory
locally.

When that location is read-only — a mounted dataset directory, common in a
deployment — Enterprise **refuses to write and asks for a destination** rather
than falling back to local disk. A local path on a shared deployment produces a
sample only the node that generated it can see, which is worse than an error.
Set the operator's *Output directory* to a writable cloud location. Open-source
falls back to the dataset's own folder, since everything there is local
anyway.

Filenames are `<task>_<source name>_<UTC timestamp>_<random suffix>`. The
random suffix matters: an agent can run several generations inside the same
second, and existing files are never overwritten — the destination is checked
before the write.

## Skills

The plugin ships three FiftyOne skills, declared in `fiftyone.yml` and living
under `skills/`. A FiftyOne Agent in any deployment where this plugin is
enabled discovers them automatically and can drive these operators from a
natural-language request — no glue code.

| Skill | Covers |
| --- | --- |
| `gemini-video-understanding` | All ten video tasks, cost control, and how results land as temporal detections |
| `gemini-video-generation` | Generating, editing and extending video with Omni |
| `gemini-image-analysis` | Chat, OCR and spatial pointing over images |
| `gemini-image-generation` | Generation, editing and composition |

Verify they are visible:

```shell
fiftyone skills list --plugin @adonaivera/gemini_vision
```

```
name                        category    plugin                     enabled
--------------------------  ----------  -------------------------  ---------
gemini-video-understanding  Curate      @adonaivera/gemini_vision  ✓
gemini-video-generation     Curate      @adonaivera/gemini_vision  ✓
gemini-image-analysis       Curate      @adonaivera/gemini_vision  ✓
gemini-image-generation     Curate      @adonaivera/gemini_vision  ✓
```

All six operators surface their parameter form before running, so the user can
review and edit the task, prompt, model and target fields before anything
executes or bills.

Then just ask the agent:

> Find every time a pedestrian appears in these videos and show me the results

The agent loads `gemini-video-understanding`, picks the `needle` task, calls
`video_understanding` with the right parameters, and puts the temporal
detections on screen.

---

## Upgrading from 1.x

- **Video is rebuilt on the Interactions API.** The old `generateContent` path
  with its 20MB hard cap is gone. Videos up to 2GB work, and agentic
  processing is the default.
- **`media_resolution` is removed.** It was accepted but never actually sent,
  and the Interactions API has no equivalent. Use `processing_mode="static"`
  with `fps` for explicit control over how much video is sampled.
- **Task names.** `segment` → `chapters`, `extract` → `needle`, `question` →
  `qa`. The old names still work.
- **Results move from a metadata blob to real labels.** 1.x appended dicts to a
  `video_analysis` list field. 2.0 writes `TemporalDetections` plus a Markdown
  summary field, and every task has its own field so running several on one
  video overwrites nothing. Existing data is untouched but is not read by the
  new operator.
- **New in 2.1.** The `video_generation` operator (Gemini Omni), a fourth
  skill, and a `result_field` output naming the sample field each run wrote to.
- **Model defaults.** `gemini-3-pro-preview` was retired by Google and now
  returns 404, so the vision default is `gemini-3.1-pro-preview` and image
  generation defaults to the GA `gemini-3-pro-image`.

## Next Steps

If you find this plugin useful, please leave a ⭐ on the repository!

### Future enhancements

- Batch video generation from a prompt list
- Pipeline support for generating variations at scale
- Dynamic per-image prompting for automated generation

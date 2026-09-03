---
name: gemini-video-understanding
description:
    Use when the user wants to analyze, search, question or edit VIDEO with
    Gemini — including "what happens in this video", "find every time X
    appears", "count how many times Y happens", "is there anything anomalous",
    "why did Z happen", "split this into chapters", "pull the best frames out",
    "transcribe this video", or any question about a moment or timestamp
    inside a video. Runs Gemini's agentic video understanding and writes the
    results back as temporal detections you can seek to in the App.
category: Curate
resources:
    - TASKS.md
---

# Gemini Agentic Video Understanding

Analyzes video with Gemini's **agentic** video processing, where the model
navigates the timeline itself — searching, scanning and re-sampling only the
segments a prompt actually needs — instead of being handed a fixed 1 FPS strip
of frames. Google reports up to 88% fewer tokens and up to 7% better accuracy
than static sampling for long, heterogeneous content. Those gains are not
automatic — read **Choosing a processing mode** before picking one.

The payoff inside FiftyOne is that timestamped answers come back **structured**,
so they land on the sample as `TemporalDetections` — seekable in the App's video
timeline and filterable like any other label field. A video question stops being
a paragraph of prose and becomes data.

Backed by the `@adonaivera/gemini_vision` plugin. Requires a `GEMINI_API_KEY`.

## Key Directives

**ALWAYS follow these rules — no exceptions:**

### 1. Verify the plugin is installed before promising anything

```
list_operators()
```

Look for `@adonaivera/gemini_vision/video_understanding`. If it is absent:

```
list_plugins(enabled=True)
```

If `@adonaivera/gemini_vision` is missing or disabled, tell the user to install
it (`fiftyone plugins download https://github.com/AdonaiVera/gemini-vision-plugin`)
or ask their FiftyOne Enterprise admin to enable it. Do not fall back to
writing raw Gemini API code unless the user asks for it.

### 2. Pick the task from what the user actually asked

This operator has ten tasks and picking the wrong one wastes a paid API call.
Match the user's intent against the routing table below before calling
anything. When two tasks both fit, say which you chose and why in one line.

### 3. Never guess the media type

```
dataset_summary(name="<name>")
```

Confirm `media_type` is `video`. On an image dataset, route to
`gemini-image-analysis` instead. On a grouped dataset, set the active slice to
the video slice first.

### 4. Bound the cost — this operator bills per video

`max_videos` defaults to **5**. Never raise it without saying what it will
cost the user in videos analyzed. The operator runs on the App's selection and
refuses to run with more than `max_videos` selected. Tell the user which videos
you are about to analyze *before* you run.

### 5. Delegate anything long

Agentic analysis of a long video takes real wall-clock time. For videos over a
couple of minutes, or for more than one video, pass `delegate=True` and then
poll:

```
list_delegated_operations(dataset_name="<name>")
```

### 6. Show the user their results in the App, don't just print them

Structured tasks write a `TemporalDetections` field. After a successful run,
make it visible:

```
set_active_fields(["<label_field>"])
open_sample(sample_id="<id>")
```

The detections appear as bars in the video timeline; clicking one seeks to it.
Say that explicitly — users do not always know temporal detections are seekable.

---

## Task routing

| The user asks | Task | Writes |
| --- | --- | --- |
| "What happens in this video?" | `describe` | `TemporalDetections` (one per beat) |
| "When does the door open / the light turn on / the part get placed?" | `state_change` | `TemporalDetections` |
| "Is anything wrong / unsafe / unusual here?" | `anomaly` | `TemporalDetections` (+ `severity`) |
| "Find every time a red forklift appears" | `needle` | `TemporalDetections` |
| "How many times does X happen?" | `count` | `TemporalDetections` + `<field>_total` |
| "Give me the best frames" / "pull keyframes" | `frame_extract` | `TemporalDetections` + a companion image dataset |
| "Break this into chapters/segments" | `chapters` | `TemporalDetections` |
| "Why did the stack fall over?" | `cause_effect` | `TemporalDetections` (one per causal link) |
| "What is the person at 0:30 holding?" | `qa` | prose |
| "Transcribe this" | `transcript` | prose |

`needle`, `count`, `cause_effect` and `qa` **require** a prompt. The rest have
sensible defaults and take a prompt only to narrow the scope.

Legacy task names still work: `segment` → `chapters`, `extract` → `needle`,
`question` → `qa`.

For the full parameter list, the response schema of each task, and the exact
fields written, read `TASKS.md`.

---

## Operator Reference

| Action | Operator URI | Requires Prompt |
| --- | --- | --- |
| Analyze video (all ten tasks) | `@adonaivera/gemini_vision/video_understanding` | True |

Every operator in this plugin surfaces its parameters before running. Each call
costs money and writes to samples, and the task, prompt, model, processing mode
and label field are all worth a look before execution — so the user always gets
the form and can edit it.

### Parameters

| Parameter | Type | Notes |
| --- | --- | --- |
| `task_type` | enum | See the routing table. Default `describe` |
| `prompt` | str | Required for `needle`, `count`, `cause_effect`, `qa` |
| `max_videos` | int | Default 5. A guard against a runaway bill |

The operator runs on the **App's selection** — there is no `sample_ids`
parameter. Set the selection before calling:

```
set_selected_samples(sample_ids=["<id1>", "<id2>"])
```

With nothing selected, or more than `max_videos` selected, the form blocks
execution rather than warning.
| `model` | str | Default `gemini-3.8-flash`. Agentic video also runs on `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` |
| `processing_mode` | enum | `agentic` (default) or `static` |
| `fps`, `start_offset`, `end_offset` | float | `static` mode only — fixed-rate sampling, optionally clipped |
| `thinking_level` | enum | `low` or `high` (default). `low` is cheaper but loses recall on long video — see Choosing a processing mode |
| `label_field` | str | Where temporal detections land. Defaults per task |
| `extract_frames` | bool | `frame_extract` only. Cuts the chosen frames to disk and into a companion image dataset |
| `source` | enum | `selected` (default) or `youtube` |
| `youtube_url` | str | With `source="youtube"` — a public YouTube URL, no upload needed |

---

## Workflow

### Step 1 — Confirm the target

```
dataset_summary(name="<name>")
```

Check `media_type == "video"` and note the sample count. The operator runs on the App's current selection, so set it before running:

```
set_selected_samples(sample_ids=["<id1>", "<id2>"])
```

### Step 2 — Run the task

```
execute_operator(
    operator_uri="@adonaivera/gemini_vision/video_understanding",
    params={
        "task_type": "needle",
        "prompt": "a forklift carrying a pallet",
        "label_field": "forklift_events",
        "thinking_level": "high",
    },
)
```

For anything long, add `delegate=True` and poll
`list_delegated_operations(dataset_name="<name>")` until it completes.

### Step 3 — Read the result

The operator returns:

- `result` — Markdown: a table of timestamped events, or the prose answer
- `result_field` — **the sample field the answer was written to**. Always report
  this to the user; it is how they and you find the answer again later
- `num_events` — how many temporal detections were written
- `label_field` — the temporal detections field, when any were written
- `processing_calls` — how many times the model reached back into the video.
  This is the agentic loop made visible; static mode always returns 0
- `total_tokens` — what the run cost in tokens

Report `result_field` always, plus `num_events` and `label_field` when
detections were written. Mention `processing_calls` only if they ask how
agentic mode differs.

Every task writes to its own field, so nothing is overwritten by running a
second task on the same video. Eight of the ten return timestamped moments as
`TemporalDetections`; only `qa` and `transcript` are prose-only.

### Step 4 — Make it visible

```
set_active_fields(["<label_field>"])
open_sample(sample_id="<id>")
```

Tell the user the events show as segments on the video timeline and that
clicking one seeks to it.

### Step 5 — Filter or aggregate

Temporal detections behave like any label field:

```
count_values("<label_field>.detections.label")
```

To keep only videos where something was found:

```
set_view(view_stages=[{"_cls": "Match", "filter": {"$expr": {"$gt": [{"$size": {"$ifNull": ["$<label_field>.detections", []]}}, 0]}}}])
```

---

## Common Use Cases

### "Find every time X happens in these videos"

`needle`, with the target described in the prompt. Set a descriptive
`label_field` (`forklift_events`, not the default) so several searches can
coexist on the same dataset. Then `count_values` on the label to see the
distribution across the dataset.

### "How many times does the door open?"

`count`. The result carries a `<label_field>_total` integer field per sample
plus one temporal detection per occurrence, so the user can verify the count by
seeking to each one rather than trusting a number. Always surface the model's
`counting_rule` — it states what it treated as one occurrence, which is where
counting disagreements actually come from.

### "Is there anything unsafe in this footage?"

`anomaly`. Each detection carries a `severity` of low/medium/high. Sort the
user's attention to high first. An empty result is a real answer — the task is
instructed not to invent anomalies — so say "nothing flagged" rather than
re-running with a louder prompt.

### "Pull out the good frames from these videos"

`frame_extract` with `extract_frames=True`. Frames are written beside the video
under `gemini_frames/` and added to a companion image dataset named
`<dataset>-gemini-frames`, each sample carrying `timestamp`, `frame_number`,
`source_sample_id` and Gemini's reason for choosing it. Tell the user the new
dataset name — that is the deliverable, not the timestamps.

### "Why did this happen?"

`cause_effect`, with the outcome in the prompt. The answer is prose with a
causal chain and cited timestamps; it separates what was observed from what was
inferred. There is no label field for this task.

### "Analyze this YouTube video"

```
params={"source": "youtube", "youtube_url": "https://...", "task_type": "chapters"}
```

No download and no upload. Results are returned but **not** written to a
sample, since there is no sample — report the Markdown to the user.

---

## Choosing a processing mode

**Default to `static` with `fps=0.5`. Escalate to `agentic` only when a
uniform pass demonstrably misses things.** Agentic is not automatically
cheaper — the loop spends tokens deciding where to look before it looks.

Decision rules, in order:

1. User knows which stretch matters → `static` with `fps` plus
   `start_offset`/`end_offset`.
2. Target is visually distinct and the footage is uniform → `static`,
   `fps=0.5`.
3. A static pass returned nothing and the user is confident the target is
   there → re-run `agentic` with `thinking_level="high"`.
4. Long, heterogeneous footage where most of the timeline is irrelevant →
   `agentic`. This is the case Google's efficiency figures describe.

**Never pair `thinking_level="low"` with `needle`, `count` or `anomaly` on
video longer than a few minutes.** It cuts cost but also cuts recall. Use
`low` only for `describe`, `chapters` and `transcript`, where a miss is cheap.

Reference measurements — one target hidden once in 30 minutes of repeated
street footage, `gemini-3.8-flash`, same prompt. Quote these only if the user
asks about cost, and say the footage was deliberately repetitive, which
favours uniform sampling:

| Mode | Tokens | Lookups | Found it |
| --- | --- | --- | --- |
| agentic, high | 553,560 | 18 | yes |
| agentic, low | 274,183 | 16 | **no** |
| static, fps=1 | 119,184 | 0 | yes |
| static, fps=0.5 | 61,053 | 0 | yes |

Uploads: ≤20MB inline, larger goes through the Files API automatically up to
2GB (≈1 minute for 163MB), and the URI is reusable for 48 hours.

---

## Troubleshooting

**"Operator not found"**

```
list_plugins(enabled=True)
```

Install with
`fiftyone plugins download https://github.com/AdonaiVera/gemini-vision-plugin`,
or have an admin enable `@adonaivera/gemini_vision`.

**"No Gemini API Key"**

`GEMINI_API_KEY` is not set in the environment the plugin runs in. In
Enterprise, it is set as a plugin secret; locally it is an environment variable.

**Result has `num_events: 0` but prose mentions events**

The model returned an answer that didn't match the response schema. Re-run once
with `thinking_level="high"`. If it recurs, the task may be a poor fit — a
question with no timestamped answer belongs in `qa`, not `needle`.

**Timestamps look shifted**

The plugin converts timestamps to frame supports using the sample's video
metadata. If `metadata` was stale, recompute it and re-run:

```
execute_operator(operator_uri="@voxel51/utils/compute_metadata")
```

**Analysis is very slow**

Expected for agentic mode on long video. Use `delegate=True`, or drop to
`processing_mode="static"` with an `fps` and a clipped range.

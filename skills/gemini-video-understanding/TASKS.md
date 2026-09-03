# Video task reference

Every task below runs through
`@adonaivera/gemini_vision/video_understanding`. The tasks split into two
kinds:

- **Structured tasks** ask Gemini for a JSON array of timestamped events and
  write them to the sample as `fo.TemporalDetections`.
- **Prose tasks** return Markdown and write it to a single string field.

Both kinds always write the readable answer to a summary field, so nothing is
lost when a structured parse comes back thin.

---

## Structured tasks

Each detection carries `label`, `support` (the `[first, last]` frame numbers),
`confidence`, `start_time` and `end_time` in seconds, and a `description`.
`anomaly` adds `severity`.

### `state_change`

Every moment where something transitions between discrete states — open to
closed, off to on, stopped to moving, absent to present. The model makes one
pass, then re-inspects at a finer frame rate only where it has positive
evidence of a transition.

- Prompt: optional, to narrow the subject
- Default field: `gemini_state_changes`
- Labels come back shaped `<subject>_<from>_to_<to>`

### `anomaly`

Establishes what normal looks like in the footage, then flags what breaks the
pattern, violates procedure, or is unsafe. Instructed to return an empty list
rather than invent anomalies, so an empty result is a real answer.

- Prompt: optional, to describe what "normal" means here
- Default field: `gemini_anomalies`
- Extra field per detection: `severity` ∈ `low` | `medium` | `high`

### `needle`

Exhaustive search of the whole timeline for a target the user describes. Does
not stop at the first match.

- Prompt: **required** — describe the target
- Default field: `gemini_matches`
- Response also carries `found` (bool) and a `summary`

### `count`

Counts occurrences, returning one detection per individual occurrence so the
total is verifiable by seeking to each one.

- Prompt: **required** — what to count
- Default field: `gemini_occurrences`
- Also writes `<label_field>_total` (int) to the sample
- Response carries `counting_rule` — what the model treated as one occurrence.
  Always surface it; counting disagreements almost always come from this

### `frame_extract`

Picks individual frames rather than spans; each detection's start and end are
the same instant, and `confidence` doubles as a quality score.

- Prompt: optional, e.g. "one sharp frame per person who appears"
- Default field: `gemini_keyframes`
- With `extract_frames=True` (the default), frames are cut with ffmpeg to
  `<video dir>/gemini_frames/<video name>/` and added to a companion image
  dataset `<dataset>-gemini-frames`, one sample per frame carrying
  `source_sample_id`, `source_filepath`, `frame_number`, `timestamp`,
  `timestamp_str`, `gemini_reason` and a `gemini_label` classification
- A video dataset cannot hold image samples, which is why the frames go to a
  separate dataset rather than back into the source

### `describe`

Breaks the timeline into consecutive moments covering the whole video, one per
distinct beat of action, plus a two-or-three sentence summary of the setting
and subjects.

- Prompt: optional, to steer what the description attends to
- Default field: `gemini_description`; summary in `gemini_description_summary`

### `cause_effect`

Identifies the outcome the user asks about, then works backwards to the events
that caused it — one detection per link in the chain, ordered earliest first.
The summary separates what was observed from what was inferred, and says
plainly when the video lacks the evidence.

- Prompt: **required** — name the outcome to explain
- Default field: `gemini_cause_effect`

### `chapters`

Contiguous, titled segments covering the full duration with no gaps or
overlap, cut on real topic and scene boundaries.

- Prompt: optional, e.g. "one chapter per surgical step"
- Default field: `gemini_chapters`

---

## Prose-only tasks

`qa` and `transcript` return Markdown with no detections, written to
`gemini_qa` and `gemini_transcript`. Every other task returns timestamped
moments. The operator reports the field it wrote in `result_field`.

### `qa`

Free-form question answering, grounded in what is visible or audible, with
timestamps cited. Says so when the video does not answer the question.

- Prompt: **required**

### `transcript`

Timestamped transcript with speaker attribution and non-speech audio noted in
brackets.

---

## Processing modes

### `agentic` (default)

`processing: "agentic"` on the video input. The model drives its own traversal
— pulling transcript, frames or audio for a segment only when its reasoning
calls for it. The `processing_calls` field in the operator's output counts how
many times it did so.

Cheaper than static sampling on long video, because most of the timeline is
never materialized. On a short clip it can cost *more* than a single static
pass, because the loop has a floor.

### `static`

`processing: {"type": "static", "fps": ..., "start_offset": ..., "end_offset": ...}`.
Fixed-rate frame sampling, roughly 100 tokens per second of video at low
resolution. Predictable and cheap. The right choice when the user already knows
which stretch matters.

Offsets are in seconds; `fps: 0.5` samples one frame every two seconds.

---

## Models

Agentic video processing runs on:

| Model | Notes |
| --- | --- |
| `gemini-3.8-flash` | Default |
| `gemini-3.7-flash` | Best quality/cost balance per Google's announcement |
| `gemini-3.6-flash` | |
| `gemini-3.5-flash-lite` | Cheapest |

The operator only offers models that both support agentic video and are
visible to the caller's API key.

---

## Video sources

| Source | Path |
| --- | --- |
| ≤ 20MB local file | Sent inline as base64 |
| > 20MB local file | Uploaded via the Files API, then referenced by URI; the upload is reusable for 48 hours |
| > 2GB | Rejected — above the Files API limit |
| Public YouTube URL | `source="youtube"`, passed straight through with no download |

YouTube results are returned to the caller but not written to a sample, since
there is no sample to write to.

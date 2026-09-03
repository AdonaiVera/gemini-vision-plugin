"""Agentic video understanding for the Gemini Vision plugin.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import base64
import json
import logging
import os
import re
import time

import requests

import fiftyone as fo

import fiftyone.core.storage as fos

from . import gemini_media as gm


logger = logging.getLogger(__name__)


API_BASE = gm.API_BASE
INTERACTIONS_URL = gm.INTERACTIONS_URL
FILES_UPLOAD_URL = gm.FILES_UPLOAD_URL

AGENTIC_VIDEO_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

DEFAULT_VIDEO_MODEL = "gemini-3.8-flash"

INLINE_LIMIT_MB = 20

MAX_UPLOAD_MB = 2048

VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "mpeg": "video/mpeg",
    "mpg": "video/mpg",
    "mov": "video/mov",
    "avi": "video/avi",
    "flv": "video/x-flv",
    "webm": "video/webm",
    "wmv": "video/wmv",
    "3gp": "video/3gpp",
}


class GeminiVideoError(Exception):
    """Raised when a video request cannot be completed."""


_TS_RE = re.compile(
    r"^\s*(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)\s*$"
)


def parse_timestamp(value):
    """Parses a Gemini timestamp into seconds.

    Args:
        value: the timestamp to parse

    Returns:
        the timestamp in seconds, or ``None`` if it can't be parsed
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    match = _TS_RE.match(text)
    if match is not None:
        hours = float(match.group("h") or 0)
        return hours * 3600 + float(match.group("m")) * 60 + float(match.group("s"))

    try:
        return float(text.rstrip("sS"))
    except ValueError:
        return None


def format_timestamp(seconds):
    """Formats seconds as ``MM:SS`` (or ``H:MM:SS`` past an hour)."""
    if seconds is None:
        return ""

    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def _event_schema(item_props, required, array_key, extra_props=None):
    """Builds a response schema for a list of timestamped events."""
    properties = {
        array_key: {
            "type": "array",
            "items": {
                "type": "object",
                "properties": item_props,
                "required": required,
            },
        }
    }
    properties.update(extra_props or {})
    return {
        "type": "object",
        "properties": properties,
        "required": [array_key],
    }


_SPAN_PROPS = {
    "start": {
        "type": "string",
        "description": "Start timestamp in MM:SS or H:MM:SS format",
    },
    "end": {
        "type": "string",
        "description": "End timestamp in MM:SS or H:MM:SS format",
    },
    "label": {
        "type": "string",
        "description": "Short snake_case label, 1-3 words",
    },
    "description": {"type": "string"},
    "confidence": {
        "type": "number",
        "description": "Confidence between 0 and 1",
    },
}

_SPAN_REQUIRED = ["start", "end", "label", "confidence"]


VIDEO_TASKS = {
    "describe": {
        "label": "Describe",
        "description": "A summary plus every beat of the video as a timestamped moment",
        "theme": "Video analysis",
        "prompt_required": False,
        "default_prompt": "Describe what happens in this video.",
        "instruction": (
            "Describe the video in two parts. First a summary: the setting, the "
            "subjects, and what the video is of, in two or three sentences. "
            "Then break the timeline into consecutive moments that together "
            "cover the whole video with no gaps and no overlap — one per "
            "distinct beat of action, not a fixed interval. Give each moment "
            "its timestamp span, a short snake_case label naming what happens "
            "in it, and a concrete one-line description of the action."
        ),
        "schema": _event_schema(
            _SPAN_PROPS,
            _SPAN_REQUIRED,
            "events",
            extra_props={"summary": {"type": "string"}},
        ),
        "events_key": "events",
        "default_field": "gemini_description",
    },
    "state_change": {
        "label": "State changes",
        "description": (
            "Every point where something in the scene changes state — the "
            "sub-second transitions static frame sampling misses"
        ),
        "theme": "Video-based QA",
        "prompt_required": False,
        "default_prompt": (
            "Find every state change in this video."
        ),
        "instruction": (
            "Identify every STATE CHANGE: a moment where an object, person or "
            "the scene transitions from one discrete state to another (open to "
            "closed, off to on, empty to full, stopped to moving, absent to "
            "present). Make one pass over the timeline, then re-inspect at a "
            "finer frame rate only those windows where you have positive "
            "evidence a transition occurs — do not re-scan a window you have "
            "already resolved. For each change, give the tightest timestamp "
            "span that contains the transition, a short snake_case label of "
            "the form <subject>_<from>_to_<to>, and a one-line description "
            "naming the before and after state. Do not report steady state as "
            "a change."
        ),
        "schema": _event_schema(_SPAN_PROPS, _SPAN_REQUIRED, "events"),
        "events_key": "events",
        "default_field": "gemini_state_changes",
    },
    "anomaly": {
        "label": "Anomaly detection",
        "description": "Unexpected, unsafe or out-of-pattern moments",
        "theme": "Video-based QA",
        "prompt_required": False,
        "default_prompt": "Find anything anomalous or unsafe in this video.",
        "instruction": (
            "Establish what NORMAL looks like in this video, then flag every "
            "ANOMALY: an event that breaks the established pattern, violates "
            "expected procedure, or presents a safety risk. Re-inspect a "
            "window at a higher frame rate once, only when rapid motion makes "
            "a first pass inconclusive. "
            "For each anomaly give the tightest timestamp span, a short "
            "snake_case label, a description stating what was expected versus "
            "what occurred, and a severity of low, medium or high. Return an "
            "empty list if the video is entirely nominal — do not invent "
            "anomalies."
        ),
        "schema": _event_schema(
            dict(
                _SPAN_PROPS,
                severity={"type": "string", "enum": ["low", "medium", "high"]},
            ),
            _SPAN_REQUIRED + ["severity"],
            "events",
        ),
        "events_key": "events",
        "default_field": "gemini_anomalies",
    },
    "needle": {
        "label": "Needle in a haystack",
        "description": "Locate every occurrence of a specific thing you describe",
        "theme": "Needle-in-a-Haystack",
        "prompt_required": True,
        "default_prompt": "",
        "instruction": (
            "Search the entire video for every occurrence of the target the "
            "user describes. Be exhaustive — scan the full timeline, do not "
            "stop at the first match, and do not skip regions. For each "
            "occurrence give the tightest timestamp span that contains it, a "
            "short snake_case label, and a description of what makes it a "
            "match. If the target never appears, return an empty list and say "
            "so in the summary."
        ),
        "schema": _event_schema(
            _SPAN_PROPS,
            _SPAN_REQUIRED,
            "events",
            extra_props={
                "found": {"type": "boolean"},
                "summary": {"type": "string"},
            },
        ),
        "events_key": "events",
        "default_field": "gemini_matches",
    },
    "count": {
        "label": "Counting",
        "description": "Count events or repetitions, with one span per occurrence",
        "theme": "Counting",
        "prompt_required": True,
        "default_prompt": "",
        "instruction": (
            "Count the occurrences the user asks about. Track each occurrence "
            "across the timeline so repeated motion is not double counted and "
            "distinct objects are not merged. Return one entry per individual "
            "occurrence with its timestamp span, plus a total that equals the "
            "number of entries. State your counting rule — what you treated as "
            "one occurrence — in the summary."
        ),
        "schema": _event_schema(
            _SPAN_PROPS,
            _SPAN_REQUIRED,
            "events",
            extra_props={
                "total": {"type": "integer"},
                "counting_rule": {"type": "string"},
                "summary": {"type": "string"},
            },
        ),
        "events_key": "events",
        "default_field": "gemini_occurrences",
    },
    "frame_extract": {
        "label": "Frame extraction",
        "description": (
            "Pick the exact frames worth keeping, and optionally cut them out "
            "of the video as images"
        ),
        "theme": "Editing",
        "prompt_required": False,
        "default_prompt": (
            "Select the most representative frames of this video."
        ),
        "instruction": (
            "Act as a video editor. Choose the individual FRAMES that best "
            "satisfy the user's request — a keyframe per distinct shot or "
            "action unless they ask for something else. Pick the precise "
            "instant, not a rough neighbourhood: prefer the frame where the "
            "subject is sharpest and least occluded. Return each pick as a "
            "timestamp span whose start and end are the same instant, a short "
            "snake_case label, a description of why the frame was chosen, and "
            "a confidence that doubles as a quality score."
        ),
        "schema": _event_schema(_SPAN_PROPS, _SPAN_REQUIRED, "events"),
        "events_key": "events",
        "default_field": "gemini_keyframes",
    },
    "chapters": {
        "label": "Chapters",
        "description": "Segment the video into contiguous, titled chapters",
        "theme": "Editing",
        "prompt_required": False,
        "default_prompt": "Break this video into chapters.",
        "instruction": (
            "Segment the video into contiguous chapters that together cover "
            "the full duration with no gaps and no overlap. Cut on real "
            "topic or scene boundaries, not on a fixed interval. Give each "
            "chapter a timestamp span, a short snake_case label, and a "
            "one-sentence description of what it covers."
        ),
        "schema": _event_schema(_SPAN_PROPS, _SPAN_REQUIRED, "events"),
        "events_key": "events",
        "default_field": "gemini_chapters",
    },
    "cause_effect": {
        "label": "Cause and effect",
        "description": "Trace why something happened, as a chain of timestamped links",
        "theme": "Video Analysis / Q&A",
        "prompt_required": True,
        "default_prompt": "",
        "instruction": (
            "Answer as a causal analysis. Identify the outcome the user asks "
            "about, then work backwards to the events that caused it. Return "
            "one entry per link in the chain, ordered earliest first, each with "
            "the timestamp span where that link is visible, a short snake_case "
            "label, and a description saying what happened and how it "
            "contributed. In the summary, state the outcome, separate what you "
            "observed from what you inferred, and say plainly if the video "
            "lacks the evidence to establish a cause."
        ),
        "schema": _event_schema(
            _SPAN_PROPS,
            _SPAN_REQUIRED,
            "events",
            extra_props={"summary": {"type": "string"}},
        ),
        "events_key": "events",
        "default_field": "gemini_cause_effect",
    },
    "qa": {
        "label": "Question answering",
        "description": "Ask anything about the video",
        "theme": "Video Analysis / Q&A",
        "prompt_required": True,
        "default_prompt": "",
        "instruction": (
            "Answer the user's question about this video. Ground every claim "
            "in what is actually visible or audible and cite MM:SS timestamps. "
            "If the video does not answer the question, say so instead of "
            "guessing. Answer in Markdown."
        ),
        "schema": None,
        "events_key": None,
        "default_field": "gemini_qa",
    },
    "transcript": {
        "label": "Transcript",
        "description": "Timestamped transcript of the spoken audio",
        "theme": "Video analysis",
        "prompt_required": False,
        "default_prompt": "Transcribe the speech in this video.",
        "instruction": (
            "Transcribe the spoken audio with a MM:SS timestamp at the start "
            "of each line, attributing speakers as Speaker 1, Speaker 2, and "
            "so on when more than one voice is present. Note significant "
            "non-speech audio in square brackets. If the video has no speech, "
            "say so."
        ),
        "schema": None,
        "events_key": None,
        "default_field": "gemini_transcript",
    },
}

LEGACY_TASK_ALIASES = {
    "segment": "chapters",
    "extract": "needle",
    "question": "qa",
}


def resolve_task(task_type):
    """Resolves a task name, following legacy aliases.

    Args:
        task_type: a task name

    Returns:
        a ``(name, task)`` tuple

    Raises:
        GeminiVideoError: if the task is unknown
    """
    name = LEGACY_TASK_ALIASES.get(task_type, task_type)
    task = VIDEO_TASKS.get(name)
    if task is None:
        raise GeminiVideoError(
            f"Unknown video task '{task_type}'. Available tasks: "
            + ", ".join(sorted(VIDEO_TASKS))
        )

    return name, task


def get_video_size_mb(video_path):
    """Returns the size of a video on disk, in MB."""
    return os.path.getsize(video_path) / (1024 * 1024)


def get_mime_type(video_path):
    """Returns the MIME type for a video path."""
    ext = os.path.splitext(video_path)[1].lstrip(".").lower()
    return VIDEO_MIME_TYPES.get(ext, "video/mp4")


def is_youtube_url(value):
    """Returns whether ``value`` is a YouTube URL."""
    if not isinstance(value, str):
        return False

    return bool(
        re.match(r"^https?://(www\.)?(youtube\.com/watch\?|youtu\.be/)", value.strip())
    )


def upload_video(video_path, api_key, timeout=600):
    """Uploads a video via the Files API and waits for it to become active.

    Args:
        video_path: the path to the video
        api_key: a Gemini API key
        timeout (600): the maximum number of seconds to wait for processing

    Returns:
        the file URI to reference in a request

    Raises:
        GeminiVideoError: if the upload or processing fails
    """
    size = os.path.getsize(video_path)
    mime_type = get_mime_type(video_path)

    resp = requests.post(
        FILES_UPLOAD_URL,
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": os.path.basename(video_path)}},
        timeout=60,
    )
    upload_url = resp.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise GeminiVideoError(
            f"Failed to start upload for '{video_path}': {resp.text[:300]}"
        )

    with open(video_path, "rb") as f:
        resp = requests.post(
            upload_url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=f,
            timeout=timeout,
        )

    try:
        info = resp.json()
    except ValueError:
        raise GeminiVideoError(f"Upload failed: {resp.text[:300]}")

    file_info = info.get("file", info)
    name = file_info.get("name")
    uri = file_info.get("uri")
    if not uri:
        raise GeminiVideoError(f"Upload returned no file URI: {str(info)[:300]}")

    state = file_info.get("state")
    deadline = time.time() + timeout
    while state == "PROCESSING" and time.time() < deadline:
        time.sleep(2)
        poll = requests.get(
            f"{API_BASE}/v1beta/{name}",
            headers={"x-goog-api-key": api_key},
            timeout=30,
        )
        state = poll.json().get("state")

    if state != "ACTIVE":
        raise GeminiVideoError(
            f"Uploaded video did not become ACTIVE (state: {state})"
        )

    return uri


def build_video_part(video_source, api_key, processing, uploaded_uri=None):
    """Builds the video element of an Interactions API request.

    Args:
        video_source: a local path or a YouTube URL
        api_key: a Gemini API key
        processing: the ``processing`` value for the part
        uploaded_uri (None): a previously uploaded file URI to reuse

    Returns:
        a ``(part, file_uri)`` tuple, where ``file_uri`` is ``None`` for
        inlined videos
    """
    if is_youtube_url(video_source):
        return {
            "type": "video",
            "uri": video_source.strip(),
            "processing": processing,
        }, None

    if uploaded_uri:
        return {
            "type": "video",
            "uri": uploaded_uri,
            "mime_type": get_mime_type(video_source),
            "processing": processing,
        }, uploaded_uri

    if not os.path.isfile(video_source):
        raise GeminiVideoError(f"Video not found: {video_source}")

    size_mb = get_video_size_mb(video_source)
    if size_mb > MAX_UPLOAD_MB:
        raise GeminiVideoError(
            f"Video is {size_mb:.1f}MB, above the {MAX_UPLOAD_MB}MB Files API limit"
        )

    mime_type = get_mime_type(video_source)

    if size_mb <= INLINE_LIMIT_MB:
        with open(video_source, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        return {
            "type": "video",
            "data": data,
            "mime_type": mime_type,
            "processing": processing,
        }, None

    file_uri = upload_video(video_source, api_key)
    return {
        "type": "video",
        "uri": file_uri,
        "mime_type": mime_type,
        "processing": processing,
    }, file_uri


def build_processing(mode="agentic", fps=None, start_offset=None, end_offset=None):
    """Builds the ``processing`` value for a video part.

    Args:
        mode ("agentic"): ``"agentic"`` to let the model navigate the timeline
            itself, or ``"static"`` for fixed-rate frame sampling
        fps (None): frames per second, static mode only
        start_offset (None): clip start in seconds, static mode only
        end_offset (None): clip end in seconds, static mode only

    Returns:
        the ``processing`` value
    """
    if mode == "agentic":
        return "agentic"

    processing = {"type": "static"}
    if fps:
        processing["fps"] = float(fps)
    if start_offset is not None:
        processing["start_offset"] = int(start_offset)
    if end_offset is not None:
        processing["end_offset"] = int(end_offset)

    return processing


def extract_output_text(response):
    """Extracts the model's final text from an Interactions API response."""
    text = response.get("output_text")
    if text:
        return text

    chunks = []
    for step in response.get("steps", []):
        if step.get("type") != "model_output":
            continue

        for item in step.get("content", []):
            if item.get("text"):
                chunks.append(item["text"])

    return "".join(chunks)


def count_processing_calls(response):
    """Returns how many times the model reached back into the video."""
    return sum(
        1 for s in response.get("steps", []) if s.get("type") == "processing_call"
    )


def run_interaction(
    video_source,
    prompt,
    api_key,
    task_type="describe",
    model=DEFAULT_VIDEO_MODEL,
    processing_mode="agentic",
    fps=None,
    start_offset=None,
    end_offset=None,
    thinking_level=None,
    max_output_tokens=None,
    uploaded_uri=None,
    timeout=900,
):
    """Runs one agentic video analysis.

    Args:
        video_source: a local video path or a YouTube URL
        prompt: the user's prompt
        api_key: a Gemini API key
        task_type ("describe"): a key of :data:`VIDEO_TASKS`
        model: the Gemini model to use
        processing_mode ("agentic"): ``"agentic"`` or ``"static"``
        fps (None): frames per second, static mode only
        start_offset (None): clip start in seconds, static mode only
        end_offset (None): clip end in seconds, static mode only
        thinking_level (None): ``"low"`` or ``"high"``
        max_output_tokens (None): a cap on the response length
        uploaded_uri (None): a previously uploaded file URI to reuse
        timeout (900): the request timeout, in seconds

    Returns:
        a dict with ``text``, ``data``, ``usage``, ``processing_calls``,
        ``file_uri`` and ``task`` keys

    Raises:
        GeminiVideoError: if the request fails
    """
    name, task = resolve_task(task_type)

    prompt = (prompt or "").strip() or task["default_prompt"]
    if task["prompt_required"] and not prompt:
        raise GeminiVideoError(f"The '{name}' task requires a prompt")

    processing = build_processing(
        processing_mode,
        fps=fps,
        start_offset=start_offset,
        end_offset=end_offset,
    )
    video_part, file_uri = build_video_part(
        video_source, api_key, processing, uploaded_uri=uploaded_uri
    )

    payload = {
        "model": model,
        "input": [video_part, {"type": "text", "text": prompt}],
        "system_instruction": task["instruction"],
    }

    if task["schema"] is not None:
        payload["response_format"] = task["schema"]

    generation_config = {}
    if thinking_level:
        generation_config["thinking_level"] = thinking_level
    if max_output_tokens:
        generation_config["max_output_tokens"] = int(max_output_tokens)
    if generation_config:
        payload["generation_config"] = generation_config

    resp = requests.post(
        INTERACTIONS_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )

    try:
        content = resp.json()
    except ValueError:
        raise GeminiVideoError(f"Gemini returned a non-JSON response: {resp.text[:300]}")

    if "error" in content:
        err = content["error"]
        raise GeminiVideoError(err.get("message") or str(err))

    text = extract_output_text(content)
    if not text:
        raise GeminiVideoError(
            f"Gemini returned no output (status: {content.get('status')})"
        )

    data = None
    if task["schema"] is not None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Gemini returned unparseable JSON for task '%s'", name)

    return {
        "task": name,
        "prompt": prompt,
        "model": model,
        "text": text,
        "data": data,
        "usage": gm.summarize_usage(content),
        "processing_calls": count_processing_calls(content),
        "file_uri": file_uri,
    }


def events_to_temporal_detections(events, sample):
    """Converts Gemini's timestamped events into temporal detections.

    Args:
        events: the list of event dicts returned by Gemini
        sample: the video :class:`fiftyone.core.sample.Sample`

    Returns:
        a ``(detections, skipped)`` tuple
    """
    if sample.metadata is None or sample.metadata.duration is None:
        sample.compute_metadata()

    duration = sample.metadata.duration
    detections = []
    skipped = 0

    for event in events or []:
        if not isinstance(event, dict):
            skipped += 1
            continue

        start = parse_timestamp(event.get("start"))
        end = parse_timestamp(event.get("end"))
        if start is None:
            skipped += 1
            continue

        if end is None or end < start:
            end = start

        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))

        label = str(event.get("label") or "event").strip() or "event"

        confidence = event.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        detection = fo.TemporalDetection.from_timestamps(
            [start, end],
            sample=sample,
            label=label,
            confidence=confidence,
        )

        for key in ("description", "severity"):
            if event.get(key):
                detection[key] = str(event[key])

        detection["start_time"] = start
        detection["end_time"] = end

        detections.append(detection)

    return fo.TemporalDetections(detections=detections), skipped


def save_result(sample, result, label_field=None, summary_field=None):
    """Writes an analysis result onto a video sample.

    Args:
        sample: the video :class:`fiftyone.core.sample.Sample`
        result: a :func:`run_interaction` result
        label_field (None): the field for temporal detections. Defaults to the
            task's ``default_field``
        summary_field (None): the field for the prose answer. Defaults to
            ``"<label_field>_summary"``, or ``"gemini_video_analysis"``

    Returns:
        a dict describing what was written
    """
    _, task = resolve_task(result["task"])

    structured = bool(task["events_key"])

    label_field = label_field or task["default_field"]
    if summary_field is None:
        summary_field = f"{label_field}_summary" if structured else label_field

    written = {"summary_field": summary_field, "num_events": 0, "skipped": 0}

    events = []
    data = result.get("data")
    if data and task["events_key"]:
        events = data.get(task["events_key"]) or []

    if structured and label_field and events:
        detections, skipped = events_to_temporal_detections(events, sample)
        sample[label_field] = detections
        written["label_field"] = label_field
        written["num_events"] = len(detections.detections)
        written["skipped"] = skipped

    summary = result["text"]
    if data:
        parts = []
        if data.get("summary"):
            parts.append(str(data["summary"]))
        if data.get("counting_rule"):
            parts.append(f"**Counting rule:** {data['counting_rule']}")
        if data.get("total") is not None:
            parts.append(f"**Total:** {data['total']}")
        parts.append(format_events_markdown(events))
        summary = "\n\n".join(p for p in parts if p)

    sample[summary_field] = summary

    if data and data.get("total") is not None:
        count_field = f"{label_field or 'gemini_video'}_total"
        sample[count_field] = int(data["total"])
        written["count_field"] = count_field

    sample.save()
    return written


def format_events_markdown(events):
    """Renders timestamped events as a Markdown table."""
    if not events:
        return "_No events found._"

    has_severity = any(e.get("severity") for e in events if isinstance(e, dict))

    header = "| Start | End | Label | Description |"
    divider = "| --- | --- | --- | --- |"
    if has_severity:
        header = "| Start | End | Label | Severity | Description |"
        divider = "| --- | --- | --- | --- | --- |"

    rows = [header, divider]
    for event in events:
        if not isinstance(event, dict):
            continue

        start = format_timestamp(parse_timestamp(event.get("start")))
        end = format_timestamp(parse_timestamp(event.get("end")))
        label = str(event.get("label") or "").replace("|", "\\|")
        desc = str(event.get("description") or "").replace("|", "\\|")

        if has_severity:
            severity = str(event.get("severity") or "").replace("|", "\\|")
            rows.append(f"| {start} | {end} | `{label}` | {severity} | {desc} |")
        else:
            rows.append(f"| {start} | {end} | `{label}` | {desc} |")

    return "\n".join(rows)


def extract_frames(sample, events, output_dir=None, dataset_name=None):
    """Cuts the frames Gemini selected out of the video as images.

    Args:
        sample: the video :class:`fiftyone.core.sample.Sample`
        events: the event dicts Gemini returned
        output_dir (None): where to write the frames. Defaults to a
            ``gemini_frames`` directory beside the video
        dataset_name (None): the companion dataset to add the frames to.
            Defaults to ``"<dataset>-gemini-frames"``

    Returns:
        a dict with ``dataset``, ``num_frames`` and ``output_dir`` keys
    """
    import fiftyone.utils.video as fouv

    source = gm.localize(sample)

    if sample.metadata is None or sample.metadata.total_frame_count is None:
        sample.compute_metadata()

    metadata = sample.metadata
    duration = metadata.duration
    total_frames = metadata.total_frame_count

    frame_numbers = []
    frame_events = []
    for event in events or []:
        if not isinstance(event, dict):
            continue

        seconds = parse_timestamp(event.get("start"))
        if seconds is None:
            continue

        seconds = max(0.0, min(seconds, duration))
        frame_number = max(
            1, min(total_frames, int(round(seconds / duration * total_frames)))
        )
        if frame_number in frame_numbers:
            continue

        frame_numbers.append(frame_number)
        frame_events.append((frame_number, seconds, event))

    if not frame_numbers:
        return {"dataset": None, "num_frames": 0, "output_dir": None}

    stem = os.path.splitext(os.path.basename(sample.filepath))[0]
    destination = output_dir or gm.media_root(sample._dataset, sample)

    import tempfile

    staging = os.path.join(
        tempfile.mkdtemp(prefix="gemini_frames_"), stem
    )
    os.makedirs(staging, exist_ok=True)
    output_patt = os.path.join(staging, "frame_%06d.jpg")

    fouv.sample_video(
        source,
        output_patt,
        frames=sorted(frame_numbers),
        original_frame_numbers=True,
    )

    if dataset_name is None:
        source = sample._dataset.name if sample._dataset is not None else "gemini"
        dataset_name = f"{source}-gemini-frames"

    if fo.dataset_exists(dataset_name):
        frames_dataset = fo.load_dataset(dataset_name)
    else:
        frames_dataset = fo.Dataset(dataset_name, persistent=True)

    samples = []
    for frame_number, seconds, event in frame_events:
        staged = output_patt % frame_number
        if not os.path.isfile(staged):
            continue

        filepath = gm.output_path(
            destination,
            f"{stem}_frame_{frame_number:06d}",
            "jpg",
            subdir=gm.FRAMES_SUBDIR,
        )
        gm.upload_local(staged, filepath)

        frame_sample = fo.Sample(filepath=filepath)
        frame_sample["source_sample_id"] = str(sample.id)
        frame_sample["source_filepath"] = sample.filepath
        frame_sample["frame_number"] = frame_number
        frame_sample["timestamp"] = seconds
        frame_sample["timestamp_str"] = format_timestamp(seconds)
        frame_sample["gemini_reason"] = str(event.get("description") or "")
        frame_sample["gemini_label"] = fo.Classification(
            label=str(event.get("label") or "keyframe"),
            confidence=(
                float(event["confidence"])
                if isinstance(event.get("confidence"), (int, float))
                else None
            ),
        )
        samples.append(frame_sample)

    frames_dataset.add_samples(samples)

    return {
        "dataset": dataset_name,
        "num_frames": len(samples),
        "output_dir": fos.join(destination, gm.FRAMES_SUBDIR),
    }

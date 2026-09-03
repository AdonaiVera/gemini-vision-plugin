"""Media access for open-source and Enterprise FiftyOne.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import uuid
from datetime import datetime, timezone

import fiftyone.core.storage as fos


API_BASE = "https://generativelanguage.googleapis.com"
INTERACTIONS_URL = f"{API_BASE}/v1beta/interactions"
FILES_UPLOAD_URL = f"{API_BASE}/upload/v1beta/files"

REQUEST_TIMEOUT = 900

GENERATED_SUBDIR = "gemini_generated"
FRAMES_SUBDIR = "gemini_frames"


class MediaError(Exception):
    """Raised when media cannot be read or written."""


def localize(sample, download=True):
    """Returns a local path to a sample's media.

    Args:
        sample: a :class:`fiftyone.core.sample.Sample`
        download (True): whether to fetch remote media that is not yet cached

    Returns:
        a local filepath

    Raises:
        MediaError: if the media cannot be made local
    """
    filepath = sample.filepath

    get_local_path = getattr(sample, "get_local_path", None)
    if callable(get_local_path):
        try:
            local = get_local_path(download=download, skip_failures=False)
        except TypeError:
            local = get_local_path()
        except Exception as e:
            raise MediaError(
                f"Could not retrieve '{filepath}': {e}. The object may have "
                "been moved or deleted, or this deployment may not have read "
                "access to that location."
            )

        if not local or not os.path.isfile(local):
            raise MediaError(f"Media could not be localized: {filepath}")

        return local

    if not os.path.isfile(filepath):
        raise MediaError(
            f"Media not found: {filepath}. Open-source FiftyOne reads local "
            "files only; cloud-backed media requires FiftyOne Enterprise."
        )

    return filepath


def is_ready(sample):
    """Whether :func:`localize` would avoid a download, when knowable."""
    for attr in ("is_local_or_cached", "is_local"):
        value = getattr(sample, attr, None)
        if value is not None:
            return bool(value)

    return os.path.isfile(sample.filepath)


def is_writable(directory):
    """Whether a local directory can be written to."""
    if is_cloud(directory):
        return True

    probe = directory
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent

    return os.access(probe, os.W_OK)


def supports_cloud_media():
    """Whether the runtime can read and write cloud-backed media."""
    return hasattr(fos, "upload_media")


def fallback_root(dataset):
    """Returns a writable local directory owned by FiftyOne for this dataset."""
    import fiftyone as fo

    name = getattr(dataset, "name", "gemini")
    return os.path.join(fo.config.default_dataset_dir, name, "data")


def media_root(dataset, sample=None):
    """Returns the directory generated media should be written beside.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        sample (None): the sample the output derives from

    Returns:
        a writable directory path

    Raises:
        MediaError: if no anchor media exists and no fallback can be built
    """
    anchor = sample
    if anchor is None and dataset is not None:
        anchor = dataset.first()

    if anchor is None:
        if dataset is None or supports_cloud_media():
            raise MediaError(
                "There is no media to write beside. Set an output directory — "
                "on a shared deployment it must be a cloud location such as "
                "gs://your-bucket/generated, so everyone can see the result."
            )
        return fallback_root(dataset)

    beside = os.path.dirname(anchor.filepath)
    if is_writable(beside):
        return beside

    if supports_cloud_media():
        raise MediaError(
            f"'{beside}' is read-only, so generated media cannot be written "
            "beside the source. Set an output directory to a writable cloud "
            "location such as gs://your-bucket/generated. Writing to local "
            "disk is not offered here: on a shared deployment the resulting "
            "sample would be visible only to the node that produced it."
        )

    if dataset is None:
        raise MediaError(
            f"'{beside}' is not writable. Pass an explicit output_dir."
        )

    return fallback_root(dataset)


def output_path(directory, stem, ext, subdir=GENERATED_SUBDIR):
    """Builds a collision-free output path under ``directory``.

    Args:
        directory: the directory to write beside
        stem: a name stem, typically ``"<task>_<source name>"``
        ext: the file extension, without a dot
        subdir (GENERATED_SUBDIR): a subdirectory to keep generated media out
            of the source media's own folder

    Returns:
        a path that does not currently exist
    """
    base = fos.join(directory, subdir) if subdir else directory
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for _ in range(10):
        name = f"{stem}_{stamp}_{uuid.uuid4().hex[:6]}.{ext}"
        path = fos.join(base, name)
        if not fos.exists(path):
            return path

    raise MediaError(f"Could not find a free filename under {base}")


def is_cloud(path):
    """Whether a path lives outside the local filesystem."""
    try:
        return fos.get_file_system(path) != fos.FileSystem.LOCAL
    except Exception:
        return False


def write_media(data, path, tmp_dir=None):
    """Writes bytes to ``path``, which may be local or cloud.

    Args:
        data: the bytes to write
        path: the destination path
        tmp_dir (None): a directory to stage cloud writes through

    Returns:
        ``path``

    Raises:
        MediaError: if the write fails
    """
    if not is_cloud(path):
        fos.ensure_basedir(path)
        with open(path, "wb") as f:
            f.write(data)
        return path

    import tempfile

    staging = tmp_dir or tempfile.gettempdir()
    os.makedirs(staging, exist_ok=True)
    local = os.path.join(staging, f"gemini_{uuid.uuid4().hex}_{os.path.basename(path)}")

    try:
        with open(local, "wb") as f:
            f.write(data)
        fos.copy_file(local, path)
    except Exception as e:
        raise MediaError(f"Could not write to {path}: {e}")
    finally:
        if os.path.isfile(local):
            os.remove(local)

    return path


def upload_local(local_path, path):
    """Moves an already-written local file to its final destination.

    Args:
        local_path: the file on disk
        path: the destination, local or cloud

    Returns:
        ``path``

    Raises:
        MediaError: if the copy fails
    """
    if os.path.abspath(local_path) == os.path.abspath(path):
        return path

    try:
        if is_cloud(path):
            fos.copy_file(local_path, path)
        else:
            fos.ensure_basedir(path)
            fos.copy_file(local_path, path)
    except Exception as e:
        raise MediaError(f"Could not write to {path}: {e}")

    return path


def encode_file(path):
    """Returns the base64 contents of a file."""
    import base64

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def summarize_usage(response):
    """Summarizes token usage from an Interactions API response."""
    usage = response.get("usage") or {}
    return {
        "total_tokens": usage.get("total_tokens"),
        "input_tokens": usage.get("total_input_tokens"),
        "output_tokens": usage.get("total_output_tokens"),
        "thought_tokens": usage.get("total_thought_tokens"),
        "tool_use_tokens": usage.get("total_tool_use_tokens"),
    }

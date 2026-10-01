"""Shared JSON and vLLM log readers for experiment reports."""

import json
import re
from pathlib import Path


def read_json(path):
    """Read a UTF-8 experiment result or report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data, *, ensure_ascii=False):
    """Write a report using the experiment runners' existing JSON format."""
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=ensure_ascii), encoding="utf-8"
    )


def http_status_counts(path, start, end, *, clamp=True):
    """Count chat completion HTTP statuses in a byte range of a vLLM log."""
    with Path(path).open("rb") as stream:
        stream.seek(start)
        length = max(0, end - start) if clamp else end - start
        blob = stream.read(length).decode("utf-8", "replace")
    counts = {}
    for code in re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})', blob):
        counts[code] = counts.get(code, 0) + 1
    return counts

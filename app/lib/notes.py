"""Per-event annotations (memo text + bookmark flag), persisted to one JSON
file next to the bundles (override with ``--notes-file``).

Keyed by ``event_key`` (= session|task|silence_start|pre_speaker), which is
VAD-derived and model-independent, so a note written in comparison mode is
visible in single-model mode and survives re-extraction of any bundle.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path

_VERSION = 1


def load_notes(path: Path) -> dict[str, dict]:
    """{event_key: {"memo": str, "bookmark": bool, ...context...}}."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("notes", {})


def _write_atomic(path: Path, notes: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _VERSION, "notes": notes}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def update_note(path: Path, event_key: str, *,
                memo: str | None = None,
                bookmark: bool | None = None,
                context: dict | None = None) -> dict[str, dict]:
    """Read-modify-write one entry; pass only the fields to change.

    An entry with empty memo and no bookmark is dropped entirely, keeping the
    file free of stale keys. Returns the updated notes mapping."""
    notes = load_notes(path)
    ent = dict(notes.get(event_key, {}))
    if memo is not None:
        ent["memo"] = memo.strip()
    if bookmark is not None:
        ent["bookmark"] = bool(bookmark)
    if not ent.get("memo") and not ent.get("bookmark"):
        notes.pop(event_key, None)
    else:
        if context:
            ent.update(context)
        ent["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        notes[event_key] = ent
    _write_atomic(path, notes)
    return notes

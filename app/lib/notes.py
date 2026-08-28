"""Per-event annotations (memo text + bookmark flag), persisted per
**bundle combination** under ``<bundles-dir>/notes/`` (override the directory
with ``--notes-dir``).

Keyed by ``event_key`` (= session|task|silence_start|pre_speaker), which is
VAD-derived and model-independent, so a note survives re-extraction of any
bundle. ただしファイルは「いま選んでいるバンドルの組み合わせ」ごとに分かれる
(単一モデル = 1個の組み合わせ)。{A, B} を比較しながら書いたメモは単一Aでは
見えない -- どの比較で気づいたことなのかを取り違えないため。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
from pathlib import Path

_VERSION = 2
_MAX_STEM = 200          # これを超える結合名は sha1 に落とす (ファイル名長対策)


def notes_path(notes_dir: Path, bundle_names) -> Path:
    """組み合わせ -> JSONファイルパス。

    キーはバンドルの**ディレクトリ名**をソートして "+" 連結したもの
    (aliases は書き換わるのでキーにしない)。長すぎるときは sha1 の頭12桁。
    どちらの場合もメンバーは JSON 本体側にも記録される (load_bundles)。"""
    joined = "+".join(sorted(str(n) for n in bundle_names))
    stem = joined if len(joined) <= _MAX_STEM else hashlib.sha1(
        joined.encode("utf-8")).hexdigest()[:12]
    return Path(notes_dir) / f"{stem or '_empty'}.json"


def _load_payload(path: Path) -> dict:
    if not Path(path).is_file():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_notes(path: Path) -> dict[str, dict]:
    """{event_key: {"memo": str, "bookmark": bool, ...context...}}."""
    return _load_payload(path).get("notes", {})


def load_bundles(path: Path) -> list[str]:
    """そのファイルが属するバンドル組み合わせ (ファイル名がsha1でも分かる)。"""
    return list(_load_payload(path).get("bundles", []))


def _write_atomic(path: Path, notes: dict[str, dict],
                  bundles: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _VERSION, "bundles": bundles or [], "notes": notes}
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
                context: dict | None = None,
                bundles: list[str] | None = None) -> dict[str, dict]:
    """Read-modify-write one entry; pass only the fields to change.

    An entry with empty memo and no bookmark is dropped entirely, keeping the
    file free of stale keys. ``bundles`` = この組み合わせのディレクトリ名
    (省略時は既存ファイルの記録を引き継ぐ)。Returns the updated notes mapping."""
    payload = _load_payload(path)
    notes = payload.get("notes", {})
    if bundles is None:
        bundles = list(payload.get("bundles", []))
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
    _write_atomic(path, notes, bundles)
    return notes

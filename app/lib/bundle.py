"""Bundle IO for the viewer. Reads only what the extractor wrote --
no vapx / torch / checkpoints (constraint C1/C2 of the spec)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

TASKS = ("shift_hold", "shift_pred")


def list_bundles(root: Path) -> list[str]:
    """Bundle names = subdirectories of ``root`` that contain a meta.json."""
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "meta.json").is_file())


@lru_cache(maxsize=64)
def load_meta(bundle_dir: str) -> dict:
    return json.loads((Path(bundle_dir) / "meta.json").read_text())


@lru_cache(maxsize=16)
def load_cases(bundle_dir: str) -> pd.DataFrame:
    return pd.read_parquet(Path(bundle_dir) / "cases.parquet")


@lru_cache(maxsize=8)
def load_probs(bundle_dir: str, sid: str) -> dict[str, np.ndarray]:
    """Per-session precomputed arrays, upcast to float32 for plotting."""
    out: dict[str, np.ndarray] = {}
    with np.load(Path(bundle_dir) / "probs" / f"{sid}.npz") as d:
        for k in d.files:
            a = d[k]
            out[k] = a.astype(np.float32) if a.dtype == np.float16 else a.copy()
    return out


@lru_cache(maxsize=8)
def load_tokens(bundle_dir: str, sid: str) -> pd.DataFrame | None:
    p = Path(bundle_dir) / "tokens" / f"{sid}.jsonl"
    if not p.is_file():
        return None
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows) if rows else None


def session_meta(meta: dict, sid: str) -> dict | None:
    for s in meta.get("sessions", []):
        if s["id"] == sid:
            return s
    return None


# --------------------------------------------------------------------------
# audio (referenced, never copied into the bundle -- constraint C5)
# --------------------------------------------------------------------------


def resolve_audio_path(recorded: str | None, audio_root: Path | None) -> Path | None:
    """Resolve one audio path recorded in meta.json.

    Try, in order: the recorded path as-is; ``audio_root / recorded`` when the
    recorded path is relative; ``audio_root / basename`` as a last resort
    (covers bundles whose absolute paths belong to another machine)."""
    if not recorded:
        return None
    p = Path(recorded)
    if p.is_file():
        return p
    if audio_root is not None:
        if not p.is_absolute():
            q = audio_root / p
            if q.is_file():
                return q
        q = audio_root / p.name
        if q.is_file():
            return q
    return None


def resolve_session_audio(meta: dict, sid: str, audio_root: Path | None
                          ) -> tuple[Path | None, Path | None]:
    """(left, right) audio paths for a session; either may be None."""
    s = session_meta(meta, sid)
    if s is None:
        return None, None
    if s.get("audio_l") or s.get("audio_r"):
        return (resolve_audio_path(s.get("audio_l"), audio_root),
                resolve_audio_path(s.get("audio_r"), audio_root))
    p = resolve_audio_path(s.get("audio"), audio_root)
    return p, p   # single stereo file: same path both sides


def read_crop(path: Path, start_sec: float, dur_sec: float) -> tuple[np.ndarray, int]:
    """Read ``dur_sec`` seconds starting at ``start_sec`` without loading the
    whole file. Returns (mono float32 array, sample_rate)."""
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        start = max(0, int(start_sec * sr))
        n = int(dur_sec * sr)
        f.seek(min(start, f.frames))
        data = f.read(min(n, f.frames - f.tell()), dtype="float32", always_2d=True)
    return data[:, 0], sr


def read_stereo_crop(path_l: Path | None, path_r: Path | None,
                     start_sec: float, dur_sec: float
                     ) -> tuple[np.ndarray | None, int | None]:
    """(2, n) crop combining the L/R session files (or one stereo file)."""
    if path_l is None and path_r is None:
        return None, None
    if path_l is not None and path_l == path_r:
        with sf.SoundFile(str(path_l)) as f:
            sr = f.samplerate
            start = max(0, int(start_sec * sr))
            n = int(dur_sec * sr)
            f.seek(min(start, f.frames))
            data = f.read(min(n, f.frames - f.tell()), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        return data[:, :2].T.copy(), sr

    parts, srs = [], []
    for p in (path_l, path_r):
        if p is None:
            parts.append(None)
            continue
        w, sr = read_crop(p, start_sec, dur_sec)
        parts.append(w)
        srs.append(sr)
    if not srs:
        return None, None
    sr = srs[0]
    n = min(len(w) for w in parts if w is not None)
    chans = [w[:n] if w is not None else np.zeros(n, dtype=np.float32) for w in parts]
    return np.stack(chans, axis=0), sr


# --------------------------------------------------------------------------
# video (referenced like audio; cropped on demand with ffmpeg)
# --------------------------------------------------------------------------


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def resolve_session_video(sid: str, video_root: Path | None) -> Path | None:
    """``video_root/<sid>.mp4`` if it exists."""
    if video_root is None:
        return None
    p = Path(video_root) / f"{sid}.mp4"
    return p if p.is_file() else None


def read_video_crop(path: Path, start_sec: float, dur_sec: float,
                    max_width: int = 960, with_audio: bool = False) -> bytes | None:
    """Transcode [start, start+dur] to a small MP4 (bytes). The in-app player
    mutes it (audio comes from the wav crop); exports keep the source audio.

    The Tabidachi zoom videos are two 16:9 face tiles letterboxed into the
    middle half of the frame -- crop that band so the faces render large."""
    exe = _ffmpeg_exe()
    if exe is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out = tmp.name
    audio = ["-c:a", "aac", "-b:a", "96k"] if with_audio else ["-an"]
    cmd = [exe, "-y", "-loglevel", "error",
           "-ss", f"{max(0.0, start_sec):.3f}", "-i", str(path),
           "-t", f"{dur_sec:.3f}", *audio,
           "-vf", f"crop=iw:ih/2:0:ih/4,scale='min({max_width},iw)':-2",
           "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
           "-movflags", "+faststart", out]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return Path(out).read_bytes()
    except Exception:
        return None
    finally:
        Path(out).unlink(missing_ok=True)

"""Per-session visual feature curves for the fixed-features detail figure.

RESTORED 2026-09-04 after the previous file was overwritten by mistake. The
public surface below is reconstructed from its call sites (``plots.py``
``fixed_features_figure`` and ``viewer.py``); the transform maths is
reproduced from ``vapx.features.visual_encoder`` and checked numerically
against those torch modules (max abs diff ~1e-7 on float32). Wording and any
internal helpers of the original are gone.

The features are not copied into the bundle: they already exist as per-session
npz under the recipe's ``vis_npz`` tree, and duplicating them per experiment
would multiply hundreds of megabytes. ``meta["vis"]["dirs"]`` records where
they live and this module reads them directly, the same way audio is read from
``--audio-root``.

Transforms are the encoder's *pre-projection* values -- what the model is
actually handed -- so what the figure shows is what the network saw.
"""

from __future__ import annotations

import numpy as np

# Display order. ``raw`` leads because the point of the comparison is what the
# dynamic transforms do *to* it -- the per-speaker offset visible in raw is what
# they remove.
TRANSFORM_ORDER = ["raw", "delta", "ddelta", "rms"]

TRANSFORM_LABEL = {
    "raw": "Raw (生値)",
    "delta": "Δ (1次微分)",
    "ddelta": "Δ² (2次微分)",
    "rms": "RMS(Δ) (運動エネルギー)",
}

# Units per modality, for the y-axis label once a single transform is shown.
UNIT_LABEL = {
    "raw": {"gaze": "rad", "headpose": "deg", "au": "強度"},
    "delta": {"gaze": "rad/f", "headpose": "deg/f", "au": "強度/f"},
    "ddelta": {"gaze": "rad/f²", "headpose": "deg/f²", "au": "強度/f²"},
    "rms": {"gaze": "rad/f", "headpose": "deg/f", "au": "強度/f"},
}

MODALITY_LABEL = {
    "gaze": "視線 (gaze)",
    "headpose": "頭部姿勢 (headpose)",
    "au": "表情 (AU)",
}

# Both AU extractors are 12-wide but name different action units, so the label
# row depends on which npz tree the bundle points at. Guessing wrong mislabels
# every AU row without changing the picture, which is worse than being unsure.
AU_DISFA = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
            "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]   # LibreFace intensity
AU_BP4D = ["AU01", "AU02", "AU04", "AU06", "AU07", "AU10",
           "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]    # ME-GraphAU occurrence

DIM_LABELS = {
    "gaze": ["pitch", "yaw"],
    "headpose": ["pitch", "yaw", "roll"],
    "au": AU_DISFA,
}


def au_labels(au_dir: str | None) -> list[str]:
    """Pick the AU name row from the directory the features came from."""
    return AU_BP4D if au_dir and "libreface" not in str(au_dir) else AU_DISFA

# meta["vis"]["dirs"] key -> modality label used everywhere else.
DIR_KEY = {"gaze": "gaze_dir", "headpose": "head_dir", "au": "au_dir"}

# Visual npz are written at this rate by local/convert_vis_npz.sh; the model
# consumes them at frame_hz / vis_audio_ratio, which is the same number.
VIS_FPS = 25.0

# Encoder defaults (_DeltaEncoder.K, _DeltaDeltaEncoder.K, _RmsEncoder).
DELTA_K = 5
RMS_WIN = 8


def _delta(x: np.ndarray, k: int = DELTA_K) -> np.ndarray:
    """Causal weighted multi-lag first derivative, sign retained.

    delta(t) = sum_{j=1..k} j * (x[t] - x[t-j]) / sum_{j=1..k} j^2
    """
    acc = np.zeros_like(x)
    norm = sum(j * j for j in range(1, k + 1))
    for j in range(1, k + 1):
        acc[j:] += j * (x[j:] - x[:-j])
    return acc / norm


def _rms(x: np.ndarray, win: int = RMS_WIN) -> np.ndarray:
    """Causal local RMS of the frame-to-frame difference, per dimension."""
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    c = np.cumsum(np.pad(d ** 2, ((win, 0), (0, 0))), axis=0)
    return np.sqrt((c[win:] - c[:-win]) / win)


def _ddelta(x: np.ndarray, k: int = DELTA_K) -> np.ndarray:
    """The same causal derivative applied twice (effective span 2k frames)."""
    return _delta(_delta(x, k), k)


TRANSFORMS = {
    "raw": lambda x: x,
    "delta": _delta,
    "ddelta": _ddelta,
    "rms": _rms,
}

# Δ and Δ² keep their sign. The heatmap this figure replaced drew magnitude on
# a 0-max colormap, which was fine for spotting *that* a channel moved; a line
# plot can show direction too, and direction is the finding -- "looks up when
# backchannelling" is a claim about the sign of gaze pitch, not its size.
# RMS(Δ) is unsigned by construction (it is an energy).
SIGNED: set[str] = set()


def load_session(vis_meta: dict, sid: str) -> dict:
    """-> {modality: {role: (T, D) float64}} for whatever is present on disk.

    Missing modalities and unreadable sessions are skipped rather than raised:
    a bundle may name a modality whose npz tree has since moved, and the detail
    view should still render everything else.
    """
    dirs = (vis_meta or {}).get("dirs") or {}
    roles = (vis_meta or {}).get("roles") or ["customer", "operator"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for modality, key in DIR_KEY.items():
        d = dirs.get(key)
        if not d:
            continue
        try:
            z = np.load(f"{d}/{sid}.npz")
        except (FileNotFoundError, OSError):
            continue
        per_role = {r: np.asarray(z[r], dtype=np.float64) for r in roles if r in z}
        if per_role:
            out[modality] = per_role
    return out


def transform_curves(vis_meta: dict, sid: str, t0: float, t1: float) -> dict:
    """-> {transform: {modality: (t, arr_a, arr_b)}} over the window [t0, t1].

    ``t`` is seconds; ``arr_a`` / ``arr_b`` are ``(len(t), D)`` for the two
    roles in ``meta["vis"]["roles"]`` order.

    Each transform runs over the whole session before slicing, never over the
    crop: every one of them is causal with a window of 5-50 frames, so
    transforming a crop would fabricate a spike at its left edge where no
    history exists.
    """
    series = load_session(vis_meta, sid)
    roles = (vis_meta or {}).get("roles") or ["customer", "operator"]
    fps = float((vis_meta or {}).get("vis_fps") or VIS_FPS)

    out: dict[str, dict[str, tuple]] = {t: {} for t in TRANSFORM_ORDER}
    for modality, per_role in series.items():
        n = min(arr.shape[0] for arr in per_role.values())
        i0 = max(0, int(np.floor(t0 * fps)))
        i1 = min(n, int(np.ceil(t1 * fps)) + 1)
        if i1 <= i0:
            continue
        t = np.arange(i0, i1) / fps
        for tname in TRANSFORM_ORDER:
            fn = TRANSFORMS[tname]
            cols = []
            for role in roles:
                arr = per_role.get(role)
                if arr is None:
                    cols.append(np.zeros((i1 - i0, 1)))
                    continue
                v = fn(arr[:n])
                if tname in SIGNED:
                    v = np.abs(v)
                cols.append(v[i0:i1])
            out[tname][modality] = (t, cols[0], cols[1])
    return out

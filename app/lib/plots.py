"""Matplotlib detail figure for one error case.

Everything is drawn from the bundle's precomputed arrays (+ optionally an
audio crop read from --audio-root); no model code runs here.

Colors: speakers keep the lab's A=blue / B=orange convention (validated
pair); comparison-mode model curves use a separate validated 4-slot theme so
they never impersonate a speaker; correct/incorrect use status green/red and
always carry an OK/NG text label (never color alone).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

COL_A = "#2a78d6"        # speaker A (= L / ch0)
COL_B = "#eb6834"        # speaker B (= R / ch1)
COL_OK = "#0ca30c"       # status good
COL_NG = "#d03b3b"       # status critical
MODEL_COLORS = ["#1baf7a", "#4a3aa7", "#eda100", "#e87ba4"]  # comparison overlays
GRID = "#e6e5e1"
INK = "#52514e"
MUTED = "#9c9a94"

_FONT_READY = False


def _setup_fonts() -> None:
    """Pick a CJK-capable font when available so Japanese tokens render."""
    global _FONT_READY
    if _FONT_READY:
        return
    candidates = [
        ("IPAexGothic", "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"),
        ("IPAGothic", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"),
        ("Noto Sans CJK JP", None),
    ]
    for name, path in candidates:
        if path and Path(path).is_file():
            font_manager.fontManager.addfont(path)
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams["font.family"] = name
        break
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def _style_axis(ax, t0: float, t1: float, ylabel: str = "") -> None:
    ax.set_xlim(t0, t1)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)


def _envelope(w: np.ndarray, sr: int, t_off: float, n_bins: int = 1500):
    """Min/max envelope of a waveform for fast plotting."""
    n = len(w)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    bins = min(n_bins, n)
    edge = np.linspace(0, n, bins + 1, dtype=int)
    lo = np.array([w[a:b].min() for a, b in zip(edge[:-1], edge[1:]) if b > a])
    hi = np.array([w[a:b].max() for a, b in zip(edge[:-1], edge[1:]) if b > a])
    t = t_off + (edge[:-1][: len(lo)] + edge[1:][: len(lo)]) / 2 / sr
    return t, lo, hi


def _eval_window_sec(case, zcfg: dict, frame_hz: float) -> tuple[float, float]:
    """The frame window whose scores decided this case, in seconds
    (mirrors zero_shot._plan_session)."""
    ss = int(case["silence_start"])
    if case["task"] == "shift_hold":
        s = ss + int(round(zcfg["sh_eval_start_sec"] * frame_hz))
        e = s + int(round(zcfg["sh_eval_dur_sec"] * frame_hz))
    else:  # shift_pred
        e = ss
        s = ss - int(round(zcfg["spred_eval_dur_sec"] * frame_hz))
        mc = int(round(zcfg.get("min_context_sec", 0) * frame_hz))
        s = max(mc, s)
    return s / frame_hz, e / frame_hz


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------


def _panel_wave(ax, wav_ch, sr, t_off, color, label):
    t, lo, hi = _envelope(wav_ch, sr, t_off)
    if len(t):
        ax.fill_between(t, lo, hi, color=color, lw=0, alpha=0.9, zorder=2)
    ax.set_yticks([])
    ax.text(0.003, 0.82, label, transform=ax.transAxes, fontsize=9,
            color=color, fontweight="bold")


def _panel_vad(ax, vad, frame_hz, t0, t1):
    a, b = int(max(0, t0 * frame_hz)), int(t1 * frame_hz) + 1
    seg = vad[a:b].astype(np.float32)   # uint8 would underflow on negation
    t = (np.arange(a, a + len(seg))) / frame_hz
    ax.fill_between(t, 0, seg[:, 0], step="post", color=COL_A, alpha=0.85, lw=0)
    ax.fill_between(t, 0, -seg[:, 1], step="post", color=COL_B, alpha=0.85, lw=0)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([1, -1])
    ax.set_yticklabels(["A", "B"], fontsize=8)


def _panel_events(ax, cases_win, selected_key, zcfg, frame_hz, models=None):
    """All same-task events inside the window; the selected one is bold.

    Single model (``models`` is None): eval window colored by OK/NG, text
    ``G:<gold> P:<pred> OK/NG``. Comparison (``models`` = bundle names,
    ``cases_win`` = joined table with ``pred_<m>``/``correct_<m>``): gold on
    top in ink, below it one line per model in that model's overlay color
    (S○ / H× ... ○=正解 ×=誤り; shift_pred: TP / FN)."""
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    n_mod = len(models) if models else 0
    for _, c in cases_win.iterrows():
        sel = c["event_key"] == selected_key
        ss, se = c["silence_start"] / frame_hz, c["silence_end"] / frame_hz
        ax.axvspan(ss, se, color=MUTED, alpha=0.28 if sel else 0.15, lw=0)
        ws, we = _eval_window_sec(c, zcfg, frame_hz)
        is_sh = c["task"] == "shift_hold"
        gold = c["gold"] if is_sh else "S"
        x = (ws + we) / 2
        fs = 8 if sel else 7
        if models:
            ax.axvspan(ws, we, color=INK, alpha=0.32 if sel else 0.15, lw=0)
            ax.text(x, 0.86, f"G:{gold}", ha="center", fontsize=fs, color=INK,
                    fontweight="bold", clip_on=True)
            ys = np.linspace(0.60, 0.10, n_mod) if n_mod > 1 else [0.35]
            for (m, y, col) in zip(models, ys, MODEL_COLORS):
                if f"correct_{m}" not in c:
                    continue          # joined table lacks this model
                corr = bool(c[f"correct_{m}"])
                if is_sh:
                    txt = f"{c[f'pred_{m}']}{'○' if corr else '×'}"
                else:
                    txt = "TP○" if corr else "FN×"
                ax.text(x, y, txt, ha="center", fontsize=fs, color=col,
                        fontweight="bold" if sel else "normal", clip_on=True)
        else:
            col = COL_OK if c["correct"] else COL_NG
            ax.axvspan(ws, we, color=col, alpha=0.55 if sel else 0.3, lw=0)
            if is_sh:
                txt = f"G:{gold} P:{c['pred']} {'OK' if c['correct'] else 'NG'}"
            else:
                txt = f"G:{gold} {'TP' if c['correct'] else 'FN'}"
            ax.text(x, 0.78 if sel else 0.18, txt,
                    ha="center", fontsize=fs, clip_on=True,
                    fontweight="bold" if sel else "normal", color=col)


def _panel_bins(ax, bin_probs, case, zcfg, frame_hz, bin_times, t0, t1):
    """Mean per-bin activity over the decision window, drawn at the future
    times each bin refers to. A = upper half (blues), B = lower (oranges)."""
    ws, we = _eval_window_sec(case, zcfg, frame_hz)
    a, b = int(ws * frame_hz), max(int(ws * frame_hz) + 1, int(we * frame_hz))
    mean = bin_probs[a:b].mean(axis=0)          # (K, 2)
    cum = np.concatenate([[0.0], np.cumsum(bin_times)])
    cm_a = plt.get_cmap("Blues")
    cm_b = plt.get_cmap("Oranges")
    for k in range(mean.shape[0]):
        x0, x1 = ws + cum[k], ws + cum[k + 1]
        for spk, (cm, y0) in enumerate(((cm_a, 0.5), (cm_b, 0.0))):
            v = float(mean[k, spk])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, 0.5,
                                   facecolor=cm(0.15 + 0.75 * v),
                                   edgecolor="white", lw=1.5, zorder=2))
            if t0 <= (x0 + x1) / 2 <= t1:
                ax.text((x0 + x1) / 2, y0 + 0.25, f"{v:.2f}", ha="center",
                        va="center", fontsize=7, color=INK, clip_on=True)
    ax.axvline(ws, color=INK, lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.75, 0.25])
    ax.set_yticklabels(["A", "B"], fontsize=8)


def _panel_tokens(ax, tokens, frame_hz, t0, t1):
    ax.set_ylim(0, 1)
    ax.set_yticks([0.72, 0.24])
    ax.set_yticklabels(["A", "B"], fontsize=8)
    a0, a1 = int(t0 * frame_hz), int(t1 * frame_hz)
    vis = tokens[(tokens["pos"] < a1) & (tokens["end"] > a0)]
    for _, r in vis.iterrows():
        y = 0.72 if r["ch"] == "L" else 0.24
        col = COL_A if r["ch"] == "L" else COL_B
        x0, x1 = r["pos"] / frame_hz, min(r["end"], a1) / frame_hz
        ax.plot([x0, x1], [y - 0.1, y - 0.1], color=col, lw=2,
                alpha=0.4, solid_capstyle="butt")
        if t0 <= x0 <= t1:
            ax.text(x0, y, str(r["text"]), fontsize=8, color=INK,
                    ha="left", va="center", rotation=30, clip_on=True)


def _panel_task_score(ax, probs, case, frame_hz, t0, t1):
    a, b = int(max(0, t0 * frame_hz)), int(t1 * frame_hz) + 1
    t = np.arange(a, min(b, len(probs["score_sh"]))) / frame_hz
    key = "score_sh" if case["task"] == "shift_hold" else "score_spred"
    curves = probs[key][a: a + len(t)]
    ax.plot(t, curves[:, 0], color=COL_A, lw=2, label="A")
    ax.plot(t, curves[:, 1], color=COL_B, lw=2, label="B")
    thr = case.get("threshold")
    if case["task"] == "shift_pred" and thr == thr:   # not NaN
        ax.axhline(thr, color=INK, lw=1, ls="--")
        ax.text(t1, thr, f" thr={thr:.3f}", fontsize=7, color=INK, va="bottom", ha="right")
    ax.set_ylim(-0.02, max(0.5, float(curves.max()) * 1.15) if len(curves) else 1)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)


def _pshift_curve(probs: dict, case: dict) -> np.ndarray:
    """Per-frame curve oriented so that UP always means SHIFT for this event.

    shift_hold: P(shift) = s_other / (s_A + s_B) -- the same ratio whose
    eval-window mean is the case score. shift_pred: the raw subset score of
    the incoming (post) speaker, which is what the threshold applies to."""
    if case["task"] == "shift_hold":
        other = 1 - int(case["pre_speaker"])
        s = probs["score_sh"]
        return s[:, other] / np.clip(s[:, 0] + s[:, 1], 1e-9, None)
    return probs["score_spred"][:, int(case["post_speaker"])]


def _panel_pshift(ax, probs, case, zcfg, frame_hz, t0, t1, overlays=None):
    """SHIFT-oriented decision panel: curve above its (dashed) threshold in
    the shaded eval window = that model predicted SHIFT."""
    a, b = int(max(0, t0 * frame_hz)), int(t1 * frame_hz) + 1
    ws, we = _eval_window_sec(case, zcfg, frame_hz)
    ax.axvspan(ws, we, color=MUTED, alpha=0.18, lw=0)

    is_sh = case["task"] == "shift_hold"
    ymax = 0.25
    if overlays:
        for i, (ov, col) in enumerate(zip(overlays, MODEL_COLORS)):
            curve = _pshift_curve(ov["probs"], case)
            t = np.arange(a, min(b, len(curve))) / frame_hz
            y = curve[a: a + len(t)]
            ax.plot(t, y, color=col, lw=2, label=ov["name"])
            thr = ov.get("threshold")
            if thr is not None and thr == thr:
                ax.axhline(thr, color=col, lw=1, ls="--", alpha=0.8)
            if len(y):
                ymax = max(ymax, float(y.max()))
            if is_sh:
                letter = str(ov.get("pred", "?"))
            else:
                letter = "TP○" if ov.get("correct") else "FN×"
            ax.text(we + 0.05, 0.9 - 0.28 * i, letter, color=col,
                    fontsize=9, fontweight="bold", clip_on=True,
                    transform=ax.get_xaxis_transform())
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    else:
        curve = _pshift_curve(probs, case)
        t = np.arange(a, min(b, len(curve))) / frame_hz
        y = curve[a: a + len(t)]
        ax.plot(t, y, color=INK, lw=2)
        thr = case.get("threshold")
        if thr is not None and thr == thr:
            ax.axhline(thr, color=INK, lw=1, ls="--", alpha=0.8)
            ax.text(t1, thr, f" thr={thr:.3f}", fontsize=7, color=INK,
                    va="bottom", ha="right")
        if len(y):
            ymax = max(ymax, float(y.max()))

    if is_sh:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        note = "P(SHIFT)  ↑=SHIFT予測 / ↓=HOLD予測 (破線=閾値)"
    else:
        ax.set_ylim(-0.02, ymax * 1.1)
        note = "S-predスコア(相手話者)  破線閾値より上=SHIFT予測"
    ax.text(0.003, 0.84, note, transform=ax.transAxes,
            fontsize=8, color=INK, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))


def _panel_prob(ax, curve, frame_hz, t0, t1, label, overlays=None):
    """p_now / p_future. >0.5 = A dominates (blue fill), <0.5 = B (orange)."""
    a, b = int(max(0, t0 * frame_hz)), int(t1 * frame_hz) + 1
    t = np.arange(a, min(b, len(curve))) / frame_hz
    y = curve[a: a + len(t)]
    ax.axhline(0.5, color=GRID, lw=1)
    if overlays:
        # comparison mode: one line per model, no fills
        for (name, ov_curve), col in zip(overlays, MODEL_COLORS):
            oy = ov_curve[a: a + len(t)]
            n = min(len(t), len(oy))
            ax.plot(t[:n], oy[:n], color=col, lw=2, label=name)
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    else:
        ax.fill_between(t, 0.5, y, where=y >= 0.5, color=COL_A, alpha=0.25, lw=0)
        ax.fill_between(t, 0.5, y, where=y < 0.5, color=COL_B, alpha=0.25, lw=0)
        ax.plot(t, y, color=INK, lw=1.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.text(0.003, 0.82, label, transform=ax.transAxes, fontsize=9,
            color=INK, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))


# --------------------------------------------------------------------------
# figure assembly
# --------------------------------------------------------------------------


def detail_figure(
    *,
    case: dict,
    probs: dict,
    meta: dict,
    cases_win,
    t0: float,
    t1: float,
    wav: np.ndarray | None = None,
    wav_sr: int | None = None,
    wav_t0: float = 0.0,
    tokens=None,
    token_sets: list[tuple[str, object]] | None = None,
    overlays: list[dict] | None = None,
):
    """One figure, panels top to bottom (only those with data):
    wave A / wave B / VAD / events / bin heatmap / tokens (one panel per
    entry of ``token_sets``) / task score (single-model only) / P(SHIFT) /
    p_now / p_future. All share the time axis [t0, t1] (seconds).

    ``token_sets``: ``[(label, tokens_df), ...]`` -- comparison mode passes
    one entry per distinct lang feature set so token contents can be
    compared across models on the same time axis. ``tokens`` (single df)
    is the single-model shorthand.

    ``overlays`` (comparison mode): ``[{name, probs, threshold, pred,
    correct}, ...]`` -- one entry per model, first entry = primary bundle."""
    if token_sets is None and tokens is not None:
        token_sets = [("", tokens)]
    token_sets = [(lbl, t) for lbl, t in (token_sets or []) if t is not None]
    _setup_fonts()
    frame_hz = float(meta["frame_hz"])
    zcfg = meta["zero_shot_config"]
    bin_times = meta.get("bin_times_sec") or [0.2, 0.4, 0.6, 0.8]

    model_names = [o["name"] for o in overlays] if overlays else None
    events_h = 0.8 if not overlays else min(1.7, 0.75 + 0.25 * len(overlays))

    panels: list[tuple[str, float]] = []
    if wav is not None:
        panels += [("wave_a", 1.4), ("wave_b", 1.4)]
    panels += [("vad", 0.8), ("events", events_h), ("bins", 0.9)]
    for i in range(len(token_sets)):
        panels += [(f"tokens{i}", 1.0)]
    if overlays is None:
        panels += [("score", 1.0)]           # raw A/B curves (single model)
    panels += [("p_shift", 1.0), ("p_now", 1.0), ("p_future", 1.0)]

    fig, axes = plt.subplots(
        len(panels), 1, sharex=True,
        figsize=(12, 1.05 * sum(h for _, h in panels)),
        gridspec_kw={"height_ratios": [h for _, h in panels], "hspace": 0.12},
    )
    # Rendered without bbox_inches="tight" (the figure-player overlay needs
    # stable pixel geometry), so trim the margins here instead.
    fig.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.07)
    ax_of = dict(zip((n for n, _ in panels), np.atleast_1d(axes)))

    if wav is not None:
        _panel_wave(ax_of["wave_a"], wav[0], wav_sr, wav_t0, COL_A, "A (L)")
        _panel_wave(ax_of["wave_b"], wav[1], wav_sr, wav_t0, COL_B, "B (R)")
    _panel_vad(ax_of["vad"], probs["vad"], frame_hz, t0, t1)
    _panel_events(ax_of["events"], cases_win, case["event_key"], zcfg, frame_hz,
                  models=model_names)
    _panel_bins(ax_of["bins"], probs["bin_probs"], case, zcfg, frame_hz,
                bin_times, t0, t1)
    for i, (lbl, tdf) in enumerate(token_sets):
        ax_t = ax_of[f"tokens{i}"]
        _panel_tokens(ax_t, tdf, frame_hz, t0, t1)
        if lbl:
            col = (MODEL_COLORS[model_names.index(lbl)]
                   if (model_names and lbl in model_names) else INK)
            ax_t.text(0.003, 0.84, lbl, transform=ax_t.transAxes, fontsize=8,
                      color=col, fontweight="bold",
                      bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    if "score" in ax_of:
        _panel_task_score(ax_of["score"], probs, case, frame_hz, t0, t1)
    _panel_pshift(ax_of["p_shift"], probs, case, zcfg, frame_hz, t0, t1,
                  overlays=overlays)
    ov_now = [(o["name"], o["probs"]["p_now"]) for o in overlays] if overlays else None
    ov_fut = [(o["name"], o["probs"]["p_future"]) for o in overlays] if overlays else None
    _panel_prob(ax_of["p_now"], probs["p_now"], frame_hz, t0, t1,
                "p_now  ↑=A / ↓=B", ov_now)
    _panel_prob(ax_of["p_future"], probs["p_future"], frame_hz, t0, t1,
                "p_future  ↑=A / ↓=B", ov_fut)

    labels = {"wave_a": "", "wave_b": "", "vad": "VAD", "events": "S/H",
              "bins": "bins", "score": "score",
              "p_shift": "P(SHIFT)", "p_now": "", "p_future": ""}
    for i in range(len(token_sets)):
        labels[f"tokens{i}"] = "tokens"
    t_event = case["silence_start"] / frame_hz
    for name, _ in panels:
        ax = ax_of[name]
        _style_axis(ax, t0, t1, labels.get(name, ""))
        ax.axvline(t_event, color=INK, lw=0.8, ls=":", alpha=0.6, zorder=1)
    ax_of[panels[-1][0]].set_xlabel("time [s]", fontsize=9, color=INK)

    ok = "OK" if case["correct"] else "NG"
    pre = "A" if int(case["pre_speaker"]) == 0 else "B"
    post = "A" if int(case["post_speaker"]) == 0 else "B"
    fig.suptitle(
        f"{case['session']}  {case['task']}  t={case['t_sec']:.2f}s   "
        f"pre={pre}→post={post}   "
        f"gold={case['gold']} pred={case['pred']} [{ok}]   "
        f"score={case['score']:.4f} thr={case['threshold']:.4f}",
        fontsize=10, color=COL_OK if case["correct"] else COL_NG, y=0.985)
    return fig

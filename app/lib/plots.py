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

from . import vis_features as VF

COL_A = "#2a78d6"        # speaker A (= L / ch0)
COL_B = "#eb6834"        # speaker B (= R / ch1)
COL_OK = "#0ca30c"       # status good
COL_NG = "#d03b3b"       # status critical
MODEL_COLORS = ["#1baf7a", "#4a3aa7", "#eda100", "#e87ba4"]  # comparison overlays
MODALITY_COLORS = {"gaze": "#1baf7a", "headpose": "#4a3aa7", "au": "#eda100"}
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
        ("Noto Sans JP",  # bundled with the app (app/assets/fonts/)
         Path(__file__).resolve().parents[1] / "assets/fonts/NotoSansJP-Regular.ttf"),
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
    """The frame window whose scores decided this case, in seconds.

    Schema-v2 bundles carry the window directly (win_start/win_end); older
    bundles re-derive it from silence_start (mirrors zero_shot._plan_session)."""
    ws, we = case.get("win_start"), case.get("win_end")
    if ws is not None and we is not None and ws == ws:   # not None / NaN
        return int(ws) / frame_hz, int(we) / frame_hz
    ss = int(case["silence_start"])
    mc = int(round(zcfg.get("min_context_sec", 0) * frame_hz))
    task = case["task"]
    if task == "shift_hold":
        s = ss + int(round(zcfg["sh_eval_start_sec"] * frame_hz))
        e = s + int(round(zcfg["sh_eval_dur_sec"] * frame_hz))
    elif task == "shift_pred":
        e = ss
        s = max(mc, ss - int(round(zcfg["spred_eval_dur_sec"] * frame_hz)))
    elif task == "bc_pred":       # window before the BC onset (= silence_start)
        e = ss
        s = max(mc, ss - int(round(zcfg["bcpred_eval_dur_sec"] * frame_hz)))
    else:                          # short_long: onset window (short=BC onset,
        anchor = ss if case.get("gold") == "short" else int(case["silence_end"])
        s = anchor                 # long=post-shift onset = silence_end)
        e = s + int(round(zcfg["sl_eval_dur_sec"] * frame_hz))
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


_ABBREV = {"short": "S", "long": "L"}      # short_long letters in tight panels


def _gold_label(c) -> str:
    task = c["task"]
    if task == "shift_hold":
        return str(c["gold"])
    if task == "shift_pred":
        return "S"
    if task == "bc_pred":
        return "BC"
    return _ABBREV.get(c["gold"], str(c["gold"]))       # short_long


def _pred_mark(c, task, pred, correct: bool) -> str:
    """Per-model outcome letter (comparison mode): ○=正解 ×=誤り; the
    miss-centric tasks (shift_pred / bc_pred) read TP/FN instead."""
    if task in ("shift_pred", "bc_pred"):
        return "TP○" if correct else "FN×"
    return f"{_ABBREV.get(pred, pred)}{'○' if correct else '×'}"


def _panel_events(ax, cases_win, selected_key, zcfg, frame_hz, models=None):
    """All same-task events inside the window; the selected one is bold.

    Single model (``models`` is None): eval window colored by OK/NG, text
    ``G:<gold> P:<pred> OK/NG``. Comparison (``models`` = bundle names,
    ``cases_win`` = joined table with ``pred_<m>``/``correct_<m>``): gold on
    top in ink, below it one line per model in that model's overlay color
    (S○ / H× ... ○=正解 ×=誤り; shift_pred / bc_pred: TP / FN)."""
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    n_mod = len(models) if models else 0
    for _, c in cases_win.iterrows():
        sel = c["event_key"] == selected_key
        task = c["task"]
        ss, se = c["silence_start"] / frame_hz, c["silence_end"] / frame_hz
        ax.axvspan(ss, se, color=MUTED, alpha=0.28 if sel else 0.15, lw=0)
        ws, we = _eval_window_sec(c, zcfg, frame_hz)
        gold = _gold_label(c)
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
                txt = _pred_mark(c, task, c.get(f"pred_{m}", "?"),
                                 bool(c[f"correct_{m}"]))
                ax.text(x, y, txt, ha="center", fontsize=fs, color=col,
                        fontweight="bold" if sel else "normal", clip_on=True)
        else:
            col = COL_OK if c["correct"] else COL_NG
            ax.axvspan(ws, we, color=col, alpha=0.55 if sel else 0.3, lw=0)
            if task in ("shift_pred", "bc_pred"):
                txt = f"G:{gold} {'TP' if c['correct'] else 'FN'}"
            else:
                txt = (f"G:{gold} P:{_ABBREV.get(c['pred'], c['pred'])} "
                       f"{'OK' if c['correct'] else 'NG'}")
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


_SCORE_KEY = {"shift_hold": "score_sh", "shift_pred": "score_spred",
              "bc_pred": "score_bc", "short_long": "score_bc"}


def _panel_task_score(ax, probs, case, frame_hz, t0, t1):
    a, b = int(max(0, t0 * frame_hz)), int(t1 * frame_hz) + 1
    t = np.arange(a, min(b, len(probs["score_sh"]))) / frame_hz
    key = _SCORE_KEY[case["task"]]
    curves = probs[key][a: a + len(t)]
    ax.plot(t, curves[:, 0], color=COL_A, lw=2, label="A")
    ax.plot(t, curves[:, 1], color=COL_B, lw=2, label="B")
    thr = case.get("threshold")
    if case["task"] != "shift_hold" and thr == thr:   # not NaN
        ax.axhline(thr, color=INK, lw=1, ls="--")
        ax.text(t1, thr, f" thr={thr:.3f}", fontsize=7, color=INK, va="bottom", ha="right")
    ax.set_ylim(-0.02, max(0.5, float(curves.max()) * 1.15) if len(curves) else 1)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)


def _pshift_curve(probs: dict, case: dict) -> np.ndarray | None:
    """Per-frame decision curve for the selected event, oriented so that UP
    always means the positive prediction (SHIFT / S-pred / BC / SHORT).

    shift_hold: P(shift) = s_other / (s_A + s_B) -- the same ratio whose
    eval-window mean is the case score. Other tasks: the raw subset score of
    the scored (post) speaker, which is what the threshold applies to
    (shift_pred: spred subset; bc_pred / short_long: bc subset)."""
    if case["task"] == "shift_hold":
        other = 1 - int(case["pre_speaker"])
        s = probs["score_sh"]
        return s[:, other] / np.clip(s[:, 0] + s[:, 1], 1e-9, None)
    key = _SCORE_KEY[case["task"]]
    if key not in probs:            # v1 bundle without score_bc
        return None
    return probs[key][:, int(case["post_speaker"])]


def _panel_pshift(ax, probs, case, zcfg, frame_hz, t0, t1, overlays=None):
    """SHIFT-oriented decision panel: curve above its (dashed) threshold in
    the shaded eval window = that model predicted SHIFT."""
    a, b = int(max(0, t0 * frame_hz)), int(t1 * frame_hz) + 1
    ws, we = _eval_window_sec(case, zcfg, frame_hz)
    ax.axvspan(ws, we, color=MUTED, alpha=0.18, lw=0)

    task = case["task"]
    is_sh = task == "shift_hold"
    ymax = 0.25
    if overlays:
        for i, (ov, col) in enumerate(zip(overlays, MODEL_COLORS)):
            curve = _pshift_curve(ov["probs"], case)
            if curve is None:               # v1 bundle without this curve
                continue
            t = np.arange(a, min(b, len(curve))) / frame_hz
            y = curve[a: a + len(t)]
            ax.plot(t, y, color=col, lw=2, label=ov["name"])
            thr = ov.get("threshold")
            if thr is not None and thr == thr:
                ax.axhline(thr, color=col, lw=1, ls="--", alpha=0.8)
            if len(y):
                ymax = max(ymax, float(y.max()))
            if is_sh or task == "short_long":
                letter = _ABBREV.get(ov.get("pred"), str(ov.get("pred", "?")))
                letter += "○" if ov.get("correct") else "×"
            else:
                letter = "TP○" if ov.get("correct") else "FN×"
            ax.text(we + 0.05, 0.9 - 0.28 * i, letter, color=col,
                    fontsize=9, fontweight="bold", clip_on=True,
                    transform=ax.get_xaxis_transform())
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    else:
        curve = _pshift_curve(probs, case)
        if curve is None:
            ax.text(0.5, 0.5, "score_bc が無い旧バンドルです (再抽出で表示可)",
                    transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
            curve = np.zeros(0)
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
        note = {
            "shift_pred": "S-predスコア(相手話者)  破線閾値より上=SHIFT予測",
            "bc_pred": "BCスコア(BC話者)  破線閾値より上=BC予測",
            "short_long": "S/Lスコア(発話開始話者のbc subset)  破線閾値より上=SHORT予測",
        }[task]
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
    panels += [("vad", 0.8), ("events", events_h)]
    n_bins_panels = len(overlays) if overlays else 1
    panels += [(f"bins{i}", 0.9) for i in range(n_bins_panels)]
    for i in range(len(token_sets)):
        panels += [(f"tokens{i}", 1.0)]
    if overlays is None and _SCORE_KEY[case["task"]] in probs:
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
    if overlays:
        # one bins panel per model so bin activities can be compared
        for i, ov in enumerate(overlays):
            ax_b = ax_of[f"bins{i}"]
            _panel_bins(ax_b, ov["probs"]["bin_probs"], case, zcfg, frame_hz,
                        bin_times, t0, t1)
            ax_b.text(0.003, 0.84, ov["name"], transform=ax_b.transAxes,
                      fontsize=8, color=MODEL_COLORS[i % len(MODEL_COLORS)],
                      fontweight="bold",
                      bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    else:
        _panel_bins(ax_of["bins0"], probs["bin_probs"], case, zcfg, frame_hz,
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

    ev_label = {"shift_hold": "S/H", "shift_pred": "S-pred",
                "bc_pred": "BC", "short_long": "S/L"}.get(case["task"], "events")
    labels = {"wave_a": "", "wave_b": "", "vad": "VAD", "events": ev_label,
              "score": "score",
              "p_shift": "P(SHIFT)", "p_now": "", "p_future": ""}
    for i in range(n_bins_panels):
        labels[f"bins{i}"] = "bins"
    for i in range(len(token_sets)):
        labels[f"tokens{i}"] = "tokens"
    t_event = float(case["t_sec"])   # == silence_start except short_long/long
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


# Line styles cycle per feature dimension; speaker stays encoded in color, so
# a dimension can be followed across the two speakers by its dash pattern.
_DIM_STYLES = ["-", "--", ":", "-."]

# Above this many dimensions a cell would be an unreadable thicket (AU is 12
# wide), so only the most active ones are drawn in full; the rest stay as faint
# context lines. Picked per cell and per window, not globally.
_MAX_HIGHLIGHT_DIMS = 3


def _line_cell(ax, t, arr_a, arr_b, dim_labels, t0, t1, dims=None):
    """One (transform, modality) cell as overlaid line plots.

    Both speakers share one y-axis: within a modality the dimensions carry the
    same unit, so the vertical offset between the two speakers is real and
    worth seeing -- it is exactly the per-speaker bias the dynamic transforms
    are meant to remove.

    ``dims`` pins which feature dimensions are drawn solid. Left as None, the
    ones that actually move inside this window are picked automatically, so one
    active AU stays visible instead of being buried under eleven flat ones."""
    D = len(dim_labels)
    if len(t) < 2 or not arr_a.size:
        ax.set_xlim(t0, t1)
        return

    if dims is not None:
        shown = [d for d in dims if 0 <= d < D]
    elif D > _MAX_HIGHLIGHT_DIMS:
        # Rank dimensions by how much they move here, across both speakers.
        span = np.maximum(np.ptp(arr_a, axis=0), np.ptp(arr_b, axis=0))
        shown = list(np.argsort(span)[::-1][:_MAX_HIGHLIGHT_DIMS])
    else:
        shown = list(range(D))
    if not shown:
        shown = [0]

    for d in range(D):
        lead = d in shown
        style = _DIM_STYLES[list(shown).index(d) % len(_DIM_STYLES)] if lead else "-"
        for arr, col in ((arr_a, COL_A), (arr_b, COL_B)):
            ax.plot(t, arr[:, d], color=col, lw=1.3 if lead else 0.6,
                    ls=style, alpha=0.95 if lead else 0.18,
                    label=f"{dim_labels[d]}" if lead and col == COL_A else None,
                    zorder=3 if lead else 1)

    # Robust limits: the causal transforms spike at a session's first frames
    # where they have no history, and one such frame would flatten everything
    # else. Percentiles keep the visible range on the real motion.
    both = np.concatenate([arr_a[:, shown].ravel(), arr_b[:, shown].ravel()])
    lo, hi = np.percentile(both, [0.5, 99.5])
    pad = max((hi - lo) * 0.12, 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    if lo < 0 < hi:
        ax.axhline(0, color=GRID, lw=1, zorder=0)

    if len(shown) < D:
        ax.text(0.995, 0.97, f"他 {D - len(shown)} 次元は薄線",
                transform=ax.transAxes, fontsize=6, color=MUTED,
                ha="right", va="top")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=len(shown),
              handlelength=1.6, columnspacing=0.9, borderpad=0.1)
    ax.set_xlim(t0, t1)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK, labelsize=7)


def dim_labels_for(vis_meta: dict) -> dict:
    """Per-modality dimension names for this bundle (AU set depends on tree)."""
    out = dict(VF.DIM_LABELS)
    out["au"] = VF.au_labels((vis_meta or {}).get("dirs", {}).get("au_dir"))
    return out


def available_modalities(vis_meta: dict, sid: str) -> list[str]:
    """Modalities with a readable npz for this session, in display order."""
    have = VF.load_session(vis_meta, sid)
    return [m for m in ("gaze", "headpose", "au") if m in have]


def fixed_features_figure(vis_meta: dict, sid: str, t0: float, t1: float,
                          t_event: float, *,
                          transform: str = "rms",
                          modality: str | None = None,
                          dim_index: int | None = None,
                          compare_raw: bool = True,
                          vad=None, frame_hz: float = 50.0,
                          win: tuple[float, float] | None = None,
                          title: str = ""):
    """Line plots of the visual features around one event.

    Rows are transforms, columns are modalities. ``compare_raw`` prepends the
    raw row whenever a dynamic transform is selected: "raw carries a
    per-speaker offset, the dynamic form does not" is only legible when both
    are drawn over the same dimension on the same time axis.

    ``dim_index`` pins one feature dimension (e.g. gaze pitch alone). Direction
    is what a single dimension buys -- "looks up before backchannelling" is a
    statement about the sign of pitch -- so the signed transforms keep their
    sign and the y-axis is not forced through zero.

    ``vad`` (frame-rate speech activity) and ``win`` (the scored window, in
    seconds) supply the context the features alone cannot: whether the mover
    was speaking, and which span the model was actually judged on.

    Returns ``None`` when the session has no usable visual-feature npz."""
    curves = VF.transform_curves(vis_meta, sid, t0, t1)
    present = [m for m in ("gaze", "headpose", "au") if m in curves[VF.TRANSFORM_ORDER[0]]]
    mods = [modality] if modality in present else present
    if not mods:
        return None
    _setup_fonts()
    dim_labels = dim_labels_for(vis_meta)

    rows = [transform] if (transform == "raw" or not compare_raw) else ["raw", transform]
    # A dimension index only means something inside one modality -- index 0 is
    # gaze pitch in one column and AU1 in the next -- so it applies only when a
    # single modality is on screen.
    dims = None if (dim_index is None or modality not in present) else [dim_index]

    heights = ([0.5] if vad is not None else []) + [1.6] * len(rows)
    fig, axes = plt.subplots(
        len(heights), len(mods), sharex=True, squeeze=False,
        figsize=(5.2 * len(mods), 1.15 * sum(heights)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.16, "wspace": 0.3},
    )
    r0 = 0
    if vad is not None:
        for c in range(len(mods)):
            _panel_vad(axes[0][c], vad, frame_hz, t0, t1)
        r0 = 1

    for r, tname in enumerate(rows):
        for c, mod in enumerate(mods):
            ax = axes[r0 + r][c]
            t, arr_a, arr_b = curves[tname][mod]
            _line_cell(ax, t, arr_a, arr_b, dim_labels[mod], t0, t1, dims=dims)
            unit = VF.UNIT_LABEL.get(tname, {}).get(mod, "")
            if unit:
                ax.set_ylabel(f"[{unit}]", fontsize=7.5, color=MUTED)
            # Row name inside the axes, matching the other panels: a long
            # ylabel gets clipped once the units line is added under it.
            ax.text(0.003, 0.03, VF.TRANSFORM_LABEL[tname],
                    transform=ax.transAxes, fontsize=8.5, color=INK,
                    fontweight="bold", va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    for row in axes:
        for ax in row:
            if win is not None:
                ax.axvspan(win[0], win[1], color=COL_NG, alpha=0.08, lw=0, zorder=0)
            ax.axvline(t_event, color=INK, lw=0.9, ls=":", alpha=0.7, zorder=4)
    for c, mod in enumerate(mods):
        head = VF.MODALITY_LABEL.get(mod, mod)
        if dim_index is not None and mod in dim_labels:
            lbl = dim_labels[mod]
            if 0 <= dim_index < len(lbl):
                head = f"{head} — {lbl[dim_index]}"
        axes[0][c].set_title(head, fontsize=10, color=INK, fontweight="bold")
        axes[-1][c].set_xlabel("time [s]", fontsize=8, color=INK)
    if title:
        fig.suptitle(title, fontsize=10, color=INK, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.09)
    return fig

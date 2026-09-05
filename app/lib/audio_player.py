"""Detail-figure player: the matplotlib figure and the audio element live in
one HTML component, with a playhead line drawn ON the figure.

`st.audio` / `st.pyplot` cannot exchange the playback position, so the figure
is embedded as a base64 <img> and a vertical line is overlaid at
``x = f(audio.currentTime)``. Because every panel shares the time axis, the
mapping only needs the pixel fractions of t0/t1 on that axis (computed from
matplotlib's data transform). Clicking anywhere on the figure seeks.

Self-contained (base64 WAV + PNG, no CDN, no extra deps).
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import soundfile as sf

COL_HEAD = "#d03b3b"
COL_INK = "#52514e"


def figure_player_html(
    fig,                       # matplotlib Figure from plots.detail_figure
    wav: np.ndarray,           # (2, n) float32 audio crop
    sr: int,
    t0: float,                 # absolute session time of crop start [s]
    t1: float,                 # absolute session time of crop end   [s]
    video: bytes | None = None,  # muted MP4 crop of the same [t0, t1] window
) -> tuple[str, int]:
    """Return (html, initial_height) for st.components.v1.html.

    The iframe auto-fits its height afterwards (same-origin srcdoc lets the
    embedded script resize window.frameElement)."""
    # PNG without bbox_inches="tight": tight would re-crop the canvas and
    # invalidate the axis pixel fractions computed below.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Pixel fractions of t0 / t1 on the shared time axis.
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    ax = fig.axes[-1]
    (x0, _), (x1, _) = ax.transData.transform([(t0, 0.0), (t1, 0.0)])
    fx0, fx1 = float(x0) / w_px, float(x1) / w_px

    buf_a = io.BytesIO()
    sf.write(buf_a, wav.T, sr, format="WAV", subtype="PCM_16")
    wav_b64 = base64.b64encode(buf_a.getvalue()).decode("ascii")

    payload = {
        "t0": round(t0, 3), "t1": round(t1, 3),
        "dur": round(wav.shape[1] / sr, 3),
        "fx0": round(fx0, 5), "fx1": round(fx1, 5),
    }
    aspect = h_px / w_px

    vid_html = ""
    vid_h = 0
    if video is not None:
        vid_b64 = base64.b64encode(video).decode("ascii")
        vid_h = 270          # height estimate at ~900px column (32:9 crop)
        vid_html = f"""
  <video id="vid" muted playsinline preload="auto"
         src="data:video/mp4;base64,{vid_b64}"
         style="display:block; width:100%; height:auto;
                margin:0 0 6px;"></video>"""

    html = f"""
<meta charset="utf-8">
<div style="font-family: sans-serif;">{vid_html}
  <div id="wrap" style="position:relative; cursor:pointer; line-height:0;">
    <img id="fig" src="data:image/png;base64,{png_b64}"
         style="width:100%; display:block;" draggable="false">
    <div id="head" style="position:absolute; top:0; bottom:0; width:2px;
         background:{COL_HEAD}; pointer-events:none; display:none;"></div>
  </div>
  <div id="pos" style="font-size:11px; color:{COL_INK}; margin:3px 0;">
    図をクリックするとその時刻から再生します</div>
  <audio id="au" controls style="width:100%; height:32px;"
         src="data:audio/wav;base64,{wav_b64}"></audio>
</div>
<script>
const D = {json.dumps(payload)};
const au = document.getElementById("au");
const head = document.getElementById("head");
const pos = document.getElementById("pos");
const wrap = document.getElementById("wrap");
const vid = document.getElementById("vid");            // null when no video

// audio is the playback master; the muted video follows it.
function sync(force) {{
  if (!vid) return;
  const t = au.currentTime || 0;
  if (force || Math.abs(vid.currentTime - t) > 0.15) vid.currentTime = t;
}}

function update() {{
  const t = au.currentTime || 0;                       // crop-relative [s]
  const frac = D.fx0 + (t / (D.t1 - D.t0)) * (D.fx1 - D.fx0);
  head.style.left = (frac * 100) + "%";
  head.style.display = "block";
  pos.textContent = "▶ " + (D.t0 + t).toFixed(2) + " s (セッション時刻)"
                  + " / 図をクリックでシーク";
  sync(false);
}}

let raf = null;
function loop() {{ update(); raf = requestAnimationFrame(loop); }}
au.addEventListener("play",  () => {{ if (!raf) loop();
                                      if (vid) {{ sync(true); vid.play().catch(() => {{}}); }} }});
au.addEventListener("pause", () => {{ cancelAnimationFrame(raf); raf = null; update();
                                      if (vid) {{ vid.pause(); sync(true); }} }});
au.addEventListener("ended", () => {{ cancelAnimationFrame(raf); raf = null; update();
                                      if (vid) vid.pause(); }});
au.addEventListener("seeked", () => {{ update(); sync(true); }});

wrap.addEventListener("click", (ev) => {{
  const r = wrap.getBoundingClientRect();
  const frac = (ev.clientX - r.left) / r.width;
  const t = (frac - D.fx0) / (D.fx1 - D.fx0) * (D.t1 - D.t0);
  au.currentTime = Math.min(Math.max(t, 0), D.dur - 0.01);
  update();
  if (au.paused) au.play();
}});

// Fit the Streamlit iframe to the (responsive) content height.
function fit() {{
  if (window.frameElement)
    window.frameElement.style.height = (document.body.scrollHeight + 8) + "px";
}}
document.getElementById("fig").addEventListener("load", fit);
window.addEventListener("resize", fit);
// the video grows the layout only once its metadata arrives -> re-fit then
if (vid) {{
  vid.addEventListener("loadedmetadata", fit);
  vid.addEventListener("loadeddata", fit);
}}
// catch any late layout change (font swap, slow data-URI decode, ...)
new ResizeObserver(fit).observe(document.body);
fit();
update();
</script>
"""
    # Initial guess before the auto-fit kicks in (assumes ~900px column).
    est_height = int(900 * aspect) + 80 + vid_h
    return html, est_height

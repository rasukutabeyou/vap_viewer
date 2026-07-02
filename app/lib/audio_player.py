"""Self-contained HTML audio player with a waveform seekbar.

`st.audio` cannot expose the playback position, so the detail view embeds a
small <audio> element plus a canvas waveform (A above / B below, same colors
as the plots) with a moving playhead. Clicking the canvas seeks. The audio
crop is inlined as a base64 WAV data URI -- no extra dependencies, works in
any browser Streamlit runs in.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import soundfile as sf

COL_A = "#2a78d6"
COL_B = "#eb6834"
COL_HEAD = "#d03b3b"
COL_MARK = "#52514e"
COL_GRID = "#e6e5e1"


def _envelope(w: np.ndarray, n_bins: int) -> list[float]:
    """Per-bin peak amplitude in [0, 1], rounded to keep the HTML small."""
    n = len(w)
    if n == 0:
        return []
    edge = np.linspace(0, n, n_bins + 1, dtype=int)
    peak = np.array([np.abs(w[a:b]).max() if b > a else 0.0
                     for a, b in zip(edge[:-1], edge[1:])])
    top = float(peak.max())
    if top > 0:
        peak = peak / top
    return [round(float(v), 3) for v in peak]


def audio_player_html(
    wav: np.ndarray,           # (2, n) float32
    sr: int,
    t0: float,                 # absolute session time of the crop start [s]
    marker_sec: float | None = None,   # absolute event time to mark
    n_bins: int = 700,
    wave_height: int = 96,
) -> tuple[str, int]:
    """Return (html, component_height) for st.components.v1.html."""
    buf = io.BytesIO()
    sf.write(buf, wav.T, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    dur = wav.shape[1] / sr
    payload = {
        "envA": _envelope(wav[0], n_bins),
        "envB": _envelope(wav[1], n_bins),
        "t0": round(t0, 3),
        "dur": round(dur, 3),
        "marker": round(marker_sec - t0, 3) if marker_sec is not None else None,
        "colA": COL_A, "colB": COL_B, "colHead": COL_HEAD,
        "colMark": COL_MARK, "colGrid": COL_GRID,
        "H": wave_height,
    }

    html = f"""
<meta charset="utf-8">
<div style="font-family: sans-serif;">
  <canvas id="wf" style="width:100%; height:{wave_height}px; display:block;
          border:1px solid {COL_GRID}; border-radius:4px; cursor:pointer;"></canvas>
  <div id="pos" style="font-size:11px; color:{COL_MARK}; margin:2px 0;"></div>
  <audio id="au" controls style="width:100%; height:32px;"
         src="data:audio/wav;base64,{b64}"></audio>
</div>
<script>
const D = {json.dumps(payload)};
const au = document.getElementById("au");
const cv = document.getElementById("wf");
const pos = document.getElementById("pos");
const ctx = cv.getContext("2d");

function draw() {{
  const w = cv.clientWidth, h = D.H;
  cv.width = w * devicePixelRatio; cv.height = h * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2, n = D.envA.length, bw = w / n;
  ctx.strokeStyle = D.colGrid; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  // A: upper half, B: lower half (peak envelopes)
  ctx.fillStyle = D.colA;
  for (let i = 0; i < n; i++) {{
    const a = D.envA[i] * (mid - 3);
    ctx.fillRect(i * bw, mid - a, Math.max(bw - 0.4, 0.6), a);
  }}
  ctx.fillStyle = D.colB;
  for (let i = 0; i < n; i++) {{
    const b = D.envB[i] * (mid - 3);
    ctx.fillRect(i * bw, mid, Math.max(bw - 0.4, 0.6), b);
  }}
  // event marker (dotted)
  if (D.marker !== null && D.marker >= 0 && D.marker <= D.dur) {{
    const x = D.marker / D.dur * w;
    ctx.strokeStyle = D.colMark; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.setLineDash([]);
  }}
  // playhead + played-region shading
  const t = au.currentTime || 0;
  const x = Math.min(t / D.dur, 1) * w;
  ctx.fillStyle = "rgba(82,81,78,0.10)";
  ctx.fillRect(0, 0, x, h);
  ctx.strokeStyle = D.colHead; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  pos.textContent = "▶ " + (D.t0 + t).toFixed(2) + " s (セッション時刻) / クリックでシーク";
}}

let raf = null;
function loop() {{ draw(); raf = requestAnimationFrame(loop); }}
au.addEventListener("play",  () => {{ if (!raf) loop(); }});
au.addEventListener("pause", () => {{ cancelAnimationFrame(raf); raf = null; draw(); }});
au.addEventListener("seeked", draw);
au.addEventListener("ended", () => {{ cancelAnimationFrame(raf); raf = null; draw(); }});
cv.addEventListener("click", (ev) => {{
  const r = cv.getBoundingClientRect();
  au.currentTime = (ev.clientX - r.left) / r.width * D.dur;
  draw();
}});
window.addEventListener("resize", draw);
draw();
</script>
"""
    return html, wave_height + 70

"""Event-video export: the cropped face video stacked on top of the detail
figure, with the red playhead line sweeping across the figure in sync --
the same picture the in-app player shows, rendered to a standalone MP4.

The playhead is an ffmpeg overlay: a 2px color bar whose x position is a
linear function of t, using the same t0/t1 pixel mapping as the player
(matplotlib data transform of the shared time axis; no bbox_inches="tight").
Audio comes from the source video's own track.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .audio_player import COL_HEAD
from .bundle import _ffmpeg_exe


def export_event_video(fig, video_path: Path, t0: float, t1: float,
                       out_path: Path, fps: int = 25) -> bool:
    """Render one event to ``out_path``. ``fig`` is the detail figure for
    exactly [t0, t1]; the caller closes it. Returns False on failure."""
    exe = _ffmpeg_exe()
    if exe is None:
        return False
    dur = t1 - t0

    # figure PNG + pixel x of t0/t1 (same math as audio_player.figure_player_html)
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    ax = fig.axes[-1]
    (x0, _), (x1, _) = ax.transData.transform([(t0, 0.0), (t1, 0.0)])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig_png = tmp.name
    fig.savefig(fig_png, format="png", dpi=fig.dpi)

    fc = (
        # face band (same crop as bundle.read_video_crop), figure width
        f"[0:v]crop=iw:ih/2:0:ih/4,scale={w_px}:-2[cam];"
        # playhead: 2px bar swept left->right over the time axis
        f"color=c=0x{COL_HEAD.lstrip('#')}:s=2x{h_px}:r={fps}[bar];"
        f"[1:v][bar]overlay=x='{x0:.1f}+(t/{dur:.3f})*{x1 - x0:.1f}':y=0[fig];"
        f"[cam][fig]vstack=inputs=2,pad={w_px}:ceil(ih/2)*2[v]"
    )
    cmd = [exe, "-y", "-loglevel", "error",
           "-ss", f"{max(0.0, t0):.3f}", "-i", str(video_path),
           "-loop", "1", "-framerate", str(fps), "-i", fig_png,
           "-filter_complex", fc,
           "-map", "[v]", "-map", "0:a?",
           "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
           "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k",
           "-movflags", "+faststart", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        Path(out_path).unlink(missing_ok=True)
        return False
    finally:
        Path(fig_png).unlink(missing_ok=True)

"""Standalone-HTML export of one case for reports.

A screenshot of the detail figure cannot play audio, so the export wraps the
existing self-contained player fragment (base64 PNG + WAV, no CDN) in a full
HTML document together with the case facts and the analyst's memo. The file
opens in any browser with no server and no dependencies.
"""

from __future__ import annotations

import datetime
import html
import io
import re

COL_INK = "#52514e"


def safe_filename(*parts: object) -> str:
    """Join parts into a filesystem-safe base name (no extension)."""
    s = "_".join(str(p) for p in parts if p not in (None, ""))
    return re.sub(r"[^\w.\-]+", "-", s)


def figure_png_bytes(fig) -> bytes:
    # No bbox_inches="tight": keep pixel geometry identical to the player.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    return buf.getvalue()


def _info_table(rows: list[tuple[str, object]]) -> str:
    tr = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows)
    return f"<table class='kv'>{tr}</table>"


def _models_table(models: list[dict]) -> str:
    """models: [{"model":..., "pred":..., "score":..., "正誤":...}, ...]"""
    if not models:
        return ""
    cols = list(models[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(m.get(c, '')))}</td>"
                         for c in cols) + "</tr>"
        for m in models)
    return (f"<h2>モデル別判定</h2><table class='models'>"
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def standalone_case_html(
    *,
    title: str,
    info_rows: list[tuple[str, object]],
    memo: str = "",
    player_fragment: str = "",
    models: list[dict] | None = None,
) -> str:
    """Full HTML document: header facts + memo + figure/audio player."""
    memo_html = ""
    if memo.strip():
        memo_html = (f"<h2>メモ</h2><div class='memo'>"
                     f"{html.escape(memo.strip())}</div>")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: sans-serif; color: {COL_INK}; margin: 24px auto;
          max-width: 1100px; padding: 0 16px; }}
  h1 {{ font-size: 18px; }}
  h2 {{ font-size: 14px; margin: 18px 0 6px; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 3px 10px; text-align: left; }}
  table.kv th {{ background: #f4f3f1; font-weight: normal; }}
  table.models thead th {{ background: #f4f3f1; }}
  .memo {{ white-space: pre-wrap; border-left: 3px solid #ccc;
           padding: 6px 10px; font-size: 13px; background: #fafaf8; }}
  .stamp {{ font-size: 11px; color: #999; margin-top: 20px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{_info_table(info_rows)}
{_models_table(models or [])}
{memo_html}
<h2>詳細図{"・音声" if player_fragment else ""}</h2>
{player_fragment}
<div class="stamp">VAP error-case viewer で書き出し ({stamp})</div>
</body>
</html>
"""

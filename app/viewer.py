"""VAP error-case viewer (Streamlit, bundle-only -- no vapx/torch).

Run:
    streamlit run app/viewer.py -- --bundles-dir bundles --audio-root /path/to/wavs

Two modes:
  * 単一モデル: browse one bundle's error cases (list -> detail).
  * 比較:       join >= 2 bundles on event_key, filter by per-model outcome
                (e.g. "CPC is correct but stream-KV is wrong"), overlay curves.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import bundle as B          # noqa: E402
from lib import export as X          # noqa: E402
from lib import notes as N           # noqa: E402
from lib import plots as P           # noqa: E402
from lib.audio_player import figure_player_html   # noqa: E402


# --------------------------------------------------------------------------
# CLI / page setup
# --------------------------------------------------------------------------

def _cli_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundles-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "bundles")
    p.add_argument("--audio-root", type=Path, default=None,
                   help="root of the shared wav storage (audio is referenced, "
                        "not bundled).")
    p.add_argument("--notes-file", type=Path, default=None,
                   help="JSON file for memos/bookmarks "
                        "(default: <bundles-dir>/notes.json).")
    args, _ = p.parse_known_args()
    return args


st.set_page_config(page_title="VAP error-case viewer", layout="wide")
ARGS = _cli_args()
NOTES_PATH = ARGS.notes_file or ARGS.bundles_dir / "notes.json"

TASK_LABEL = {"shift_hold": "shift_hold (S/H)", "shift_pred": "shift_pred (見逃し中心)"}


# -- memo / bookmark callbacks (run BEFORE the script reruns, so the list
#    already shows the updated note when the page redraws) ------------------

def _save_memo(event_key: str, ctx: dict) -> None:
    N.update_note(NOTES_PATH, event_key,
                  memo=st.session_state.get(f"memo_{event_key}", ""),
                  context=ctx)

def _save_bookmark(event_key: str, ctx: dict) -> None:
    N.update_note(NOTES_PATH, event_key,
                  bookmark=st.session_state.get(f"bm_{event_key}", False),
                  context=ctx)


@st.cache_data(show_spinner=False)
def _cases(bundle_dir: str) -> pd.DataFrame:
    return B.load_cases(bundle_dir)


@st.cache_data(show_spinner=False)
def _meta(bundle_dir: str) -> dict:
    return B.load_meta(bundle_dir)


# --------------------------------------------------------------------------
# sidebar: mode / bundle / task / filters
# --------------------------------------------------------------------------

st.sidebar.title("VAP error cases")
names = B.list_bundles(ARGS.bundles_dir)
if not names:
    st.error(f"バンドルが見つかりません: {ARGS.bundles_dir}\n"
             "build/extract_error_cases.py で生成してください。")
    st.stop()

mode = st.sidebar.radio("モード", ["単一モデル", "比較"], horizontal=True)


def _bundle_label(n: str) -> str:
    """Flag debug bundles extracted with --limit-sessions."""
    try:
        return f"{n} ⚠partial" if _meta(str(ARGS.bundles_dir / n)).get("limited") else n
    except Exception:
        return n


full_names = [n for n in names if not _meta(str(ARGS.bundles_dir / n)).get("limited")]
default_names = (full_names or names)

if mode == "単一モデル":
    sel_names = [st.sidebar.selectbox("バンドル", names,
                                      index=names.index(default_names[0]),
                                      format_func=_bundle_label)]
else:
    sel_names = st.sidebar.multiselect("バンドル (2つ以上)", names,
                                       default=default_names[:2],
                                       format_func=_bundle_label)
    if len(sel_names) < 2:
        st.info("比較モードではバンドルを2つ以上選んでください。")
        st.stop()

dirs = {n: str(ARGS.bundles_dir / n) for n in sel_names}
metas = {n: _meta(d) for n, d in dirs.items()}

# comparison-mode precondition: same event universe (§7.3)
if mode == "比較":
    ref = metas[sel_names[0]]
    for n in sel_names[1:]:
        m = metas[n]
        for key in ("plan_cfg_hash", "frame_hz", "split"):
            if m.get(key) != ref.get(key):
                st.warning(f"⚠ {n} の {key} が {sel_names[0]} と異なります "
                           f"({m.get(key)} != {ref.get(key)}) -- join結果は不正確かもしれません。")

task = st.sidebar.selectbox("タスク", ["shift_hold", "shift_pred"],
                            format_func=lambda t: TASK_LABEL[t])

# -- assemble the working table -------------------------------------------
if mode == "単一モデル":
    df = _cases(dirs[sel_names[0]])
    df = df[df["task"] == task].reset_index(drop=True)
else:
    base_cols = ["event_key", "session", "task", "t_sec", "silence_start",
                 "silence_end", "pre_speaker", "post_speaker", "gold", "threshold"]
    joined = None
    for n in sel_names:
        c = _cases(dirs[n])
        c = c[c["task"] == task]
        keep = c[base_cols + ["pred", "score", "correct"]].rename(
            columns={"pred": f"pred_{n}", "score": f"score_{n}",
                     "correct": f"correct_{n}", "threshold": f"threshold_{n}"})
        if joined is None:
            joined = keep
        else:
            keep = keep.drop(columns=[col for col in base_cols
                                      if col not in ("event_key",)] ,
                             errors="ignore")
            joined = joined.merge(keep, on="event_key", how="inner")
    df = joined.reset_index(drop=True)

# unfiltered snapshot: the detail view lists neighbouring events regardless
# of the sidebar filters (comparison mode keeps per-model columns this way)
df_unfiltered = df.copy()

meta0 = metas[sel_names[0]]

# -- filters ----------------------------------------------------------------
notes = N.load_notes(NOTES_PATH)   # {event_key: {"memo","bookmark",...}}

st.sidebar.subheader("フィルタ")
if st.sidebar.checkbox("★ ブックマークのみ", value=False):
    marked = {k for k, v in notes.items() if v.get("bookmark")}
    df = df[df["event_key"].isin(marked)]
sessions = sorted(df["session"].unique())
sel_sessions = st.sidebar.multiselect("セッション (空=全て)", sessions)
if sel_sessions:
    df = df[df["session"].isin(sel_sessions)]

if mode == "単一モデル":
    only_ng = st.sidebar.checkbox("NGのみ", value=True)
    if only_ng:
        df = df[~df["correct"]]
    gold_vals = sorted(df["gold"].unique())
    sel_gold = st.sidebar.multiselect("gold (空=全て)", gold_vals)
    if sel_gold:
        df = df[df["gold"].isin(sel_gold)]
    pred_vals = sorted(df["pred"].unique())
    sel_pred = st.sidebar.multiselect("pred (空=全て)", pred_vals)
    if sel_pred:
        df = df[df["pred"].isin(sel_pred)]
    if len(df):
        lo, hi = float(df["score"].min()), float(df["score"].max())
        if lo < hi:
            r = st.sidebar.slider("スコア範囲", lo, hi, (lo, hi))
            df = df[(df["score"] >= r[0]) & (df["score"] <= r[1])]
else:
    patterns = ["すべて", "いずれかNG", "全モデルNG"] + \
               [f"{n} のみNG" for n in sel_names]
    pat = st.sidebar.selectbox("正誤パターン", patterns, index=1)
    corr = df[[f"correct_{n}" for n in sel_names]]
    if pat == "いずれかNG":
        df = df[~corr.all(axis=1)]
    elif pat == "全モデルNG":
        df = df[~corr.any(axis=1)]
    elif pat.endswith("のみNG"):
        ng_model = pat[: -len(" のみNG")]
        m = ~df[f"correct_{ng_model}"]
        for n in sel_names:
            if n != ng_model:
                m &= df[f"correct_{n}"]
        df = df[m]

sort_keys = {"確信度(|score-thr|)": "margin", "時刻": "t_sec", "セッション": "session"}
if mode == "比較":
    sort_keys = {"時刻": "t_sec", "セッション": "session"}
    for n in sel_names:
        sort_keys[f"score({n})"] = f"score_{n}"
sk = st.sidebar.selectbox("ソート", list(sort_keys))
asc = st.sidebar.checkbox("昇順", value=True)
df = df.sort_values(sort_keys[sk], ascending=asc).reset_index(drop=True)

# --------------------------------------------------------------------------
# main: summary + list
# --------------------------------------------------------------------------

full = {n: _cases(d) for n, d in dirs.items()}
cols = st.columns(2 + len(sel_names))
cols[0].metric("表示ケース数", len(df))
cols[1].metric("split", meta0.get("split", "?"))
for i, n in enumerate(sel_names):
    t_all = full[n][full[n]["task"] == task]
    acc = t_all["correct"].mean() if len(t_all) else float("nan")
    cols[2 + i].metric(f"{n} 正解率", f"{acc:.3f}",
                       help=f"{task} 全{len(t_all)}件での正解率")

df = df.assign(
    **{"★": df["event_key"].map(lambda k: "⭐" if notes.get(k, {}).get("bookmark") else ""),
       "メモ": df["event_key"].map(lambda k: notes.get(k, {}).get("memo", ""))})
show_cols = ["★", "メモ"] + [c for c in df.columns
                             if c not in ("task", "exp", "silence_start",
                                          "silence_end", "★", "メモ")]
event = st.dataframe(
    df[show_cols],
    height=380, width="stretch", hide_index=True, key="case_table",
    on_select="rerun", selection_mode="single-row",
    column_config={"★": st.column_config.TextColumn("★", width="small"),
                   "メモ": st.column_config.TextColumn("メモ", width="medium")},
)

# Selecting a row shows the detail view. Saving a memo/bookmark rewrites the
# table data, which can drop the dataframe selection -- fall back to the last
# selected event so the detail view survives the save.
sel_rows = event.selection.rows if event and event.selection else []
if sel_rows and sel_rows[0] < len(df):
    case = df.iloc[sel_rows[0]]
    st.session_state["last_event_key"] = case["event_key"]
else:
    hit = df.index[df["event_key"] == st.session_state.get("last_event_key")]
    if len(hit) == 0:
        st.caption("↑ 行をクリックすると詳細を表示します。")
        st.stop()
    case = df.loc[hit[0]]
sid = case["session"]

# --------------------------------------------------------------------------
# detail view
# --------------------------------------------------------------------------

st.divider()
left, right = st.columns([3, 1])
with right:
    margin_sec = st.slider("表示幅 (イベント前後, 秒)", 2.0, 20.0, 6.0, 0.5)
    show_audio = st.checkbox("音声", value=True)

    # -- memo / bookmark (persisted to NOTES_PATH, shared across modes) ----
    ek = case["event_key"]
    note = notes.get(ek, {})
    note_ctx = {"session": sid, "task": task, "t_sec": float(case["t_sec"])}
    st.toggle("★ ブックマーク", value=bool(note.get("bookmark")),
              key=f"bm_{ek}", on_change=_save_bookmark, args=(ek, note_ctx))
    st.text_area("メモ", value=note.get("memo", ""), key=f"memo_{ek}",
                 height=110, placeholder="このケースの特徴・気づきを記録")
    st.button("メモを保存", key=f"savememo_{ek}",
              on_click=_save_memo, args=(ek, note_ctx), width="stretch")
    if note.get("updated"):
        st.caption(f"最終更新: {note['updated']}")

frame_hz = float(meta0["frame_hz"])
t_ev = float(case["t_sec"])
t0, t1 = max(0.0, t_ev - margin_sec), t_ev + margin_sec

# audio crop (referenced storage; may be missing on this machine)
wav = wav_sr = None
if show_audio:
    pl, pr = B.resolve_session_audio(meta0, sid, ARGS.audio_root)
    if pl is None and pr is None:
        right.caption("音声ファイルが見つかりません。--audio-root を指定してください。")
    else:
        wav, wav_sr = B.read_stereo_crop(pl, pr, t0, t1 - t0)

# neighbouring same-task cases in the visible window. Comparison mode uses
# the joined table so the events panel can show every model's outcome.
primary = sel_names[0]
if mode == "比較":
    c_all = df_unfiltered          # already task-filtered, has pred_<m> cols
else:
    c_all = full[primary]
    c_all = c_all[c_all["task"] == task]
cases_win = c_all[(c_all["session"] == sid)
                  & (c_all["t_sec"] >= t0 - 2) & (c_all["t_sec"] <= t1 + 2)]

probs0 = B.load_probs(dirs[primary], sid)

# token sets: comparison mode shows one panel per DISTINCT lang feature set
# (models sharing the same lang_dir produce identical tokens -> one panel).
if mode == "比較":
    _groups: dict[str, list[str]] = {}
    for n in sel_names:
        if metas[n].get("has_tokens"):
            _groups.setdefault(metas[n].get("lang_dir") or n, []).append(n)
    token_sets = []
    for names_g in _groups.values():
        tdf = B.load_tokens(dirs[names_g[0]], sid)
        if tdf is not None:
            token_sets.append((", ".join(names_g), tdf))
else:
    tdf = (B.load_tokens(dirs[primary], sid)
           if metas[primary].get("has_tokens") else None)
    token_sets = [("", tdf)] if tdf is not None else []

case0 = case.to_dict()
if mode == "比較":
    # detail panels describe the FIRST selected bundle; curves are overlaid.
    case0.update(pred=case[f"pred_{primary}"], score=case[f"score_{primary}"],
                 correct=bool(case[f"correct_{primary}"]),
                 threshold=case.get(f"threshold_{primary}", case.get("threshold")),
                 exp=primary)
    overlays = [
        {"name": n,
         "probs": B.load_probs(dirs[n], sid),
         "threshold": float(case[f"threshold_{n}"]),
         "pred": case[f"pred_{n}"],
         "correct": bool(case[f"correct_{n}"])}
        for n in sel_names
    ]
else:
    overlays = None

model_rows = None
if mode == "比較":
    model_rows = [{"model": n,
                   "pred": case[f"pred_{n}"],
                   "score": f"{case[f'score_{n}']:.4f}",
                   "正誤": "OK" if case[f"correct_{n}"] else "NG"}
                  for n in sel_names]

player_html = ""
with left:
    fig = P.detail_figure(
        case=case0, probs=probs0, meta=meta0, cases_win=cases_win,
        t0=t0, t1=t1, wav=wav, wav_sr=wav_sr, wav_t0=t0,
        token_sets=token_sets, overlays=overlays,
    )
    if wav is not None and wav_sr:
        # figure + audio in one component: playhead overlaid on the figure,
        # click anywhere on the figure to seek.
        player_html, height = figure_player_html(fig, wav, wav_sr, t0, t1)
        components.html(player_html, height=height, scrolling=False)
    else:
        st.pyplot(fig, width="stretch")

with right:
    if mode == "比較":
        st.markdown("**モデル別判定**")
        st.dataframe(pd.DataFrame(model_rows), hide_index=True, width="stretch")
    if wav is not None and wav_sr:
        st.caption(f"音声: {t0:.1f}s – {t1:.1f}s (図の下のプレーヤーで再生。"
                   f"再生位置は図上の赤線、点線=イベント時刻)")
    if token_sets:
        st.markdown("**可視トークン**")
        a0, a1 = int(t0 * frame_hz), int(t1 * frame_hz)
        for lbl, tdf in token_sets:
            vis = tdf[(tdf["pos"] < a1) & (tdf["end"] > a0)]
            if lbl:
                st.caption(f"◆ {lbl}")
            txt = {ch: "".join(str(t) for t in vis[vis["ch"] == ch]["text"])
                   for ch in ("L", "R")}
            st.caption(f"A: {txt.get('L', '')}")
            st.caption(f"B: {txt.get('R', '')}")

    # -- export (for reports: a screenshot cannot play audio, the HTML can) --
    st.markdown("**保存 (レポート用)**")
    fname = X.safe_filename("case", sid, f"{t_ev:.1f}s", task)
    info_rows = [("バンドル", ", ".join(sel_names)),
                 ("セッション", sid),
                 ("タスク", task),
                 ("イベント時刻", f"{t_ev:.2f} s"),
                 ("gold", case["gold"]),
                 ("表示範囲", f"{t0:.1f} – {t1:.1f} s")]
    if mode == "単一モデル":
        info_rows += [("pred", case["pred"]),
                      ("score", f"{float(case['score']):.4f}"),
                      ("threshold", f"{float(case['threshold']):.4f}"),
                      ("正誤", "OK" if case["correct"] else "NG")]
    if player_html:
        doc = X.standalone_case_html(
            title=f"VAPケース {sid} @ {t_ev:.2f}s ({task})",
            info_rows=info_rows,
            memo=st.session_state.get(f"memo_{ek}", note.get("memo", "")),
            player_fragment=player_html,
            models=model_rows,
        )
        st.download_button("📄 HTML (図+音声, 単体で再生可)", data=doc,
                           file_name=f"{fname}.html", mime="text/html",
                           width="stretch")
    else:
        st.caption("音声付きHTMLは音声表示ON時のみ保存できます。")
    st.download_button("🖼 PNG (図のみ)", data=X.figure_png_bytes(fig),
                       file_name=f"{fname}.png", mime="image/png",
                       width="stretch")
    if wav is not None and wav_sr:
        _wb = io.BytesIO()
        sf.write(_wb, wav.T, wav_sr, format="WAV", subtype="PCM_16")
        st.download_button("🔊 WAV (音声のみ)", data=_wb.getvalue(),
                           file_name=f"{fname}.wav", mime="audio/wav",
                           width="stretch")

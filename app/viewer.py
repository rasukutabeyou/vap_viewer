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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import bundle as B          # noqa: E402
from lib import plots as P           # noqa: E402
from lib import vis_features as VF   # noqa: E402
from lib.audio_player import figure_player_html   # noqa: E402
from lib.export import export_event_video         # noqa: E402


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
    p.add_argument("--video-root", type=Path, default=None,
                   help="root of the session videos (<sid>.mp4); shown next "
                        "to the detail figure, synced to audio playback.")
    p.add_argument("--au-dir", type=Path, default=None,
                   help="override the AU npz tree used for DISPLAY. Bundles "
                        "record whatever au_dir the checkpoint was trained "
                        "with, which for gaze/headpose models is the recipe "
                        "default (ME-GraphAU) even though those models never "
                        "read AU. Point this at the LibreFace tree to inspect "
                        "the same AU set the AU experiments use.")
    args, _ = p.parse_known_args()
    return args


st.set_page_config(page_title="VAP error-case viewer", layout="wide")
ARGS = _cli_args()

TASK_LABEL = {
    "shift_hold": "shift_hold (S/H)",
    "shift_pred": "shift_pred (見逃し中心)",
    "bc_pred": "bc_pred (BC予測, 見逃し中心)",
    "short_long": "short_long (S/L)",
}
TASK_ORDER = list(TASK_LABEL)


@st.cache_data(show_spinner=False)
def _cases(bundle_dir: str) -> pd.DataFrame:
    return B.load_cases(bundle_dir)


@st.cache_data(show_spinner=False)
def _meta(bundle_dir: str) -> dict:
    return B.load_meta(bundle_dir)


@st.cache_data(show_spinner="動画を切り出し中…", max_entries=8)
def _video_crop(path: str, start_sec: float, dur_sec: float) -> bytes | None:
    return B.read_video_crop(Path(path), start_sec, dur_sec, max_width=960)


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

# offer only the tasks the selected bundles actually contain (old bundles
# were extracted with shift_hold / shift_pred only)
_present = set()
for n in sel_names:
    _present |= set(_cases(dirs[n])["task"].unique())
task_opts = [t for t in TASK_ORDER if t in _present] or ["shift_hold"]
task = st.sidebar.selectbox("タスク", task_opts,
                            format_func=lambda t: TASK_LABEL.get(t, t))

# -- assemble the working table -------------------------------------------
if mode == "単一モデル":
    df = _cases(dirs[sel_names[0]])
    df = df[df["task"] == task].reset_index(drop=True)
else:
    base_cols = ["event_key", "session", "task", "t_sec", "silence_start",
                 "silence_end", "pre_speaker", "post_speaker", "gold", "threshold"]
    # schema v2 eval-window columns, only when every bundle has them
    if all("win_start" in _cases(d).columns for d in dirs.values()):
        base_cols += ["win_start", "win_end"]
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
st.sidebar.subheader("フィルタ")
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

show_cols = [c for c in df.columns
             if c not in ("task", "exp", "silence_start", "silence_end",
                          "win_start", "win_end")]
event = st.dataframe(
    df[show_cols],
    height=380, width="stretch", hide_index=True,
    on_select="rerun", selection_mode="single-row",
)

# --------------------------------------------------------------------------
# detail figure for one case row (shared by the detail view and video export)
# --------------------------------------------------------------------------

def _detail_figure_for(case_row, margin: float, with_wav: bool = True):
    """(fig, wav, wav_sr, token_sets, t0, t1) honouring single/comparison
    mode. ``case_row`` is one row of ``df``."""
    sid = case_row["session"]
    t_ev = float(case_row["t_sec"])
    t0, t1 = max(0.0, t_ev - margin), t_ev + margin

    # audio crop (referenced storage; may be missing on this machine)
    wav = wav_sr = None
    if with_wav:
        pl, pr = B.resolve_session_audio(meta0, sid, ARGS.audio_root)
        if pl is not None or pr is not None:
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

    case0 = case_row.to_dict()
    if mode == "比較":
        # detail panels describe the FIRST selected bundle; curves are overlaid.
        case0.update(pred=case_row[f"pred_{primary}"],
                     score=case_row[f"score_{primary}"],
                     correct=bool(case_row[f"correct_{primary}"]),
                     threshold=case_row.get(f"threshold_{primary}",
                                            case_row.get("threshold")),
                     exp=primary)
        overlays = [
            {"name": n,
             "probs": B.load_probs(dirs[n], sid),
             "threshold": float(case_row[f"threshold_{n}"]),
             "pred": case_row[f"pred_{n}"],
             "correct": bool(case_row[f"correct_{n}"])}
            for n in sel_names
        ]
    else:
        overlays = None

    fig = P.detail_figure(
        case=case0, probs=probs0, meta=meta0, cases_win=cases_win,
        t0=t0, t1=t1, wav=wav, wav_sr=wav_sr, wav_t0=t0,
        token_sets=token_sets, overlays=overlays,
    )
    return fig, wav, wav_sr, token_sets, t0, t1


# Visual-feature curves are computed from the raw vis npz -- independent of
# which model bundle is selected, so any selected bundle recording 'vis' dirs
# works. Used by the 視覚特徴 draw mode in the detail view further down.
vis_meta = next((m.get("vis") for m in metas.values() if m.get("vis")), None)

# --au-dir retargets the AU column without touching the bundle. Copy rather
# than mutate: metas is cached across reruns, and the recorded au_dir is the
# provenance of what the model was trained on -- only the display moves.
au_override_note = None
if vis_meta is not None and ARGS.au_dir is not None:
    vis_meta = {**vis_meta, "dirs": {**vis_meta["dirs"], "au_dir": str(ARGS.au_dir)}}
    # Models that actually consume AU would now be shown features they never
    # saw, which is worth saying out loud rather than silently redrawing.
    consumers = [n for n in sel_names
                 if "au" in (metas[n].get("vis") or {}).get("modalities", [])
                 and (metas[n]["vis"]["dirs"].get("au_dir") != str(ARGS.au_dir))]
    if consumers:
        au_override_note = (
            f"AU表示を --au-dir で差し替えています: {ARGS.au_dir}。"
            f"{', '.join(consumers)} は別のAU ("
            + metas[consumers[0]]["vis"]["dirs"]["au_dir"] + ") で学習されており、"
            "表示中のAUはモデルが見たものではありません。")

# --------------------------------------------------------------------------
# video export: check events in the table below, then save their clips
# --------------------------------------------------------------------------

with st.expander("🎬 エクスポート (チェックしたイベントの動画を保存)"):
    if not len(df):
        st.caption("対象のイベントがありません。")
    else:
        exp_cols = [c for c in ("session", "t_sec", "gold", "pred") if c in df.columns]
        exp_cols += [c for c in df.columns if c.startswith("correct")]
        edit = df[exp_cols].copy()
        edit.insert(0, "保存", False)
        c1, c2 = st.columns([1, 3])
        exp_margin = c1.number_input("イベント前後 [秒]", 1.0, 30.0, 6.0, 0.5)
        out_dir = c2.text_input(
            "保存先ディレクトリ",
            str(Path(__file__).resolve().parent.parent / "exports"))
        # no explicit key: streamlit then keys the widget on its data, so the
        # checkboxes reset whenever the filtered table changes (stale checks
        # would otherwise point at different rows)
        checked = st.data_editor(edit, hide_index=True, height=250,
                                 width="stretch", disabled=exp_cols)
        picked = df[checked["保存"].to_numpy()]
        n_pick = len(picked)

        if ARGS.video_root is None:
            st.caption("--video-root が未指定のため動画は保存できません。")
        elif st.button(f"🎬 選択した {n_pick} 件の動画を保存", disabled=n_pick == 0):
            outp = Path(out_dir)
            outp.mkdir(parents=True, exist_ok=True)
            prog = st.progress(0.0)
            saved, failed = [], []
            for i, (_, r) in enumerate(picked.iterrows()):
                t_ev = float(r["t_sec"])
                name = f"{r['session']}_{task}_t{t_ev:.2f}s.mp4"
                pv = B.resolve_session_video(r["session"], ARGS.video_root)
                done = False
                if pv is not None:
                    # face video stacked on the detail figure, with the red
                    # playhead swept across -- what the in-app player shows
                    figx, _, _, _, e0, e1 = _detail_figure_for(r, exp_margin)
                    done = export_event_video(figx, pv, e0, e1, outp / name)
                    plt.close(figx)
                (saved if done else failed).append(name)
                prog.progress((i + 1) / n_pick)
            prog.empty()
            if saved:
                st.success(f"{len(saved)} 件を {outp} に保存しました:\n\n"
                           + "\n".join(f"- {n}" for n in saved))
            if failed:
                st.warning("保存できませんでした (動画なし/ffmpeg失敗): "
                           + ", ".join(failed))

sel_rows = event.selection.rows if event and event.selection else []
if not sel_rows:
    st.caption("↑ 行をクリックすると詳細を表示します。")
    st.stop()

case = df.iloc[sel_rows[0]]
sid = case["session"]

# --------------------------------------------------------------------------
# detail view
# --------------------------------------------------------------------------

st.divider()
left, right = st.columns([3, 1])
with right:
    margin_sec = st.slider("表示幅 (イベント前後, 秒)", 2.0, 20.0, 6.0, 0.5)
    show_audio = st.checkbox("音声", value=True)
    show_video = (st.checkbox("動画", value=False)
                  if ARGS.video_root is not None else False)
    # Two exclusive views of the same case: the prediction view is where an
    # oddity is spotted, the feature view is where its cause is checked. Showing
    # both at once made each of them small enough to be useless.
    view = st.radio("描画モード", ["発話活動予測", "視覚特徴"],
                    index=0, horizontal=True,
                    disabled=vis_meta is None,
                    help=("視覚特徴モードは vis 情報を持つバンドルでのみ有効"
                          if vis_meta is None else None))
    vis_opts: dict = {}
    if view == "視覚特徴" and vis_meta is not None:
        mods = P.available_modalities(vis_meta, sid)
        if mods:
            labels = P.dim_labels_for(vis_meta)
            vis_opts["transform"] = st.radio(
                "変換", VF.TRANSFORM_ORDER, index=VF.TRANSFORM_ORDER.index("rms"),
                format_func=lambda t: VF.TRANSFORM_LABEL[t])
            vis_opts["modality"] = st.selectbox(
                "モダリティ", mods + ["すべて"],
                format_func=lambda m: VF.MODALITY_LABEL.get(m, m))
            if vis_opts["modality"] == "すべて":
                vis_opts["modality"] = None
            else:
                names = labels[vis_opts["modality"]]
                pick = st.selectbox("次元", ["すべて"] + list(names))
                vis_opts["dim_index"] = (None if pick == "すべて"
                                         else list(names).index(pick))
            vis_opts["compare_raw"] = st.checkbox(
                "Raw と並べる", value=True,
                help="選んだ変換の上に Raw を同じ次元・同じ時間軸で描きます")
            vis_opts["show_vad"] = st.checkbox("発話区間を重ねる", value=True)

frame_hz = float(meta0["frame_hz"])
fig, wav, wav_sr, token_sets, t0, t1 = _detail_figure_for(
    case, margin_sec, with_wav=show_audio)
if show_audio and wav is None:
    right.caption("音声ファイルが見つかりません。--audio-root を指定してください。")

# video crop (needs audio playback as the sync master)
video_bytes = None
if show_video and wav is not None:
    pv = B.resolve_session_video(sid, ARGS.video_root)
    if pv is None:
        right.caption(f"動画が見つかりません: {ARGS.video_root}/{sid}.mp4")
    else:
        video_bytes = _video_crop(str(pv), t0, t1 - t0)
        if video_bytes is None:
            right.caption("動画の切り出しに失敗しました (ffmpeg が必要です)。")
elif show_video:
    right.caption("動画は音声プレーヤーと同期再生のため、音声も有効にしてください。")

with left:
    if view == "発話活動予測":
        if wav is not None and wav_sr:
            # figure + audio in one component: playhead overlaid on the figure,
            # click anywhere on the figure to seek.
            html, height = figure_player_html(fig, wav, wav_sr, t0, t1,
                                              video=video_bytes)
            components.html(html, height=height, scrolling=False)
        else:
            st.pyplot(fig, width="stretch")

    if view == "視覚特徴" and vis_meta is not None and vis_opts:
        t_ev = float(case["t_sec"])
        fhz = float(meta0["frame_hz"])
        # win_start/win_end are frame indices of the span the model was scored
        # on; shading it separates "the model looked here" from "the event is
        # here", which the event line alone cannot show.
        win = None
        if case.get("win_start") is not None and case.get("win_end") is not None:
            win = (float(case["win_start"]) / fhz, float(case["win_end"]) / fhz)
        figv = P.fixed_features_figure(
            vis_meta, sid, t0, t1, t_ev,
            transform=vis_opts["transform"],
            modality=vis_opts.get("modality"),
            dim_index=vis_opts.get("dim_index"),
            compare_raw=vis_opts.get("compare_raw", True),
            vad=(B.load_probs(dirs[sel_names[0]], sid)["vad"]
                 if vis_opts.get("show_vad") else None),
            frame_hz=fhz, win=win,
            title=f"{sid}  {task}  t={t_ev:.2f}s  "
                  f"gold={case['gold']} pred={case['pred']}"
                  f" [{'OK' if case['correct'] else 'NG'}]")
        if figv is None:
            st.caption("このセッションの視覚特徴npzが見つかりません。")
        else:
            st.pyplot(figv, width="stretch")
            plt.close(figv)
        if au_override_note:
            st.warning(au_override_note)

with right:
    if mode == "比較":
        st.markdown("**モデル別判定**")
        rows = [{"model": n,
                 "pred": case[f"pred_{n}"],
                 "score": f"{case[f'score_{n}']:.4f}",
                 "正誤": "OK" if case[f"correct_{n}"] else "NG"}
                for n in sel_names]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
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

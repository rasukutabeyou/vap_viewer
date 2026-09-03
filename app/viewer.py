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
    p.add_argument("--notes-dir", type=Path, default=None,
                   help="directory holding the per-bundle-combination "
                        "memo/bookmark JSONs (default: <bundles-dir>/notes).")
    args, _ = p.parse_known_args()
    return args


st.set_page_config(page_title="VAP error-case viewer", layout="wide")
ARGS = _cli_args()
NOTES_DIR = ARGS.notes_dir or ARGS.bundles_dir / "notes"

TASK_LABEL = {"shift_hold": "shift_hold (S/H)", "shift_pred": "shift_pred (見逃し中心)"}

# (gold, pred) -> 判定タイプフィルタの表示ラベル。データに現れた組だけ選択肢に出す
OUTCOME_LABEL = {
    ("S", "H"): "S→H: SHIFTをHOLDと誤り (見逃し)",
    ("H", "S"): "H→S: HOLDをSHIFTと誤り (早とちり)",
    ("S", "S"): "S→S: 正解",
    ("H", "H"): "H→H: 正解",
    ("pos", "neg"): "pos→neg: SHIFT予測の見逃し (FN)",
    ("pos", "pos"): "pos→pos: 正解 (TP)",
    ("neg", "pos"): "neg→pos: 誤検出 (FP)",
    ("neg", "neg"): "neg→neg: 正解 (TN)",
}

# 詳細図のパネル表示トグル: (単位キー, ラベル, 既定)。p_now/p_future は
# 優先度が低いので既定OFF。キーは固定なのでモード/バンドルを切り替えても残る
PANEL_TOGGLES = [
    ("wave", "波形 A/B", True),
    ("vad", "VAD", True),
    ("events", "S/Hイベント", True),
    ("bins", "binヒートマップ", True),
    ("tokens", "トークン (lang系)", True),
    ("score", "score曲線 (単一モデル)", True),
    ("p_shift", "P(SHIFT)", True),
    ("p_now", "p_now", False),
    ("p_future", "p_future", False),
]

# 比較モードの正誤フィルタ。プリセット(既定=いずれかNG)と、モデル毎の3択
# (先頭 = 既定 = 絞り込まない)。3択なら 3^N 通り全部表せる
OKNG_PATTERNS = ["すべて", "いずれかNG", "全モデルNG", "モデル別指定"]
OKNG_CHOICES = ["指定なし", "正解のみ", "誤りのみ"]


# -- memo / bookmark callbacks (run BEFORE the script reruns, so the list
#    already shows the updated note when the page redraws). 保存先はいま選んで
#    いるバンドル組み合わせのファイル -- ウィジェット生成時に args で束ねる --

def _save_memo(wkey: str, event_key: str, ctx: dict, path: Path,
               bundles: list[str]) -> None:
    N.update_note(path, event_key, memo=st.session_state.get(wkey, ""),
                  context=ctx, bundles=bundles)

def _save_bookmark(wkey: str, event_key: str, ctx: dict, path: Path,
                   bundles: list[str]) -> None:
    N.update_note(path, event_key, bookmark=st.session_state.get(wkey, False),
                  context=ctx, bundles=bundles)


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

# 表示パネルのトグルは下の st.stop()(比較モードでバンドル1つ以下)より後に
# 描かれる。描かれない run では streamlit がキーを掃除してしまうので、
# stop を跨いでも設定が残るよう毎 run ここで書き戻す(無条件の代入が延命になる)。
# 既定値もここで入れて checkbox には value= を渡さない -- value= と
# session_state の併用は streamlit がスタック付きの警告をログに吐くため
for _k, _, _dflt in PANEL_TOGGLES:
    _pk = f"panel_{_k}"
    st.session_state[_pk] = st.session_state.get(_pk, _dflt)
st.session_state["pshift_zoom"] = st.session_state.get("pshift_zoom", False)

# モデル別正誤フィルタも同じ理由で延命する。描かれるのは「モデル別指定」を
# 選んでいる run のバンドルの分だけなので、これが無いとプリセットを一度見て
# 戻す/バンドルを外して戻すだけで 3モデル分の指定が消える。radio は index=0 が
# 既定なので default_value=None 扱いになり、警告ログは出ない
st.session_state["okng_pattern"] = st.session_state.get("okng_pattern",
                                                        OKNG_PATTERNS[1])
for _n in names:
    _ok = f"okng_{_n}"
    st.session_state[_ok] = st.session_state.get(_ok, OKNG_CHOICES[0])


# -- display names: alias (bundles/aliases.json) > meta.json の exp (--name
#    で記録) > ディレクトリ名。内部キー(dirs / 結合列のサフィックス)は
#    ディレクトリ名のまま、画面に出る文字列だけこのラベルを使う。 ----------

ALIASES = B.load_aliases(ARGS.bundles_dir)


def _display_name(n: str) -> str:
    if ALIASES.get(n):
        return ALIASES[n]
    try:
        exp = _meta(str(ARGS.bundles_dir / n)).get("exp")
        if exp and exp != n:
            return exp
    except Exception:
        pass
    return n


LABEL: dict[str, str] = {}
for _n in names:
    _lbl = _display_name(_n)
    if _lbl in LABEL.values():          # 図の列名にも使うので一意にしておく
        _lbl = f"{_lbl} ({_n})"
    LABEL[_n] = _lbl


def _bundle_label(n: str) -> str:
    """Display name; flag debug bundles extracted with --limit-sessions."""
    try:
        limited = _meta(str(ARGS.bundles_dir / n)).get("limited")
    except Exception:
        limited = False
    lbl = LABEL.get(n, n)
    return f"{lbl} ⚠partial" if limited else lbl


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

with st.sidebar.expander("表示名の編集"):
    st.caption("空欄=exp名のまま。bundles/aliases.json に保存され、"
               "凡例・正解率などすべてのラベルに反映されます。")
    for n in names:
        st.text_input(n, value=ALIASES.get(n, ""), key=f"alias_{n}",
                      placeholder="表示名 (例: CPC音声のみ)")
    if st.button("表示名を保存", key="save_aliases", width="stretch"):
        B.save_aliases(ARGS.bundles_dir,
                       {n: st.session_state[f"alias_{n}"].strip()
                        for n in names if st.session_state[f"alias_{n}"].strip()})
        st.rerun()

with st.sidebar.expander("表示パネル"):
    st.caption("詳細図に出すパネル。判定に不要なものを消して画面を軽くできます"
               "(音声の有無は下の「音声」チェックが別に効きます)。")
    for _k, _lbl, _ in PANEL_TOGGLES:
        st.checkbox(_lbl, key=f"panel_{_k}")   # 既定は上で session_state に投入済み
    st.divider()
    st.checkbox("P(SHIFT) を評価窓の判定域まで拡大", key="pshift_zoom")
    st.caption("閾値は 0.1 前後で、評価窓の中の差は数pxしかありません。"
               "ONで評価窓の値と全モデルの閾値が収まる範囲までY軸を拡大します"
               "(窓の外のピークは切れます)。")
visible_panels = {k for k, _, _ in PANEL_TOGGLES if st.session_state[f"panel_{k}"]}

dirs = {n: str(ARGS.bundles_dir / n) for n in sel_names}
metas = {n: _meta(d) for n, d in dirs.items()}

# メモ/★ はいま選んでいるバンドルの組み合わせ単位で保存する (ディレクトリ名
# キー。表示名は編集できるのでキーにしない)。単一モデル = 1個の組み合わせ
NOTES_PATH = N.notes_path(NOTES_DIR, sel_names)
# ウィジェットキーにも組み合わせを混ぜる。混ぜないと key が既に session_state に
# ある扱いになり、組み合わせを切り替えても前の組のメモ文字列が居座る
NOTE_SCOPE = NOTES_PATH.stem
st.sidebar.caption("メモ/★の保存単位: " + " + ".join(LABEL[n] for n in sorted(sel_names)))

# comparison-mode precondition: same event universe (§7.3)
if mode == "比較":
    ref = metas[sel_names[0]]
    for n in sel_names[1:]:
        m = metas[n]
        for key in ("plan_cfg_hash", "frame_hz", "split"):
            if m.get(key) != ref.get(key):
                st.warning(f"⚠ {LABEL[n]} の {key} が {LABEL[sel_names[0]]} と異なります "
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
notes = N.load_notes(NOTES_PATH)   # この組み合わせ専用 {event_key: {"memo",...}}

st.sidebar.subheader("フィルタ")
if st.sidebar.checkbox("★ ブックマークのみ", value=False):
    marked = {k for k, v in notes.items() if v.get("bookmark")}
    df = df[df["event_key"].isin(marked)]
sessions = sorted(df["session"].unique())
sel_sessions = st.sidebar.multiselect("セッション (空=全て)", sessions)
if sel_sessions:
    df = df[df["session"].isin(sel_sessions)]

if mode == "単一モデル":
    # combos from the unfiltered task table so options stay stable while
    # other filters narrow df down
    combos = sorted(set(zip(df_unfiltered["gold"], df_unfiltered["pred"])),
                    key=lambda c: (c[0] == c[1], c))     # 誤り系を先頭に
    opts = {OUTCOME_LABEL.get(c, f"{c[0]}→{c[1]}"): c for c in combos}
    ng_default = [lbl for lbl, (g, p) in opts.items() if g != p]
    sel_out = st.sidebar.multiselect("判定タイプ gold→pred (空=全て)",
                                     list(opts), default=ng_default)
    if sel_out:
        keep = {opts[lbl] for lbl in sel_out}
        df = df[[gp in keep for gp in zip(df["gold"], df["pred"])]]
    if len(df):
        lo, hi = float(df["score"].min()), float(df["score"].max())
        if lo < hi:
            r = st.sidebar.slider("スコア範囲", lo, hi, (lo, hi))
            df = df[(df["score"] >= r[0]) & (df["score"] <= r[1])]
else:
    gold_vals = sorted(df["gold"].unique())
    sel_gold = st.sidebar.multiselect("gold (空=全て)", gold_vals)
    if sel_gold:
        df = df[df["gold"].isin(sel_gold)]
    # 正誤の絞り込み: 安価なプリセット + モデル毎の3択 (指定なし/正解のみ/
    # 誤りのみ)。3択なら「Xのみ正解」も任意の組み合わせも 3^N 通り表せる
    # ("いずれかNG" はORなのでモデル毎のANDでは書けず、プリセットに残す)
    # 既定値は上で session_state に投入済み (index= を渡すと警告ログが出る)
    pat = st.sidebar.selectbox("正誤パターン", OKNG_PATTERNS, key="okng_pattern")
    corr = df[[f"correct_{n}" for n in sel_names]]
    if pat == "いずれかNG":
        df = df[~corr.all(axis=1)]
    elif pat == "全モデルNG":
        df = df[~corr.any(axis=1)]
    elif pat == "モデル別指定":
        st.sidebar.caption("モデル毎に 正解のみ / 誤りのみ を指定 "
                           "(例: A=正解のみ + B=誤りのみ = 「Aだけ当てた」)。")
        for n in sel_names:
            pick = st.sidebar.radio(LABEL[n], OKNG_CHOICES,
                                    horizontal=True, key=f"okng_{n}")
            if pick == "正解のみ":
                df = df[df[f"correct_{n}"]]
            elif pick == "誤りのみ":
                df = df[~df[f"correct_{n}"]]

sort_keys = {"確信度(|score-thr|)": "margin", "時刻": "t_sec", "セッション": "session"}
if mode == "比較":
    sort_keys = {"時刻": "t_sec", "セッション": "session"}
    for n in sel_names:
        sort_keys[f"score({LABEL[n]})"] = f"score_{n}"
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
    cols[2 + i].metric(f"{LABEL[n]} 正解率", f"{acc:.3f}",
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

    # -- memo / bookmark (persisted to NOTES_PATH = この組み合わせ専用) -----
    ek = case["event_key"]
    note = notes.get(ek, {})
    note_ctx = {"session": sid, "task": task, "t_sec": float(case["t_sec"])}
    memo_key, bm_key = f"memo_{NOTE_SCOPE}_{ek}", f"bm_{NOTE_SCOPE}_{ek}"
    note_ctx_args = (ek, note_ctx, NOTES_PATH, sorted(sel_names))
    st.toggle("★ ブックマーク", value=bool(note.get("bookmark")),
              key=bm_key, on_change=_save_bookmark,
              args=(bm_key, *note_ctx_args))
    st.text_area("メモ", value=note.get("memo", ""), key=memo_key,
                 height=110, placeholder="このケースの特徴・気づきを記録")
    st.button("メモを保存", key=f"savememo_{NOTE_SCOPE}_{ek}",
              on_click=_save_memo, args=(memo_key, *note_ctx_args),
              width="stretch")
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
if mode == "比較":
    # events パネルは overlay の name をサフィックスに列を引くので表示名に揃える
    cases_win = cases_win.rename(columns={
        f"{c}_{n}": f"{c}_{LABEL[n]}" for n in sel_names
        for c in ("pred", "score", "correct", "threshold") if LABEL[n] != n})

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
            token_sets.append((", ".join(LABEL[x] for x in names_g), tdf))
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
        {"name": LABEL[n],
         "probs": B.load_probs(dirs[n], sid),
         "threshold": float(case[f"threshold_{n}"]),
         "pred": case[f"pred_{n}"],
         "score": float(case[f"score_{n}"]),
         "correct": bool(case[f"correct_{n}"])}
        for n in sel_names
    ]
else:
    overlays = None

model_rows = None
if mode == "比較":
    model_rows = [{"model": LABEL[n],
                   "pred": case[f"pred_{n}"],
                   "score": f"{case[f'score_{n}']:.4f}",
                   "正誤": "OK" if case[f"correct_{n}"] else "NG"}
                  for n in sel_names]

player_html = ""
with left:
    fig = P.detail_figure(
        case=case0, probs=probs0, meta=meta0, cases_win=cases_win,
        t0=t0, t1=t1, wav=wav, wav_sr=wav_sr, wav_t0=t0,
        token_sets=token_sets, overlays=overlays, visible=visible_panels,
        pshift_zoom=st.session_state["pshift_zoom"],
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
    # event-time snapshot per token set: tokens that became input inside the
    # visible window (end==2^30 = open sentinel, so an end-only filter would
    # return the whole session) and were still valid at the event frame.
    # Shown here and embedded in the HTML export below.
    token_snapshots = []
    if token_sets:
        ev_frame = int(case["silence_start"])
        a0 = int(t0 * frame_hz)
        for lbl, tdf in token_sets:
            act = tdf[(tdf["pos"] >= a0) & (tdf["pos"] <= ev_frame)
                      & (tdf["end"] > ev_frame)]
            txt = {ch: "".join(str(t) for t in act[act["ch"] == ch]["text"])
                   for ch in ("L", "R")}
            token_snapshots.append((lbl, txt))
        st.markdown("**イベント時点の有効トークン**")
        for lbl, txt in token_snapshots:
            if lbl:
                st.caption(f"◆ {lbl}")
            st.caption(f"A: {txt.get('L', '')}")
            st.caption(f"B: {txt.get('R', '')}")
        st.caption("表示窓内で入力に加わり、無音開始フレーム(図の点線)の時点で"
                   "まだ有効だったトークン列。撤回済みの仮説は含みません。")

    # -- export (for reports: a screenshot cannot play audio, the HTML can) --
    st.markdown("**保存 (レポート用)**")
    fname = X.safe_filename("case", sid, f"{t_ev:.1f}s", task)
    info_rows = [("バンドル", ", ".join(
                     LABEL[n] if LABEL[n] == n else f"{LABEL[n]} ({n})"
                     for n in sel_names)),
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
            memo=st.session_state.get(memo_key, note.get("memo", "")),
            player_fragment=player_html,
            models=model_rows,
            token_snapshots=token_snapshots,
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

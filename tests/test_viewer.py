"""UI回帰テスト (Streamlit AppTest) + 詳細図パネルの単体テスト。

実行: .venv/bin/python -m pytest tests -q
実バンドル (bundles/) を読むので、このリポで抽出済みの環境でだけ動きます。
メモ/★の保存先は必ず tmp_path に向ける (本番の bundles/notes/ を汚さない)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app" / "viewer.py"
BUNDLES = REPO / "bundles"

sys.path.insert(0, str(REPO / "app"))
from lib import bundle as B      # noqa: E402
from lib import notes as N       # noqa: E402
from lib import plots as P       # noqa: E402

A_NAME = "audio_cpc_reg"
B_NAME = "lang_kv_sarashina_stream_nospace_reg"
# 両モデルとも誤る shift_hold イベント (どちらのバンドルでも既定フィルタに残る)
EVENT_KEY = "101_1_2|shift_hold|2985|0"

pytestmark = pytest.mark.skipif(
    not (BUNDLES / A_NAME / "meta.json").is_file(),
    reason="bundles/ が未生成の環境ではスキップ")


def _app(notes_dir: Path, timeout: float = 180) -> AppTest:
    sys.argv = ["viewer.py", "--bundles-dir", str(BUNDLES),
                "--notes-dir", str(notes_dir)]
    return AppTest.from_file(str(APP), default_timeout=timeout)


def _n_cases(at: AppTest) -> int:
    return int(at.metric[0].value)


# --------------------------------------------------------------------------
# 1 / 2. 両モードが例外なく描画できる
# --------------------------------------------------------------------------


def test_single_mode_runs(tmp_path):
    at = _app(tmp_path).run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "単一モデル"


def test_compare_mode_runs(tmp_path):
    at = _app(tmp_path).run()
    at.sidebar.radio[0].set_value("比較").run()
    assert not at.exception
    assert len(at.sidebar.multiselect[0].value) == 2


# --------------------------------------------------------------------------
# 6. p_now / p_future は既定OFF、それ以外は既定ON
# --------------------------------------------------------------------------


def test_panel_toggle_defaults(tmp_path):
    at = _app(tmp_path).run()
    assert at.checkbox(key="panel_p_now").value is False
    assert at.checkbox(key="panel_p_future").value is False
    for k in ("wave", "vad", "events", "bins", "tokens", "score", "p_shift"):
        assert at.checkbox(key=f"panel_{k}").value is True


def test_panel_toggle_survives_stop_and_mode_switch(tmp_path):
    at = _app(tmp_path).run()
    at.checkbox(key="panel_p_now").set_value(True).run()
    at.checkbox(key="panel_bins").set_value(False).run()
    # 比較モードでバンドルを1つに減らすと st.stop() でトグルが描かれない run
    at.sidebar.radio[0].set_value("比較").run()
    at.sidebar.multiselect[0].set_value([A_NAME]).run()
    assert at.info                                   # 「2つ以上選んでください」
    at.sidebar.multiselect[0].set_value([A_NAME, B_NAME]).run()
    assert not at.exception
    assert at.checkbox(key="panel_p_now").value is True
    assert at.checkbox(key="panel_bins").value is False


# --------------------------------------------------------------------------
# 4. visible でパネル数が変わる (detail_figure 直叩き)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fig_kwargs():
    d = str(BUNDLES / A_NAME)
    meta = B.load_meta(d)
    cases = B.load_cases(d)
    case = cases[cases["event_key"] == EVENT_KEY].iloc[0]
    sid = case["session"]
    win = cases[(cases["session"] == sid) & (cases["task"] == "shift_hold")]
    return dict(case=case.to_dict(), probs=B.load_probs(d, sid), meta=meta,
                cases_win=win, t0=max(0.0, case["t_sec"] - 6),
                t1=case["t_sec"] + 6)


def _n_axes(kwargs, visible):
    fig = P.detail_figure(**kwargs, visible=visible)
    n = len(fig.axes)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return n


def test_visible_changes_panel_count(fig_kwargs):
    # wav=None / has_tokens=False / 単一モデル
    # -> vad, events, bins, score, p_shift, p_now, p_future = 7
    assert _n_axes(fig_kwargs, None) == 7
    default = {"wave", "vad", "events", "bins", "tokens", "score", "p_shift"}
    assert _n_axes(fig_kwargs, default) == 5          # p_now/p_future が消える
    assert _n_axes(fig_kwargs, {"vad", "p_shift"}) == 2
    assert _n_axes(fig_kwargs, {"p_now"}) == 1        # 1枚でも np.atleast_1d でOK
    assert _n_axes(fig_kwargs, set()) == 1            # 全部OFFでも軸1枚は残す


def test_figure_player_mapping_survives_toggles(fig_kwargs):
    """図プレーヤーの時刻→ピクセル対応 (fig.axes[-1].transData) が
    どのパネル構成でも成立すること (bbox_inches='tight' 禁止の前提)。

    再生ヘッドが依存する性質は「fx0/fx1 が図の内側にあり、パネル構成を変えても
    同じ値であること」。値そのもの (subplots_adjust) は固定しない。"""
    import json
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    from lib.audio_player import figure_player_html
    sr = 8000
    dur = fig_kwargs["t1"] - fig_kwargs["t0"]
    wav = np.zeros((2, int(sr * dur)), dtype=np.float32)
    seen = []
    for vis in (None, {"vad"}, {"wave", "p_shift"}, {"p_now"},
                {"wave", "vad", "events", "bins", "tokens", "score", "p_shift"},
                set()):
        fig = P.detail_figure(**fig_kwargs, wav=wav, wav_sr=sr,
                              wav_t0=fig_kwargs["t0"], visible=vis)
        html, h = figure_player_html(fig, wav, sr,
                                     fig_kwargs["t0"], fig_kwargs["t1"])
        assert h > 0
        d = json.loads(re.search(r"const D = (\{.*?\});", html).group(1))
        assert 0 < d["fx0"] < d["fx1"] < 1, (vis, d)
        seen.append((d["fx0"], d["fx1"]))
        plt.close(fig)
    assert len(set(seen)) == 1, seen        # パネル構成に依らず同じ対応


def test_visible_hidden_panels_do_not_draw(fig_kwargs):
    """非表示にしたパネルの中身が別の軸に紛れ込んでいないこと。"""
    import matplotlib.pyplot as plt
    fig = P.detail_figure(**fig_kwargs, visible={"vad"})
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. メモ/★ は組み合わせ単位に保存され、別の組み合わせからは見えない
# --------------------------------------------------------------------------


def test_notes_path_is_per_combination(tmp_path):
    p1 = N.notes_path(tmp_path, [A_NAME])
    p2 = N.notes_path(tmp_path, [A_NAME, B_NAME])
    p3 = N.notes_path(tmp_path, [B_NAME, A_NAME])
    assert p1 != p2 and p2 == p3            # 順序非依存
    assert p2.name == f"{A_NAME}+{B_NAME}.json"
    # 200文字を超える結合名は sha1 に落ちる
    long = [f"{A_NAME}_{i:03d}" * 2 for i in range(8)]
    assert len(N.notes_path(tmp_path, long).stem) == 12


def test_notes_payload_records_bundles(tmp_path):
    p = N.notes_path(tmp_path, [A_NAME, B_NAME])
    N.update_note(p, EVENT_KEY, memo="x", bundles=[A_NAME, B_NAME])
    assert N.load_bundles(p) == [A_NAME, B_NAME]
    N.update_note(p, EVENT_KEY, memo="")     # 空メモ+★なし -> エントリ削除
    assert N.load_notes(p) == {}
    assert N.load_bundles(p) == [A_NAME, B_NAME]


def _wkeys(bundles) -> tuple[str, str, str]:
    """(memo, bookmark, 保存ボタン) のウィジェットキー: 組み合わせで分かれる。"""
    scope = N.notes_path(Path("."), bundles).stem
    return (f"memo_{scope}_{EVENT_KEY}", f"bm_{scope}_{EVENT_KEY}",
            f"savememo_{scope}_{EVENT_KEY}")


def test_memo_and_bookmark_are_scoped_to_the_combination(tmp_path):
    mk_a, bk_a, sk_a = _wkeys([A_NAME])
    mk_ab, bk_ab, sk_ab = _wkeys([A_NAME, B_NAME])

    # (a) 単一モデル A でメモと★を保存
    at = _app(tmp_path)
    at.session_state["last_event_key"] = EVENT_KEY
    at.run()
    assert not at.exception
    at.text_area(key=mk_a).set_value("Aで気づいたこと").run()
    at.button(key=sk_a).click().run()
    at.toggle(key=bk_a).set_value(True).run()
    assert not at.exception

    path_a = N.notes_path(tmp_path, [A_NAME])
    saved = N.load_notes(path_a)
    assert saved[EVENT_KEY]["memo"] == "Aで気づいたこと"
    assert saved[EVENT_KEY]["bookmark"] is True
    assert N.load_bundles(path_a) == [A_NAME]

    # (b) 比較 {A, B} は別ファイル -> まだ空
    path_ab = N.notes_path(tmp_path, [A_NAME, B_NAME])
    assert not path_ab.is_file()

    at2 = _app(tmp_path)
    at2.session_state["last_event_key"] = EVENT_KEY
    at2.run()
    at2.sidebar.radio[0].set_value("比較").run()
    assert not at2.exception
    assert at2.text_area(key=mk_ab).value == ""
    assert at2.toggle(key=bk_ab).value is False

    # (c) 比較側に書いても単一A側は壊れない
    at2.text_area(key=mk_ab).set_value("比較で気づいたこと").run()
    at2.button(key=sk_ab).click().run()
    assert N.load_notes(path_ab)[EVENT_KEY]["memo"] == "比較で気づいたこと"
    assert N.load_notes(path_a)[EVENT_KEY]["memo"] == "Aで気づいたこと"
    assert sorted(N.load_bundles(path_ab)) == sorted([A_NAME, B_NAME])

    # (d) 逆向き: 単一Aに戻ると比較側のメモは見えず、Aのメモ/★がそのまま残る
    at2.sidebar.radio[0].set_value("単一モデル").run()
    assert not at2.exception
    assert at2.text_area(key=mk_a).value == "Aで気づいたこと"
    assert at2.toggle(key=bk_a).value is True


def test_notes_dir_is_not_picked_up_as_a_bundle(tmp_path):
    (tmp_path / "notes").mkdir()                 # meta.json を持たない
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "meta.json").write_text("{}")
    assert B.list_bundles(tmp_path) == ["real"]
    assert "notes_archive" not in B.list_bundles(BUNDLES)


# --------------------------------------------------------------------------
# 5. 比較モードのモデル別 正解/誤り フィルタ
# --------------------------------------------------------------------------


def test_per_model_ok_ng_filter(tmp_path):
    at = _app(tmp_path).run()
    at.sidebar.radio[0].set_value("比較").run()
    names = list(at.sidebar.multiselect[0].value)
    assert len(names) == 2

    pat = [s for s in at.sidebar.selectbox if s.label == "正誤パターン"][0]
    pat.set_value("モデル別指定").run()
    assert not at.exception
    n_all = _n_cases(at)

    # A=正解のみ -> 「Aは当てた」ケースだけに減る (従来は指定できなかった側)
    at.radio(key=f"okng_{names[0]}").set_value("正解のみ").run()
    n_a_ok = _n_cases(at)
    assert 0 < n_a_ok < n_all

    # さらに B=誤りのみ -> 「AだけがOK」= 旧「BのみNG」と一致するはず
    at.radio(key=f"okng_{names[1]}").set_value("誤りのみ").run()
    n_only_b_ng = _n_cases(at)
    assert 0 < n_only_b_ng < n_a_ok

    task = "shift_hold"
    ca = B.load_cases(str(BUNDLES / names[0]))
    cb = B.load_cases(str(BUNDLES / names[1]))
    m = (ca[ca.task == task][["event_key", "correct"]]
         .merge(cb[cb.task == task][["event_key", "correct"]],
                on="event_key", suffixes=("_a", "_b")))
    assert n_only_b_ng == int(((m.correct_a) & (~m.correct_b)).sum())


def test_per_model_picks_survive_preset_and_bundle_churn(tmp_path):
    """モデル別指定は「モデル別指定」を選んでいる run でしか描かれない。
    延命しないと、プリセットを一度見て戻す/バンドルを外して戻すだけで
    3モデル分の指定が 指定なし に戻ってしまう。"""
    at = _app(tmp_path).run()
    at.sidebar.radio[0].set_value("比較").run()
    names = list(at.sidebar.multiselect[0].value)

    pat = [s for s in at.sidebar.selectbox if s.label == "正誤パターン"][0]
    pat.set_value("モデル別指定").run()
    at.radio(key=f"okng_{names[0]}").set_value("正解のみ").run()
    at.radio(key=f"okng_{names[1]}").set_value("誤りのみ").run()
    n_want = _n_cases(at)

    # プリセットを往復
    pat = [s for s in at.sidebar.selectbox if s.label == "正誤パターン"][0]
    pat.set_value("すべて").run()
    pat = [s for s in at.sidebar.selectbox if s.label == "正誤パターン"][0]
    pat.set_value("モデル別指定").run()
    assert at.radio(key=f"okng_{names[0]}").value == "正解のみ"
    assert at.radio(key=f"okng_{names[1]}").value == "誤りのみ"
    assert _n_cases(at) == n_want

    # バンドルを外して戻す (描かれない run を跨ぐ)
    at.sidebar.multiselect[0].set_value([names[0]]).run()
    at.sidebar.multiselect[0].set_value(names).run()
    assert not at.exception
    assert at.radio(key=f"okng_{names[0]}").value == "正解のみ"
    assert at.radio(key=f"okng_{names[1]}").value == "誤りのみ"
    assert _n_cases(at) == n_want


def test_single_mode_can_select_correct_outcomes(tmp_path):
    """単一モードは既定こそ誤りのみだが、正解の組も選べる (既存挙動の確認)。"""
    at = _app(tmp_path).run()
    ms = [m for m in at.sidebar.multiselect if m.label.startswith("判定タイプ")][0]
    assert any("正解" in o for o in ms.options)
    n_err = _n_cases(at)
    ms.set_value([]).run()                      # 空 = 全件
    assert _n_cases(at) > n_err


def test_bookmark_only_filter_uses_the_combination_file(tmp_path):
    path_a = N.notes_path(tmp_path, [A_NAME])
    N.update_note(path_a, EVENT_KEY, bookmark=True, bundles=[A_NAME])
    at = _app(tmp_path).run()
    bm = [c for c in at.sidebar.checkbox if c.label.startswith("★")][0]
    bm.set_value(True).run()
    assert not at.exception
    assert _n_cases(at) == 1


# --------------------------------------------------------------------------
# 16. P(SHIFT) の評価窓ズーム: 既定OFF・st.stop() を跨いでも残る
# --------------------------------------------------------------------------


def test_pshift_zoom_default_off_and_survives_stop(tmp_path):
    at = _app(tmp_path).run()
    assert at.checkbox(key="pshift_zoom").value is False
    at.checkbox(key="pshift_zoom").set_value(True).run()
    at.sidebar.radio[0].set_value("比較").run()
    at.sidebar.multiselect[0].set_value([A_NAME]).run()      # st.stop() する run
    at.sidebar.multiselect[0].set_value([A_NAME, B_NAME]).run()
    assert not at.exception
    assert at.checkbox(key="pshift_zoom").value is True


# --------------------------------------------------------------------------
# 17. ズームは評価窓の値と閾値を必ず含み、固定レンジより狭くなる
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zoom_case():
    """評価窓の中が閾値よりずっと低い shift_hold ケースを探して返す。

    ハードコードすると「窓の中で曲線が 1.0 近くまで振れるケース」を引いた
    ときに拡大が効かず、テストが本質と無関係に落ちる。"""
    d = str(BUNDLES / A_NAME)
    meta = B.load_meta(d)
    cases = B.load_cases(d)
    fhz = float(meta["frame_hz"])
    zcfg = meta["zero_shot_config"]
    sid = cases[cases["event_key"] == EVENT_KEY].iloc[0]["session"]
    probs = B.load_probs(d, sid)
    win = cases[(cases["session"] == sid) & (cases["task"] == "shift_hold")]
    best = None
    for _, row in win.iterrows():
        c = row.to_dict()
        ws, we = P._eval_window_sec(c, zcfg, fhz)
        seg = P._pshift_curve(probs, c)[int(ws * fhz): int(we * fhz) + 1]
        if not len(seg):
            continue
        m = float(seg.max())
        if best is None or m < best[0]:
            best = (m, c)
    if best is None:
        pytest.skip("評価窓に収まる shift_hold ケースが無い")
    wmax, case = best
    return dict(case=case, probs=probs, meta=meta, cases_win=win,
                t0=max(0.0, case["t_sec"] - 6), t1=case["t_sec"] + 6), wmax


def test_pshift_zoom_fits_the_eval_window(zoom_case):
    import matplotlib.pyplot as plt
    kw, wmax = zoom_case
    thr = float(kw["case"]["threshold"])

    fixed = P.detail_figure(**kw, visible={"p_shift"}, pshift_zoom=False)
    lo_f, hi_f = fixed.axes[-1].get_ylim()
    plt.close(fixed)
    zoomed = P.detail_figure(**kw, visible={"p_shift"}, pshift_zoom=True)
    lo_z, hi_z = zoomed.axes[-1].get_ylim()
    plt.close(zoomed)

    assert (lo_f, hi_f) == (0.0, 1.0)        # shift_hold の従来レンジ
    assert hi_z < hi_f                       # 実際に拡大されている
    assert lo_z <= thr <= hi_z               # 閾値の破線が画面内に残る
    assert hi_z >= wmax and lo_z <= 0        # 評価窓の値も入る

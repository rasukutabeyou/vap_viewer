# vap_viewer — VAP zero-shot 誤りケース確認GUI（研究室配布用・vapx非依存）

利用者向けの使い方は `README.md`、要件定義（v1.1確定+v1.2）は
`~/work/AI_prompt/vapx/error_case_viewer_requirements.md`。vapx 本体の地図は
`~/work/vapx/CLAUDE.md`（このリポで起動したセッションは vapx の auto-memory を読めない）。

## 絶対に守る設計契約
- **`app/` は vapx / torch / checkpoint / GPU に依存しない**（依存は streamlit, pandas,
  numpy, matplotlib, soundfile のみ = `app/requirements.txt`）。モデル前向きはビューアで行わない
- checkpoint に触るのは `build/extract_error_cases.py`（producer専用・vapx環境で1回）だけ。
  実行: `cd ~/work/vapx && uv run --with pandas --with pyarrow python
  ~/work/vap_viewer/build/extract_error_cases.py ...`（vapx の lockfile を汚さない）
- 音声はバンドルに**コピーしない**。閲覧時に `--audio-root`（研究室共有ストレージ）から解決
- **抽出後は一致検証necessario**: バンドルの tp/fp/tn/fn が該当 exp の `zero_shot-test.json` と
  完全一致すること（shift_hold / shift_pred とも）。不一致なら
  そのexpのzero_shot-test.jsonが2026-06-24のlang評価バグ修正**前**の産物でないか疑う
- pred の再現は全タスク閾値ベース `score >= 保存threshold`（shift_hold も argmax ではない）

## 構成
- `app/viewer.py` + `app/lib/{bundle,plots,audio_player,notes,export}.py` — Streamlit。
  単一モデル / 複数モデル比較の2モード。比較joinキーは
  `(session, task, silence_start, pre_speaker)`（イベント集合はVAD由来でモデル非依存）
- `bundles/<exp名>/` — cases.parquet, probs/<session>.npz, tokens/<session>.jsonl, meta.json。
  表示名は `bundles/aliases.json`、メモ/★は `bundles/notes/<組み合わせ>.json`
  （`--notes-dir`。event_keyキー・atomic write）。**組み合わせ = 選択中バンドルの
  ディレクトリ名をソートして `+` 連結**（>200字は sha1[:12]。JSON の `bundles` に
  メンバーを常に記録）。{A,B} で書いたメモは単一Aからは見えない（意図的に厳密分離）。
  ver1.1.1 までの共有 `notes.json` は `bundles/notes_archive/` に隔離済み・互換なし
- 詳細図のパネルは `plots.detail_figure(visible=...)` で出し入れ（単位は
  `wave/vad/events/bins/tokens/score/p_shift/p_now/p_future`、p_now/p_future は既定OFF）。
  UIは `st.session_state["panel_<単位>"]`。全部OFFのときは vad を1枚残す
  （`plt.subplots(0,1)` は作れず、図プレーヤーの transData 対応にも軸が1つ要る）
- `detail_figure(pshift_zoom=True)`（UIは `st.session_state["pshift_zoom"]`・既定OFF・
  `panel_*` と同じ延命が要る）は P(SHIFT) の y 軸を**評価窓の中**の曲線＋
  **全モデルの閾値**に合わせる（`_yrange`）。窓の外まで含めて合わせると発話中に
  0.95 まで振れるので 0..1 のままになり無意味 — 拡大の基準は必ず評価窓。
  拡大時は `set_yticks([0,0.5,1])` を打たない（拡大先に1つも入らないと目盛りが消える）
- 起動: `.venv/bin/streamlit run app/viewer.py -- --bundles-dir bundles --audio-root <wav共有>`

## ハマり所（再発させない）
- vad は uint8 → 符号演算前に float 化（plots で対応済み）
- `clip_on=False` の窓外テキストで bbox 爆発 → clip_on=True
- lang npz の end_positions/finalized_frames は未確定センチネル **2^30**。
  トークンの窓フィルタは **pos 基準**にする(`end > a0` だと open センチネルのせいで
  セッション開始からの全トークンが該当し、パネルも右ペインも全文まみれになる)
- figure_player_html: `bbox_inches='tight'` 禁止（時刻→ピクセル対応が狂う）。
  座標は ax.transData で fx0/fx1 を渡す方式。代わりに `subplots_adjust` で余白を
  詰めるが、上下は**割合固定にしない**（パネルを絞ると図が低くなり suptitle や
  `time [s]` が切れる）。必要インチ数から比率を出し、合計が 0.6 を超えたら
  縮める（vad 1枚=0.84in で `bottom >= top` になり ValueError）。左右は
  fx0/fx1 の前提なので 0.055/0.995 のまま触らない
- チャンネル契約は **L=operator / R=user**（花川氏環境は逆。彼らのバンドルと直接joinしない）
- `st.stop()`（比較モードでバンドル1つ以下）を跨ぐと、その run で描かれなかった
  ウィジェットの session_state キーが掃除される → `panel_*` / `okng_pattern` /
  `okng_<バンドル>` はモード radio の直後で毎run
  `st.session_state[k] = st.session_state.get(k, 既定)` して延命している
  （消すと表示パネル設定やモデル別正誤指定が既定に戻る。`okng_*` は
  「モデル別指定」を選んだ run しか描かれないので、プリセットを往復するだけでも
  消える。回帰テストあり）。既定値はこの代入で入れ、
  `st.checkbox` に `value=` は渡さない（併用すると streamlit が
  "created with a default value but also had its value set via the Session State API"
  をスタック付きでログに吐く）

## 検証方法
UIの回帰は Streamlit `AppTest` = `tests/test_viewer.py`（単一/比較・メモ/★の
組み合わせ別永続化・パネルトグル・モデル別正誤フィルタ・P(SHIFT)の評価窓
ズーム）。実行は
`.venv/bin/python -m pytest tests -q`（pytest は dev専用。`app/requirements.txt`
には入れない）。実 `bundles/` を読むので未生成環境ではskip、メモは必ず tmp_path へ。
描画整合は headless Chrome で実描画確認。抽出器は既存expとのビット一致回帰。

- AppTest は `st.dataframe` の行選択を再現できない → `at.session_state
  ["last_event_key"]` を実 event_key にしてフォールバック経路で詳細を出す。
  引数は `AppTest.from_file` に渡せないので `sys.argv` を差し替える
- メモ系ウィジェットの `key` には**組み合わせを混ぜる**（`memo_<scope>_<ek>`）。
  混ぜないと session_state に値が残り、組み合わせを切り替えても前の組の
  メモ文字列が居座る（`value=` は既存キーがあると無視される）

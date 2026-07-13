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
  表示名は `bundles/aliases.json`、メモ/★は `bundles/notes.json`（event_keyキー・atomic write）
- 起動: `.venv/bin/streamlit run app/viewer.py -- --bundles-dir bundles --audio-root <wav共有>`

## ハマり所（再発させない）
- vad は uint8 → 符号演算前に float 化（plots で対応済み）
- `clip_on=False` の窓外テキストで bbox 爆発 → clip_on=True
- lang npz の end_positions/finalized_frames は未確定センチネル **2^30**。
  トークンの窓フィルタは **pos 基準**にする(`end > a0` だと open センチネルのせいで
  セッション開始からの全トークンが該当し、パネルも右ペインも全文まみれになる)
- figure_player_html: `bbox_inches='tight'` 禁止（時刻→ピクセル対応が狂う）。
  座標は ax.transData で fx0/fx1 を渡す方式
- チャンネル契約は **L=operator / R=user**（花川氏環境は逆。彼らのバンドルと直接joinしない）

## 検証方法
UIの回帰は Streamlit `AppTest`（単一/比較・メモ/★永続化・フィルタの6項目）、
描画整合は headless Chrome で実描画確認。抽出器は既存expとのビット一致回帰。

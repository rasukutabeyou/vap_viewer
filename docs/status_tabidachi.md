# tabidachi 環境での適用状況(花川側・2026-07-03 時点)

`docs/handoff_vis_extraction.md` の手順を tabidachi 実データで実行した際の
判明事項と、修正済み内容・残りの手順のメモ。

## 判明したこと

### 1. 旧 `exp/ablation` 系列(6/23学習・38 vis exp)は現行コードでロード不可

`vapx/features/visual_encoder.py` の**未コミット変更でエンコーダの
アーキテクチャが変わっている**ため、6月学習の checkpoint と形状が合わない:

- 例: gaze delta の checkpoint は `proj: Linear(2→256)→GELU→Linear(256→256)`、
  現行コードは `Linear(2→128)→GELU→Linear(128→256)`(`_make_proj` の
  `mid_dim = hidden_dim // 2` 化)。CNN系も同様(`input_proj+convs` → `_CnnProj` 段階拡張)
- HEAD(ff0e4ca)のコードとも一致しない(HEAD の delta は `[raw,Δ]` concat で入力4次元)。
  **学習時点のコードはリポジトリのどこにも残っていない**(コミット・stash なし)
- 全39exp走査の結果: `audio_only` のみ互換、視覚38expすべて MISMATCH
- → 旧系列をバンドル化するには当時のコード復元が必要。形状だけ合わせても
  変換の意味(例: 単純Δ → multi-lag加重|Δ|)が変わっており zero_shot json と
  一致しないため、実質は**再学習(または当時コードの発掘)が必要**

### 2. 新 enc_compare 系列(ablation_cnn / ablation_mlp)は現行コードと互換(こちらを使う)

- 40条件 = {gaze, headpose, au, vis_all} × {raw, delta, rms, zscore, abd}
  × {cnn(Conv1D×2), mlp}。`run_ablation_cnn.sh && run_ablation_mlp.sh` が実行中
  (7/4 時点: cnn 20exp の学習完了・stage 5 進行中、mlp は未着手)。
  stage 5 が各expの `zero_shot-test.json` を生成する
- **ablation_cnn 全20exp の checkpoint は現行コードでロードOKを確認済み**
  (7/4 読み取りスキャン)。ablation_mlp も現行コードで学習されるので互換のはず
- **音声のみベースラインは旧系列の `exp/ablation/audio_only` を使う**
  (現行コードでロード可)。data/label 設定が新系列と同一
  (同manifest・同frame_hz・同チャンネル対応)であることを確認済みなので、
  ビューアの比較モードで新系列と直接joinできる
- 全て完了後に `bash build/extract_all_tabidachi.sh` で一括バンドル化
  (audio_only + cnn20 + mlp20 = 41バンドル、GPUで1expあたり数分)
- 注意: audio_only の `zero_shot-test.json`(6/29生成)が現行コードで
  再現できるかは未検証(CPU抽出が遅すぎて中断)。**再学習は不要**
  (checkpoint 自体は現行コードでロード可)。抽出時の一致検証が
  MISMATCH になった場合のみ、**評価だけ**を GPU で再実行(数分)してから抽出する:
  `uv run vapx-zero-shot --config conf/ablation/audio_only.yaml --include-dir conf
  --include-dir ../../TEMPLATE/vap1/conf --checkpoint exp/ablation/audio_only/checkpoints/best.pt
  --exp-dir exp/ablation/audio_only`

### 3. CPUでの抽出・zero-shot は非現実的

tabidachi のセッションは長く(~20分)、CPU では **1セッション約20分**
(zero-shot 全体で約24時間)。抽出・zero-shot は GPU が空いてから行うこと。

## vap_viewer 側の修正(このリポジトリ、7/3)

- `build/extract_error_cases.py`
  - manifest のセッションIDキーが `id` 固定だったのを `session` / `id` 両対応に
    (tabidachi は `session`。vapx `iter_session_inputs` と同じ解決順)
  - 視覚エンコーダの checkpoint 形状不一致時に、原因(コード版数の不一致)を
    説明するエラーで停止するように(従来は生の torch エラー)
  - `meta.json` に `vapx_git_dirty` を追加(vapx 作業ツリーが dirty のまま
    生成したバンドルは commit hash がコードを特定しないため)
- `build/extract_all_tabidachi.sh` 追加(ablation_cnn 一括抽出)
- **bc_pred / short_long タスク対応(7/4, schema v2)**: 抽出器が
  BC予測(見逃し中心)と SHORT/LONG 判定もイベント化し
  (`win_start/win_end` 列と `probs/*.npz` の `score_bc` を追加)、
  ビューアの4タスク(一覧・詳細・比較)を audio_only partial バンドルで検証済み。
  イベント窓は `_plan_session` と (start,end,speaker) の完全一致で照合される。
  v1 バンドルもビューアはそのまま読める(タスク選択肢は存在するもののみ表示)

## 残りの手順(GPUが空いたら)

```bash
# 1) stage 5 完了確認(zero_shot-test.json が揃っていること)
ls /home/hanakawa/project/2026/vapx/egs/tabidachi/vap1/exp/ablation_cnn/*/zero_shot-test.json | wc -l   # → 20
ls /home/hanakawa/project/2026/vapx/egs/tabidachi/vap1/exp/ablation_mlp/*/zero_shot-test.json | wc -l   # → 20

# 2) 一括抽出(揃った分だけでも実行可。未完了expは自動スキップ)
cd /home/hanakawa/project/2026/vap_viewer
bash build/extract_all_tabidachi.sh
#   一部だけなら: bash build/extract_all_tabidachi.sh audio_only gaze_cnn_delta gaze_mlp_delta

# 3) ビューア起動(.venv は uv で作成済み)
.venv/bin/streamlit run app/viewer.py -- --bundles-dir bundles
# 音声パス(/autofs/diamond3/.../wav)は manifest の絶対パスがそのまま見えるため
# --audio-root は不要
```

比較の典型パターン(比較モードでバンドルを複数選択):
- **音声のみ vs 視覚あり**: `audio_only` + `gaze_cnn_delta` など
- **固定変換の比較**: `gaze_cnn_raw` + `gaze_cnn_delta` + `gaze_cnn_rms` + `gaze_cnn_zscore`(+`gaze_cnn_abd`)
- **エンコーダ(MLP vs Conv1D×2)の比較**: `gaze_cnn_delta` + `gaze_mlp_delta` のような同モダリティ・同変換ペア

推奨: バンドル生成の前に vapx の未コミット変更(88ファイル)をコミットする
(meta.json に記録される commit hash がコードを特定できるようになる。
`docs/handoff_vis_extraction.md` §3.1)。

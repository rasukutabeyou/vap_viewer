# build/ — バンドル抽出器(producer専用・vapx依存)

`extract_error_cases.py` は vapx の学習済み checkpoint を1回だけ動かし、
ビューアが読む自己完結バンドルを生成します。**配布物には含めません**。

## 実行方法

vapx リポジトリの venv(uv 管理)で動かします。pandas/pyarrow は vapx の
lockfile を汚さないよう `--with` で一時追加します:

```bash
cd ~/work/vapx
uv run --with pandas --with pyarrow python ~/work/vap_viewer/build/extract_error_cases.py \
    --recipe-dir egs/tabidachi/vap1 \
    --exp-dir exp/<EXP_NAME> \
    --split test \
    --out ~/work/vap_viewer/bundles/<バンドル名>
```

主な引数:

| 引数 | 意味 |
|---|---|
| `--recipe-dir` | レシピdir。checkpoint内configの相対パス(manifest等)の基準 |
| `--exp-dir` | `checkpoints/best.pt` と `zero_shot-test.json` をこの中から使う |
| `--checkpoint` + `--zs-json` | 任意のckpt/閾値jsonを直接指定(命名が違うexp向け) |
| `--split` | test / valid / train(検証は test/valid のみ対応) |
| `--name` | バンドルに記録する表示名(省略時 `--out` の basename) |
| `--chunk-sec` `--warmup-sec` `--min-context-sec` | 前向き計算の条件。**json生成時と揃えること**(既定=vapx既定) |
| `--no-tokens` | lang系でもトークンテキストを書き出さない |
| `--limit-sessions N` | デバッグ用(検証スキップ・バンドルに partial 記録) |
| `--allow-mismatch` | 一致検証失敗でも書き出す(結果は meta.json に記録) |

## 何をしているか

1. checkpoint 埋め込み config から feature/model/labeler を再構築
   (`vapx.inference.load_for_inference`)。
2. 全セッションを `vapx.eval.inference.predict_session` で前向き
   (zero-shot 評価と同一のチャンク処理)。lang系は lang特徴を自動配線。
3. `vapx.eval.zero_shot._plan_session` / `_score_session` でイベント抽出・スコアリング
   (評価コードそのものを流用)。
4. `zero_shot-test.json` 保存の閾値で各イベントの pred/correct を確定
   (**shift_hold も閾値ベース**。argmax ではない)。
5. フレームレベルの指標を再計算し、json の tp/fp/tn/fn と**完全一致**を検証。
6. cases.parquet / probs npz / tokens jsonl / meta.json を書き出し。

## 一致検証が失敗したら

- json が **lang未配線バグ修正前**に生成された → `vapx-zero-shot` で再生成。
- 前向き条件(chunk/warmup/min-context)が json 生成時と異なる → 揃える。
- checkpoint が違う(best.pt 以外で評価していた等) → `--checkpoint`/`--zs-json` で明示。
- 切り分け用に meta.json へ vapx の git commit と `plan_cfg_hash` を記録しています。

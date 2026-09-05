#!/usr/bin/env bash
# extract_all_tabidachi.sh -- エンコーダ比較 (enc_compare) 一式をバンドル化する。
#
# 対象 (最大41バンドル):
#   audio_only                     exp/ablation/audio_only   (音声のみベースライン)
#   {gaze,headpose,au,vis_all}_cnn_{raw,delta,rms,zscore,abd}   exp/ablation_cnn/
#   {gaze,headpose,au,vis_all}_mlp_{raw,delta,rms,zscore,abd}   exp/ablation_mlp/
#
# バンドル名 = exp名なので、ビューアの比較モードでそのまま
#   「audio_only vs gaze_cnn_delta vs gaze_mlp_delta」のように選べる
# (全バンドルは同一レシピ/同一イベント空間なので任意の組で比較可)。
#
# 前提: 各expに checkpoints/best.pt と zero_shot-test.json があること
# (run_ablation_{cnn,mlp}.sh の stage 5 が生成する)。無いものはスキップ。
# GPU で1expあたり数分。GPUが学習で埋まっている間は実行しないこと。
#
# Usage:
#   bash build/extract_all_tabidachi.sh              # 揃っているもの全部
#   bash build/extract_all_tabidachi.sh gaze_cnn_delta gaze_mlp_delta audio_only

set -uo pipefail

VAPX=/home/hanakawa/project/2026/vapx
VIEWER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="$VAPX/egs/tabidachi/vap1/exp"

# name -> exp dir (recipe相対)
declare -A EXP_OF
EXP_OF[audio_only]="exp/ablation/audio_only"
for MOD in gaze headpose au vis_all; do
  for TR in raw delta rms zscore abd; do
    EXP_OF[${MOD}_cnn_${TR}]="exp/ablation_cnn/${MOD}_cnn_${TR}"
    EXP_OF[${MOD}_mlp_${TR}]="exp/ablation_mlp/${MOD}_mlp_${TR}"
  done
done

if [[ $# -gt 0 ]]; then
  NAMES=("$@")
else
  NAMES=(audio_only)
  for MOD in gaze headpose au vis_all; do
    for TR in raw delta rms zscore abd; do
      NAMES+=("${MOD}_cnn_${TR}" "${MOD}_mlp_${TR}")
    done
  done
fi

for NAME in "${NAMES[@]}"; do
  REL="${EXP_OF[$NAME]:-}"
  if [[ -z "$REL" ]]; then
    echo "[SKIP] $NAME: 未知のexp名"
    continue
  fi
  EXP="$VAPX/egs/tabidachi/vap1/$REL"
  if [[ ! -f "$EXP/checkpoints/best.pt" ]]; then
    echo "[SKIP] $NAME: checkpoints/best.pt がまだ無い (学習未完了)"
    continue
  fi
  if [[ ! -f "$EXP/zero_shot-test.json" ]]; then
    echo "[SKIP] $NAME: zero_shot-test.json がまだ無い (stage 5 未実行)"
    continue
  fi
  if [[ -f "$VIEWER/bundles/$NAME/meta.json" ]]; then
    echo "[SKIP] $NAME: bundles/$NAME は生成済み"
    continue
  fi
  echo "=== [$NAME] extracting ($REL) ==="
  (cd "$VAPX" && uv run --with pandas --with pyarrow python \
      "$VIEWER/build/extract_error_cases.py" \
      --recipe-dir egs/tabidachi/vap1 \
      --exp-dir "$REL" \
      --split test \
      --out "$VIEWER/bundles/$NAME") \
    || echo "[FAIL] $NAME (ログを確認してください)"
done

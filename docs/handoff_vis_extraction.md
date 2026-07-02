# 引き継ぎ: 視覚(gaze/head/AU)モデルの誤りケースバンドル作成手順

対象: 花川さん側の環境で作業する方(Claude での実行を想定した粒度で書いています)
作成: 2026-07-02(佐飼側 / vap_viewer dev_vis ブランチ)

## 0. これは何をするものか

`vap_viewer` は VAP zero-shot 評価(shift_hold / shift_pred)の**誤りケースを
イベント単位で抽出し、GUI(Streamlit)で波形・確率曲線・音声つきで確認・
モデル間比較する**ツールです。2段構成です:

```
[build/extract_error_cases.py]  vapx依存・GPU・exp毎に1回
   checkpoint + 全セッション前向き → bundles/<name>/ (自己完結・配布可能)
[app/viewer.py]                 vapx/torch非依存・Streamlit
   bundles/ を読むだけ(単一モデル確認 + 複数モデル比較)
```

**あなた(花川側)にお願いしたいのはバンドル生成まで**です。生成された
`bundles/<name>/` ディレクトリ(1expあたり50〜100MB程度、音声は含まない)を
佐飼に渡せば、閲覧・比較はこちらでできます。もちろんそちらでビューアを
起動して使うこともできます(§6)。

抽出器は import した vapx の API を自動判別します。**dev-hanakawa 系統
(predict_session が vis_encoders/vis_feats_a/b を受ける)に対応済み**で、
合成データでの統合テスト(視覚配線済み `run_zero_shot_eval` の出力との
tp/fp/tn/fn 完全一致)を通過しています。

## 1. 前提状況(2026-07-02 に佐飼側から read-only で確認した内容)

- `silver12:/home/hanakawa/project/2026/vapx`(branch `dev-hanakawa`)に
  **未コミットの変更が約87ファイル**あり、肝心の zero-shot 視覚配線
  (`vapx/eval/inference.py` の `predict_session(vis_encoders=...)`、
  `vapx/eval/zero_shot.py` の `run_zero_shot_eval(gaze_dir=...)` 等)は
  この未コミット分に含まれています
- 評価コードに **lang 配線はありません**。`lang_dim > 0` の構成
  (audio_lang / all_vis 等)は、zero-shot でも本抽出器でも **lang が
  供給されないまま**(縮退値)になります。抽出器は警告を出し
  `meta.json` に `lang_eval_wired: false` を記録します(§7)
- exp / data は `/autofs/diamond4/share/users/hanakawa/...` にあり、
  佐飼のアカウントからは Permission denied(そのため実データでの最終確認は
  そちらで行っていただく必要があります)

## 2. 作業手順(概要)

1. vapx の未コミット変更をコミット(再現性のため。バンドルに git commit を記録します)
2. バンドル化したい各 exp について `zero_shot-test.json` を**視覚配線済みの
   評価コードで再生成**
3. `vap_viewer` を入手し、exp 毎に抽出器を実行 → 一致検証が PASS することを確認
4. `bundles/` を佐飼へ受け渡し(または自分でビューア起動)

## 3. 手順詳細

### 3.1 vapx のコミット

```bash
cd /home/hanakawa/project/2026/vapx
git status            # 視覚配線を含む変更が未コミットのはず
git add -A && git commit -m "zero-shot visual wiring 他"
```

理由: 抽出器は `meta.json` に vapx の git commit を記録します。未コミットのまま
だと「どのコードで作ったバンドルか」が後から辿れません。

### 3.2 zero_shot-test.json の再生成(exp 毎)

視覚配線が入る**前**に作られた `zero_shot-test.json` が残っている場合、
それは視覚特徴なしの縮退評価値なので、**必ず再生成**してください
(古い json のままだと抽出器の一致検証が失敗します)。

```bash
cd /home/hanakawa/project/2026/vapx/egs/tabidachi/vap1
uv run vapx-zero-shot \
    --config conf/<この exp を学習した config>.yaml \
    --include-dir conf \
    --checkpoint exp/ablation/<EXP>/checkpoints/best.pt \
    --exp-dir exp/ablation/<EXP>
# → exp/ablation/<EXP>/zero_shot-test.json が更新される
```

注意:
- `--chunk-sec / --warmup-sec / --min-context-sec` は**既定値のまま**に
  してください(抽出器も同じ既定値で前向きします。変えた場合は抽出時に
  同じ値を指定する必要があります)
- config は学習時と同じもの(`exp/<EXP>/config.resolved.json` があれば
  それと同内容のもの)を使ってください

### 3.3 vap_viewer の入手と抽出

`vap_viewer` 一式(`dev_vis` ブランチ)を佐飼から受け取ってください
(リポジトリごとコピーで構いません。`bundles/` は除いてOK)。

exp 毎に抽出器を実行します。**pandas / pyarrow は vapx の lockfile を汚さず
`uv run --with` で一時追加**します:

```bash
cd /home/hanakawa/project/2026/vapx        # ここから実行するのが重要(uv がこの venv を使う)
uv run --with pandas --with pyarrow python <vap_viewer>/build/extract_error_cases.py \
    --recipe-dir egs/tabidachi/vap1 \
    --exp-dir exp/ablation/<EXP> \
    --split test \
    --out <vap_viewer>/bundles/<バンドル名>
```

- `--exp-dir` は `checkpoints/best.pt` と `zero_shot-test.json` を中から使います。
  別名の checkpoint/json を使う場合は `--checkpoint` + `--zs-json` で明示できます
- バンドル名は自由(例: `audio_gaze`, `audio_only`, `all_vis`)。ビューアの
  モデル選択肢・凡例にそのまま表示されます
- 動作確認だけしたい場合は `--limit-sessions 2` を付けると数分で終わります
  (一致検証はスキップされ、バンドルは partial 扱いになります)

正常終了時のログの見どころ:

```
[extract] visual model; modalities=['gaze', ...] dirs={...} roles=(customer,operator) ratio=2
[extract] [1/36] <session> T=... sh=... spred+=...
[verify] shift_hold (test): OK        ← ここが OK であること
[verify] shift_pred (test): OK
[extract] wrote bundle to .../bundles/<名前>
```

### 3.4 一致検証が MISMATCH になったら

抽出器は json の tp/fp/tn/fn と完全一致するまでバンドルを書きません。
失敗時の切り分け(可能性の高い順):

1. `zero_shot-test.json` が**視覚配線前**の生成物 → §3.2 で再生成
2. json 生成時と抽出時で `--chunk-sec / --warmup-sec / --min-context-sec` が
   違う → 揃える
3. json を作った checkpoint と `--exp-dir` の best.pt が違う → `--checkpoint`/
   `--zs-json` で明示
4. json 生成後に vapx のコード(モデル/特徴/評価)を変更した → json 再生成

`meta.json` に vapx の git commit と config ハッシュが記録されるので、
再現条件の照合に使えます。

## 4. 比較モードの前提(重要)

- **同じレシピ(同じ VAD・同じ split・同じ frame_hz)で作ったバンドル同士**
  だけが比較できます。花川側レシピの exp 同士(gaze ablation 等)は問題
  ありません
- **佐飼側のバンドル(audio_cpc_reg 等)とは混ぜられません**。チャンネル
  対応が逆(花川側: L=user/customer、佐飼側: L=operator)で、イベントの
  キーが鏡像になるためです。ビューアはバンドル間の設定ハッシュ不一致を
  警告します

## 5. lang 入り構成(audio_lang / all_vis 等)の扱い

現状の dev-hanakawa 系統の評価には lang 配線が無いため、これらの構成は
**lang ブランチが働かないまま**評価・抽出されます(zero-shot の数値も
バンドルの誤りケースも「lang なし相当」)。抽出器は以下で明示します:

```
[extract] WARNING: model has a lang branch but this vapx has no lang wiring in eval -> ...
```

lang も効かせた評価にしたい場合は、佐飼側 main の zero-shot lang 配線
(`_load_session_lang` + `predict_session(lang=...)`)の取り込みが必要です。
その際は佐飼まで相談してください(マージ済みの抽出器はどちらの系統でも
自動で正しく動きます)。

## 6. ビューアをそちらで使う場合

```bash
cd <vap_viewer>
python3 -m venv .venv && .venv/bin/pip install -r app/requirements.txt   # 初回のみ
.venv/bin/streamlit run app/viewer.py -- \
    --bundles-dir bundles \
    --audio-root <音声wavのルート>   # manifestのパスがそのまま見えるなら不要
```

torch / vapx / GPU は不要です。使い方はリポジトリの `README.md` を参照。

## 7. バンドルの中身(参考)

```
bundles/<name>/
├── cases.parquet      # 1行 = 1イベント×このモデル (gold/pred/score/正誤/時刻)
├── probs/<sid>.npz    # p_now/p_future・binごとの確率・スコア曲線・VAD (float16)
├── tokens/<sid>.jsonl # lang供給時のみ(視覚系統では生成されません)
└── meta.json          # exp/閾値/git commit/一致検証結果/セッション毎のvis有無 等
```

- `meta.json` の `sessions[].vis` に、そのセッションで実際に供給できた
  モダリティが入ります(video 無しセッションは空リスト = 視覚ブランチが
  no-op のまま評価されたという意味)
- `lang_eval_wired: false` は §5 の縮退状態を示します

## 8. 問い合わせ

不明点・一致検証が解決しない場合は佐飼まで。バンドルを渡してもらえれば
こちらのビューアで一緒に確認できます。

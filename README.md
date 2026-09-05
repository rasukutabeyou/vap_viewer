# vap_viewer — VAP 誤りケース確認ツール

vapx の zero-shot 評価(SHIFT/HOLD・SHIFT予測)の**誤りケースを横断的に抽出・一覧・可視化**する
独立ツールです。要件定義: `~/work/AI_prompt/vapx/error_case_viewer_requirements.md`(v1.1)。

```
[build/ 抽出器]  vapx依存・GPU・exp毎に1回
   checkpoint + 全セッション → bundles/<name>/ に自己完結データ
                                     │
                                     ▼
[app/ ビューア]  vapx非依存・Streamlit・何度も
   bundles/ を読むだけ → 単一/比較モードで一覧⇄詳細+音声
```

- **ビューアは vapx / torch / checkpoint / GPU に依存しません**(配布単位 = `app/` + `bundles/`)。
- 音声はバンドルに**含めません**。閲覧時に `--audio-root`(研究室共有ストレージ)から参照します。

## 1. バンドル生成(producer、vapx 環境で1回)

```bash
cd ~/project/2026/vapx
uv run --with pandas --with pyarrow python \
    ~/project/2026/vap_viewer/build/extract_error_cases.py \
    --recipe-dir egs/tabidachi/vap1 \
    --exp-dir /autofs/diamond4/share/users/hanakawa/work/vapx/tabidachi/exp/xspk/seed0/gaze_raw \
    --split test \
    --name xspk_gaze_raw \
    --out ~/project/2026/vap_viewer/bundles/xspk_gaze_raw
```

- `--exp-dir` は絶対パスでも、`--recipe-dir` からの相対でも構いません。
- **manifest・特徴パスはチェックポイント埋め込みの config から来ます**
  (`conf/` を後から編集しても既存バンドルは変わりません)。xspk の
  チェックポイントを渡せば xspk の test split (30セッション) が使われます。
- 抽出後、`zero_shot-test.json` の保存値と**正解率一致検証**が走ります
  (MISMATCH の場合バンドルは拒否されます → §注意)。
- lang系モデルは lang特徴を自動で配線し、可視トークン(テキスト+時刻)も
  `tokens/` に書き出します。
- GPU で 1条件あたり約 1〜2 分。詳細は `build/README.md`。

**`bundles/` にあるのは xspk seed0 の18条件のみです** (話者disjoint split、
gaze は diamond3 の `l2cs-gaze360` 統一版)。

```
xspk_audio_only
xspk_gaze_{raw,delta,ddelta,rms}       xspk_headpose_{raw,delta,ddelta,rms}
xspk_au_{raw,delta,ddelta,rms}         xspk_au_raw_nospeech
xspk_vis_all_{raw,delta,ddelta,rms}
```

旧 41 バンドル (`gaze_cnn_raw` などの命名) は `bundles_legacy/` に退避しました。
**話者リークのある split で学習され、gaze は 330 セッション中 165 が符号の反転した
別出処**という二重の問題があるため、分析には使えません。参照したい場合のみ
`--bundles-dir bundles_legacy` を指定してください。

## 2. ビューア起動

```bash
cd ~/project/2026/vap_viewer
python -m venv .venv && .venv/bin/pip install -r app/requirements.txt   # 初回のみ
.venv/bin/streamlit run app/viewer.py -- \
    --bundles-dir bundles \
    --audio-root /autofs/diamond3/share/corpus/Tabidachi/processed/wav \
    --video-root /autofs/diamond3/share/corpus/Tabidachi/processed/mp4 \
    --au-dir /autofs/diamond4/share/users/hanakawa/work/vapx/tabidachi/vis_npz/au_libreface
```

`--` の後ろがビューアの引数です(前は streamlit の引数)。省略時の挙動:

| オプション | 省略時 | 効果 |
|---|---|---|
| `--bundles-dir` | `./bundles` | 読み込むバンドル置き場 |
| `--audio-root` | meta のパスをそのまま使う | 音声が見つからない時の探索先。無いと再生とクリックシークが使えない |
| `--video-root` | 無効 | `<sid>.mp4` があれば動画を音声と同期再生 |
| `--au-dir` | バンドル記録値 | **AU の表示系列を差し替える**。下記参照 |

**`--au-dir` を付ける理由**: バンドルは学習時の `au_dir` を記録します。視線・頭部姿勢の
モデルは AU を使いませんが、レシピ既定の ME-GraphAU (BP4D 12種) が記録に残るため、
指定しないとバンドルごとに AU の中身が変わります。LibreFace (DISFA 12種) に揃えるには
上記のように指定してください。表示だけが変わり、バンドルは書き換えません。
AU を実際に使うモデルで別ツリーを指定した場合は画面に警告が出ます。

## 2.1 使い方の流れ

```
① バンドル選択      サイドバー。1つ = 単一モード / 2つ以上 = 比較モード
② タスク選択        shift_hold / shift_pred / bc_pred / short_long
③ 絞り込み          「NGのみ」(既定ON) / gold / pred / セッション / スコア範囲
④ 一覧から行をクリック → 下に詳細が出る
⑤ 描画モードを切替  「発話活動予測」⇄「視覚特徴」
```

**描画モードは排他です。** 予測モードで違和感のあるケースを見つけ、視覚特徴モードに
切り替えてその原因を特徴側で確かめる、という順で使います。両方同時に出すと
どちらも小さくなって読めないため分けてあります。

### 発話活動予測モード(既定)

波形A/B・VAD・イベント・binヒートマップ・トークン・score曲線・P(SHIFT)・
p_now/p_future。音声プレーヤーと一体で、**再生位置が図上の赤い縦線**として全パネルに
重なり、**図をクリックするとその時刻から再生**されます(点線=イベント時刻)。

### 視覚特徴モード

視線・頭部姿勢・AU の実際の値を折れ線で表示します。右ペインに4つのコントロール:

| コントロール | 選択肢 | 備考 |
|---|---|---|
| 変換 | Raw / Δ / Δ² / RMS(Δ) | 既定 RMS(Δ)。Δ と Δ² は**符号あり**(上を向いた/下を向いたが読める) |
| モダリティ | 視線 / 頭部姿勢 / 表情 / すべて | |
| 次元 | すべて / pitch / yaw / roll / AU… | **単一モダリティ選択時のみ**有効 |
| Raw と並べる | 既定ON | 選んだ変換の上に Raw を同じ次元・同じ時間軸で描く |

- **話者A(青)とB(橙)が同じ軸に重なります。** Raw では2人が別のベースラインに乗り、
  動的変換ではゼロ近傍に集まります。この差が「動的特徴が個人差を消す」ことの中身です。
- **桃色の網掛け = モデルが実際にスコアを計算した窓**、点線 = イベント時刻。
  「どこを見て間違えたか」が分かります。
- **最上段の帯 = 発話区間。** 動いたのが話し手か聞き手かはこれを見ないと決まりません。
  相槌なら聞き手の動き、話者交替なら話し手の終了合図か聞き手の開始準備かで解釈が変わります。
- 次元が多いモダリティ(AU の12次元)は、**その窓で実際に動いた上位3次元だけ濃く**描き、
  残りは薄線にします。次元を1つ選べば、その次元だけを符号つきで読めます。

「次元」を選ぶと「この人は相槌を打つ時に上を見る」のような、方向を伴う傾向が読めます。
全次元を重ねた状態では符号と大きさが潰れるので、傾向を確かめる段階では1次元に絞ってください。

- **単一モデルモード**: バンドルを1つ選び、NGのみ/gold/pred/セッション/スコア範囲で
  フィルタ→行クリックで詳細(波形A/B・VAD・S/Hイベント・binヒートマップ・トークン・
  score曲線・p_now/p_future・音声再生)。
- 音声プレーヤーは詳細図と一体化: **再生位置が詳細図上の赤い縦線**で全パネル
  (波形・VAD・確率曲線)に重なって表示され、**図の任意の位置をクリックするとその
  時刻から再生**されます(点線=イベント時刻)。
- **動画**: `--video-root`(`<sid>.mp4`)を指定して右側の「動画」チェックを
  入れると、詳細図の上にセッション動画(表示窓と同じ時間範囲を ffmpeg で
  切り出し)が表示され、音声プレーヤーと**同期再生**されます(音声=マスター、
  動画はミュート追従。ffmpeg は `imageio-ffmpeg` 同梱バイナリを使用)。
- **イベントパネル(VAD直下)**: 各イベントに **gold を常時表示**(`G:S`/`G:H`)。
  比較モードでは**全モデルの判定を縦に並べて表示**します
  (shift_hold: `S○`/`H×`(○=正解, ×=誤り)、shift_pred: `TP○`/`FN×`。色=モデル色)。
- **P(SHIFT)パネル**: 常に「上=SHIFT予測 / 下=HOLD予測」になるよう正規化した曲線。
  比較モードではモデル毎の曲線+同色破線の閾値+評価窓右の S/H(TP/FN)判定文字で、
  各モデルがどちらに予測したか一目で分かります。p_now/p_future は「↑=A / ↓=B」
  (次話者視点)のままです。
- **比較モード**: バンドルを2つ以上選ぶと `event_key` で厳密join。
  「CPCは正解だが stream-KV は誤り」等の正誤パターンでフィルタでき、
  p_now/p_future はモデル毎に重ね描画されます。
  lang系バンドル同士の比較では**トークンパネルがモデル毎に縦に並び**、同一時間軸で
  トークン内容(ASR結果や遅延の違い)を比較できます(同じ lang_dir を共有する
  モデルは1パネルに集約)。右ペインの可視トークンもモデル別に表示されます。
- meta.json の音声パスがそのまま見つかる場合 `--audio-root` は不要です。
  見つからない場合は `--audio-root/<パス>` → `--audio-root/<ファイル名>` の順で解決します。

## 2.2 リポジトリに入れないもの

`.gitignore` 済み。**いずれも生成物なので、消しても手順どおり作り直せます。**

| パス | 中身 | 復旧方法 |
|---|---|---|
| `bundles/` | 分析用バンドル (~940 MB) | §1 の抽出コマンド |
| `bundles_legacy/` | 旧バンドル (~2.3 GB) | 再生成しない (旧expは非推奨) |
| `.venv/` | ビューアの実行環境 (~520 MB) | §2 の venv 作成コマンド |
| `exports/` | 動画エクスポートの出力先 | 実行時に自動作成 |
| `__pycache__/`, `*.pyc` | バイトコード | 自動生成 |

**逆に、リポジトリに必要なもの**は `app/` `build/` `docs/` `README.md` と
`app/assets/fonts/NotoSansJP-Regular.ttf` (4.4 MB) です。フォントは図の日本語
ラベル描画に使い、システムに IPAex 系が無い環境でも同じ図が出るよう同梱しています
(`app/lib/plots.py` の `_setup_fonts`)。

## 3. バンドルの中身(配布単位)

```
bundles/<name>/
├── cases.parquet      # 1行 = 1イベント×このモデル (shift_hold / shift_pred / bc_pred / short_long)
├── probs/<sid>.npz    # p_now/p_future, bin_probs, subsetスコア曲線, VAD (float16)
├── tokens/<sid>.jsonl # lang系のみ: 可視トークン {ch, text, pos, end, fin}
└── meta.json          # exp/閾値/frame_hz/config hash/git commit/セッション→音声パス
```

比較モードの前提(同一 split・VAD・frame_hz)は meta.json の `plan_cfg_hash` で
検証され、不一致なら警告が出ます。

## 注意

- **一致検証が失敗する場合**: `zero_shot-test.json` が古い(lang未配線バグ修正前の生成)
  可能性が高いです。`uv run vapx-zero-shot ...` で再生成してから抽出してください。
- `--limit-sessions` 付きで作ったデバッグ用バンドルは UI 上 ⚠partial と表示されます。
- タスクは shift_hold / shift_pred / bc_pred / short_long の4つ(schema v2)。
  shift_pred / bc_pred は見逃し(FN)中心の正例イベントのみ、short_long は
  SHORT(相槌開始)と LONG(交替後の発話開始)の両クラスをイベント化します。
  v1 バンドル(shift_hold / shift_pred のみ)もビューアはそのまま読めます。

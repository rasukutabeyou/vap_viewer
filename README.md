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
cd ~/work/vapx
uv run --with pandas --with pyarrow python ~/work/vap_viewer/build/extract_error_cases.py \
    --recipe-dir egs/tabidachi/vap1 \
    --exp-dir exp/train_audio_cpc_reg \
    --split test \
    --out ~/work/vap_viewer/bundles/audio_cpc_reg
```

- 抽出後、`zero_shot-test.json` の保存値と**正解率一致検証**が走ります
  (MISMATCH の場合バンドルは拒否されます → §注意)。
- lang系モデルは lang特徴を自動で配線し、可視トークン(テキスト+時刻)も
  `tokens/` に書き出します。
- 詳細は `build/README.md`。

## 2. ビューア起動

```bash
cd ~/work/vap_viewer
python -m venv .venv && .venv/bin/pip install -r app/requirements.txt   # 初回のみ
.venv/bin/streamlit run app/viewer.py -- \
    --bundles-dir bundles \
    --audio-root /autofs/diamond3/share/corpus/Tabidachi/processed/wav
```

- **単一モデルモード**: バンドルを1つ選び、**判定タイプ (gold→pred)**・セッション・
  スコア範囲でフィルタ→行クリックで詳細。判定タイプは「S→H: SHIFTをHOLDと誤り
  (見逃し)」のような gold と pred の組で選べます(既定は誤りのみ。空にすると全件)(波形A/B・VAD・S/Hイベント・binヒートマップ・トークン・
  score曲線・p_now/p_future・音声再生)。
- 音声プレーヤーは詳細図と一体化: **再生位置が詳細図上の赤い縦線**で全パネル
  (波形・VAD・確率曲線)に重なって表示され、**図の任意の位置をクリックするとその
  時刻から再生**されます(点線=イベント時刻)。
- **イベントパネル(VAD直下)**: 各イベントに **gold を常時表示**(`G:S`/`G:H`)。
  比較モードでは**全モデルの判定を縦に並べて表示**します
  (shift_hold: `S○`/`H×`(○=正解, ×=誤り)、shift_pred: `TP○`/`FN×`。色=モデル色)。
- **P(SHIFT)パネル**: 常に「上=SHIFT予測 / 下=HOLD予測」になるよう正規化した曲線。
  比較モードではモデル毎の曲線+同色破線の閾値+評価窓右の S/H(TP/FN)判定文字で、
  各モデルがどちらに予測したか一目で分かります。p_now/p_future は「↑=A / ↓=B」
  (次話者視点)のままです。
- **トークンパネル (lang系)**: 各トークンの **▎ティック = モデル入力として有効に
  なった時刻**(cross-attention は `pos <= t < end` のトークンを参照。delay系
  バンドルでは発話とティックのずれ=遅延がそのまま見えます)。ASRの改訂で
  **撤回された partial 仮説は灰色**、─線は撤回までの間入力に残っていた期間です。
  テキストは話者レーンごとに3段へ自動配置して重なりを回避します
  (右にずらされたテキストは点線でティックに接続)。右ペインの
  「イベント時点の有効トークン」は無音開始時点でモデルに入っていた
  テキストのスナップショットです(撤回済み仮説は含まず)。
- **比較モード**: バンドルを2つ以上選ぶと `event_key` で厳密join。
  gold と「CPCは正解だが stream-KV は誤り」等の正誤パターンでフィルタでき
  (例: gold=S + 「XのみNG」= X だけが SHIFT を見逃したケース)、
  p_now/p_future はモデル毎に重ね描画されます。
  lang系バンドル同士の比較では**トークンパネルがモデル毎に縦に並び**、同一時間軸で
  トークン内容(ASR結果や遅延の違い)を比較できます(同じ lang_dir を共有する
  モデルは1パネルに集約)。右ペインの可視トークンもモデル別に表示されます。
- **表示名**: サイドバーの「表示名の編集」でバンドルに短い表示名を付けられます
  (`bundles/aliases.json` に保存され、選択肢・凡例・正解率・エクスポートの
  ラベルすべてに反映。空欄なら meta.json の `exp`(抽出時 `--name`)→
  ディレクトリ名の順で表示)。
- meta.json の音声パスがそのまま見つかる場合 `--audio-root` は不要です。
  見つからない場合は `--audio-root/<パス>` → `--audio-root/<ファイル名>` の順で解決します。

### メモ・ブックマーク

- 詳細画面の右ペインで各ケースに**メモ**(比較時に気づいた特徴の記録)と
  **★ブックマーク**(要再確認マーク)を付けられます。一覧の先頭列に ★/メモが
  表示され、サイドバーの「★ ブックマークのみ」で絞り込めます。
- キーは `event_key`(VAD由来・モデル非依存)なので、**単一/比較どちらのモードで
  書いても共有**され、バンドルを再抽出しても消えません。
- 保存先は `<bundles-dir>/notes.json`(`--notes-file` で変更可)。メモも★も空に
  なったエントリはファイルから自動削除されます。

### 保存(レポート用エクスポート)

詳細画面の右ペイン下部からダウンロードできます:

- **HTML(図+音声)**: 詳細図・音声・ケース情報・モデル別判定・
  イベント時点の有効トークン(lang系)・メモを1ファイルに
  埋め込んだ**自己完結HTML**。ブラウザで開くだけで図クリック→シーク再生まで動くので、
  スクリーンショットでは音声を伝えられない報告書の添付資料に使えます
  (base64埋込のためサーバ・依存物・ネット接続は不要。1ファイル約1MB)。
- **PNG(図のみ)** / **WAV(音声のみ)**: 個別素材が必要な場合用。

## 3. バンドルの中身(配布単位)

```
bundles/<name>/
├── cases.parquet      # 1行 = 1イベント×このモデル (shift_hold + shift_pred)
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
- タスクは初版 shift_hold / shift_pred のみ。bc_pred / short_long は将来拡張。

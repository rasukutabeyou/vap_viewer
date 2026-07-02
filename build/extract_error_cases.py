#!/usr/bin/env python
"""Extract per-event error cases from a trained vapx checkpoint into a bundle.

Producer-side tool (runs inside the vapx environment, needs torch/GPU).
The produced bundle is self-contained: the Streamlit viewer reads it without
vapx / torch / checkpoints.

Reproduces `vapx.eval.zero_shot` exactly:
  * the model forward (`predict_session`, same chunking / warmup),
  * the VAD event plan (`_plan_session`),
  * the frame/event scoring (`_score_session`),
and verifies that metrics recomputed at the thresholds saved in
`zero_shot-test.json` match that file (exact tp/fp/tn/fn). A mismatch means
the json is stale (e.g. generated before the lang-wiring fix) or the config /
checkpoint differ -- the bundle is then refused unless --allow-mismatch.

Bundle layout (out dir):
  cases.parquet          1 row = 1 event x this model (shift_hold + shift_pred)
  probs/<sid>.npz        p_now/p_future, bin_probs, per-frame subset scores, vad
  tokens/<sid>.jsonl     lang models only: visible tokens (text + frame times)
  meta.json              exp/config/thresholds/frame_hz/sessions/verification

Usage (from the vapx repo, so `uv run` picks up the vapx project; pandas and
pyarrow are added ephemerally without touching the lockfile):

  cd ~/work/vapx
  uv run --with pandas --with pyarrow python ~/work/vap_viewer/build/extract_error_cases.py \
      --recipe-dir egs/tabidachi/vap1 \
      --exp-dir exp/train_lang_kv_sarashina_reg \
      --split test \
      --out ~/work/vap_viewer/bundles/lang_kv_sarashina_reg
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# Tolerance for float metric comparison against the saved json. Counts
# (tp/fp/tn/fn/n_pos/n_neg) are compared exactly.
FLOAT_RTOL = 1e-6

CASE_COLUMNS = [
    "exp", "session", "task", "event_key", "t_sec",
    "silence_start", "silence_end", "pre_speaker", "post_speaker",
    "gold", "pred", "score", "threshold", "correct", "margin",
]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--recipe-dir", required=True, type=Path,
                   help="vapx recipe dir (e.g. egs/tabidachi/vap1); relative "
                        "paths in the checkpoint's embedded config resolve here.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp-dir", type=Path,
                   help="exp dir (relative to --recipe-dir ok); uses "
                        "checkpoints/best.pt and zero_shot-test.json inside it.")
    g.add_argument("--checkpoint", type=Path,
                   help="explicit checkpoint path (then --zs-json is required).")
    p.add_argument("--zs-json", type=Path, default=None,
                   help="zero_shot-*.json with tuned thresholds "
                        "(default: <exp-dir>/zero_shot-test.json).")
    p.add_argument("--split", default="test", choices=("train", "valid", "test"))
    p.add_argument("--out", required=True, type=Path, help="bundle output dir.")
    p.add_argument("--name", default=None,
                   help="exp display name recorded in the bundle "
                        "(default: basename of --out).")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    # Forward-pass knobs: keep identical to the run that wrote zero_shot-test.json.
    p.add_argument("--chunk-sec", type=float, default=20.0)
    p.add_argument("--warmup-sec", type=float, default=3.0)
    p.add_argument("--min-context-sec", type=float, default=3.0)
    p.add_argument("--no-tokens", action="store_true",
                   help="skip token text export for lang models.")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="write the bundle even when verification fails "
                        "(result is still recorded in meta.json).")
    p.add_argument("--limit-sessions", type=int, default=None,
                   help="DEBUG: only process the first N sessions "
                        "(verification is skipped).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# probability-curve helpers (mirrors notebooks/model_output_zero_shot.ipynb)
# --------------------------------------------------------------------------


def _bit_mask(num_bins: int) -> np.ndarray:
    """(num_classes, 2K) 0/1 matrix; bit k*2+s == speaker s active in bin k."""
    bits = num_bins * 2
    idx = np.arange(1 << bits)
    return ((idx[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.float32)


def _now_future(bin_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p_now / p_future from (T, K, 2) marginal bin activity."""
    K = bin_probs.shape[1]
    mid = K // 2
    near_a = bin_probs[:, :mid, 0].mean(axis=-1)
    near_b = bin_probs[:, :mid, 1].mean(axis=-1)
    far_a = bin_probs[:, mid:, 0].mean(axis=-1)
    far_b = bin_probs[:, mid:, 1].mean(axis=-1)
    p_now = near_a / np.clip(near_a + near_b, 1e-9, None)
    p_future = far_a / np.clip(far_a + far_b, 1e-9, None)
    return p_now, p_future


# --------------------------------------------------------------------------
# per-session case extraction (must mirror zero_shot._plan_session/_score_session)
# --------------------------------------------------------------------------


def _session_cases(sid, probs, vad_full, T, plan, zcfg, masks, thresholds, exp_name,
                   find_shift_hold_events):
    """Build case rows for one session. Score arithmetic replicates
    `_score_session` expression-for-expression so the parquet rows are
    bit-identical with the samples used for metric verification."""
    frame_hz = zcfg.frame_hz
    vad = vad_full[:T].astype(np.float32)
    min_ctx = int(round(zcfg.min_context_sec * frame_hz))
    score = {k: probs[:T] @ m.astype(probs.dtype) for k, m in masks.items()}

    # -- shift_hold: recompute events (same call as _plan_session) to keep
    #    silence_end etc. that plan.sh does not carry.
    sh_events = find_shift_hold_events(
        vad, frame_hz,
        pre_offset_sec=zcfg.sh_pre_offset_sec, post_onset_sec=zcfg.sh_post_onset_sec,
    )
    eval_start = int(round(zcfg.sh_eval_start_sec * frame_hz))
    eval_dur = int(round(zcfg.sh_eval_dur_sec * frame_hz))
    kept = []
    for ev in sh_events:
        s = ev.silence_start + eval_start
        e = s + eval_dur
        if s < min_ctx or e > T:
            continue
        kept.append((ev, s, e))
    if len(kept) != plan.sh.shape[0]:
        raise RuntimeError(
            f"{sid}: shift_hold event recount mismatch "
            f"({len(kept)} != plan {plan.sh.shape[0]})")

    rows = []
    thr_sh = thresholds.get("shift_hold")
    for ev, s, e in kept:
        sA = float(score["sh_0"][s:e].mean())
        sB = float(score["sh_1"][s:e].mean())
        prob_shift = sB if ev.pre_speaker == 0 else sA
        prob_hold = sA if ev.pre_speaker == 0 else sB
        denom = prob_shift + prob_hold
        score_shift = prob_shift / denom if denom > 0 else 0.5
        gold = "S" if ev.is_shift else "H"
        pred = "S" if (thr_sh is not None and score_shift >= thr_sh) else "H"
        rows.append(dict(
            exp=exp_name, session=sid, task="shift_hold",
            event_key=f"{sid}|shift_hold|{ev.silence_start}|{ev.pre_speaker}",
            t_sec=ev.silence_start / frame_hz,
            silence_start=int(ev.silence_start), silence_end=int(ev.silence_end),
            pre_speaker=int(ev.pre_speaker), post_speaker=int(ev.post_speaker),
            gold=gold, pred=pred, score=score_shift,
            threshold=float(thr_sh) if thr_sh is not None else math.nan,
            correct=(gold == pred),
            margin=abs(score_shift - thr_sh) if thr_sh is not None else math.nan,
        ))

    # -- shift_pred: positive (SHIFT) event windows, frame scores aggregated
    #    by window mean; miss(FN)-centric per the spec.
    spred_dur = int(round(zcfg.spred_eval_dur_sec * frame_hz))
    kept_pred = []
    for ev in sh_events:
        if not ev.is_shift:
            continue
        s = max(min_ctx, ev.silence_start - spred_dur)
        e = ev.silence_start
        if e - s < 1:
            continue
        kept_pred.append((ev, s, e))
    if len(kept_pred) != plan.spred_pos.shape[0]:
        raise RuntimeError(
            f"{sid}: shift_pred event recount mismatch "
            f"({len(kept_pred)} != plan {plan.spred_pos.shape[0]})")

    thr_sp = thresholds.get("shift_pred")
    for ev, s, e in kept_pred:
        w = score[f"spred_{int(ev.post_speaker)}"][s:e]
        sc = float(w.mean())
        pred_pos = thr_sp is not None and sc >= thr_sp
        rows.append(dict(
            exp=exp_name, session=sid, task="shift_pred",
            event_key=f"{sid}|shift_pred|{ev.silence_start}|{ev.pre_speaker}",
            t_sec=ev.silence_start / frame_hz,
            silence_start=int(ev.silence_start), silence_end=int(ev.silence_end),
            pre_speaker=int(ev.pre_speaker), post_speaker=int(ev.post_speaker),
            gold="pos", pred="pos" if pred_pos else "neg", score=sc,
            threshold=float(thr_sp) if thr_sp is not None else math.nan,
            correct=bool(pred_pos),
            margin=abs(sc - thr_sp) if thr_sp is not None else math.nan,
        ))

    return rows, score


# --------------------------------------------------------------------------
# token text export (lang models only)
# --------------------------------------------------------------------------


class _TokenExporter:
    def __init__(self, lang_dir: Path, out_dir: Path):
        self.lang_dir = lang_dir
        self.out_dir = out_dir
        self._tokenizers: dict[str, object] = {}
        self.tokenizer_name: str | None = None

    def _tokenizer(self, name: str):
        if name not in self._tokenizers:
            from transformers import AutoTokenizer
            self._tokenizers[name] = AutoTokenizer.from_pretrained(name)
            self.tokenizer_name = name
        return self._tokenizers[name]

    def export(self, sid: str) -> bool:
        rows = []
        for side in ("L", "R"):
            npz_path = self.lang_dir / f"{sid}-{side}.npz"
            side_json = self.lang_dir / f"{sid}-{side}.json"
            with np.load(npz_path) as d:
                if "token_ids" not in d.files:
                    return False
                ids = d["token_ids"].tolist()
                pos = d["positions"].tolist()
                end = d["end_positions"].tolist()
                fin = d["finalized_frames"].tolist()
            model_name = json.loads(side_json.read_text())["model"]
            tok = self._tokenizer(model_name)
            texts = tok.batch_decode([[i] for i in ids], skip_special_tokens=False)
            rows.extend(
                {"ch": side, "text": t, "pos": p, "end": e, "fin": f}
                for t, p, e, f in zip(texts, pos, end, fin)
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / f"{sid}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return True


# --------------------------------------------------------------------------
# verification against the saved zero_shot json
# --------------------------------------------------------------------------


def _verify_task(name, store, saved: dict, split: str, binary_metrics) -> dict:
    """Recompute metrics at the saved threshold and diff against the json.
    For split=test compare the top-level numbers, for split=valid the nested
    valid_metrics (same threshold, per zero_shot.compute_task_metrics)."""
    ref = saved if split == "test" else saved.get("valid_metrics", {})
    thr = saved["threshold"]
    scores, labels = store.as_arrays()
    m = binary_metrics(labels, scores, threshold=thr).as_dict()

    diffs = []
    for k in ("n_pos", "n_neg", "tp", "fp", "tn", "fn"):
        if k in ref and int(m[k]) != int(ref[k]):
            diffs.append(f"{k}: recomputed {m[k]} != saved {ref[k]}")
    for k in ("acc", "bacc", "precision", "recall", "f1", "auc"):
        if k in ref and not math.isclose(m[k], ref[k], rel_tol=FLOAT_RTOL, abs_tol=1e-9):
            diffs.append(f"{k}: recomputed {m[k]:.10f} != saved {ref[k]:.10f}")
    return {
        "task": name, "split": split, "threshold": thr,
        "ok": not diffs, "diffs": diffs,
        "recomputed": {k: m[k] for k in ("n_pos", "n_neg", "tp", "fp", "tn", "fn", "acc", "bacc", "f1", "auc")},
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    args = _parse_args(argv)

    recipe_dir = args.recipe_dir.resolve()
    out_dir = args.out.resolve()
    exp_name = args.name or out_dir.name

    # Resolve exp paths before chdir; relative --exp-dir/--checkpoint are
    # taken relative to the recipe dir (matching how vapx CLIs are run).
    def _rel_to_recipe(p: Path) -> Path:
        return p if p.is_absolute() else (recipe_dir / p)

    if args.exp_dir is not None:
        exp_dir = _rel_to_recipe(args.exp_dir).resolve()
        ckpt_path = exp_dir / "checkpoints" / "best.pt"
        zs_json_path = args.zs_json or (exp_dir / "zero_shot-test.json")
    else:
        ckpt_path = _rel_to_recipe(args.checkpoint).resolve()
        if args.zs_json is None:
            sys.exit("--checkpoint requires --zs-json (tuned thresholds).")
        zs_json_path = args.zs_json
    zs_json_path = _rel_to_recipe(Path(zs_json_path)).resolve()

    for p, what in ((ckpt_path, "checkpoint"), (zs_json_path, "zero_shot json")):
        if not p.is_file():
            sys.exit(f"{what} not found: {p}")

    # Relative paths inside the checkpoint's embedded config (manifests, CPC
    # weights, ...) resolve from the recipe dir, exactly like vapx CLI runs.
    os.chdir(recipe_dir)

    import inspect

    import torch

    from vapx.eval.events import find_shift_hold_events
    from vapx.eval.inference import iter_session_inputs, predict_session
    from vapx.eval.metrics import binary_metrics
    from vapx.eval.zero_shot import (
        TaskStores,
        ZeroShotConfig,
        _build_subset_masks,
        _plan_cfg_hash,
        _plan_session,
        _score_session,
    )
    from vapx.inference import load_for_inference

    # The extractor supports two vapx lineages, detected from what the
    # installed vapx actually provides:
    #   * lang-wired eval (dev_sakai / main): predict_session(..., lang=...)
    #     and zero_shot._load_session_lang exist.
    #   * visual-wired eval (dev-hanakawa): predict_session(...,
    #     vis_encoders=, vis_feats_a/b=, vis_audio_ratio=) and
    #     iter_session_inputs(..., gaze_dir=, ...) exist.
    try:
        from vapx.eval.zero_shot import _load_session_lang
    except ImportError:
        _load_session_lang = None
    ps_params = set(inspect.signature(predict_session).parameters)
    has_lang_eval = _load_session_lang is not None and "lang" in ps_params
    has_vis_eval = "vis_encoders" in ps_params

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] loading {ckpt_path} on {device}")
    bundle = load_for_inference(ckpt_path, device=device)
    feature, model, labeler = bundle["feature"], bundle["model"], bundle["labeler"]
    cfg = bundle["config"]
    data_cfg = cfg["data"]
    sample_rate = data_cfg.get("sample_rate", 16000)

    zcfg = ZeroShotConfig(
        frame_hz=labeler.frame_hz,
        num_bins=labeler.num_bins,
        min_context_sec=args.min_context_sec,
    )
    masks = _build_subset_masks(zcfg.num_bins)
    bitmask = _bit_mask(zcfg.num_bins)

    manifest_key = f"{args.split}_manifest"
    if manifest_key not in data_cfg:
        sys.exit(f"checkpoint config has no data.{manifest_key}")
    manifest_path = Path(data_cfg[manifest_key]).resolve()
    with manifest_path.open(encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    base_dir = manifest_path.parent
    if args.limit_sessions:
        entries = entries[: args.limit_sessions]

    # Same lang detection as vapx.training.entry.zero_shot_main (post-fix).
    lang_rel = data_cfg.get("lang_dir") if hasattr(model, "lang_in_proj") else None
    if lang_rel and not has_lang_eval:
        # dev-hanakawa lineage: the model HAS a language branch but this
        # vapx's eval cannot feed it -- zero-shot (and this bundle) score the
        # model with the lang cross-attention idle. Consistent with the
        # zero_shot json produced by the same code, so verification still
        # passes, but the numbers are lang-degraded. Surface it loudly.
        print("[extract] WARNING: model has a lang branch but this vapx has "
              "no lang wiring in eval -> lang is NOT fed (degraded, matches "
              "this vapx's zero_shot json). Tokens are not exported.")
        lang_rel = None
    lang_dir = (base_dir / lang_rel) if lang_rel else None
    if lang_rel:
        print(f"[extract] language model; feeding lang features from {lang_dir}")

    # Visual modalities (dev-hanakawa lineage): mirror zero_shot_main --
    # rebuild the per-modality encoders and load their checkpoint weights.
    vis_encoders: dict = {}
    if has_vis_eval:
        ckpt_vis = bundle["state"]["model"].get("vis_encoders", {})
        for key, enc in (bundle.get("vis_encoders") or {}).items():
            enc = enc.to(device)
            if key in ckpt_vis:
                enc.load_state_dict(ckpt_vis[key])
            vis_encoders[key] = enc.eval()
    vis_dirs = {k: data_cfg.get(k) for k in ("gaze_dir", "head_dir", "au_dir")}
    use_vis = bool(has_vis_eval and vis_encoders and any(vis_dirs.values()))
    vis_role_a = data_cfg.get("vis_role_a", "customer")
    vis_role_b = data_cfg.get("vis_role_b", "operator")
    vis_fps = data_cfg.get("vis_fps", 25.0)
    vis_audio_ratio = max(1, round(float(zcfg.frame_hz) / float(vis_fps)))
    if use_vis:
        print(f"[extract] visual model; modalities={sorted(vis_encoders)} "
              f"dirs={ {k: v for k, v in vis_dirs.items() if v} } "
              f"roles=({vis_role_a},{vis_role_b}) ratio={vis_audio_ratio}")

    zs_saved = json.loads(zs_json_path.read_text())
    thresholds = {
        t: zs_saved[t]["threshold"]
        for t in ("shift_hold", "shift_pred")
        if t in zs_saved and "threshold" in zs_saved[t]
    }
    if not thresholds:
        sys.exit(f"no tuned thresholds found in {zs_json_path}")
    print(f"[extract] thresholds: {thresholds}")

    out_dir.mkdir(parents=True, exist_ok=True)
    probs_dir = out_dir / "probs"
    probs_dir.mkdir(exist_ok=True)
    token_exporter = (
        _TokenExporter(lang_dir, out_dir / "tokens")
        if (lang_dir is not None and not args.no_tokens) else None
    )
    tokens_ok = token_exporter is not None

    stores = TaskStores()
    all_rows: list[dict] = []
    sessions_meta: list[dict] = []

    if use_vis:
        session_iter = iter_session_inputs(
            entries, base_dir, sample_rate,
            gaze_dir=vis_dirs["gaze_dir"], head_dir=vis_dirs["head_dir"],
            au_dir=vis_dirs["au_dir"], role_a=vis_role_a, role_b=vis_role_b)
    else:
        session_iter = iter_session_inputs(entries, base_dir, sample_rate)

    for i, item in enumerate(session_iter):
        entry, wav, vad_full = item[0], item[1], item[2]
        vis_a = item[3] if len(item) > 3 else None
        vis_b = item[4] if len(item) > 4 else None
        sid = entry["id"]
        fwd_kwargs: dict = {}
        if lang_dir is not None:
            fwd_kwargs["lang"] = _load_session_lang(lang_dir, sid)
        if use_vis:
            fwd_kwargs.update(
                vis_encoders=vis_encoders,
                vis_feats_a=vis_a, vis_feats_b=vis_b,
                vis_audio_ratio=vis_audio_ratio,
            )
        probs = predict_session(
            feature, model, wav,
            sample_rate=sample_rate,
            chunk_sec=args.chunk_sec, warmup_sec=args.warmup_sec,
            device=device,
            **fwd_kwargs,
        )
        T = min(probs.shape[0], vad_full.shape[0])
        plan = _plan_session(T, vad_full, zcfg)
        _score_session(probs, plan, masks, stores)   # verification samples

        rows, score = _session_cases(
            sid, probs, vad_full, T, plan, zcfg, masks, thresholds, exp_name,
            find_shift_hold_events)
        all_rows.extend(rows)

        bin_probs = (probs[:T] @ bitmask).reshape(T, zcfg.num_bins, 2)
        p_now, p_future = _now_future(bin_probs)
        np.savez_compressed(
            probs_dir / f"{sid}.npz",
            p_now=p_now.astype(np.float16),
            p_future=p_future.astype(np.float16),
            bin_probs=bin_probs.astype(np.float16),
            score_sh=np.stack([score["sh_0"], score["sh_1"]], axis=1).astype(np.float16),
            score_spred=np.stack([score["spred_0"], score["spred_1"]], axis=1).astype(np.float16),
            vad=(vad_full[:T] > 0).astype(np.uint8),
            frame_hz=np.float32(zcfg.frame_hz),
        )

        if token_exporter is not None and tokens_ok:
            try:
                tokens_ok = token_exporter.export(sid)
                if not tokens_ok:
                    print(f"[extract] WARNING: {sid} lang npz has no token_ids; "
                          f"token export disabled")
            except Exception as exc:  # noqa: BLE001 - non-fatal side channel
                tokens_ok = False
                print(f"[extract] WARNING: token export failed ({exc}); disabled")

        n_sh = sum(1 for r in rows if r["task"] == "shift_hold")
        n_sp = len(rows) - n_sh
        print(f"[extract] [{i + 1}/{len(entries)}] {sid}  T={T}  sh={n_sh}  spred+={n_sp}")

        smeta = {
            "id": sid,
            "audio_l": entry.get("audio_l"), "audio_r": entry.get("audio_r"),
            "audio": entry.get("audio"),
            "duration": entry.get("duration"), "n_frames": int(T),
        }
        if use_vis:
            # which modalities actually had features for this session
            # (sessions without video run with the visual branches idle)
            smeta["vis"] = sorted(set(vis_a or {}) | set(vis_b or {}))
        sessions_meta.append(smeta)

    # ---- verification --------------------------------------------------
    verification = []
    if args.limit_sessions:
        print("[extract] --limit-sessions given: skipping metric verification")
    else:
        for task, store in (("shift_hold", stores.sh), ("shift_pred", stores.spred)):
            if task not in thresholds:
                continue
            v = _verify_task(task, store, zs_saved[task], args.split, binary_metrics)
            verification.append(v)
            status = "OK" if v["ok"] else "MISMATCH"
            print(f"[verify] {task} ({args.split}): {status}")
            for d in v["diffs"]:
                print(f"[verify]   {d}")
        failed = [v for v in verification if not v["ok"]]
        if failed and not args.allow_mismatch:
            print("[verify] FAILED: recomputed metrics differ from "
                  f"{zs_json_path}.\n"
                  "  Likely causes: the json predates the lang-wiring fix "
                  "(regenerate with vapx-zero-shot), a different checkpoint, "
                  "or different forward-pass args (--chunk-sec/--warmup-sec/"
                  "--min-context-sec).\n"
                  "  Use --allow-mismatch to write the bundle anyway.")
            return 1

    # ---- cases.parquet --------------------------------------------------
    import pandas as pd

    df = pd.DataFrame(all_rows, columns=CASE_COLUMNS)
    df.to_parquet(out_dir / "cases.parquet", index=False)

    # ---- meta.json -------------------------------------------------------
    def _git_commit(repo: Path) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return None

    import vapx

    meta = {
        "schema_version": SCHEMA_VERSION,
        "exp": exp_name,
        "checkpoint": str(ckpt_path),
        "zs_json": str(zs_json_path),
        "split": args.split,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "vapx_git_commit": _git_commit(Path(vapx.__file__).parent),
        "frame_hz": float(zcfg.frame_hz),
        "num_bins": int(zcfg.num_bins),
        "bin_times_sec": list(getattr(labeler, "bin_times_sec", [])),
        "thresholds": thresholds,
        "zero_shot_config": asdict(zcfg),
        "plan_cfg_hash": _plan_cfg_hash(zcfg),
        "forward": {"chunk_sec": args.chunk_sec, "warmup_sec": args.warmup_sec},
        "sample_rate": sample_rate,
        "lang_dir": lang_rel,
        "lang_eval_wired": has_lang_eval,
        "vis": ({
            "modalities": sorted(vis_encoders),
            "dirs": {k: v for k, v in vis_dirs.items() if v},
            "roles": [vis_role_a, vis_role_b],
            "vis_audio_ratio": vis_audio_ratio,
        } if use_vis else None),
        "has_tokens": bool(token_exporter is not None and tokens_ok),
        "tokenizer": token_exporter.tokenizer_name if token_exporter else None,
        "n_sessions": len(sessions_meta),
        "sessions": sessions_meta,
        "verification": verification,
        "limited": bool(args.limit_sessions),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    n_ng = int((~df["correct"]).sum())
    print(f"[extract] wrote bundle to {out_dir}")
    print(f"[extract]   cases: {len(df)}  (NG: {n_ng})  sessions: {len(sessions_meta)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

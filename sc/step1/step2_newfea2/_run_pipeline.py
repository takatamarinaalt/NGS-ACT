#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2 全ステップ一括実行スクリプト（ステップ1〜9）

実行順序:
  1. step1: 予測
  2. step2: 閾値の再設定（任意, BAM再スキャンなしでTYPE列を再判定）
  3. DEL パイプライン: step3 → step5 → step6 → step7
  4. INS パイプライン: step3 → step5 → step6 → step7
  5. 結合ステップ: DEL と INS の nested ラベルを統合して {TOP}-{DEL}-{INS} 形式の txt を生成
  6. step9: PCA プロット（DEL・INS それぞれ、結合ラベルで色付け）

最終的な label 形式:
  {TOP クラスタ番号}-{DEL サブクラスタ番号}-{INS サブクラスタ番号}
  例: 1-2-3

DEL / INS は step1_predictions/predictions フォルダのファイル名から自動判別する（--mode 指定不要）。
--start-step / --end-step で途中のステップから開始・終了できる。

ステップ番号: 1, 2, 3, 5, 6, 7, combine, 9

step2（閾値の再設定）について:
  1_predict_combined.py の出力は、スキャンした全座位について PROB_INS
  （または PROB_DEL_st / PROB_DEL_end）と PROB_NONE を保持しているため、
  BAM再スキャンなしで閾値だけを変更して TYPE 列を再分類できる。
  --threshold-del / --threshold-ins のどちらかを指定すると自動的に実行される
  （両方とも未指定なら step2 はスキップされ、step1 の元の分類がそのまま使われる）。
  出力は <outdir>/step2_threshold/predictions/ に書き出され、
  以降の step3 はデフォルトでこのフォルダを入力として使う。

使用例:
  # ステップ1から、閾値再設定も含めて全て実行
  python3 run_pipeline.py \\
      --outdir results/ \\
      --prefix Hd1 \\
      --bam-dir /path/to/bams/ \\
      --model-del /path/to/DEL_model.joblib \\
      --model-ins /path/to/INS_model.joblib \\
      --hclust-model /path/to/step1_kmeans_model.joblib \\
      --threshold-del 0.95 --threshold-ins 0.90

  # 既存の step1 予測結果に対し、閾値再設定（step2）から最後まで再開
  python3 run_pipeline.py \\
      --outdir results/ \\
      --prefix Hd1 \\
      --start-step 2 \\
      --threshold-del 0.95 --threshold-ins 0.90 \\
      --hclust-model /path/to/step1_kmeans_model.joblib

  # ステップ3（DEL/INS wide table化）から再開（閾値再設定なし）
  python3 run_pipeline.py \\
      --outdir results/ \\
      --prefix Hd1 \\
      --start-step 3 \\
      --hclust-model /path/to/step1_kmeans_model.joblib

  # コマンド確認のみ（実行しない）
  python3 run_pipeline.py --outdir results/ --prefix Hd1 --dry-run ...
"""

import argparse
import csv
import gzip
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _run_cmd_raw(cmd: list, capture: bool = False):
    """Runs cmd via Popen and waits for it to finish. Returns
    (returncode, stdout, stderr). stdout/stderr are None when capture=False
    (inherited, not piped, matching subprocess.run() with no capture_output)."""
    if capture:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
    else:
        proc = subprocess.Popen(cmd)
        proc.wait()
        out, err = None, None
    return proc.returncode, out, err

# ステップ番号 (combine は 7 と 9 の間に挿入)
STEP_SEQUENCE = [1, 2, 3, 5, 6, 7, "combine", 9]

# DEL/INS の両方が揃っているときだけ実行するステップ
INDEL_STEPS = [3, 5, 6, 7]   # DEL を先に、次に INS を実行する


# ============================================================
# ヘルパ
# ============================================================

def run_cmd(cmd: list, dry_run: bool) -> None:
    """コマンドを表示して実行する。失敗したら終了。"""
    print("\n$ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return
    returncode, _out, _err = _run_cmd_raw(cmd)
    if returncode != 0:
        print(f"[ERROR] Command failed (exit code: {returncode})", file=sys.stderr)
        sys.exit(returncode)


def run_cmd_capture(cmd: list, dry_run: bool) -> str:
    """コマンドを実行して標準出力を返す。コンソールにも出力する。dry_run 時は空文字を返す。"""
    print("\n$ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return ""
    returncode, out, err = _run_cmd_raw(cmd, capture=True)
    if out:
        print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    if returncode != 0:
        print(f"[ERROR] Command failed (exit code: {returncode})", file=sys.stderr)
        sys.exit(returncode)
    return out


def parse_elbow_k(output: str) -> int | None:
    """6_elbow.py --print_elbow の出力から k の値を取り出す。"""
    for line in output.splitlines():
        m = re.search(r"k\s*=\s*(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def _count_number_tsv_samples(tsv_path: Path):
    """
    step5（one-hot数値化）TSVのデータ行数（＝サンプル数）を返す。
    ファイルが存在しない場合（--dry-run 等で未生成）は None を返す。
    """
    if not tsv_path.exists():
        return None
    with open(tsv_path) as f:
        n = sum(1 for _ in f) - 1  # ヘッダー行を除く
    return max(n, 0)


def _capped_max_k(requested_max_k: int, n_samples) -> int:
    """
    KMeans は n_clusters <= n_samples でなければ ValueError で異常終了する
    （scikit-learn: "n_samples=X should be >= n_clusters=Y"）。6_elbow.py は
    k=1..max_k を順に KMeans.fit() するため、サンプル数が少ない svtype
    （例: 品種数4に対し --max-k のデフォルト10）でそのまま実行すると
    途中の k でここに落ちる。SNP側の _run_pipeline.py の同名関数と同じ方針で、
    実際のサンプル数に応じて安全な上限に丸める。
    n_samples が不明（--dry-run 等でTSV未生成）な場合は丸めない。
    """
    if n_samples is None:
        return requested_max_k
    return max(1, min(requested_max_k, n_samples - 1))


def parse_cluster_txt(path: Path) -> dict[str, str]:
    """
    cluster txt ファイルを読んで {sample: label} の辞書を返す。

    フォーマット:
      # cluster 1-2
      SAMPLE_A
      SAMPLE_B

      # cluster 1-3
      SAMPLE_C
    """
    result: dict[str, str] = {}
    current_label = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                parts = line.lstrip("# ").strip().split()
                current_label = parts[-1] if parts else None
            elif line and current_label:
                result[line] = current_label
    return result


def detect_types(directory: Path) -> tuple[bool, bool]:
    """フォルダ内の TSV ファイル名から DEL / INS の有無を自動検出する。"""
    tsv_files = list(directory.glob("*.tsv")) + list(directory.glob("*.tsv.gz"))
    has_del = any("_DEL" in p.stem for p in tsv_files)
    has_ins = any("_INS" in p.stem for p in tsv_files)
    return has_del, has_ins


def detect_svtype(path: Path) -> str | None:
    """ファイル名から DEL / INS を判別する（detect_types と同じ方式、1ファイル単位）。"""
    stem = path.name[:-3] if path.name.endswith(".gz") else path.stem
    if "_DEL" in stem:
        return "DEL"
    elif "_INS" in stem:
        return "INS"
    return None


def open_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


# INS_notTP のみ、通常の閾値（--threshold-ins）とは別に専用の閾値を使う。
# INS_TP / DEL は引き続き指定された閾値をそのまま使う。
INS_NOTTP_THRESHOLD = 1.0


def reclassify_file(in_path: Path, out_path: Path,
                     threshold: float, drop_none: bool) -> tuple[int, int]:
    """
    step2: 1ファイル分の TYPE 列を新しい閾値で再判定して out_path に書き出す。

    "PROB_NONE" 以外の "PROB_*" 列すべてを候補クラスとして扱う
    （例: INS なら PROB_INS の1列、DEL なら PROB_DEL_st / PROB_DEL_end の2列）。
    候補クラスのうち最大値が閾値以上ならその列名（PROB_ を除いた部分、
    例: DEL_st, DEL_end, INS）を TYPE とし、全て閾値未満なら NONE とする。
    BAM の再スキャン（再予測）は不要。

    戻り値: (全座位数, 新閾値で NONE 以外と判定された座位数)
    """
    n_total = 0
    n_pos = 0

    with open_maybe_gzip(in_path, "r") as fin, open_maybe_gzip(out_path, "w") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            print(f"[WARN] {in_path} is an empty file (no header); treating all positions as missing "
                  f"and writing an empty output as-is. Will be filled with N via 3_predict_tsv.py's --fill.")
            return 0, 0

        prob_cols = [c for c in fieldnames if c.startswith("PROB_") and c != "PROB_NONE"]
        if not prob_cols:
            print(f"[WARN] No PROB_* columns (other than PROB_NONE) found in {in_path}; "
                  f"treating all positions as missing and writing an empty output as-is "
                  f"(columns: {fieldnames}).")
            return 0, 0

        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()

        for row in reader:
            n_total += 1
            values = {c: float(row[c]) for c in prob_cols}
            best_col = max(values, key=values.get)
            best_val = values[best_col]
            threshold_eff = INS_NOTTP_THRESHOLD if best_col == "PROB_INS_notTP" else threshold

            if best_val >= threshold_eff:
                row["TYPE"] = best_col[len("PROB_"):]
                row["PROB"] = row[best_col]
                n_pos += 1
                writer.writerow(row)
            else:
                row["TYPE"] = "NONE"
                if "PROB_NONE" in row:
                    row["PROB"] = row["PROB_NONE"]
                if not drop_none:
                    writer.writerow(row)

    return n_total, n_pos


def resolve_step3_input_dir(args, outdir: Path) -> Path:
    """
    step3 の入力フォルダを解決する。
    --filtered-dir が明示指定されていればそれを最優先。
    次に step2_threshold/predictions（step2 の出力）が存在すればそちらを使う。
    どちらもなければ step1_predictions/predictions（step1 の生出力）を使う。
    """
    if args.filtered_dir:
        return Path(args.filtered_dir)
    step2_dir = outdir / "step2_threshold" / "predictions"
    if step2_dir.exists() and (list(step2_dir.glob("*.tsv")) or list(step2_dir.glob("*.tsv.gz"))):
        return step2_dir
    return outdir / "step1_predictions" / "predictions"


def resolve_input_dir(args, outdir: Path, start_step) -> Path:
    """
    開始ステップの入力フォルダを返す。
    --predictions-dir / --filtered-dir / --wide-dir / --number-dir で明示指定された場合はそちらを優先する。
    """
    if start_step == 2:
        return Path(args.predictions_dir) if args.predictions_dir else outdir / "step1_predictions" / "predictions"
    elif start_step == 3:
        return resolve_step3_input_dir(args, outdir)
    elif start_step == 5:
        return Path(args.wide_dir) if args.wide_dir else outdir / "step3_wide"
    elif start_step in (6, 7, "combine", 8, 9):
        return Path(args.number_dir) if args.number_dir else outdir / "step5_number"
    return outdir


def resolve_types(args, outdir: Path, start_step) -> tuple[bool, bool]:
    """
    DEL/INS の有無を開始ステップの入力フォルダから検出する。
    step1/2 から始める場合はどちらも True として進める。
    """
    if start_step in (1, 2):
        return True, True

    ref_dir = resolve_input_dir(args, outdir, start_step)

    if not ref_dir.exists():
        print(f"[ERROR] Input folder not found: {ref_dir}", file=sys.stderr)
        print(f"[ERROR] You can also specify it explicitly with --predictions-dir / --filtered-dir / --wide-dir / --number-dir.",
              file=sys.stderr)
        sys.exit(1)

    has_del, has_ins = detect_types(ref_dir)
    if not has_del and not has_ins:
        print(f"[ERROR] No DEL/INS TSV found in {ref_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Auto-detected: DEL={'yes' if has_del else 'no'}, "
          f"INS={'yes' if has_ins else 'no'}  ({ref_dir})")
    return has_del, has_ins


# ============================================================
# 各ステップの実行関数（svtype = "DEL" or "INS" ごとに独立実行）
# ============================================================

def run_step1(args, script_dir: Path, outdir: Path) -> None:
    """step1: BAM → 予測 TSV"""
    out = outdir / "step1_predictions"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script_dir / "1_predict_combined.py"),
        "-b", args.bam_dir,
        "-o", str(out),
        "--step", str(args.predict_step),
        "--threshold-del", str(args.predict_threshold_del),
        "--threshold-ins", str(args.predict_threshold_ins),
        "--ins-mq", str(args.ins_mq),
        "--jobs", str(args.jobs),
    ]
    if not args.keep_features:
        cmd.append("--no-features")
    if args.model_del:
        cmd += ["--model-del", args.model_del]
    if args.model_ins:
        cmd += ["--model-ins", args.model_ins]
    if args.targets:
        cmd += ["--targets", args.targets]
    if args.region:
        cmd += ["--region", args.region]
    if args.mean_depth_tsv:
        cmd += ["--mean-depth-tsv", args.mean_depth_tsv]

    run_cmd(cmd, args.dry_run)


def run_step2(args, outdir: Path) -> bool:
    """
    step2: 閾値を再設定して予測TSVのTYPE列を再判定する（BAM再スキャンなし）。
    --threshold-del / --threshold-ins のいずれも未指定なら何もせずスキップする。
    片方のみ指定された場合、指定のない方の svtype のファイルは元のまま（無変更）でコピーする。

    戻り値: 実際に再分類（step2_threshold/predictions への出力）を行ったかどうか。
    """
    if args.threshold_del is None and args.threshold_ins is None:
        print("[INFO] step2: skipping because neither --threshold-del nor --threshold-ins was specified"
              " (using step1's original classification as-is).")
        return False

    in_dir  = Path(args.predictions_dir) if args.predictions_dir else outdir / "step1_predictions" / "predictions"
    out_dir = outdir / "step2_threshold" / "predictions"

    if args.dry_run:
        print(f"\n[step2] Input: {in_dir}")
        print(f"[step2] Output: {out_dir}")
        print(f"[step2] Thresholds: DEL={args.threshold_del}, INS={args.threshold_ins}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] step2 input folder not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    tsv_files = sorted(list(in_dir.glob("*.tsv")) + list(in_dir.glob("*.tsv.gz")))
    if not tsv_files:
        print(f"[ERROR] No TSV files found in {in_dir}.", file=sys.stderr)
        sys.exit(1)

    for f in tsv_files:
        svtype = detect_svtype(f)
        out_path = out_dir / f.name

        if svtype is None:
            print(f"[WARN] Cannot determine DEL/INS; copying as-is: {f.name}")
            shutil.copy(f, out_path)
            continue

        threshold = args.threshold_del if svtype == "DEL" else args.threshold_ins
        if threshold is None:
            print(f"[INFO] {svtype}: threshold not specified, copying original classification: {f.name}")
            shutil.copy(f, out_path)
            continue

        n_total, n_pos = reclassify_file(f, out_path, threshold, args.drop_none)
        print(f"[INFO] {f.name}: threshold={threshold} → {svtype} classified {n_pos}/{n_total} position(s)")

    print(f"[INFO] step2 complete → {out_dir}")
    return True


def run_step3(args, script_dir: Path, outdir: Path, svtype: str) -> None:
    """step3: 予測 TSV → wide table（DEL / INS それぞれ）"""
    in_dir  = resolve_step3_input_dir(args, outdir)
    out_dir = outdir / "step3_wide"
    out_dir.mkdir(parents=True, exist_ok=True)

    if svtype == "DEL":
        flag = "--del-only"
    elif svtype == "INS_TP":
        flag = "--instp-only"
    elif svtype == "INS_notTP":
        flag = "--insnottp-only"
    else:  # svtype == "INS"（--ins-mode split 以外）
        flag = {"ins": "--ins-only",
                 "instp": "--instp-only",
                 "insnottp": "--insnottp-only"}[args.ins_mode]
    cmd = [
        sys.executable, str(script_dir / "3_predict_tsv.py"),
        "-i", str(in_dir),
        "-o", str(out_dir / f"{args.prefix}_{svtype}.tsv"),
        "--fill", args.fill,
        flag,
    ]
    if args.keep_chrom:
        cmd.append("--keep-chrom")

    run_cmd(cmd, args.dry_run)


def run_step5(args, script_dir: Path, outdir: Path, svtype: str) -> None:
    """step5: wide table → one-hot 数値化（DEL=2クラス, INS=2クラス）"""
    in_dir  = Path(args.wide_dir) if args.wide_dir else outdir / "step3_wide"
    out_dir = outdir / "step5_number"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5_number.py の --mode は del/ins の2択のみ。INS_TP/INS_notTP はどちらも
    # NONE/INS の2クラスone-hot（mode=ins）で扱う。
    number_mode = "del" if svtype == "DEL" else "ins"
    cmd = [
        sys.executable, str(script_dir / "5_number.py"),
        "-i", str(in_dir  / f"{args.prefix}_{svtype}.tsv"),
        "-o", str(out_dir / f"{args.prefix}_{svtype}_number.tsv"),
        "--mode", number_mode,
        "--n-fill", args.n_fill,
    ]
    run_cmd(cmd, args.dry_run)


def run_step6(args, script_dir: Path, outdir: Path, svtype: str) -> int | None:
    """step6: エルボー法で最適 k を推定。検出した k を返す。"""
    in_dir  = Path(args.number_dir) if args.number_dir else outdir / "step5_number"
    out_dir = outdir / "step6_elbow"
    out_dir.mkdir(parents=True, exist_ok=True)

    number_tsv = in_dir / f"{args.prefix}_{svtype}_number.tsv"
    n_samples = _count_number_tsv_samples(number_tsv)
    max_k = _capped_max_k(args.max_k, n_samples)
    if n_samples is not None and max_k < args.max_k:
        print(f"[INFO] step6 {svtype}: adjusted --max_k from {args.max_k} to {max_k} "
              f"to fit n_samples={n_samples}.")

    cmd = [
        sys.executable, str(script_dir / "6_elbow.py"),
        "--input",      str(number_tsv),
        "--max_k",      str(max_k),
        "--output_fig", str(out_dir / f"{args.prefix}_{svtype}_elbow.png"),
        "--print_elbow",
    ]
    output = run_cmd_capture(cmd, args.dry_run)
    k = parse_elbow_k(output)
    if k is not None:
        print(f"[INFO] step6 elbow method -> {svtype}: optimal k = {k}")
    elif not args.dry_run:
        print(f"[WARN] Could not determine elbow k for {svtype}. step7 will use the manual setting.")
    return k


def run_step7(args, script_dir: Path, outdir: Path, svtype: str,
              elbow_k: int | None) -> None:
    """step7: K-Means クラスタリング。k はエルボー法結果を優先、手動指定があればそちらを使う。"""
    in_dir  = Path(args.number_dir) if args.number_dir else outdir / "step5_number"
    out_dir = outdir / "step7_kmeans"
    out_dir.mkdir(parents=True, exist_ok=True)

    hclust_model = args.hclust_model
    manual_k     = args.sub_n_clusters_del if svtype == "DEL" else args.sub_n_clusters_ins

    if not hclust_model:
        print(f"[WARN] step7: --hclust-model not specified, skipping {svtype}.")
        return

    # Determine k: manual setting > elbow method > default 2
    if manual_k is not None:
        n_clusters = manual_k
        print(f"[INFO] {svtype}: k = {n_clusters} (manual)")
    elif elbow_k is not None:
        n_clusters = elbow_k
        print(f"[INFO] {svtype}: k = {n_clusters} (auto, from elbow method)")
    else:
        n_clusters = 2
        print(f"[WARN] {svtype}: k undetermined, using default k=2.")

    out_prefix = str(out_dir / f"{args.prefix}_{svtype}")

    # step3_wide の wide table TSV をハプロタイプ照合用に渡す
    wide_dir = Path(args.wide_dir) if args.wide_dir else outdir / "step3_wide"
    wide_tsv = wide_dir / f"{args.prefix}_{svtype}.tsv"

    cmd = [
        sys.executable, str(script_dir / "7_kmeans.py"),
        "-m", hclust_model,
        "-i", str(in_dir / f"{args.prefix}_{svtype}_number.tsv"),
        "-o", out_prefix,
        "--sub-n-clusters", str(n_clusters),
        "--indel-weight",   str(args.indel_weight),
        "--sub-n-init",     str(args.sub_n_init),
        "--fig",
    ]
    if args.map_file:
        cmd += ["--map-file", args.map_file]
    if wide_tsv.exists():
        cmd += ["--simplified-tsv", str(wide_tsv)]
    if getattr(args, "haplotype_tsv", None):
        cmd += ["--haplotype-tsv", args.haplotype_tsv]
    if getattr(args, "gff", None):
        cmd += ["--gff", args.gff]
    if getattr(args, "region", None):
        cmd += ["--region", args.region]

    run_cmd(cmd, args.dry_run)


def run_combine(args, outdir: Path, svtypes: list[str]) -> Path | None:
    """
    各 svtype（DEL / INS、または --ins-mode split 時の DEL / INS_TP / INS_notTP）の
    nested.txt（この多型専用のSUBクラスタ番号。既報アリルに一致していればアリル名）と
    TOPラベル（SNP側の一次クラスタ番号。どのsvtypeでも同じサンプルなら同じ値）を統合して
    {TOP}-{svtype1}-{svtype2}-... 形式の combined.txt を生成する。svtypes の並び順が
    そのままラベルの並び順になる。

    TOP は単一の数字（例: "3"）とは限らず、二段階クラスタリング（--two-stage）由来の
    combined joblib を --hclust-model に使った場合は「一次クラスタ番号-ローカル番号」
    （例: "3-2"）の複合表記になることがあり、--snp-cluster-txt が指定されていれば
    既報アリルに一致したサンプルのTOPはアリル名になる。複合表記・アリル名いずれも
    分割せず、そのまま1つのラベルとして扱う。

    --snp-cluster-txt が指定されていない場合（未指定・単体CLI実行など）は、
    各svtypeの top.txt（数値のみ、アリル一致情報を持たない）にフォールバックする。

    戻り値: 生成された combined.txt のパス（dry_run 時、または svtypes が空の場合は None）
    """
    kmeans_dir = outdir / "step7_kmeans"
    out_path   = kmeans_dir / f"{args.prefix}_combined.txt"
    nested_paths = {sv: kmeans_dir / f"{args.prefix}_{sv}_out" / "nested.txt" for sv in svtypes}
    top_paths    = {sv: kmeans_dir / f"{args.prefix}_{sv}_out" / "top.txt" for sv in svtypes}
    snp_cluster_txt = getattr(args, "snp_cluster_txt", None)

    if args.dry_run:
        for sv in svtypes:
            print(f"\n[combine] {sv} nested: {nested_paths[sv]}")
            print(f"[combine] {sv} top: {top_paths[sv]}")
        if snp_cluster_txt:
            print(f"[combine] SNP cluster (TOP source): {snp_cluster_txt}")
        print(f"[combine] output: {out_path}")
        return None

    if not svtypes:
        print("[WARN] combine: no target svtypes, not creating combined.txt.", file=sys.stderr)
        return None

    for sv in svtypes:
        if not nested_paths[sv].exists():
            print(f"[ERROR] {sv} nested.txt not found: {nested_paths[sv]}", file=sys.stderr)
            sys.exit(1)
        if not top_paths[sv].exists():
            print(f"[ERROR] {sv} top.txt not found: {top_paths[sv]}", file=sys.stderr)
            sys.exit(1)

    sub_maps = {sv: parse_cluster_txt(nested_paths[sv]) for sv in svtypes}
    top_maps = {sv: parse_cluster_txt(top_paths[sv]) for sv in svtypes}
    all_samples = sorted(set().union(*sub_maps.values()))

    # SNP側の最終cluster.txt（既報アリル一致時はアリル名が入っている）をTOPの
    # 情報源として優先する。sc/内部は常にサンプル名を大文字化して扱うが、この
    # ファイルはGUI側で既に元の表記に復元済みの場合があるため、突き合わせは
    # 大文字化したキーで行う（nested.txt/top.txt 側は元々大文字のまま）。
    snp_cluster_map: dict[str, str] = {}
    if snp_cluster_txt:
        snp_path = Path(snp_cluster_txt)
        if snp_path.exists():
            snp_cluster_map = {s.upper(): label for s, label in parse_cluster_txt(snp_path).items()}
        else:
            print(f"[WARN] combine: --snp-cluster-txt not found, falling back to top.txt: {snp_path}",
                  file=sys.stderr)

    combined: dict[str, list] = defaultdict(list)

    for sample in all_samples:
        top = snp_cluster_map.get(sample.upper())
        subs = []
        for sv in svtypes:
            sub = sub_maps[sv].get(sample)
            if sub is not None:
                subs.append(sub)
                if top is None:
                    top = top_maps[sv].get(sample)
            else:
                # No data for this svtype (0 predictions or filtered out entirely) -> sub-cluster 1
                subs.append("1")
        if top is None:
            # No data for any svtype (in the hclust model but with no predictions at all) -> all 1
            top = "1"
        label = "-".join([top] + subs)
        combined[label].append(sample)

    # txt ファイル書き出し
    with open(out_path, "w", encoding="utf-8") as f:
        for label in sorted(combined.keys()):
            f.write(f"# Allele {label}\n")
            for sample in sorted(combined[label]):
                f.write(f"{sample}\n")
            f.write("\n")

    n_samples = sum(len(v) for v in combined.values())
    print(f"\n[combine] Generated {out_path} "
          f"({n_samples} samples, {len(combined)} clusters)")
    print(f"[combine]   Labels: {sorted(combined.keys())}")
    return out_path


def run_step9(args, script_dir: Path, outdir: Path, svtype: str) -> None:
    """
    step9: PCA プロット。
    クラスタ色付けは各タイプ自身の nested.txt（{TOP}-{DEL} or {TOP}-{INS} 形式）を使用する。
    結合ラベル（{TOP}-{DEL}-{INS}）は step9 のプロットには使わない。
    """
    number_dir   = Path(args.number_dir) if args.number_dir else outdir / "step5_number"
    kmeans_dir   = outdir / "step7_kmeans"
    out_dir      = outdir / "step9_plot"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 各タイプ自身の nested.txt を使用（例: {TOP}-{DEL} or {TOP}-{INS}）
    cluster_file = kmeans_dir / f"{args.prefix}_{svtype}_out" / f"{args.cluster_key}.txt"

    pca_model_out = out_dir / f"{args.prefix}_{svtype}_pca_model.joblib"

    cmd = [
        sys.executable, str(script_dir / "9_plot.py"),
        "--input",           str(number_dir / f"{args.prefix}_{svtype}_number.tsv"),
        "--cluster",         str(cluster_file),
        "--output",          str(out_dir / f"{args.prefix}_{svtype}_plot.png"),
        "--variance_output", str(out_dir / f"{args.prefix}_{svtype}_variance.txt"),
        "--components",      str(args.components),
        "--plot_pc",
    ] + [str(p) for p in args.plot_pc] + [
        "--pca-model-out",   str(pca_model_out),
    ]
    if args.case_map:
        cmd += ["--case-map", args.case_map]

    run_cmd(cmd, args.dry_run)


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "step2 全ステップ（1〜9）一括実行スクリプト。\n"
            "DEL を先に全ステップ処理 → INS を全ステップ処理 → 結合ラベル生成 → step8・9 の順で実行。\n"
            "DEL / INS はファイル名から自動判別する（--mode 指定不要）。\n"
            "最終ラベル形式: {TOPクラスタ}-{DELサブクラスタ}-{INSサブクラスタ}  例: 1-2-3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 実行制御
    ap.add_argument("--outdir",      required=True, help="出力ベースディレクトリ")
    ap.add_argument("--prefix",      default="result", help="出力ファイルのプレフィックス（デフォルト: result）")
    ap.add_argument("--start-step",  default="1",
                    help=f"開始ステップ {STEP_SEQUENCE}（デフォルト: 1）")
    ap.add_argument("--end-step",    default="9",
                    help=f"終了ステップ {STEP_SEQUENCE}（デフォルト: 9）")
    ap.add_argument("--script-dir",  default=None,
                    help="1〜9 の .py スクリプトが入ったフォルダ（デフォルト: このスクリプトと同じフォルダ）")
    ap.add_argument("--dry-run",     action="store_true",
                    help="コマンドを表示するだけで実際には実行しない")

    # 途中再開用: 各ステップの入力フォルダを明示指定
    resume = ap.add_argument_group("途中再開用: 入力フォルダの明示指定（省略時は --outdir 以下のデフォルトパスを使用）")
    resume.add_argument("--predictions-dir", default=None,
                        help="step2 の入力フォルダ（step1 の予測 TSV が入っているフォルダ）。"
                             "省略時は <outdir>/step1_predictions/predictions を使用。")
    resume.add_argument("--filtered-dir",   default=None,
                        help="step3 の入力フォルダを明示的に上書きする（省略時は "
                             "<outdir>/step2_threshold/predictions があればそれを、"
                             "なければ <outdir>/step1_predictions/predictions を自動使用）。"
                             "指定すると自動判定より必ず優先される点に注意。")
    resume.add_argument("--wide-dir",       default=None,
                        help="step5 の入力フォルダ（step3 の wide table TSV が入っているフォルダ）")
    resume.add_argument("--number-dir",     default=None,
                        help="step6〜8 の入力フォルダ（step5 の number TSV が入っているフォルダ）")

    # Step 1
    s1 = ap.add_argument_group("Step 1: 予測")
    s1.add_argument("--bam-dir",           default=None, help="入力 BAM フォルダ")
    s1.add_argument("--model-del",         default=None, help="DEL モデル (.joblib)")
    s1.add_argument("--model-ins",         default=None, help="INS モデル (.joblib)")
    s1.add_argument("--targets",           default=None, help="targets TSV（指定座標のみ予測）")
    s1.add_argument("--region",            default=None, help="予測対象領域（例: chr01:38382382-38385504）")
    s1.add_argument("--predict-step",      type=int,   default=1,    help="スキャンステップ（デフォルト: 1）")
    s1.add_argument("--predict-threshold-del", type=float, default=0.98, help="DEL予測確率閾値（デフォルト: 0.98）")
    s1.add_argument("--predict-threshold-ins", type=float, default=0.98, help="INS予測確率閾値（デフォルト: 0.98）")
    s1.add_argument("--ins-mq",            type=int,   default=60,
                    help="INS特徴量計算のMQ閾値（MQ>=この値のリードのみ対象。1_predict_combined.py にそのまま渡す。デフォルト: 60）")
    s1.add_argument("--jobs",              type=int,   default=2,   help="並列処理数（デフォルト: 1）")
    s1.add_argument("--keep-features",     action="store_true",
                    help="1_predict_combined.py の特徴量TSV出力を有効にする（デフォルトは --no-features で出力しない）")
    s1.add_argument("--mean-depth-tsv",    default=None,
                    help="estimate_mean_depth.py が出力したTSV（sample, mean_depth）。"
                         "領域抽出済みBAM（±10kb等）を --bam-dir に渡す場合は必須。"
                         "1_predict_combined.py にそのまま渡され、BAMからの再計算を省略する。")

    # Step 2
    s2 = ap.add_argument_group("Step 2: 閾値の再設定（任意）")
    s2.add_argument("--threshold-del", type=float, default=None,
                    help="DEL の新しい閾値。指定するとstep2でTYPE列を再判定する（PROB_DEL_st/PROB_DEL_endの大きい方で判定）")
    s2.add_argument("--threshold-ins", type=float, default=None,
                    help="INS の新しい閾値。指定するとstep2でTYPE列を再判定する（PROB_INSで判定）")
    s2.add_argument("--drop-none",     action="store_true",
                    help="step2でNONEと判定された行を出力から除外する（デフォルトは全座位を保持）")

    # Step 3
    s3 = ap.add_argument_group("Step 3: wide table 化")
    s3.add_argument("--fill",           default="N",  help="欠損補完値（デフォルト: N）")
    s3.add_argument("--keep-chrom",     action="store_true", help="染色体名を正規化しない")
    s3.add_argument("--ins-mode", choices=["ins", "instp", "insnottp", "split"], default="ins",
                    help="INS の wide table 化モード（DELには影響しない）。"
                         "ins: INS_TP/INS_notTP混在の1本（デフォルト、3_predict_tsv.pyの--ins-only相当）。"
                         "instp: INS_TPのみの1本（--instp-only相当）。"
                         "insnottp: INS_notTPのみの1本（--insnottp-only相当）。"
                         "split: INS_TPとINS_notTPを別々のsvtype（INS_TP/INS_notTP）として"
                         "wide table化からクラスタリングまで完全に別々に実行する。")

    # Step 5
    s5 = ap.add_argument_group("Step 5: one-hot 数値化")
    s5.add_argument("--n-fill", choices=["mean", "zero"], default="zero",
                    help="N 補完方法: mean=座位平均, zero=ゼロベクトル（デフォルト: zero）")

    # Step 6
    s6 = ap.add_argument_group("Step 6: エルボー法")
    s6.add_argument("--max-k", type=int, default=10, help="最大クラスタ数（デフォルト: 10）")

    # Step 7
    s7 = ap.add_argument_group("Step 7: K-Means クラスタリング")
    s7.add_argument("--hclust-model", default=None,
                    help="step1 の 7_kmeans.py が出力した joblib（samples・labels_top を含む）。DEL・INS 共通で使用する。")
    s7.add_argument("--snp-cluster-txt", default=None,
                    help="combineステップでTOPラベルの元にするSNP側の最終cluster.txt"
                         "（step1 single-stage: step7_{prefix}_cluster.txt / two-stage: "
                         "combined/step7_{prefix}_cluster.txt）。指定すると、既報アリルに"
                         "一致したサンプルのTOPラベルがアリル名になる（未指定時は "
                         "top.txt の数値のみのTOPにフォールバックする）。")
    s7.add_argument("--sub-n-clusters-del", type=int, default=None,
                    help="DEL の K-Means k（省略時は step6 エルボー法の結果を自動使用）")
    s7.add_argument("--sub-n-clusters-ins", type=int, default=None,
                    help="INS の K-Means k（省略時は step6 エルボー法の結果を自動使用）")
    s7.add_argument("--indel-weight", type=float, default=1.0,
                    help="INS/DEL 列への重み（デフォルト: 3.0）")
    s7.add_argument("--sub-n-init",   type=int, default=10,
                    help="K-Means の n_init（デフォルト: 10）")
    s7.add_argument("--map-file",     default=None,
                    help="サンプル名対応表 TSV（model_name / sub_name の 2 列）")
    s7.add_argument("--haplotype-tsv", default=None,
                    help="ハプロタイプ定義TSV。gene_model_make/cds/ 形式（variant_type 列あり）と"
                         "既存形式（hap_name/vt_kind 列あり）を自動検出。"
                         "cds 形式の場合は DEL/INS のみ使用し SNP はスキップ。"
                         "nested.txt にハプロタイプ名を反映し joblib にも保存される。")
    s7.add_argument("--gff", default=None,
                    help="GFF3 ファイル。cds.tsv に coord_type=cds/aa/gdna のエントリがある場合に必要。"
                         "7_kmeans.py に渡され coord_converter.py でゲノム座標に変換される。")

    # Step 9
    s9 = ap.add_argument_group("Step 9: PCA プロット")
    s9.add_argument("--components", type=int, default=2,
                    help="PCA 成分数（デフォルト: 2）")
    s9.add_argument("--plot-pc",    nargs="+", type=int, default=[1, 2],
                    help="プロットする PC のペア（デフォルト: 1 2）")
    s9.add_argument("--cluster-key", choices=["top", "sub", "nested"], default="sub",
                    help="step7 のクラスタファイル種別 combined.txt がない場合のフォールバック（デフォルト: sub）")
    s9.add_argument("--case-map", default=None,
                    help="2列TSV（大文字化サンプル名<TAB>元の表記）。指定すると step9 の "
                         "PCAプロット点ラベルを元の表記で描画する（9_plot.py にそのまま渡す）。")

    args = ap.parse_args()

    outdir     = Path(args.outdir)
    script_dir = Path(args.script_dir) if args.script_dir else Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # 開始・終了ステップを解決（文字列 "combine" も含む）
    def to_step_key(s):
        try:
            return int(s)
        except ValueError:
            return s  # "combine"

    start_key = to_step_key(args.start_step)
    end_key   = to_step_key(args.end_step)

    if start_key not in STEP_SEQUENCE:
        print(f"[ERROR] Invalid --start-step value: {args.start_step}", file=sys.stderr)
        sys.exit(1)
    if end_key not in STEP_SEQUENCE:
        print(f"[ERROR] Invalid --end-step value: {args.end_step}", file=sys.stderr)
        sys.exit(1)

    start_idx = STEP_SEQUENCE.index(start_key)
    end_idx   = STEP_SEQUENCE.index(end_key)
    if start_idx > end_idx:
        print("[ERROR] --start-step comes after --end-step.", file=sys.stderr)
        sys.exit(1)

    steps_to_run = STEP_SEQUENCE[start_idx: end_idx + 1]
    print(f"[INFO] Steps to run: {steps_to_run}")
    if args.dry_run:
        print("[INFO] --dry-run mode: only prints commands, does not execute them")

    # DEL/INS 自動検出
    has_del, has_ins = resolve_types(args, outdir, start_key)

    # --ins-mode split の場合、INS を INS_TP / INS_notTP という
    # 2つの独立した svtype として wide table化〜クラスタリングまで別々に実行する。
    ins_svtypes = ["INS_TP", "INS_notTP"] if args.ins_mode == "split" else ["INS"]
    del_svtypes = ["DEL"] if has_del else []
    all_svtypes = del_svtypes + (ins_svtypes if has_ins else [])

    # ステップ実行
    elbow_k: dict[str, int | None] = {}
    combined_txt: Path | None = None

    for step in steps_to_run:
        print(f"\n{'='*60}")
        if step != "combine":
            label = f"Step {step}"
        else:
            combine_label_format = "{TOP}-" + "-".join("{" + sv + "}" for sv in all_svtypes)
            label = f"Combine step (generating {combine_label_format} labels)"
        print(f"  Starting {label}")
        print(f"{'='*60}")

        if step == 1:
            if not args.bam_dir:
                print("[ERROR] step1 requires --bam-dir.", file=sys.stderr)
                sys.exit(1)
            run_step1(args, script_dir, outdir)

        elif step == 2:
            run_step2(args, outdir)

        elif step == 3:
            # Process DEL first, then INS (INS_TP -> INS_notTP order when split)
            for sv in all_svtypes:
                print(f"\n--- step3 {sv} ---")
                run_step3(args, script_dir, outdir, sv)

        elif step == 5:
            for sv in all_svtypes:
                print(f"\n--- step5 {sv} ---")
                run_step5(args, script_dir, outdir, sv)

        elif step == 6:
            for sv in all_svtypes:
                print(f"\n--- step6 {sv} ---")
                elbow_k[sv] = run_step6(args, script_dir, outdir, sv)

        elif step == 7:
            for sv in all_svtypes:
                print(f"\n--- step7 {sv} ---")
                run_step7(args, script_dir, outdir, sv, elbow_k.get(sv))

        elif step == "combine":
            combined_txt = run_combine(args, outdir, all_svtypes)

        elif step == 9:
            for sv in all_svtypes:
                print(f"\n--- step9 {sv} (colored by {sv}'s own nested labels) ---")
                run_step9(args, script_dir, outdir, sv)

        print(f"  {label} done")

    # Look for an existing file even if the combine step wasn't run
    if combined_txt is None and not args.dry_run:
        candidate = outdir / "step7_kmeans" / f"{args.prefix}_combined.txt"
        if candidate.exists():
            combined_txt = candidate

    print(f"\n{'='*60}")
    print(f"  All steps done!  Output: {outdir}")
    print(f"\n  - Per-type nested label (this svtype's own sub-cluster number)")
    for svtype in all_svtypes:
        p = outdir / "step7_kmeans" / f"{args.prefix}_{svtype}_out" / "nested.txt"
        print(f"    {svtype}: {p}")
    if combined_txt:
        label_format = "{TOP}-" + "-".join("{" + sv + "}" for sv in all_svtypes)
        print(f"\n  - Final combined label ({label_format} format)")
        print(f"    {combined_txt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
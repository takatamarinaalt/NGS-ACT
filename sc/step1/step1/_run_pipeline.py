#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1 全ステップ一括実行スクリプト

実行順序:
  1. 1_bam_choice.py           : BAM から遺伝子領域を抽出
  2. 2_vcf.py                  : VCF 生成
  3. 3_vcf_select.py           : VCF フィルタリング（all / exon / intron モード）
  4. 4_hetero_N_one_vcf_tsv.py : VCF → TSV 変換
  5. 5_snp_number.py           : バイナリ特徴量変換
  6. 6_elbow.py                : エルボー法でクラスタ数選定
  7. 7_kmeans.py               : K-means クラスタリング
  8. 9_plot.py                 : PCA プロット（step5 特徴量から直接PCA）
  9. 10_haplotype_match.py     : ハプロタイプ照合（--haplotype-tsv 指定時のみ）

【変異領域のモード選択】
  --vcf-mode all   : RAP-DB 遺伝子領域全体の多型を対象（デフォルト）
  --vcf-mode exon  : GFF3 エキソン領域の多型のみを対象

使用例:
  # 遺伝子領域全体（all モード）
  python3 _run_pipeline.py \\
      --outdir results/Hd1/ \\
      --prefix Hd1 \\
      --bam-dir /path/to/bams/ \\
      --ref /path/to/ref.fa \\
      --regions chr06:9336376-9338569 \\
      --vcf-mode all

  # エキソンのみ（exon モード）
  python3 _run_pipeline.py \\
      --outdir results/Hd1/ \\
      --prefix Hd1 \\
      --bam-dir /path/to/bams/ \\
      --ref /path/to/ref.fa \\
      --regions chr06:9336376-9338569 \\
      --vcf-mode exon \\
      --gff /path/to/transcripts.gff \\
      --gene Hd1

  # step5 から再開（BAM/VCF 生成済みの場合）
  python3 _run_pipeline.py \\
      --outdir results/Hd1/ \\
      --prefix Hd1 \\
      --start-step 5

  # コマンド確認のみ（実行しない）
  python3 _run_pipeline.py --dry-run --outdir results/Hd1/ --prefix Hd1 ...

  # 二段階クラスタリング（イントロンSNPで一次クラスタ → クラスタごとに遺伝子全体で二次クラスタ）
  python3 _run_pipeline.py \\
      --outdir results/Hd1/ \\
      --prefix Hd1 \\
      --bam-dir /path/to/bams/ \\
      --ref /path/to/ref.fa \\
      --regions chr06:9336376-9338569 \\
      --gff /path/to/transcripts.gff \\
      --gene Hd1 \\
      --haplotype-tsv /path/to/Hd1_cds.tsv \\
      --two-stage

中間ファイル構成:
  {outdir}/
    step1_bams/                     - 抽出済み BAM
    step2_raw.vcf                   - 生成 VCF
    step3_selected.vcf              - フィルタ済み VCF
    step4_{prefix}.tsv              - VCF → TSV
    step5_{prefix}.tsv              - バイナリ特徴量
    step6_{prefix}_elbow.png        - エルボープロット
    step7_{prefix}_cluster.txt      - クラスタ結果
    step7_{prefix}_model.joblib     - K-means モデル（ハプロタイプ情報も追記される）
    step9_{prefix}_pca.png          - PCA プロット
    step9_{prefix}_variance.txt     - 寄与率
    step9_{prefix}_pca_model.joblib - PCA モデル（step10 と同じ空間を共有するため）
    step10_{prefix}_haplotype_pca.png    - ハプロタイプ矢印付きPCAプロット（--haplotype-tsv 指定時）
    step10_{prefix}_haplotype_match.txt  - ハプロタイプ一致結果（--haplotype-tsv 指定時）

【二段階クラスタリング（--two-stage）を指定した場合】
  亜種間（japonica/indica）の違いに埋もれてしまいがちな、品種内の細かいSNPの
  違い（1つや2つの違い）を捉えるため、以下の2段階でクラスタリングを行う。
    stage0: イントロン領域のSNPのみで一次クラスタリング
    stage1: 一次クラスタごとに、遺伝子領域全体のSNPで二次クラスタリング
            （--vcf-mode all 相当の処理を、各一次クラスタの品種のみに絞って実行）
  二次クラスタの番号は、全stage1を通した連番（cluster1, cluster2, ...）を用いる。
  一次クラスタ番号と二次クラスタ番号の対応関係は two_stage_cluster_summary.txt に保存される。

  {outdir}/
    step1_bams/, step2_raw.vcf   - stage0/stage1共通（BAM抽出・VCF生成は1回のみ）
    stage0_intron/
      step3_intron.vcf                    - イントロン領域のみのVCF
      step4_{prefix}_intron.tsv
      step5_{prefix}_intron.tsv
      step6_{prefix}_intron_elbow.png
      step7_{prefix}_intron_cluster.txt   - 一次クラスタ結果（stage1で参照）
      step7_{prefix}_intron_model.joblib
    stage1_cluster{N}/              - 一次クラスタNに属する品種のみを対象とした二次クラスタリング
      step3_0_sample_subset.vcf     - 一次クラスタNの品種のみに絞ったVCF
      step3_selected.vcf            - 遺伝子領域全体で選択したVCF（--vcf-mode all相当）
      step4_{prefix}_cN.tsv ... step9_{prefix}_cN_pca.png / _variance.txt / _pca_model.joblib
      step10_{prefix}_cN_haplotype_pca.png / _haplotype_match.txt （--haplotype-tsv 指定時）
    two_stage_cluster_summary.txt   - 一次クラスタ ⇔ 二次クラスタ（全体通し番号）の対応関係
"""

import argparse
import subprocess
import sys
from pathlib import Path

STEP_SEQUENCE = [1, 2, 3, 4, 5, 6, 7, 9, 10]


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

SCRIPT_NAMES = {
    1:  "1_bam_choice.py",
    2:  "2_vcf.py",
    3:  "3_vcf_select.py",
    4:  "4_hetero_N_one_vcf_tsv.py",
    5:  "5_snp_number.py",
    6:  "6_elbow.py",
    7:  "7_kmeans.py",
    9:  "9_plot.py",
    10: "10_haplotype_match.py",
}


# ============================================================
# ヘルパ
# ============================================================

def run_cmd(cmd: list, dry_run: bool) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return
    returncode, _out, _err = _run_cmd_raw(cmd)
    if returncode != 0:
        print(f"[ERROR] Command failed (exit code: {returncode})", file=sys.stderr)
        sys.exit(returncode)


def run_cmd_optional(cmd: list, dry_run: bool, step_desc: str = "") -> bool:
    """
    run_cmd と同じだが、失敗しても sys.exit() せず警告を出して処理を継続する。
    戻り値: 成功したら True、失敗したら False。

    ハプロタイプ照合（step10）のように、失敗してもそれ以外の（既に完了した）
    クラスタリング結果を丸ごと無効にすべきでない補助的なステップに使用する。
    例: --haplotype-tsv のファイルが1つ見つからないだけで、
    そのgeneのstage0/stage1/stage2の全結果が失われるのを防ぐ。
    """
    print("\n$ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return True
    returncode, _out, _err = _run_cmd_raw(cmd)
    if returncode != 0:
        label = f"{step_desc}" if step_desc else "Command"
        print(f"[WARN] {label} failed (exit code: {returncode}). "
              f"Skipping this step's result and continuing with the rest of the pipeline.", file=sys.stderr)
        return False
    return True


def _count_tsv_feature_columns(tsv_path: Path):
    """
    step5（バイナリ特徴量）TSVの特徴量列数（先頭のサンプル名列を除く）を返す。
    ファイルが存在しない場合は None を返す。

    一次クラスタ内で、対象となる全SNP位置が同一遺伝子型になってしまうと
    （二段階クラスタリング特有: 元々「亜種差」だった多型がstage0で分離された結果、
    そのstage1サブグループ内では1/1を持つ品種がゼロになり、
    4_hetero_N_one_vcf_tsv.py がその行を丸ごと除外してしまう）、
    特徴量列が0個のTSVになりうる。KMeans/PCAはこの状態を扱えずエラーになるため、
    事前に検出して回避する。
    """
    if not tsv_path.exists():
        return None
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n")
    if not header:
        return 0
    cols = header.split("\t")
    return max(len(cols) - 1, 0)  # 先頭列（サンプル名）を除く


_DUMMY_FEATURE_COL = "NO_VARIANT_DUMMY"


def _ensure_nonzero_features(tsv_path: Path, dry_run: bool, context: str) -> None:
    """
    step5（バイナリ特徴量）TSVの特徴量列が0個の場合、KMeans/エルボー法が
    そのままでは ValueError で異常終了してしまう（sklearn は0特徴量の
    入力を受け付けない）。対象領域にALTホモの多型が1つも無い品種群
    （＝品種間の差が検出できない）などで実際に起こりうるため、
    ダミー列（全員0）を1つ追加して回避する
    （run_stage0_intron_clustering / run_stage1_subcluster と同じ方針。
    このファイルを直接読む run_step6/run_step7 は、ダミー列だけが
    存在することを _is_dummy_only_features() で検出して1クラスタとして扱う）。
    """
    if dry_run or not tsv_path.exists():
        return
    if _count_tsv_feature_columns(tsv_path) != 0:
        return
    print(f"[INFO] {context}: 0 feature columns (no difference detected between samples); "
          f"adding a dummy column and treating this as a single cluster.")
    import pandas as _pd
    df = _pd.read_csv(tsv_path, sep="\t")
    df[_DUMMY_FEATURE_COL] = 0
    df.to_csv(tsv_path, sep="\t", index=False)


def _is_dummy_only_features(tsv_path: Path, dry_run: bool) -> bool:
    """True if tsv_path's only feature column is the dummy one added by
    _ensure_nonzero_features() above (i.e. there was genuinely no usable
    SNP feature for this sample set)."""
    if dry_run or not tsv_path.exists():
        return False
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t")[1:]  # 先頭列（サンプル名）を除く
    return cols == [_DUMMY_FEATURE_COL]


def _capped_pca_components(requested: int, n_samples, n_features) -> int:
    """
    PCA(n_components=k) は k <= min(n_samples, n_features) でなければ
    ValueError で異常終了する（scikit-learn: "n_components=k must be between
    0 and min(n_samples, n_features)=N"）。
    サンプル数・特徴量数が少ない一次クラスタ（二段階クラスタリングのstage1で
    1品種しかいないクラスタなど）でデフォルトの --components（2）のまま
    9_plot.py を実行すると、ここで落ちてしまうため安全な範囲に丸める。
    n_samples / n_features が不明（--dry-run 等でTSV未生成）な場合は丸めない。
    特徴量が0列（対象領域にALTホモの多型が1つも無い場合など）の場合、
    PCAは原理的に実行不可能なため 0 を返す（呼び出し側は 0 のとき
    9_plot.py の実行自体をスキップすること -- 1に丸めてしまうと
    「0特徴量に対して1次元PCA」という無効な呼び出しになり、
    9_plot.py が ValueError で終了コード1を返して落ちる）。
    """
    if n_samples is None or n_features is None:
        return requested
    upper = min(n_samples, n_features)
    if upper <= 0:
        return 0
    return max(1, min(requested, upper))


def _count_tsv_samples(tsv_path: Path):
    """
    step5（バイナリ特徴量）TSVのデータ行数（＝サンプル数）を返す。
    ファイルが存在しない場合（--dry-run 実行時など）は None を返す。
    """
    if not tsv_path.exists():
        return None
    with open(tsv_path) as f:
        n = sum(1 for _ in f) - 1  # ヘッダー行を除く
    return max(n, 0)


def _read_vcf_samples(vcf_path: Path) -> list[str]:
    """
    VCFの #CHROM ヘッダー行からサンプル名一覧を取得する（見つからなければ空リスト）。
    4_hetero_N_one_vcf_tsv.py / vcf_sample_subset.py と同じ正規化
    （basename化 → .sorted.bam/.sort.bam/.bam を除去 → 大文字化）を適用する。
    これをしないと、VCFヘッダーに残る生のBAMパス（例: "dir/Sample.sort.bam"）が
    そのままクラスタファイルに書かれてしまい、後続の vcf_sample_subset.py 等が
    正規化済み名前（例: "SAMPLE"）と一致させられなくなる。
    """
    if not vcf_path.exists():
        return []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                return [
                    name.split("/")[-1]
                        .replace(".sorted.bam", "")
                        .replace(".sort.bam", "")
                        .replace(".bam", "")
                        .upper()
                    for name in cols[9:]
                ]
            if not line.startswith("#"):
                break
    return []


def _capped_max_k(requested_max_k: int, n_samples) -> int:
    """
    KMeans は n_clusters <= n_samples でなければ ValueError で異常終了する
    （scikit-learn: "n_samples=X should be >= n_clusters=Y"）。
    品種数が少ないクラスタ（二段階クラスタリングの一次クラスタ内など）で
    デフォルトの --max_k（10）のままエルボー法を回すとここで落ちてしまうため、
    実際のサンプル数に応じて安全な上限に丸める。
    n_samples が不明（--dry-run 等でTSV未生成）な場合は丸めずそのまま返す。
    """
    if n_samples is None:
        return requested_max_k
    return max(1, min(requested_max_k, n_samples - 1))


def _capped_n_clusters(requested_k: int, n_samples, context: str) -> int:
    """
    --n-clusters で手動指定されたクラスタ数が実際のサンプル数を超える場合、
    KMeans がエラーになる前に安全な値へ丸めて警告を出す。
    """
    if n_samples is None:
        return requested_k
    if requested_k > n_samples:
        capped = max(1, n_samples)
        print(f"[WARN] {context}: the requested cluster count {requested_k} exceeds the sample count {n_samples}; "
              f"adjusting to {capped}.", file=sys.stderr)
        return capped
    return requested_k


def run_cmd_capture(cmd: list, dry_run: bool) -> str:
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
    for line in output.splitlines():
        if "Final selected optimal cluster count: k =" in line:
            try:
                return int(line.split("=")[-1].strip())
            except ValueError:
                pass
    return None


def collect_bam_files(bam_dir: Path) -> list[Path]:
    bams = sorted(bam_dir.glob("*.bam"))
    bams = [b for b in bams if not b.name.endswith(".bam.bai")]
    # macOS writes AppleDouble sidecar files (e.g. "._sample.bam") next to real files
    # on non-native filesystems (exFAT/FAT32 external drives, network shares). These
    # match "*.bam" but aren't real BAMs and make pysam/samtools fail with a cryptic
    # "Exec format error" -- filter them out everywhere we glob for BAMs.
    bams = [b for b in bams if not b.name.startswith("._")]
    return bams


# ============================================================
# 各ステップの実行関数
# ============================================================

def run_step1(args, script_dir: Path, outdir: Path) -> None:
    """BAM から遺伝子領域を抽出"""
    out_bam_dir = outdir / "step1_bams"
    out_bam_dir.mkdir(parents=True, exist_ok=True)

    if not args.bam_dir and not args.bams:
        print("[ERROR] step1 requires --bam-dir or --bams.", file=sys.stderr)
        sys.exit(1)
    if not args.regions:
        print("[ERROR] step1 requires --regions.", file=sys.stderr)
        sys.exit(1)

    if args.bams:
        bam_files = list(args.bams)
    else:
        bam_dir_path = Path(args.bam_dir)
        bam_files = collect_bam_files(bam_dir_path) if bam_dir_path.exists() else []
        if not bam_files and not args.dry_run:
            print(f"[ERROR] No BAM files found: {args.bam_dir}", file=sys.stderr)
            sys.exit(1)
        if not bam_files and args.dry_run:
            bam_files = [Path(args.bam_dir) / "*.bam"]  # dry-run 用プレースホルダ

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[1]),
        "--input_bams", *[str(b) for b in bam_files],
        "--regions",    *args.regions,
        "--output_dir", str(out_bam_dir),
    ]
    run_cmd(cmd, args.dry_run)


def run_step2(args, script_dir: Path, outdir: Path) -> None:
    """VCF 生成"""
    out_vcf = outdir / "step2_raw.vcf"

    bam_dir = outdir / "step1_bams"
    if bam_dir.exists() and list(bam_dir.glob("*.bam")):
        bam_files = collect_bam_files(bam_dir)
    elif args.bams:
        bam_files = list(args.bams)
    elif args.bam_dir:
        bam_dir_path = Path(args.bam_dir)
        bam_files = collect_bam_files(bam_dir_path) if bam_dir_path.exists() else []
        if not bam_files and args.dry_run:
            bam_files = [bam_dir_path / "*.bam"]
    else:
        if not args.dry_run:
            print("[ERROR] step2 requires --bam-dir or --bams.", file=sys.stderr)
            sys.exit(1)
        bam_files = [Path("*.bam")]

    if not args.ref:
        print("[ERROR] step2 requires --ref.", file=sys.stderr)
        sys.exit(1)

    if not args.bcftools:
        print("[ERROR] step2 requires --bcftools (please specify your environment's bcftools path).",
              file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[2]),
        "-b",  *[str(b) for b in bam_files],
        "-r",  args.ref,
        "-o",  str(out_vcf),
        "-t",  str(args.threads),
        "--min-dp", str(args.min_dp),
        "--bcftools", args.bcftools,
    ]
    run_cmd(cmd, args.dry_run)


def run_step3(args, script_dir: Path, outdir: Path) -> None:
    """VCF フィルタリング（all / exon / intron モード）"""
    in_vcf  = outdir / "step2_raw.vcf"
    out_vcf = outdir / "step3_selected.vcf"

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[3]),
        "--input_vcf",  str(in_vcf),
        "--output_vcf", str(out_vcf),
        "--mode",       args.vcf_mode,
    ]

    if args.vcf_mode == "all":
        if not args.regions:
            print("[ERROR] --vcf-mode all requires --regions.", file=sys.stderr)
            sys.exit(1)
        cmd += ["--regions", *args.regions]
    else:  # exon / intron
        if not args.gff or not args.gene:
            print(f"[ERROR] --vcf-mode {args.vcf_mode} requires --gff and --gene.", file=sys.stderr)
            sys.exit(1)
        cmd += ["--gff", args.gff, "--gene", args.gene]

    run_cmd(cmd, args.dry_run)


def run_step4(args, script_dir: Path, outdir: Path) -> None:
    """VCF → TSV 変換"""
    in_vcf  = outdir / "step3_selected.vcf"
    out_tsv = outdir / f"step4_{args.prefix}.tsv"

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[4]),
        "--input",  str(in_vcf),
        "--output", str(out_tsv),
    ]
    run_cmd(cmd, args.dry_run)


def run_step5(args, script_dir: Path, outdir: Path) -> None:
    """バイナリ特徴量変換"""
    in_tsv  = outdir / f"step4_{args.prefix}.tsv"
    out_tsv = outdir / f"step5_{args.prefix}.tsv"

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[5]),
        "-i", str(in_tsv),
        "-o", str(out_tsv),
    ]
    run_cmd(cmd, args.dry_run)

    # 対象領域にALTホモの多型が1つも無い品種群だと特徴量0列になり、このあとの
    # step6（エルボー法）/step7（KMeans）/step9（PCA）がそのままでは異常終了する
    # ため、事前にダミー列を追加しておく（two-stage の stage0/stage1 側で
    # 既に使われているのと同じ回避策 -- 詳細は _ensure_nonzero_features 参照）。
    _ensure_nonzero_features(out_tsv, args.dry_run, "step5")


def run_step6(args, script_dir: Path, outdir: Path) -> int | None:
    """エルボー法で最適クラスタ数を推定"""
    in_tsv   = outdir / f"step5_{args.prefix}.tsv"
    out_fig  = outdir / f"step6_{args.prefix}_elbow.png"

    if _is_dummy_only_features(in_tsv, args.dry_run):
        print("[INFO] step6: 0 feature columns (no difference detected between samples); "
              "skipping the elbow method and treating this as a single cluster.")
        return 1

    n_samples = _count_tsv_samples(in_tsv)
    max_k = _capped_max_k(args.max_k, n_samples)
    if n_samples is not None and max_k < args.max_k:
        print(f"[INFO] step6: adjusted --max_k from {args.max_k} to {max_k} to match the sample count ({n_samples}).")

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[6]),
        "--input",      str(in_tsv),
        "--max_k",      str(max_k),
        "--output_fig", str(out_fig),
        "--print_elbow",
    ]
    output = run_cmd_capture(cmd, args.dry_run)

    k = parse_elbow_k(output)
    if k is not None:
        print(f"[INFO] step6: elbow method -> optimal k = {k}")
    else:
        print("[WARN] step6: could not obtain the elbow k. Using the --n-clusters value instead.")
    return k


def run_step7(args, script_dir: Path, outdir: Path, elbow_k: int | None) -> None:
    """K-means クラスタリング"""
    in_tsv       = outdir / f"step5_{args.prefix}.tsv"
    out_cluster  = outdir / f"step7_{args.prefix}_cluster.txt"
    out_model    = outdir / f"step7_{args.prefix}_model.joblib"

    n_samples = _count_tsv_samples(in_tsv)

    # クラスタ数の決定: ダミー列（特徴量0列の代替） > 手動指定 > エルボー法。
    # ダミー列の場合は品種間の差が検出できていないため、--n-clusters が
    # 指定されていても分割せず強制的に1クラスタとして扱う。
    if _is_dummy_only_features(in_tsv, args.dry_run):
        k = 1
        print("[INFO] step7: 0 feature columns; treating this as a single cluster.")
    elif args.n_clusters:
        k = _capped_n_clusters(args.n_clusters, n_samples, "step7")
        print(f"[INFO] step7: --n-clusters specified, k = {k}")
    elif elbow_k is not None:
        k = elbow_k
        print(f"[INFO] step7: using elbow method k = {k}")
    else:
        print("[ERROR] step7: cluster count unknown. Please specify --n-clusters.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[7]),
        "--input",      str(in_tsv),
        "--n_clusters", str(k),
        "--output",     str(out_cluster),
        "--model",      str(out_model),
    ]
    if getattr(args, "regions", None):
        cmd += ["--region", args.regions[0]]
    run_cmd(cmd, args.dry_run)


def run_step9(args, script_dir: Path, outdir: Path) -> None:
    """PCA プロット"""
    input_tsv   = outdir / f"step5_{args.prefix}.tsv"
    cluster     = outdir / f"step7_{args.prefix}_cluster.txt"
    out_png     = outdir / f"step9_{args.prefix}_pca.png"
    out_var     = outdir / f"step9_{args.prefix}_variance.txt"
    out_pca_model = outdir / f"step9_{args.prefix}_pca_model.joblib"

    n_samples  = _count_tsv_samples(input_tsv)
    n_features = _count_tsv_feature_columns(input_tsv)
    n_components = _capped_pca_components(args.pca_components, n_samples, n_features)
    if n_components == 0:
        print(f"[WARN] step9: PCA cannot run with 0 feature columns (n_samples={n_samples}). "
              f"Skipping the PCA plot/model generation (this does not affect the cluster results themselves).")
        return
    if n_components < args.pca_components:
        print(f"[INFO] step9: adjusted --components from {args.pca_components} to {n_components} "
              f"to match the sample count ({n_samples}) / feature count ({n_features}).")

    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[9]),
        "--input",           str(input_tsv),
        "--cluster",         str(cluster),
        "--output",          str(out_png),
        "--components",      str(n_components),
        "--plot_pc",         *[str(p) for p in args.plot_pc],
        "--variance_output", str(out_var),
        "--pca-model-out",   str(out_pca_model),
    ]
    if args.case_map:
        cmd += ["--case-map", args.case_map]
    run_cmd(cmd, args.dry_run)


def run_step10(args, script_dir: Path, outdir: Path) -> None:
    """ハプロタイプ照合・矢印付き PCA プロット"""
    if not args.haplotype_tsv:
        print("[SKIP] step10: --haplotype-tsv was not specified; skipping")
        return

    vcf_path      = outdir / "step3_selected.vcf"
    pca_model     = outdir / f"step9_{args.prefix}_pca_model.joblib"
    cmd = [
        sys.executable, str(script_dir / SCRIPT_NAMES[10]),
        "--outdir",        str(outdir),
        "--prefix",        args.prefix,
        "--haplotype-tsv", args.haplotype_tsv,
        "--pca-components", str(args.pca_components),
        "--plot-pc",       *[str(p) for p in args.plot_pc],
        "--vcf",           str(vcf_path),
    ]
    if args.gff:
        cmd += ["--gff", args.gff]
    if pca_model.exists():
        cmd += ["--pca-model", str(pca_model)]
    if args.case_map:
        cmd += ["--case-map", args.case_map]

    # step10 はあくまで付加的な処理（ハプロタイプ照合）のため、失敗しても
    # 既に完了しているstep1〜9の結果は残す（sys.exitで丸ごと無効にしない）
    run_cmd_optional(cmd, args.dry_run, step_desc="step10 (haplotype matching)")


# ============================================================
# 二段階クラスタリング（--two-stage）
#   stage0: イントロン領域SNPのみで一次クラスタリング
#   stage1: 一次クラスタごとに、遺伝子領域全体のSNPで二次クラスタリング
# ============================================================

def _list_clusters_from_file(cluster_file: Path):
    """cluster.txt (7_kmeans.py 形式) から (cluster_id, label, n_samples) のリストを取得する。"""
    result = []
    idx = -1
    label = None
    count = 0
    with open(cluster_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if idx >= 0:
                    result.append((idx + 1, label, count))
                idx += 1
                label = line[1:].strip()
                count = 0
            elif line.strip():
                count += 1
        if idx >= 0:
            result.append((idx + 1, label, count))
    return result


def run_stage0_intron_clustering(args, script_dir: Path, outdir: Path) -> Path:
    """
    stage0: イントロン領域のSNPのみを用いた一次クラスタリング。
    3(mode=intron) → 4 → 5 → 6 → 7 を実行し、一次クラスタ結果ファイルのパスを返す。
    """
    stage0_dir = outdir / "stage0_intron"
    stage0_dir.mkdir(parents=True, exist_ok=True)

    raw_vcf = outdir / "step2_raw.vcf"
    if not raw_vcf.exists() and not args.dry_run:
        print(f"[ERROR] {raw_vcf} not found. step1/step2 must complete first.",
              file=sys.stderr)
        sys.exit(1)

    # 3: イントロン領域のみ抽出
    # 3_vcf_select.py はシングルエキソン遺伝子などイントロンが1つも見つからない場合、
    # sys.exit(1) で異常終了する仕様（スタンドアロン実行時の通常のエラー動作としては妥当）。
    # ここを run_cmd（失敗時にパイプライン全体を sys.exit させる）で呼ぶと、この1遺伝子の
    # stage0〜stage3処理が丸ごと止まり、stage0_intron/ に何も出力されないまま終了してしまう
    # （D61 で実際に発生した症状）。run_cmd_optional に変え、失敗時は VCF のサンプル一覧から
    # 全品種を1つの一次クラスタとして扱う合成クラスタファイルを作り、以降の
    # step4〜7（イントロン特徴量に基づく処理）をスキップして処理を継続する。
    intron_vcf = stage0_dir / "step3_intron.vcf"
    ok = run_cmd_optional([
        sys.executable, str(script_dir / SCRIPT_NAMES[3]),
        "--input_vcf",  str(raw_vcf),
        "--output_vcf", str(intron_vcf),
        "--mode",       "intron",
        "--gff",        args.gff,
        "--gene",       args.gene,
    ], args.dry_run, step_desc="stage0 (intron region extraction)")

    if not ok and not args.dry_run:
        samples = _read_vcf_samples(raw_vcf)
        print(f"[INFO] stage0: intron extraction failed for gene '{args.gene}' "
              f"(it may be a single-exon gene, or the gene may not be found in the GFF3). "
              f"Skipping intron-based primary clustering and treating all {len(samples)} sample(s) "
              f"as a single primary cluster.")
        intron_cluster = stage0_dir / f"step7_{args.prefix}_intron_cluster.txt"
        with open(intron_cluster, "w", encoding="utf-8") as f:
            f.write("# cluster 1\n")
            for s in samples:
                f.write(f"{s}\n")
            f.write("\n")
        return intron_cluster

    # 4: VCF → TSV
    intron_tsv4 = stage0_dir / f"step4_{args.prefix}_intron.tsv"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[4]),
        "--input",  str(intron_vcf),
        "--output", str(intron_tsv4),
    ], args.dry_run)

    # 5: バイナリ特徴量変換
    intron_tsv5 = stage0_dir / f"step5_{args.prefix}_intron.tsv"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[5]),
        "-i", str(intron_tsv4),
        "-o", str(intron_tsv5),
    ], args.dry_run)

    # 特徴量が0列になっていないか確認する。シングルエキソン遺伝子などイントロンが
    # GFF3から見つからない場合、step3(mode=intron)が空のVCFを生成し、5_snp_number.py
    # の出力が0列になりうる。KMeans/エルボー法はこの状態を扱えず ValueError で
    # 異常終了し、この遺伝子のstage0〜stage3処理全体が止まってしまうため、事前に
    # 検出してダミー列を追加し、一次クラスタを1つ（分割なし）として扱う
    # （stage1側の同種の対処 [run_stage1_subcluster] と同じ方針）。
    force_k1_stage0 = False
    if not args.dry_run:
        n_features_check = _count_tsv_feature_columns(intron_tsv5)
        if n_features_check == 0:
            print(f"[INFO] stage0: 0 intron-region feature columns "
                  f"(the gene may have no introns, or may not be found in the GFF3); "
                  f"adding a dummy column and treating this as a single primary cluster.")
            import pandas as _pd
            _df = _pd.read_csv(intron_tsv5, sep="\t")
            _df["NO_VARIANT_DUMMY"] = 0
            _df.to_csv(intron_tsv5, sep="\t", index=False)
            force_k1_stage0 = True

    if force_k1_stage0:
        # 分割せず1クラスタに固定するため、エルボー法は行わない
        k = 1
    else:
        # 6: エルボー法でクラスタ数を推定（サンプル数に応じて max_k を丸める）
        max_k_intron_req = args.max_k_intron if args.max_k_intron else args.max_k
        n_samples = _count_tsv_samples(intron_tsv5)
        max_k_intron = _capped_max_k(max_k_intron_req, n_samples)
        if n_samples is not None and max_k_intron < max_k_intron_req:
            print(f"[INFO] stage0: adjusted --max_k from "
                  f"{max_k_intron_req} to {max_k_intron} to match the sample count ({n_samples}).")

        intron_fig = stage0_dir / f"step6_{args.prefix}_intron_elbow.png"
        output6 = run_cmd_capture([
            sys.executable, str(script_dir / SCRIPT_NAMES[6]),
            "--input",      str(intron_tsv5),
            "--max_k",      str(max_k_intron),
            "--output_fig", str(intron_fig),
            "--print_elbow",
        ], args.dry_run)
        elbow_k = parse_elbow_k(output6)

        if args.n_clusters_intron:
            k = _capped_n_clusters(args.n_clusters_intron, n_samples, "stage0")
            print(f"[INFO] stage0: --n-clusters-intron specified, k = {k}")
        elif elbow_k is not None:
            k = elbow_k
            print(f"[INFO] stage0: using elbow method k = {k}")
        else:
            print("[ERROR] stage0: cluster count unknown. Please specify --n-clusters-intron.", file=sys.stderr)
            sys.exit(1)

    # 7: K-means（イントロン領域SNPによる一次クラスタリング）
    intron_cluster = stage0_dir / f"step7_{args.prefix}_intron_cluster.txt"
    intron_model   = stage0_dir / f"step7_{args.prefix}_intron_model.joblib"
    cmd7 = [
        sys.executable, str(script_dir / SCRIPT_NAMES[7]),
        "--input",      str(intron_tsv5),
        "--n_clusters", str(k),
        "--output",     str(intron_cluster),
        "--model",      str(intron_model),
    ]
    if args.regions:
        cmd7 += ["--region", args.regions[0]]
    run_cmd(cmd7, args.dry_run)

    return intron_cluster


def run_stage1_subcluster(args, script_dir: Path, outdir: Path, cluster_id: int,
                           label: str, n_samples: int, intron_cluster_file: Path):
    """
    stage1: 一次クラスタ内の品種のみを対象に、遺伝子領域全体のSNPで二次クラスタリングを行う
    （既存パイプラインの --vcf-mode all（step3〜step9、および --haplotype-tsv 指定時は step10）
    と同一処理を、サブセットVCFに対して実行）。

    二次クラスタの番号は、一次クラスタごとに常に1から振り直すローカル番号を用いる
    （7_kmeans.py の --start_number はデフォルト値の1のまま渡す）。
    以前は全stage1を通した連番（例: 一次クラスタ1がcluster1・2を使ったら
    一次クラスタ2はcluster3から）を用いていたが、この番号だけを見ても
    どの一次クラスタ由来か判別できず、stage0とstage1の結果が実質的に混ざって
    見えてしまう問題があった。stage2（combine）側で「一次クラスタ番号-ローカル番号」
    （例: cluster 1-1, cluster 1-2, cluster 2-1）の複合表記に組み直すことで、
    stage0/stage1 それぞれの番号を区別できるようにする。

    サンプル数が --min-cluster-size 未満の一次クラスタは、統計的に意味のある分割が
    難しいと判断し、エルボー法を行わずクラスタ数 k=1（分割なし）として扱う。
    完全にスキップ（結果を出力しない）のではなく k=1 で処理を続けることで、
    combined（stage2）で全品種を漏れなく1つのPCA・クラスタファイルにまとめられるようにする。

    戻り値: (k, cluster_file) 常にタプルを返す。
            クラスタ数を決定できない致命的エラー時のみ (None, None)。
    """
    sub_prefix = f"{args.prefix}_c{cluster_id}"
    sub_dir = outdir / f"stage1_cluster{cluster_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] --- stage1: primary cluster {cluster_id} (label='{label}', n={n_samples}) ---")

    force_k1 = n_samples < args.min_cluster_size
    if force_k1:
        print(f"[INFO] stage1: primary cluster {cluster_id} has {n_samples} sample(s) "
              f"(below --min-cluster-size {args.min_cluster_size}); not splitting further and "
              f"treating it as a single secondary cluster.")

    raw_vcf = outdir / "step2_raw.vcf"

    # 0: 一次クラスタに属する品種のみに絞ったVCFを作成
    sub_vcf = sub_dir / "step3_0_sample_subset.vcf"
    run_cmd([
        sys.executable, str(script_dir / "vcf_sample_subset.py"),
        "--input_vcf",    str(raw_vcf),
        "--output_vcf",   str(sub_vcf),
        "--cluster_file", str(intron_cluster_file),
        "--cluster_id",   str(cluster_id),
    ], args.dry_run)

    # 3: 遺伝子領域全体を対象に選択（all モード、既存パイプラインと同じ処理）
    sel_vcf = sub_dir / "step3_selected.vcf"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[3]),
        "--input_vcf",  str(sub_vcf),
        "--output_vcf", str(sel_vcf),
        "--mode",       "all",
        "--regions",    *args.regions,
    ], args.dry_run)

    # 4: VCF → TSV
    tsv4 = sub_dir / f"step4_{sub_prefix}.tsv"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[4]),
        "--input",  str(sel_vcf),
        "--output", str(tsv4),
    ], args.dry_run)

    # 5: バイナリ特徴量変換
    tsv5 = sub_dir / f"step5_{sub_prefix}.tsv"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[5]),
        "-i", str(tsv4),
        "-o", str(tsv5),
    ], args.dry_run)

    # 特徴量が0列になっていないか確認する。
    # 二段階クラスタリング特有の問題: 元々「亜種差」だった多型がstage0で
    # 分離された結果、そのstage1サブグループ内では1/1を持つ品種が
    # ゼロになり、4_hetero_N_one_vcf_tsv.py がその行を丸ごと除外してしまう
    # ことがある。この場合 step5 の特徴量列が0個になり、KMeans/PCAが
    # そのままでは実行できないため、ダミー列を1つ追加して回避する
    # （分割不能なので force_k1 と同様に k=1 として扱う）。
    if not args.dry_run:
        n_features_check = _count_tsv_feature_columns(tsv5)
        if n_features_check == 0:
            print(f"[INFO] stage1 primary cluster {cluster_id}: 0 feature columns "
                  f"(no difference detected between samples within this cluster); "
                  f"adding a dummy column and treating this as a single cluster.")
            import pandas as _pd
            _df = _pd.read_csv(tsv5, sep="\t")
            _df["NO_VARIANT_DUMMY"] = 0
            _df.to_csv(tsv5, sep="\t", index=False)
            force_k1 = True

    if force_k1:
        # 分割せず1クラスタに固定するため、エルボー法は行わない
        k = 1
    else:
        # 6: エルボー法（サンプル数に応じて max_k を安全な範囲に丸める）
        n_samples_sub = _count_tsv_samples(tsv5)
        max_k_sub = _capped_max_k(args.max_k, n_samples_sub)
        if n_samples_sub is not None and max_k_sub < args.max_k:
            print(f"[INFO] stage1 primary cluster {cluster_id}: adjusted --max_k from "
                  f"{args.max_k} to {max_k_sub} to match the sample count ({n_samples_sub}).")

        fig6 = sub_dir / f"step6_{sub_prefix}_elbow.png"
        out6 = run_cmd_capture([
            sys.executable, str(script_dir / SCRIPT_NAMES[6]),
            "--input",      str(tsv5),
            "--max_k",      str(max_k_sub),
            "--output_fig", str(fig6),
            "--print_elbow",
        ], args.dry_run)
        ek = parse_elbow_k(out6)

        if args.n_clusters:
            k = _capped_n_clusters(args.n_clusters, n_samples_sub, f"stage1 primary cluster {cluster_id}")
            print(f"[INFO] stage1 primary cluster {cluster_id}: --n-clusters specified, k = {k}")
        elif ek is not None:
            k = ek
            print(f"[INFO] stage1 primary cluster {cluster_id}: using elbow method k = {k}")
        else:
            print(f"[ERROR] stage1 primary cluster {cluster_id}: cluster count unknown. Please specify --n-clusters.",
                  file=sys.stderr)
            return None, None

    # 7: K-means（二次クラスタの番号は一次クラスタ内でのローカル番号。--start_number は
    #    デフォルトの1のまま渡さない。cid を付けた複合表記は stage2（combine）側で行う）
    cluster7 = sub_dir / f"step7_{sub_prefix}_cluster.txt"
    model7   = sub_dir / f"step7_{sub_prefix}_model.joblib"
    cmd7 = [
        sys.executable, str(script_dir / SCRIPT_NAMES[7]),
        "--input",        str(tsv5),
        "--n_clusters",   str(k),
        "--output",       str(cluster7),
        "--model",        str(model7),
    ]
    if args.regions:
        cmd7 += ["--region", args.regions[0]]
    run_cmd(cmd7, args.dry_run)

    # 9: PCA プロット（step5特徴量から直接PCA。PCAモデルも保存しstep10と共有する）
    png9 = sub_dir / f"step9_{sub_prefix}_pca.png"
    var9 = sub_dir / f"step9_{sub_prefix}_variance.txt"
    pca_model9 = sub_dir / f"step9_{sub_prefix}_pca_model.joblib"

    n_samples_pca  = _count_tsv_samples(tsv5)
    n_features_pca = _count_tsv_feature_columns(tsv5)
    n_components9  = _capped_pca_components(args.pca_components, n_samples_pca, n_features_pca)
    if n_components9 == 0:
        print(f"[WARN] stage1 primary cluster {cluster_id}: PCA cannot run with 0 feature columns "
              f"(n_samples={n_samples_pca}). Skipping the PCA plot/model generation "
              f"(this does not affect the cluster results themselves).")
    else:
        if n_components9 < args.pca_components:
            print(f"[INFO] stage1 primary cluster {cluster_id}: adjusted --components from "
                  f"{args.pca_components} to {n_components9} to match the sample count "
                  f"({n_samples_pca}) / feature count ({n_features_pca}).")

        cmd9 = [
            sys.executable, str(script_dir / SCRIPT_NAMES[9]),
            "--input",           str(tsv5),
            "--cluster",         str(cluster7),
            "--output",          str(png9),
            "--components",      str(n_components9),
            "--plot_pc",         *[str(p) for p in args.plot_pc],
            "--variance_output", str(var9),
            "--pca-model-out",   str(pca_model9),
        ]
        if args.case_map:
            cmd9 += ["--case-map", args.case_map]
        run_cmd(cmd9, args.dry_run)

    # 10: ハプロタイプ照合（--haplotype-tsv 指定時のみ。失敗しても致命的エラーにしない）
    if args.haplotype_tsv:
        cmd10 = [
            sys.executable, str(script_dir / SCRIPT_NAMES[10]),
            "--outdir",        str(sub_dir),
            "--prefix",        sub_prefix,
            "--haplotype-tsv", args.haplotype_tsv,
            "--pca-components", str(n_components9),
            "--plot-pc",       *[str(p) for p in args.plot_pc],
            "--vcf",           str(sel_vcf),
        ]
        if args.gff:
            cmd10 += ["--gff", args.gff]
        if pca_model9.exists():
            cmd10 += ["--pca-model", str(pca_model9)]
        if args.case_map:
            cmd10 += ["--case-map", args.case_map]
        run_cmd_optional(cmd10, args.dry_run,
                          step_desc=f"stage1 primary cluster {cluster_id} step10 (haplotype matching)")
    else:
        print(f"[SKIP] stage1 primary cluster {cluster_id}: step10 (skipped; --haplotype-tsv not specified)")

    return k, cluster7


def run_two_stage_pipeline(args, script_dir: Path, outdir: Path) -> None:
    """
    二段階クラスタリングの一括実行:
      stage0: イントロン領域SNPのみで一次クラスタリング
      stage1: 一次クラスタごとに、遺伝子領域全体のSNPで二次クラスタリング
              （二次クラスタの番号は一次クラスタごとに1から振り直すローカル番号。
                最終的な表記は stage2（combine）で「一次クラスタ番号-ローカル番号」
                （例: cluster 1-1, cluster 1-2, cluster 2-1）に組み直され、
                stage0/stage1 それぞれの番号が独立して残る）
    """
    if not args.gff or not args.gene:
        print("[ERROR] --two-stage requires --gff and --gene (for intron region extraction).",
              file=sys.stderr)
        sys.exit(1)
    if not args.regions:
        print("[ERROR] --two-stage requires --regions (used for stage1's whole-gene-region selection).",
              file=sys.stderr)
        sys.exit(1)

    intron_cluster_file = run_stage0_intron_clustering(args, script_dir, outdir)

    if args.dry_run:
        print("\n[INFO] Because of --dry-run, the actual cluster breakdown cannot be obtained. "
              "Only showing command construction for stage1 (primary cluster 1) and stage2 (combine).")
        run_stage1_subcluster(args, script_dir, outdir, 1, "dry-run", args.min_cluster_size,
                               intron_cluster_file)
        run_stage2_combine(args, script_dir, outdir, [1])
        return

    clusters = _list_clusters_from_file(intron_cluster_file)
    print(f"\n[INFO] stage0: split into {len(clusters)} primary cluster(s) based on intron SNPs.")
    for cid, label, n in clusters:
        print(f"       primary cluster {cid} (label='{label}'): {n} sample(s)")

    summary_rows = []  # (stage0_cluster_id, stage0_label, n_samples, k or None)
    processed_cluster_ids = []  # combined ステップで参照する成功したcluster_idの並び

    for cid, label, n in clusters:
        k, _cluster_file = run_stage1_subcluster(
            args, script_dir, outdir, cid, label, n, intron_cluster_file
        )
        summary_rows.append((cid, label, n, k))
        if k:
            processed_cluster_ids.append(cid)

    # 一次クラスタ番号 ⇔ 二次クラスタ番号（一次クラスタごとのローカル番号）の対応関係をまとめて保存
    summary_path = outdir / "two_stage_cluster_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Two-stage clustering correspondence summary\n")
        f.write("=" * 60 + "\n")
        f.write("Correspondence between stage0 (intron region) primary clusters and\n")
        f.write("stage1 (whole gene region) secondary clusters (local number within each primary cluster).\n")
        f.write("The final cluster number (combined/step7_{prefix}_cluster.txt) is a 1-based\n")
        f.write("sequential number (flat number) assigned in order of appearance of each\n")
        f.write("\"primary cluster number-local number\" combination; the mapping back to the\n")
        f.write("original combinations is saved in combined/step7_{prefix}_cluster_number_map.txt.\n")
        f.write("(A primary cluster with fewer samples than --min-cluster-size is treated as a single cluster, unsplit)\n\n")
        for cid, lbl, n, k in summary_rows:
            if k:
                cluster_desc = (f"cluster {cid}-1" if k == 1
                                 else f"cluster {cid}-1 - cluster {cid}-{k}")
                f.write(f"Primary cluster {cid} (label='{lbl}', {n} sample(s)) -> {cluster_desc}\n")
                f.write(f"  Details: stage1_cluster{cid}/step7_{args.prefix}_c{cid}_cluster.txt\n")
            else:
                f.write(f"Primary cluster {cid} (label='{lbl}', {n} sample(s)) -> could not be processed due to an error\n")

    print(f"\n[INFO] Saved the primary-cluster <-> secondary-cluster correspondence summary: {summary_path}")

    print(f"\n{'='*60}")
    print(f"  Done: two-stage clustering complete")
    print(f"  stage0 (intron primary clusters): {outdir / 'stage0_intron'}")
    print(f"  stage1 (secondary clusters per primary cluster): {outdir}/stage1_cluster*/")
    print(f"  Correspondence summary: {summary_path}")
    print(f"{'='*60}")

    # stage2: 全品種を1つのPCA・クラスタファイル・joblibにまとめる
    run_stage2_combine(args, script_dir, outdir, processed_cluster_ids)


def _apply_allele_match_to_combined_cluster(combined_dir: Path, prefix: str, combined_cluster: Path) -> None:
    """stage2のstep10（アリル照合）が combined/ に出力した
    step10_{prefix}_allele_match.txt を読み、既報アリルに一致したサンプルの
    ラベルを、そのままの数値フラット番号からアリル名に置き換えて
    combined_cluster を書き直す。

    DEL/INS側（7_kmeans.pyのnested.txt）は既にアリル一致時にラベルを
    アリル名へ置き換えているが、SNP側の最終cluster.txtは従来この置き換えを
    行っていなかった（step10は別ファイル(*_allele_match.txt)に結果を書くのみで、
    cluster.txt自体は書き換えない）。そのため、SNP側がアリルに一致していても、
    最終的な結合ラベル（sc/step1/step2_newfea2/_run_pipeline.pyのrun_combine()が
    --snp-cluster-txt でこのファイルを読む）にはTOP部分が数値のまま出てしまう
    問題があった。ここで一致結果を反映させることで、DEL/INS側と同じ挙動にする。

    置き換えるのは表示用のこのテキストファイルのみで、bundleの
    "labels_top"/"cluster_labels"（二段階階層predictが実際に使う数値ラベル）は
    変更しない -- 新品種のSNPアリル判定は determining 側の
    bundle["haplotype_info"] を使った独立の照合で行われており、
    flat番号→アリル名の対応に依存していないため、表示用ファイルだけを
    直しても階層predictの動作には影響しない。

    allele_match.txt が無い/一致が1件も無い場合は何もしない。
    """
    match_path = combined_dir / f"step10_{prefix}_allele_match.txt"
    if not match_path.exists():
        return

    sample_to_alleles: dict[str, list[str]] = {}
    with open(match_path, encoding="utf-8") as f:
        f.readline()  # header: allele\tsample\tcluster
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            allele_name, sample, cluster = parts[0], parts[1], parts[2]
            if cluster == "-" or not sample or sample.startswith("("):
                continue  # no-match placeholder row
            sample_to_alleles.setdefault(sample, []).append(allele_name)

    if not sample_to_alleles:
        return

    # allele_match.txt のサンプル表記と combined_cluster 側のサンプル表記の大文字小文字が
    # 必ずしも一致しない（GUI側の restore_case_in_cluster_file() がどのタイミングで
    # どちらに実行されるかに関わらず安全にするため）、突き合わせは大文字化したキーで行う
    # -- run_combine() の --snp-cluster-txt 突き合わせと同じ方針。
    sample_to_label = {s.upper(): "/".join(sorted(set(names))) for s, names in sample_to_alleles.items()}

    groups: dict[str, list[str]] = {}
    current_label = None
    with open(combined_cluster, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_parts = line.lstrip("#").strip().split()
                current_label = header_parts[-1] if header_parts else None
                continue
            sample = line
            label = sample_to_label.get(sample.upper(), current_label)
            groups.setdefault(label, []).append(sample)

    def _sort_key(label: str):
        return (0, int(label)) if label.isdigit() else (1, 0, label)

    with open(combined_cluster, "w", encoding="utf-8") as out:
        for label in sorted(groups.keys(), key=_sort_key):
            header_line = f"# cluster {label}" if label.isdigit() else f"# {label}"
            out.write(header_line + "\n")
            for s in groups[label]:
                out.write(f"{s}\n")
            out.write("\n")

    print(f"[OK] Applied allele-name matches from {match_path.name} to {combined_cluster.name} "
          f"({len(sample_to_label)} sample(s) matched)")


def run_stage2_combine(args, script_dir: Path, outdir: Path, cluster_ids: list) -> None:
    """
    stage2: stage1の全結果を1つにまとめ、全品種を同一のPCA空間で可視化する。

    クラスタの判定自体は stage0/stage1 で意図的に別モデルに分けて行っている
    （亜種差に細かいSNP差が埋もれるのを避けるため）ので、判定モデルそのものを
    1つに統合することはしない。一方、可視化目的のPCAは全品種で共有した方が
    有用なため、ここで以下を1回だけ計算・統合する。
      - 全品種・遺伝子領域全体の特徴量（step3(all)〜5相当。stage1ごとの
        サブセットではなく、全品種を対象に計算し直す）
      - 全stage1の cluster.txt を1つに結合。各stage1内のローカル番号
        （一次クラスタごとに1から振り直したもの）は、一次クラスタ番号 cid と
        組み合わせた「{cid}-{ローカル番号}」の出現順に、1始まりの通し番号
        （フラット番号）を割り振り、最終的なクラスタ番号として使う
        （stage0/stage1 の番号を混ぜずに区別しつつ、ユーザーへの表示は
        単一の数字にする）。元の「{cid}-{ローカル番号}」との対応表は
        bundle["compound_label_map"] と step7_{prefix}_cluster_number_map.txt
        に保存し、情報は失わない。
      - 上記2つから、全品種を同一空間にプロットした1枚のPCA・PCAモデル・joblib
      - --haplotype-tsv 指定時は、結合結果に対して1回だけ step10 を実行
        （stage1ごとに分散していたハプロタイプ照合結果を1つの結果にまとめる）
    """
    if not cluster_ids:
        print("[WARN] stage2: no clusters to combine; skipping the combined step.",
              file=sys.stderr)
        return

    combined_dir = outdir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    print(f"\n{'='*60}")
    print(f"  stage2: combining all samples into a single PCA/cluster file")
    print(f"{'='*60}")

    raw_vcf = outdir / "step2_raw.vcf"

    # 3: 全品種・遺伝子領域全体でVCF選択（可視化のための特徴量を1回だけ計算し直す）
    sel_vcf = combined_dir / "step3_selected.vcf"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[3]),
        "--input_vcf",  str(raw_vcf),
        "--output_vcf", str(sel_vcf),
        "--mode",       "all",
        "--regions",    *args.regions,
    ], args.dry_run)

    # 4: VCF → TSV
    tsv4 = combined_dir / f"step4_{prefix}.tsv"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[4]),
        "--input",  str(sel_vcf),
        "--output", str(tsv4),
    ], args.dry_run)

    # 5: バイナリ特徴量変換
    tsv5 = combined_dir / f"step5_{prefix}.tsv"
    run_cmd([
        sys.executable, str(script_dir / SCRIPT_NAMES[5]),
        "-i", str(tsv4),
        "-o", str(tsv5),
    ], args.dry_run)

    if args.dry_run:
        print("[INFO] --dry-run: skipping cluster.txt combine and step9/step10 execution.")
        return

    # 7相当: 全stage1のcluster.txtを1つに結合。各stage1のラベルはローカル番号
    # （一次クラスタごとに1始まり）なので、一次クラスタ番号 cid と組み合わせた
    # 「{cid}-{ローカル番号}」の出現順に1始まりの通し番号（フラット番号）を
    # 割り振り、それを最終的なクラスタ番号として書き出す。
    import re as _re_combine
    combined_cluster = combined_dir / f"step7_{prefix}_cluster.txt"
    n_total_samples = 0
    samples_ordered: list[str] = []
    cluster_of_sample: dict[str, int] = {}
    compound_label_map: dict[str, int] = {}  # "{cid}-{local}" -> フラット番号（1始まり）

    def _flat_number(compound: str) -> int:
        if compound not in compound_label_map:
            compound_label_map[compound] = len(compound_label_map) + 1
        return compound_label_map[compound]

    blocks_by_flat: dict[int, list[str]] = {}
    for cid in cluster_ids:
        src = outdir / f"stage1_cluster{cid}" / f"step7_{prefix}_c{cid}_cluster.txt"
        if not src.exists():
            print(f"[WARN] stage2: {src} not found, skipping.", file=sys.stderr)
            continue
        text = src.read_text(encoding="utf-8")

        compound_label = None
        block_samples: list[str] = []
        blocks: list[tuple[str, list[str]]] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                if compound_label is not None:
                    blocks.append((compound_label, block_samples))
                local_label = line[1:].strip()
                m = _re_combine.search(r"(\d+)", local_label)
                local_num = m.group(1) if m else local_label
                compound_label = f"{cid}-{local_num}"
                block_samples = []
            elif line:
                block_samples.append(line)
        if compound_label is not None:
            blocks.append((compound_label, block_samples))

        for compound_label, block_samples in blocks:
            flat = _flat_number(compound_label)
            blocks_by_flat.setdefault(flat, [])
            for s in block_samples:
                blocks_by_flat[flat].append(s)
                samples_ordered.append(s)
                cluster_of_sample[s] = flat
                n_total_samples += 1

    with open(combined_cluster, "w", encoding="utf-8") as out:
        for flat in sorted(blocks_by_flat):
            out.write(f"# cluster {flat}\n")
            for s in blocks_by_flat[flat]:
                out.write(f"{s}\n")
            out.write("\n")

    # フラット番号と元の「一次クラスタ-ローカル番号」の対応表（情報を失わないため）
    number_map_path = combined_dir / f"step7_{prefix}_cluster_number_map.txt"
    with open(number_map_path, "w", encoding="utf-8") as out:
        for compound, flat in sorted(compound_label_map.items(), key=lambda kv: kv[1]):
            cid_str, local_str = compound.split("-", 1)
            out.write(f"{flat}\tprimary_cluster={cid_str}\tlocal_cluster={local_str}\n")

    print(f"[OK] Combined cluster results: {combined_cluster} (total {n_total_samples} samples)")
    print(f"[OK] Saved cluster number map: {number_map_path}")

    # 9: 結合PCA（全品種を同一空間にプロット。PCAモデルも保存しstep10と共有する）
    png9 = combined_dir / f"step9_{prefix}_pca.png"
    var9 = combined_dir / f"step9_{prefix}_variance.txt"
    pca_model9 = combined_dir / f"step9_{prefix}_pca_model.joblib"

    n_samples_pca  = _count_tsv_samples(tsv5)
    n_features_pca = _count_tsv_feature_columns(tsv5)
    n_components9  = _capped_pca_components(args.pca_components, n_samples_pca, n_features_pca)
    if n_components9 == 0:
        print(f"[WARN] stage2: cannot run PCA with 0 feature columns (n_samples={n_samples_pca}). "
              f"Skipping PCA plot/model generation (this does not affect the cluster results themselves).")
    else:
        if n_components9 < args.pca_components:
            print(f"[INFO] stage2: adjusted --components from {args.pca_components} to {n_components9} "
                  f"to fit n_samples={n_samples_pca}, n_features={n_features_pca}.")

        cmd9 = [
            sys.executable, str(script_dir / SCRIPT_NAMES[9]),
            "--input",           str(tsv5),
            "--cluster",         str(combined_cluster),
            "--output",          str(png9),
            "--components",      str(n_components9),
            "--plot_pc",         *[str(p) for p in args.plot_pc],
            "--variance_output", str(var9),
            "--pca-model-out",   str(pca_model9),
        ]
        if args.case_map:
            cmd9 += ["--case-map", args.case_map]
        run_cmd(cmd9, args.dry_run)

    # 7相当のjoblib: 実際に単一モデルで学習したわけではないため、
    # 完全なKMeansモデルは持たないが、10_haplotype_match.py が要求する
    # bundle(dict)形式で保存しておくことで、ハプロタイプ情報の追記に対応する。
    model_path = combined_dir / f"step7_{prefix}_model.joblib"
    try:
        import joblib as _jl

        # cluster_labels/labels_top は、元の「一次クラスタ番号-ローカル番号」
        # 複合表記の出現順に振った1始まりのフラット番号（通し番号）を使う。
        # 元の複合表記との対応は compound_label_map に保存するため、情報は失われない。
        # step2/7_kmeans.py（--hclust-model）は "samples" + "labels_top"（サンプル順に
        # 対応するTOPラベル。整数である必要はなく、文字列でもそのままラベルとして使われる）
        # か "linkage" のどちらかを要求するが、二段階クラスタリングでは単一の
        # KMeans/linkage は存在しないため、そのままでは "labels_top" も "linkage" も無い
        # joblib になり、--hclust-model にこの combined joblib を渡すと
        # 「joblib に linkage もありません」で失敗し、何も出力されなくなる。
        labels_top = [str(cluster_of_sample[s]) for s in samples_ordered]

        bundle = {
            "model": None,  # 二段階クラスタリングのため単一のKMeansオブジェクトは存在しない
            "samples": samples_ordered,
            "labels_top": labels_top,  # step2 --hclust-model が参照するTOPラベル（フラット番号の文字列）
            "cluster_labels": [f"cluster {cluster_of_sample[s]}" for s in samples_ordered],
            "compound_label_map": compound_label_map,  # "{cid}-{local}" -> フラット番号
            "method": "two_stage_combined",
            "n_clusters": len(set(cluster_of_sample.values())),
            "gene_region": args.regions[0] if args.regions else None,
            "note": "Combined result of the two-stage clustering (stage0 intron -> stage1 whole gene). "
                    "The model key is not a single trained model but cluster_labels represents the "
                    "final assignment. cluster_labels/labels_top are flat, 1-based sequential numbers "
                    "assigned to the original \"primary cluster number-local number\" compound labels "
                    "(e.g. cluster 4 / labels_top='4'); the mapping back to the original compound "
                    "labels is saved in compound_label_map and step7_{prefix}_cluster_number_map.txt.",
        }
        _jl.dump(bundle, model_path)
        print(f"[OK] Saved combined joblib: {model_path}")
    except Exception as e:
        print(f"[WARN] stage2: failed to save combined joblib: {e}", file=sys.stderr)

    # 10: アリル照合（--haplotype-tsv 指定時のみ、結合結果に対して1回だけ実行。
    #     失敗しても既に完了しているstage0/stage1/stage2の結果は残す）
    if args.haplotype_tsv:
        cmd10 = [
            sys.executable, str(script_dir / SCRIPT_NAMES[10]),
            "--outdir",        str(combined_dir),
            "--prefix",        prefix,
            "--haplotype-tsv", args.haplotype_tsv,
            "--pca-components", str(n_components9),
            "--plot-pc",       *[str(p) for p in args.plot_pc],
            "--vcf",           str(sel_vcf),
        ]
        if args.gff:
            cmd10 += ["--gff", args.gff]
        if pca_model9.exists():
            cmd10 += ["--pca-model", str(pca_model9)]
        if args.case_map:
            cmd10 += ["--case-map", args.case_map]
        run_cmd_optional(cmd10, args.dry_run, step_desc="stage2 step10 (allele matching)")
        if not args.dry_run:
            _apply_allele_match_to_combined_cluster(combined_dir, prefix, combined_cluster)
    else:
        print("[SKIP] stage2: step10 (skipped, --haplotype-tsv not specified)")

    print(f"\n{'='*60}")
    print(f"  Done: combined result")
    print(f"  {combined_dir}/step9_{prefix}_pca.png    - all samples plotted in one PCA space")
    print(f"  {combined_dir}/step7_{prefix}_cluster.txt - final cluster assignment for all samples")
    print(f"  {combined_dir}/step7_{prefix}_model.joblib - combined joblib")
    print(f"{'='*60}")

    # stage3: 新品種の階層的predict用に、既存の学習済みモデルをそのまま1つのjoblibへ格納
    run_stage3_unified_model(args, script_dir, outdir, cluster_ids)


def run_stage3_unified_model(args, script_dir: Path, outdir: Path, cluster_ids: list) -> None:
    """
    stage3: stage0/stage1で個別に学習済みのモデルを、stage2が作った
    combined/step7_{prefix}_model.joblib に追記してまとめる（別ファイルは作らない）。

    重要: ここでは一切の再学習（fit）を行わない。既に学習済みの
    KMeansオブジェクトをPythonの辞書に詰めて joblib.dump するだけの
    パッケージング処理であり、各モデルの学習結果・predict()の挙動は
    一切変化しない。

    combined/step7_{prefix}_model.joblib には既に stage2 が
    "model"/"samples"/"labels_top"/"cluster_labels"（step2 --hclust-model 用）
    を書き込んでおり、--haplotype-tsv 指定時は 10_haplotype_match.py が
    "haplotype_info"（結合PCA空間での照合結果）も追記済み。
    ここではそれを壊さず、階層的predict用のキーだけを追加で書き込む。
    既に "haplotype_info" がある場合、stage1側の集約結果は
    "stage1_haplotype_info" という別キーに保存し、上書きしない。

    新品種を分類する場合の使い方（階層的predict）:
      1) 新品種のイントロン領域特徴量を stage0_feature_cols の列順に揃え、
         stage0_model.predict() で一次クラスタ番号（0始まり）を得る
      2) (0始まり結果 + 1) が一次クラスタ番号。対応する
         stage1_models[一次クラスタ番号] を選択する
      3) 新品種の遺伝子領域全体特徴量を stage1_feature_cols[一次クラスタ番号]
         の列順に揃え、選択したモデルで .predict() する
      4) 得られたローカルラベル（0始まり）を
         stage1_label_map[一次クラスタ番号] で最終的なフラット番号（例: 4）に変換する
         （フラット番号はstage2が書き込んだ compound_label_map の割り当てをそのまま使う）

    stage0/stage1いずれかのjoblibが見つからない場合は、その旨を警告して
    可能な範囲だけをまとめる（致命的エラーにはしない）。
    """
    import joblib as _jl
    import re as _re

    combined_model_path = outdir / "combined" / f"step7_{args.prefix}_model.joblib"
    if not combined_model_path.exists():
        print(f"[WARN] stage3: {combined_model_path} not found, skipping appending hierarchical-predict keys.",
              file=sys.stderr)
        return

    stage0_model_path = outdir / "stage0_intron" / f"step7_{args.prefix}_intron_model.joblib"
    if not stage0_model_path.exists():
        # stage0がシングルエキソン遺伝子/GFF3に見つからない等の理由で
        # run_stage0_intron_clustering() のフォールバック（全品種を一次クラスタ1つに
        # 強制し、モデル学習自体をスキップ）を通った場合にここへ来る。この場合
        # 一次クラスタは cluster_ids == [1] の1つだけで、その中のstage1モデルは
        # 実質的に全品種を対象にした（二段階でない）通常のクラスタリングと等価。
        # 階層predict用のキーは追記できないが、そのstage1モデルをそのまま
        # bundle["model"]（フラットなKMeans）として使えば、通常predictで
        # 新品種を分類できるため、method="two_stage_hierarchical"以外の値に
        # しておいて 4_cluster.py の通常predict分岐に処理させる。
        if len(cluster_ids) == 1:
            cid = cluster_ids[0]
            stage1_model_path = outdir / f"stage1_cluster{cid}" / f"step7_{args.prefix}_c{cid}_model.joblib"
            if not stage1_model_path.exists():
                print(f"[WARN] stage3: {stage1_model_path} not found, skipping appending hierarchical-predict keys.",
                      file=sys.stderr)
                return
            stage1_bundle = _jl.load(stage1_model_path)
            stage1_model = stage1_bundle.get("model")
            if stage1_model is None:
                print(f"[WARN] stage3: {stage1_model_path} has no model, skipping appending hierarchical-predict keys.",
                      file=sys.stderr)
                return
            bundle = _jl.load(combined_model_path)
            bundle["model"] = stage1_model
            bundle["feature_cols"] = stage1_bundle.get("feature_cols")
            bundle["method"] = "kmeans"
            bundle["note"] = bundle.get("note", "") + (
                "\n[Automatic fallback to a single primary cluster]\n"
                f"Gene '{args.prefix}' could not be clustered at stage0 (intron region) -- "
                "possibly a single-exon gene, or the gene could not be found in the GFF3 -- "
                "so all samples were treated as one primary cluster, and the single model "
                "trained at stage1 (whole gene region) is used directly as this bundle's "
                "model (plain predict, not hierarchical predict)."
            )
            _jl.dump(bundle, combined_model_path)
            print(f"\n[OK] stage0 fell back to a single cluster, so stage1's single model "
                  f"was stored as a flat model in {combined_model_path} as-is "
                  f"(new samples can be classified via plain predict, not hierarchical predict).")
            return
        print(f"[WARN] stage3: {stage0_model_path} not found, skipping appending hierarchical-predict keys.",
              file=sys.stderr)
        return

    stage0_bundle = _jl.load(stage0_model_path)

    # stage2が書き込んだ「{cid}-{local}」->フラット番号の対応表を再利用する
    # （ここで新たに割り当て直すと、stage2のcluster.txt/labels_topと番号がずれるため）
    bundle = _jl.load(combined_model_path)
    compound_label_map: dict = bundle.get("compound_label_map", {})

    stage1_models: dict = {}
    stage1_feature_cols: dict = {}
    # cluster_id -> {ローカルラベル(0始まりint): フラット番号(int)}
    stage1_label_map: dict = {}

    # 各stage1モデルの haplotype_info（ローカルのクラスタ番号）を、
    # フラット番号に変換しながら1つに統合する
    combined_hap_info: dict = {}  # hap_name -> {"variants":..., "matched_samples":set, "clusters":set}

    for cid in cluster_ids:
        stage1_model_path = outdir / f"stage1_cluster{cid}" / f"step7_{args.prefix}_c{cid}_model.joblib"
        if not stage1_model_path.exists():
            print(f"[WARN] stage3: {stage1_model_path} not found, skipping this primary cluster.",
                  file=sys.stderr)
            continue

        stage1_bundle = _jl.load(stage1_model_path)
        stage1_models[cid] = stage1_bundle.get("model")
        stage1_feature_cols[cid] = stage1_bundle.get("feature_cols")

        labels_top = stage1_bundle.get("labels_top", [])          # このモデル内でのローカル番号（1始まり）
        cluster_labels = stage1_bundle.get("cluster_labels", [])  # このモデル内のローカル文字列（例: "cluster 2"）
        # cluster_labels は一次クラスタ cid の中でのローカル番号（7_kmeans.py の
        # --start_number デフォルト1のまま生成されており、cid をまたいだ一意性は無い）。
        # compound_label_map で cid 付きの通し番号（フラット番号）に変換する。
        label_map = {}
        for lt, cl in zip(labels_top, cluster_labels):
            m_local = _re.search(r"(\d+)", cl)
            local_num = m_local.group(1) if m_local else cl
            compound = f"{cid}-{local_num}"
            flat = compound_label_map.get(compound)
            if flat is None:
                print(f"[WARN] stage3: {compound} not found in compound_label_map, skipping.",
                      file=sys.stderr)
                continue
            label_map[int(lt) - 1] = flat  # KMeans.labels_ は0始まりのため -1 して揃える
        stage1_label_map[cid] = label_map

        # この一次クラスタの haplotype_info（10_haplotype_match.py が追記したもの）を統合
        hap_info_local = stage1_bundle.get("haplotype_info", {})
        for hap_name, info in hap_info_local.items():
            entry = combined_hap_info.setdefault(hap_name, {
                "variants": info.get("variants", []),
                "matched_samples": set(),
                "clusters": set(),
            })
            entry["matched_samples"].update(info.get("matched_samples", []))
            for local_cluster in info.get("clusters", []):
                flat = label_map.get(int(local_cluster) - 1)
                if flat is None:
                    continue
                entry["clusters"].add(flat)

    if not stage1_models:
        print("[WARN] stage3: no valid stage1 models found, not creating the combined joblib.",
              file=sys.stderr)
        return

    # set → ソート済みlistに変換（フラット番号はint）
    haplotype_info = {
        hap_name: {
            "variants": entry["variants"],
            "matched_samples": sorted(entry["matched_samples"]),
            "clusters": sorted(entry["clusters"]),
        }
        for hap_name, entry in combined_hap_info.items()
    }

    bundle["stage0_model"] = stage0_bundle.get("model")
    bundle["stage0_feature_cols"] = stage0_bundle.get("feature_cols")
    bundle["stage0_gene_region"] = stage0_bundle.get("gene_region")
    bundle.setdefault("gene_region", stage0_bundle.get("gene_region"))
    bundle["stage1_models"] = stage1_models
    bundle["stage1_feature_cols"] = stage1_feature_cols
    bundle["stage1_label_map"] = stage1_label_map

    # "haplotype_info" は既に 10_haplotype_match.py が結合PCA空間での照合結果を
    # 書き込んでいる可能性があるため上書きしない。無ければそのまま採用する。
    if "haplotype_info" in bundle:
        bundle["stage1_haplotype_info"] = haplotype_info
    else:
        bundle["haplotype_info"] = haplotype_info

    # sc/step3/1_v2/4_cluster.py が exact match でこの文字列をチェックするため、
    # 過去の値に追記するのではなく明示的にこの文字列で上書きする。
    bundle["method"] = "two_stage_hierarchical"
    bundle["note"] = bundle.get("note", "") + (
        "\n[Appended for hierarchical predict]\n"
        "Steps to classify a new sample:\n"
        "  1) Align intron-region features to stage0_feature_cols' column order, then stage0_model.predict()\n"
        "     -> result (0-based) + 1 is the primary cluster number\n"
        "  2) Select the corresponding stage1_models[primary cluster number]\n"
        "  3) Align whole-gene-region features to stage1_feature_cols[primary cluster number]'s column order, then .predict()\n"
        "  4) Convert the result via stage1_label_map[primary cluster number] into the final "
        "flat number (a 1-based integer)\n"
        "The stage0/stage1 models are stored as already-trained -- no retraining happened when appending this."
    )

    _jl.dump(bundle, combined_model_path)
    print(f"\n[OK] Appended hierarchical-predict keys to {combined_model_path} "
          f"(stored 1 stage0 model + {len(stage1_models)} stage1 model(s) as-is. No retraining. Merged into one file.)")


def main():
    parser = argparse.ArgumentParser(
        description="step1 全ステップ一括実行（BAM 抽出 → VCF → クラスタリング → PCA）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- 必須 ---
    parser.add_argument("--outdir",  required=True, help="出力ディレクトリ")
    parser.add_argument("--prefix",  required=True, help="出力ファイル名のプレフィックス（遺伝子名など）")

    # --- 入力 BAM ---
    bam_group = parser.add_mutually_exclusive_group()
    bam_group.add_argument("--bam-dir", help="BAM ファイルが入ったフォルダ（step1 必須）")
    bam_group.add_argument("--bams", nargs="+", help="BAM ファイルを直接指定（複数可）")

    # --- 参照・領域 ---
    parser.add_argument("--ref",     help="参照 FASTA ファイル（step2 必須）")
    parser.add_argument("--regions", nargs="+",
                        help="遺伝子領域 (例: chr06:9336376-9338569)。step1 と mode=all の step3 に必須")

    # --- VCF フィルタリングモード ---
    parser.add_argument("--vcf-mode", choices=["all", "exon", "intron"], default="all",
                        help="all: 遺伝子領域全体の多型（デフォルト） / exon: GFF3 エキソン領域のみ / "
                             "intron: GFF3 イントロン領域のみ")
    parser.add_argument("--gff",  help="GFF3 ファイル（--vcf-mode exon/intron、または --two-stage で必須）")
    parser.add_argument("--gene", help="対象遺伝子の ID / シンボル（--vcf-mode exon/intron、または --two-stage で必須）")

    # --- VCF 変換オプション ---
    parser.add_argument("--threads", type=int, default=8,
                        help="VCF 生成のスレッド数（デフォルト: 8）")
    parser.add_argument("--min-dp", type=int, default=10,
                        help="1/1 ホモの最低デプス閾値（デフォルト: 10）")

    # --- 外部ツールのパス ---
    # VCF生成（step2）を実際に実行する時だけ必要（run_step2内でチェックする）。
    # --start-step で step2 をスキップする場合は未指定でもよい。
    parser.add_argument("--bcftools", default=None,
                        help="bcftools 実行パス（2_vcf.py の VCF 生成で使用。ご環境のパスを指定してください）")

    # --- クラスタリングオプション ---
    parser.add_argument("--max-k",     type=int, default=10,
                        help="エルボー法の最大クラスタ数（デフォルト: 10）")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="クラスタ数を手動指定（省略時はエルボー法で自動決定）")

    # --- PCA プロットオプション ---
    parser.add_argument("--pca-components", type=int, default=2,
                        help="PCA の主成分数（デフォルト: 2）")
    parser.add_argument("--plot-pc", nargs="+", type=int, default=[1, 2],
                        help="プロットする PC のペア（デフォルト: 1 2）")

    # --- ハプロタイプ照合（step10）---
    parser.add_argument("--haplotype-tsv", default=None,
                        help="ハプロタイプ定義 TSV（cds.tsv 形式）。指定時のみ step10 を実行"
                             "（--two-stage 時は各stage1サブクラスタでも実行される）")
    parser.add_argument("--case-map", default=None,
                        help="2列TSV（大文字化サンプル名<TAB>元の表記）。指定すると step9/step10 の "
                             "PCAプロット点ラベルを元の表記で描画する（9_plot.py/10_haplotype_match.py "
                             "にそのまま渡す）。")

    # --- 二段階クラスタリング ---
    parser.add_argument("--two-stage", action="store_true",
                        help="二段階クラスタリングを実行する: "
                             "①イントロン領域SNPのみで一次クラスタリング → "
                             "②各一次クラスタ内で遺伝子領域全体のSNPを用いて二次クラスタリング。"
                             "指定時は --gff / --gene / --regions が必須（--vcf-mode は無視される）")
    parser.add_argument("--n-clusters-intron", type=int, default=None,
                        help="stage0（イントロン領域）のクラスタ数を手動指定（省略時はエルボー法で自動決定）")
    parser.add_argument("--max-k-intron", type=int, default=None,
                        help="stage0（イントロン領域）のエルボー法の最大クラスタ数（省略時は --max-k と同じ値）")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="stage1（二次クラスタリング）を実行する一次クラスタの最小サンプル数。"
                             "これ未満の一次クラスタは二次クラスタリングをスキップする（デフォルト: 3）")

    # --- 実行制御 ---
    parser.add_argument("--start-step", type=int, default=1, choices=STEP_SEQUENCE,
                        help="開始ステップ（デフォルト: 1）。--two-stage 指定時は無視される")
    parser.add_argument("--end-step",   type=int, default=9, choices=STEP_SEQUENCE,
                        help="終了ステップ（デフォルト: 9）。step10 を実行するには 10 を指定。"
                             "--two-stage 指定時は無視される")
    parser.add_argument("--dry-run", action="store_true",
                        help="コマンドを表示するだけで実行しない")

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 実行するステップ一覧
    steps = [s for s in STEP_SEQUENCE if args.start_step <= s <= args.end_step]

    print(f"[INFO] prefix   : {args.prefix}")
    print(f"[INFO] outdir   : {outdir}")
    print(f"[INFO] vcf-mode : {args.vcf_mode}")
    if args.vcf_mode in ("exon", "intron"):
        print(f"[INFO] gff      : {args.gff}")
        print(f"[INFO] gene     : {args.gene}")
    if args.two_stage:
        print(f"[INFO] two-stage: enabled (intron primary cluster -> per-cluster whole-gene secondary cluster)")
    print(f"[INFO] steps    : {steps}")
    if args.dry_run:
        print("[INFO] --dry-run: only prints the commands, does not execute them")

    if args.two_stage:
        # step1/2（BAM抽出・VCF生成）のみ通常通り実行し、
        # step3以降は run_two_stage_pipeline() が stage0/stage1 として担う。
        pre_steps = [s for s in (1, 2) if args.start_step <= s <= args.end_step]
        for step in pre_steps:
            print(f"\n{'='*60}")
            print(f"  STEP {step}: {SCRIPT_NAMES[step]}")
            print(f"{'='*60}")
            if step == 1:
                run_step1(args, script_dir, outdir)
            else:
                run_step2(args, script_dir, outdir)

        print(f"\n{'='*60}")
        print(f"  Starting two-stage clustering")
        print(f"{'='*60}")
        run_two_stage_pipeline(args, script_dir, outdir)
        return

    elbow_k = None

    step_funcs = {
        1:  lambda: run_step1(args, script_dir, outdir),
        2:  lambda: run_step2(args, script_dir, outdir),
        3:  lambda: run_step3(args, script_dir, outdir),
        4:  lambda: run_step4(args, script_dir, outdir),
        5:  lambda: run_step5(args, script_dir, outdir),
        6:  lambda: run_step6(args, script_dir, outdir),
        7:  lambda: run_step7(args, script_dir, outdir, elbow_k),
        9:  lambda: run_step9(args, script_dir, outdir),
        10: lambda: run_step10(args, script_dir, outdir),
    }

    for step in steps:
        print(f"\n{'='*60}")
        print(f"  STEP {step}: {SCRIPT_NAMES[step]}")
        print(f"{'='*60}")

        if step == 6:
            elbow_k = run_step6(args, script_dir, outdir)
        elif step == 7:
            run_step7(args, script_dir, outdir, elbow_k)
        else:
            step_funcs[step]()

    print(f"\n{'='*60}")
    print(f"  Done: all steps completed")
    print(f"  Output: {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3/1 パイプライン一括実行スクリプト

実行順:
  1. 1_vcf_call.py    : BAM + 座標TSV + 参照FASTA → bcftoolsでVCF作成（targets座標のみ）
  2. 2_vcf_to_calls.py: VCF → SNP/INS/DEL/NONE/RATIO/N に変換
  3. 3_snp_number.py  : バイナリ/実測比率特徴量に変換（ATGC 4ビット / INDEL 3ビット）
  4. 4_cluster.py     : step1 KMeansモデルで新サンプルをクラスタに割当
  5. 6_pcapkl.py      : 学習データの特徴量TSVからPCAモデルを作成
  7. 7_newplot.py     : 新サンプルを既存PCA空間に重ねてプロット

Step1/Step2は学習側（sc/step1/step1/2_vcf.py, 4_hetero_N_one_vcf_tsv.py）と
同じbcftoolsベースの方式で新サンプルのジェノタイプを決定する
（旧: pysamで直接pileupして多数決 → 現在は撤去）。そのため --ref は必須。

使用例:
  python3 run_pipeline.py \
    --bam sample1.bam sample2.bam \
    --targets targets.tsv \
    --model kmeans_model.joblib \
    --old-input training_features.tsv \
    --old-cluster cluster_result.txt \
    -o output_dir
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def _has_haplotype_info(model_path: str) -> bool:
    """step1 joblib に haplotype_info が存在するか確認する。"""
    try:
        import joblib as _jl
        bundle = _jl.load(model_path)
        if isinstance(bundle, dict):
            return bool(bundle.get("haplotype_info"))
    except Exception:
        pass
    return False


def run_step10_if_needed(args):
    """
    step1 joblib に haplotype_info がない場合に限り、
    step1/10_haplotype_match.py を実行して haplotype_info を追記する。
    --haplotype-tsv が未指定の場合はスキップ。
    """
    if not getattr(args, "haplotype_tsv", None):
        return

    if _has_haplotype_info(args.model):
        print("[INFO] Skipping step10 because joblib already has haplotype_info")
        return

    step10_script = SCRIPT_DIR.parent.parent / "step1" / "10_haplotype_match.py"
    if not step10_script.exists():
        print(f"[WARN] 10_haplotype_match.py not found: {step10_script}", file=sys.stderr)
        print("[WARN] Skipping haplotype matching", file=sys.stderr)
        return

    step1_outdir = Path(args.model).parent
    vcf_path     = step1_outdir / "step3_selected.vcf"

    cmd = [sys.executable, step10_script,
           "--outdir",        str(step1_outdir),
           "--prefix",        args.prefix,
           "--haplotype-tsv", args.haplotype_tsv]
    if args.gff:
        cmd += ["--gff", args.gff]
    if vcf_path.exists():
        cmd += ["--vcf", str(vcf_path)]

    run_step(cmd,
             "10_haplotype_match.py : haplotype matching -> append haplotype_info to step1 joblib")


def load_gene_region_from_joblib(model_path: str):
    """step1 joblib に保存された gene_region を返す。失敗時は None。"""
    if not model_path:
        return None
    try:
        import joblib as _joblib
        bundle = _joblib.load(model_path)
        if isinstance(bundle, dict):
            return bundle.get("gene_region")
    except Exception:
        pass
    return None


def run_step(cmd, desc):
    """1ステップを実行し、失敗したら終了する。"""
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"{'='*60}")
    print("CMD:", " ".join(str(x) for x in cmd), "\n")
    proc = subprocess.Popen([str(x) for x in cmd])
    returncode = proc.wait()
    if returncode != 0:
        print(f"[ERROR] Failed: {desc}  (exit={returncode})", file=sys.stderr)
        sys.exit(returncode)


def run_piped_step(cmds: list, desc: str):
    """パイプで繋いだ複数コマンドを実行し、失敗したら終了する。"""
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"{'='*60}")
    cmd_strs = [" ".join(str(x) for x in c) for c in cmds]
    print("CMD:", " | ".join(cmd_strs), "\n")

    procs = []
    prev_stdout = None
    for i, cmd in enumerate(cmds):
        is_last = (i == len(cmds) - 1)
        stdin  = prev_stdout
        stdout = None if is_last else subprocess.PIPE
        p = subprocess.Popen([str(c) for c in cmd], stdin=stdin, stdout=stdout)
        if prev_stdout is not None:
            prev_stdout.close()
        prev_stdout = p.stdout if not is_last else None
        procs.append(p)

    rets = [p.wait() for p in procs]
    failed = [i for i, r in enumerate(rets) if r != 0]
    if failed:
        ret_str = ", ".join(f"exit{i+1}={rets[i]}" for i in failed)
        print(f"[ERROR] Failed: {desc}  ({ret_str})", file=sys.stderr)
        sys.exit(max(rets))


def expand_bam_list(bam_inputs: list[str]) -> list[str]:
    """--bam に渡されたパスのうちディレクトリのものを .bam ファイル一覧に展開する"""
    result = []
    for inp in bam_inputs:
        p = Path(inp)
        if p.is_dir():
            # macOS writes AppleDouble sidecar files (e.g. "._sample.bam") on
            # exFAT/FAT32 external drives / network shares -- they match "*.bam"
            # but aren't real BAMs and make pysam fail with "Exec format error".
            found = sorted(str(b) for b in p.glob("*.bam") if not b.name.startswith("._"))
            if not found:
                print(f"[WARN] No .bam files found in directory: {inp}",
                      file=sys.stderr)
            result.extend(found)
        else:
            result.append(str(inp))
    return result


def main():
    ap = argparse.ArgumentParser(
        description="step3/1 パイプライン一括実行（BAM → クラスタ割当 → PCAプロット）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- 必須引数 ---
    ap.add_argument("--bam", nargs="+", required=True,
                    help="新サンプルのBAMファイル（複数可。.bai インデックス必須）")
    ap.add_argument("--targets", required=True,
                    help="対象座標TSV（chr / posi 列を含む。step1で作成したVCF由来TSVなど）")
    ap.add_argument("--model", required=True,
                    help="step1 7_kmeans.py で作成したKMeansモデル (.joblib)")
    ap.add_argument("--old-input", required=True,
                    help="step1 の学習サンプルの特徴量TSV（step5_{prefix}.tsv）")
    ap.add_argument("--old-cluster", required=True,
                    help="step1 7_kmeans.py で出力したクラスタTXT (cluster_result.txt)")
    ap.add_argument("-o", "--outdir", required=True,
                    help="出力ディレクトリ（存在しなければ作成）")

    # --- 参照ゲノム（必須：ジェノタイプ判定の主経路がbcftools VCF方式になったため）---
    ap.add_argument("--ref", required=True,
                    help="参照ゲノムFASTA（.faiが必要）。Step1/Step2のVCF作成と、"
                         "新規多型クラスタ追加（--no-novel-clusterで無効化可）の両方に使用")
    ap.add_argument("--region", default=None,
                    help="新規多型クラスタ検出用VCF作成対象の遺伝子領域"
                         "（例: chr01:38000000-38100000）。省略時は --model の"
                         "KMeansモデルに保存された gene_region を使用する。"
                         "どちらも無い場合は新規多型クラスタ検出をスキップする")
    ap.add_argument("--no-novel-cluster", action="store_true",
                    help="新規多型クラスタ追加ステップをスキップする")
    ap.add_argument("--bcftools", default="bcftools",
                    help="bcftools の実行パス（default: bcftools。"
                         "PATH に含まれない場合は絶対パスを指定）")

    # --- オプション ---
    ap.add_argument("--prefix", default="new",
                    help="中間ファイル・出力ファイルのプレフィックス（default: new）")
    ap.add_argument("--min-mapq", type=int, default=0,
                    help="最小MAPQ（新規多型クラスタ用mpileup用、default: 0。"
                         "step1 2_vcf.py の bcftools mpileup（MQ/BQフィルタなし）と一致）")
    ap.add_argument("--min-bq", type=int, default=1,
                    help="最小ベースクオリティ（新規多型クラスタ用mpileup用、default: 1。"
                         "step1 2_vcf.py の bcftools mpileup（MQ/BQフィルタなし）と一致）")
    ap.add_argument("--min-dp", type=int, default=10,
                    help="VCF作成時のDPフィルタ（1/1かつDP<この値を./. に置換、default: 10。"
                         "step1 2_vcf.py の bcftools filter と一致。Step1/新規多型クラスタ両方に使用）")
    ap.add_argument("--binom-alpha", type=float, default=0.01,
                    help="GT再判定に使う二項検定の有意水準（default: 0.01。"
                         "step1 2_vcf.py の reclassify_gt_by_binomial_test と一致。Step1で使用）")
    ap.add_argument("--n-fill", choices=["mean", "zero"], default="zero",
                    help="RATIOトークンも取れない真の欠損Nの補完方法（3_snp_number.py 用、"
                         "default: zero。学習側5_snp_number.pyの既定と一致）")
    ap.add_argument("--n-components", type=int, default=2,
                    help="PCA主成分数（6_pcapkl.py 用、default: 2）")
    ap.add_argument("--plot-pc", nargs="+", type=int, default=[1, 2],
                    help="プロットするPC番号をペアで指定（default: 1 2）")
    ap.add_argument("--pca-model", default=None,
                    help="step1 9_plot.py が保存したPCAモデル（.joblib）。"
                         "指定時は 6_pcapkl.py をスキップして直接使用する（プロット空間を完全一致させる場合に指定）")
    ap.add_argument("--haplotype-tsv", default=None,
                    help="ハプロタイプ定義 TSV（cds.tsv 形式）。"
                         "joblib に haplotype_info がない場合に step1/10_haplotype_match.py を実行して追記する")
    ap.add_argument("--gff", default=None,
                    help="GFF3 ファイル（--haplotype-tsv の CDS/AA 座標変換に使用）")
    ap.add_argument("--case-map", default=None,
                    help="2列TSV（UPPER(sample)<TAB>元の表記）。指定時は7_newplot.pyの"
                         "新サンプルラベル表示を、sc内部処理で大文字化される前の元の"
                         "BAM名の表記に戻す（determining_gui.pyがバッチ単位で1回だけ生成）。")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    px = args.prefix

    # --bam に渡されたディレクトリを実ファイル一覧に展開しておく（1_vcf_call.py /
    # 2_vcf_to_calls.py の両方で、bcftoolsに渡す順序と同じ順序で必要になる）。
    bam_files = expand_bam_list(args.bam)
    if not bam_files:
        print("[ERROR] No BAM files found to process", file=sys.stderr)
        sys.exit(1)

    # 中間ファイルパス
    f_match_vcf = outdir / f"{px}_match.vcf"
    f_simple   = outdir / f"{px}_simplified.tsv"
    f_features = outdir / f"{px}_features.tsv"
    f_cluster  = outdir / f"{px}_cluster.txt"
    f_vcf      = outdir / f"{px}_novel.vcf"
    f_pca      = outdir / f"{px}_pca_model.joblib"
    f_plot     = outdir / f"{px}_plot.png"

    # ------------------------------------------------------------------
    # Step 1: BAM + targets座標 + 参照FASTA → bcftoolsでVCF作成
    # ------------------------------------------------------------------
    run_step(
        [sys.executable, SCRIPT_DIR / "1_vcf_call.py",
         "--targets", args.targets,
         "--bam"] + bam_files + [
         "--ref", args.ref,
         "--output", f_match_vcf,
         "--bcftools", args.bcftools,
         "--min-dp", str(args.min_dp),
         "--binom-alpha", str(args.binom_alpha)],
        "1_vcf_call.py : BAM + targets positions + reference FASTA -> create VCF"
    )

    # ------------------------------------------------------------------
    # Step 2: VCF → SNP/INS/DEL/NONE/RATIO/N に変換
    # ------------------------------------------------------------------
    run_step(
        [sys.executable, SCRIPT_DIR / "2_vcf_to_calls.py",
         "--vcf",       f_match_vcf,
         "--targets",   args.targets,
         "--old-input", args.old_input,
         "--bam"] + bam_files + [
         "--output",    f_simple],
        "2_vcf_to_calls.py : VCF -> simplified TSV"
    )

    # ------------------------------------------------------------------
    # Step 3: バイナリ特徴量変換
    # ------------------------------------------------------------------
    run_step(
        [sys.executable, SCRIPT_DIR / "3_snp_number.py",
         "-i", f_simple,
         "-o", f_features,
         "--n-fill", args.n_fill],
        "3_snp_number.py : binary feature conversion (ATGC 4bit / INDEL 3bit)"
    )

    # ------------------------------------------------------------------
    # Step 3.5: haplotype_info が joblib にない場合は step10 を実行して追記
    # --haplotype-tsv 指定時のみ動作する
    # ------------------------------------------------------------------
    run_step10_if_needed(args)

    # ------------------------------------------------------------------
    # Step 4: step1 KMeansモデルでクラスタ割当 + ハプロタイプ自動判定
    # joblib に haplotype_info が保存されていれば混在クラスタ内を自動判定する
    # ------------------------------------------------------------------
    cmd4 = [sys.executable, SCRIPT_DIR / "4_cluster.py",
            "--input_tsv",      f_features,
            "--model_pkl",      args.model,
            "--output_txt",     f_cluster,
            "--simplified-tsv", f_simple]   # ← 2_vcf_to_calls.py の出力を渡す
    if args.gff:
        cmd4 += ["--gff", args.gff]
    run_step(cmd4, "4_cluster.py : cluster assignment (step1 KMeans model) + automatic haplotype matching")

    # ------------------------------------------------------------------
    # Step 4b: VCF作成（bcftools mpileup | bcftools call）
    # Step 4c: 新規多型クラスタ追加
    # --no-novel-cluster が指定されていない場合のみ実行（--ref は必須化されたため常に利用可能）
    # ------------------------------------------------------------------
    if not args.no_novel_cluster:
        if args.region:
            region = args.region
        else:
            region = load_gene_region_from_joblib(args.model)
            if region:
                print(f"[INFO] Got gene region from KMeans model: {region} ({args.model})")

        if not region:
            print("[INFO] Could not resolve a gene region, so skipping novel-cluster detection "
                  "(specify --region, or use a model that includes gene_region).")
        else:
            print(f"[INFO] VCF creation region: {region}")
            # bam_files はファイル冒頭で --bam から展開済み（Step1と同じ一覧を再利用）

            mpileup_cmd = [
                args.bcftools, "mpileup",
                "-f", args.ref,
                "-q", str(args.min_mapq),
                "-Q", str(args.min_bq),
                "-a", "AD,DP",
                "-Ou",  # uncompressed BCF（パイプ用）
                "-r", region,
            ]
            mpileup_cmd += bam_files

            call_cmd = [
                args.bcftools, "call",
                "-mv",
                "-Ou",  # uncompressed BCF（パイプ用）
            ]

            filter_cmd = [
                args.bcftools, "filter",
                "-e", f'GT="1/1" & FMT/DP<{args.min_dp}',
                "-S", ".",  # フィルタ条件を満たしたGTを./. に置換
                "-Ov",      # plain VCF
                "-o", str(f_vcf),
            ]

            run_piped_step(
                [mpileup_cmd, call_cmd, filter_cmd],
                "bcftools mpileup | call | filter : create novel-polymorphism VCF"
            )

            step4c_cmd = [
                sys.executable, SCRIPT_DIR / "4b_novel_cluster.py",
                "--cluster-txt", f_cluster,
                "--vcf",         f_vcf,
                "--targets",     args.targets,
                "--model",       args.model,   # haplotype_info 取得用
                "--output",      f_cluster,    # 同ファイルを上書き
                "--bcftools",    args.bcftools,
                "--bam",
            ] + bam_files
            run_step(step4c_cmd, "4b_novel_cluster.py : VCF haplotype matching / add novel-polymorphism cluster")

    # ------------------------------------------------------------------
    # Step 5: PCAモデル作成（外部モデル指定時はスキップ）
    # ------------------------------------------------------------------
    if args.pca_model and Path(args.pca_model).exists():
        f_pca = Path(args.pca_model)
        print(f"\n[INFO] --pca-model given, skipping 6_pcapkl.py: {f_pca}")
    else:
        run_step(
            [sys.executable, SCRIPT_DIR / "6_pcapkl.py",
             "--input_tsv",        args.old_input,
             "--output_pca_model", f_pca,
             "--n_components",     str(args.n_components)],
            "6_pcapkl.py : create PCA model (using training-data feature TSV)"
        )

    # ------------------------------------------------------------------
    # Step 7: 新サンプルを既存PCA空間にプロット
    # ------------------------------------------------------------------
    step7_cmd = [
        sys.executable, SCRIPT_DIR / "7_newplot.py",
        "--old_input",  args.old_input,
        "--new_input",  f_features,
        "--pca_model",  f_pca,
        "--cluster_txt",  args.old_cluster,
        "--new_cluster_txt", f_cluster,
        "--model",        args.model,
        "--output",       f_plot,
        "--plot_pc"] + [str(x) for x in args.plot_pc]
    if args.case_map:
        step7_cmd += ["--case-map", args.case_map]
    run_step(step7_cmd, "7_newplot.py : PCA plot (with new sample added)")

    # ------------------------------------------------------------------
    # 完了メッセージ
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[DONE] All steps completed successfully")
    print(f"  Cluster assignment : {f_cluster}")
    print(f"  Matching VCF        : {f_match_vcf}")
    if not args.no_novel_cluster:
        print(f"  Novel-polymorphism VCF : {f_vcf}")
    print(f"  PCA plot            : {f_plot}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3/2 パイプライン一括実行スクリプト

DEL と INS は特徴量系・KMeansモデルが独立しているため、
ステップ2以降を DEL と INS それぞれで別々に実行する。

【実行順】
  1a. 1_predict_combined.py     : BAM → DEL予測TSV（DELモデル + DEL座標TSV）
  1b. 1_predict_combined.py     : BAM → INS予測TSV（INSモデル + INS座標TSV）
  2a. 2_predict_tsv.py : DEL予測TSVを wide table に統合（--del-only）
  2b. 2_predict_tsv.py : INS予測TSVを wide table に統合（--ins-only）
  3a. 4_cluster.py     : DEL クラスタ割当（one-hot変換・KMeans予測・ハプロタイプ照合を内包）
  3b. 4_cluster.py     : INS クラスタ割当（one-hot変換・KMeans予測・ハプロタイプ照合を内包）
  [PCA が必要な場合のみ]
  4a. 3_number.py      : DEL one-hot変換（6_pcapkl / 7_newplot 用）
  4b. 3_number.py      : INS one-hot変換（6_pcapkl / 7_newplot 用）
  5a. 6_pcapkl.py      : DEL PCAモデル作成（学習データ特徴量TSV）
  5b. 6_pcapkl.py      : INS PCAモデル作成（学習データ特徴量TSV）
  6a. 7_newplot.py     : DEL PCAプロット（新サンプル追加）
  6b. 7_newplot.py     : INS PCAプロット（新サンプル追加）

PROB閾値フィルタ（旧 2_filter_predictions.py）は廃止。
1_predict_combined.py 自身の --threshold（emit_n判定）のみで信頼度を制御する。

【targets が空の場合】
  --targets-del / --targets-ins がヘッダーのみ（データ行0件）の場合、
  predict/merge をスキップし、--bam から抽出した全サンプルを
  「# cluster 1」として {prefix}_DEL_cluster.txt / {prefix}_INS_cluster.txt に出力する。

【途中再開】
  --skip-predict : 予測済みの場合にpredictをスキップし、マージから再開（--predictions-dir が必須）

使用例（全ステップ実行）:
  python3 run_pipeline.py \
    --bam sample1.bam sample2.bam \
    --targets-del del_targets.tsv \
    --targets-ins ins_targets.tsv \
    --model-del del_model.joblib \
    --model-ins ins_model.joblib \
    --kmeans-del del_kmeans.joblib \
    --old-input-del del_features.tsv \
    --old-clus-del del_cluster.txt \
    --kmeans-ins ins_kmeans.joblib \
    --old-input-ins ins_features.tsv \
    --old-clus-ins ins_cluster.txt \
    -o output_dir

使用例（予測済み・マージから再開）:
  python3 run_pipeline.py \
    --skip-predict \
    --predictions-dir output_dir/predictions \
    --kmeans-del del_kmeans.joblib \
    ... （その他必須引数） \
    -o output_dir
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent


# ============================================================
# KMeansモデルから gene_region を取得
# ============================================================

def load_gene_region_from_joblib(model_path: Optional[str]) -> Optional[str]:
    """KMeans joblib に保存された gene_region を返す。失敗時は None。"""
    if not model_path:
        return None
    try:
        import joblib
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            return bundle.get("gene_region")
    except Exception:
        pass
    return None


# ============================================================
# targets 空チェック・フォールバック出力
# ============================================================

def is_targets_tsv_empty(path: str) -> bool:
    """targets TSV にヘッダー以外のデータ行が1つも無いか判定する。"""
    p = Path(path)
    if not p.exists():
        return True
    with open(p) as f:
        f.readline()  # ヘッダーを読み飛ばす
        for line in f:
            if line.strip():
                return False
    return True


def _bam_sample_basename(bam_path: str) -> str:
    name = Path(bam_path).name
    for suf in (".rg.bam", ".sort.bam", ".bam"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def collect_sample_names_from_bam(bam_args: list) -> list:
    """--bam 引数（ディレクトリ or ファイル、複数可）からサンプル名一覧を抽出する。"""
    names = []
    for inp in bam_args:
        p = Path(inp)
        if p.is_dir():
            # skip macOS AppleDouble sidecar files (e.g. "._sample.bam") from
            # exFAT/FAT32 external drives / network shares -- not real BAMs.
            for b in sorted(p.glob("*.bam")):
                if b.name.startswith("._"):
                    continue
                names.append(_bam_sample_basename(str(b)).upper())
        elif p.is_file() and str(p).endswith(".bam"):
            names.append(_bam_sample_basename(str(p)).upper())
    return sorted(set(names))


def collect_sample_names_from_pred_dir(pred_dir) -> list:
    """
    予測TSVファイル名（{SAMPLE}_DEL.tsv / {SAMPLE}_INS.tsv 等）からサンプル名一覧を抽出する。
    --skip-predict 使用時（--bam が None）のフォールバックに使う。
    """
    names = []
    if pred_dir is None:
        return names
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        return names
    files = list(pred_dir.glob("*.tsv")) + list(pred_dir.glob("*.tsv.gz"))
    for f in sorted(files):
        name = f.name
        if name.endswith(".gz"):
            name = name[:-3]
        if name.endswith(".sort.tsv"):
            name = name[:-9]
        elif name.endswith(".tsv"):
            name = name[:-4]
        if name.lower().endswith("_predictions"):
            name = name[:-len("_predictions")]
        if name.endswith("_INS") or name.endswith("_DEL"):
            name = name[:-4]
        names.append(name.upper())
    return sorted(set(names))


def collect_sample_names(args, pred_dir) -> list:
    """--bam があればBAMから、無ければ予測TSVディレクトリから、サンプル名一覧を取得する。"""
    if args.bam:
        return collect_sample_names_from_bam(args.bam)
    return collect_sample_names_from_pred_dir(pred_dir)


def write_fallback_cluster_txt(samples: list, output_txt: Path, label: str) -> None:
    """targets が空のとき、全サンプルをクラスタ1として出力する。"""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("# cluster 1\n")
        for s in samples:
            f.write(f"{s}\n")
        f.write("\n")
    print(f"[OK] {label}: targets is empty, skipping predict/merge and "
          f"outputting all samples as cluster 1 -> {output_txt}")


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


def run_del_pca_plot(args, outdir, f_merged) -> bool:
    """
    DEL の PCA プロット生成（--old-input-del が指定された場合のみ実行）。
    f_merged が存在しない、または --old-input-del 未指定の場合は何もせず False を返す。
    f_merged の行数が0（新サンプル側でDEL予測が1つも無かった場合）でも、
    3_number.py/6_pcapkl.py/7_newplot.py はいずれも特徴量0列を検知して
    プレースホルダー画像を出す仕組みを持っているため、クラッシュせず実行できる
    （学習データ側にも特徴量が無ければ「PCA not available」画像に、学習データ側に
    特徴量があれば、新サンプルは原点扱いで既存のPCA空間に重ねて表示される）。
    """
    px = args.prefix
    f_pca     = outdir / f"{px}_DEL_pca_model.joblib"
    f_plot    = outdir / f"{px}_DEL_plot.png"
    f_cluster = outdir / f"{px}_DEL_cluster.txt"

    _del_pca_input = args.old_input_del or getattr(args, "old_dist_del", None)
    if not _del_pca_input:
        print("[INFO] Skipped DEL PCA (--old-input-del not specified)")
        return False
    if not Path(f_merged).exists():
        print(f"[INFO] Skipped DEL PCA ({f_merged} not found)")
        return False

    f_features = outdir / f"{px}_DEL_features.tsv"

    # 4a: one-hot encoding (for 7_newplot.py)
    run_step(
        [sys.executable, SCRIPT_DIR / "3_number.py",
         "-i",       f_merged,
         "-o",       f_features,
         "--mode",   "del",
         "--n-fill", args.n_fill],
        "3_number.py [DEL] : one-hot encoding (for PCA)"
    )
    # 5a: build PCA model (skipped if an external model is specified)
    _ext_pca_del = getattr(args, "pca_model_del", None)
    if _ext_pca_del and Path(_ext_pca_del).exists():
        f_pca = Path(_ext_pca_del)
        print(f"\n[INFO] Skipping 6_pcapkl.py [DEL] because --pca-model-del was specified: {f_pca}")
    else:
        run_step(
            [sys.executable, SCRIPT_DIR / "6_pcapkl.py",
             "--input_tsv",        _del_pca_input,
             "--output_pca_model", f_pca,
             "--n_components",     str(args.n_components)],
            "6_pcapkl.py [DEL] : build PCA model"
        )
    # 6a: PCA plot
    del_newplot_cmd = [sys.executable, SCRIPT_DIR / "7_newplot.py",
         "--old_input",   _del_pca_input,
         "--new_input",   f_features,
         "--pca_model",   f_pca,
         "--cluster_txt", args.old_clus_del,
         "--model",       args.kmeans_del,
         "--output",      f_plot,
         "--plot_pc"] + [str(x) for x in args.plot_pc]
    if f_cluster.exists():
        del_newplot_cmd += ["--new_cluster_txt", f_cluster]
    if getattr(args, "case_map", None):
        del_newplot_cmd += ["--case-map", args.case_map]
    run_step(del_newplot_cmd, "7_newplot.py [DEL] : PCA plot (add new sample)")
    print(f"  PCA plot  : {f_plot}")
    return True


def run_del_pipeline(args, outdir, pred_dir):
    """DEL サブパイプライン（マージ以降）を実行する。"""
    px = args.prefix

    f_merged  = outdir / f"{px}_DEL_merged.tsv"
    f_cluster = outdir / f"{px}_DEL_cluster.txt"

    # Step 3a: DEL クラスタ割当（one-hot変換・KMeans予測・ハプロタイプ照合を内包）
    step3a_cmd = [sys.executable, SCRIPT_DIR / "4_cluster.py",
                  "--input-tsv",  f_merged,
                  "--model-pkl",  args.kmeans_del,
                  "--output-txt", f_cluster,
                  "--mode",       "del"]
    if args.targets_del:
        step3a_cmd += ["--targets", args.targets_del]
    if getattr(args, "haplotype_tsv", None):
        step3a_cmd += ["--haplotype-tsv", args.haplotype_tsv]
    if getattr(args, "gff", None):
        step3a_cmd += ["--gff", args.gff]
    step3a_cmd += ["--gene-absent-file", str(outdir / "gene_absent_samples.txt")]
    run_step(step3a_cmd, "4_cluster.py [DEL] : cluster assignment (one-hot encoding, KMeans prediction, haplotype matching)")

    # Step 4a/5a/6a: PCA (only run if --old-input-del was specified)
    run_del_pca_plot(args, outdir, f_merged)

    print(f"\n[DEL complete]")
    print(f"  Cluster assignment : {f_cluster}")


def run_ins_pca_plot(args, outdir, f_merged) -> bool:
    """
    INS の PCA プロット生成（--old-input-ins が指定された場合のみ実行）。
    詳細は run_del_pca_plot() のdocstring参照（DEL/INSで同じ方針）。
    """
    px = args.prefix
    f_pca     = outdir / f"{px}_INS_pca_model.joblib"
    f_plot    = outdir / f"{px}_INS_plot.png"
    f_cluster = outdir / f"{px}_INS_cluster.txt"

    _ins_pca_input = args.old_input_ins or getattr(args, "old_dist_ins", None)
    if not _ins_pca_input:
        print("[INFO] Skipped INS PCA (--old-input-ins not specified)")
        return False
    if not Path(f_merged).exists():
        print(f"[INFO] Skipped INS PCA ({f_merged} not found)")
        return False

    f_features = outdir / f"{px}_INS_features.tsv"

    # 4b: one-hot encoding (for 7_newplot.py)
    run_step(
        [sys.executable, SCRIPT_DIR / "3_number.py",
         "-i",       f_merged,
         "-o",       f_features,
         "--mode",   "ins",
         "--n-fill", args.n_fill],
        "3_number.py [INS] : one-hot encoding (for PCA)"
    )
    # 5b: build PCA model (skipped if an external model is specified)
    _ext_pca_ins = getattr(args, "pca_model_ins", None)
    if _ext_pca_ins and Path(_ext_pca_ins).exists():
        f_pca = Path(_ext_pca_ins)
        print(f"\n[INFO] Skipping 6_pcapkl.py [INS] because --pca-model-ins was specified: {f_pca}")
    else:
        run_step(
            [sys.executable, SCRIPT_DIR / "6_pcapkl.py",
             "--input_tsv",        _ins_pca_input,
             "--output_pca_model", f_pca,
             "--n_components",     str(args.n_components)],
            "6_pcapkl.py [INS] : build PCA model"
        )
    # 6b: PCA plot
    ins_newplot_cmd = [sys.executable, SCRIPT_DIR / "7_newplot.py",
         "--old_input",   _ins_pca_input,
         "--new_input",   f_features,
         "--pca_model",   f_pca,
         "--cluster_txt", args.old_clus_ins,
         "--model",       args.kmeans_ins,
         "--output",      f_plot,
         "--plot_pc"] + [str(x) for x in args.plot_pc]
    if f_cluster.exists():
        ins_newplot_cmd += ["--new_cluster_txt", f_cluster]
    if getattr(args, "case_map", None):
        ins_newplot_cmd += ["--case-map", args.case_map]
    run_step(ins_newplot_cmd, "7_newplot.py [INS] : PCA plot (add new sample)")
    print(f"  PCA plot  : {f_plot}")
    return True


def run_ins_pipeline(args, outdir, pred_dir):
    """INS サブパイプライン（マージ以降）を実行する。"""
    px = args.prefix

    f_merged  = outdir / f"{px}_INS_merged.tsv"
    f_cluster = outdir / f"{px}_INS_cluster.txt"

    # Step 3b: INS クラスタ割当（one-hot変換・KMeans予測・ハプロタイプ照合を内包）
    step3b_cmd = [sys.executable, SCRIPT_DIR / "4_cluster.py",
                  "--input-tsv",  f_merged,
                  "--model-pkl",  args.kmeans_ins,
                  "--output-txt", f_cluster,
                  "--mode",       "ins"]
    if args.targets_ins:
        step3b_cmd += ["--targets", args.targets_ins]
    if getattr(args, "haplotype_tsv", None):
        step3b_cmd += ["--haplotype-tsv", args.haplotype_tsv]
    if getattr(args, "gff", None):
        step3b_cmd += ["--gff", args.gff]
    step3b_cmd += ["--gene-absent-file", str(outdir / "gene_absent_samples.txt")]
    run_step(step3b_cmd, "4_cluster.py [INS] : クラスタ割当（one-hot変換・KMeans予測・ハプロタイプ照合）")

    # Step 4b/5b/6b: PCA（--old-input-ins が指定された場合のみ実行）
    run_ins_pca_plot(args, outdir, f_merged)

    print(f"\n[INS done]")
    print(f"  Cluster assignment: {f_cluster}")


def main():
    ap = argparse.ArgumentParser(
        description="step3/2 パイプライン一括実行（DEL / INS それぞれ独立して実行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- 必須引数 ---
    ap.add_argument("-o", "--outdir", required=True,
                    help="出力ディレクトリ（存在しなければ作成）")

    # --- 途中再開 ---
    resume = ap.add_argument_group("途中再開オプション")
    resume.add_argument("--skip-predict", action="store_true",
                        help="予測をスキップしてマージから開始（--predictions-dir が必須）")
    resume.add_argument("--predictions-dir", default=None,
                        help="既存の予測TSVフォルダ（--skip-predict 時に使用）")

    # --- 予測用引数 ---
    ap.add_argument("--bam", nargs="+", default=None,
                    help="新サンプルのBAMファイルまたはBAMフォルダ（複数可・フォルダ指定時は *.bam を自動展開。予測スキップ時は不要）")

    del_group = ap.add_argument_group("DEL セット（5つ全てセットで指定する）")
    del_group.add_argument("--targets-del", default=None,
                           help="DEL学習ポジションの座標TSV（予測スキップ時は不要）")
    del_group.add_argument("--model-del", default=None,
                           help="step2で作成したDEL予測モデル (.joblib)（予測スキップ時は不要）")
    del_group.add_argument("--kmeans-del", default=None,
                           help="step2 7_kmeans.py で作成したDEL KMeansモデル (.joblib)")
    del_group.add_argument("--old-input-del", default=None,
                           help="step2 の DEL学習サンプルの特徴量TSV（step5_number/{prefix}_DEL_number.tsv）")
    del_group.add_argument("--old-clus-del", default=None,
                           help="step2 7_kmeans.py で出力したDELクラスタTXT")

    ins_group = ap.add_argument_group("INS セット（5つ全てセットで指定する）")
    ins_group.add_argument("--targets-ins", default=None,
                           help="INS学習ポジションの座標TSV（予測スキップ時は不要）")
    ins_group.add_argument("--model-ins", default=None,
                           help="step2で作成したINS予測モデル (.joblib)（予測スキップ時は不要）")
    ins_group.add_argument("--kmeans-ins", default=None,
                           help="step2 7_kmeans.py で作成したINS KMeansモデル (.joblib)")
    ins_group.add_argument("--old-input-ins", default=None,
                           help="step2 の INS学習サンプルの特徴量TSV（step5_number/{prefix}_INS_number.tsv）")
    ins_group.add_argument("--old-clus-ins", default=None,
                           help="step2 7_kmeans.py で出力したINSクラスタTXT")

    # --- その他オプション ---
    ap.add_argument("--prefix", default="new",
                    help="出力ファイルのプレフィックス（default: new）")
    ap.add_argument("--threshold", type=float, default=0.98,
                    help="予測確率閾値（1_predict_combined.py 用、default: 0.6）")
    ap.add_argument("--ins-mq", type=int, default=60,
                    help="INS特徴量計算のMQ閾値（MQ>=この値のリードのみ対象。"
                         "1_predict_combined.py にそのまま渡す。default: 60）")
    ap.add_argument("--n-fill", choices=["mean", "zero"], default="zero",
                    help="Nの補完方法（3_number.py 用、default: zero）")
    ap.add_argument("--n-components", type=int, default=2,
                    help="PCA主成分数（default: 2）")
    ap.add_argument("--plot-pc", nargs="+", type=int, default=[1, 2],
                    help="プロットするPC番号をペアで指定（default: 1 2）")
    ap.add_argument("--no-features", action="store_true",
                    help="特徴量TSVを出力しない（1_predict_combined.py 高速化）")
    ap.add_argument("--jobs", "-j", type=int, default=2,
                    help="並列処理数（1_predict_combined.py 用、default: 1）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="既に出力済みの予測ファイルをスキップ（1_predict_combined.py 用）")
    ap.add_argument("--region", default=None,
                    help="予測範囲の遺伝子領域（例: chr01:38382000-38390000）。"
                         "省略時は --kmeans-del / --kmeans-ins の joblib から自動取得する。"
                         "指定した場合は --targets-del / --targets-ins を使わず領域全体をスキャンする。")
    ap.add_argument("--step", type=int, default=1,
                    help="--region スキャン時のポジション間隔（default: 1）")
    ap.add_argument("--skip-extraction", action="store_true",
                    help="--bam に渡したBAMが既に遺伝子領域±10kbへ抽出済みであることを示し、"
                         "1_bam_choice.py による再抽出をスキップする（呼び出し元がSNP/INDEL両方の"
                         "ステップで共有する抽出済みBAMを用意している場合に指定。単独実行時は不要）")
    ap.add_argument("--mean-depth-tsv", default=None,
                    help="事前計算済みの平均デプス・ライブラリinsert size TSV"
                         "（estimate_mean_depth.py の出力）。指定時はこのファイルをそのまま使い、"
                         "自前でのestimate_mean_depth.py呼び出しをスキップする"
                         "（--skip-extraction と併用する想定。抽出前の全長BAMに対して計算された"
                         "ものであること）")
    ap.add_argument("--gff", default=None,
                    help="GFF3 ファイル（--haplotype-tsv の CDS/AA 座標を genomic に変換する場合に指定）")
    ap.add_argument("--haplotype-tsv", default=None,
                    help="ハプロタイプ定義TSV（gene_model_make/cds/ 形式）。"
                         "coord_type=genomic のエントリのみ使用。SNP はスキップ。"
                         "DEL・INS の両クラスタ割当に使用される。")

    ap.add_argument("--pca-model-del", default=None,
                    help="step2 9_plot.py が保存したDEL PCAモデル（.joblib）。"
                         "指定時は 6_pcapkl.py [DEL] をスキップして直接使用する")
    ap.add_argument("--pca-model-ins", default=None,
                    help="step2 9_plot.py が保存したINS PCAモデル（.joblib）。"
                         "指定時は 6_pcapkl.py [INS] をスキップして直接使用する")
    ap.add_argument("--case-map", default=None,
                    help="2列TSV（UPPER(sample)<TAB>元の表記）。指定時は7_newplot.pyの"
                         "新サンプルラベル表示を、sc内部処理で大文字化される前の元の"
                         "BAM名の表記に戻す（determining_gui.pyがバッチ単位で1回だけ生成）。")

    # 旧引数（廃止済み）: エラーにせず無視する
    ap.add_argument("--old-dist-del", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--old-dist-ins", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    if args.old_dist_del or args.old_dist_ins:
        print("[INFO] --old-dist-del / --old-dist-ins are deprecated and ignored.")

    # --bam にフォルダが指定された場合は *.bam を自動展開
    if args.bam:
        expanded = []
        for b in args.bam:
            p = Path(b)
            if p.is_dir():
                # macOS writes AppleDouble sidecar files (e.g. "._sample.bam") on
                # exFAT/FAT32 external drives / network shares -- they match "*.bam"
                # but aren't real BAMs and make pysam fail with "Exec format error".
                found = sorted(f for f in p.glob("*.bam") if not f.name.startswith("._"))
                if not found:
                    ap.error(f"No .bam files found in --bam directory '{b}'.")
                expanded.extend(str(f) for f in found)
            else:
                expanded.append(b)
        args.bam = expanded

    # --- バリデーション ---
    run_del = any(x is not None for x in [
        args.targets_del, args.model_del, args.kmeans_del,
        args.old_input_del, args.old_clus_del
    ])
    run_ins = any(x is not None for x in [
        args.targets_ins, args.model_ins, args.kmeans_ins,
        args.old_input_ins, args.old_clus_ins
    ])

    if not run_del and not run_ins:
        ap.error("Please specify the DEL set, the INS set, or both.")

    # Check the arguments needed from merge onward
    if run_del:
        missing = [n for n, v in [
            ("--kmeans-del",  args.kmeans_del),
            ("--old-clus-del", args.old_clus_del),
        ] if v is None]
        if missing:
            ap.error(f"The DEL set is incomplete. Missing: {' '.join(missing)}")
        if args.old_input_del is None:
            print("[INFO] --old-input-del not specified, skipping the PCA step (6_pcapkl / 7_newplot).")

    if run_ins:
        missing = [n for n, v in [
            ("--kmeans-ins",   args.kmeans_ins),
            ("--old-clus-ins", args.old_clus_ins),
        ] if v is None]
        if missing:
            ap.error(f"The INS set is incomplete. Missing: {' '.join(missing)}")
        if args.old_input_ins is None:
            print("[INFO] --old-input-ins not specified, skipping the PCA step (6_pcapkl / 7_newplot).")

    # --region: use if explicitly given, otherwise auto-detect from the KMeans joblib
    gene_region: Optional[str] = args.region
    if gene_region is None:
        for model_path in [args.kmeans_del, args.kmeans_ins]:
            r = load_gene_region_from_joblib(model_path)
            if r:
                gene_region = r
                print(f"[INFO] Got gene region from the KMeans model: {gene_region} ({model_path})")
                break

    # Argument checks for when prediction is skipped
    if args.skip_predict:
        if args.predictions_dir is None:
            ap.error("--predictions-dir is required when using --skip-predict.")
    else:
        if args.bam is None:
            ap.error("--bam is required when running prediction.")
        # --region or --targets is required (either one is fine)
        if run_del and args.model_del is None:
            ap.error("--model-del is required for DEL prediction.")
        if run_ins and args.model_ins is None:
            ap.error("--model-ins is required for INS prediction.")
        if gene_region is None:
            if run_del and args.targets_del is None:
                ap.error("--targets-del or --region (or gene_region in the joblib) is required for DEL prediction.")
            if run_ins and args.targets_ins is None:
                ap.error("--targets-ins or --region (or gene_region in the joblib) is required for INS prediction.")

    # --- ディレクトリ準備 ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_dir = outdir / "predictions"

    # --predictions-dir で外部指定された場合はそちらを使う
    if args.predictions_dir:
        pred_dir = Path(args.predictions_dir)

    # --- targets が空かどうかを事前判定（--region 使用時は targets 不使用なのでスキップ扱い） ---
    # targets が None または空TSV のいずれかで、かつ gene_region があれば領域スキャンを優先する
    _del_targets_is_empty = (args.targets_del is None or is_targets_tsv_empty(args.targets_del))
    _ins_targets_is_empty = (args.targets_ins is None or is_targets_tsv_empty(args.targets_ins))

    # gene_region が分かっているなら、targets の有無に関わらず常に領域全体を
    # スキャンする（旧仕様は targets が空の時だけのフォールバックだったが、
    # 新規多型（学習時のtargets外のDEL/INS位置）を検出できるようにするため、
    # 常時スキャンに変更した -- 4_cluster.py 側で targets 外の座位を検出し、
    # 未使用のクラスタ番号を割り当てる）。
    use_region_del = (run_del and gene_region is not None)
    use_region_ins = (run_ins and gene_region is not None)

    # gene_region もなく targets も空の場合のみクラスタ1フォールバック
    del_targets_empty = (
        run_del and not args.skip_predict and not use_region_del
        and _del_targets_is_empty
    )
    ins_targets_empty = (
        run_ins and not args.skip_predict and not use_region_ins
        and _ins_targets_is_empty
    )
    if del_targets_empty:
        print(f"\n[INFO] DEL: --targets-del is empty and gene_region is not set -> skipping predict/merge "
              f"and outputting all samples as cluster 1.")
    if ins_targets_empty:
        print(f"\n[INFO] INS: --targets-ins is empty and gene_region is not set -> skipping predict/merge "
              f"and outputting all samples as cluster 1.")
    if use_region_del:
        print(f"\n[INFO] DEL: scanning the whole region with --region {gene_region} "
              f"(new positions outside targets will be detected by 4_cluster.py).")
    if use_region_ins:
        print(f"\n[INFO] INS: scanning the whole region with --region {gene_region} "
              f"(new positions outside targets will be detected by 4_cluster.py).")

    # ============================================================
    # 領域全体スキャンを行う場合、Clustering Alleles と同じ方針で、先に
    # BAMを遺伝子領域±10kbへ抽出し（1_bam_choice.py）、平均デプス・ライブラリ
    # median insert size を「抽出前の」全長BAMから事前計算しておく
    # （estimate_mean_depth.py）。抽出済みBAMから直接これらを再計算すると
    # DEPTH_RATIO系の特徴量が壊れるため、この順序は必須。
    #
    # --skip-extraction / --mean-depth-tsv が指定されている場合（呼び出し元の
    # determining_gui.py が、SNP側のステップとも共有する抽出済みBAM・事前計算済み
    # 平均デプスを既に用意している場合）は、この抽出・事前計算を再度行わず、
    # 渡された値をそのまま使う。単独CLI実行時（どちらも未指定）は今まで通り
    # 自前で抽出・計算する。
    # ============================================================
    region_bam_files = args.bam
    mean_depth_tsv_path: Optional[str] = args.mean_depth_tsv
    if (use_region_del or use_region_ins) and not args.skip_predict and args.bam:
        if args.skip_extraction:
            region_bam_files = args.bam
            print("[INFO] --skip-extraction specified, using --bam as already-extracted BAMs as-is")
        else:
            region_bams_dir = outdir / "_region_bams"
            region_bams_dir.mkdir(parents=True, exist_ok=True)

            run_step(
                [sys.executable,
                 SCRIPT_DIR.parent.parent / "step1" / "step1" / "1_bam_choice.py",
                 "--input_bams"] + args.bam + [
                 "--regions", gene_region,
                 "--output_dir", str(region_bams_dir)],
                f"1_bam_choice.py : extract BAMs to gene region +/- 10kb ({gene_region})"
            )

            region_bam_files = [str(region_bams_dir / Path(b).name) for b in args.bam]

        if mean_depth_tsv_path is None:
            mean_depth_tsv_path = str(outdir / "mean_depth.tsv")
            run_step(
                [sys.executable,
                 SCRIPT_DIR.parent.parent / "step1" / "step2_newfea2" / "estimate_mean_depth.py",
                 "--bam"] + args.bam + ["-o", mean_depth_tsv_path],
                "estimate_mean_depth.py : estimate mean depth / library insert size (full-length BAM)"
            )
        else:
            print(f"[INFO] --mean-depth-tsv specified, using the precomputed value: {mean_depth_tsv_path}")

    # ============================================================
    # Step 1a: BAM → DEL 予測TSV
    # ============================================================
    if not args.skip_predict:
        pred_dir.mkdir(parents=True, exist_ok=True)

        if run_del and not del_targets_empty:
            del_bam_files = region_bam_files if use_region_del else args.bam
            cmd_del = [sys.executable, SCRIPT_DIR / "1_predict_combined.py",
                       "-b"] + del_bam_files + [
                       "-o",           outdir,
                       "--model-del",  args.model_del,
                       "--threshold",  str(args.threshold),
                       "--jobs",       str(args.jobs),
                       "--step",       str(args.step)]
            if use_region_del:
                cmd_del += ["--region", gene_region]
                if mean_depth_tsv_path:
                    cmd_del += ["--mean-depth-tsv", mean_depth_tsv_path]
            elif args.targets_del:
                cmd_del += ["--targets", args.targets_del]
            if args.no_features:
                cmd_del += ["--no-features"]
            if args.skip_existing:
                cmd_del += ["--skip-existing"]
            run_step(cmd_del, "1_predict_combined.py [DEL] : BAM → DEL予測TSV")

        # Step 1b: BAM → INS 予測TSV
        if run_ins and not ins_targets_empty:
            ins_bam_files = region_bam_files if use_region_ins else args.bam
            cmd_ins = [sys.executable, SCRIPT_DIR / "1_predict_combined.py",
                       "-b"] + ins_bam_files + [
                       "-o",           outdir,
                       "--model-ins",  args.model_ins,
                       "--threshold",  str(args.threshold),
                       "--ins-mq",     str(args.ins_mq),
                       "--jobs",       str(args.jobs),
                       "--step",       str(args.step)]
            if use_region_ins:
                cmd_ins += ["--region", gene_region]
                if mean_depth_tsv_path:
                    cmd_ins += ["--mean-depth-tsv", mean_depth_tsv_path]
            elif args.targets_ins:
                cmd_ins += ["--targets", args.targets_ins]
            if args.no_features:
                cmd_ins += ["--no-features"]
            if args.skip_existing:
                cmd_ins += ["--skip-existing"]
            run_step(cmd_ins, "1_predict_combined.py [INS] : BAM → INS予測TSV")

    # ============================================================
    # DEL / INS サブパイプライン（マージ以降）
    # predictions ディレクトリに実際にファイルが存在するか確認してから実行
    # ============================================================
    def has_tsv(directory: Path, suffix: str) -> bool:
        """directory 内に suffix を含む .tsv ファイルが存在するか確認。"""
        return directory.exists() and any(
            f.name.endswith(".tsv") and suffix in f.name
            for f in directory.glob("*.tsv")
        )

    # ============================================================
    # Step 2 [DEL+INS]: 予測TSV統合（DEL/INS を1回で同時処理）
    # ============================================================
    px = args.prefix
    f_merged_del = outdir / f"{px}_DEL_merged.tsv"
    f_merged_ins = outdir / f"{px}_INS_merged.tsv"

    do_del_merge = run_del and not del_targets_empty and has_tsv(pred_dir, "_DEL")
    do_ins_merge = run_ins and not ins_targets_empty and has_tsv(pred_dir, "_INS")

    if do_del_merge or do_ins_merge:
        merge_cmd = [sys.executable, SCRIPT_DIR / "2_predict_tsv.py",
                     "-i",               pred_dir]
        if do_del_merge:
            merge_cmd += ["--del-output", f_merged_del]
        if do_ins_merge:
            merge_cmd += ["--ins-output", f_merged_ins]
        modes = "+".join(filter(None, ["DEL" if do_del_merge else "", "INS" if do_ins_merge else ""]))
        run_step(merge_cmd, f"2_predict_tsv.py [{modes}] : merge prediction TSVs (combined)")

    if run_del:
        _del_merged_empty = is_targets_tsv_empty(str(f_merged_del))
        if del_targets_empty or _del_merged_empty:
            samples = collect_sample_names(args, pred_dir)
            if _del_merged_empty and not del_targets_empty:
                print(f"\n[INFO] DEL: merged TSV is empty (all predictions NONE), "
                      f"outputting all samples as cluster 1.")
            write_fallback_cluster_txt(
                samples, outdir / f"{args.prefix}_DEL_cluster.txt", "DEL")
            # Even with a fallback cluster assignment, still produce a PCA plot if possible
            # (even with an empty new-sample prediction, if the training data has features,
            #  the new sample can be overlaid at the origin of the existing PCA space; if the
            #  training data also has 0 columns, run_del_pca_plot's own 6_pcapkl.py/7_newplot.py
            #  produce a "PCA not available" placeholder image)
            run_del_pca_plot(args, outdir, f_merged_del)
        elif has_tsv(pred_dir, "_DEL"):
            print("\n" + "#"*60)
            print("# DEL pipeline (merge -> cluster -> distance -> PCA -> plot)")
            print("#"*60)
            run_del_pipeline(args, outdir, pred_dir)
        else:
            print(f"\n[WARN] DEL: no _DEL.tsv found in {pred_dir}. Skipping the DEL pipeline.")

    if run_ins:
        _ins_merged_empty = is_targets_tsv_empty(str(f_merged_ins))
        if ins_targets_empty or _ins_merged_empty:
            samples = collect_sample_names(args, pred_dir)
            if _ins_merged_empty and not ins_targets_empty:
                print(f"\n[INFO] INS: merged TSV is empty (all predictions NONE), "
                      f"outputting all samples as cluster 1.")
            write_fallback_cluster_txt(
                samples, outdir / f"{args.prefix}_INS_cluster.txt", "INS")
            # Same policy as the DEL side (see the comment at the run_del_pca_plot call)
            run_ins_pca_plot(args, outdir, f_merged_ins)
        elif has_tsv(pred_dir, "_INS"):
            print("\n" + "#"*60)
            print("# INS pipeline (merge -> cluster -> distance -> PCA -> plot)")
            print("#"*60)
            run_ins_pipeline(args, outdir, pred_dir)
        else:
            print(f"\n[WARN] INS: no _INS.tsv found in {pred_dir}. Skipping the INS pipeline.")

    print(f"\n{'='*60}")
    print("[Done] All pipelines completed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
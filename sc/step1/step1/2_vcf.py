#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import argparse
import shlex
import os
import sys

from scipy.stats import binomtest

# ご環境の bcftools 実行パスを指定してください（GUIから実行する場合は
# clustering_gui.py の DEFAULT_BCFTOOLS、または画面上の入力欄で指定されます）。
bcftools = None

# bcftools call (-m) はサンプル横断のアレル頻度を事前分布として使うため、
# 集団内で稀な変異では「ALTリードが大多数（例: 38本中35本）でも0/1と判定される」
# という直感に反する結果が起こりうる（多数決ではなく、モデル上の事後確率で決まるため）。
# これを補正するため、「真にヘテロ接合（ALT比0.5）」という帰無仮説のもとで、
# 観測されたAD（リード数）がこれだけALT側に偏る確率（片側二項検定のp値）を計算し、
# 有意水準未満（=偶然ではとても説明できないほど偏っている）の場合だけ1/1に
# 上書きする（片方向のみ）。0/0・0/1同士の間の判定は bcftools call の
# モデルベースの判断（deep有り/塩基クオリティ考慮/集団頻度事前分布）に任せ、
# 一切変更しない -- 0/1の範囲を人為的に狭めたり広げたりしない。
# 固定のALT比閾値（例: 0.9以上）とは異なり、depthに応じて判定の厳しさが
# 自動的に変わる（同じALT比でも、depthが高いほど「偶然の偏り」である可能性は
# 小さくなるため、より低いALT比でも有意になり得る。depthが低いと逆に、
# 高いALT比でも偶然の範囲に収まりやすい）。
# 二対立遺伝子（AD が2値）のサイトのみ対象。既に ./.（DPフィルタ等で欠損）の
# サンプルや、AD が無い/2値でないサンプルはそのまま変更しない。
BINOM_ALPHA = 0.01


def reclassify_gt_by_binomial_test(vcf_path, alpha=BINOM_ALPHA, min_dp=10):
    """VCF内の各サンプルのGTのうち、bcftools call が1/1以外（主に0/1）と
    判定したにもかかわらず、そのサンプル自身のAD（リード数）が、
    「真にヘテロ接合（ALT比0.5）」という帰無仮説のもとでは統計的に
    説明できないほどALTに偏っている（片側二項検定のp値 < alpha）場合だけ、
    1/1に上書きする（片方向のみ）。0/0や、有意水準に達しない0/1は
    bcftoolsの判定のまま変更しない。PL/DP/AD 等、GT以外のFORMATフィールドも
    変更しない。

    min_dp: 既存のbcftools filter（DP<min_dpの1/1を./.にする）は、bcftoolsが
    「元々」1/1と判定したケースにしか効かない。ここでの二項検定による
    再判定によって新たに1/1になったケース（元は0/1などだった）が同じ保護
    なしにすり抜けないよう、AD合計（≈DP）がmin_dp未満なら1/1にはせず
    ./. にする。"""
    with open(vcf_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = []
    n_changed = 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            out_lines.append(line)
            continue

        fields = line.rstrip("\n").split("\t")
        if len(fields) < 10:
            out_lines.append(line)
            continue

        fmt_keys = fields[8].split(":")
        try:
            gt_idx = fmt_keys.index("GT")
            ad_idx = fmt_keys.index("AD")
        except ValueError:
            out_lines.append(line)
            continue

        for i in range(9, len(fields)):
            sample = fields[i]
            if sample in (".", "./."):
                continue
            parts = sample.split(":")
            if len(parts) <= max(gt_idx, ad_idx):
                continue

            gt = parts[gt_idx]
            if gt in (".", "./.", ".|."):
                continue

            ad_str = parts[ad_idx]
            ad_vals = ad_str.split(",")
            if len(ad_vals) != 2:
                continue
            try:
                ref_n, alt_n = int(ad_vals[0]), int(ad_vals[1])
            except ValueError:
                continue

            total = ref_n + alt_n
            if total == 0:
                continue

            # 片方向のみ: 「真にヘテロ（ALT比0.5）」という帰無仮説のもとで、
            # 観測されたALTリード数以上に偏る確率（p値）が有意水準未満の
            # ケースだけを1/1へ上書きする。それ以外（0/0や、有意水準に
            # 達しない0/1）は bcftools call の判定をそのまま尊重し、
            # 一切変更しない。
            if gt == "1/1":
                continue
            p_value = binomtest(alt_n, total, 0.5, alternative="greater").pvalue
            if p_value >= alpha:
                continue
            new_gt = "1/1" if total >= min_dp else "./."

            if new_gt != gt:
                parts[gt_idx] = new_gt
                fields[i] = ":".join(parts)
                n_changed += 1

        out_lines.append("\t".join(fields) + "\n")

    with open(vcf_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"[INFO] GT reclassification by binomial test: overwrote {n_changed} genotype(s) to 1/1 "
          f"(only cases where P(ALT reads >= observed | true het, p=0.5) < {alpha} "
          f"that were not already 1/1; 0/0 and other 0/1 calls are left as bcftools call decided).")


def generate_vcf(bam_files, ref_file, output_vcf, threads=8, min_dp=10, bcftools_bin=None,
                  alpha=BINOM_ALPHA):
    bcftools_bin = bcftools_bin or bcftools

    # --- 前提チェック ---
    if not bcftools_bin:
        print("[ERROR] bcftools path is not specified (specify it with --bcftools).",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(ref_file):
        print(f"[ERROR] Reference FASTA not found: {ref_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(bcftools_bin):
        print(f"[ERROR] bcftools not found: {bcftools_bin}", file=sys.stderr)
        sys.exit(1)
    for b in bam_files:
        if not os.path.exists(b):
            print(f"[ERROR] BAM not found: {b}", file=sys.stderr)
            sys.exit(1)

    # シェル安全のためクォート
    ref_q = shlex.quote(ref_file)
    out_q = shlex.quote(output_vcf)
    bams_q = " ".join(shlex.quote(b) for b in bam_files)
    bcftools_q = shlex.quote(bcftools_bin)

    # -A は入れない / MQ・BQ フィルタは付けない（bcftools のデフォルト挙動のまま）/
    # DP と AD を FORMAT に付与。
    # PL は bcftools call (-m) が自動で FORMAT に付けます（GT, AD, DP, PL が出力されます）
    # bcftools filter: DP < min_dp のサンプルのジェノタイプを ./. に置換（-S '.'）-- これだけは残す。
    command = (
        f"{bcftools_q} mpileup -Ou "
        f"-f {ref_q} "
        f"-a FORMAT/DP,FORMAT/AD "                  # サンプルごとの DP と AD を出力
        f"--threads {threads} "
        f"{bams_q} "
        f"| {bcftools_q} call -mv -Ou --threads {threads} "
        f"| {bcftools_q} filter -e 'GT=\"1/1\" & FMT/DP<{min_dp}' -S '.' -Ov "  # ALT ホモ(1/1)のみ DP < min_dp → ./.
        f"-o {out_q}"
    )

    try:
        print(f"[INFO] Generating VCF file: {output_vcf}")
        # 実行コマンドを表示（デバッグ用）
        print("[CMD]", command)
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            print("[STDOUT]\n" + result.stdout)
        if result.stderr.strip():
            print("[STDERR]\n" + result.stderr)
        print(f"[OK] VCF file written: {output_vcf}")

        reclassify_gt_by_binomial_test(output_vcf, alpha, min_dp)
    except subprocess.CalledProcessError as e:
        print("[ERROR] Command execution failed.", file=sys.stderr)
        print("        returncode:", e.returncode, file=sys.stderr)
        if e.stdout:
            print("[STDOUT]\n" + e.stdout, file=sys.stderr)
        if e.stderr:
            print("[STDERR]\n" + e.stderr, file=sys.stderr)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="複数の BAM から VCF を生成（MQ/BQフィルタなし・bcftoolsのデフォルト挙動, FORMAT: DP, AD, PL 付き）"
    )
    parser.add_argument(
        "-b", "--bams", nargs="+", required=True, help="入力 BAM ファイル（複数可）"
    )
    parser.add_argument(
        "-r", "--ref", required=True, help="参照リファレンス FASTA（.fai が必要）"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="出力 VCF ファイル名（例: out.vcf）"
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=8, help="使用するスレッド数（既定: 8）"
    )
    parser.add_argument(
        "--min-dp", type=int, default=10,
        help="この値未満の DP のサンプルを ./. に置換（既定: 10）"
    )
    parser.add_argument(
        "--bcftools", default=bcftools, required=(bcftools is None),
        help="bcftools 実行パス（必須。ご環境のパスを指定してください）"
    )
    parser.add_argument(
        "--binom-alpha", type=float, default=BINOM_ALPHA,
        help=f"GT再判定に使う二項検定の有意水準（既定: {BINOM_ALPHA}）。"
             "「真にヘテロ（ALT比0.5）」という帰無仮説のもとでのp値がこの値未満の"
             "サンプルだけを1/1に上書きする（片方向のみ。0/0・0/1同士の判定は"
             "bcftools callのまま）。"
    )
    args = parser.parse_args()

    generate_vcf(args.bams, args.ref, args.output, args.threads, args.min_dp,
                bcftools_bin=args.bcftools, alpha=args.binom_alpha)


if __name__ == "__main__":
    main()

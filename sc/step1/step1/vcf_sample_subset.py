#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCF から指定サンプル（品種）のみを抽出するスクリプト。

二段階クラスタリング
  stage0: イントロン領域SNPのみで一次クラスタリング（3_vcf_select.py --mode intron 等）
  stage1: 一次クラスタごとに、遺伝子領域全体のSNPで二次クラスタリング
において、stage1 の対象を「一次クラスタに属する品種のみ」に絞り込むために使用する。

サンプル名の照合は 4_hetero_N_one_vcf_tsv.py がTSV出力時に行う正規化ルール
（basename化 → .sorted.bam / .sort.bam / .bam を除去 → 大文字化）と同じ方式で行う。
これにより、7_kmeans.py が出力する cluster.txt 中のサンプル名（TSVの列名由来）と、
生の VCF の #CHROM 行のサンプル列（BAMファイル名由来）を正しく対応付けられる。
"""

import argparse
import sys


def normalize_sample_name(raw_name: str) -> str:
    """4_hetero_N_one_vcf_tsv.py と同じ正規化ルールでサンプル名を揃える。"""
    return (
        raw_name.split("/")[-1]
        .replace(".sorted.bam", "")
        .replace(".sort.bam", "")
        .replace(".bam", "")
        .upper()
    )


def load_cluster_samples(cluster_file, cluster_id):
    """
    7_kmeans.py / 9_plot.py と同じフォーマットの cluster.txt を読み、
    指定クラスタ（'# cluster N' の出現順、1始まり）に属するサンプル名リストを返す。
    """
    samples = []
    current_idx = -1
    target_idx = cluster_id - 1
    with open(cluster_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                current_idx += 1
            elif line.strip():
                if current_idx == target_idx:
                    samples.append(line.strip())
    return samples


def list_clusters(cluster_file):
    """cluster.txt に含まれる全クラスタの (通し番号, ラベル, サンプル数) を返す。"""
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


def subset_vcf(input_vcf, output_vcf, target_samples):
    """target_samples に一致する列のみを残した VCF を書き出す。"""
    target_set = {s.upper() for s in target_samples}
    kept_count = 0

    with open(input_vcf, 'r') as vin, open(output_vcf, 'w') as vout:
        keep_indices = None
        for line in vin:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                header_fixed = fields[:9]
                sample_fields = fields[9:]

                keep_indices = [
                    i for i, name in enumerate(sample_fields)
                    if normalize_sample_name(name) in target_set
                ]
                if not keep_indices:
                    print("[ERROR] None of the target samples were found in the VCF. "
                          "Please check the sample name correspondence.", file=sys.stderr)
                    sys.exit(1)

                matched = {normalize_sample_name(sample_fields[i]) for i in keep_indices}
                missing = target_set - matched
                if missing:
                    print(f"[WARN] Samples not found in the VCF: {sorted(missing)}", file=sys.stderr)

                kept_header = header_fixed + [sample_fields[i] for i in keep_indices]
                kept_count = len(keep_indices)
                vout.write("\t".join(kept_header) + "\n")
                continue

            if line.startswith("#"):
                vout.write(line)
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            fixed = fields[:9]
            sample_fields = fields[9:]
            kept = [sample_fields[i] for i in keep_indices]
            vout.write("\t".join(fixed + kept) + "\n")

    print(f"[OK] Wrote subset VCF: {output_vcf} (samples: {kept_count})")


def main():
    p = argparse.ArgumentParser(
        description="VCF から指定サンプル（品種）のみを抽出する（二段階クラスタリングのstage1用）"
    )
    p.add_argument("--input_vcf",  help="入力 VCF ファイル（--list_clusters のみの場合は不要）")
    p.add_argument("--output_vcf", help="出力 VCF ファイル（--list_clusters のみの場合は不要）")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--samples", nargs="+", help="抽出するサンプル名を直接指定")
    src.add_argument("--cluster_file", help="cluster.txt（7_kmeans.py / 9_plot.py 形式）のパス")

    p.add_argument("--cluster_id", type=int,
                    help="--cluster_file 指定時、抽出するクラスタ番号（1始まり）")
    p.add_argument("--list_clusters", action="store_true",
                    help="--cluster_file の内容（クラスタ番号・ラベル・サンプル数）を表示して終了する"
                         "（この場合 --input_vcf / --output_vcf は不要）")

    args = p.parse_args()

    if args.cluster_file:
        if args.list_clusters:
            for cid, label, cnt in list_clusters(args.cluster_file):
                print(f"cluster {cid}\tlabel={label}\tn_samples={cnt}")
            return
        if not args.input_vcf or not args.output_vcf:
            p.error("--input_vcf and --output_vcf are required")
        if args.cluster_id is None:
            p.error("--cluster_id is required when using --cluster_file (use --list_clusters to just list clusters)")
        target_samples = load_cluster_samples(args.cluster_file, args.cluster_id)
        if not target_samples:
            print(f"[ERROR] No samples found in cluster {args.cluster_id}.", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.input_vcf or not args.output_vcf:
            p.error("--input_vcf and --output_vcf are required")
        target_samples = args.samples

    subset_vcf(args.input_vcf, args.output_vcf, target_samples)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_vcf_call.py

新サンプルのBAM(複数可)から、--targets（学習時に決定した座標一覧）のみを
対象にbcftoolsでVCFを作成する。sc/step1/step1/2_vcf.py（学習側のVCF作成）と
同じ mpileup/call/filter/ALT比補正の流れを、新サンプル用に座標制限して行う。

学習側との違い:
  - bcftools call は -v（variants-only）を付けない。全ターゲット座標に
    レコードを残す（0/0 も含む）ことで、「VCFに存在しない = そのポジションの
    filterを通過した読み取りが1本も無かった（真の欠損）」という前提を保つ。
    -v を付けてしまうと 0/0 の座標も VCF から消え、真の欠損（N）と
    区別できなくなる。
  - -R でtargetsの座標のみに制限する（学習側は遺伝子領域全体をスキャンする）。

使い方:
  python3 1_vcf_call.py \
    --targets targets.tsv \
    --bam sample1.bam [sample2.bam ...] \
    --ref reference.fasta \
    --output new_sample.vcf \
    [--bcftools /path/to/bcftools] [--min-dp 10] [--binom-alpha 0.01]
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pysam
from scipy.stats import binomtest

# bcftools call (-m) はサンプル横断のアレル頻度を事前分布として使うため、
# ALTリードが大多数でも0/1と判定される直感に反する結果が起こりうる。
# これを補正するため、「真にヘテロ接合（ALT比0.5）」という帰無仮説のもとで、
# 観測されたAD（リード数）がこれだけALT側に偏る確率（片側二項検定のp値）を計算し、
# 有意水準未満の場合だけ1/1に上書きする（片方向のみ）。
# sc/step1/step1/2_vcf.py と同一ロジック。
BINOM_ALPHA = 0.01


def resolve_chr_for_bam(raw_chr: str, bam_contigs: set) -> str | None:
    """
    targets TSV の chr表記（学習時に "chr" プレフィックスを剥がした値。
    sc/step1/step1/4_hetero_N_one_vcf_tsv.py 参照）を、実際のBAM/参照配列の
    contig名に解決する。染色体名が数字のみ（"6" -> "chr06"）とは限らない
    （"A01" -> "chrA01" のような英字+数字の命名の参照ゲノムも実在する ---
    CLAUDE.md の "Species-agnostic by design" と同じ方針）ため、複数の候補を
    作り、実際にBAMのreferencesに存在するものだけを採用する。
    解決できなければ None を返す（呼び出し側でこの座標をスキップする）。

    候補の例（raw_chr="A01" の場合）: "chrA01", "A01"
    候補の例（raw_chr="6" の場合）: "chr06", "chr6", "6"
    """
    s = str(raw_chr).strip()
    candidates = []
    if s.isdigit():
        candidates.append(f"chr{int(s):02d}")
        candidates.append(f"chr{int(s)}")
    m = re.fullmatch(r"chr(\d+)", s, re.IGNORECASE)
    if m:
        candidates.append(f"chr{int(m.group(1)):02d}")
    candidates.append(f"chr{s}")
    candidates.append(s)
    for c in candidates:
        if c in bam_contigs:
            return c
    return None


def reclassify_gt_by_binomial_test(vcf_path, alpha=BINOM_ALPHA, min_dp=10):
    """VCF内の各サンプルのGTのうち、bcftools call が1/1以外（主に0/1）と
    判定したにもかかわらず、そのサンプル自身のAD（リード数）が、
    「真にヘテロ接合（ALT比0.5）」という帰無仮説のもとでは統計的に
    説明できないほどALTに偏っている（片側二項検定のp値 < alpha）場合だけ、
    1/1に上書きする（片方向のみ）。sc/step1/step1/2_vcf.py と同一ロジック
    （コメントもそのまま踏襲）。

    0/0や、有意水準に達しない0/1はbcftoolsの判定のまま変更しない。
    PL/DP/AD 等、GT以外のFORMATフィールドも変更しない。

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
          f"that were not already 1/1; 0/0 and other 0/1 calls from bcftools call are left as-is).")


def build_regions_file(targets_tsv: str, regions_path: str, bam_contigs: set) -> int:
    """targets TSV（chr, posi列）から bcftools -R 用の regions ファイルを作る。
    2列（CHROM, POS）形式 -- bcftoolsはこれを1塩基分の領域として扱う。
    chrom, posi でソートして書き出す。染色体名はresolve_chr_for_bam()で
    実際のBAMのreferencesと突き合わせて解決し、解決できない座標は警告して
    スキップする。戻り値: 書き出した座標数。"""
    df = pd.read_csv(targets_tsv, sep="\t", dtype=str)
    if not {"chr", "posi"}.issubset(df.columns):
        raise ValueError("--targets には chr, posi 列が必要です")

    rows = []
    unresolved = set()
    for chrom_raw, posi_raw in zip(df["chr"], df["posi"]):
        bam_chrom = resolve_chr_for_bam(chrom_raw, bam_contigs)
        if bam_chrom is None:
            unresolved.add(str(chrom_raw))
            continue
        pos = int(posi_raw)
        rows.append((bam_chrom, pos))

    if unresolved:
        print(f"[WARN] No matching chromosome name found in the BAM's references; "
              f"skipped these positions: {sorted(unresolved)}", file=sys.stderr)

    rows = sorted(set(rows))
    with open(regions_path, "w", encoding="utf-8") as f:
        for chrom, pos in rows:
            f.write(f"{chrom}\t{pos}\n")
    return len(rows)


def run_piped(cmds: list, desc: str):
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"{'='*60}")
    cmd_strs = [" ".join(shlex.quote(str(c)) for c in cmd) for cmd in cmds]
    print("CMD:", " | ".join(cmd_strs), "\n")

    procs = []
    prev_stdout = None
    for i, cmd in enumerate(cmds):
        is_last = (i == len(cmds) - 1)
        stdout = None if is_last else subprocess.PIPE
        p = subprocess.Popen([str(c) for c in cmd], stdin=prev_stdout, stdout=stdout)
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


def main():
    ap = argparse.ArgumentParser(
        description="新サンプルBAMから、targets座標のみを対象にVCFを作成する"
                    "（bcftools mpileup | call | filter、学習側2_vcf.pyと同一ロジック + 二項検定によるGT補正）"
    )
    ap.add_argument("--targets", required=True, help="対象座標TSV（chr, posi列）")
    ap.add_argument("--bam", nargs="+", required=True, help="新サンプルのBAMファイル（複数可、.baiインデックス必須）")
    ap.add_argument("--ref", required=True, help="参照ゲノムFASTA（.fai が必要）")
    ap.add_argument("--output", required=True, help="出力VCFファイルパス")
    ap.add_argument("--bcftools", default="bcftools", help="bcftools実行パス（default: bcftools）")
    ap.add_argument("--min-dp", type=int, default=10,
                    help="この値未満のDPの1/1サンプルを./.に置換（default: 10。学習側2_vcf.pyと一致）")
    ap.add_argument("--binom-alpha", type=float, default=BINOM_ALPHA,
                    help=f"GT再判定に使う二項検定の有意水準（default: {BINOM_ALPHA}。学習側2_vcf.pyと一致）")
    args = ap.parse_args()

    for b in args.bam:
        if not os.path.exists(b):
            print(f"[ERROR] BAM not found: {b}", file=sys.stderr)
            sys.exit(1)
    if not os.path.exists(args.ref):
        print(f"[ERROR] Reference FASTA not found: {args.ref}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # targetsのchr表記をBAMの実際のcontig名へ解決するために、1つ目のBAMの
    # referencesを基準にする（--bamに複数渡された場合も、同じ参照ゲノムに
    # 揃っている前提）。
    try:
        with pysam.AlignmentFile(args.bam[0], "rb") as _bam:
            bam_contigs = set(_bam.references)
    except Exception as e:
        print(f"[ERROR] Could not get references from BAM: {args.bam[0]}: {e}", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".regions.tsv", delete=False) as tf:
        regions_path = tf.name
    try:
        n_regions = build_regions_file(args.targets, regions_path, bam_contigs)
        print(f"[INFO] Created regions file: {n_regions} position(s) -> {regions_path}")
        if n_regions == 0:
            print("[ERROR] Could not resolve any targets position to a BAM contig name. "
                  "Check that the reference genome is the same one used when targets were created (Clustering Alleles).",
                  file=sys.stderr)
            sys.exit(1)

        mpileup_cmd = [
            args.bcftools, "mpileup", "-Ou",
            "-f", args.ref,
            "-a", "FORMAT/DP,FORMAT/AD",
            "-R", regions_path,
        ] + list(args.bam)

        call_cmd = [args.bcftools, "call", "-m", "-Ou"]

        filter_cmd = [
            args.bcftools, "filter",
            "-e", f'GT="1/1" & FMT/DP<{args.min_dp}',
            "-S", ".",
            "-Ov",
            "-o", str(output_path),
        ]

        run_piped(
            [mpileup_cmd, call_cmd, filter_cmd],
            "bcftools mpileup | call | filter : create new-sample VCF (targets positions only, all sites output)",
        )
    finally:
        try:
            os.unlink(regions_path)
        except OSError:
            pass

    reclassify_gt_by_binomial_test(str(output_path), args.binom_alpha, args.min_dp)
    print(f"\n[OK] VCF file written: {output_path}")


if __name__ == "__main__":
    main()

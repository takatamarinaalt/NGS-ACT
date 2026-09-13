#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import argparse

def parse_gt_to_alleles(gt: str, ref: str, alt_list):
    """GT を整数インデックスに変換して対応するアリルを返す。"""
    if gt in ('.', './.', '.|.'):
        return [None]
    parts = re.split(r'[\/|]', gt)
    alleles = []
    for p in parts:
        if p == '.':
            alleles.append(None)
            continue
        try:
            idx = int(p)
        except ValueError:
            alleles.append(None)
            continue
        if idx == 0:
            alleles.append(ref)
        elif 1 <= idx <= len(alt_list):
            alleles.append(alt_list[idx - 1])
        else:
            alleles.append(None)
    return alleles

def is_homozygous_nonnull(alleles):
    """全コピーが同一かつ None でない → ホモ判定"""
    vals = [a for a in alleles if a is not None]
    return len(vals) == len(alleles) and len(vals) > 0 and all(v == vals[0] for v in vals)

def classify_alt(ref: str, alt: str) -> str:
    """ALT と REF の長さ差で SNP/INS/DEL を判定"""
    if alt is None:
        return 'NA'
    if len(alt) == len(ref):
        return 'SNP'
    elif len(alt) > len(ref):
        return 'INS'
    else:
        return 'DEL'

def parse_sample_fields(fmt: str, sample: str):
    """FORMAT列(fmt)とサンプル列(sample)から {key: value} を作る。"""
    if not fmt or fmt == "." or not sample:
        return {}
    keys = fmt.split(":")
    vals = sample.split(":")
    d = {}
    for i, k in enumerate(keys):
        d[k] = vals[i] if i < len(vals) else "."
    return d

def get_depth_from_sample(fmt: str, sample: str):
    """
    サンプルのDepthを取得する。
    優先順位:
      1) FORMATの DP
      2) FORMATの AD を合計（AD=ref,alt1,alt2...）
    取れない場合は None。
    """
    d = parse_sample_fields(fmt, sample)
    if not d:
        return None

    if "DP" in d and d["DP"] not in (".", ""):
        try:
            return int(d["DP"])
        except ValueError:
            pass

    if "AD" in d and d["AD"] not in (".", ""):
        try:
            parts = d["AD"].split(",")
            nums = []
            for x in parts:
                if x in (".", ""):
                    return None
                nums.append(int(x))
            return sum(nums)
        except Exception:
            return None

    return None

def is_gt_11(gt: str) -> bool:
    """GTが 1/1 または 1|1 かどうか（ALT1ホモのみ）。"""
    if gt is None:
        return False
    gt = str(gt).strip()
    return gt in ("1/1", "1|1")


def get_ad_from_sample(fmt: str, sample: str):
    """サンプルのAD（allele depth）を (ref_count, alt_count) で返す。
    2値（REF, ALT1）のみ対応。取れない/多対立/不正な値の場合は None。"""
    d = parse_sample_fields(fmt, sample)
    if not d or "AD" not in d or d["AD"] in (".", ""):
        return None
    parts = d["AD"].split(",")
    if len(parts) != 2:
        return None
    try:
        ref_n, alt_n = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if ref_n < 0 or alt_n < 0:
        return None
    return ref_n, alt_n


def ratio_token(ref_class: str, alt_class: str, ref_n: int, alt_n: int) -> str:
    """0/1（ヘテロ）やDPフィルタで ./. にされた1/1など、GTだけでは確定的な
    クラスを決められないがADで実際のリード比率が分かる場合に使うトークン。
    "RATIO|{ref_class}={ref_frac}|{alt_class}={alt_frac}" の形式。
    5_snp_number.py 側でこの形式を認識し、対応する one-hot 列に直接
    実測比率を書き込む（推定で埋めるのではなく、実際のリード数比をそのまま使う）。"""
    total = ref_n + alt_n
    ref_frac = ref_n / total
    alt_frac = alt_n / total
    return f"RATIO|{ref_class}={ref_frac:.4f}|{alt_class}={alt_frac:.4f}"

def vcf_to_tsv_one_line(vcf_file, tsv_file):
    """
    chr・posi ごとに1行出力。

    DPフィルタは 2_vcf.py（bcftools filter）で実施済みのため、
    このスクリプトでは行わない。

    - ヘテロ（0/1）/ DPフィルタ由来の欠損（./.）は、ADの実測リード比率が
      取れれば "RATIO|..." トークン（5_snp_number.py 側で実測比率のまま
      one-hot 列に反映される）。比率が取れない場合のみ N。
    - 0/0（REF ホモ）はリファレンス塩基または NONE
    - 1/1（ALT ホモ）は ALT 塩基 / INS / DEL
    - そのPOSに 1/1 が1つも無い場合、その行を出力しない
    """
    with open(vcf_file, 'r') as vcf, open(tsv_file, 'w', newline='') as tsv:
        w = csv.writer(tsv, delimiter='\t')

        sample_names = []
        for line in vcf:
            if line.startswith("#CHROM"):
                header_fields = line.strip().split("\t")
                sample_names = [
                    name.split("/")[-1]
                        .replace(".sorted.bam", "")
                        .replace(".sort.bam", "")
                        .replace(".bam", "")
                        .upper()
                    for name in header_fields[9:]
                ]
                w.writerow(["chr", "posi"] + sample_names)
                continue

            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom, pos, _id, ref, alt, qual, filt, info, fmt = fields[:9]
            sample_data = fields[9:]
            alt_list = [] if alt == '.' else alt.split(',')

            chrom_numeric = re.sub(r"^(?:chr|CHR|Chr)0?", "", chrom)

            # --- INDEL判定（元スクリプトのまま） ---
            alt_classes = [classify_alt(ref, a) for a in alt_list]
            has_indel = any(c in ('INS', 'DEL') for c in alt_classes)
            if "INDEL" in info:
                has_indel = True

            # --- 行フィルタ条件: 1/1 が1つでもあるか ---
            # DP フィルタは 2_vcf.py で実施済み。ここでは 1/1 の有無のみ確認。
            has_any_11 = any(
                is_gt_11(s.split(':', 1)[0] if s else '.')
                for s in sample_data
            )
            if not has_any_11:
                continue

            # REF側・ALT側のクラスラベル（0/0・1/1 判定と同じロジック）。
            # ヘテロ/DPフィルタ由来の ./. をADの実測比率で置き換える際、
            # どのクラス同士の比率かを決めるのに使う。単一ALT（biallelic）の
            # サイトのみ対応 -- 多対立サイトは alt_class=None のままにして
            # 従来通り N に落とす（誤ったクラスに比率を割り当てないため）。
            ref_class = 'NONE' if has_indel else ('NONE' if len(ref) > 1 else ref)
            alt_class = None
            if len(alt_list) == 1:
                acls0 = alt_classes[0]
                if acls0 == "SNP":
                    alt_class = alt_list[0]
                elif acls0 in ("INS", "DEL"):
                    alt_class = acls0

            row_vals = []
            for sample in sample_data:
                gt = sample.split(':', 1)[0] if sample else '.'
                alleles = parse_gt_to_alleles(gt, ref, alt_list)

                # ヘテロ/欠損: ADの実測リード比率が取れればそれをそのまま使う
                # （0/1本来の意味、およびDP<10で./.にされた1/1も対象 -- どちらも
                # bcftools filter -S '.' はGTのみ書き換えるためADは残っている）。
                # 比率が取れない（AD無し・多対立・DP=0など）場合のみ N のまま。
                if not is_homozygous_nonnull(alleles):
                    ad = get_ad_from_sample(fmt, sample) if alt_class is not None else None
                    if ad is not None and (ad[0] + ad[1]) > 0:
                        row_vals.append(ratio_token(ref_class, alt_class, ad[0], ad[1]))
                    else:
                        row_vals.append('N')
                    continue

                allele = alleles[0]

                # --- 0/0（REFホモ）: 元スクリプトのロジック ---
                if allele == ref:
                    if has_indel:
                        row_vals.append('NONE')
                    else:
                        row_vals.append('NONE' if len(ref) > 1 else ref)
                    continue

                # --- ALTホモ ---
                # まず、このALTが何番目か（1,2,...）を判定
                try:
                    alt_index0 = alt_list.index(allele)  # 0-based
                    alt_number = alt_index0 + 1          # 1-based (GTの番号)
                    acls = alt_classes[alt_index0]
                except ValueError:
                    row_vals.append('N')
                    continue

                # 出力は元スクリプト同様
                if acls == "SNP":
                    row_vals.append(allele)
                elif acls == "INS":
                    row_vals.append("INS")
                elif acls == "DEL":
                    row_vals.append("DEL")
                else:
                    row_vals.append("N")

            w.writerow([chrom_numeric, pos] + row_vals)

    print(f"✅ Saved TSV file '{tsv_file}'.")

def main():
    p = argparse.ArgumentParser(
        description="VCF→TSV（DPフィルタは 2_vcf.py 実施済み。ヘテロ/欠損はADの実測比率があれば"
                    "RATIOトークン・無ければN、1/1なし行は除外）"
    )
    p.add_argument("--input", required=True, help="入力VCFファイル")
    p.add_argument("--output", required=True, help="出力TSVファイル")
    args = p.parse_args()

    vcf_to_tsv_one_line(args.input, args.output)

if __name__ == "__main__":
    main()

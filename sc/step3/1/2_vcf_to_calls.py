#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_vcf_to_calls.py

1_vcf_call.py が作成した新サンプルVCFと --targets を突き合わせ、
sc/step1/step1/4_hetero_N_one_vcf_tsv.py と同じ規約
（0/0=REF塩基/NONE、1/1=ALT塩基/INS/DEL、ヘテロ/欠損はAD実測比率の
RATIOトークン、比率も取れない/そもそもVCFに存在しない座標はN）で
chr, posi, {sample}... 形式のTSVに変換する。

学習側 4_hetero_N_one_vcf_tsv.py との違い:
  - 「1/1が1つも無い行は出力しない」というフィルタは行わない
    （Determining Allelesでは全targets座標を常に出力する必要があるため）。
  - このVCFに存在しない座標（1_vcf_call.py側で全座標にレコードを残す設計に
    しているため、存在しないのは真に読み取りが0本だった場合のみ）は N。
  - 0/0 の座標がSNP由来かINDEL由来か（NONE を出すか実際の塩基を出すか）を、
    このVCF行自体のALT（monomorphicで空のことがある）から判定できない場合に
    備えて、--old-input（学習時の特徴量TSVのカラム名）から座標ごとの
    SNP/INDEL種別を引けるようにしている。

使い方:
  python3 2_vcf_to_calls.py \
    --vcf new_sample.vcf --targets targets.tsv --old-input step5_prefix.tsv \
    --output new_simplified.tsv
"""

import argparse
import os
import re
import sys

import pandas as pd


def guess_sample_name(bam_path: str) -> str:
    """BAMファイルパスからサンプル名を推定する（1_vcf_call.py/4b_novel_cluster.pyと同じロジック）。
    bcftoolsが付けるVCFのサンプル列名は（読み取りグループのSMタグ次第で）フルパスの
    場合があるため、常にこちらの名前を採用する（--bam に渡した順序とVCFのサンプル列の
    順序が一致することを利用して、名前文字列ではなく渡した順序で対応付ける）。"""
    sample = os.path.basename(bam_path)
    for suf in (".rg.bam", ".sort.bam", ".bam"):
        if sample.endswith(suf):
            sample = sample[: -len(suf)]
            break
    return sample


def norm_chrom(c: str) -> str:
    """'chr07' / '7' / 'Chr07' → '7' に正規化（4b_novel_cluster.pyと同一）"""
    s = re.sub(r"^[Cc]hr0*", "", str(c).strip())
    try:
        return str(int(s))
    except ValueError:
        return s


def classify_alt(ref: str, alt) -> str:
    """ALT と REF の長さ差で SNP/INS/DEL を判定（4_hetero_N_one_vcf_tsv.pyと同一）"""
    if alt is None:
        return 'NA'
    if len(alt) == len(ref):
        return 'SNP'
    elif len(alt) > len(ref):
        return 'INS'
    else:
        return 'DEL'


def parse_gt_to_alleles(gt: str, ref: str, alt_list):
    """GT を整数インデックスに変換して対応するアリルを返す（同一ロジック）"""
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


def is_homozygous_nonnull(alleles) -> bool:
    """全コピーが同一かつ None でない → ホモ判定（同一ロジック）"""
    vals = [a for a in alleles if a is not None]
    return len(vals) == len(alleles) and len(vals) > 0 and all(v == vals[0] for v in vals)


def parse_sample_fields(fmt: str, sample: str):
    if not fmt or fmt == "." or not sample:
        return {}
    keys = fmt.split(":")
    vals = sample.split(":")
    return {k: (vals[i] if i < len(vals) else ".") for i, k in enumerate(keys)}


def get_ad_from_sample(fmt: str, sample: str):
    """サンプルのAD（allele depth）を (ref_count, alt_count) で返す。
    2値（REF, ALT1）のみ対応（同一ロジック）"""
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
    """"RATIO|{ref_class}={ref_frac}|{alt_class}={alt_frac}" 形式（同一ロジック）。
    3_snp_number.py 側でこの形式を認識し、実測比率をそのまま one-hot 列に書き込む。"""
    total = ref_n + alt_n
    ref_frac = ref_n / total
    alt_frac = alt_n / total
    return f"RATIO|{ref_class}={ref_frac:.4f}|{alt_class}={alt_frac:.4f}"


def load_locus_types(old_input_path: str) -> dict:
    """--old-input（5_snp_number.py/3_snp_number.py が出力する one-hot 特徴量TSV）の
    カラム名から、座標ごとのSNP/INDEL種別を復元する。
    列名規則: SNP="{chr}_{posi}_{A|T|G|C}", INDEL="{chr}_{posi}_short_{NONE|INS|DEL}"
    （重複列は make_unique() により "__DUPn" が付くのでそれも許容する）。
    戻り値: {(chr_str, posi_int): "SNP"|"INDEL"}"""
    with open(old_input_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")

    indel_re = re.compile(r"^(.+)_(\d+)_short_(?:NONE|INS|DEL)(?:__DUP\d+)?$")
    snp_re = re.compile(r"^(.+)_(\d+)_[ATGC](?:__DUP\d+)?$")

    types: dict = {}
    for col in header[1:]:
        m = indel_re.match(col)
        if m:
            types[(m.group(1), int(m.group(2)))] = "INDEL"
            continue
        m = snp_re.match(col)
        if m:
            types.setdefault((m.group(1), int(m.group(2))), "SNP")
    return types


def parse_vcf(vcf_path: str):
    """VCF → {(norm_chrom, pos): {"ref", "alt_list", "fmt", "samples": {name: raw_field}}}"""
    vcf_data = {}
    sample_names = []
    with open(vcf_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#CHROM"):
                header_fields = line.rstrip("\n").split("\t")
                sample_names = header_fields[9:]
                continue
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, pos, _id, ref, alt, _qual, _filt, _info, fmt = fields[:9]
            sample_fields = fields[9:]
            alt_list = [] if alt == "." else alt.split(",")
            key = (norm_chrom(chrom), int(pos))
            vcf_data[key] = {
                "ref": ref,
                "alt_list": alt_list,
                "fmt": fmt,
                "samples": dict(zip(sample_names, sample_fields)),
            }
    return vcf_data, sample_names


def convert(vcf_path: str, targets_path: str, old_input_path: str, output_path: str,
            bam_paths: list) -> None:
    vcf_data, raw_sample_names = parse_vcf(vcf_path)
    if not raw_sample_names:
        raise ValueError("VCFにサンプル列が見つかりません")
    if len(raw_sample_names) != len(bam_paths):
        raise ValueError(
            f"VCFのサンプル数({len(raw_sample_names)})と--bamに渡したBAM数({len(bam_paths)})が"
            f"一致しません。1_vcf_call.pyに渡した--bamと同じ順序・同じ本数を渡してください。"
        )

    # bcftoolsが付けるVCFのサンプル列名（BAMのフルパス等になりうる）を、
    # --bam に渡した順序で対応付けて guess_sample_name() のクリーンな名前に置き換える。
    sample_names = [guess_sample_name(b) for b in bam_paths]
    name_map = dict(zip(raw_sample_names, sample_names))
    for rec in vcf_data.values():
        rec["samples"] = {name_map[k]: v for k, v in rec["samples"].items()}

    locus_types = load_locus_types(old_input_path)

    targets_df = pd.read_csv(targets_path, sep="\t", dtype=str)
    if not {"chr", "posi"}.issubset(targets_df.columns):
        raise ValueError("--targets には chr, posi 列が必要です")

    rows = []
    n_missing_type = 0
    n_no_record = 0

    for chrom_raw, posi_raw in zip(targets_df["chr"], targets_df["posi"]):
        pos = int(posi_raw)
        key = (norm_chrom(chrom_raw), pos)
        rec = vcf_data.get(key)

        if rec is None:
            # このtargets座標に対応するVCFレコードが無い
            # = 1_vcf_call.py は全座標にレコードを残す設計なので、
            #   これは「全サンプルでfilter通過リードが0本」の真の欠損を意味する
            n_no_record += 1
            row_vals = ["N"] * len(sample_names)
        else:
            ref = rec["ref"]
            alt_list = rec["alt_list"]
            fmt = rec["fmt"]
            alt_classes = [classify_alt(ref, a) for a in alt_list]
            row_has_alt_indel = any(c in ("INS", "DEL") for c in alt_classes)

            if row_has_alt_indel:
                has_indel = True
            elif alt_list:
                # ALTはあるがINS/DELではない（SNPのALT）→ このサイトはSNP行として扱う
                has_indel = False
            else:
                # monomorphic（ALT無し）→ 実際のALTから種別を判断できないので
                # 学習時に決定した種別（--old-input）にフォールバックする
                locus_type = locus_types.get((str(chrom_raw), pos))
                if locus_type is not None:
                    has_indel = (locus_type == "INDEL")
                else:
                    has_indel = False
                    n_missing_type += 1

            ref_class = 'NONE' if has_indel else ('NONE' if len(ref) > 1 else ref)
            alt_class = None
            if len(alt_list) == 1:
                acls0 = alt_classes[0]
                if acls0 == "SNP":
                    alt_class = alt_list[0]
                elif acls0 in ("INS", "DEL"):
                    alt_class = acls0

            row_vals = []
            for sample in sample_names:
                sample_field = rec["samples"].get(sample, ".")
                gt = sample_field.split(":", 1)[0] if sample_field else "."
                alleles = parse_gt_to_alleles(gt, ref, alt_list)

                if not is_homozygous_nonnull(alleles):
                    ad = get_ad_from_sample(fmt, sample_field) if alt_class is not None else None
                    if ad is not None and (ad[0] + ad[1]) > 0:
                        row_vals.append(ratio_token(ref_class, alt_class, ad[0], ad[1]))
                    else:
                        row_vals.append("N")
                    continue

                allele = alleles[0]

                if allele == ref:
                    row_vals.append("NONE" if has_indel else ("NONE" if len(ref) > 1 else ref))
                    continue

                try:
                    alt_index0 = alt_list.index(allele)
                    acls = alt_classes[alt_index0]
                except ValueError:
                    row_vals.append("N")
                    continue

                if acls == "SNP":
                    row_vals.append(allele)
                elif acls == "INS":
                    row_vals.append("INS")
                elif acls == "DEL":
                    row_vals.append("DEL")
                else:
                    row_vals.append("N")

        rows.append([chrom_raw, posi_raw] + row_vals)

    if n_missing_type:
        print(f"[WARN] Monomorphic position(s) with no SNP/INDEL type found in --old-input: "
              f"{n_missing_type} (treated as SNP)", file=sys.stderr)
    if n_no_record:
        print(f"[INFO] Position(s) with no VCF record (= 0 reads, treated as N for all samples): "
              f"{n_no_record} / {len(targets_df)}")

    out_df = pd.DataFrame(rows, columns=["chr", "posi"] + sample_names)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"[OK] Output: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="1_vcf_call.py が作成したVCFをtargets座標のTSV（chr/posi/sample...）に変換する"
                    "（0/0=REF, 1/1=ALT/INS/DEL, ヘテロ/欠損=RATIOトークン, VCFに無い座標=N）"
    )
    ap.add_argument("--vcf", required=True, help="1_vcf_call.py が出力したVCF")
    ap.add_argument("--targets", required=True, help="対象座標TSV（chr, posi列）")
    ap.add_argument("--old-input", required=True,
                    help="学習時の特徴量TSV（step5_{prefix}.tsv など）。"
                        "座標ごとのSNP/INDEL種別をカラム名から復元するために使用")
    ap.add_argument("--bam", nargs="+", required=True,
                    help="1_vcf_call.py に渡したのと同じBAMファイル一覧（同じ順序で）。"
                        "VCFのサンプル列名をこの順序でクリーンな名前に対応付けるために使用")
    ap.add_argument("--output", required=True, help="出力TSV")
    args = ap.parse_args()

    convert(args.vcf, args.targets, args.old_input, args.output, args.bam)


if __name__ == "__main__":
    main()

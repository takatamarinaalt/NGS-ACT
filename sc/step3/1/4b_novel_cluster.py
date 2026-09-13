#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4b_novel_cluster.py

4_cluster.py の出力クラスタTXTに、VCFベースのハプロタイプ照合と新規多型クラスタを追加する。

処理の優先順位:
  1. 既知ハプロタイプ（4_cluster.py が既に割当済みの非数字ラベル）     → そのまま
  2. VCF + haplotype_info で一致するハプロタイプが見つかった場合       → ハプロタイプ名
  3. --targets 外の新規ポジションで 1/1 を持つ品種（同一セットでグループ）→ 新クラスタ番号
  4. 上記いずれにも該当しない場合                                       → KMeans番号のまま

条件:
  - VCFハプロタイプ照合: 全ポジション（targets内外問わず）1/1 かつ SNPはALT一致
  - 新規クラスタ: targets外ポジションのみ / 1/1 のみ / 同一frozensetが同一クラスタ
  - 複数ハプロタイプ一致: 全て "/" で連結して出力

使い方:
  python3 4b_novel_cluster.py \
    --cluster-txt  4_cluster_output.txt \
    --vcf          novel.vcf.gz \
    --targets      targets.tsv \
    --model        kmeans_model.joblib \
    --bam          sample1.bam sample2.bam ... \
    --output       updated_cluster.txt
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #

def guess_sample_name(bam_path: str) -> str:
    """BAMファイルパスからサンプル名を推定する（1_vcf_call.py と同じロジック）"""
    sample = os.path.basename(bam_path)
    for suf in (".rg.bam", ".sort.bam", ".bam"):
        if sample.endswith(suf):
            sample = sample[: -len(suf)]
            break
    return sample


def norm_chrom(c: str) -> str:
    """'chr01' / '1' / 'Chr7' → '1', '7' など数字文字列に正規化"""
    s = re.sub(r"^[Cc]hr0*", "", str(c).strip())
    try:
        return str(int(s))
    except ValueError:
        return s


# ------------------------------------------------------------------ #
# クラスタTXT 読み書き
# ------------------------------------------------------------------ #

def read_cluster_txt(path: Path) -> dict[str, str]:
    """クラスタTXT → {sample: label}"""
    sample_label: dict[str, str] = {}
    current_label = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                m = re.search(r"#\s*(?:cluster\s+)?(\S+)", stripped)
                if m:
                    current_label = m.group(1)
            else:
                if current_label is not None:
                    sample_label[stripped] = current_label
    return sample_label


def write_cluster_txt(path: Path, sample_label: dict[str, str]) -> None:
    """{sample: label} → クラスタTXT"""
    groups: dict[str, list[str]] = defaultdict(list)
    for sample, label in sample_label.items():
        groups[label].append(sample)

    def sort_key(lbl: str):
        return (0, int(lbl), "") if lbl.isdigit() else (1, 0, lbl)

    with open(path, "w", encoding="utf-8") as f:
        for label in sorted(groups, key=sort_key):
            header = f"# cluster {label}" if label.isdigit() else f"# {label}"
            f.write(header + "\n")
            for s in sorted(groups[label]):
                f.write(f"{s}\n")
            f.write("\n")


# ------------------------------------------------------------------ #
# モデル読み込み
# ------------------------------------------------------------------ #

def load_haplotype_info(model_path: Path) -> dict:
    """joblib から haplotype_info を取得する。なければ空dict。"""
    obj = joblib.load(str(model_path))
    if isinstance(obj, dict):
        return obj.get("haplotype_info", {})
    return {}


# ------------------------------------------------------------------ #
# targets / VCF 読み込み
# ------------------------------------------------------------------ #

def load_targets_set(targets_tsv: Path) -> set[tuple[str, int]]:
    """targets TSV → {(norm_chrom, pos_int)} の集合"""
    df = pd.read_csv(targets_tsv, sep="\t", dtype=str)
    result: set[tuple[str, int]] = set()
    for _, row in df.iterrows():
        try:
            c = norm_chrom(str(row["chr"]))
            p = int(str(row["posi"]).strip())
            result.add((c, p))
        except (ValueError, KeyError):
            continue
    return result


def get_vcf_samples(vcf_path: Path, bcftools_bin: str = "bcftools") -> list[str]:
    """bcftools query -l でVCFのサンプル名一覧を取得"""
    result = subprocess.run(
        [bcftools_bin, "query", "-l", str(vcf_path)],
        capture_output=True, text=True, check=True,
    )
    return [s for s in result.stdout.strip().split("\n") if s]


def query_vcf_full(vcf_path: Path, bcftools_bin: str = "bcftools") -> pd.DataFrame:
    """
    bcftools query で VCF から全サンプルの CHROM, POS, REF, ALT, SAMPLE, GT を取得する。
    GT の種類を問わず全行を返す（ハプロタイプ照合と新規クラスタ両方に使用）。
    REF は、ハプロタイプの定義塩基がNipponbare参照ゲノム側の塩基と一致する場合
    （GT=0/0 が正当な一致になるケース。例: マイナス鎖遺伝子でNipponbare自身が
    ハプロタイプ定義アレルを持つ場合）の判定に使う。
    返り値: DataFrame(CHROM, POS, REF, ALT, SAMPLE, GT)
    """
    result = subprocess.run(
        [bcftools_bin, "query", "-f", "[%CHROM\t%POS\t%REF\t%ALT\t%SAMPLE\t%GT\n]",
         str(vcf_path)],
        capture_output=True, text=True, check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 6:
            continue
        chrom, pos_str, ref, alt, sample, gt = parts
        try:
            rows.append((chrom, int(pos_str), ref, alt, sample, gt))
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=["CHROM", "POS", "REF", "ALT", "SAMPLE", "GT"])


# ------------------------------------------------------------------ #
# サンプル名マッピング
# ------------------------------------------------------------------ #

def build_name_map(vcf_samples: list[str], bam_paths: list[str]) -> dict[str, str]:
    """
    VCFサンプル名 → クラスタ用サンプル名（BAMファイル名ベース）のマッピングを作成する。
    完全一致を優先し、前方/後方一致でフォールバック。
    """
    bam_names = [guess_sample_name(b) for b in bam_paths] if bam_paths else []
    name_map: dict[str, str] = {}
    for vs in vcf_samples:
        if vs in bam_names:
            name_map[vs] = vs
            continue
        matched = [bn for bn in bam_names if bn in vs or vs in bn]
        if len(matched) == 1:
            name_map[vs] = matched[0]
        elif len(matched) > 1:
            print(f"[WARN] VCF sample '{vs}' matched multiple BAM names -> skipped",
                  file=sys.stderr)
    return name_map


# ------------------------------------------------------------------ #
# VCFベース ハプロタイプ照合
# ------------------------------------------------------------------ #

def check_haplotype_from_vcf(
    sample_name: str,
    vcf_df: pd.DataFrame,
    haplotype_info: dict,
) -> list[str]:
    """
    VCFデータと haplotype_info を照合し、一致するハプロタイプ名をリストで返す。

    照合ロジック（4_cluster.py の check_haplotype と同じ考え方、VCFベース版）:
      - ポジションが VCF に存在しない                  → REF扱い → ミスマッチ
      - SNP: GT=1/1 かつ ALT が haplotype_info の alt と一致           → 一致
             GT=0/0 かつ REF が haplotype_info の alt と一致           → 一致
             （ハプロタイプの定義塩基がNipponbare参照側の塩基と一致するケースに対応。
               例: マイナス鎖遺伝子でNipponbare自身がそのハプロタイプのアレルを持つ場合、
               真の保有品種はVCF上ではGT=0/0＝REF側ホモとして現れる）
             上記いずれでもない（0/1等のヘテロを含む）                  → ミスマッチ
      - INS/DEL: GT が 1/1 でない                                      → ミスマッチ
    全ポジション（targets内外問わず）を照合対象とする。
    複数ハプロタイプに一致した場合は全て返す。
    """
    samp_rows = vcf_df[vcf_df["SAMPLE"] == sample_name]
    # (norm_chrom, pos) → {ref, alt, gt}
    sample_variants: dict[tuple[str, int], dict] = {}
    for _, row in samp_rows.iterrows():
        key = (norm_chrom(str(row["CHROM"])), int(row["POS"]))
        sample_variants[key] = {
            "ref": str(row["REF"]),
            "alt": str(row["ALT"]),
            "gt": str(row["GT"]),
        }

    matched = []
    for hap_name, hap_info in haplotype_info.items():
        variants = hap_info.get("variants", [])
        if not variants:
            continue

        all_match = True
        for v in variants:
            key = (norm_chrom(str(v["chrom"])), int(v["pos"]))
            vt_kind = v.get("vt_kind", "UNKNOWN")
            expected_alt = (v.get("alt") or "").upper()

            sv = sample_variants.get(key)
            if sv is None:
                all_match = False
                break

            gt = sv["gt"]

            if vt_kind == "SNP":
                if gt in ("1/1", "1|1") and sv["alt"].upper() == expected_alt:
                    pass  # ALT側ホモがハプロタイプ定義塩基と一致
                elif gt == "0/0" and sv["ref"].upper() == expected_alt:
                    pass  # REF側ホモ（Nipponbare一致）がハプロタイプ定義塩基と一致
                else:
                    all_match = False
                    break
            elif vt_kind in ("INS", "DEL"):
                if gt not in ("1/1", "1|1"):
                    all_match = False
                    break
            else:
                all_match = False
                break

        if all_match:
            matched.append(hap_name)

    return matched


# ------------------------------------------------------------------ #
# 新規多型抽出（targets外 & 1/1 のみ）
# ------------------------------------------------------------------ #

def extract_novel_variants(
    vcf_df: pd.DataFrame,
    targets: set[tuple[str, int]],
    name_map: dict[str, str],
    cluster_samples: set[str],
) -> dict[str, frozenset[tuple[str, int]]]:
    """
    クラスタサンプル名 → 新規ポジション（1/1 かつ targets外）の frozenset を返す。
    """
    hom_alt = {"1/1", "1|1"}
    result: dict[str, set] = {}
    for _, row in vcf_df.iterrows():
        if row["GT"] not in hom_alt:
            continue

        chrom_norm = norm_chrom(str(row["CHROM"]))
        pos = int(row["POS"])
        if (chrom_norm, pos) in targets:
            continue

        vcf_samp = str(row["SAMPLE"])
        cluster_samp = name_map.get(vcf_samp, vcf_samp)
        if cluster_samp not in cluster_samples:
            continue

        result.setdefault(cluster_samp, set()).add((chrom_norm, pos))

    return {s: frozenset(v) for s, v in result.items()}


# ------------------------------------------------------------------ #
# ラベル更新（優先順位: 既知ハプロタイプ > VCFハプロタイプ > 新規クラスタ > KMeans）
# ------------------------------------------------------------------ #

def assign_labels(
    sample_label: dict[str, str],
    vcf_df: pd.DataFrame,
    haplotype_info: dict,
    novel_map: dict[str, frozenset],
    name_map: dict[str, str],
) -> dict[str, str]:
    """
    優先順位に従ってラベルを更新する。
    """
    updated = dict(sample_label)
    max_cluster = max(
        (int(lbl) for lbl in sample_label.values() if lbl.isdigit()),
        default=0,
    )

    # VCFサンプル名 → クラスタサンプル名の逆引き（ハプロタイプ照合用）
    vcf_sample_set = set(vcf_df["SAMPLE"].unique()) if not vcf_df.empty else set()

    def resolve_vcf_name(cluster_samp: str) -> str | None:
        """クラスタサンプル名に対応するVCFサンプル名を返す"""
        if cluster_samp in vcf_sample_set:
            return cluster_samp
        for vcf_s, cs in name_map.items():
            if cs == cluster_samp:
                return vcf_s
        return None

    fs_to_label: dict[frozenset, str] = {}
    next_cluster = max_cluster + 1

    for sample, label in sample_label.items():
        # --- 優先度1: 既知ハプロタイプ（既に非数字ラベル）---
        if not label.isdigit():
            continue

        # --- 優先度2: VCFハプロタイプ照合 ---
        if haplotype_info and not vcf_df.empty:
            vcf_name = resolve_vcf_name(sample)
            if vcf_name is not None:
                hap_matches = check_haplotype_from_vcf(vcf_name, vcf_df, haplotype_info)
                if hap_matches:
                    updated[sample] = "/".join(sorted(hap_matches))
                    continue

        # --- 優先度3: 新規多型クラスタ ---
        fs = novel_map.get(sample)
        if fs:
            if fs not in fs_to_label:
                fs_to_label[fs] = str(next_cluster)
                next_cluster += 1
            updated[sample] = fs_to_label[fs]
            continue

        # --- 優先度4: KMeans番号のまま ---

    return updated


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="クラスタTXTに VCFハプロタイプ照合・新規多型クラスタを追加する"
    )
    ap.add_argument("--cluster-txt", required=True,
                    help="4_cluster.py の出力TXT")
    ap.add_argument("--vcf", required=True,
                    help="bcftools call で作成したVCF（.vcf または .vcf.gz）")
    ap.add_argument("--targets", required=True,
                    help="1_bam_tsv.py に渡した座標TSV（chr / posi 列）")
    ap.add_argument("--model", default=None,
                    help="step1 の joblib（haplotype_info 取得用。省略時はハプロタイプ照合スキップ）")
    ap.add_argument("--bam", nargs="*", default=None,
                    help="BAMファイルパス（VCFサンプル名とクラスタ名のマッピング用。"
                         "省略時はVCFサンプル名をそのまま使用）")
    ap.add_argument("--output", required=True,
                    help="出力TXT")
    ap.add_argument("--bcftools", default="bcftools",
                    help="bcftools 実行パス（default: bcftools。PATH に無い場合は絶対パスを指定）")
    args = ap.parse_args()

    # --- 入力読み込み ---
    sample_label = read_cluster_txt(Path(args.cluster_txt))
    if not sample_label:
        print("[ERROR] No samples found in cluster TXT", file=sys.stderr)
        sys.exit(1)

    targets = load_targets_set(Path(args.targets))
    print(f"[INFO] Number of targets positions: {len(targets)}")

    haplotype_info: dict = {}
    if args.model:
        haplotype_info = load_haplotype_info(Path(args.model))
        print(f"[INFO] Number of registered haplotype_info entries: {len(haplotype_info)}")
    else:
        print("[INFO] --model omitted -> skipping VCF haplotype matching")

    vcf_samples = get_vcf_samples(Path(args.vcf), bcftools_bin=args.bcftools)
    print(f"[INFO] Number of VCF samples: {len(vcf_samples)}")

    vcf_df = query_vcf_full(Path(args.vcf), bcftools_bin=args.bcftools)
    print(f"[INFO] Number of VCF records (all GT): {len(vcf_df)}")

    # VCFサンプル名 ↔ クラスタサンプル名のマッピング
    name_map = build_name_map(vcf_samples, args.bam or [])
    unmapped = [vs for vs in vcf_samples
                if vs not in name_map and vs not in sample_label]
    if unmapped:
        print(f"[WARN] VCF samples that could not be mapped to cluster TXT: {unmapped}",
              file=sys.stderr)

    # 新規多型の抽出（targets外 & 1/1）
    cluster_samples = set(sample_label.keys())
    novel_map = extract_novel_variants(vcf_df, targets, name_map, cluster_samples)

    novel_samples = {s for s, v in novel_map.items() if v}
    if novel_samples:
        print(f"\n[INFO] Samples with novel polymorphism(s) (outside targets, 1/1): {len(novel_samples)} sample(s)")
        for s in sorted(novel_samples):
            print(f"  {s}: {sorted(novel_map[s])}")

    # ラベル更新
    updated = assign_labels(
        sample_label, vcf_df, haplotype_info, novel_map, name_map
    )

    # 変更サマリ
    changed = [
        (s, sample_label[s], updated[s])
        for s in updated
        if updated[s] != sample_label.get(s)
    ]
    if changed:
        print(f"\n[INFO] Label changes: {len(changed)} sample(s)")
        for s, old, new in sorted(changed):
            print(f"  {s}: {old} -> {new}")
    else:
        print("\n[INFO] No label changes")

    write_cluster_txt(Path(args.output), updated)
    print(f"\n[OK] Output: {args.output}")


if __name__ == "__main__":
    main()

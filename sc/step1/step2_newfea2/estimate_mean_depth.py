#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_mean_depth.py

サンプルごとの平均デプス（BAM平均カバレッジ）を計算し、TSV（sample, mean_depth）に保存する。

1_predict_combined.py の estimate_mean_depth() と同一ロジック（コピー）。
DEL/INS予測の特徴量（DEPTH_RATIO 等）は「該当ポジションのカバレッジ ÷ このBAM平均デプス」
で正規化されるため、この値は必ず領域抽出（±10kb等）前の全長BAMに対して計算する必要がある。
抽出後のBAM（縮小されたゲノム長ではなくヘッダーのgenome_lengthはそのままなのに
total_mappedだけ激減する）に対して計算すると、平均デプスが実際より大幅に小さく
見積もられ、DEPTH_RATIO系の特徴量が全て壊れる。

出力TSVには mean_depth に加えて median_isize（ライブラリ全体の中央値insert size、
INSERT_SIZE_DIFF 特徴量の基準値）も含む。これも mean_depth と同じ理由で、必ず
領域抽出前の全長BAMに対して計算する必要がある -- 抽出済みの狭い領域だけでは、
ライブラリ全体を代表する insert size のサンプリング元になる読み取りが
ほとんど残っておらず、正しく推定できない（実際にテストデータの染色体先頭付近の
遺伝子で、この問題が起きることを確認済み）。

計算したTSVは 1_predict_combined.py --mean-depth-tsv にそのまま渡せる
（サンプル名の正規化は 1_predict_combined.py の sample_basename().upper() と同一）。

使用例:
  python3 estimate_mean_depth.py --bam /path/to/bam_dir -o mean_depth.tsv
  python3 estimate_mean_depth.py --bam sample1.bam sample2.bam -o mean_depth.tsv
"""

import argparse
import math
import multiprocessing
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pysam
# BAMより.baiが古い場合に htslib が出す "The index file is older than the
# data file" 警告を抑える（実害は無い。GUIログにエラーのように出るのを防ぐ）。
pysam.set_verbosity(0)

COVER_MAPQ_THRESHOLD = 20
MEAN_DEPTH_SAMPLE_READS = 500
INSERT_SIZE_SAMPLE_READS = 5000


def sample_basename(bam_path: str) -> str:
    name = Path(bam_path).name
    for suf in (".rg.bam", ".sort.bam", ".bam"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def dedup_sample_names_upper(bam_paths: List[str]):
    seen = {}
    mapping = {}
    for p in bam_paths:
        base = sample_basename(p).upper()
        n = seen.get(base, 0)
        mapping[p] = base if n == 0 else f"{base}.{n}"
        seen[base] = n + 1
    return mapping


def expand_bam_inputs(inputs: List[str]) -> List[str]:
    bam_files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            # Exclude macOS AppleDouble sidecar files (e.g. "._sample.bam") that show
            # up on exFAT/FAT32 external drives and network shares -- they match
            # "*.bam" but aren't real BAMs and make pysam fail with "Exec format error".
            dir_bams = sorted(b for b in p.glob("*.bam") if not b.name.startswith("._"))
            if not dir_bams:
                print(f"[WARN] No .bam files found in directory: {inp}")
            else:
                print(f"[INFO] Found {len(dir_bams)} .bam file(s) in directory {inp}")
                bam_files.extend([str(b) for b in dir_bams])
        elif p.is_file():
            if str(p).endswith(".bam"):
                bam_files.append(str(p))
            else:
                print(f"[WARN] Not a .bam file (skipped): {inp}")
        else:
            print(f"[WARN] Path not found or inaccessible: {inp}")
    return bam_files


def estimate_mean_depth(bam_path: str) -> float:
    """1_predict_combined.py の同名関数と同一ロジック。"""
    bam = pysam.AlignmentFile(bam_path, "rb")

    genome_length = sum(bam.lengths)
    if genome_length == 0:
        print("[WARN] Genome length is 0. mean_depth will be NaN.", file=sys.stderr)
        bam.close()
        return float("nan")

    try:
        stats = bam.get_index_statistics()
        total_mapped = sum(s.mapped for s in stats)
    except Exception as e:
        print(f"[WARN] Failed to get index statistics ({e}).", file=sys.stderr)
        bam.close()
        return float("nan")

    if total_mapped == 0:
        print("[WARN] Mapped read count is 0. mean_depth will be NaN.", file=sys.stderr)
        bam.close()
        return float("nan")

    read_lengths = []
    for read in bam.fetch():
        if read.is_unmapped:
            continue
        if read.mapping_quality < COVER_MAPQ_THRESHOLD:
            continue
        if read.query_length and read.query_length > 0:
            read_lengths.append(read.query_length)
        if len(read_lengths) >= MEAN_DEPTH_SAMPLE_READS:
            break

    bam.close()

    if not read_lengths:
        print("[WARN] Failed to sample read lengths. mean_depth will be NaN.", file=sys.stderr)
        return float("nan")

    read_lengths.sort()
    median_read_len = read_lengths[len(read_lengths) // 2]
    return (total_mapped * median_read_len) / genome_length


def estimate_median_insert_size(bam_path: str) -> float:
    """1_predict_combined.py / INS_features_v2.py の同名関数と同一ロジック。
    proper pairのTLENからライブラリ全体の中央値insert sizeを推定する。"""
    bam = pysam.AlignmentFile(bam_path, "rb")
    references = list(bam.references)
    lengths = list(bam.lengths)

    anchors = []
    for chrom, length in zip(references, lengths):
        if length <= 0:
            continue
        for frac in (0.02, 0.5, 0.98):
            anchors.append((chrom, int(length * frac)))

    if not anchors:
        bam.close()
        return float("nan")

    per_region_target = max(1, math.ceil(INSERT_SIZE_SAMPLE_READS / len(anchors)))

    isizes = []
    for chrom, start in anchors:
        if len(isizes) >= INSERT_SIZE_SAMPLE_READS:
            break
        n_before = len(isizes)
        for read in bam.fetch(chrom, start, None):
            if read.is_unmapped or not read.is_paired:
                continue
            if read.mate_is_unmapped or not read.is_proper_pair:
                continue
            if read.mapping_quality < COVER_MAPQ_THRESHOLD:
                continue
            if read.next_reference_id != read.reference_id:
                continue
            if read.template_length == 0:
                continue
            isizes.append(abs(read.template_length))
            if len(isizes) - n_before >= per_region_target:
                break
            if len(isizes) >= INSERT_SIZE_SAMPLE_READS:
                break
    bam.close()

    if not isizes:
        return float("nan")
    return statistics.median(isizes)


def estimate_both(bam_path: str):
    """1BAM分の mean_depth と median_isize をまとめて計算する（並列ワーカー用）。"""
    return estimate_mean_depth(bam_path), estimate_median_insert_size(bam_path)


def main():
    ap = argparse.ArgumentParser(
        description="サンプルごとの平均デプス・ライブラリ中央値insert sizeを計算し"
                     "TSV(sample, mean_depth, median_isize)に保存する。"
                     "必ず領域抽出前の全長BAMに対して実行すること。"
    )
    ap.add_argument("--bam", nargs="+", required=True,
                    help="入力BAMファイル（複数可）またはBAMが入ったディレクトリ（全長BAMを指定すること）")
    ap.add_argument("-o", "--output", required=True, help="出力TSVパス")
    ap.add_argument("--jobs", "-j", type=int, default=0,
                    help="並列処理数（0=全コア使用。デフォルト: 0）。"
                         "この処理はBAMファイルごとに独立しているため並列化の効果が大きい。")
    args = ap.parse_args()

    bam_files = expand_bam_inputs(args.bam)
    if not bam_files:
        print("[ERROR] No BAM files found to process", file=sys.stderr)
        sys.exit(1)

    name_map = dedup_sample_names_upper(bam_files)
    n_jobs = args.jobs if args.jobs > 0 else multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, len(bam_files)))
    print(f"[INFO] Processing {len(bam_files)} BAM(s) with parallelism {n_jobs}.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_bam = {executor.submit(estimate_both, bam_path): bam_path
                         for bam_path in bam_files}
        for future in as_completed(future_to_bam):
            bam_path = future_to_bam[future]
            sample = name_map[bam_path]
            depth, isize = future.result()
            results[bam_path] = (depth, isize)
            depth_str = "NaN" if math.isnan(depth) else f"{depth:.2f}"
            isize_str = "NaN" if math.isnan(isize) else f"{isize:.1f}"
            print(f"[INFO] {sample}: mean_depth={depth_str} median_isize={isize_str}")

    # repr() round-trips a float exactly (unlike a fixed-precision format like .2f,
    # which would truncate and make downstream DEPTH_RATIO features computed from the
    # extracted BAM differ from features computed against the original full-length
    # BAM at the ~1e-6 relative level). Write in original discovery order for a
    # stable, human-scannable file (parallel completion order is nondeterministic).
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sample\tmean_depth\tmedian_isize\n")
        for bam_path in bam_files:
            sample = name_map[bam_path]
            depth, isize = results[bam_path]
            depth_str = "" if math.isnan(depth) else repr(depth)
            isize_str = "" if math.isnan(isize) else repr(isize)
            f.write(f"{sample}\t{depth_str}\t{isize_str}\n")

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

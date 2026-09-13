#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INS / DEL 統合予測スクリプト

DELモデルはDEL特徴量系、INSモデルはINS特徴量系を用いて予測する（特徴量・モデルは完全に独立）。
DEL/INSは同一のBAM走査ループ内でまとめて計算し、モデルロードも1回にまとめることで、
従来のDEL単独スキャン→INS単独スキャンの2回実行に比べてループ回数・モデルロード回数を半分にする。

高速化の本命は _force_single_threaded_predict()：DEL/INSモデルは学習時に
n_jobs=-1（全コア並列）で保存されているが、本スクリプトは1ポジションずつ
predict_proba を呼ぶため、そのままだと呼び出しのたびに木の並列化オーバーヘッド
（プロセス/スレッド間通信）が発生し支配的なボトルネックになる
（外側で positions/chrom 単位に並列化しているため二重並列でもある）。
ロード直後に n_jobs=1 を強制することで実測で約10倍高速化する
（出力される確率値はn_jobsの値に関わらず同一）。

出力ファイルは従来通り1品種あたり {SAMPLE}_DEL.tsv と {SAMPLE}_INS.tsv の2本のまま
（列フォーマットも不変。下流の 3_predict_tsv.py / _run_pipeline.py の
 ファイル名判定・閾値再設定ロジックに影響しない）。

[DEL特徴量] - DEL_features.py 準拠
- HMQ_RATIO         : MQ60リード数 / BAM平均デプス                     (当該POSのみ)
- HMQ_RATIO_DELTA   : 前10bpのHMQ_RATIO平均 - 後10bpのHMQ_RATIO平均   (±10bp)
- LARGE_INSERT_RATIO: insert>=ライブラリ中央値insert sizeのリード数の合計 / pairedリード数の合計  (±10bp, MQ60のみ)
- LOW_MQ_RATIO      : MQ0-20のリード数 / BAM平均デプス                 (当該POSのみ)
- RIGHT_SOFT_RATIO  : 右softclipリード数 / softclipリード数            (±5bp, MQ60)
- LEFT_SOFT_RATIO   : 左softclipリード数 / softclipリード数            (±5bp, MQ60)
- DEPTH_RATIO       : MQ>=20のcover / BAM平均デプス                    (当該POSのみ)

[INS特徴量] - INS_predict.py / INS_features.py 準拠（全てMAPQ=60のリードのみ対象）
- START_RATIO          : リード始点数 / BAM平均デプス
- END_RATIO            : リード終点数 / BAM平均デプス
- INTERNAL_RATIO       : 内部カバーリード数 / BAM平均デプス
- SOFTCLIP_LONG_RATIO  : softclip >= read長の半分 / BAM平均デプス
- SOFTCLIP_SHORT_RATIO : softclip < read長の半分 / BAM平均デプス
- LEFT_SOFT_RATIO      : 左softclip / (左+右softclip)
- RIGHT_SOFT_RATIO     : 右softclip / (左+右softclip)
- LARGE_INSERT_RATIO   : insertサイズ>10000のリード数 / ペアリード数   (当該POSのみ)
- SMALL_INSERT_RATIO   : insertサイズ<300のリード数 / ペアリード数     (当該POSのみ)
- MATE_UNMAP_RATIO     : mateがunmappedのリード数 / ペアリード数
- MATE_DIFFCHR_RATIO   : mateが他染色体のリード数 / ペアリード数
- DEPTH_RATIO          : MQ>=20のcover / BAM平均デプス
- DEPTH_RATIO_PREV     : MQ>=20のcover / BAM平均デプス (POS-1)
- DEPTH_RATIO_NEXT     : MQ>=20のcover / BAM平均デプス (POS+1)
"""

import pysam
# BAMより.baiが古い場合に htslib が出す "The index file is older than the
# data file" 警告を抑える（実害は無い。GUIログにエラーのように出るのを防ぐ）。
pysam.set_verbosity(0)
import pandas as pd
import argparse
import joblib
import math
import statistics
from tqdm import tqdm
from pathlib import Path
import re
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==== 共通パラメータ ====
MAPQ_THRESHOLD         = 60
COVER_MAPQ_THRESHOLD   = 20
MEAN_DEPTH_SAMPLE_READS = 500

# INS特徴量専用のMQ閾値のデフォルト（--ins-mq で上書き可能。DEL側のMAPQ_THRESHOLDとは独立）。
# INS_features_v2.py の --mq-mate 相当。MQ>=ins_mq の以上判定。
INS_MQ_DEFAULT = 20

# INS_notTP のみ、通常のINS閾値（threshold_ins、既定0.98）とは別に専用の閾値を使う。
# INS_TP は引き続き threshold_ins をそのまま使う。
INS_NOTTP_THRESHOLD = 1.0

# ==== DEL パラメータ ====
DEL_LOW_MQ_MAX        = 20
DEL_WINDOW_MAX        = 10
# LARGE_INSERT_RATIOの閾値は固定値ではなく、estimate_median_insert_size()で
# サンプルごとに推定したライブラリ中央値insert sizeを使う（DEL_features.py準拠）。

# ==== INS パラメータ（INS_features_v2.py 準拠） ====
INS_RAW_WINDOW_MAX    = 10   # rawデータ出力範囲、かつ CLEAN_END_RATIO/INSERT_SIZE_DIFF の
                              # window集計にも使う範囲（両方ともこれ以下の窓幅なので兼用できる）
INS_CLEAN_END_WINDOW  = 5    # CLEAN_END_RATIO の window（INS_features_v2.py のデフォルトと同じ）
INS_INSERT_DIFF_WINDOW = 10  # INSERT_SIZE_DIFF の window（INS_features_v2.py のデフォルトと同じ）

# ==== 特徴量列定義 ====
DEL_FEATURE_COLS = [
    "HMQ_RATIO",
    "HMQ_RATIO_DELTA",
    "LARGE_INSERT_RATIO",
    "LOW_MQ_RATIO",
    "RIGHT_SOFT_RATIO",
    "LEFT_SOFT_RATIO",
    "DEPTH_RATIO",
]

INS_FEATURE_COLS = [
    "START_RATIO",
    "END_RATIO",
    "INTERNAL_RATIO",
    "SOFTCLIP_LONG_RATIO",
    "SOFTCLIP_SHORT_RATIO",
    "LEFT_SOFT_RATIO",
    "RIGHT_SOFT_RATIO",
    "CLEAN_END_RATIO",
    "INSERT_SIZE_DIFF",
    "MATE_UNMAP_RATIO",
    "MATE_DIFFCHR_RATIO",
    "MATE_DIFFCHR_DIVERSITY_RATIO",
    "DEPTH_RATIO",
    "DEPTH_RATIO_PREV",
    "DEPTH_RATIO_NEXT",
]

DEL_RAW_COLS = [
    "CHROM", "BASE_POS", "WIN_POS",
    "ALL_COVER", "MQ60_COVER",
    "LEFT_SOFT", "RIGHT_SOFT",
    "PAIRED", "LARGE_INSERT", "LOW_MQ_COVER",
]

INS_RAW_COLS = [
    "CHROM", "BASE_POS", "WIN_POS",
    "START", "END", "COVER",
    "SOFTCLIP_LONG", "SOFTCLIP_SHORT",
    "LEFT_SOFT", "RIGHT_SOFT",
    "START_CLEAN", "START_CLIPPED", "END_CLEAN", "END_CLIPPED",
    "ISIZE_DIFF_N",
    "PAIRED",
    "MATE_UNMAPPED", "MATE_DIFFCHR", "MATE_DIFFCHR_NCHROM",
]


# ============================================================
# 共通: BAM平均デプス推定
# ============================================================

def estimate_mean_depth(bam_path: str) -> float:
    import sys
    bam = pysam.AlignmentFile(bam_path, "rb")

    genome_length = sum(bam.lengths)
    if genome_length == 0:
        print("[WARN] Genome length is 0. DEPTH_RATIO will be NaN.", file=sys.stderr)
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
        print("[WARN] Mapped read count is 0. DEPTH_RATIO will be NaN.", file=sys.stderr)
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
        print("[WARN] Failed to sample read lengths. DEPTH_RATIO will be NaN.", file=sys.stderr)
        return float("nan")

    read_lengths.sort()
    median_read_len = read_lengths[len(read_lengths) // 2]
    return (total_mapped * median_read_len) / genome_length


INSERT_SIZE_SAMPLE_READS = 5000  # ライブラリinsert size中央値推定に使うサンプリング数


def estimate_median_insert_size(bam_path: str) -> float:
    """proper pairのTLEN(insert size)からライブラリの中央値insert sizeを推定する
    （INSERT_SIZE_DIFF の基準値。INS_features_v2.py の estimate_median_insert_size()
    と同じロジック）。BAM先頭に偏らないよう、各染色体の先頭/中央/末尾付近から
    サンプリングする。推定失敗時はNaNを返す。"""
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


def collect_cover_mq20_for_positions(bam_path: str, chrom: str, positions: List[int]) -> Dict[int, int]:
    """MAPQ >= COVER_MAPQ_THRESHOLD のcoverカウントを収集する（DEPTH_RATIO用）。"""
    if not positions:
        return {}

    positions = sorted(set(positions))
    pos_set = set(positions)

    cover = {p: 0 for p in positions}

    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam.fetch(chrom, positions[0] - 1, positions[-1]):
        if read.is_unmapped:
            continue
        if read.mapping_quality < COVER_MAPQ_THRESHOLD:
            continue
        if read.reference_start is None or read.reference_end is None:
            continue
        for p0 in read.get_reference_positions(full_length=False):
            p1 = p0 + 1
            if p1 in pos_set:
                cover[p1] += 1
    bam.close()
    return cover


# ============================================================
# 共通: BAM入力展開 / chr正規化 / targets TSV / ヘルパ
# ============================================================

def expand_bam_inputs(inputs: List[str]) -> List[str]:
    bam_files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            # macOS writes AppleDouble sidecar files (e.g. "._sample.bam") on
            # exFAT/FAT32 external drives / network shares -- they match "*.bam"
            # but aren't real BAMs and make pysam fail with "Exec format error".
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


def _strip_chr_prefix(s: str) -> str:
    return re.sub(r"^(chr|CHR|Chr)", "", str(s).strip())

def _normalize_chrom_key(chrom: str) -> str:
    c = str(chrom).strip()
    if c == "":
        return ""
    c = _strip_chr_prefix(c)
    cu = c.upper()
    if cu in ("M", "MT", "MITO", "MITOCHONDRIA", "MITOCHONDRION"):
        return "MT"
    if re.fullmatch(r"\d+", c):
        return str(int(c))
    return cu

def build_bam_chrom_alias(bam_references: List[str]) -> Dict[str, str]:
    def score(ref: str) -> int:
        sc = 0
        if re.match(r"^(chr|CHR|Chr)", str(ref)):
            sc += 10
        if re.search(r"\d{2,}$", str(ref)):
            sc += 1
        return sc

    best: Dict[str, Tuple[int, str]] = {}
    for ref in bam_references:
        k = _normalize_chrom_key(ref)
        if k == "":
            continue
        sc = score(ref)
        if (k not in best) or (sc > best[k][0]):
            best[k] = (sc, ref)
    return {k: v for k, (sc, v) in best.items()}

def map_chrom_to_bam(chrom: str, alias: Dict[str, str]) -> Optional[str]:
    return alias.get(_normalize_chrom_key(chrom), None)

def map_chrom_list_to_bam(chroms: Optional[List[str]], alias: Dict[str, str]) -> Optional[List[str]]:
    if not chroms:
        return None
    out = []
    for c in chroms:
        mapped = map_chrom_to_bam(c, alias)
        if mapped is None:
            print(f"[WARN] --chroms '{c}' not found in BAM references. Skipping.")
            continue
        out.append(mapped)
    return out if out else []

def parse_region(region_str: Optional[str]) -> Tuple[str, Optional[int], Optional[int]]:
    if not region_str:
        return "", None, None
    m = re.fullmatch(r"([^:]+):(\d+)-(\d+)", region_str)
    if m:
        chrom = m.group(1)
        start, end = int(m.group(2)), int(m.group(3))
        if end <= start:
            raise ValueError(f"Invalid region: end ({end}) must be > start ({start})")
        return chrom, start, end
    return region_str, None, None

def parse_region_and_map_to_bam(region_str: Optional[str], alias: Dict[str, str]) -> Tuple[str, Optional[int], Optional[int]]:
    chrom, start, end = parse_region(region_str)
    if not chrom:
        return "", start, end
    mapped = map_chrom_to_bam(chrom, alias)
    if mapped is None:
        print(f"[WARN] --region chrom '{chrom}' not found in BAM references.")
        return chrom, start, end
    return mapped, start, end


def _normalize_colname(s: str) -> str:
    return str(s).strip().lower()

def _detect_targets_columns(tsv_path: str) -> Tuple[str, str, List[str]]:
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\n\r")
    header_cols = header_line.split("\t")
    if len(header_cols) < 2:
        raise ValueError(f"Targets TSV header looks invalid: {tsv_path}")

    cols_norm = {c: _normalize_colname(c) for c in header_cols}
    chrom_col = next((c for c, cn in cols_norm.items() if cn in ("chr", "chrom", "chromosome", "chrom_name")), None)
    pos_col   = next((c for c, cn in cols_norm.items() if cn in ("posi", "pos", "position", "site", "bp")), None)

    if chrom_col is None or pos_col is None:
        raise ValueError(
            f"Targets TSV は chrom列とpos列が必要です。\n"
            f"検出ヘッダー: {header_cols}"
        )
    return chrom_col, pos_col, header_cols

def load_targets_tsv(tsv_path: str) -> pd.DataFrame:
    chrom_col, pos_col, _ = _detect_targets_columns(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", usecols=[chrom_col, pos_col],
                     dtype={chrom_col: "string", pos_col: "string"}, engine="c")
    if df.empty:
        raise ValueError(f"Targets TSV is empty: {tsv_path}")

    out = pd.DataFrame()
    out["CHROM_RAW"] = df[chrom_col].astype(str).str.strip()
    out["POS"] = pd.to_numeric(df[pos_col].astype(str).str.strip(), errors="coerce")
    out = out.dropna(subset=["POS"])
    out["POS"] = out["POS"].astype(int)
    out = out[out["POS"] > 0].drop_duplicates(subset=["CHROM_RAW", "POS"]).reset_index(drop=True)
    return out

def targets_to_pos0_by_chrom(
    bam: pysam.AlignmentFile,
    targets_df: pd.DataFrame,
    alias: Dict[str, str],
    chroms: Optional[List[str]] = None,
    region: Optional[str] = None,
) -> Dict[str, List[int]]:
    t = targets_df.copy()
    t["CHROM"] = t["CHROM_RAW"].map(lambda x: map_chrom_to_bam(x, alias))
    before = len(t)
    t = t.dropna(subset=["CHROM"])
    if len(t) < before:
        print(f"[INFO] targets chrom map: dropped {before - len(t)} row(s) (no match in BAM references)")

    if chroms is not None:
        t = t[t["CHROM"].isin(set(chroms))]
    if region:
        r_chrom, r_start, r_end = parse_region(region)
        if r_start is None or r_end is None:
            t = t[t["CHROM"] == r_chrom]
        else:
            t = t[(t["CHROM"] == r_chrom) & (t["POS"] >= r_start) & (t["POS"] <= r_end)]

    pos_dict: Dict[str, List[int]] = {}
    for chrom, sub in t.groupby("CHROM"):
        chrom_len = bam.get_reference_length(chrom)
        pos0_list = [pos1 - 1 for pos1 in sub["POS"].tolist() if 1 <= pos1 <= chrom_len]
        if pos0_list:
            pos_dict[chrom] = sorted(set(pos0_list))
    return pos_dict


def sample_basename(bam_path: str) -> str:
    name = Path(bam_path).name
    for suf in (".rg.bam", ".sort.bam", ".bam"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name

def dedup_sample_names_upper(bam_paths: List[str]) -> Dict[str, str]:
    seen: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for p in bam_paths:
        base = sample_basename(p).upper()
        n = seen.get(base, 0)
        mapping[p] = base if n == 0 else f"{base}.{n}"
        seen[base] = n + 1
    return mapping

def build_spans_for_scan(bam: pysam.AlignmentFile, chroms=None, region=None) -> List[Tuple[str, int, int]]:
    spans = []
    if region:
        chrom_r, r_start, r_end = parse_region(region)
        if chrom_r not in bam.references:
            print(f"[WARN] {chrom_r} not in BAM references. Skipping.")
            return spans
        chrom_len = bam.get_reference_length(chrom_r)
        r_start = max(0, r_start or 0)
        r_end   = min(chrom_len, r_end or chrom_len)
        if r_end > r_start:
            spans.append((chrom_r, r_start, r_end))
        return spans

    target_refs = chroms if chroms else bam.references
    for chrom in target_refs:
        if chrom not in bam.references:
            print(f"[WARN] {chrom} not in BAM references. Skipping.")
            continue
        spans.append((chrom, 0, bam.get_reference_length(chrom)))
    return spans

def _nan_to_zero(v: float) -> float:
    return 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)

def _safe_div(num: float, den: float) -> float:
    if den == 0 or (isinstance(den, float) and math.isnan(den)):
        return float("nan")
    return num / den

def _is_nan(v: float) -> bool:
    return isinstance(v, float) and math.isnan(v)

def _align_features(X: pd.DataFrame, model, fallback_cols: List[str]) -> pd.DataFrame:
    """Reindex columns into the exact order the model expects (by name if the
    model recorded feature_names_in_ at fit time, otherwise by the hardcoded
    fallback list) and keep it as a DataFrame. Models are now trained on a
    DataFrame (see modelsc/random_meandepth_{DEL,INS}.py), so they have
    feature_names_in_ set -- passing a DataFrame with matching names/order is
    what sklearn expects, and it lets sklearn actually verify the names match
    (raising a clear error if they don't) instead of just trusting position.
    """
    names = getattr(model, "feature_names_in_", None)
    cols = list(names) if names is not None else fallback_cols
    return X.reindex(columns=cols, fill_value=0.0)

def _force_single_threaded_predict(model):
    """学習時に n_jobs=-1 で保存されたRandomForest等は、1行ずつ predict_proba を
    呼ぶ本スクリプトのループ内では木ごとの並列化オーバーヘッド（プロセス/スレッド間
    通信）がボトルネックになる（外側で positions/chrom 単位に並列化しているため、
    内側の木レベル並列は二重並列で無駄が大きい）。予測時は n_jobs=1 に強制する。
    出力される確率値は n_jobs の値に関わらず同一（計算方法が変わるだけ）。
    """
    if hasattr(model, "n_jobs") and model.n_jobs != 1:
        model.n_jobs = 1
    return model

def _maybe_add_prob_ins(row: Dict[str, Any], classes: List[str], probs) -> None:
    cls_set = set(classes)
    if "INS" in cls_set:
        try:
            row["PROB_INS"] = round(float(probs[classes.index("INS")]), 4)
        except Exception:
            pass
        return
    if "INS_TP" in cls_set or "INS_notTP" in cls_set:
        p = sum(float(pr) for c, pr in zip(classes, probs) if c in ("INS_TP", "INS_notTP"))
        row["PROB_INS"] = round(p, 4)


# ============================================================
# DEL特徴量計算
# ============================================================

def _collect_counts_del(bam_path: str, chrom: str, positions: List[int],
                        median_isize: float = float("nan")) -> Dict[int, Dict]:
    """DEL特徴量用カウンタ収集。median_isize: LARGE_INSERT_RATIOの閾値
    （ライブラリ中央値insert size。DEL_features.py準拠、固定値ではない）。"""
    if not positions:
        return {}

    positions = sorted(set(positions))
    pos_set = set(positions)

    counts = {
        p: {"all_cover": 0, "mq60_cover": 0, "low_mq_cover": 0,
            "left_soft": 0, "right_soft": 0, "paired": 0, "large_insert": 0}
        for p in positions
    }

    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam.fetch(chrom, positions[0] - 1, positions[-1]):
        if read.is_unmapped:
            continue
        if read.reference_start is None or read.reference_end is None:
            continue
        if read.reference_end <= read.reference_start:
            continue

        mq = read.mapping_quality
        if read.query_length is None or read.query_length <= 0:
            continue

        is_mq60   = (mq == MAPQ_THRESHOLD)
        is_low_mq = (0 <= mq <= DEL_LOW_MQ_MAX)

        covered = {p0 + 1 for p0 in read.get_reference_positions(full_length=False) if p0 + 1 in pos_set}
        if not covered:
            continue

        left_soft = right_soft = 0
        large_insert = False
        if is_mq60:
            cigar = read.cigartuples
            if cigar:
                if cigar[0][0] == 4:
                    left_soft = cigar[0][1]
                if cigar[-1][0] == 4:
                    right_soft = cigar[-1][1]
            if read.is_paired:
                same_chr = (not read.mate_is_unmapped) and (read.next_reference_id == read.reference_id)
                if same_chr and not math.isnan(median_isize):
                    large_insert = abs(read.template_length) >= median_isize

        for pos in covered:
            c = counts[pos]
            c["all_cover"] += 1
            if is_mq60:
                c["mq60_cover"] += 1
                if left_soft > 0:
                    c["left_soft"] += 1
                if right_soft > 0:
                    c["right_soft"] += 1
                if read.is_paired:
                    c["paired"] += 1
                    if large_insert:
                        c["large_insert"] += 1
            if is_low_mq:
                c["low_mq_cover"] += 1

    bam.close()
    return counts


def _hmq_ratio_at(pos_counts: Dict[int, Dict], pos: int, mean_depth: float) -> float:
    if pos not in pos_counts:
        return float("nan")
    return _safe_div(pos_counts[pos]["mq60_cover"], mean_depth)

def _calc_hmq_ratio_delta(pos_counts: Dict[int, Dict], center_pos: int, mean_depth: float) -> float:
    def mean_ratio(positions):
        vals = [_hmq_ratio_at(pos_counts, p, mean_depth) for p in positions if p in pos_counts]
        vals = [v for v in vals if not _is_nan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    before = mean_ratio(range(center_pos - DEL_WINDOW_MAX, center_pos))
    after  = mean_ratio(range(center_pos + 1, center_pos + DEL_WINDOW_MAX + 1))
    if _is_nan(before) or _is_nan(after):
        return float("nan")
    return before - after

def _large_insert_ratio_at_del(pos_counts: Dict[int, Dict], center_pos: int, window: int,
                               median_isize: float) -> float:
    """insert>=median_isize のリード数合計 / pairedリード数合計（±window bp, MQ60のみ）。
    median_isize がNaN（推定失敗）のときはNaNを返す（DEL_features.py準拠）。"""
    if math.isnan(median_isize):
        return float("nan")
    large_sum = paired_sum = 0
    for p in range(center_pos - window, center_pos + window + 1):
        if p in pos_counts:
            c = pos_counts[p]
            large_sum  += c["large_insert"]
            paired_sum += c["paired"]
    return _safe_div(large_sum, paired_sum)

def _low_mq_ratio_at(pos_counts: Dict[int, Dict], pos: int, mean_depth: float) -> float:
    if pos not in pos_counts:
        return float("nan")
    return _safe_div(pos_counts[pos]["low_mq_cover"], mean_depth)

def _soft_side_ratio_window_del(pos_counts: Dict[int, Dict], center_pos: int, window: int, side_key: str) -> float:
    side_sum = total_sum = 0
    for p in range(center_pos - window, center_pos + window + 1):
        if p in pos_counts:
            c = pos_counts[p]
            side_sum  += c[side_key]
            total_sum += c["left_soft"] + c["right_soft"]
    return _safe_div(side_sum, total_sum)

def _features_del(bam_path: str, chrom: str, pos1: int, mean_depth: float,
                  median_isize: float = float("nan")) -> Tuple[Dict[str, float], Dict[int, Dict]]:
    """DEL特徴量を1ポジション計算する（1-based pos1）。"""
    all_positions = [p for p in range(pos1 - DEL_WINDOW_MAX, pos1 + DEL_WINDOW_MAX + 1) if p >= 1]
    pos_counts = _collect_counts_del(bam_path, chrom, all_positions, median_isize)

    hmq_ratio          = _hmq_ratio_at(pos_counts, pos1, mean_depth)
    hmq_ratio_delta    = _calc_hmq_ratio_delta(pos_counts, pos1, mean_depth)
    large_insert_ratio = _large_insert_ratio_at_del(pos_counts, pos1, DEL_WINDOW_MAX, median_isize)
    low_mq_ratio       = _low_mq_ratio_at(pos_counts, pos1, mean_depth)
    right_soft_ratio   = _soft_side_ratio_window_del(pos_counts, pos1, 5, "right_soft")
    left_soft_ratio    = _soft_side_ratio_window_del(pos_counts, pos1, 5, "left_soft")

    cover_mq20 = collect_cover_mq20_for_positions(bam_path, chrom, [pos1]).get(pos1, 0)
    depth_ratio = _safe_div(cover_mq20, mean_depth) if not _is_nan(mean_depth) else float("nan")

    feats = {
        "HMQ_RATIO":          _nan_to_zero(hmq_ratio),
        "HMQ_RATIO_DELTA":    _nan_to_zero(hmq_ratio_delta),
        "LARGE_INSERT_RATIO": _nan_to_zero(large_insert_ratio),
        "LOW_MQ_RATIO":       _nan_to_zero(low_mq_ratio),
        "RIGHT_SOFT_RATIO":   _nan_to_zero(right_soft_ratio),
        "LEFT_SOFT_RATIO":    _nan_to_zero(left_soft_ratio),
        "DEPTH_RATIO":        _nan_to_zero(depth_ratio),
    }
    return feats, pos_counts


def _build_raw_rows_del(chrom: str, pos1: int, pos_counts: Dict[int, Dict]) -> List[Dict]:
    rows = []
    for dp in range(-DEL_WINDOW_MAX, DEL_WINDOW_MAX + 1):
        p = pos1 + dp
        c = pos_counts.get(p)
        if c is None:
            row = {k: 0 for k in DEL_RAW_COLS}
            row["CHROM"] = chrom; row["BASE_POS"] = pos1; row["WIN_POS"] = p
        else:
            row = {"CHROM": chrom, "BASE_POS": pos1, "WIN_POS": p,
                   "ALL_COVER": c["all_cover"], "MQ60_COVER": c["mq60_cover"],
                   "LEFT_SOFT": c["left_soft"], "RIGHT_SOFT": c["right_soft"],
                   "PAIRED": c["paired"], "LARGE_INSERT": c["large_insert"],
                   "LOW_MQ_COVER": c["low_mq_cover"]}
        rows.append(row)
    return rows


# ============================================================
# INS特徴量計算
# ============================================================

def _collect_counts_ins(bam_path: str, chrom: str, positions: List[int],
                        ins_mq: int = INS_MQ_DEFAULT,
                        median_isize: float = float("nan")) -> Dict[int, Dict]:
    """INS特徴量用カウンタ収集（MQ >= ins_mq のリードのみ対象。INS_features_v2.py
    の collect_counts_for_positions() と同じロジック -- LARGE/SMALL_INSERT_RATIO用の
    集計は廃止し、代わりに CLEAN_END_RATIO / INSERT_SIZE_DIFF 用の集計を行う）。"""
    if not positions:
        return {}

    positions = sorted(set(positions))
    pos_set = set(positions)

    counts = {
        p: {"start": 0, "end": 0, "cover": 0,
            "softclip_long": 0, "softclip_short": 0,
            "left_soft": 0, "right_soft": 0,
            "start_clean": 0, "start_clipped": 0,
            "end_clean": 0, "end_clipped": 0,
            "isize_diffs": [],
            "paired": 0,
            "mate_unmapped": 0, "mate_diffchr": 0,
            "mate_diffchr_chroms": set()}
        for p in positions
    }

    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam.fetch(chrom, positions[0] - 1, positions[-1]):
        if read.is_unmapped:
            continue
        if read.mapping_quality < ins_mq:
            continue
        if read.reference_start is None or read.reference_end is None:
            continue
        if read.reference_end <= read.reference_start:
            continue

        read_len = read.query_length
        if read_len is None or read_len <= 0:
            continue

        ref_start_1 = read.reference_start + 1
        ref_end_1   = read.reference_end

        covered = {p0 + 1 for p0 in read.get_reference_positions(full_length=False) if p0 + 1 in pos_set}
        if not covered and ref_start_1 not in pos_set and ref_end_1 not in pos_set:
            continue

        left_soft = right_soft = 0
        cigar = read.cigartuples
        if cigar:
            if cigar[0][0] == 4:
                left_soft = cigar[0][1]
            if cigar[-1][0] == 4:
                right_soft = cigar[-1][1]
        max_soft = max(left_soft, right_soft)
        has_soft = (left_soft > 0 or right_soft > 0)
        is_softclip_long  = has_soft and (max_soft * 2 >= read_len)
        is_softclip_short = has_soft and (max_soft * 2 <  read_len)

        is_paired      = read.is_paired
        mate_unmapped  = read.mate_is_unmapped if is_paired else False
        mate_diffchr   = (is_paired and not mate_unmapped
                          and read.next_reference_id != read.reference_id)
        mate_diffchr_chrom = read.next_reference_name if mate_diffchr else None

        # INSERT_SIZE_DIFF用: ライブラリ中央値insert sizeからのズレ
        isize_diff = None
        if (is_paired and not mate_unmapped
                and read.next_reference_id == read.reference_id
                and read.template_length != 0
                and not math.isnan(median_isize)):
            isize_diff = abs(read.template_length) - median_isize

        for pos in covered:
            c = counts[pos]
            c["cover"] += 1
            if is_softclip_long:
                c["softclip_long"] += 1
            elif is_softclip_short:
                c["softclip_short"] += 1
            if left_soft > 0:
                c["left_soft"] += 1
            if right_soft > 0:
                c["right_soft"] += 1
            if is_paired:
                c["paired"] += 1
                if mate_unmapped:
                    c["mate_unmapped"] += 1
                if mate_diffchr:
                    c["mate_diffchr"] += 1
                    c["mate_diffchr_chroms"].add(mate_diffchr_chrom)

        # CLEAN_END_RATIO / INSERT_SIZE_DIFF: そのポジションで終わる/始まるリードを判定
        if ref_start_1 in pos_set:
            counts[ref_start_1]["start"] += 1
            if left_soft > 0:
                counts[ref_start_1]["start_clipped"] += 1
            else:
                counts[ref_start_1]["start_clean"] += 1
            if isize_diff is not None:
                counts[ref_start_1]["isize_diffs"].append(isize_diff)
        if ref_end_1 in pos_set:
            counts[ref_end_1]["end"] += 1
            if right_soft > 0:
                counts[ref_end_1]["end_clipped"] += 1
            else:
                counts[ref_end_1]["end_clean"] += 1
            if isize_diff is not None:
                counts[ref_end_1]["isize_diffs"].append(isize_diff)

    bam.close()
    return counts


def _point_ratio_by_depth_ins(pos_counts: Dict[int, Dict], pos: int, num_key: str, mean_depth: float) -> float:
    if pos not in pos_counts:
        return float("nan")
    return _safe_div(pos_counts[pos][num_key], mean_depth)

def _internal_ratio_by_depth_ins(pos_counts: Dict[int, Dict], pos: int, mean_depth: float) -> float:
    if pos not in pos_counts:
        return float("nan")
    c = pos_counts[pos]
    internal = max(0, c["cover"] - c["start"] - c["end"])
    return _safe_div(internal, mean_depth)

def _soft_side_ratio_ins(pos_counts: Dict[int, Dict], pos: int, side_key: str) -> float:
    if pos not in pos_counts:
        return float("nan")
    c = pos_counts[pos]
    total_soft = c["left_soft"] + c["right_soft"]
    return _safe_div(c[side_key], total_soft)

def _point_ratio_ins(pos_counts: Dict[int, Dict], pos: int, num_key: str, den_key: str) -> float:
    if pos not in pos_counts:
        return float("nan")
    c = pos_counts[pos]
    return _safe_div(c[num_key], c[den_key])

def _mate_diffchr_diversity_ratio_ins(pos_counts: Dict[int, Dict], pos: int,
                                      total_chrom_count: Optional[int]) -> float:
    """MATE_DIFFCHR_DIVERSITY_RATIO = mateが張り付いている他染色体の種類数 / BAMの染色体総数。
    total_chrom_countが無効(None/0以下)ならNaN。"""
    if pos not in pos_counts:
        return float("nan")
    if total_chrom_count is None or total_chrom_count <= 0:
        return float("nan")
    n_chroms = len(pos_counts[pos]["mate_diffchr_chroms"])
    return n_chroms / total_chrom_count

def _window_sum_ins(pos_counts: Dict[int, Dict], center_pos: int, window: int, key: str):
    total = 0
    for p in range(center_pos - window, center_pos + window + 1):
        if p in pos_counts:
            total += pos_counts[p][key]
    return total

def _window_concat_list_ins(pos_counts: Dict[int, Dict], center_pos: int, window: int, key: str):
    out = []
    for p in range(center_pos - window, center_pos + window + 1):
        if p in pos_counts:
            out.extend(pos_counts[p][key])
    return out

def _clean_end_ratio_ins(pos_counts: Dict[int, Dict], pos: int, window: int = INS_CLEAN_END_WINDOW) -> float:
    """CLEAN_END_RATIO = (start_clean+end_clean) / (境界リード総数)  (pos±windowで合算)。
    境界リードが0本ならNaN。INS_features_v2.py の clean_end_ratio() と同じ。"""
    clean = (_window_sum_ins(pos_counts, pos, window, "start_clean")
             + _window_sum_ins(pos_counts, pos, window, "end_clean"))
    clipped = (_window_sum_ins(pos_counts, pos, window, "start_clipped")
               + _window_sum_ins(pos_counts, pos, window, "end_clipped"))
    total = clean + clipped
    if total == 0:
        return float("nan")
    return clean / total

def _insert_size_diff_median_ins(pos_counts: Dict[int, Dict], pos: int, window: int = INS_INSERT_DIFF_WINDOW) -> float:
    """INSERT_SIZE_DIFF = pos±windowで終わる/始まるペアリードのinsert sizeと
    ライブラリ中央値との差の中央値。サンプルが0件ならNaN。
    INS_features_v2.py の insert_size_diff_median() と同じ。"""
    diffs = _window_concat_list_ins(pos_counts, pos, window, "isize_diffs")
    if not diffs:
        return float("nan")
    return statistics.median(diffs)


def _features_ins(bam_path: str, chrom: str, pos1: int,
                  mean_depth: float, ins_mq: int = INS_MQ_DEFAULT,
                  total_chrom_count: Optional[int] = None,
                  median_isize: float = float("nan")) -> Tuple[Dict[str, float], Dict[int, Dict]]:
    """INS特徴量を1ポジション計算する（1-based pos1）。CLEAN_END_RATIO/INSERT_SIZE_DIFF
    のみ INS_features_v2.py のデフォルト通り pos1±window で集計し、他は従来通り
    pos1単独（window=0相当）で計算する。"""
    all_positions = [p for p in range(pos1 - INS_RAW_WINDOW_MAX, pos1 + INS_RAW_WINDOW_MAX + 1) if p >= 1]
    pos_counts = _collect_counts_ins(bam_path, chrom, all_positions, ins_mq, median_isize)

    center = {pos1: pos_counts[pos1]} if pos1 in pos_counts else {}

    start_ratio          = _point_ratio_by_depth_ins(center, pos1, "start",         mean_depth)
    end_ratio            = _point_ratio_by_depth_ins(center, pos1, "end",            mean_depth)
    internal_ratio       = _internal_ratio_by_depth_ins(center, pos1, mean_depth)
    softclip_long_ratio  = _point_ratio_by_depth_ins(center, pos1, "softclip_long",  mean_depth)
    softclip_short_ratio = _point_ratio_by_depth_ins(center, pos1, "softclip_short", mean_depth)
    left_soft_ratio      = _soft_side_ratio_ins(center, pos1, "left_soft")
    right_soft_ratio     = _soft_side_ratio_ins(center, pos1, "right_soft")
    # CLEAN_END_RATIO / INSERT_SIZE_DIFF は pos_counts 全体（±INS_RAW_WINDOW_MAX分）から
    # それぞれの window で集計する -- center（pos1単独）ではなく pos_counts を渡す。
    clean_end_ratio_v   = _clean_end_ratio_ins(pos_counts, pos1)
    insert_size_diff_v  = _insert_size_diff_median_ins(pos_counts, pos1)
    mate_unmap_ratio     = _point_ratio_ins(center, pos1, "mate_unmapped", "paired")
    mate_diffchr_ratio   = _point_ratio_ins(center, pos1, "mate_diffchr",  "paired")
    mate_diffchr_diversity_ratio = _mate_diffchr_diversity_ratio_ins(center, pos1, total_chrom_count)

    mq20_positions = [p for p in [pos1 - 1, pos1, pos1 + 1] if p >= 1]
    cover_mq20_dict = collect_cover_mq20_for_positions(bam_path, chrom, mq20_positions)

    def _depth_ratio_at(p: int) -> float:
        if _is_nan(mean_depth) or mean_depth <= 0:
            return float("nan")
        return cover_mq20_dict.get(p, 0) / mean_depth

    feats = {
        "START_RATIO":           _nan_to_zero(start_ratio),
        "END_RATIO":             _nan_to_zero(end_ratio),
        "INTERNAL_RATIO":        _nan_to_zero(internal_ratio),
        "SOFTCLIP_LONG_RATIO":   _nan_to_zero(softclip_long_ratio),
        "SOFTCLIP_SHORT_RATIO":  _nan_to_zero(softclip_short_ratio),
        "LEFT_SOFT_RATIO":       _nan_to_zero(left_soft_ratio),
        "RIGHT_SOFT_RATIO":      _nan_to_zero(right_soft_ratio),
        "CLEAN_END_RATIO":       _nan_to_zero(clean_end_ratio_v),
        "INSERT_SIZE_DIFF":      _nan_to_zero(insert_size_diff_v),
        "MATE_UNMAP_RATIO":      _nan_to_zero(mate_unmap_ratio),
        "MATE_DIFFCHR_RATIO":    _nan_to_zero(mate_diffchr_ratio),
        "MATE_DIFFCHR_DIVERSITY_RATIO": _nan_to_zero(mate_diffchr_diversity_ratio),
        "DEPTH_RATIO":           _nan_to_zero(_depth_ratio_at(pos1)),
        "DEPTH_RATIO_PREV":      _nan_to_zero(_depth_ratio_at(pos1 - 1) if pos1 - 1 >= 1 else float("nan")),
        "DEPTH_RATIO_NEXT":      _nan_to_zero(_depth_ratio_at(pos1 + 1)),
    }
    return feats, pos_counts


def _build_raw_rows_ins(chrom: str, pos1: int, pos_counts: Dict[int, Dict]) -> List[Dict]:
    rows = []
    for dp in range(-INS_RAW_WINDOW_MAX, INS_RAW_WINDOW_MAX + 1):
        p = pos1 + dp
        c = pos_counts.get(p)
        if c is None:
            row = {k: 0 for k in INS_RAW_COLS}
            row["CHROM"] = chrom; row["BASE_POS"] = pos1; row["WIN_POS"] = p
        else:
            row = {"CHROM": chrom, "BASE_POS": pos1, "WIN_POS": p,
                   "START": c["start"], "END": c["end"], "COVER": c["cover"],
                   "SOFTCLIP_LONG": c["softclip_long"], "SOFTCLIP_SHORT": c["softclip_short"],
                   "LEFT_SOFT": c["left_soft"], "RIGHT_SOFT": c["right_soft"],
                   "START_CLEAN": c["start_clean"], "START_CLIPPED": c["start_clipped"],
                   "END_CLEAN": c["end_clean"], "END_CLIPPED": c["end_clipped"],
                   "ISIZE_DIFF_N": len(c["isize_diffs"]),
                   "PAIRED": c["paired"],
                   "MATE_UNMAPPED": c["mate_unmapped"], "MATE_DIFFCHR": c["mate_diffchr"],
                   "MATE_DIFFCHR_NCHROM": len(c["mate_diffchr_chroms"])}
        rows.append(row)
    return rows


# ============================================================
# 並列チャンク処理（DEL/INSを同一ポジションループ内で計算）
# ============================================================

def _load_del_ins_models(args: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """個別joblib（--model-del/--model-ins）からモデルをロードする。"""
    model_del_path = args.get("model_del_path")
    model_ins_path = args.get("model_ins_path")
    model_del = _force_single_threaded_predict(joblib.load(model_del_path)) if model_del_path else None
    model_ins = _force_single_threaded_predict(joblib.load(model_ins_path)) if model_ins_path else None
    return model_del, model_ins


def _process_chunk_combined(args: Dict[str, Any]) -> Dict[str, List]:
    """1チャンク（1染色体分のポジション列）についてDEL/INS両方を1パスで処理する。

    DEL特徴量とモデル、INS特徴量とモデルはそれぞれ従来通り完全に独立したまま、
    ポジションループとモデルロードだけを共有することで、DEL/INS個別スキャンだった
    従来実装に比べてループ回数・モデルロード回数を半分にする。
    出力の行フォーマット（CHROM,POS,TYPE,PROB,PROB_<class>）はDEL/INSそれぞれ従来通り。
    """
    bam_path   = args["bam_path"]
    chrom      = args["chrom"]
    positions0 = args["positions0"]
    threshold_del = args["threshold_del"]
    threshold_ins = args["threshold_ins"]
    emit_n     = args.get("emit_n_for_targets", False)
    mean_depth = args.get("mean_depth", float("nan"))
    ins_mq     = args.get("ins_mq", INS_MQ_DEFAULT)
    total_chrom_count = args.get("total_chrom_count")
    median_isize = args.get("median_isize", float("nan"))

    model_del, model_ins = _load_del_ins_models(args)
    del_classes = list(model_del.classes_) if model_del is not None else None
    ins_classes = list(model_ins.classes_) if model_ins is not None else None

    results_del = []
    results_ins = []
    results_del_feat = []
    results_ins_feat = []
    results_del_raw  = []
    results_ins_raw  = []

    for pos0 in positions0:
        pos1 = pos0 + 1

        if model_del is not None:
            feats, pos_counts = _features_del(bam_path, chrom, pos1, mean_depth, median_isize)
            feat_row = {"CHROM": chrom, "POS": pos1}
            feat_row.update(feats)
            results_del_feat.append(feat_row)
            results_del_raw.extend(_build_raw_rows_del(chrom, pos1, pos_counts))

            X = _align_features(pd.DataFrame([feats]), model_del, DEL_FEATURE_COLS)
            probs = model_del.predict_proba(X)[0]
            idx   = probs.argmax()
            label = str(del_classes[idx])
            prob  = float(probs[idx])

            if prob >= threshold_del:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": label, "PROB": round(prob, 4)}
                for cls, p in zip(del_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_del.append(row)
            elif emit_n:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": "N", "PROB": round(prob, 4)}
                for cls, p in zip(del_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_del.append(row)

        if model_ins is not None:
            feats, pos_counts = _features_ins(bam_path, chrom, pos1, mean_depth, ins_mq, total_chrom_count, median_isize)
            feat_row = {"CHROM": chrom, "POS": pos1}
            feat_row.update(feats)
            results_ins_feat.append(feat_row)
            results_ins_raw.extend(_build_raw_rows_ins(chrom, pos1, pos_counts))

            X = _align_features(pd.DataFrame([feats]), model_ins, INS_FEATURE_COLS)
            probs = model_ins.predict_proba(X)[0]
            idx   = probs.argmax()
            label = str(ins_classes[idx])
            prob  = float(probs[idx])
            threshold_ins_eff = INS_NOTTP_THRESHOLD if label == "INS_notTP" else threshold_ins

            if prob >= threshold_ins_eff:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": label, "PROB": round(prob, 4)}
                for cls, p in zip(ins_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_ins.append(row)
            elif emit_n:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": "N", "PROB": round(prob, 4)}
                for cls, p in zip(ins_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_ins.append(row)

    return {
        "del": results_del,
        "ins": results_ins,
        "del_features": results_del_feat,
        "ins_features": results_ins_feat,
        "del_raw": results_del_raw,
        "ins_raw": results_ins_raw,
    }


# ============================================================
# 1BAMの予測実行（DEL / INS を同一ループで計算、出力は従来通り別ファイル）
# ============================================================

def _run_combined_prediction(
    bam_path: str,
    model_del,
    model_ins,
    model_del_path: Optional[str],
    model_ins_path: Optional[str],
    positions_by_chrom: Optional[Dict[str, List[int]]],
    spans_scan: Optional[List[Tuple[str, int, int]]],
    mean_depth: float,
    step: int,
    threshold_del: float,
    threshold_ins: float,
    n_jobs: int,
    out_del: Optional[str],
    out_ins: Optional[str],
    out_del_feat: Optional[str],
    out_ins_feat: Optional[str],
    out_del_raw: Optional[str],
    out_ins_raw: Optional[str],
    ins_mq: int = INS_MQ_DEFAULT,
    total_chrom_count: Optional[int] = None,
    median_isize: float = float("nan"),
):
    """DEL/INSを同一のBAM走査ループで予測する。

    特徴量計算・モデルはDEL/INSで完全に独立したまま、ポジションのスキャン
    （全ゲノム or targets）とモデルロードを1回にまとめることで、従来の
    DEL単独スキャン→INS単独スキャンの2回実行に比べて処理時間を短縮する。
    出力TSV（予測・特徴量・raw）は従来通りDEL/INSそれぞれ別ファイルのまま。
    """
    del_classes = list(model_del.classes_) if model_del is not None else None
    ins_classes = list(model_ins.classes_) if model_ins is not None else None
    del_feat_order = ["CHROM", "POS"] + DEL_FEATURE_COLS
    ins_feat_order = ["CHROM", "POS"] + INS_FEATURE_COLS

    if model_del is not None:
        print(f"  [DEL] classes: {del_classes}")
        print(f"  [DEL] features: {DEL_FEATURE_COLS}")
    if model_ins is not None:
        print(f"  [INS] classes: {ins_classes}")
        print(f"  [INS] features: {INS_FEATURE_COLS}")

    results_del = []
    results_ins = []
    results_del_feat = []
    results_ins_feat = []
    results_del_raw  = []
    results_ins_raw  = []

    emit_n = (positions_by_chrom is not None)

    def _run_at_pos(chrom: str, pos0: int) -> None:
        pos1 = pos0 + 1

        if model_del is not None:
            feats, pos_counts = _features_del(bam_path, chrom, pos1, mean_depth, median_isize)
            feat_row = {"CHROM": chrom, "POS": pos1}
            feat_row.update(feats)
            results_del_feat.append(feat_row)
            results_del_raw.extend(_build_raw_rows_del(chrom, pos1, pos_counts))

            X = _align_features(pd.DataFrame([feats]), model_del, DEL_FEATURE_COLS)
            probs = model_del.predict_proba(X)[0]
            idx   = probs.argmax()
            label = str(del_classes[idx])
            prob  = float(probs[idx])

            if prob >= threshold_del:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": label, "PROB": round(prob, 4)}
                for cls, p in zip(del_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_del.append(row)
            elif emit_n:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": "N", "PROB": round(prob, 4)}
                for cls, p in zip(del_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_del.append(row)

        if model_ins is not None:
            feats, pos_counts = _features_ins(bam_path, chrom, pos1, mean_depth, ins_mq, total_chrom_count, median_isize)
            feat_row = {"CHROM": chrom, "POS": pos1}
            feat_row.update(feats)
            results_ins_feat.append(feat_row)
            results_ins_raw.extend(_build_raw_rows_ins(chrom, pos1, pos_counts))

            X = _align_features(pd.DataFrame([feats]), model_ins, INS_FEATURE_COLS)
            probs = model_ins.predict_proba(X)[0]
            idx   = probs.argmax()
            label = str(ins_classes[idx])
            prob  = float(probs[idx])
            threshold_ins_eff = INS_NOTTP_THRESHOLD if label == "INS_notTP" else threshold_ins

            if prob >= threshold_ins_eff:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": label, "PROB": round(prob, 4)}
                for cls, p in zip(ins_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_ins.append(row)
            elif emit_n:
                row = {"CHROM": chrom, "POS": pos1, "TYPE": "N", "PROB": round(prob, 4)}
                for cls, p in zip(ins_classes, probs):
                    row[f"PROB_{cls}"] = round(float(p), 4)
                results_ins.append(row)

    def _collect_chunk_result(result: Dict[str, List]) -> None:
        results_del.extend(result["del"])
        results_ins.extend(result["ins"])
        results_del_feat.extend(result["del_features"])
        results_ins_feat.extend(result["ins_features"])
        results_del_raw.extend(result["del_raw"])
        results_ins_raw.extend(result["ins_raw"])

    def _build_task(chrom: str, positions0: List[int], emit_n_for_targets: bool) -> Dict[str, Any]:
        return {
            "bam_path": bam_path, "chrom": chrom, "positions0": positions0,
            "threshold_del": threshold_del, "threshold_ins": threshold_ins,
            "emit_n_for_targets": emit_n_for_targets, "mean_depth": mean_depth,
            "model_del_path": model_del_path, "model_ins_path": model_ins_path,
            "ins_mq": ins_mq, "total_chrom_count": total_chrom_count, "median_isize": median_isize,
        }

    if positions_by_chrom is not None:
        if n_jobs == 1:
            for chrom in sorted(positions_by_chrom.keys()):
                pos_list0 = positions_by_chrom[chrom]
                for pos0 in tqdm(pos_list0, desc=f"  [DEL+INS] Targets {chrom} ({len(pos_list0)} pos)"):
                    _run_at_pos(chrom, pos0)
        else:
            tasks = [
                _build_task(chrom, pos_list0, True)
                for chrom, pos_list0 in positions_by_chrom.items()
            ]
            print(f"  [DEL+INS] Running {len(tasks)} task(s) in parallel with {n_jobs} process(es) (targets mode)...")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(_process_chunk_combined, t): t for t in tasks}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="  [DEL+INS] Processing"):
                    try:
                        _collect_chunk_result(future.result())
                    except Exception as e:
                        print(f"[ERROR] {e}")
    else:
        if n_jobs == 1:
            for chrom, r_start, r_end in spans_scan:
                for pos0 in tqdm(range(r_start, r_end, step), desc=f"  [DEL+INS] Scanning {chrom}"):
                    _run_at_pos(chrom, pos0)
        else:
            tasks = [
                _build_task(chrom, list(range(r_start, r_end, step)), False)
                for chrom, r_start, r_end in spans_scan
            ]
            print(f"  [DEL+INS] Running {len(tasks)} task(s) in parallel with {n_jobs} process(es) (full scan)...")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(_process_chunk_combined, t): t for t in tasks}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="  [DEL+INS] Processing"):
                    try:
                        _collect_chunk_result(future.result())
                    except Exception as e:
                        print(f"[ERROR] {e}")

    # 予測TSV出力（DEL/INSそれぞれ従来通り別ファイル）
    if out_del is not None:
        out_df = pd.DataFrame(results_del)
        if not out_df.empty:
            out_df = out_df.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)
        out_df.to_csv(out_del, sep="\t", index=False)
        print(f"  ✓ Saved [DEL]: {out_del} ({len(out_df)} predictions)")

    if out_ins is not None:
        out_df = pd.DataFrame(results_ins)
        if not out_df.empty:
            out_df = out_df.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)
        out_df.to_csv(out_ins, sep="\t", index=False)
        print(f"  ✓ Saved [INS]: {out_ins} ({len(out_df)} predictions)")

    # 特徴量TSV出力
    if out_del_feat:
        df_feat = pd.DataFrame(results_del_feat)
        if not df_feat.empty:
            df_feat = df_feat.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)
            df_feat = df_feat[[c for c in del_feat_order if c in df_feat.columns]]
        df_feat.to_csv(out_del_feat, sep="\t", index=False)
        print(f"  ✓ Saved [DEL] features: {out_del_feat} ({len(df_feat)} positions)")

    if out_ins_feat:
        df_feat = pd.DataFrame(results_ins_feat)
        if not df_feat.empty:
            df_feat = df_feat.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)
            df_feat = df_feat[[c for c in ins_feat_order if c in df_feat.columns]]
        df_feat.to_csv(out_ins_feat, sep="\t", index=False)
        print(f"  ✓ Saved [INS] features: {out_ins_feat} ({len(df_feat)} positions)")

    if out_del_raw:
        df_raw = pd.DataFrame(results_del_raw, columns=DEL_RAW_COLS)
        if not df_raw.empty:
            df_raw = df_raw.sort_values(by=["CHROM", "BASE_POS", "WIN_POS"]).reset_index(drop=True)
        df_raw.to_csv(out_del_raw, sep="\t", index=False)
        print(f"  ✓ Saved [DEL] raw: {out_del_raw} ({len(df_raw)} rows)")

    if out_ins_raw:
        df_raw = pd.DataFrame(results_ins_raw, columns=INS_RAW_COLS)
        if not df_raw.empty:
            df_raw = df_raw.sort_values(by=["CHROM", "BASE_POS", "WIN_POS"]).reset_index(drop=True)
        df_raw.to_csv(out_ins_raw, sep="\t", index=False)
        print(f"  ✓ Saved [INS] raw: {out_ins_raw} ({len(df_raw)} rows)")


# ============================================================
# メイン: 1BAM処理
# ============================================================

def predict_combined(
    bam_path: str,
    model_del,
    model_ins,
    model_del_path: Optional[str],
    model_ins_path: Optional[str],
    out_del: Optional[str],
    out_ins: Optional[str],
    out_del_feat: Optional[str],
    out_ins_feat: Optional[str],
    out_del_raw: Optional[str],
    out_ins_raw: Optional[str],
    step: int,
    threshold_del: float,
    threshold_ins: float,
    target_chroms: Optional[List[str]],
    region: Optional[str],
    n_jobs: int,
    targets_tsv: Optional[str],
    precomputed_mean_depth: Optional[float] = None,
    ins_mq: int = INS_MQ_DEFAULT,
    precomputed_median_isize: Optional[float] = None,
):
    print(f"\n[{Path(bam_path).name}] Starting prediction...")

    if precomputed_mean_depth is not None and not math.isnan(precomputed_mean_depth):
        # 事前計算済みの値（--mean-depth-tsv）を使う。領域抽出済みBAMを渡す場合、
        # このBAM自身から再計算すると total_mapped が激減する一方で genome_length は
        # ヘッダーの宣言値のまま変わらないため、mean_depthが実際より大幅に小さく
        # 見積もられ、DEPTH_RATIO系の特徴量が全て壊れる。必ず全長BAMに対して
        # estimate_mean_depth.py で事前計算した値をここで再利用する。
        mean_depth = precomputed_mean_depth
        print(f"  [INFO] Mean depth (using precomputed value): {mean_depth:.2f}x")
    else:
        print(f"  [INFO] Estimating mean depth...")
        mean_depth = estimate_mean_depth(bam_path)
        if not math.isnan(mean_depth):
            print(f"  [INFO] Estimated mean depth: {mean_depth:.2f}x")
        else:
            print("  [INFO] Estimated mean depth: NaN")

    if precomputed_median_isize is not None and not math.isnan(precomputed_median_isize):
        # mean_depth と同じ理由（領域抽出済みBAMでは推定に必要な読み取りの分布が
        # 失われる）で、必ず全長BAMに対して事前計算した値をここで再利用する。
        median_isize = precomputed_median_isize
        print(f"  [INFO] Library insert size (using precomputed value): {median_isize:.1f}bp")
    else:
        print(f"  [INFO] Estimating library insert size...")
        median_isize = estimate_median_insert_size(bam_path)
        if not math.isnan(median_isize):
            print(f"  [INFO] Estimated median insert size: {median_isize:.1f}bp")
        else:
            print("  [INFO] Estimated median insert size: NaN")

    bam = pysam.AlignmentFile(bam_path, "rb")
    alias = build_bam_chrom_alias(list(bam.references))
    print(f"  [INFO] BAM references sample: {list(bam.references)[:5]} (n={len(bam.references)})")

    # MATE_DIFFCHR_DIVERSITY_RATIO の分母（BAMヘッダの@SQ行数 = アラインメントに
    # 使用したfastaの染色体総数）。1_bam_choice.py の領域抽出は元BAMのヘッダを
    # そのままコピーして書き出すため、±10kb抽出済みBAMでもこの値は元の全長BAMと
    # 一致する（実データで確認済み）。genome_length/mean_depthのケースと異なり、
    # 事前計算・全長BAMへのフォールバックは不要。
    total_chrom_count = bam.nreferences

    mapped_chroms = map_chrom_list_to_bam(target_chroms, alias) if target_chroms else None
    mapped_region = None
    if region:
        rchrom, rstart, rend = parse_region_and_map_to_bam(region, alias)
        mapped_region = rchrom if (rstart is None or rend is None) else f"{rchrom}:{rstart}-{rend}"

    positions_by_chrom: Optional[Dict[str, List[int]]] = None
    spans_scan: Optional[List[Tuple[str, int, int]]] = None

    if targets_tsv:
        print(f"  [INFO] targets TSV: {targets_tsv}")
        tdf = load_targets_tsv(targets_tsv)
        positions_by_chrom = targets_to_pos0_by_chrom(bam, tdf, alias, mapped_chroms, mapped_region)
        total = sum(len(v) for v in positions_by_chrom.values())
        print(f"  [INFO] targets: {total} positions, {len(positions_by_chrom)} chroms")
        if total == 0:
            print("  [WARN] No valid target positions. Skipping.")
            bam.close()
            return
    else:
        spans_scan = build_spans_for_scan(bam, mapped_chroms, mapped_region)
        if not spans_scan:
            print("  [WARN] No spans to scan. Skipping.")
            bam.close()
            return

    bam.close()

    # DEL/INSを同一BAM走査ループで予測（出力は従来通り別ファイル）
    _run_combined_prediction(
        bam_path, model_del, model_ins, model_del_path, model_ins_path,
        positions_by_chrom, spans_scan,
        mean_depth, step, threshold_del, threshold_ins, n_jobs,
        out_del if model_del is not None else None,
        out_ins if model_ins is not None else None,
        out_del_feat, out_ins_feat, out_del_raw, out_ins_raw,
        ins_mq=ins_mq, total_chrom_count=total_chrom_count, median_isize=median_isize,
    )


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "INS / DEL 統合予測スクリプト。"
            "DELモデルはDEL特徴量系、INSモデルはINS特徴量系を使用して予測する。"
            "DEL/INSは同一のBAM走査ループでまとめて計算し、"
            "1品種あたり {SAMPLE}_DEL.tsv と {SAMPLE}_INS.tsv を出力する。"
        )
    )
    parser.add_argument("-b", "--bam", nargs="+", required=True,
                        help="入力BAMファイル（複数可）またはBAMが入ったディレクトリ")

    model_group = parser.add_argument_group("モデル（少なくとも一方を指定）")
    model_group.add_argument("--model-del", default=None, help="DELモデル (.joblib)")
    model_group.add_argument("--model-ins", default=None, help="INSモデル (.joblib)")

    parser.add_argument("-o", "--outdir", required=True, help="出力ディレクトリ")
    parser.add_argument("--step", type=int, default=1,
                        help="スキャンステップ（full scanモード、デフォルト: 1）")
    parser.add_argument("--threshold-del", type=float, default=0.98,
                        help="DEL予測確率の閾値（デフォルト: 0.98）")
    parser.add_argument("--threshold-ins", type=float, default=0.98,
                        help="INS予測確率の閾値（デフォルト: 0.98）")
    parser.add_argument("--ins-mq", type=int, default=INS_MQ_DEFAULT,
                        help="INS特徴量計算のMQ閾値（MQ>=この値のリードのみ対象。"
                             f"デフォルト: {INS_MQ_DEFAULT}）")
    parser.add_argument("--chroms", nargs="+", help="対象染色体フィルタ")
    parser.add_argument("--region", help="対象リージョン (chr or chr:start-end)")
    parser.add_argument("--skip-existing", action="store_true", help="既存の出力ファイルをスキップ")
    parser.add_argument("--no-features", action="store_true", help="特徴量TSVを出力しない")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="並列処理数（0=全コア使用）")
    parser.add_argument("--targets", default=None,
                        help="targets TSV（chr/posi列必須）。指定した座標のみ予測する。")
    parser.add_argument("--mean-depth-tsv", default=None,
                        help="estimate_mean_depth.py が出力したTSV（sample, mean_depth）。"
                             "指定時はサンプルごとにこの値を使い、BAMからの再計算を行わない。"
                             "領域抽出済みBAM（±10kb等）を --bam に渡す場合は必須"
                             "（抽出前の全長BAMに対して estimate_mean_depth.py を実行した結果を渡すこと）。")

    args = parser.parse_args()

    if args.model_del is None and args.model_ins is None:
        parser.error("Specify at least one of --model-del or --model-ins.")

    n_jobs = args.jobs if args.jobs > 0 else multiprocessing.cpu_count()

    bam_files = expand_bam_inputs(args.bam)
    if not bam_files:
        print("[ERROR] No BAM files found to process")
        return

    print(f"[INFO] BAM files to process: {len(bam_files)}")
    if args.targets:
        print(f"[INFO] targets mode: {args.targets}")

    mean_depth_map: Dict[str, float] = {}
    median_isize_map: Dict[str, float] = {}
    if args.mean_depth_tsv:
        # estimate_mean_depth.py が出力するTSVは sample, mean_depth, median_isize の
        # 3列（median_isize列が無い旧形式のTSVでも、列不足の行は単に無視してmean_depthだけ
        # 読み込む形でそのまま動く）。
        with open(args.mean_depth_tsv, encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or not parts[1]:
                    continue
                try:
                    mean_depth_map[parts[0]] = float(parts[1])
                except ValueError:
                    continue
                if len(parts) >= 3 and parts[2]:
                    try:
                        median_isize_map[parts[0]] = float(parts[2])
                    except ValueError:
                        pass
        print(f"[INFO] Loaded mean-depth-tsv: {args.mean_depth_tsv} "
              f"({len(mean_depth_map)} sample(s), median_isize {len(median_isize_map)} sample(s))")

    model_del = _force_single_threaded_predict(joblib.load(args.model_del)) if args.model_del else None
    model_ins = _force_single_threaded_predict(joblib.load(args.model_ins)) if args.model_ins else None
    model_del_path = args.model_del
    model_ins_path = args.model_ins

    outdir_pred     = Path(args.outdir) / "predictions"
    outdir_del_feat = Path(args.outdir) / "features" / "del_values"
    outdir_ins_feat = Path(args.outdir) / "features" / "ins_values"
    outdir_del_raw  = Path(args.outdir) / "features" / "del_raw"
    outdir_ins_raw  = Path(args.outdir) / "features" / "ins_raw"

    outdir_pred.mkdir(parents=True, exist_ok=True)
    if not args.no_features:
        if model_del:
            outdir_del_feat.mkdir(parents=True, exist_ok=True)
            outdir_del_raw.mkdir(parents=True, exist_ok=True)
        if model_ins:
            outdir_ins_feat.mkdir(parents=True, exist_ok=True)
            outdir_ins_raw.mkdir(parents=True, exist_ok=True)

    name_map = dedup_sample_names_upper(bam_files)

    for bam in bam_files:
        sample = name_map[bam]

        out_del = str(outdir_pred / f"{sample}_DEL.tsv") if model_del else None
        out_ins = str(outdir_pred / f"{sample}_INS.tsv") if model_ins else None

        if args.skip_existing:
            existing = [f for f in [out_del, out_ins] if f and Path(f).exists()]
            expected = sum([model_del is not None, model_ins is not None])
            if len(existing) == expected:
                print(f"• Skipped (already exists): {sample}")
                continue

        out_del_feat = str(outdir_del_feat / f"{sample}_del_features.tsv") if (model_del and not args.no_features) else None
        out_ins_feat = str(outdir_ins_feat / f"{sample}_ins_features.tsv") if (model_ins and not args.no_features) else None
        out_del_raw  = str(outdir_del_raw  / f"{sample}_del_raw.tsv")      if (model_del and not args.no_features) else None
        out_ins_raw  = str(outdir_ins_raw  / f"{sample}_ins_raw.tsv")      if (model_ins and not args.no_features) else None

        predict_combined(
            bam, model_del, model_ins,
            model_del_path, model_ins_path,
            out_del, out_ins,
            out_del_feat, out_ins_feat,
            out_del_raw, out_ins_raw,
            args.step, args.threshold_del, args.threshold_ins,
            args.chroms, args.region,
            n_jobs,
            args.targets,
            precomputed_mean_depth=mean_depth_map.get(sample),
            ins_mq=args.ins_mq,
            precomputed_median_isize=median_isize_map.get(sample),
        )


if __name__ == "__main__":
    main()

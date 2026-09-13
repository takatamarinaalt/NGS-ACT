#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple prediction TSVs (CHROM, POS, TYPE, PROB) into a single wide table.

- Input files: sample_INS.tsv, sample_DEL.tsv, sample_predictions.tsv (gz optional)
- _INS and _DEL files for the same sample are merged (non-NONE takes priority)
- Output columns: chr, posi, <sample> (no ".tsv"/".sort.tsv"/"_predictions"), filled with TYPE.
- Duplicates resolved by highest PROB if available.
- DEL_st / DEL_end are paired per sample (nearest downstream DEL_end for each
  DEL_st, stack-based). Only the two paired positions become DEL. DEL_mid and
  any unpaired DEL_st/DEL_end are dropped for that sample.
- INS_ prefixed types are converted to INS.

[Filtering rule]
- Keep ONLY rows where at least one sample has DEL or INS.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

def norm_chr_to_number(chrom: str) -> str:
    c = str(chrom).strip().replace("CHR", "chr")
    if c.lower().startswith("chr"):
        c = c[3:]
    try:
        return str(int(c))
    except Exception:
        return c

def guess_sample_name(path: Path) -> str:
    """
    拡張子とサフィックスを除いたファイル名を列名として使う
    例:
      AADRIBATTA_INS.tsv         -> AADRIBATTA
      AADRIBATTA_DEL.tsv         -> AADRIBATTA
      AADRIBATTA_predictions.tsv -> AADRIBATTA
      SAMPLE.tsv                 -> SAMPLE
    """
    name = path.name

    # .gz を除去
    if name.endswith(".gz"):
        name = name[:-3]

    # 拡張子を除去
    if name.endswith(".sort.tsv"):
        name = name[:-9]
    elif name.endswith(".tsv"):
        name = name[:-4]

    # 末尾の _predictions を除去（大文字小文字を吸収）
    if name.lower().endswith("_predictions"):
        name = name[:-(len("_predictions"))]

    # _INS, _DEL サフィックスを除去
    if name.endswith("_INS") or name.endswith("_DEL"):
        name = name[:-4]

    return name

def safe_read_csv(path: Path) -> pd.DataFrame:
    """pandas 1.x/2.x 両対応で重複列名に .1, .2 を付与。空ファイルはNoneを返す"""
    try:
        return pd.read_csv(path, sep="\t", dtype={"CHROM": str}, compression="infer",
                           mangle_dupe_cols=True)
    except TypeError:
        # pandas >= 2.0 では mangle_dupe_cols が削除済み → デフォルトで一意化される
        try:
            return pd.read_csv(path, sep="\t", dtype={"CHROM": str}, compression="infer")
        except pd.errors.EmptyDataError:
            return None
    except pd.errors.EmptyDataError:
        return None

def normalize_type(type_val: str) -> str:
    """
    TYPE値を正規化する
    - DEL_st, DEL_mid, DEL_end など DEL_ で始まるもの → そのまま保持（サブタイプ区別のため）
    - INS_long, INS_short など INS_ で始まるもの → INS
    - それ以外はそのまま
    """
    x = str(type_val)
    if x.startswith("INS_"):
        return "INS"
    return type_val

def pair_del_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    1サンプル分の予測テーブル（CHROM, POS, TYPE, [PROB]）に対して、
    DEL_st / DEL_end のペアリングを行う。

    ルール:
      - 染色体ごとに POS 昇順で走査し、スタック方式でペアリングする
        （DEL_st が出たら積む。DEL_end が出たら直近未消化の DEL_st と1組にする）
      - ペアが成立した2ポジション（st側・end側）だけを TYPE="DEL" に変換して残す
      - DEL_mid、および相方が見つからなかった DEL_st / DEL_end 単独ポジションは
        行ごと除外する（このサンプルではその座位にデータが無かった扱いになる）
      - DEL_st / DEL_end / DEL_mid 以外（NONE, INS, SNP等）はそのまま残す
    """
    type_upper = df["TYPE"].astype(str).str.upper()
    is_st  = type_upper == "DEL_ST"
    is_end = type_upper == "DEL_END"
    is_del_other = type_upper.str.startswith("DEL_") & ~is_st & ~is_end  # DEL_MID など

    non_del_df = df[~(is_st | is_end | is_del_other)]

    keep_indices = []
    for chrom, sub in df[is_st | is_end].groupby("CHROM"):
        sub = sub.sort_values("POS")
        pending_st = []  # スタック：未消化のDEL_stのindex
        for idx, row in sub.iterrows():
            t = str(row["TYPE"]).upper()
            if t == "DEL_ST":
                pending_st.append(idx)
            elif t == "DEL_END":
                if pending_st:
                    st_idx = pending_st.pop()
                    keep_indices.append(st_idx)
                    keep_indices.append(idx)
                # 相方がいなければこのDEL_endは除外（keep_indicesに追加しない）

    del_pairs_df = df.loc[keep_indices].copy()
    if len(del_pairs_df) > 0:
        del_pairs_df["TYPE"] = "DEL"

    result = pd.concat([non_del_df, del_pairs_df], ignore_index=True)
    return result

def read_prediction_table(path: Path) -> pd.DataFrame:
    """
    予測テーブルを読み込む。空ファイルや必須カラムがない場合はNoneを返す
    """
    df = safe_read_csv(path)

    if df is None:
        print(f"  [WARN] Empty file skipped: {path.name}")
        return None

    if not {"CHROM","POS","TYPE"}.issubset(df.columns):
        print(f"  [WARN] Missing required columns in {path.name}, skipped")
        return None

    if len(df) == 0:
        print(f"  [WARN] No data rows in {path.name}, skipped")
        return None

    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    invalid = df["POS"].isna().sum()
    if invalid:
        print(f"  [WARN] {path.name}: excluded {invalid} row(s) with non-numeric POS.")
    df = df.dropna(subset=["POS"])
    if len(df) == 0:
        print(f"  [WARN] {path.name}: no valid rows remain. Skipping.")
        return None
    df["TYPE"] = df["TYPE"].apply(normalize_type)
    return df[["CHROM","POS","TYPE"] + (["PROB"] if "PROB" in df.columns else [])]

def dedup_by_prob(df: pd.DataFrame) -> pd.DataFrame:
    if "PROB" in df.columns:
        return df.sort_values(["CHROM","POS","PROB"], ascending=[True,True,False]) \
                 .drop_duplicates(["CHROM","POS"], keep="first")
    return df.drop_duplicates(["CHROM","POS"], keep="first")

def merge_ins_del_files(file_list: list) -> pd.DataFrame:
    """
    同じサンプルのINSとDEL（および predictions）ファイルを統合する
    同じ(CHROM, POS)で異なるTYPEがある場合、NONEより他の予測（INDEL）を優先
    空ファイルはスキップ。全ファイルが空の場合はNoneを返す
    """
    dfs = []
    for f in file_list:
        df = read_prediction_table(f)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None

    if len(dfs) == 1:
        return dedup_by_prob(dfs[0])

    combined = pd.concat(dfs, ignore_index=True)

    def get_priority(type_val):
        return 0 if type_val == "NONE" else 1

    combined["_priority"] = combined["TYPE"].apply(get_priority)

    if "PROB" in combined.columns:
        combined = combined.sort_values(
            ["CHROM", "POS", "_priority", "PROB"],
            ascending=[True, True, False, False]
        )
    else:
        combined = combined.sort_values(
            ["CHROM", "POS", "_priority"],
            ascending=[True, True, False]
        )

    result = combined.drop_duplicates(["CHROM", "POS"], keep="first")
    result = result.drop(columns=["_priority"])
    return result

def group_files_by_sample(files: list) -> dict:
    """ファイルをサンプル名でグループ化"""
    groups = defaultdict(list)
    for f in files:
        sample_name = guess_sample_name(f)
        groups[sample_name].append(f)
    return dict(groups)

def filter_files_by_mode(files: list, mode: str) -> list:
    """
    モードに応じてファイルをフィルタリング
    - all: 全て
    - del: _DEL.tsv のみ
    - ins: _INS.tsv のみ
    """
    if mode == "all":
        return files

    filtered = []
    for f in files:
        name = f.name
        if name.endswith(".gz"):
            name = name[:-3]

        if mode == "del":
            if "_DEL.tsv" in name or "_DEL.sort.tsv" in name:
                filtered.append(f)
        elif mode == "ins":
            if "_INS.tsv" in name or "_INS.sort.tsv" in name:
                filtered.append(f)

    return filtered

def merge_wide(files, fill="N", keep_chrom=False, mode="all"):
    per_sample = {}
    idx = None
    no_data_samples = []

    sample_groups = group_files_by_sample(files)
    print(f"[INFO] Processing {len(files)} files from {len(sample_groups)} samples...")

    for sample_name, sample_files in sample_groups.items():
        print(f"[INFO] Processing {sample_name} ({len(sample_files)} file(s): {[f.name for f in sample_files]})...")

        df = merge_ins_del_files(sample_files)
        if df is None or len(df) == 0:
            # データが0件でも品種は残す（全ポジションが fill 値になる）
            print(f"  [INFO] Sample {sample_name}: 0 predictions. Keeping all positions as '{fill}'.")
            per_sample[sample_name] = pd.Series(dtype=object)
            no_data_samples.append(sample_name)
            continue

        before_pair = len(df)
        df = pair_del_boundaries(df)
        after_pair = len(df)
        if before_pair != after_pair:
            print(f"  [INFO] Sample {sample_name}: after DEL_st/DEL_end pairing "
                  f"{before_pair} -> {after_pair} row(s) (excluding DEL_mid / unpaired)")

        if keep_chrom:
            df["_chr"] = df["CHROM"]
        else:
            df["_chr"] = df["CHROM"].map(norm_chr_to_number)
        df["_posi"] = df["POS"].astype(int)

        s = df.set_index(["_chr","_posi"])["TYPE"]
        per_sample[sample_name] = s
        idx = s.index if idx is None else idx.union(s.index)

    if not per_sample:
        print("[ERROR] No samples to process.", file=sys.stderr)
        sys.exit(1)

    # idx が None = 全サンプルが0件（ポジションが1件も存在しない）
    if idx is None:
        print("[WARN] All samples have 0 predictions. Output will be an empty table.")
        idx = pd.MultiIndex.from_tuples([], names=["_chr", "_posi"])

    if no_data_samples:
        print(f"[INFO] Samples with 0 predictions, treated as all-N ({len(no_data_samples)}): {no_data_samples}")

    out = pd.DataFrame(index=idx)
    for sm, s in per_sample.items():
        out[sm] = s

    out = out.fillna(fill).reset_index(names=["chr","posi"])
    sample_cols = sorted([c for c in out.columns if c not in {"chr","posi"}])
    n_samples = len(sample_cols)

    mat = out[sample_cols]

    # --- ポジションフィルタ: DEL/INSを少なくとも1つ持つ行のみ残す ---
    # pair_del_boundaries() によりDELはペアリング済みの正味の "DEL" 文字列のみになっている
    if mode == "del":
        indel_mask = mat.isin(["DEL"])
    elif mode == "ins":
        indel_mask = mat.isin(["INS"])
    else:
        indel_mask = mat.isin(["DEL", "INS"])

    keep = indel_mask.any(axis=1)
    before = len(out)
    out    = out[keep]
    after  = len(out)

    print(f"[INFO] After position filter: {before} -> {after} row(s)"
          f"  (excluding positions with no DEL/INS predicted in any sample)")

    return out[["chr","posi"] + sample_cols]

def _run_merge(in_dir, output, mode, fill, keep_chrom, recursive):
    pats = ["*.tsv", "*.tsv.gz"]
    files = []
    for pat in pats:
        files += (in_dir.rglob(pat) if recursive else in_dir.glob(pat))
    if not files:
        print(f"[ERROR] No TSVs found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(files)} TSV files in {in_dir}")
    files = filter_files_by_mode(files, mode)
    if not files:
        print(f"[ERROR] No matching TSV files found for mode '{mode}'", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] [{mode.upper()}] Using {len(files)} TSV files after filtering")
    merged = merge_wide(files, fill=fill, keep_chrom=keep_chrom, mode=mode)
    merged.to_csv(output, sep="\t", index=False)
    print(f"[OK] wrote {output} rows={len(merged)} cols={len(merged.columns)}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge multiple prediction TSVs into a single wide table."
    )
    ap.add_argument("-i", "--input-dir", required=True,
                    help="Input directory containing TSV files")
    ap.add_argument("-o", "--output", default=None,
                    help="Output TSV file path（--del-only / --ins-only と組み合わせて使用）")
    ap.add_argument("--del-output", default=None,
                    help="DEL merged TSV出力パス（--ins-outputと同時指定でDEL/INS両方を1回で処理）")
    ap.add_argument("--ins-output", default=None,
                    help="INS merged TSV出力パス（--del-outputと同時指定でDEL/INS両方を1回で処理）")
    ap.add_argument("--fill", default="N",
                    help="Fill value for missing positions (default: N)")
    ap.add_argument("--keep-chrom", action="store_true",
                    help="Keep original chromosome names (don't normalize)")
    ap.add_argument("--recursive", action="store_true",
                    help="Search for TSV files recursively")

    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--del-only", action="store_true",
                            help="_DEL.tsvファイルのみを統合（-o と組み合わせて使用）")
    mode_group.add_argument("--ins-only", action="store_true",
                            help="_INS.tsvファイルのみを統合（-o と組み合わせて使用）")

    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    kw = dict(fill=args.fill, keep_chrom=args.keep_chrom, recursive=args.recursive)

    # --del-output / --ins-output が指定された場合は両方を1回で処理
    if args.del_output or args.ins_output:
        if args.del_output:
            print("[INFO] Mode: DEL (_DEL.tsv files)")
            _run_merge(in_dir, args.del_output, "del", **kw)
        if args.ins_output:
            print("[INFO] Mode: INS (_INS.tsv files)")
            _run_merge(in_dir, args.ins_output, "ins", **kw)
        return

    # 従来の -o / --del-only / --ins-only モード
    if args.output is None:
        ap.error("-o / --output is required unless --del-output / --ins-output is used.")

    if args.del_only:
        mode = "del"
        print("[INFO] Mode: DEL only (_DEL.tsv files)")
    elif args.ins_only:
        mode = "ins"
        print("[INFO] Mode: INS only (_INS.tsv files)")
    else:
        mode = "all"
        print("[INFO] Mode: ALL (both _DEL.tsv and _INS.tsv files)")

    _run_merge(in_dir, args.output, mode, **kw)


if __name__ == "__main__":
    main()
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
- INS_ prefixed types (INS_TP, INS_notTP) are kept as-is; they are NOT
  collapsed into a generic "INS" label.

[Filtering rule]
- Keep ONLY rows where at least one sample has DEL / INS_TP / INS_notTP.

[モード]
  --del-only      : _DEL.tsv のみ統合（TYPE=DELの行のみ残す）
  --ins-only      : _INS.tsv のみ統合（TYPE=INS_TP または INS_notTP のどちらかがある行を残す。
                     セルの値は INS_TP / INS_notTP のまま、1つのTSVに混在する）
  --instp-only    : _INS.tsv のうち INS_TP のみを対象に統合（別TSVとして出力）
  --insnottp-only : _INS.tsv のうち INS_notTP のみを対象に統合（別TSVとして出力）

[領域モード --vcf-mode]
  all  : 全ポジションを対象（デフォルト）
  exon : --gff と --gene で指定した遺伝子のエキソン領域に座乗するポジションのみ対象
"""

import argparse
import sys
import urllib.parse
from pathlib import Path
from collections import defaultdict
import pandas as pd


# ============================================================
# GFF3 エキソン領域パーサ（step1/3_vcf_select.py と同仕様）
# ============================================================

def parse_gff_exons(gff_path: str, gene_query: str) -> list:
    """
    GFF3 から指定遺伝子のエキソン座標リストを返す。

    標準 GFF3（gene → mRNA → exon）と
    RAP-DB 形式（mRNA 最上位 → CDS/UTR）の両方に対応。

    戻り値: list of (chrom, start, end)  ※1-based, closed interval, マージ済み
    """
    id_feats  = {}
    sub_feats = []

    EXON_TYPES = frozenset({'exon', 'CDS', 'five_prime_UTR', 'three_prime_UTR'})

    with open(gff_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 9:
                continue
            chrom, _, ftype = cols[0], cols[1], cols[2]
            start, end = int(cols[3]), int(cols[4])
            attrs = {}
            for item in cols[8].split(';'):
                item = item.strip()
                if '=' in item:
                    k, v = item.split('=', 1)
                    attrs[k.strip()] = v.strip()
            fid     = attrs.get('ID', '')
            parents = [p.strip() for p in attrs.get('Parent', '').split(',') if p.strip()]

            if fid:
                id_feats[fid] = {
                    'type': ftype, 'chrom': chrom,
                    'start': start, 'end': end,
                    'attrs': attrs, 'parents': parents,
                }
                if ftype in EXON_TYPES:
                    for p in parents:
                        sub_feats.append((ftype, chrom, start, end, p))
            else:
                if ftype in EXON_TYPES:
                    for p in parents:
                        sub_feats.append((ftype, chrom, start, end, p))

    def attr_symbols(attrs, key):
        raw = attrs.get(key, '')
        if not raw:
            return []
        return [s.strip() for s in urllib.parse.unquote(raw).split(',') if s.strip()]

    def matches(feat_attrs, query):
        q = query.lower()
        a = feat_attrs
        for key in ('ID', 'Name'):
            if q in a.get(key, '').lower():
                return True
        if a.get('Locus_id', '').lower() == q:
            return True
        for key in ('RAP-DB Gene Symbol Synonym(s)',
                    'CGSNL Gene Symbol',
                    'Oryzabase Gene Symbol Synonym(s)'):
            for sym in attr_symbols(a, key):
                if q == sym.lower():
                    return True
        return False

    gene_ids = {fid for fid, feat in id_feats.items()
                if feat['type'] == 'gene' and matches(feat['attrs'], gene_query)}

    if gene_ids:
        mrna_ids = {fid for fid, feat in id_feats.items()
                    if feat['type'] in ('mRNA', 'transcript', 'pseudogenic_transcript')
                    and any(p in gene_ids for p in feat['parents'])}
    else:
        mrna_ids = {fid for fid, feat in id_feats.items()
                    if feat['type'] == 'mRNA' and matches(feat['attrs'], gene_query)}

    if not mrna_ids:
        return []

    regions = []
    seen = set()
    for _, chrom, start, end, parent in sub_feats:
        if parent in mrna_ids:
            key = (chrom, start, end)
            if key not in seen:
                seen.add(key)
                regions.append(key)

    if not regions:
        return []

    by_chrom = {}
    for chrom, s, e in regions:
        by_chrom.setdefault(chrom, []).append((s, e))

    merged = []
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((chrom, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((chrom, cur_s, cur_e))

    return sorted(merged, key=lambda x: (x[0], x[1]))

def norm_chr_to_number(chrom: str) -> str:
    c = str(chrom).strip().replace("CHR", "chr")
    if c.lower().startswith("chr"):
        c = c[3:]
    try:
        return str(int(c))
    except:
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
    - DEL_st, DEL_mid, DEL_end などは、この時点ではサブタイプのまま保持する
      （最終的なDEL判定は pair_del_boundaries() でst/endのペアリングを行った上で決める）
    - INS_TP, INS_notTP はサブタイプのまま保持する（INS への統合はしない）
    - それ以外はそのまま
    """
    return str(type_val)

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

    df["POS"] = pd.to_numeric(df["POS"], errors="raise")
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
    - ins / instp / insnottp: _INS.tsv のみ（サブタイプでの絞り込みは行フィルタ側で行う）
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
        elif mode in ("ins", "instp", "insnottp"):
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
                  f"{before_pair} → {after_pair} rows (DEL_mid / unpaired excluded)")

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
        print(f"[INFO] Samples with 0 predictions treated as all-N ({len(no_data_samples)}): {no_data_samples}")

    out = pd.DataFrame(index=idx)
    for sm, s in per_sample.items():
        out[sm] = s

    out = out.fillna(fill).reset_index(names=["chr","posi"])
    sample_cols = sorted([c for c in out.columns if c not in {"chr","posi"}])
    n_samples = len(sample_cols)

    mat = out[sample_cols]

    # --- ポジションフィルタ: 対象タイプを少なくとも1つ持つ行のみ残す ---
    if mode == "del":
        indel_mask = mat.isin(["DEL"])
    elif mode == "instp":
        indel_mask = mat.isin(["INS_TP"])
    elif mode == "insnottp":
        indel_mask = mat.isin(["INS_notTP"])
    elif mode == "ins":
        indel_mask = mat.isin(["INS_TP", "INS_notTP"])
    else:
        indel_mask = mat.isin(["DEL", "INS_TP", "INS_notTP"])

    keep = indel_mask.any(axis=1)
    before = len(out)
    out    = out[keep]
    after  = len(out)

    print(f"[INFO] After position filter: {before} → {after} rows"
          f"  (positions where no sample has the target type[mode={mode}] excluded)")

    return out[["chr","posi"] + sample_cols]

def filter_by_exon_regions(df: pd.DataFrame, exon_regions: list, keep_chrom: bool) -> pd.DataFrame:
    """
    wide table の行をエキソン領域でフィルタリングする。

    exon_regions: [(chrom_str, start_1based, end_1based), ...]
    df の chr 列は keep_chrom=False のとき数字（例: "6"）になっているため、
    GFF の染色体名（例: "chr06"）と正規化して照合する。
    """
    if not exon_regions:
        return df

    # GFF の chrom を df の chr 列形式に合わせて正規化
    def norm(c: str) -> str:
        if keep_chrom:
            return c
        return norm_chr_to_number(c)

    # (chr_normalized, start, end) のリストを構築
    norm_regions = [(norm(ch), s, e) for ch, s, e in exon_regions]

    def in_exon(row) -> bool:
        chr_val  = str(row["chr"])
        posi_val = int(row["posi"])
        for ch, s, e in norm_regions:
            if chr_val == ch and s <= posi_val <= e:
                return True
        return False

    before = len(df)
    mask   = df.apply(in_exon, axis=1)
    df     = df[mask].reset_index(drop=True)
    after  = len(df)
    print(f"[INFO] After exon filter: {before} → {after} rows")
    return df


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Merge multiple prediction TSVs into a single wide table.\n\n"
            "【領域モード --vcf-mode】\n"
            "  all  : 全ポジションを対象（デフォルト）\n"
            "  exon : --gff と --gene で指定した遺伝子のエキソン領域のみを対象"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-i","--input-dir",required=True,
                    help="Input directory containing TSV files")
    ap.add_argument("-o","--output",required=True,
                    help="Output TSV file path")
    ap.add_argument("--fill",default="N",
                    help="Fill value for missing positions (default: N)")
    ap.add_argument("--keep-chrom",action="store_true",
                    help="Keep original chromosome names (don't normalize)")
    ap.add_argument("--recursive",action="store_true",
                    help="Search for TSV files recursively")

    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--del-only", action="store_true",
                           help="_DEL.tsvファイルのみを統合（_INS.tsvは無視）")
    mode_group.add_argument("--ins-only", action="store_true",
                           help="_INS.tsvファイルのみを統合（_DEL.tsvは無視）。"
                                "INS_TP/INS_notTPは統合せず、そのままセルに残す。")
    mode_group.add_argument("--instp-only", action="store_true",
                           help="_INS.tsvのうち INS_TP の行のみを統合した別TSVを作る")
    mode_group.add_argument("--insnottp-only", action="store_true",
                           help="_INS.tsvのうち INS_notTP の行のみを統合した別TSVを作る")

    # --- 領域モード ---
    ap.add_argument(
        "--vcf-mode", choices=["all", "exon"], default="all",
        help="all: 全ポジションを対象（デフォルト） / exon: GFF3 エキソン領域のみを対象"
    )
    ap.add_argument(
        "--gff",
        help="[--vcf-mode exon 必須] GFF3 ファイルのパス"
    )
    ap.add_argument(
        "--gene",
        help="[--vcf-mode exon 必須] 対象遺伝子の ID / Name / Locus_id / 遺伝子シンボル"
    )

    args = ap.parse_args()

    # vcf-mode の引数チェック
    if args.vcf_mode == "exon":
        if not args.gff or not args.gene:
            ap.error("--vcf-mode exon requires --gff and --gene")

    in_dir = Path(args.input_dir)
    pats = ["*.tsv","*.tsv.gz"]
    files = []
    for pat in pats:
        files += (in_dir.rglob(pat) if args.recursive else in_dir.glob(pat))
    if not files:
        print("No TSVs found", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(files)} TSV files in {in_dir}")

    if args.del_only:
        mode = "del"
        print("[INFO] Mode: DEL only (_DEL.tsv files)")
    elif args.ins_only:
        mode = "ins"
        print("[INFO] Mode: INS only (_INS.tsv files, INS_TP/INS_notTP mixed)")
    elif args.instp_only:
        mode = "instp"
        print("[INFO] Mode: INS_TP only (_INS.tsv files)")
    elif args.insnottp_only:
        mode = "insnottp"
        print("[INFO] Mode: INS_notTP only (_INS.tsv files)")
    else:
        mode = "all"
        print("[INFO] Mode: ALL (both _DEL.tsv and _INS.tsv files)")

    files = filter_files_by_mode(files, mode)
    if not files:
        print(f"No matching TSV files found for mode '{mode}'", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Using {len(files)} TSV files after filtering")

    merged = merge_wide(files, fill=args.fill, keep_chrom=args.keep_chrom, mode=mode)

    # --- エキソンフィルタ ---
    if args.vcf_mode == "exon":
        print(f"[INFO] vcf-mode=exon: fetching exons for gene '{args.gene}' from GFF '{args.gff}'...")
        exon_regions = parse_gff_exons(args.gff, args.gene)
        if not exon_regions:
            print(
                f"[ERROR] No exons found for gene '{args.gene}' in the GFF3.\n"
                f"        Check ID= / Name= / Locus_id= / gene symbol.",
                file=sys.stderr
            )
            sys.exit(1)
        print(f"[INFO] Using {len(exon_regions)} exon region(s)")
        for ch, s, e in exon_regions:
            print(f"       {ch}:{s}-{e}  ({e - s + 1} bp)")
        merged = filter_by_exon_regions(merged, exon_regions, args.keep_chrom)

    merged.to_csv(args.output, sep="\t", index=False)
    print(f"[OK] wrote {args.output} rows={len(merged)} cols={len(merged.columns)}")

if __name__=="__main__":
    main()
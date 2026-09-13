#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd

# モードごとのクラス定義
_MODE_CONFIG = {
    "all": {
        "classes":  ["NONE", "INS", "DEL"],
        "map":      {"NONE": 0, "INS": 1, "INS_SHORT": 1, "INS_LONG": 1, "DEL": 2},
    },
    "del": {
        "classes":  ["NONE", "DEL"],
        "map":      {"NONE": 0, "DEL": 1, "DEL_ST": 1, "DEL_MID": 1, "DEL_END": 1},
    },
    "ins": {
        "classes":  ["NONE", "INS"],
        "map":      {"NONE": 0, "INS": 1, "INS_SHORT": 1, "INS_LONG": 1},
    },
}

def normalize_frame_strings(df: pd.DataFrame) -> pd.DataFrame:
    """文字列を正規化する。N は NONE と同一視せず、独立した欠損トークンとして残す
    （--n-fill の指定に従って処理される。mean なら座位ごとの平均ベクトルで補完、
    zero なら全クラス0のベクトルのまま残る）。"""
    g = df.astype("string")
    g = g.apply(lambda c: c.str.strip().str.upper()
                          .str.replace("-", "_").str.replace(" ", "_"))
    return g

def convert_data(input_file, output_file=None, float_dtype="float32", n_fill_mode="zero", mode="all"):
    """
    入力（wide: chr,posi + samples）を one-hot に変換するだけ。
    - mode="all": NONE/INS/DEL の 3クラス
    - mode="del": NONE/DEL の 2クラス（DEL専用データ用）
    - mode="ins": NONE/INS の 2クラス（INS専用データ用）
    - 座位の除外は一切しない（NONE/N だけの座位も残す）
    - N は独立した欠損トークンとして扱われる（NONEとは同一視しない）
    - --n-fill mean（デフォルト）: 欠損位置は座位ごとの平均ベクトル（他サンプルの
      クラス構成比）で補完する
    - --n-fill zero: 欠損位置は全クラス0のベクトルのまま残す
    """
    cfg = _MODE_CONFIG[mode]
    CLASSES        = cfg["classes"]
    MAP_TO_CLASS   = cfg["map"]
    ALLOWED_TOKENS = set(MAP_TO_CLASS.keys()) | {"N"}
    n_classes      = len(CLASSES)

    if output_file is None:
        base_name = os.path.basename(input_file)
        prefix = "indel2" if mode in ("del", "ins") else "indel3"
        output_file = f"{prefix}_{base_name}"

    df = pd.read_csv(input_file, sep="\t")
    if df.shape[1] < 3:
        raise ValueError("列が足りません。先頭2列に 'chr','posi'、以降にサンプル列を想定しています。")
    if df.columns[0].lower() != "chr" or df.columns[1].lower() != "posi":
        raise ValueError("先頭2列は 'chr' と 'posi' である必要があります。")

    sample_cols = list(df.columns[2:])
    chr_pos = df[["chr", "posi"]].copy()
    loci = [f"{r.chr}_{r.posi}" for r in chr_pos.itertuples(index=False)]

    mat = normalize_frame_strings(df[sample_cols])

    uniq = pd.unique(mat.stack().dropna())
    unknown = sorted(set(uniq) - ALLOWED_TOKENS)
    if unknown:
        ex = ", ".join(list(map(str, unknown[:10])))
        more = " ..." if len(unknown) > 10 else ""
        raise ValueError(
            f"未知の値が見つかりました（許可: {sorted(ALLOWED_TOKENS)} / N は欠損トークンとして許可されます）。例: {ex}{more}"
        )

    codes = mat.apply(lambda c: c.map(MAP_TO_CLASS))
    codes_np = codes.to_numpy(dtype="float64").copy()
    codes_np[np.isnan(codes_np)] = -1
    codes_np = codes_np.astype(np.int16)

    n_loci, n_samples = codes_np.shape

    out = np.zeros((n_samples, n_loci, n_classes), dtype=float_dtype)
    miss = (codes_np == -1).T

    if n_fill_mode == "mean":
        out[miss, :] = np.nan

    for cls_idx in range(n_classes):
        m = (codes_np == cls_idx)
        if m.any():
            out[m.T, cls_idx] = 1.0

    if n_fill_mode == "mean":
        with np.errstate(invalid="ignore"):
            means = np.nanmean(out, axis=0)
        sums1d = np.nansum(means, axis=1)
        good = np.isfinite(sums1d) & (sums1d > 0)
        means[good] = means[good] / sums1d[good, None]
        means[~good] = (1.0 / n_classes)
        nan_mask_out = np.isnan(out)
        out[nan_mask_out] = np.broadcast_to(means, out.shape)[nan_mask_out]
    elif n_fill_mode == "zero":
        pass
    else:
        raise ValueError("n_fill_mode must be 'mean' or 'zero'.")

    headers = [f"{l}_{c}" for l in loci for c in CLASSES]
    out2d = out.reshape(n_samples, n_loci * n_classes)
    out_df = pd.DataFrame(out2d, columns=headers, dtype=float_dtype)
    out_df.insert(0, "chara_value", sample_cols)

    out_df.to_csv(output_file, sep="\t", index=False, float_format="%.3f")
    fill_desc = "per-locus mean" if n_fill_mode == "mean" else f"[{','.join(['0']*n_classes)}]"
    print(f"✅ Output: {output_file} (rows={out_df.shape[0]}, cols={out_df.shape[1]}) "
          f"[mode={mode}, classes={CLASSES}, N fill={fill_desc}]")
    return out_df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "INDEL予測のwide tableを one-hot 化する。"
            "座位の除外なし。N は独立した欠損トークンとして扱われ、--n-fill で"
            "指定した方法（デフォルト: 座位平均）で補完されます。\n"
            "  --mode all: NONE/INS/DEL の3クラス（デフォルト）\n"
            "  --mode del: NONE/DEL の2クラス（DEL専用データ用）\n"
            "  --mode ins: NONE/INS の2クラス（INS専用データ用）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-i", "--input", required=True, help="入力TSV（wide: chr,posi + samples）")
    ap.add_argument("-o", "--output", required=False,
                    help="出力TSV（省略時: all→indel3_<input>, del/ins→indel2_<input>）")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="出力数値型（default: float32）")
    ap.add_argument("--n-fill", choices=["mean", "zero"], default="zero",
                    help="Nの補完方法: mean=座位平均, zero=ゼロベクトル（default: zero）")
    ap.add_argument("--mode", choices=["all", "del", "ins"], default="all",
                    help="エンコードモード: all=3クラス, del=2クラス(NONE/DEL), ins=2クラス(NONE/INS)（default: all）")
    args = ap.parse_args()
    convert_data(args.input, args.output, float_dtype=args.dtype, n_fill_mode=args.n_fill, mode=args.mode)
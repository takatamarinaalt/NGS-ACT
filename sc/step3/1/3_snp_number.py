#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd

BASES = ['A', 'T', 'G', 'C']
STATUS_COLS = ['NONE', 'INS', 'DEL']


def _norm_cell(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s.upper()


def make_unique(names):
    """
    列名の重複を回避しつつ、順序と長さを保つ。
    例: ["a","a","b"] -> ["a","a__DUP1","b"]
    """
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__DUP{seen[n]}")
    return out


def _parse_ratio_token(val, classes):
    """2_vcf_to_calls.py が出力する "RATIO|{class}={frac}|{class}={frac}"
    形式を解析し、classes の順番に並んだ比率の配列を返す。解析できなければ
    None（呼び出し側で通常のN/欠損として扱う）。
    sc/step1/step1/5_snp_number.py と同一ロジック。"""
    if not val.startswith("RATIO|"):
        return None
    probs = np.zeros(len(classes), dtype="float32")
    class_idx = {c: i for i, c in enumerate(classes)}
    ok = False
    for part in val.split("|")[1:]:
        if "=" not in part:
            continue
        cls, frac_s = part.split("=", 1)
        if cls not in class_idx:
            continue
        try:
            probs[class_idx[cls]] = float(frac_s)
            ok = True
        except ValueError:
            continue
    return probs if ok else None


def _fill_matrix(codes_np, n_classes, n_fill_mode, ratio_np=None):
    """codes_np: (n_loci, n_samples) の int 配列。有効クラスは 0..n_classes-1、
    それ以外（N など未知の値）は -1（欠損）。
    ratio_np: 同shapeで、RATIOトークン（ADの実測比率）が取れたセルだけ
    (n_classes,) の比率配列、それ以外は None のオブジェクト配列（省略可）。
    戻り値: (n_samples, n_loci, n_classes) の one-hot。
    sc/step1/step1/5_snp_number.py と同一ロジック。

    優先順位:
      1) 実クラス（A/T/G/C や NONE/INS/DEL）が確定しているセル -> one-hot 1.0
      2) RATIOトークンが取れたセル -> その実測比率をそのまま使う（推定しない）
      3) それ以外の欠損（N。ADが取れなかった場合のみ） -> n_fill_mode に従う
         (mean: 座位ごとの平均ベクトルで補完 / zero: 全クラス0のまま)
    """
    n_loci, n_samples = codes_np.shape
    out = np.zeros((n_samples, n_loci, n_classes), dtype="float32")
    miss = (codes_np == -1).T  # (n_samples, n_loci)

    if n_fill_mode == "mean":
        out[miss, :] = np.nan

    for cls_idx in range(n_classes):
        m = (codes_np == cls_idx)
        if m.any():
            out[m.T, cls_idx] = 1.0

    if ratio_np is not None:
        for locus_idx in range(n_loci):
            for sample_idx in range(n_samples):
                r = ratio_np[locus_idx, sample_idx]
                if r is not None:
                    out[sample_idx, locus_idx, :] = r

    if n_fill_mode == "mean":
        fill_target = np.isnan(out)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(out, axis=0)
        sums1d = np.nansum(means, axis=1)
        good = np.isfinite(sums1d) & (sums1d > 0)
        means[good] = means[good] / sums1d[good, None]
        means[~good] = (1.0 / n_classes)
        out[fill_target] = np.broadcast_to(means, out.shape)[fill_target]
    elif n_fill_mode != "zero":
        raise ValueError("n_fill_mode must be 'mean' or 'zero'.")

    return out


def convert_data(input_file, output_file=None, n_fill_mode="zero"):
    """
    SNP列（A/T/G/Cの4クラス）とINDEL列（NONE/INS/DELの3クラス）を one-hot 化する。
    セルの値は3種類ありうる:
      - 確定クラス（A/T/G/C や NONE/INS/DEL） -> そのまま one-hot 1.0
      - RATIOトークン（2_vcf_to_calls.py が、ヘテロ接合やAD比率が取れた欠損に
        対して出力する）-> その実測比率をそのまま使う
      - N（比率も取れない、真に情報が無い欠損） -> --n-fill に従う
    """
    if output_file is None:
        base_name = os.path.basename(input_file)
        output_file = f"dummy_ATGC_{base_name}"

    df = pd.read_csv(input_file, sep="\t", dtype="string")
    sample_cols = list(df.columns[2:])
    if not sample_cols:
        raise ValueError("入力TSVにサンプル列が見つかりません（chr,posi 以外の列が必要）。")

    # 行ごとにINDEL行か判定（どこかに NONE/INS/DEL、または
    # RATIOトークンのクラス名がNONE/INS/DELならINDEL。RATIOトークンは
    # 2_vcf_to_calls.pyがヘテロ接合/AD比率の取れた欠損に対して出力するため、
    # 確定クラスが1つも無くRATIOトークンしか無い行でも正しく判定できるようにする）
    def row_is_indel(row) -> bool:
        for c in sample_cols:
            v = _norm_cell(row[c])
            if v in STATUS_COLS:
                return True
            if v.startswith("RATIO|"):
                for part in v.split("|")[1:]:
                    cls = part.split("=", 1)[0]
                    if cls in STATUS_COLS:
                        return True
        return False

    df["_IS_INDEL_"] = df.apply(row_is_indel, axis=1)
    snp_rows = df[~df["_IS_INDEL_"]].reset_index(drop=True)
    indel_rows = df[df["_IS_INDEL_"]].reset_index(drop=True)

    n_samples = len(sample_cols)

    def _codes_and_ratios_for(rows, classes):
        class_map = {c: i for i, c in enumerate(classes)}
        n_loci = len(rows)
        codes = np.full((n_loci, n_samples), -1, dtype=np.int16)
        ratios = np.full((n_loci, n_samples), None, dtype=object)
        for r_idx, row in rows.iterrows():
            for c_idx, sample_col in enumerate(sample_cols):
                val = _norm_cell(row[sample_col])
                if val in class_map:
                    codes[r_idx, c_idx] = class_map[val]
                elif val.startswith("RATIO|"):
                    r = _parse_ratio_token(val, classes)
                    if r is not None:
                        ratios[r_idx, c_idx] = r
        return codes, ratios

    snp_codes, snp_ratios = _codes_and_ratios_for(snp_rows, BASES)
    indel_codes, indel_ratios = _codes_and_ratios_for(indel_rows, STATUS_COLS)

    snp_out = _fill_matrix(snp_codes, len(BASES), n_fill_mode, snp_ratios)
    indel_out = _fill_matrix(indel_codes, len(STATUS_COLS), n_fill_mode, indel_ratios)

    raw_snp_headers = [f"{row['chr']}_{row['posi']}_{base}"
                       for _, row in snp_rows.iterrows() for base in BASES]
    raw_indel_headers = [f"{row['chr']}_{row['posi']}_short_{st}"
                         for _, row in indel_rows.iterrows() for st in STATUS_COLS]
    new_columns_unique = make_unique(raw_snp_headers + raw_indel_headers)
    snp_headers = new_columns_unique[:len(raw_snp_headers)]
    indel_headers = new_columns_unique[len(raw_snp_headers):]

    snp_2d = snp_out.reshape(n_samples, len(snp_rows) * len(BASES))
    indel_2d = indel_out.reshape(n_samples, len(indel_rows) * len(STATUS_COLS))

    out_df = pd.DataFrame(
        np.concatenate([snp_2d, indel_2d], axis=1),
        columns=snp_headers + indel_headers,
    )
    out_df.insert(0, "chara_value", sample_cols)

    out_df.to_csv(output_file, sep="\t", index=False, float_format="%.3f")
    fill_desc = "per-locus mean" if n_fill_mode == "mean" else "zero vector"
    print(f"[OK] Saved output data to {output_file}. "
          f"(RATIO tokens use the observed ratio as-is; remaining N fill={fill_desc})")
    print(f"[INFO] SNP rows={len(snp_rows)}, INDEL rows={len(indel_rows)}, "
          f"features={len(new_columns_unique)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SNPはATGCを4ビット、INDELはNONE/INS/DELを3ビットに変換。"
                    "RATIOトークン（ヘテロ/欠損のAD実測比率）は実測比率のまま反映"
    )
    parser.add_argument("-i", "--input", required=True, help="入力TSV（chr, posi, 以降はサンプル列）")
    parser.add_argument("-o", "--output", required=False, help="出力TSV（省略可）")
    parser.add_argument("--n-fill", choices=["mean", "zero"], default="zero",
                         help="RATIOトークンも取れない真の欠損Nの補完方法: "
                              "zero=ゼロベクトル（default。学習側5_snp_number.pyの既定と一致）, "
                              "mean=座位平均")
    args = parser.parse_args()
    convert_data(args.input, args.output, n_fill_mode=args.n_fill)

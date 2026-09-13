#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd

# ステータスと3ビットの対応（INDEL行用）
status_cols = ['NONE', 'INS', 'DEL']
SNP_CLASSES = ['A', 'T', 'G', 'C']


def _parse_ratio_token(val, classes):
    """4_hetero_N_one_vcf_tsv.py が出力する "RATIO|{class}={frac}|{class}={frac}"
    形式を解析し、classes の順番に並んだ比率の配列を返す。解析できなければ
    None（呼び出し側で通常のN/欠損として扱う）。"""
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
                    # 実測比率で埋める。以降のnanmean計算からは自動的に除外される
                    # （もうNaNではなくなるため、欠損として平均計算に混ざらない）。
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
    列順は元のまま：SNP座位を先に、INDEL座位を後ろにまとめる。

    セルの値は3種類ありうる:
      - 確定クラス（A/T/G/C や NONE/INS/DEL） -> そのまま one-hot 1.0
      - RATIOトークン（0/1やDPフィルタ由来の./.でも、ADから実測リード比率が
        取れた場合。4_hetero_N_one_vcf_tsv.py が生成） -> その実測比率をそのまま使う
      - N（比率も取れない、真に情報が無い欠損） -> --n-fill に従う
        (mean: 座位ごとの平均ベクトルで補完 / zero: 全クラス0のまま)
    """
    if output_file is None:
        base_name = os.path.basename(input_file)
        output_file = f"dummy_ATGC_{base_name}"

    df = pd.read_csv(input_file, sep='\t')
    sample_cols = list(df.columns[2:])

    # 行ごとに indel ステータスがあるか判定
    def row_has_indel(row):
        vals = [str(row[c]).strip() if pd.notna(row[c]) else '' for c in sample_cols]
        return any(v in status_cols for v in vals)

    df['_HAS_INDEL_ROW_'] = df.apply(row_has_indel, axis=1)

    # SNP行とINDEL行を分離
    snp_rows = df[~df['_HAS_INDEL_ROW_']]
    indel_rows = df[df['_HAS_INDEL_ROW_']]

    n_samples = len(sample_cols)

    def _codes_and_ratios_for(rows, classes):
        """rows（DataFrame）× sample_cols の値を classes の中でのインデックスに
        変換した (n_loci, n_samples) 配列（確定クラス以外・RATIO込みで -1）と、
        RATIOトークンが取れたセルだけ比率配列が入ったオブジェクト配列を返す。"""
        class_map = {c: i for i, c in enumerate(classes)}
        n_loci = len(rows)
        codes = np.full((n_loci, n_samples), -1, dtype=np.int16)
        ratios = np.full((n_loci, n_samples), None, dtype=object)
        for r_idx, (_, row) in enumerate(rows.iterrows()):
            for c_idx, sample_col in enumerate(sample_cols):
                val = str(row[sample_col]).strip() if pd.notna(row[sample_col]) else ''
                if val in class_map:
                    codes[r_idx, c_idx] = class_map[val]
                elif val.startswith("RATIO|"):
                    r = _parse_ratio_token(val, classes)
                    if r is not None:
                        ratios[r_idx, c_idx] = r
        return codes, ratios

    snp_codes, snp_ratios = _codes_and_ratios_for(snp_rows, SNP_CLASSES)
    indel_codes, indel_ratios = _codes_and_ratios_for(indel_rows, status_cols)

    snp_out = _fill_matrix(snp_codes, len(SNP_CLASSES), n_fill_mode, snp_ratios)
    indel_out = _fill_matrix(indel_codes, len(status_cols), n_fill_mode, indel_ratios)

    snp_headers = [f"{row['chr']}_{row['posi']}_{base}"
                   for _, row in snp_rows.iterrows() for base in SNP_CLASSES]
    indel_headers = [f"{row['chr']}_{row['posi']}_short_{st}"
                      for _, row in indel_rows.iterrows() for st in status_cols]

    snp_2d = snp_out.reshape(n_samples, len(snp_rows) * len(SNP_CLASSES))
    indel_2d = indel_out.reshape(n_samples, len(indel_rows) * len(status_cols))

    output_df = pd.DataFrame(
        np.concatenate([snp_2d, indel_2d], axis=1),
        columns=snp_headers + indel_headers,
    )
    output_df.insert(0, "chara_value", sample_cols)

    output_df.to_csv(output_file, sep='\t', index=False, float_format="%.3f")
    fill_desc = "per-locus mean" if n_fill_mode == "mean" else "zero vector"
    print(f"✅ Saved output data to {output_file}. "
          f"(RATIO tokens use the measured ratio as-is; remaining N fill={fill_desc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATGC/shortバイナリ変換：SNP列を先に、INDEL列を後ろにまとめる")
    parser.add_argument("-i", "--input", required=True, help="入力ファイル（TSV）")
    parser.add_argument("-o", "--output", required=False, help="出力ファイル（省略可）")
    parser.add_argument("--n-fill", choices=["mean", "zero"], default="zero",
                         help="RATIOトークンも取れない真の欠損Nの補完方法: "
                              "zero=ゼロベクトル（default。全クラスタに対して中立で、"
                              "多数派側に誤って偏らない）, mean=座位平均")
    args = parser.parse_args()
    convert_data(args.input, args.output, n_fill_mode=args.n_fill)

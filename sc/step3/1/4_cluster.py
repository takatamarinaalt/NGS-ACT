#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import joblib


def _predict_matching_dtype(model, X):
    """model.predict(X) after casting X to the dtype model was trained with.

    KMeans stores cluster_centers_ in whatever dtype the training data had
    (often float32); a freshly-built feature array here defaults to float64
    (pandas/numpy default), which makes sklearn's Cython predict path raise
    "ValueError: Buffer dtype mismatch, expected 'const float' but got
    'double'" even though the values themselves are fine. Casting to match
    avoids that without changing the predicted labels.
    """
    arr = X.to_numpy() if hasattr(X, "to_numpy") else X
    centers = getattr(model, "cluster_centers_", None)
    if centers is not None:
        arr = arr.astype(centers.dtype, copy=False)
    return model.predict(arr)


def load_model_bundle(model_path):
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        if obj.get("method") == "two_stage_hierarchical":
            # 二段階クラスタリング（stage0→stage1）の統合joblib。
            # 単一の model キーは持たないため None を返し、
            # 実際のpredictは _predict_hierarchical() で行う。
            return None, obj
        if "model" not in obj:
            raise ValueError(f"Loaded object is dict but missing key 'model': {list(obj.keys())}")
        return obj["model"], obj
    return obj, None


def _predict_hierarchical(X: pd.DataFrame, bundle: dict) -> list[int]:
    """
    二段階クラスタリング（stage0=イントロン領域→stage1=遺伝子領域全体）の
    統合joblib（method="two_stage_hierarchical"）を使って、新品種の
    最終的なグローバルクラスタ番号（1始まり、通し番号）を階層的に予測する。

    手順:
      1) X を stage0_feature_cols（イントロン領域の学習列）に合わせて reindex し、
         stage0_model.predict() で一次クラスタ番号（0始まり）を得る
      2) (0始まり結果 + 1) が一次クラスタ番号。対応する stage1_models[一次クラスタ番号] を選択
      3) X を stage1_feature_cols[一次クラスタ番号]（遺伝子領域全体の学習列）に合わせて
         reindex し直し、選択したモデルで .predict() する
      4) 得られたローカルラベルを stage1_label_map[一次クラスタ番号] で
         最終的なグローバルクラスタ番号（フラット番号、int）に変換する

    X は reindex 前の生の特徴量（全ゲノム領域分の列を含む想定）。
    stage0/stage1 いずれの列サブセットも、この X から reindex で切り出す。
    """
    stage0_model = bundle.get("stage0_model")
    stage0_cols  = bundle.get("stage0_feature_cols") or []
    stage1_models = bundle.get("stage1_models", {})
    stage1_cols   = bundle.get("stage1_feature_cols", {})
    stage1_map    = bundle.get("stage1_label_map", {})

    if stage0_model is None:
        raise ValueError("bundle に stage0_model が見つかりません"
                          "（method=two_stage_hierarchical ですが不完全なjoblibです）")

    X_stage0 = X.reindex(columns=stage0_cols, fill_value=0)
    stage0_pred0 = _predict_matching_dtype(stage0_model, X_stage0)  # 0始まり

    results: list[int] = []
    for i, p0 in enumerate(stage0_pred0):
        primary_cluster_id = int(p0) + 1
        stage1_model = stage1_models.get(primary_cluster_id)
        if stage1_model is None:
            raise ValueError(
                f"stage1_models に一次クラスタ {primary_cluster_id} のモデルが見つかりません"
                f"（存在するキー: {sorted(stage1_models.keys())}）"
            )

        cols1 = stage1_cols.get(primary_cluster_id) or []
        X_row = X.iloc[[i]].reindex(columns=cols1, fill_value=0)
        local_label = int(_predict_matching_dtype(stage1_model, X_row)[0])

        label_map = stage1_map.get(primary_cluster_id, {})
        global_label = label_map.get(local_label)
        if global_label is None:
            raise ValueError(
                f"stage1_label_map[{primary_cluster_id}] にローカルラベル {local_label} が"
                f"見つかりません（存在するキー: {sorted(label_map.keys())}）"
            )

        results.append(int(global_label))

    return results


def _norm_chrom(c: str) -> str:
    """'chr07' / '7' / 'Chr7' → '7' に正規化"""
    s = str(c).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    try:
        return str(int(s))
    except ValueError:
        return s


def check_haplotype(sample_name: str, simplified_df,
                    haplotype_info: dict):
    """
    簡略化 TSV の DataFrame（2_vcf_to_calls.py の出力）と、joblib に保存されている
    全てのハプロタイプ変異情報を照合し、一致するハプロタイプ名のリストを返す。
    一致なし / 情報なし の場合は空リスト。

    クラスタ番号には依存せず、haplotype_info に登録されている
    全ハプロタイプ（--haplotype-tsv 由来）を対象に照合する。
    複数ハプロタイプに一致した場合は全て返す。

    simplified_df の形式: chr, posi, <sample>, <sample>, ...
    """
    df = simplified_df
    if df is None or df.empty:
        return []

    if sample_name not in df.columns:
        return []

    if "_chr_norm" not in df.columns:
        df = df.copy()
        df["_chr_norm"] = df["chr"].map(_norm_chrom)

    matched = []
    for hap_name, hap_info in haplotype_info.items():
        variants = hap_info.get("variants", [])
        if not variants:
            continue

        all_match = True
        for v in variants:
            chrom_norm = _norm_chrom(str(v["chrom"]))
            pos        = int(v["pos"])
            vt_kind    = v.get("vt_kind", "UNKNOWN")
            alt        = v.get("alt")

            row = df[(df["_chr_norm"] == chrom_norm) & (df["posi"].astype(int) == pos)]
            if row.empty:
                all_match = False
                break

            val = str(row.iloc[0][sample_name]).strip().upper()

            if vt_kind == "SNP":
                if val != (alt or "").upper():
                    all_match = False; break
            elif vt_kind == "INS":
                if val != "INS":
                    all_match = False; break
            elif vt_kind == "DEL":
                if val != "DEL":
                    all_match = False; break
            else:
                all_match = False; break

        if all_match:
            matched.append(hap_name)

    return matched


def run_clustering_to_txt(input_tsv, model_pkl, output_txt,
                          one_based=True, simplified_tsv=None):
    df = pd.read_csv(input_tsv, sep="\t")
    if "chara_value" not in df.columns:
        raise ValueError("Input TSV must contain 'chara_value' column.")

    feature_cols = [c for c in df.columns if c != "chara_value"]
    X = df[feature_cols]

    model, bundle = load_model_bundle(model_pkl)

    if bundle is not None and bundle.get("method") == "two_stage_hierarchical":
        # 二段階クラスタリングの統合joblib: stage0→stage1の順に階層的にpredict
        global_labels = _predict_hierarchical(X, bundle)  # 常に1始まりのグローバル番号
        labels_out = global_labels if one_based else [v - 1 for v in global_labels]
    else:
        if bundle is not None and "feature_cols" in bundle:
            train_cols = list(bundle["feature_cols"])
            X = X.reindex(columns=train_cols, fill_value=0)

        pred0      = _predict_matching_dtype(model, X)
        labels_out = (pred0 + 1) if one_based else pred0

    df["cluster"] = labels_out

    # --- ハプロタイプ判定 ---
    # joblib に haplotype_info が保存されていれば、新品種ごとに
    # 登録されている全ハプロタイプとの一致を調べる（クラスタ番号には依存しない）
    haplotype_info = bundle.get("haplotype_info", {}) if bundle else {}

    # label: 通常はクラスタ番号、ハプロタイプに一致した場合はハプロタイプ名
    df["label"] = df["cluster"].astype(str)

    if haplotype_info and simplified_tsv:
        try:
            simplified_df = pd.read_csv(simplified_tsv, sep="\t", dtype={"chr": str})
            simplified_df["_chr_norm"] = simplified_df["chr"].map(_norm_chrom)
        except Exception:
            simplified_df = None
        for idx, row in df.iterrows():
            sample = row["chara_value"]
            haps = check_haplotype(sample, simplified_df, haplotype_info)
            if haps:
                df.at[idx, "label"] = "/".join(sorted(haps))

    # --- 出力 ---
    def _label_sort_key(x):
        # 数字のみ → 数値順、それ以外（ハプロタイプ名）→ 数字の後に文字列順
        return (0, int(x), "") if x.isdigit() else (1, 0, x)

    with open(output_txt, "w", encoding="utf-8") as f:
        for label in sorted(df["label"].unique(), key=_label_sort_key):
            header = f"# cluster {label}" if label.isdigit() else f"# {label}"
            f.write(header + "\n")
            for s in df.loc[df["label"] == label, "chara_value"]:
                f.write(f"{s}\n")
            f.write("\n")

    print(f"[OK] Saved clustering result: {output_txt}")
    if haplotype_info:
        print(f"[INFO] Applied haplotype matching (registered haplotypes: {len(haplotype_info)})")
    elif bundle is not None:
        print(f"[INFO] Loaded bundle keys: {list(bundle.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="クラスタリング結果をtxtで保存（bundle/joblib両対応、ハプロタイプ自動判定対応）"
    )
    parser.add_argument("--input_tsv",       required=True,
                        help="特徴量付きTSV（chara_value列を含む）")
    parser.add_argument("--model_pkl",        required=True,
                        help="学習済みモデル（joblib: KMeans単体 or bundle辞書）")
    parser.add_argument("--output_txt",       required=True,
                        help="出力txt")
    parser.add_argument("--simplified-tsv",   default=None,
                        help="2_vcf_to_calls.py の出力TSV（ハプロタイプ判定用。省略時はクラスタ番号のみ）")
    parser.add_argument("--gff",              default=None,
                        help="GFF3 ファイル（joblib の haplotype_info は genomic 座標済みのため現在は使用しない）")
    parser.add_argument("--zero_based",       action="store_true",
                        help="クラスタ番号を0始まりで出力（デフォルトは1始まり）")
    args = parser.parse_args()

    run_clustering_to_txt(
        args.input_tsv,
        args.model_pkl,
        args.output_txt,
        one_based=(not args.zero_based),
        simplified_tsv=args.simplified_tsv,
    )
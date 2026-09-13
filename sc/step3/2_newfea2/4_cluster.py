#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_wide TSV（step2出力の横持ち予測テーブル）を入力として、
クラスタリングとハプロタイプ照合を行う。

[Case 1] Target TSV にポジションが存在する場合:
  - One-hot エンコード → step2 joblib のモデルでクラスタ予測
  - --haplotype-tsv 指定時: DEL/INS エントリのポジションをワイドテーブルで照合
  - 全エントリ完全一致時のみクラスタラベルをハプロタイプ名で上書き

[Case 2] Target TSV にポジションが存在しない場合:
  - step2 joblib から gene_region を取得
  - 1_predict_combined.py の予測ロジックで全BAMを再予測
  - 予測結果を横持ちワイドテーブルに変換
  - KMeans を再学習してクラスタリング
  - ハプロタイプ照合を実施
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans


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


# ─────────────────────────────────────────────
# One-hot エンコード設定
# ─────────────────────────────────────────────
_MODE_CONFIG = {
    "del": {
        "classes": ["NONE", "DEL"],
        "map": {"NONE": 0, "DEL": 1, "DEL_ST": 1, "DEL_MID": 1, "DEL_END": 1},
    },
    "ins": {
        "classes": ["NONE", "INS"],
        "map": {"NONE": 0, "INS": 1, "INS_SHORT": 1, "INS_LONG": 1},
    },
    "all": {
        "classes": ["NONE", "INS", "DEL"],
        "map": {
            "NONE": 0,
            "INS": 1, "INS_SHORT": 1, "INS_LONG": 1,
            "DEL": 2, "DEL_ST": 2, "DEL_MID": 2, "DEL_END": 2,
        },
    },
}


# ─────────────────────────────────────────────
# 1_predict_combined.py の動的インポート
# ─────────────────────────────────────────────
def _load_predict_module():
    script_dir = Path(__file__).resolve().parent
    predict_path = script_dir.parent.parent / "step2" / "1_predict_combined.py"
    if not predict_path.exists():
        raise FileNotFoundError(f"1_predict_combined.py が見つかりません: {predict_path}")
    spec = importlib.util.spec_from_file_location("predict_combined_mod", str(predict_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────
# One-hot エンコード
# ─────────────────────────────────────────────
def one_hot_encode(wide_df: pd.DataFrame, mode: str = "del") -> pd.DataFrame:
    """
    wide_df（chr, posi, sample...）→ one-hot TSV（chara_value + 特徴量列）
    """
    cfg = _MODE_CONFIG[mode]
    classes = cfg["classes"]
    type_map = cfg["map"]
    n_classes = len(classes)

    sample_cols = [c for c in wide_df.columns if c not in ("chr", "posi")]
    loci = [f"{r.chr}_{r.posi}" for r in wide_df[["chr", "posi"]].itertuples(index=False)]
    n_loci = len(loci)

    mat = wide_df[sample_cols].astype(str).apply(
        lambda col: col.str.strip().str.upper().str.replace("-", "_", regex=False)
    )
    codes = mat.apply(lambda col: col.map(lambda x: type_map.get(x, -1)))
    codes_np = codes.to_numpy(dtype=np.int16).T  # (n_samples, n_loci)
    n_samples = len(sample_cols)

    out = np.zeros((n_samples, n_loci * n_classes), dtype=np.float32)
    for cls_idx in range(n_classes):
        mask = codes_np == cls_idx
        s_idx, l_idx = np.where(mask)
        out[s_idx, l_idx * n_classes + cls_idx] = 1.0

    headers = [f"{loci[li]}_{classes[ci]}"
               for li in range(n_loci) for ci in range(n_classes)]
    out_df = pd.DataFrame(out, columns=headers, dtype=np.float32)
    out_df.insert(0, "chara_value", sample_cols)
    return out_df


# ─────────────────────────────────────────────
# 重み付け（学習時と同仕様）
# ─────────────────────────────────────────────
def apply_weights(X: pd.DataFrame, weights_info: dict) -> pd.DataFrame:
    if not weights_info:
        return X
    indel_w = float(weights_info.get("indel_weight", 3.0))
    del_w   = float(weights_info.get("del_weight",   indel_w))
    ins_w   = float(weights_info.get("ins_weight",   indel_w))
    short_w = float(weights_info.get("short_weight", indel_w))
    Xw = X.copy()
    for c in Xw.columns:
        cu = str(c).upper()
        if "_SHORT_" in cu:
            Xw[c] *= short_w
        elif "INS" in cu:
            Xw[c] *= ins_w
        elif "DEL" in cu and "DEL_MID" not in cu:
            Xw[c] *= del_w
    return Xw


# ─────────────────────────────────────────────
# ハプロタイプ TSV 読み込み（CDS 形式）
# ─────────────────────────────────────────────
def _load_coord_converter():
    """coord_converter.py を動的読み込み。見つからない場合は None を返す。"""
    import importlib.util
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "coord_converter.py",
        script_dir.parent.parent / "step1" / "coord_converter.py",
        script_dir.parent / "cds" / "coord_converter.py",
    ]
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("coord_converter", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def load_haplotype_defs(path: str, gff_path: str | None = None) -> dict:
    """
    gene_model_make/cds/ 形式 TSV を読み込み、
    ハプロタイプ定義 dict を返す。
    - SNP エントリはスキップ
    - coord_type=genomic: 直接使用
    - coord_type=cds/aa: gff_path 指定時に coord_converter で変換（DEL/INS のみ）

    Returns:
        {hap_name: [{"pos": int, "kind": "DEL"|"INS"}, ...], ...}
    """
    conv_mod = None
    gene_data = None
    name_to_id = None
    if gff_path:
        conv_mod = _load_coord_converter()
        if conv_mod:
            try:
                gene_data, name_to_id = conv_mod.parse_gff(gff_path)
                print(f"[INFO] Successfully loaded GFF: {gff_path}")
            except Exception as e:
                print(f"[WARN] Error loading GFF: {e}", file=sys.stderr)
                conv_mod = None
        else:
            print("[WARN] coord_converter.py not found. Non-genomic coordinates will be skipped.",
                  file=sys.stderr)

    if path is None:
        return {}
    if not Path(path).exists():
        print(f"[WARN] --haplotype-tsv not found: {path}"
              f" (skipping haplotype matching, outputting normal cluster numbers only)", file=sys.stderr)
        return {}

    df = pd.read_csv(path, sep="\t", dtype=str, comment="#").fillna("")
    hap_defs = defaultdict(list)

    for _, row in df.iterrows():
        vt = str(row.get("variant_type", "")).strip().upper()
        if vt.startswith("SNP"):
            continue

        name = str(row.get("name", "")).strip()
        if not name:
            continue

        coord_type = str(row.get("coord_type", "")).strip().lower()
        pos_str    = str(row.get("pos", "")).strip()
        gene_id    = str(row.get("gene_id", "")).strip()

        if coord_type in ("genome", "genomic"):
            # "chr07:9155047" 形式（染色体付き）と、素の整数（従来互換）の両方に対応
            pos_tail = pos_str.rsplit(":", 1)[-1] if ":" in pos_str else pos_str
            try:
                pos = int(pos_tail)
            except ValueError:
                print(f"[WARN] Haplotype '{name}': cannot convert pos='{pos_str}' to an integer. Skipping.",
                      file=sys.stderr)
                continue
        elif conv_mod and gene_data is not None:
            try:
                pos_int = int(pos_str)
            except ValueError:
                print(f"[WARN] Haplotype '{name}': cannot convert pos='{pos_str}' to an integer. Skipping.",
                      file=sys.stderr)
                continue
            resolved_id = conv_mod.resolve_gene_id(gene_id, gene_data, name_to_id)
            result = conv_mod.convert(resolved_id, coord_type, pos_int, gene_data)
            if result["status"] != "OK":
                print(f"[WARN] Haplotype '{name}': coordinate conversion failed ({coord_type} {pos_str})"
                      f" -> {result['message']}", file=sys.stderr)
                continue
            pos = result["genome_pos"]
        else:
            print(f"[WARN] Haplotype '{name}': skipping coord_type='{coord_type}'"
                  f" (--gff not specified or coord_converter.py not found)", file=sys.stderr)
            continue

        if vt.startswith("DEL"):
            hap_defs[name].append({"pos": pos, "kind": "DEL"})
        elif vt.startswith("INS"):
            hap_defs[name].append({"pos": pos, "kind": "INS"})

    print(f"[INFO] Loaded haplotype definitions (TSV): {len(hap_defs)} haplotype(s)")
    return dict(hap_defs)


def haplotype_info_to_hap_defs(haplotype_info: dict) -> dict:
    """
    joblib に保存されている haplotype_info 形式を
    check_haplotypes() が使う hap_defs 形式に変換する。

    joblib 形式:
      {hap_name: {"variants": [{"vt_kind": "DEL"|"INS", "chrom": ..., "pos": int}, ...]}}
    hap_defs 形式:
      {hap_name: [{"pos": int, "kind": "DEL"|"INS"}, ...]}
    """
    hap_defs = {}
    for hap_name, hap_data in haplotype_info.items():
        variants = hap_data.get("variants", []) if isinstance(hap_data, dict) else []
        entries = []
        for v in variants:
            vt_kind = str(v.get("vt_kind", "")).upper()
            pos = v.get("pos")
            if pos is None or vt_kind not in ("DEL", "INS"):
                continue
            entries.append({"pos": int(pos), "kind": vt_kind})
        if entries:
            hap_defs[hap_name] = entries
    return hap_defs


# ─────────────────────────────────────────────
# ハプロタイプ照合
# ─────────────────────────────────────────────
_DEL_VALS  = {"DEL", "DEL_ST", "DEL_END", "DEL_MID"}
_INS_VALS  = {"INS", "INS_SHORT", "INS_LONG"}

def check_haplotypes(sample: str, wide_df: pd.DataFrame, hap_defs: dict) -> list:
    """
    ワイドテーブルとハプロタイプ定義を照合し、一致するハプロタイプ名のリストを返す。
    全エントリが一致した場合のみ一致と見なす。
    INS/DEL はブレークポイント推定のズレを考慮し、position ±2bp 以内の一致を許容する。
    """
    if sample not in wide_df.columns:
        return []

    posi_index = wide_df["posi"].astype(int)

    matched = []
    for hap_name, entries in hap_defs.items():
        if not entries:
            continue
        all_match = True
        for entry in entries:
            pos = entry["pos"]
            kind = entry["kind"]

            row_mask = (posi_index >= pos - 2) & (posi_index <= pos + 2)
            if not row_mask.any():
                all_match = False
                break

            vals = wide_df.loc[row_mask, sample].astype(str).str.strip().str.upper()

            if kind == "DEL" and not vals.isin(_DEL_VALS).any():
                all_match = False
                break
            if kind == "INS" and not vals.isin(_INS_VALS).any():
                all_match = False
                break

        if all_match:
            matched.append(hap_name)

    return matched


# ─────────────────────────────────────────────
# 予測結果 → ワイドテーブル変換
# ─────────────────────────────────────────────
def _norm_type(t: str) -> str:
    """DEL_st / DEL_end → DEL、INS_short / INS_long → INS に正規化。"""
    u = t.strip().upper().replace("-", "_")
    if u.startswith("DEL"):
        return "DEL"
    if u.startswith("INS"):
        return "INS"
    return "NONE"


def pair_del_boundaries_rows(rows: list) -> list:
    """
    rows: 1サンプル分の (CHROM, POS, TYPE) タプルのリスト。

    DEL_st / DEL_end を染色体ごとにPOS昇順でスタック方式ペアリングし、
    ペアが成立した2ポジションだけを TYPE="DEL" として残す。
    DEL_mid、および相方が見つからなかった DEL_st / DEL_end は除外する。
    それ以外（NONE, INS等）はそのまま返す。
    """
    if not rows:
        return rows
    df = pd.DataFrame(rows, columns=["CHROM", "POS", "TYPE"])
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
                # 相方がいなければこのDEL_endは除外

    del_pairs_df = df.loc[keep_indices].copy()
    if len(del_pairs_df) > 0:
        del_pairs_df["TYPE"] = "DEL"

    result = pd.concat([non_del_df, del_pairs_df], ignore_index=True)
    return list(result[["CHROM", "POS", "TYPE"]].itertuples(index=False, name=None))


def merge_predictions_to_wide(pred_dir: Path, sample_names: list,
                               mode: str, fill: str = "NONE") -> pd.DataFrame:
    """
    per-sample {SAMPLE}_DEL.tsv / {SAMPLE}_INS.tsv → ワイドテーブル
    少なくとも1サンプルが DEL/INS を持つポジションのみ保持する。
    DEL_st/DEL_end は pair_del_boundaries_rows() でペアリングしてからDEL判定する。
    """
    per_sample: dict[str, dict] = {}
    all_keys: set[tuple] = set()

    suffixes = []
    if mode in ("del", "all"):
        suffixes.append("DEL")
    if mode in ("ins", "all"):
        suffixes.append("INS")

    for sample in sample_names:
        raw_rows = []
        for sv in suffixes:
            tsv = pred_dir / f"{sample}_{sv}.tsv"
            if not tsv.exists():
                continue
            try:
                df = pd.read_csv(tsv, sep="\t", dtype={"CHROM": str})
            except Exception:
                continue
            for _, row in df.iterrows():
                raw_rows.append((str(row["CHROM"]), int(row["POS"]), str(row.get("TYPE", "NONE"))))

        paired_rows = pair_del_boundaries_rows(raw_rows)

        sample_vals: dict[tuple, str] = {}
        for chr_raw, posi, raw_type in paired_rows:
            norm = _norm_type(raw_type)
            if norm == "NONE":
                continue
            key = (chr_raw, posi)
            # DEL/INS が両方ある場合は先着優先
            if key not in sample_vals:
                sample_vals[key] = norm
            all_keys.add(key)
        per_sample[sample] = sample_vals

    if not all_keys:
        return pd.DataFrame(columns=["chr", "posi"] + sample_names)

    # chr を正規化（"chr01" → "1" のような形式で統一）
    def _norm_chr(c: str) -> str:
        s = c.strip()
        if s.lower().startswith("chr"):
            s = s[3:]
        try:
            return str(int(s))
        except ValueError:
            return s

    rows = []
    for chr_raw, posi in sorted(all_keys, key=lambda x: (_norm_chr(x[0]), x[1])):
        chr_norm = _norm_chr(chr_raw)
        row = {"chr": chr_norm, "posi": posi}
        for sample in sample_names:
            row[sample] = per_sample[sample].get((chr_raw, posi), fill)
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# クラスタ txt 出力
# ─────────────────────────────────────────────
def load_gene_absent_samples(path) -> set:
    """1_predict_combined.py が出力する gene_absent_samples.txt（走査した全座位で
    生リード深度が0だったサンプル名、1行1サンプル）を読み込む。path が
    None／存在しない／空の場合は空集合を返す（--gene-absent-file 省略時や、
    該当サンプルが1つも無かった場合の通常のケース）。"""
    if not path or not Path(path).is_file():
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip()}


def write_cluster_txt(labels: list, sample_names: list, output_path: Path,
                      wide_df: pd.DataFrame, hap_defs: dict,
                      gene_absent_samples: set = None) -> None:
    """
    labels と sample_names からクラスタ txt を出力。
    hap_defs が空でなければハプロタイプ照合を実施し、ラベルを上書きする。
    gene_absent_samples が指定されていれば、該当サンプルのラベルを
    "GENE_ABSENT" で最終上書きする（ハプロタイプ照合より後に適用し、
    そちらより優先させる -- 走査した全座位でリード深度が0というのは、
    たまたまのハプロタイプ一致より強いシグナルのため）。
    """
    df = pd.DataFrame({"sample": sample_names, "label": [str(l) for l in labels]})

    if hap_defs and wide_df is not None and not wide_df.empty:
        for idx, row in df.iterrows():
            haps = check_haplotypes(row["sample"], wide_df, hap_defs)
            if haps:
                df.at[idx, "label"] = "/".join(sorted(haps))

    if gene_absent_samples:
        for idx, row in df.iterrows():
            if row["sample"].upper() in gene_absent_samples:
                df.at[idx, "label"] = "GENE_ABSENT"

    def _sort_key(x):
        return (0, int(x), "") if x.isdigit() else (1, 0, x)

    with open(output_path, "w", encoding="utf-8") as f:
        for label in sorted(df["label"].unique(), key=_sort_key):
            header = f"# cluster {label}" if label.isdigit() else f"# {label}"
            f.write(header + "\n")
            for s in df.loc[df["label"] == label, "sample"]:
                f.write(f"{s}\n")
            f.write("\n")

    n_hap = df["label"].apply(lambda x: not x.isdigit() and x != "GENE_ABSENT").sum()
    n_gene_absent = (df["label"] == "GENE_ABSENT").sum()
    print(f"[OK] Saved: {output_path}")
    if hap_defs:
        print(f"[INFO] Haplotype matching: {n_hap} sample(s) matched")
    if n_gene_absent:
        print(f"[INFO] GENE_ABSENT: {n_gene_absent} sample(s)")


# ─────────────────────────────────────────────
# 新規ポジション検出・未使用クラスタ番号割当
# （sc/step3/1/4b_novel_cluster.py の新規多型クラスタと同じ考え方を、
#   DEL/INS の one-hot ベースの割当に適用したもの）
# ─────────────────────────────────────────────

def _load_known_positions(targets_path: str | None) -> set[tuple[str, int]]:
    """--targets TSV（chr, posi 列）から {(chr, posi)} の集合を返す。
    未指定/読み込み失敗時は空集合（＝この場合、全座位が「新規」扱いになって
    しまうため、呼び出し側で --targets が実際に渡っているか確認しておくこと）。"""
    if not targets_path or not Path(targets_path).exists():
        return set()
    try:
        df = pd.read_csv(targets_path, sep="\t", dtype=str)
    except Exception:
        return set()
    known: set[tuple[str, int]] = set()
    for _, row in df.iterrows():
        try:
            known.add((str(row["chr"]).strip(), int(str(row["posi"]).strip())))
        except (KeyError, ValueError):
            continue
    return known


def _find_novel_positions(wide_df: pd.DataFrame, mode: str,
                          known_positions: set[tuple[str, int]]) -> dict[str, frozenset]:
    """wide_df の各座位のうち known_positions に含まれないもの（＝学習時の
    targets 外の座位）について、そこで実際に DEL/INS 判定（NONE 以外）が
    出ているサンプルを集め、{sample: frozenset of (chr, posi)} を返す。"""
    cfg = _MODE_CONFIG[mode]
    type_map = cfg["map"]
    sample_cols = [c for c in wide_df.columns if c not in ("chr", "posi")]
    novel: dict[str, set] = {}
    for _, row in wide_df.iterrows():
        key = (str(row["chr"]).strip(), int(row["posi"]))
        if key in known_positions:
            continue
        for s in sample_cols:
            val = str(row[s]).strip().upper().replace("-", "_")
            if type_map.get(val, -1) >= 1:
                novel.setdefault(s, set()).add(key)
    return {s: frozenset(v) for s, v in novel.items()}


def _apply_novel_overrides(labels: list, sample_cols: list, novel_map: dict[str, frozenset],
                           known_max_cluster: int | None = None) -> list:
    """novel_map に載っているサンプル（targets外に実際のDEL/INSが見つかった
    サンプル）のラベルを、既存クラスタ番号の最大値+1から始まる未使用番号に
    差し替える。同一の新規ポジション集合を持つサンプルは同じ新番号にする
    （4b_novel_cluster.py の assign_labels() と同じグルーピング方針）。

    known_max_cluster: 学習時に実際に存在したクラスタ数の最大値（bundleの
    labels_sub から取得）。これを渡さず labels（今回判定した新サンプルだけの
    ラベル）から max を取ると、新サンプルの数が少ない時に既存の実クラスタ番号と
    衝突しうる（例: 学習時にクラスタ1〜3が存在するのに、新サンプル1件が
    たまたまクラスタ1と判定された場合、max(labels)=1から次の番号を2にすると
    既存のクラスタ2と衝突する）。必ずこちらを優先して使うこと。"""
    if not novel_map:
        return labels
    updated = list(labels)
    max_cluster = known_max_cluster if known_max_cluster is not None else max(labels, default=0)
    fs_to_label: dict[frozenset, int] = {}
    next_cluster = max_cluster + 1
    for i, sample in enumerate(sample_cols):
        fs = novel_map.get(sample)
        if fs:
            if fs not in fs_to_label:
                fs_to_label[fs] = next_cluster
                next_cluster += 1
            updated[i] = fs_to_label[fs]
    return updated


# ─────────────────────────────────────────────
# Case 1: ワイドテーブルがある場合
# ─────────────────────────────────────────────
def run_case1(wide_df: pd.DataFrame, bundle: dict, args, hap_defs: dict) -> None:
    print(f"[INFO] Case 1: {len(wide_df)} position(s) x {len(wide_df.columns) - 2} sample(s)")

    encoded = one_hot_encode(wide_df, args.mode)
    sample_cols = [c for c in wide_df.columns if c not in ("chr", "posi")]

    feature_cols = [c for c in encoded.columns if c != "chara_value"]

    model = bundle.get("sub_kmeans_model") if bundle else None

    if model is not None:
        train_cols = bundle.get("sub_feature_names") or feature_cols
        X = encoded[feature_cols].reindex(columns=train_cols, fill_value=0.0)
        if not args.no_apply_weights:
            weights_info = bundle.get("weights_info", {}) if bundle else {}
            X = apply_weights(X, weights_info)
        pred = _predict_matching_dtype(model, X)
        labels = [(int(p) + 1) for p in pred]
        print("[INFO] Cluster prediction complete using existing model")
    else:
        # 学習時にこのsvtypeで有意な多型が見つからず、KMeansモデル自体が
        # 保存されていない（＝学習コホート全体でこのsvtypeの多型が実質無かった）
        # ことを意味するため、新しいサンプルに対してKMeansを再学習することは
        # しない（サンプル数が少ないと n_samples < n_clusters でクラッシュする
        # うえ、そもそも「学習時に確立されたクラスタ」自体が存在しない）。
        # 全サンプルを暫定的にクラスタ1とし、下の新規ポジション判定で実際に
        # 多型を検出できたサンプルだけを区別する。
        print("[INFO] No model (no significant polymorphism was found during training). "
              "Not retraining KMeans; provisionally assigning all samples to cluster 1.")
        labels = [1] * len(sample_cols)

    # 新規ポジション（学習時のtargets外でDEL/INSが実際に検出された座位）の検出と、
    # そのサンプルへの未使用クラスタ番号の割当。モデルの有無に関わらず行う
    # （モデルが無い場合の「クラスタ1」も、新規ポジションが見つかったサンプルは
    # ここで区別される）。--targets が渡されなかった場合（known_positionsが空）は
    # 誤って全座位を新規扱いしないようスキップする。
    if args.targets:
        known_positions = _load_known_positions(args.targets)
        novel_map = _find_novel_positions(wide_df, args.mode, known_positions)
        novel_samples = {s for s, v in novel_map.items() if v}
        if novel_samples:
            print(f"\n[INFO] Sample(s) with novel {args.mode.upper()} position(s) (outside targets): "
                  f"{len(novel_samples)} sample(s)")
            for s in sorted(novel_samples):
                print(f"  {s}: {sorted(novel_map[s])}")
            # 学習時に実際に存在したクラスタ数（bundleのlabels_subの最大値）を
            # 基準に未使用番号を割り当てる。今回判定した新サンプルだけの
            # labelsから最大値を取ると、既存の実クラスタ番号と衝突しうるため。
            labels_sub_train = bundle.get("labels_sub") if bundle else None
            known_max_cluster = (int(max(labels_sub_train))
                                 if labels_sub_train is not None and len(labels_sub_train) > 0
                                 else None)
            labels = _apply_novel_overrides(labels, sample_cols, novel_map, known_max_cluster)
    else:
        print("[INFO] --targets not specified, skipping novel position detection.")

    write_cluster_txt(labels, sample_cols, Path(args.output_txt), wide_df, hap_defs,
                      gene_absent_samples=load_gene_absent_samples(args.gene_absent_file))


# ─────────────────────────────────────────────
# Case 2: ワイドテーブルが空の場合
# ─────────────────────────────────────────────
def run_case2(bundle: dict, args, hap_defs: dict) -> None:
    gene_region = bundle.get("gene_region") if bundle else None
    if not gene_region:
        print("[ERROR] No gene_region saved in step2 joblib. "
              "Please specify it directly with --region.", file=sys.stderr)
        sys.exit(1)

    if not args.bam:
        print("[ERROR] --bam is required for Case 2 prediction.", file=sys.stderr)
        sys.exit(1)

    if not args.model_del and not args.model_ins:
        print("[ERROR] --model-del or --model-ins is required for Case 2 prediction.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Case 2: starting re-prediction with gene_region={gene_region}")

    # 1_predict_combined.py をインポート
    try:
        pm = _load_predict_module()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # BAM ファイル展開
    bam_files = pm.expand_bam_inputs(args.bam)
    if not bam_files:
        print("[ERROR] No BAM files found.", file=sys.stderr)
        sys.exit(1)
    name_map = pm.dedup_sample_names_upper(bam_files)

    # 予測モデル読み込み
    model_del = joblib.load(args.model_del) if args.model_del else None
    model_ins = joblib.load(args.model_ins) if args.model_ins else None

    # 出力先
    pred_dir = Path(args.outdir_pred) if args.outdir_pred else \
        Path(args.output_txt).parent / "_pred_case2"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 予測実行
    sample_names = []
    for bam in bam_files:
        sample = name_map[bam]
        sample_names.append(sample)
        out_del = str(pred_dir / f"{sample}_DEL.tsv") if model_del else None
        out_ins = str(pred_dir / f"{sample}_INS.tsv") if model_ins else None

        pm.predict_combined(
            bam_path=bam,
            model_del=model_del,
            model_ins=model_ins,
            model_del_path=args.model_del,
            model_ins_path=args.model_ins,
            out_del=out_del,
            out_ins=out_ins,
            out_del_feat=None,
            out_ins_feat=None,
            out_del_raw=None,
            out_ins_raw=None,
            step=args.step,
            threshold=args.threshold,
            target_chroms=None,
            region=gene_region,
            n_jobs=args.jobs,
            targets_tsv=None,
        )

    # ワイドテーブルに変換
    wide_df = merge_predictions_to_wide(pred_dir, sample_names, args.mode)
    print(f"[INFO] Wide table: {len(wide_df)} position(s) x {len(sample_names)} sample(s)")

    if wide_df.empty or len(wide_df) == 0:
        print("[WARN] Still 0 positions after re-prediction. Assigning all samples to cluster 1.")
        labels = [1] * len(sample_names)
        write_cluster_txt(labels, sample_names, Path(args.output_txt), wide_df, hap_defs,
                          gene_absent_samples=load_gene_absent_samples(args.gene_absent_file))
        return

    # One-hot エンコード
    encoded = one_hot_encode(wide_df, args.mode)
    feature_cols = [c for c in encoded.columns if c != "chara_value"]

    # KMeans 再学習
    n_clusters = args.n_clusters or 2
    print(f"[INFO] Retraining KMeans (k={n_clusters})")
    X = encoded[feature_cols].fillna(0.0)
    if not args.no_apply_weights and bundle:
        X = apply_weights(X, bundle.get("weights_info", {}))
    km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
    pred = km.fit_predict(X.to_numpy())
    labels = [(int(p) + 1) for p in pred]

    write_cluster_txt(labels, sample_names, Path(args.output_txt), wide_df, hap_defs,
                      gene_absent_samples=load_gene_absent_samples(args.gene_absent_file))


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="step3_wide TSV からクラスタリングとハプロタイプ照合を行う。"
    )

    # 共通
    ap.add_argument("--input-tsv",   required=True,
                    help="step3_wide TSV（chr, posi, サンプル... / DEL or INS ワイドテーブル）")
    ap.add_argument("--model-pkl",   default=None,
                    help="step2 の joblib（既存 KMeans モデルと gene_region を含む）")
    ap.add_argument("--output-txt",  required=True,
                    help="出力クラスタ txt")
    ap.add_argument("--haplotype-tsv", default=None,
                    help="ハプロタイプ定義 TSV（gene_model_make/cds/ 形式。SNP はスキップ）")
    ap.add_argument("--gene-absent-file", default=None,
                    help="1_predict_combined.py が出力した gene_absent_samples.txt。"
                         "列挙されたサンプルのラベルを GENE_ABSENT に上書きする"
                         "（走査した全座位で生リード深度が0だった＝遺伝子が"
                         "存在しないと推定されるサンプル）")
    ap.add_argument("--gff", default=None,
                    help="GFF3 ファイル（--haplotype-tsv の CDS/AA 座標を genomic に変換する場合に指定）")
    ap.add_argument("--mode", choices=["del", "ins", "all"], default="del",
                    help="エンコードモード: del=NONE/DEL, ins=NONE/INS, all=NONE/INS/DEL（default: del）")
    ap.add_argument("--n-clusters", type=int, default=None,
                    help="KMeans クラスタ数（モデルなし時 or Case 2 の再学習時に使用。default: 2）")
    ap.add_argument("--no-apply-weights", action="store_true",
                    help="joblib の weights_info に基づく重み付けを適用しない")
    ap.add_argument("--targets", default=None,
                    help="学習時のtargets座標TSV（chr, posi列）。指定すると --input-tsv の座位の"
                         "うちこのtargets外にあるものを「新規ポジション」として検出し、既存クラスタ"
                         "への割当ではなく未使用のクラスタ番号を割り当てる"
                         "（sc/step3/1/4b_novel_cluster.py の新規多型クラスタと同じ考え方）。")

    # Case 2 用（ポジションなし時の再予測に必要）
    ap.add_argument("--bam", nargs="+", default=None,
                    help="[Case 2] BAM ファイルまたはディレクトリ（複数可）")
    ap.add_argument("--model-del", default=None,
                    help="[Case 2] DEL 予測モデル (.joblib)")
    ap.add_argument("--model-ins", default=None,
                    help="[Case 2] INS 予測モデル (.joblib)")
    ap.add_argument("--step", type=int, default=50,
                    help="[Case 2] スキャンステップ bp（default: 50）")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="[Case 2] 予測確率の閾値（default: 0.6）")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="[Case 2] 並列処理数（default: 1）")
    ap.add_argument("--outdir-pred", default=None,
                    help="[Case 2] 再予測 TSV の一時保存先（省略時は output-txt の隣に _pred_case2/）")

    args = ap.parse_args()

    # joblib 読み込み
    bundle = None
    if args.model_pkl:
        try:
            bundle = joblib.load(args.model_pkl)
            if not isinstance(bundle, dict):
                bundle = None
        except Exception as e:
            print(f"[WARN] Error loading joblib: {e}", file=sys.stderr)

    # ハプロタイプ定義読み込み: joblib 優先、なければ TSV をフォールバック
    hap_defs = {}
    if bundle and bundle.get("haplotype_info"):
        hap_defs = haplotype_info_to_hap_defs(bundle["haplotype_info"])
        print(f"[INFO] Loaded haplotype_info from joblib ({len(hap_defs)} haplotype(s))")
    elif args.haplotype_tsv:
        hap_defs = load_haplotype_defs(args.haplotype_tsv, gff_path=getattr(args, "gff", None))

    # step3_wide TSV 読み込み
    wide_df = pd.read_csv(args.input_tsv, sep="\t", dtype={"chr": str})
    sample_cols = [c for c in wide_df.columns if c not in ("chr", "posi")]

    # ポジションの有無で分岐
    has_positions = len(wide_df) > 0 and len(sample_cols) > 0

    if has_positions:
        run_case1(wide_df, bundle, args, hap_defs)
    else:
        print("[INFO] No positions in step3_wide TSV. Falling back to Case 2 (re-prediction).")
        run_case2(bundle, args, hap_defs)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- helpers ----------

def _load_coord_converter(script_dir: Path):
    """coord_converter.py を動的に読み込む。見つからなければ None を返す。"""
    for candidate in [
        script_dir / "coord_converter.py",
        script_dir.parent / "cds" / "coord_converter.py",
        script_dir / "cds" / "coord_converter.py",
    ]:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("coord_converter", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def _norm_chrom(c: str) -> str:
    """'chr07' / '7' / 'Chr7' → '7' に正規化"""
    s = str(c).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    try:
        return str(int(s))
    except ValueError:
        return s


def check_haplotype(sample_name: str, simplified_tsv: str, haplotype_info: dict) -> list:
    """
    wide table TSV（chr, posi, <sample>...）とハプロタイプ情報を照合し、
    一致するハプロタイプ名のリストを返す。一致なしは空リスト。
    vt_kind:
      SNP        → exact position + ALT 一致（chrom あり）
      INS        → position ±2bp のいずれかに INS が予測されている
      DEL        → position ±2bp のいずれかに DEL が予測されている
      DEL_region → region_start〜region_end 内に DEL が1件以上ある
    chrom が None の場合は染色体チェックをスキップする（cds 形式由来エントリ）。
    """
    try:
        df = pd.read_csv(simplified_tsv, sep="\t", dtype={"chr": str})
    except Exception:
        return []

    if sample_name not in df.columns:
        return []

    df["_chr_norm"] = df["chr"].map(_norm_chrom)

    matched = []
    for hap_name, hap_info in haplotype_info.items():
        variants = hap_info.get("variants", [])
        if not variants:
            continue

        all_match = True
        for v in variants:
            vt_kind   = v.get("vt_kind", "UNKNOWN")
            chrom_raw = v.get("chrom")
            use_chrom = chrom_raw is not None

            if vt_kind == "DEL_region":
                chrom_norm   = _norm_chrom(str(chrom_raw)) if use_chrom else None
                region_start = int(v["region_start"])
                region_end   = int(v["region_end"])
                mask = (df["posi"].astype(int) >= region_start) & (df["posi"].astype(int) <= region_end)
                if use_chrom:
                    mask &= (df["_chr_norm"] == chrom_norm)
                rows = df[mask]
                if rows.empty or not rows[sample_name].astype(str).str.strip().str.upper().eq("DEL").any():
                    all_match = False
                    break
            else:
                pos  = int(v["pos"])
                alt  = v.get("alt")
                if vt_kind in ("INS", "DEL"):
                    # INS/DEL はブレークポイント推定のズレを考慮し前後2bpを許容する
                    posi_int = df["posi"].astype(int)
                    mask = (posi_int >= pos - 2) & (posi_int <= pos + 2)
                else:
                    mask = df["posi"].astype(int) == pos
                if use_chrom:
                    chrom_norm = _norm_chrom(str(chrom_raw))
                    mask &= (df["_chr_norm"] == chrom_norm)
                rows = df[mask]
                if rows.empty:
                    all_match = False
                    break
                if vt_kind == "SNP":
                    val = str(rows.iloc[0][sample_name]).strip().upper()
                    if val != (alt or "").upper():
                        all_match = False; break
                elif vt_kind == "INS":
                    vals = rows[sample_name].astype(str).str.strip().str.upper()
                    if not vals.eq("INS").any():
                        all_match = False; break
                elif vt_kind == "DEL":
                    vals = rows[sample_name].astype(str).str.strip().str.upper()
                    if not vals.eq("DEL").any():
                        all_match = False; break
                else:
                    all_match = False; break

        if all_match:
            matched.append(hap_name)

    return matched


def _load_haplotype_tsv_cds(hap_df: "pd.DataFrame",
                             conv_mod=None, gene_data: dict | None = None,
                             name_to_id: dict | None = None) -> dict:
    """
    gene_model_make/cds/ 形式 TSV を haplotype_info 形式に変換する。
    SNP はスキップ。coord_type=genomic は pos を整数として直接使用（chrom=None）。
    coord_type が genomic 以外（cds / aa / gdna）は conv_mod で座標変換する。
    conv_mod が None の場合、非 genomic エントリは WARN でスキップ。

    cds 形式列: gene_id, coord_type, pos, variant_type, name, reference
    """
    import sys
    haplotype_info: dict = {}
    for _, row in hap_df.iterrows():
        if not str(row.get("name", "")).strip() and not str(row.get("variant_type", "")).strip():
            continue
        vt = str(row.get("variant_type", "")).strip().upper()
        if vt.startswith("SNP"):
            continue
        coord_type = str(row.get("coord_type", "")).strip().lower()
        hap_name = str(row.get("name", "")).strip()
        if not hap_name:
            continue
        pos_str = str(row.get("pos", "")).strip()
        gene_id = str(row.get("gene_id", "")).strip()

        if vt.startswith("DEL"):
            vt_kind = "DEL"
        elif vt.startswith("INS"):
            vt_kind = "INS"
        else:
            continue

        if coord_type in ("genome", "genomic"):
            # "chr07:9155047" 形式（染色体付き）と、素の整数（染色体なし・従来互換）の両方に対応
            if ":" in pos_str:
                chrom_str, pos_tail = pos_str.rsplit(":", 1)
                try:
                    pos = int(pos_tail)
                except ValueError:
                    print(f"[WARN] haplotype-tsv: '{hap_name}' pos='{pos_str}' is not an integer, skipped",
                          file=sys.stderr)
                    continue
                chrom = _norm_chrom(chrom_str)
            else:
                try:
                    pos = int(pos_str)
                except ValueError:
                    print(f"[WARN] haplotype-tsv: '{hap_name}' pos='{pos_str}' is not an integer, skipped",
                          file=sys.stderr)
                    continue
                chrom = None
            v = {"vt_kind": vt_kind, "chrom": chrom, "pos": pos}
        else:
            # cds / aa / gdna → coord_converter で座標変換
            if conv_mod is None or gene_data is None:
                print(f"[WARN] haplotype-tsv: '{hap_name}' coord_type='{coord_type}' skipped (--gff not specified)",
                      file=sys.stderr)
                continue
            try:
                pos_int = int(pos_str)
            except ValueError:
                print(f"[WARN] haplotype-tsv: '{hap_name}' pos='{pos_str}' is not an integer, skipped",
                      file=sys.stderr)
                continue
            resolved_id = conv_mod.resolve_gene_id(gene_id, gene_data, name_to_id or {})
            result = conv_mod.convert(resolved_id, coord_type, pos_int, gene_data)
            if result["status"] != "OK":
                print(f"[WARN] haplotype-tsv: '{hap_name}' coordinate conversion failed → {result['message']}",
                      file=sys.stderr)
                continue
            chrom = _norm_chrom(result["chrom"])
            pos = result["genome_pos"]
            v = {"vt_kind": vt_kind, "chrom": chrom, "pos": pos}

        haplotype_info.setdefault(hap_name, {"variants": []})["variants"].append(v)
    return haplotype_info


def norm(name: str) -> str:
    """サンプル名の正規化：前後空白除去 → 大文字化 → 空白をアンダースコアに"""
    return str(name).strip().upper().replace(" ", "_")

def load_map(map_path: str):
    """2列TSV: model_name  sub_name → dict[model_name] = sub_name"""
    if not map_path:
        return None
    mdf = pd.read_csv(map_path, sep="\t", header=None,
                      names=["model_name", "sub_name"], usecols=[0, 1])
    mdf = mdf.dropna().astype(str)
    return dict(zip(mdf["model_name"], mdf["sub_name"]))

def apply_name_map(seq, mapping, direction="model2sub"):
    if not mapping:
        return list(seq)
    if direction == "model2sub":
        return [mapping.get(s, s) for s in seq]
    else:
        inv = {v: k for k, v in mapping.items()}
        return [inv.get(s, s) for s in seq]


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description=("TOP=joblib の labels_top を最優先で再利用（無ければ linkage に保存された階層結果でフォールバック）。"
                     "SUB=K-Means で再クラスタリング。DEL/INS に重み付け可（DEL_mid は除外）。")
    )

    # 入出力
    ap.add_argument("-m", "--model", required=True,
                    help="TOP 情報を含む joblib（少なくとも 'samples' と 'labels_top'。無い場合は 'linkage' が必要）")
    ap.add_argument("-i", "--sub-input", required=True,
                    help="SUBクラスタ用特徴量TSV（1列目=sample, 以降=数値特徴量）")
    ap.add_argument("-o", "--out-prefix", default="suball",
                    help="出力プレフィックス（<prefix>_out/ に保存）")
    ap.add_argument("--map-file", default=None,
                    help="2列TSV: model_name  sub_name（TOP→SUB名対応を行う場合）")

    # TOP のフォールバック（labels_top が無い時のみ使われます）
    ap.add_argument("--distance", type=float, default=None,
                    help="labels_top が無いとき linkage で距離カット（TOPフォールバック用）")
    ap.add_argument("--maxclust", type=int, default=None,
                    help="labels_top が無いとき linkage でクラスタ数指定（TOPフォールバック用）")

    # SUB = K-Means の設定
    ap.add_argument("--sub-n-clusters", type=int, default=2,
                    help="SUB の K-Means クラスタ数 (default: 2)")
    ap.add_argument("--sub-random-state", type=int, default=42,
                    help="K-Means の random_state")
    ap.add_argument("--sub-n-init", type=int, default=10,
                    help="K-Means の n_init")
    ap.add_argument("--sub-max-iter", type=int, default=300,
                    help="K-Means の max_iter")

    # 重み付けオプション（共通 or 個別）
    ap.add_argument("--indel-weight", type=float, default=3.0,
                    help="INS/DEL/_notTP_ 系列に掛ける共通重み（個別指定が無ければこれが使われます）")
    ap.add_argument("--del-weight", type=float, default=None,
                    help="DEL 系列の重み（'DEL_mid' は除外）。未指定なら --indel-weight を使用")
    ap.add_argument("--ins-weight", type=float, default=None,
                    help="INS 系列の重み。未指定なら --indel-weight を使用")
    ap.add_argument("--nottp-weight", type=float, default=None,
                    help="_notTP_ 系列の重み。未指定なら --indel-weight を使用")
    ap.add_argument("--list-weighted-cols", action="store_true",
                    help="重み付け対象列を一覧表示する")

    # 遺伝子領域（joblibに保存して 1_predict_combined.py で使用）
    ap.add_argument("--region", default=None,
                    help="遺伝子領域（例: chr01:38382000-38390000）。joblibに保存され、"
                         "predict 時に --targets なしで全スキャンする際に参照される。")

    # ハプロタイプ照合用の wide table TSV（step3 の出力）
    ap.add_argument("--simplified-tsv", default=None,
                    help="ハプロタイプ照合用の wide table TSV（step3_wide/{prefix}_{svtype}.tsv）。"
                         "--haplotype-tsv と合わせて指定すると nested.txt にハプロタイプ名を反映する。")

    # ハプロタイプ情報（joblibに保存して 4_cluster.py で使用）
    ap.add_argument("--haplotype-tsv", default=None,
                    help="ハプロタイプ定義TSV（タブ区切り）。joblib に保存され predict 時のクラスタ名判定に使用される。\n"
                         "必須列: hap_name, vt_kind, chrom\n"
                         "  vt_kind=SNP      → pos, alt が必要\n"
                         "  vt_kind=INS      → pos が必要\n"
                         "  vt_kind=DEL_region → region_start, region_end が必要\n"
                         "例:\n"
                         "  hap_name\\tvt_kind\\tchrom\\tpos\\tregion_start\\tregion_end\\talt\n"
                         "  sd1-d\\tDEL_region\\tchr01\\t\\t38382500\\t38383200\\t\n"
                         "  ins_hap1\\tINS\\tchr01\\t38383735\\t\\t\\t")
    ap.add_argument("--gff", default=None,
                    help="GFF3 ファイル。cds.tsv に coord_type=cds/aa/gdna のエントリがある場合に必要。"
                         "指定時は coord_converter.py を動的に読み込んでゲノム座標に変換する。")

    # 図オプション（PCAプロット）
    ap.add_argument("--fig", action="store_true",
                    help="SUBクラスタのPCA散布図PNGも保存")
    ap.add_argument("--fig-width", type=float, default=10.0)
    ap.add_argument("--fig-height", type=float, default=6.0)
    ap.add_argument("--fig-dpi", type=int, default=150)

    args = ap.parse_args()

    # --- GFF 読み込み（--gff 指定時のみ）---
    conv_mod  = None
    gene_data = {}
    name_to_id: dict = {}
    if args.gff:
        script_dir_here = Path(__file__).parent.resolve()
        conv_mod = _load_coord_converter(script_dir_here)
        if conv_mod is None:
            print("[WARN] coord_converter.py not found. CDS coordinate conversion will be skipped.",
                  file=__import__("sys").stderr)
        else:
            gene_data, name_to_id = conv_mod.parse_gff(args.gff)
            print(f"[INFO] GFF loading complete: {len(gene_data)} genes")

    # --- haplotype_info 読み込み ---
    haplotype_info = {}
    if args.haplotype_tsv and not Path(args.haplotype_tsv).exists():
        print(f"[WARN] haplotype-tsv not found, skipping matching: {args.haplotype_tsv}",
              file=__import__("sys").stderr)
        args.haplotype_tsv = None
    if args.haplotype_tsv:
        hap_df = pd.read_csv(args.haplotype_tsv, sep="\t", dtype=str).fillna("")
        if "variant_type" in hap_df.columns:
            # cds 形式（gene_model_make/cds/ フォーマット）
            haplotype_info = _load_haplotype_tsv_cds(
                hap_df, conv_mod=conv_mod, gene_data=gene_data, name_to_id=name_to_id
            )
        else:
            # 既存形式（hap_name, vt_kind, chrom, pos, ...）
            for _, row in hap_df.iterrows():
                hap_name = str(row["hap_name"]).strip()
                vt_kind  = str(row["vt_kind"]).strip()
                chrom    = str(row["chrom"]).strip()
                v = {"vt_kind": vt_kind, "chrom": chrom}
                if vt_kind == "DEL_region":
                    v["region_start"] = int(row["region_start"])
                    v["region_end"]   = int(row["region_end"])
                else:
                    v["pos"] = int(row["pos"])
                    if row.get("alt", ""):
                        v["alt"] = str(row["alt"]).strip()
                haplotype_info.setdefault(hap_name, {"variants": []})["variants"].append(v)
        print(f"[INFO] haplotype_info loading complete: {len(haplotype_info)} haplotype(s)")

    # --- joblib 読み込み ---
    meta = joblib.load(args.model)
    samples_model_raw = meta.get("samples")
    if samples_model_raw is None:
        raise SystemExit("[ERROR] joblib has no 'samples'.")
    Z_top = meta.get("linkage")
    labels_top_from_model = meta.get("labels_top")

    # 名前対応・正規化
    mapping = load_map(args.map_file)
    samples_model_mapped = apply_name_map(samples_model_raw, mapping, "model2sub")
    samples_top_all = [norm(s) for s in samples_model_mapped]

    # --- SUB 特徴量読み込み ---
    sub_df = pd.read_csv(args.sub_input, sep="\t")
    id_col = sub_df.columns[0]
    sub_df[id_col] = sub_df[id_col].astype(str).map(norm)
    sub_df = sub_df.set_index(id_col)

    # TOPサンプルと SUB特徴量の共通部分
    inter = [s for s in samples_top_all if s in sub_df.index]
    if len(inter) < 2:
        raise SystemExit("[ERROR] Too few samples in common between SUB features and TOP samples.")
    X_sub = sub_df.loc[inter].copy()
    samples = list(inter)

    # ===== 特徴量の重み付け =====
    del_w = args.del_weight if args.del_weight is not None else args.indel_weight
    ins_w = args.ins_weight if args.ins_weight is not None else args.indel_weight
    nottp_w = args.nottp_weight if args.nottp_weight is not None else args.indel_weight
    weighted_cols = []

    def apply_weight_by_rule(df: pd.DataFrame):
        nonlocal weighted_cols
        for c in df.columns:
            cu = c.upper()
            if "_NOTTP_" in cu:
                df[c] = df[c] * float(nottp_w)
                weighted_cols.append((c, nottp_w, "NOTTP"))
                continue
            if "INS" in cu:
                df[c] = df[c] * float(ins_w)
                weighted_cols.append((c, ins_w, "INS"))
                continue
            if "DEL" in cu and "DEL_MID" not in cu:
                df[c] = df[c] * float(del_w)
                weighted_cols.append((c, del_w, "DEL"))
                continue
        return df

    X_sub = apply_weight_by_rule(X_sub)

    if weighted_cols:
        print(f"[INFO] Applied weighting: {len(weighted_cols)} column(s)")
        if args.list_weighted_cols:
            for c, w, tag in weighted_cols:
                print(f"   - {c}  (type={tag}, weight={w})")
    else:
        print("[INFO] No columns found for weighting.")

    X_sub_vals = X_sub.values.astype(float)

    # --- TOPラベル確定 ---
    if labels_top_from_model is not None:
        print("[INFO] Reusing labels_top from joblib")
        ser_top = pd.Series(labels_top_from_model, index=[norm(x) for x in samples_model_raw])
        labels_top = ser_top.loc[samples].to_numpy()
    else:
        print("[INFO] labels_top not found, fallback to linkage clustering (TOP only)")
        if Z_top is None:
            raise SystemExit("[ERROR] joblib has no linkage either. Cannot assign TOP clusters.")
        if args.distance is not None:
            labels_top_all = fcluster(Z_top, t=float(args.distance), criterion="distance")
        elif args.maxclust is not None:
            labels_top_all = fcluster(Z_top, t=int(args.maxclust), criterion="maxclust")
        else:
            raise SystemExit("[ERROR] When using linkage, specify either --distance or --maxclust.")
        ser_top_all = pd.Series(labels_top_all, index=samples_top_all)
        labels_top = ser_top_all.loc[samples].to_numpy()

    # ===== K-Means クラスタリング（特徴量が0列の場合は全サンプルをサブクラスタ1に割り当て） =====
    # DEL/INS の SUB クラスタリングは、SNP側の two-stage（stage0一次クラスタ）分けとは
    # 独立に、常に全サンプル一括で1回だけ行う（stage0グループ単位に分割しない）。
    sub_kmeans_model = None    # 学習済みKMeans（1クラスタ/1サンプルの場合はNone）
    sub_cluster_centers: list = []  # cluster_centers_（リスト）
    has_sub_features = X_sub_vals.shape[1] != 0

    if not has_sub_features:
        print("[WARN] Feature count is 0 (no valid positions). Assigning all samples to sub-cluster 1.")
        labels_sub = np.ones(len(samples), dtype=int)
    else:
        print(f"[INFO] KMeans: n_clusters={args.sub_n_clusters}, n_init={args.sub_n_init}, "
              f"max_iter={args.sub_max_iter}, random_state={args.sub_random_state}")

        k_req = int(args.sub_n_clusters)
        k = max(1, min(k_req, len(samples)))
        if k < k_req:
            print(f"[WARN] Sample count ({len(samples)}) is less than --sub-n-clusters "
                  f"({k_req}), adjusting to k={k}.")

        if k <= 1:
            # 1クラスタ、または1サンプルしかいない場合はKMeansを回さず全員サブ1
            labels_sub = np.ones(len(samples), dtype=int)
        else:
            km = KMeans(
                n_clusters=k,
                n_init=int(args.sub_n_init),
                max_iter=int(args.sub_max_iter),
                random_state=int(args.sub_random_state),
            )
            local_labels0 = km.fit_predict(X_sub_vals)
            labels_sub = local_labels0 + 1
            sub_kmeans_model = km
            sub_cluster_centers = km.cluster_centers_.tolist()

    sub_display = labels_sub

    # nested はこの多型（DEL/INS_TP/INS_notTP）専用のSUBクラスタ番号のみを表示用ラベルとする
    # （TOP＝SNP側の一次クラスタ番号を混ぜると、どちらの番号を見ればよいか分かりにくくなるため）
    nested_labels = np.array([str(s) for s in labels_sub], dtype=object)

    # --- 出力ディレクトリ ---
    # args.out_prefix がフルパスの場合でも正しく動作するよう basename のみファイル名に使う
    out_dir  = Path(f"{args.out_prefix}_out")
    out_name = Path(args.out_prefix).name   # ファイル名部分のみ（例: "Hd1_DEL"）
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- クラスタ結果をテキスト出力 ---
    df_all = pd.DataFrame({"sample": samples, "top": labels_top, "sub": sub_display, "nested": nested_labels})

    def _nested_sort_key(x):
        try:
            return (0, tuple(int(p) for p in str(x).split("-")), "")
        except ValueError:
            return (1, (), str(x))

    for key in ["top", "sub", "nested"]:
        p = out_dir / f"{key}.txt"

        # nested のみ haplotype_info + simplified_tsv があれば照合してラベルを置換
        if key == "nested" and haplotype_info and args.simplified_tsv:
            df_all["_label"] = df_all["nested"].astype(str)
            for idx, row in df_all.iterrows():
                haps = check_haplotype(row["sample"], args.simplified_tsv, haplotype_info)
                if haps:
                    # 一致した場合はSUB番号の代わりにハプロタイプ名をそのままラベルにする
                    # 例: hap="FA6345" → "FA6345"
                    df_all.at[idx, "_label"] = "/".join(sorted(haps))

            with p.open("w", encoding="utf-8") as f:
                for label in sorted(df_all["_label"].unique(), key=_nested_sort_key):
                    is_cluster_num = all(c.isdigit() or c == "-" for c in str(label))
                    header = f"# cluster {label}" if is_cluster_num else f"# {label}"
                    f.write(header + "\n")
                    for s in sorted(df_all.loc[df_all["_label"] == label, "sample"]):
                        f.write(f"{s}\n")
                    f.write("\n")
            df_all.drop(columns=["_label"], inplace=True)
            print(f"[OK] wrote {p}  (haplotype matching applied)")
        else:
            with p.open("w", encoding="utf-8") as f:
                for cl, subdf in df_all.groupby(key, sort=True):
                    f.write(f"# cluster {cl}\n")
                    for s in sorted(subdf["sample"]):
                        f.write(f"{s}\n")
                    f.write("\n")
            print(f"[OK] wrote {p}")

    # --- PCA 散布図（特徴量がある場合のみ） ---
    if args.fig:
        import matplotlib.patches as mpatches
        png = out_dir / "sub.kmeans.pca.png"

        if not has_sub_features:
            fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=args.fig_dpi)
            ax.text(0.5, 0.5, "No valid positions after filtering\n(all samples → sub-cluster 1)",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title("K-Means — No Data")
            plt.tight_layout()
            plt.savefig(png, dpi=args.fig_dpi, bbox_inches="tight")
            plt.close()
        else:
            pca = PCA(n_components=2, random_state=args.sub_random_state)
            X2  = pca.fit_transform(X_sub_vals)
            var = pca.explained_variance_ratio_

            # sub.txt と同じ表示ラベル（sub番号）で色分けする。
            def _sub_display_sort_key(x):
                try:
                    return tuple(int(p) for p in str(x).split("-"))
                except ValueError:
                    return (float("inf"),)

            unique_labels = sorted(pd.unique(sub_display), key=_sub_display_sort_key)
            cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
            label_colors = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

            fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=args.fig_dpi)

            for lab in unique_labels:
                idx   = (sub_display == lab)
                color = label_colors[lab]
                ax.scatter(X2[idx, 0], X2[idx, 1], c=[color], edgecolor="k", s=40, alpha=0.9)
                for i, show in enumerate(idx):
                    if show:
                        ax.text(X2[i, 0], X2[i, 1], samples[i], fontsize=8)

            ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
            ax.set_title(f"K-Means (K={args.sub_n_clusters}) on SUB features")
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            patches = [
                mpatches.Patch(color=label_colors[lab], label=f"sub{lab}")
                for lab in unique_labels
            ]
            fig.legend(handles=patches, title="Clusters", loc="upper right",
                       bbox_to_anchor=(1.1, 1))

            plt.tight_layout()
            plt.savefig(png, dpi=args.fig_dpi, bbox_inches="tight")
            plt.close()

        print(f"[OK] saved {png}")

    # --- モデル保存（.joblibのみ） ---
    bundle = {
        "samples": samples,
        "labels_top": labels_top,
        "labels_sub": labels_sub,
        "labels_nested": nested_labels,
        # 全サンプル一括で学習した単一のKMeans（1クラスタ/1サンプルの場合はNone）
        "sub_kmeans_model": sub_kmeans_model,
        "sub_scaler": None,
        "sub_feature_names": list(X_sub.columns),
        "method": "kmeans",
        "metric": None,
        # 追加部分 ↓↓↓
        "weights_info": {
            "indel_weight": args.indel_weight,
            "del_weight": del_w,
            "ins_weight": ins_w,
            "nottp_weight": nottp_w,
            "applied": True,
        },
        "sub_cluster_centers": sub_cluster_centers,  # クラスタ中心（リスト）
        "gene_region": args.region,
        "haplotype_info": haplotype_info,
    }
    joblib.dump(bundle, out_dir / f"{out_name}.joblib")
    print(f"[OK] wrote {out_dir / f'{out_name}.joblib'}")

    # --- クラスタ中心のTSV出力（KMeansを実際に学習した場合のみ） ---
    if sub_kmeans_model is not None:
        centers_path = out_dir / f"{out_name}.kmeans.centers.tsv"
        centers_index = [f"sub{c}" for c in range(1, sub_kmeans_model.n_clusters + 1)]
        centers_df = pd.DataFrame(sub_kmeans_model.cluster_centers_, columns=list(X_sub.columns))
        centers_df.index = centers_index
        centers_df.to_csv(centers_path, sep="\t")
        print(f"[OK] wrote {centers_path}")


if __name__ == "__main__":
    main()
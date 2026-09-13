#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_haplotype_match.py
クラスタリング結果と既報ハプロタイプの照合・可視化スクリプト

cds.tsv に記載されたハプロタイプ定義と step4_{prefix}.tsv を照合し、
一致した品種を step9 の PCA プロットに矢印で表示する。

【cds.tsv のカラム仕様】
  gene_id     : 遺伝子 ID またはシンボル（例: Ghd7）
  coord_type  : genome / cds / aa / gdna のいずれか
                genome 以外は --gff で GFF3 ファイルを指定する必要がある
  pos         : 座標値
                coord_type=genome の場合は "chr07:9155047" 形式
                それ以外は整数
  variant_type: SNP:C>A / SNP:A / INS / Insertion / DEL / Deletion
                大文字小文字を区別しない
  name        : ハプロタイプ名（同じ name の行は AND 条件でマッチング）

使用例:
  python3 10_haplotype_match.py \\
      --outdir results/Ghd7/ \\
      --prefix Ghd7 \\
      --haplotype-tsv /path/to/cds.tsv \\
      --gff /path/to/transcripts.gff
"""

import argparse
import importlib.util
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
#  coord_converter の動的読み込み
# ============================================================

def _load_coord_converter(script_dir: Path):
    """coord_converter.py を動的に読み込む。失敗時は None を返す。
    検索順: 同フォルダ → ../cds/ → ./cds/
    同フォルダを最優先にすることで、明示的に置いたファイルが確実に使われる。
    """
    for candidate in [
        script_dir / "coord_converter.py",             # 同フォルダ（最優先）
        script_dir.parent / "cds" / "coord_converter.py",  # 隣の cds/
        script_dir / "cds" / "coord_converter.py",     # 子の cds/
    ]:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("coord_converter", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


# ============================================================
#  染色体名正規化
# ============================================================

def norm_chrom(c: str) -> str:
    """'chr07' / 'Chr7' / '7' → '7' に正規化"""
    c = str(c).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    try:
        return str(int(c))
    except ValueError:
        return c


# ============================================================
#  VCF パーサ（SNP塩基の自動解決用）
# ============================================================

def parse_vcf_alt(vcf_path: str) -> dict[tuple[str, int], str]:
    """
    VCF ファイルを読み込んで {(chrom_norm, pos): alt_allele} を返す。
    variant_type=SNP（塩基指定なし）のハプロタイプ定義をVCFから自動解決するために使用する。
    マルチアレリックの場合は最初の非spanning-deletion alleleを採用する。
    """
    alt_dict: dict[tuple[str, int], str] = {}
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            chrom = norm_chrom(parts[0])
            try:
                pos = int(parts[1])
            except ValueError:
                continue
            alts = [a for a in parts[4].split(",") if a != "*"]
            if alts:
                alt_dict[(chrom, pos)] = alts[0].upper()
    return alt_dict


# ============================================================
#  cds.tsv パーサ
# ============================================================

def parse_haplotype_tsv(tsv_path: str) -> list[dict]:
    """
    cds.tsv を読み込んでハプロタイプ定義リストを返す。

    戻り値: list of dict
      {gene_id, coord_type, pos_raw, variant_type, name}
    """
    records = []
    with open(tsv_path) as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            if header is None:
                header = [c.lower().strip() for c in cols]
                continue
            row = dict(zip(header, cols))
            records.append({
                "gene_id":      row.get("gene_id", "").strip(),
                "coord_type":   row.get("coord_type", "").strip().lower(),
                "pos_raw":      row.get("pos", "").strip(),
                "variant_type": row.get("variant_type", "").strip(),
                "name":         row.get("name", "").strip(),
            })
    return records


# ============================================================
#  座標変換
# ============================================================

def resolve_genome_coord(record: dict, conv_mod, gene_data: dict, name_to_id: dict) -> tuple[str | None, int | None, str | None, str]:
    """
    1 レコードのゲノム座標を解決する。

    戻り値: (chrom_normalized, genome_pos, strand, error_message)
      strand は coord_type=genome の場合 None、coord_type=aa/cds/gdna の場合は
      GFF から取得した "+"/"-"。呼び出し側の parse_variant_type() では現在この
      strand を使った塩基の相補変換は行っていない（variant_type の塩基は
      coord_type によらず常にゲノムplus鎖表記として扱う）。
    """
    coord_type = record["coord_type"]
    pos_raw    = record["pos_raw"]
    gene_id    = record["gene_id"]

    if coord_type == "genome":
        # "chr07:9155047" 形式（variant_type は genome plus鎖表記である前提）
        if ":" not in pos_raw:
            return None, None, None, f"genome 形式は 'chrom:pos' で指定してください: {pos_raw!r}"
        chrom_str, pos_str = pos_raw.rsplit(":", 1)
        try:
            return norm_chrom(chrom_str), int(pos_str), None, ""
        except ValueError:
            return None, None, None, f"pos が整数ではありません: {pos_str!r}"

    # aa / cds / gdna → coord_converter を使う
    if conv_mod is None:
        return None, None, None, "coord_converter.py が見つかりません。--gff とともに配置してください"

    try:
        pos_int = int(pos_raw)
    except ValueError:
        return None, None, None, f"pos が整数ではありません: {pos_raw!r}"

    resolved_id = conv_mod.resolve_gene_id(gene_id, gene_data, name_to_id)
    result = conv_mod.convert(resolved_id, coord_type, pos_int, gene_data)

    if result["status"] != "OK":
        return None, None, None, result["message"]

    chrom = norm_chrom(result["chrom"])
    return chrom, result["genome_pos"], result["strand"], ""


# ============================================================
#  variant_type パーサ・マッチング
# ============================================================

def parse_variant_type(vt: str, strand: str | None = None) -> tuple[str, str | None]:
    """
    variant_type を (種別, alt_allele_or_None) に分解する。

    種別: "SNP" / "INS" / "DEL" / "UNKNOWN"

    strand:
      未使用（後方互換のため引数は残す）。coord_type=aa/cds/gdna であっても、
      variant_type の塩基は常にゲノムplus鎖表記（VCFのREF/ALTの基準）として扱う。
      以前は「coord_type=aa/cds/gdnaはコード鎖(mRNA鎖)表記」という前提で、
      マイナス鎖遺伝子の場合に自動で相補変換していたが、cds.tsv の実際の記載が
      ゲノムplus鎖表記のケース（例: Ghd7-0_H0_LOF）と混在しており、遺伝子ごとに
      どちらの表記かを機械的に判別できないため、常にゲノムplus鎖表記として扱う
      方針に変更した。mRNA鎖表記で書かれた既存エントリは、cds.tsv 側でゲノム
      plus鎖表記に書き換える必要がある。
    """
    vt_strip = vt.strip()
    vt_upper = vt_strip.upper()

    if vt_upper.startswith("SNP"):
        body = vt_strip[3:].lstrip(":").strip()
        if ">" in body:
            alt_raw = body.split(">", 1)[1].strip()
        else:
            alt_raw = body
        # "T(derived_from_aa_change)" のような注記付き表記から塩基文字だけを取り出す
        # （文字自体は塩基でなければ何でもよいが、想定は "T" や複数塩基の "TA" 等）。
        m = re.match(r"[A-Za-z]+", alt_raw)
        alt = m.group(0).upper() if m else None
        return "SNP", alt

    if vt_upper in ("INS", "INSERTION"):
        return "INS", None

    if vt_upper in ("DEL", "DELETION"):
        return "DEL", None

    return "UNKNOWN", None


def sample_matches_variant(sample_value: str, vt_kind: str, vt_alt: str | None) -> bool:
    """サンプルの塩基値が variant_type にマッチするか判定"""
    sv = str(sample_value).strip().upper()
    if sv in ("N", "NA", "NONE", ""):
        return False

    if vt_kind == "SNP":
        return sv == (vt_alt or "").upper()
    if vt_kind == "INS":
        return sv == "INS"
    if vt_kind == "DEL":
        return sv == "DEL"
    return False


# ============================================================
#  ハプロタイプマッチング
# ============================================================

def match_haplotypes(
    records: list[dict],
    step4_df: pd.DataFrame,
    conv_mod,
    gene_data: dict,
    name_to_id: dict,
    vcf_alt_dict: dict | None = None,
) -> dict[str, set[str]]:
    """
    ハプロタイプ定義と step4 TSV を照合する。

    同じ name を持つ複数行は AND 条件（全変異に一致するサンプルのみ）。

    戻り値: {name: {matching_sample_names}}
    """
    sample_cols = [c for c in step4_df.columns if c not in ("chr", "posi")]

    # step4 を (chr_norm, posi) インデックスで引けるように準備
    step4_indexed = step4_df.copy()
    step4_indexed["_chr_norm"] = step4_indexed["chr"].astype(str).map(norm_chrom)
    step4_indexed = step4_indexed.set_index(["_chr_norm", "posi"])

    # ハプロタイプ名ごとにレコードをグループ化
    by_name = defaultdict(list)
    for rec in records:
        by_name[rec["name"]].append(rec)

    results: dict[str, set[str]] = {}

    for hap_name, recs in by_name.items():
        matching_sets: list[set[str]] = []

        for rec in recs:
            chrom, gpos, strand, err = resolve_genome_coord(rec, conv_mod, gene_data, name_to_id)
            if err or chrom is None or gpos is None:
                print(f"  [WARN] haplotype '{hap_name}': coordinate conversion failed -> {err}", file=sys.stderr)
                matching_sets.append(set())
                continue

            # step4 でその位置の行を取得
            try:
                row = step4_indexed.loc[(chrom, gpos)]
            except KeyError:
                print(f"  [WARN] haplotype '{hap_name}': "
                      f"{chrom}:{gpos} not found in step4", file=sys.stderr)
                matching_sets.append(set())
                continue

            # 複数行がヒットした場合（通常は1行）
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            vt_kind, vt_alt = parse_variant_type(rec["variant_type"], strand=strand)
            if vt_kind == "UNKNOWN":
                print(f"  [WARN] haplotype '{hap_name}': "
                      f"unsupported variant_type: {rec['variant_type']!r}", file=sys.stderr)
                matching_sets.append(set())
                continue

            # SNP塩基が未指定の場合、VCFからALTを自動解決する
            if vt_kind == "SNP" and vt_alt is None:
                if vcf_alt_dict and (chrom, gpos) in vcf_alt_dict:
                    vt_alt = vcf_alt_dict[(chrom, gpos)]
                    print(f"  [VCF]  '{hap_name}' {chrom}:{gpos}: got ALT={vt_alt} from VCF")
                else:
                    print(f"  [WARN] '{hap_name}' {chrom}:{gpos}: "
                          f"SNP base not specified and not found in VCF -> skipping", file=sys.stderr)
                    matching_sets.append(set())
                    continue

            matched = {
                s for s in sample_cols
                if sample_matches_variant(str(row[s]), vt_kind, vt_alt)
            }
            matching_sets.append(matched)

        # AND 条件で絞り込み
        if matching_sets:
            final = matching_sets[0]
            for s in matching_sets[1:]:
                final = final & s
        else:
            final = set()

        results[hap_name] = final

    return results


# ============================================================
#  cluster.txt パーサ（step9_plot.py と同仕様）
# ============================================================

def load_cluster_file(cluster_path: str) -> tuple[pd.DataFrame, dict[int, str]]:
    cluster_dict: dict[str, int] = {}
    cluster_labels: dict[int, str] = {}
    current_cluster_id = -1
    with open(cluster_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                current_cluster_id += 1
                cluster_labels[current_cluster_id] = line[1:].strip()
            elif line:
                cluster_dict[line] = current_cluster_id
    df = pd.DataFrame(list(cluster_dict.items()), columns=["sample", "cluster"])
    return df, cluster_labels


# ============================================================
#  PCA + プロット（矢印付き）
# ============================================================

CLUSTER_COLORS = {
    0: "red", 1: "blue", 2: "gold", 3: "green", 4: "gray",
    5: "purple", 6: "deepskyblue", 7: "hotpink", 8: "saddlebrown", 9: "darkorange",
}


def get_cluster_color(cluster_id):
    """0..9 は固定色。それ以上は matplotlib の tab20 から循環で割り当てる
    （sc/step3/1/7_newplot.py の同名関数と同じ方針）。二段階クラスタリングの
    結合後（stage2）は flat cluster 番号が10を超えることがあり、CLUSTER_COLORS
    に無い番号を一律 "gray" にフォールバックすると（cluster 4 と被る上に）
    複数の別クラスタが見分けられなくなるため、tab20で別々の色にする。"""
    if cluster_id in CLUSTER_COLORS:
        return CLUSTER_COLORS[cluster_id]
    cmap = plt.get_cmap("tab20")
    return cmap(cluster_id % 20)

# ハプロタイプ矢印に使う色（クラスタ色と被らないよう選定）
HAPLOTYPE_COLORS = [
    "#E64C4C", "#4C9BE6", "#4CE64C", "#E6A04C",
    "#B04CE6", "#4CE6E6", "#E64CB0", "#808080",
]


def load_case_map(case_map_path):
    """2-column TSV (UPPER(sample) <TAB> sample-as-originally-named). Optional
    -- lets per-point labels below show each sample's original BAM-dir/list
    casing instead of the upper-cased name sc/'s own processing uses
    internally for matching."""
    if not case_map_path:
        return {}
    case_map = {}
    with open(case_map_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                case_map[parts[0]] = parts[1]
    return case_map


def plot_pca_with_haplotypes(
    feature_path: str,
    cluster_path: str,
    hap_matches: dict[str, set[str]],
    output_path: str,
    n_components: int = 2,
    pc_pairs: list[tuple[int, int]] = None,
    pca_model_path: str | None = None,
    case_map: dict | None = None,
) -> None:
    """
    step9_plot.py と完全に同じ描画設定で PCA プロットを生成し、
    ハプロタイプ一致品種に矢印アノテーションを追加する。
    figsize / dpi / fontsize / 凡例配置はすべて step9 に合わせる。
    入力は step5 の特徴量 TSV（セントロイド距離ではなく元特徴量を使用）。
    """
    import math

    case_map = case_map or {}
    if pc_pairs is None:
        pc_pairs = [(0, 1)]

    # --- データ読み込み（step9 と同じ処理） ---
    feat_df    = pd.read_csv(feature_path, sep="\t")
    feat_df    = feat_df.rename(columns={feat_df.columns[0]: "sample"})
    cluster_df, cluster_labels = load_cluster_file(cluster_path)

    merged = feat_df.merge(cluster_df, on="sample", how="inner").dropna(subset=["cluster"])
    if merged.empty:
        print("[ERROR] The merged result of distance and cluster is empty", file=sys.stderr)
        return

    X        = merged.iloc[:, 1:-1].values.astype(float)
    samples  = merged["sample"].tolist()
    clusters = merged["cluster"].astype(int).tolist()

    # --- PCA（step9 で保存したモデルを再利用、なければ独自に fit） ---
    if pca_model_path and Path(pca_model_path).exists():
        pca     = joblib.load(pca_model_path)
        reduced = pca.transform(X)
        print(f"[INFO] Loaded PCA model: {pca_model_path}")
    else:
        pca     = PCA(n_components=n_components)
        reduced = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_

    sample_to_idx = {s: i for i, s in enumerate(samples)}

    # ハプロタイプ → 色
    hap_names  = [n for n, s in hap_matches.items() if s]
    hap_colors = {n: HAPLOTYPE_COLORS[i % len(HAPLOTYPE_COLORS)] for i, n in enumerate(hap_names)}

    # --- figsize: step9 と完全に同じ ---
    n_plots = len(pc_pairs)
    cols    = min(n_plots, 3)
    rows    = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_arr = np.array(axes).reshape(-1)

    # 凡例パッチ（step9 と同じクラスタ部分 + ハプロタイプ追加）
    valid_clusters  = sorted(set(clusters))
    cluster_patches = [
        mpatches.Patch(color=get_cluster_color(c),
                       label=cluster_labels.get(c, f"Cluster {c+1}"))
        for c in valid_clusters
    ]
    hap_patches = [
        mpatches.Patch(color=hap_colors[n], label=f"Allele: {n}")
        for n in hap_names
    ]

    for plot_idx, (pc_x, pc_y) in enumerate(pc_pairs):
        ax = axes_arr[plot_idx]

        if max(pc_x, pc_y) >= reduced.shape[1]:
            print(f"Skipping PC{pc_x+1} vs PC{pc_y+1}: exceeds available components")
            continue

        # =============================================
        # step9_plot.py と一字一句同じ描画ブロック
        # =============================================
        colors = [get_cluster_color(c) for c in clusters]
        ax.scatter(reduced[:, pc_x], reduced[:, pc_y], c=colors, edgecolor='k')
        for i, label in enumerate(samples):
            ax.text(reduced[i, pc_x], reduced[i, pc_y], case_map.get(label, label), fontsize=8)

        ax.set_xlabel(f"PC{pc_x + 1} ({var_ratio[pc_x]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y + 1} ({var_ratio[pc_y]*100:.1f}%)")
        ax.set_title(f"PC{pc_x + 1} vs PC{pc_y + 1}")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # =============================================

        # --- ハプロタイプ矢印（step10 で追加する部分のみ） ---
        all_xy  = reduced[:, [pc_x, pc_y]]
        x_range = all_xy[:, 0].ptp()
        y_range = all_xy[:, 1].ptp()

        # ラベルを探索するオフセット距離
        offset = max(x_range, y_range) * 0.30

        # プロット内の許容範囲（データ範囲 + 余白）
        pad   = max(x_range, y_range) * 0.08
        x_lo  = all_xy[:, 0].min() - pad
        x_hi  = all_xy[:, 0].max() + pad
        y_lo  = all_xy[:, 1].min() - pad
        y_hi  = all_xy[:, 1].max() + pad

        for hap_name, matched_samples in hap_matches.items():
            if not matched_samples:
                continue
            color = hap_colors.get(hap_name, "black")

            for sample_name in sorted(matched_samples):
                if sample_name not in sample_to_idx:
                    continue
                idx    = sample_to_idx[sample_name]
                sx, sy = reduced[idx, pc_x], reduced[idx, pc_y]

                # ラベル位置の自動探索:
                # 360°を30°刻みで試し、プロット内かつ他品種から最も離れた位置を選ぶ
                best_tx, best_ty = sx, sy
                best_score = -1.0
                for deg in range(0, 360, 30):
                    rad = np.radians(deg)
                    tx = sx + np.cos(rad) * offset
                    ty = sy + np.sin(rad) * offset
                    if not (x_lo <= tx <= x_hi and y_lo <= ty <= y_hi):
                        continue
                    # 最も近い他サンプルまでの距離
                    min_dist = min(
                        np.sqrt((reduced[j, pc_x] - tx) ** 2 + (reduced[j, pc_y] - ty) ** 2)
                        for j in range(len(samples)) if j != idx
                    )
                    if min_dist > best_score:
                        best_score = min_dist
                        best_tx, best_ty = tx, ty

                ax.annotate(
                    hap_name,
                    xy=(sx, sy),
                    xytext=(best_tx, best_ty),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.8,
                        connectionstyle="arc3,rad=0.1",
                    ),
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.85,
                        linewidth=1.2,
                    ),
                    zorder=10,
                )

    # 余ったパネルを非表示
    for ax in axes_arr[n_plots:]:
        ax.set_axis_off()

    # --- 凡例: step9 と同じ位置・形式、ハプロタイプ行を追加。軸領域と被らないよう、
    # 先にレイアウトを整えてから右側に専用の余白を確保して配置する（アリル名が多い
    # ほど凡例が縦長/横長になり、tight_layout() のみでは軸領域と重なりやすいため）---
    plt.tight_layout()
    fig.subplots_adjust(right=0.70)
    fig.legend(
        handles=cluster_patches + hap_patches,
        title="Clusters",
        loc='upper left',
        bbox_to_anchor=(1.03, 1),
    )

    # --- 保存: step9 と同じ dpi / bbox_inches ---
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Plot saved: {output_path}")


# ============================================================
#  ハプロタイプ変異情報の収集（joblib 保存用）
# ============================================================

def build_haplotype_variants(
    records: list[dict],
    conv_mod,
    gene_data: dict,
    name_to_id: dict,
    vcf_alt_dict: dict | None = None,
) -> dict[str, list[dict]]:
    """
    cds.tsv レコードからハプロタイプ変異の解決済み座標情報を構築する。
    joblib への保存に使用する。

    variant_type=SNP（塩基指定なし）の場合、vcf_alt_dict からALT塩基を解決する。
    戻り値: {hap_name: [{"chrom", "pos", "variant_type", "vt_kind", "alt"}, ...]}
    """
    result: dict[str, list[dict]] = {}
    for rec in records:
        hap_name = rec["name"]
        chrom, gpos, strand, err = resolve_genome_coord(rec, conv_mod, gene_data, name_to_id)
        if err or chrom is None or gpos is None:
            continue
        vt_kind, vt_alt = parse_variant_type(rec["variant_type"], strand=strand)
        if vt_kind == "UNKNOWN":
            continue
        # SNP塩基が未指定の場合、VCFから解決する
        if vt_kind == "SNP" and vt_alt is None:
            if vcf_alt_dict and (chrom, gpos) in vcf_alt_dict:
                vt_alt = vcf_alt_dict[(chrom, gpos)]
            else:
                continue  # 解決できなければjoblibに保存しない
        result.setdefault(hap_name, []).append({
            "chrom":        chrom,
            "pos":          gpos,
            "variant_type": rec["variant_type"],
            "vt_kind":      vt_kind,
            "alt":          vt_alt,
        })
    return result


def save_haplotype_to_joblib(
    model_path: str,
    hap_matches: dict[str, set[str]],
    hap_variants: dict[str, list[dict]],
    cluster_path: str,
) -> None:
    """
    step7 の joblib にハプロタイプ情報を追記して上書き保存する。

    --haplotype-tsv に記載された全ハプロタイプ（座標解決できたもの）を保存する。
    今回の学習データに一致品種が0件のハプロタイプも matched_samples=[] / clusters=[] として保存する
    （step3 で新品種が現れたときに照合できるようにするため）。

    追加されるキー:
      haplotype_info : {name: {variants, matched_samples, clusters}}
      mixed_clusters : [cluster_id, ...]  ハプロタイプと非ハプロタイプが混在するクラスタ（1始まり）
    """
    import joblib as jl

    bundle = jl.load(model_path)
    if not isinstance(bundle, dict):
        print(f"[WARN] Skipping update because the joblib is not in bundle format: {model_path}",
              file=sys.stderr)
        return

    cluster_df, _ = load_cluster_file(cluster_path)
    sample_to_cluster = dict(zip(
        cluster_df["sample"], cluster_df["cluster"].astype(int) + 1
    ))

    haplotype_info: dict = {}
    mixed_clusters: set[int] = set()

    # --haplotype-tsv に記載された全ハプロタイプを保存する（一致品種が0件でも保存する）。
    # 座標解決できた（hap_variants に存在する）ハプロタイプのみが対象。
    for hap_name, variants in hap_variants.items():
        if not variants:
            continue
        matched = hap_matches.get(hap_name, set())

        hap_clusters = {sample_to_cluster[s] for s in matched if s in sample_to_cluster}

        # 同クラスタに非ハプロタイプ品種がいれば「混在クラスタ」
        for cl in hap_clusters:
            all_in_cluster = {s for s, c in sample_to_cluster.items() if c == cl}
            if all_in_cluster - matched:
                mixed_clusters.add(cl)

        haplotype_info[hap_name] = {
            "variants":        variants,
            "matched_samples": sorted(matched),
            "clusters":        sorted(hap_clusters),
        }

    bundle["haplotype_info"] = haplotype_info
    bundle["mixed_clusters"] = sorted(mixed_clusters)
    jl.dump(bundle, model_path)

    print(f"[OK] joblib updated: added haplotype info -> {model_path}"
          f" (registered haplotypes: {len(haplotype_info)})")
    if mixed_clusters:
        print(f"     Mixed clusters: {sorted(mixed_clusters)}")
    for name, info in haplotype_info.items():
        if info["matched_samples"]:
            print(f"     {name}: {info['matched_samples']} (cluster {info['clusters']})")
        else:
            print(f"     {name}: no matching samples (saved only)")


# ============================================================
#  ハプロタイプ一致結果のテキスト出力
# ============================================================

def write_haplotype_txt(
    hap_matches: dict[str, set[str]],
    cluster_path: str,
    output_path: str,
) -> None:
    """一致品種 + クラスタ番号 + ハプロタイプ名を TSV で出力"""
    cluster_df, cluster_labels = load_cluster_file(cluster_path)
    sample_to_cluster = dict(zip(cluster_df["sample"], cluster_df["cluster"].astype(int) + 1))

    lines = ["allele\tsample\tcluster"]
    for hap_name in sorted(hap_matches):
        matched = hap_matches[hap_name]
        if not matched:
            lines.append(f"{hap_name}\t(一致なし)\t-")
        else:
            for sample in sorted(matched):
                cl = sample_to_cluster.get(sample, "?")
                lines.append(f"{hap_name}\t{sample}\t{cl}")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Haplotype match results: {output_path}")


# ============================================================
#  main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="クラスタリング結果と既報ハプロタイプを照合し、矢印付き PCA プロットを出力する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--outdir",         required=True,
                        help="step1 パイプライン出力ディレクトリ（step4/step7/step8 が入っているフォルダ）")
    parser.add_argument("--prefix",         required=True,
                        help="遺伝子名プレフィックス（例: Ghd7）")
    parser.add_argument("--haplotype-tsv",  required=True,
                        help="ハプロタイプ定義 TSV（cds.tsv 形式）")
    parser.add_argument("--gff",            default=None,
                        help="GFF3 ファイル（coord_type が genome 以外の行がある場合に必須）")
    parser.add_argument("--gff-feature",    default="CDS",
                        help="GFF3 から読み込むフィーチャータイプ（デフォルト: CDS）")
    parser.add_argument("--pca-components", type=int, default=2,
                        help="PCA 主成分数（デフォルト: 2）")
    parser.add_argument("--plot-pc",        nargs="+", type=int, default=[1, 2],
                        help="プロットする PC ペア（デフォルト: 1 2）")
    parser.add_argument("--output-plot",    default=None,
                        help="出力プロットパス（省略時: {outdir}/step10_{prefix}_haplotype_pca.png）")
    parser.add_argument("--output-txt",     default=None,
                        help="出力 TXT パス（省略時: {outdir}/step10_{prefix}_haplotype_match.txt）")
    parser.add_argument("--vcf",            default=None,
                        help="フィルタ済み VCF ファイル（step3_selected.vcf 等）。"
                             "variant_type=SNP（塩基指定なし）のハプロタイプ定義がある場合、"
                             "VCF の ALT 塩基を用いて自動的に塩基を解決する。")
    parser.add_argument("--pca-model",     default=None,
                        help="step9 で保存した PCA モデル（.joblib）。"
                             "指定すると step9 と完全に同じ PCA 空間でプロットする。")
    parser.add_argument("--case-map",      default=None,
                        help="2列TSV（大文字化サンプル名<TAB>元の表記）。指定すると点ラベルを元の表記で描画する。")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    prefix = args.prefix

    # ファイルパス
    step4_path   = outdir / f"step4_{prefix}.tsv"
    step5_path   = outdir / f"step5_{prefix}.tsv"
    step7_path   = outdir / f"step7_{prefix}_cluster.txt"
    out_plot     = Path(args.output_plot)  if args.output_plot  else outdir / f"step10_{prefix}_haplotype_pca.png"
    out_txt      = Path(args.output_txt)   if args.output_txt   else outdir / f"step10_{prefix}_haplotype_match.txt"

    for p in [step4_path, step7_path]:
        if not p.exists():
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)

    if not step5_path.exists():
        print(f"[INFO] {step5_path} not found. Skipping the PCA plot.")

    # --- coord_converter と GFF の読み込み ---
    script_dir = Path(__file__).parent.resolve()
    conv_mod   = _load_coord_converter(script_dir)
    gene_data: dict = {}
    name_to_id: dict = {}

    records = parse_haplotype_tsv(args.haplotype_tsv)
    needs_conversion = any(r["coord_type"] != "genome" for r in records)

    if needs_conversion:
        if args.gff is None:
            print("[ERROR] Found coord_type values other than genome. Please specify --gff", file=sys.stderr)
            sys.exit(1)
        if conv_mod is None:
            print("[ERROR] coord_converter.py not found. "
                  "Please check that it is placed in the cds/ directory", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Loading GFF: {args.gff}")
        gene_data, name_to_id = conv_mod.parse_gff(args.gff, feature_type=args.gff_feature)
        print(f"[INFO] Loaded {len(gene_data)} gene(s)")

    # --- step4 TSV 読み込み ---
    step4_df = pd.read_csv(step4_path, sep="\t", dtype={"chr": str})

    # --- VCF からALT塩基辞書を構築（SNP塩基未指定の自動解決用） ---
    vcf_alt_dict: dict | None = None
    if args.vcf:
        if Path(args.vcf).exists():
            vcf_alt_dict = parse_vcf_alt(args.vcf)
            print(f"[INFO] Loaded VCF: {len(vcf_alt_dict)} position(s) ({args.vcf})")
        else:
            print(f"[WARN] VCF not found: {args.vcf}", file=sys.stderr)

    # --- ハプロタイプマッチング ---
    print(f"[INFO] Haplotype definitions: {len(records)} row(s) / {len(set(r['name'] for r in records))} haplotype(s)")
    hap_matches = match_haplotypes(records, step4_df, conv_mod, gene_data, name_to_id,
                                   vcf_alt_dict=vcf_alt_dict)

    # 結果サマリ
    for hap_name, matched in sorted(hap_matches.items()):
        if matched:
            print(f"  {hap_name}: {len(matched)} sample(s) matched -> {sorted(matched)}")
        else:
            print(f"  {hap_name}: no match")

    # --- PC ペアに変換（1-indexed → 0-indexed） ---
    pcs_raw = args.plot_pc
    if len(pcs_raw) % 2 != 0:
        print("[ERROR] --plot-pc requires an even number of values (pairs)", file=sys.stderr)
        sys.exit(1)
    pc_pairs = [(pcs_raw[i] - 1, pcs_raw[i + 1] - 1) for i in range(0, len(pcs_raw), 2)]

    # --- プロット生成（step5 特徴量 TSV がある場合のみ） ---
    if step5_path.exists():
        plot_pca_with_haplotypes(
            str(step5_path), str(step7_path),
            hap_matches, str(out_plot),
            n_components=args.pca_components,
            pc_pairs=pc_pairs,
            pca_model_path=args.pca_model,
            case_map=load_case_map(args.case_map),
        )
    else:
        print(f"[INFO] Skipped the PCA plot ({step5_path.name} not present)")

    # --- 結果 TXT 出力 ---
    write_haplotype_txt(hap_matches, str(step7_path), str(out_txt))

    # --- ハプロタイプ情報を step7 joblib に追記保存 ---
    step7_model_path = outdir / f"step7_{prefix}_model.joblib"
    if step7_model_path.exists():
        hap_variants = build_haplotype_variants(records, conv_mod, gene_data, name_to_id,
                                                vcf_alt_dict=vcf_alt_dict)
        save_haplotype_to_joblib(
            str(step7_model_path), hap_matches, hap_variants, str(step7_path)
        )
    else:
        print(f"[WARN] joblib not found (skipping): {step7_model_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

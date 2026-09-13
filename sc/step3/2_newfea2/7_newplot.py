#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import re
import math


def load_case_map(case_map_path):
    """2-column TSV (UPPER(sample) <TAB> sample-as-originally-named). Optional
    -- lets per-point labels below show each sample's original BAM-dir/list
    casing instead of the upper-cased name sc/'s DEL/INS-side processing uses
    internally for matching (dedup_sample_names_upper() in
    1_predict_combined.py). Identical to sc/step1/step1/9_plot.py's helper of
    the same name."""
    if not case_map_path:
        return {}
    case_map = {}
    with open(case_map_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                case_map[parts[0]] = parts[1]
    return case_map


def get_text_bbox(ax, text_obj, renderer):
    """テキストオブジェクトのバウンディングボックスをデータ座標で取得"""
    bbox = text_obj.get_window_extent(renderer=renderer)
    return bbox.transformed(ax.transData.inverted())


def check_overlap(bbox1, bbox2):
    """2つのバウンディングボックスが重なっているかチェック"""
    return (bbox1.x0 < bbox2.x1 and bbox1.x1 > bbox2.x0 and
            bbox1.y0 < bbox2.y1 and bbox1.y1 > bbox2.y0)


def bbox_hits_axis_zone(bbox, xlim, ylim, pad_x, pad_y):
    """
    軸（プロット端）付近の保護領域に入っているか判定
    - pad_x/pad_y はデータ座標の余白
    """
    left = xlim[0] + pad_x
    right = xlim[1] - pad_x
    bottom = ylim[0] + pad_y
    top = ylim[1] - pad_y

    # bbox が安全域に収まっていればOK（= False）
    if (bbox.x0 >= left and bbox.x1 <= right and
        bbox.y0 >= bottom and bbox.y1 <= top):
        return False
    return True


def generate_candidates(x, y, base_dx, base_dy, max_radius_mult=12, n_angles=16):
    """
    候補座標を生成：
      - 円周上 n_angles 方向
      - 半径を 1..max_radius_mult まで増やす
      - 近傍（右上）も最初に含める
    """
    yield (x + base_dx, y + base_dy, 1)

    angles = [2 * math.pi * i / n_angles for i in range(n_angles)]
    for r in range(1, max_radius_mult + 1):
        for ang in angles:
            dx = math.cos(ang) * base_dx * r
            dy = math.sin(ang) * base_dy * r
            yield (x + dx, y + dy, r)


def find_best_position(
    ax, x, y, label, placed_bboxes, renderer,
    fontsize=8, fontweight='bold',
    axis_pad_frac=0.05,
    max_radius_mult=30,
    n_angles=32,
    hard_avoid_axis=True
):
    """
    placed_bboxes（既に置いたラベル bbox）と被らない位置を探索。
    ※この版では「新（new）ラベル」用に使う想定。
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    base_dx = x_range * 0.025
    base_dy = y_range * 0.025
    pad_x = x_range * axis_pad_frac
    pad_y = y_range * axis_pad_frac

    best_pos = (x + base_dx, y + base_dy)
    best_score = float('inf')
    best_overlap = float('inf')

    for tx, ty, r in generate_candidates(x, y, base_dx, base_dy,
                                         max_radius_mult=max_radius_mult, n_angles=n_angles):
        # テスト用テキスト（不可視）で bbox を評価
        test_text = ax.text(
            tx, ty, label,
            fontsize=fontsize, fontweight=fontweight,
            ha='left', va='bottom',
            alpha=0, clip_on=True
        )
        bbox = get_text_bbox(ax, test_text, renderer)
        test_text.remove()

        overlap_count = 0
        for pb in placed_bboxes:
            if check_overlap(bbox, pb):
                overlap_count += 1

        axis_bad = bbox_hits_axis_zone(bbox, xlim, ylim, pad_x, pad_y)
        if hard_avoid_axis and axis_bad:
            axis_penalty = 10000  # 軸付近へのペナルティを大幅に増加
        else:
            axis_penalty = 100 if axis_bad else 0

        # 重なり最優先 → 軸 → 距離
        score = overlap_count * 1000 + axis_penalty + r

        if overlap_count < best_overlap or (overlap_count == best_overlap and score < best_score):
            best_overlap = overlap_count
            best_score = score
            best_pos = (tx, ty)

            if overlap_count == 0 and (not axis_bad):
                return best_pos

    return best_pos


def parse_cluster_txt(cluster_txt_path):
    """
    cluster_txt を読み込んで
      - cluster_dict[sample] = cluster_id (0-based int)
      - cluster_labels[cluster_id] = label
    を返す。
    """
    cluster_dict = {}
    cluster_labels = {}
    cluster_id_map = {}  # raw header token (e.g. "1", "6-2") -> sequential 0-based int
    current_cluster = None

    # DEL/INS の nested.txt 由来の cluster_txt は "# cluster 6-2" のような
    # "{top}-{sub}" 複合番号のヘッダを持つ（SNP側は単純な "# cluster N"）。
    # \S+ で番号トークン全体（"6-2" 等）を1つのIDとして捉える -- \d+ だと
    # 先頭の数字しか拾えず、"6-1"/"6-2" が同じクラスタに潰れてしまい、
    # 残りの "-2" が誤って凡例ラベルとして使われてしまう。
    header_re = re.compile(r"^#\s*(クラスタ|cluster)\s+(\S+)\s*(.*)$", re.IGNORECASE)

    with open(cluster_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            m = header_re.match(s)
            if m:
                raw_id = m.group(2)
                if raw_id not in cluster_id_map:
                    cluster_id_map[raw_id] = len(cluster_id_map)
                current_cluster = cluster_id_map[raw_id]
                label_tail = (m.group(3) or "").strip()
                cluster_labels[current_cluster] = label_tail if label_tail else f"Cluster {raw_id}"
                continue

            if current_cluster is None:
                current_cluster = 0
                cluster_labels.setdefault(0, "Cluster 1")

            cluster_dict[s] = current_cluster

    return cluster_dict, cluster_labels


def parse_new_cluster_txt(path):
    """
    4_cluster.py が出力するクラスタ TXT を読む。
    ヘッダは "# cluster N"（通常）または "# H22" などハプロタイプ名どちらでも可。
    戻り値: {sample_name: label_str}
    """
    result = {}
    current_label = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                current_label = s[1:].strip()
            elif current_label is not None:
                result[s] = current_label
    return result


# ハプロタイプラベル用の色リスト（クラスタ色と被らないよう選定）
_HAP_COLORS = [
    "#E64C4C", "#4C9BE6", "#4CE64C", "#E6A04C",
    "#B04CE6", "#4CE6E6", "#E64CB0", "#808080",
]


def detect_sample_col(df):
    # サンプル列名を自動判定
    for cand in ["sample", "chara_value", "Sample", "SAMPLE"]:
        if cand in df.columns:
            return cand
    return df.columns[0]


def get_cluster_color(cluster_id):
    """
    0..9 は固定色。
    それ以上は matplotlib の tab20 から循環で割り当て。
    """
    fixed = {
        0: "red", 1: "blue", 2: "gold", 3: "green", 4: "gray",
        5: "purple", 6: "deepskyblue", 7: "hotpink", 8: "saddlebrown", 9: "darkorange"
    }
    if cluster_id in fixed:
        return fixed[cluster_id]
    cmap = plt.get_cmap("tab20")
    return cmap(cluster_id % 20)


def main(old_input_path, new_input_path, pca_model_path, cluster_txt_path, output_file, plot_pc,
         model_path=None, new_cluster_txt_path=None, case_map=None):
    case_map = case_map or {}
    # 1) PCAモデル（joblib）をロード
    pca = joblib.load(pca_model_path)

    # nullモデル（学習時に特徴量0列）の場合はPCAプロットをスキップ
    if isinstance(pca, dict) and pca.get("null_model"):
        print("[WARN] PCA model is null (0 feature columns in training data). Skipping PCA plot.")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "PCA not available\n(0 feature columns in training data)",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Skipped (null PCA): {output_file}")
        return

    # 2) kmeansモデルからハプロタイプ情報を取得（オプション）
    haplotype_info = {}
    if model_path:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            haplotype_info = bundle.get("haplotype_info", {})

    hap_names = [n for n, info in haplotype_info.items() if info.get("matched_samples")]
    hap_color_dict = {n: _HAP_COLORS[i % len(_HAP_COLORS)] for i, n in enumerate(hap_names)}
    sample_to_hap = {}
    for hap_name, info in haplotype_info.items():
        for s in info.get("matched_samples", []):
            sample_to_hap[s] = hap_name

    # 3) 旧データ
    old_df = pd.read_csv(old_input_path, sep="\t")
    old_sample_col = detect_sample_col(old_df)
    old_feat_cols = [c for c in old_df.columns if c != old_sample_col]
    X_old = old_df[old_feat_cols].values

    # 特徴量0列、またはPCAモデルの学習時特徴量数と食い違う場合はプレースホルダーを出して終了
    # （--pca-model-* が古い実行のjoblibで、--old-input-* が修正後のロジックで
    #   再生成され0列になった、というミスマッチをここで検知する）
    n_expected = getattr(pca, "n_features_in_", None)
    if X_old.shape[1] == 0 or (n_expected is not None and X_old.shape[1] != n_expected):
        msg = ("PCA not available\n(0 feature columns in training data)"
               if X_old.shape[1] == 0 else
               f"PCA not available\n(feature mismatch: model={n_expected}, data={X_old.shape[1]})")
        print(f"[WARN] {msg.replace(chr(10), ' ')}. Skipping PCA plot.")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Skipped (feature mismatch): {output_file}")
        return

    reduced_old = pca.transform(X_old)

    # 4) 新データ（旧データの列順に揃える。新TSVにない列は0で補完）
    new_df = pd.read_csv(new_input_path, sep="\t")
    new_sample_col = detect_sample_col(new_df)
    new_aligned = new_df.reindex(columns=[new_sample_col] + old_feat_cols, fill_value=0)
    X_new = new_aligned[old_feat_cols].values
    n_missing = len([c for c in old_feat_cols if c not in new_df.columns])
    if n_missing:
        print(f"[WARN] {n_missing}/{len(old_feat_cols)} old-data column(s) are missing from the new data; filled with 0.")
    reduced_new = pca.transform(X_new)

    # 5) クラスタ情報（旧データ用）
    cluster_dict, cluster_labels = parse_cluster_txt(cluster_txt_path)

    # 6) サンプル名
    old_samples = list(old_df[old_sample_col].astype(str))
    new_samples = list(new_df[new_sample_col].astype(str))
    old_sample_idx = {s: i for i, s in enumerate(old_samples)}

    # old_input（sc内部処理で大文字化されたサンプル名）と cluster_txt（元のBAM名の
    # ケースに復元済み -- restore_case_in_cluster_file()）とでは大文字小文字が
    # 一致しないことがあるため、大文字化した版でも引けるようにしておく。
    # これが無いとほぼ全サンプルがクラスタ検索に失敗し、デフォルトの0
    # （Cluster 1）に落ちてしまう。
    cluster_dict_upper = {k.upper(): v for k, v in cluster_dict.items()}

    # 7) 色：旧サンプルはクラスタ色（ハプロタイプ一致品種はハプロタイプ色で上書き）
    #         新サンプルは、配属されたクラスタと同じ色（四角マーカーの色分けはこの下で行う）。
    #         --new_cluster_txt が渡されなかった場合や対応が見つからない場合は黒にフォールバックする。
    colors_old = []
    old_cluster_ids = []
    for s in old_samples:
        cid = cluster_dict.get(s)
        if cid is None:
            cid = cluster_dict_upper.get(s.upper())
        if cid is None:
            cid = 0
        old_cluster_ids.append(cid)
        hap = sample_to_hap.get(s)
        colors_old.append(hap_color_dict[hap] if hap else get_cluster_color(cid))

    colors_new = ["black"] * len(new_samples)
    if new_cluster_txt_path:
        new_cluster_map = parse_new_cluster_txt(new_cluster_txt_path)
        colors_new = []
        for s in new_samples:
            label = new_cluster_map.get(s)
            color = "black"
            if label is not None:
                m = re.match(r"cluster\s+(\d+)$", label, re.IGNORECASE)
                if m:
                    color = get_cluster_color(int(m.group(1)) - 1)
                elif label in hap_color_dict:
                    color = hap_color_dict[label]
            colors_new.append(color)

    # 8) 寄与率
    explained_variance = pca.explained_variance_ratio_ * 100

    # 9) PC指定チェック
    if len(plot_pc) % 2 != 0:
        raise ValueError("Please provide an even number of PC values (pairs).")

    pc_pairs = [(plot_pc[i] - 1, plot_pc[i + 1] - 1) for i in range(0, len(plot_pc), 2)]
    max_pc_index = max(max(a, b) for a, b in pc_pairs)
    if max_pc_index >= reduced_old.shape[1]:
        raise ValueError(f"Requested PC index {max_pc_index+1} exceeds available components: {reduced_old.shape[1]}")

    # 10) subplot配置（元の形）
    num_plots = len(pc_pairs)
    cols = min(num_plots, 3)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    # 11) プロット
    for idx, (pc_x, pc_y) in enumerate(pc_pairs):
        ax = axes[idx]

        # 旧サンプル: ハプロタイプ一致は三角、それ以外は円
        non_hap_idx = [i for i, s in enumerate(old_samples) if s not in sample_to_hap]
        hap_idx     = [i for i, s in enumerate(old_samples) if s in sample_to_hap]
        if non_hap_idx:
            ax.scatter(reduced_old[non_hap_idx, pc_x], reduced_old[non_hap_idx, pc_y],
                       c=[colors_old[i] for i in non_hap_idx], marker='o', s=40, edgecolor='k')
        if hap_idx:
            ax.scatter(reduced_old[hap_idx, pc_x], reduced_old[hap_idx, pc_y],
                       c=[colors_old[i] for i in hap_idx], marker='^', s=60, edgecolor='k')

        ax.scatter(reduced_new[:, pc_x], reduced_new[:, pc_y],
                   c=colors_new, marker='s', s=40, edgecolor='k')

        ax.set_xlabel(f"PC{pc_x + 1} ({explained_variance[pc_x]:.1f}%)")
        ax.set_ylabel(f"PC{pc_y + 1} ({explained_variance[pc_y]:.1f}%)")
        ax.set_title(f"PCA Plot (PC{pc_x + 1} vs PC{pc_y + 1})")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # renderer取得（bbox計算用）
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # ★ 既存（old）のラベルは元の位置にそのまま描く
        placed_bboxes = []

        for i, label in enumerate(old_samples):
            x, y = reduced_old[i, pc_x], reduced_old[i, pc_y]
            display_label = case_map.get(label, label)
            t = ax.text(x, y, display_label, fontsize=7, ha='left', va='bottom', clip_on=False)
            placed_bboxes.append(get_text_bbox(ax, t, renderer))

        # ★ 新（new）のラベルは被らない位置へ（黒矢印、サンプル名）
        for i, label in enumerate(new_samples):
            x, y = reduced_new[i, pc_x], reduced_new[i, pc_y]
            display_label = case_map.get(label, label)

            best_pos = find_best_position(
                ax, x, y, display_label, placed_bboxes, renderer,
                fontsize=8, fontweight='bold',
                axis_pad_frac=0.05,
                max_radius_mult=30,
                n_angles=32,
                hard_avoid_axis=True
            )

            ax.annotate(
                display_label, xy=(x, y), xytext=best_pos,
                fontsize=8, fontweight='bold', ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                annotation_clip=False
            )

            tmp = ax.text(best_pos[0], best_pos[1], display_label,
                          fontsize=8, fontweight='bold', alpha=0, clip_on=True)
            placed_bboxes.append(get_text_bbox(ax, tmp, renderer))
            tmp.remove()

    # 使わないsubplotを消す
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")

    # 12) 凡例
    unique_cluster_ids = sorted(set(old_cluster_ids))
    patches = [
        mpatches.Patch(color=get_cluster_color(c),
                       label=cluster_labels.get(c, f"Cluster {c+1}"))
        for c in unique_cluster_ids
    ]
    for hap in hap_names:
        patches.append(mpatches.Patch(color=hap_color_dict[hap], label=f"Allele: {hap}"))
    # 新サンプルは配属クラスタの色で塗られるため、凡例では色ではなく四角い形で示す。
    from matplotlib.lines import Line2D
    patches.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                           markeredgecolor='black', markersize=8, label="New Samples"))
    # 軸領域と被らないよう、先にレイアウトを整えてから右側に専用の余白を確保して
    # 配置する -- tight_layout() は fig.legend() の存在を考慮しないため、凡例を
    # 先に置いてから tight_layout() を呼ぶと軸領域が凡例のスペースまで広がって
    # しまい重なる。
    plt.tight_layout()
    plt.subplots_adjust(right=0.70)  # 右側に凡例用の余白を確保
    fig.legend(handles=patches, title="Clusters / Alleles", loc='upper left', bbox_to_anchor=(1.03, 1))

    # 凡例を含めて保存
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)

    print(f"[Done] Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="既存のPCA空間に新しいデータを追加してプロット（PCAモデルはjoblib）")
    parser.add_argument("--old_input", required=True, help="もとのサンプルの特徴量TSVファイル（k-meansに使ったデータと同じ）")
    parser.add_argument("--new_input", required=True, help="新しいサンプルの特徴量TSVファイル")
    parser.add_argument("--pca_model", required=True, help="保存されたPCAモデル（.joblib）")
    parser.add_argument("--cluster_txt", required=True, help="もとのクラスタリング結果テキスト（# cluster / # クラスタ 両対応）")
    parser.add_argument("--new_cluster_txt", default=None,
                        help="新サンプルのクラスタ割当テキスト（4_cluster.py の出力）。"
                             "指定時は新サンプルの四角マーカーを配属クラスタと同じ色で着色する"
                             "（未指定時は従来通り黒）。")
    parser.add_argument("--output", required=True, help="出力PNGファイル名")
    parser.add_argument("--plot_pc", nargs="+", type=int, required=True,
                        help="プロットするPC番号をペアで指定 (例: --plot_pc 1 2 1 3 2 3)")
    parser.add_argument("--model", default=None,
                        help="step1 の KMeansモデル（.joblib）。haplotype_info が含まれる場合、"
                             "対応する旧サンプルをハプロタイプ色で着色し矢印アノテーションを追加する。")
    parser.add_argument("--case-map", default=None,
                        help="2列TSV（UPPER(sample)<TAB>元の表記）。指定時は新サンプルの"
                             "ラベル表示を、sc内部処理で大文字化される前の元のBAM名の"
                             "表記に戻す。")
    args = parser.parse_args()

    main(args.old_input, args.new_input, args.pca_model, args.cluster_txt, args.output, args.plot_pc,
         args.model, args.new_cluster_txt, case_map=load_case_map(args.case_map))
import pandas as pd
import numpy as np
import argparse
import math
import os
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def load_cluster_file(cluster_path):
    cluster_dict = {}
    cluster_labels = {}
    current_cluster_id = -1
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                current_cluster_id += 1
                label = line[1:].strip()  # '#' を除いて空白を除去
                cluster_labels[current_cluster_id] = label
            elif line:
                cluster_dict[line] = current_cluster_id
    df = pd.DataFrame(list(cluster_dict.items()), columns=["sample", "cluster"])
    return df, cluster_labels

def main(input_path, cluster_path, output_file, n_components, plot_pc, variance_output, pca_model_out=None, case_map=None):
    case_map = case_map or {}
    # 1. 元特徴量データの読み込み
    df = pd.read_csv(input_path, sep="\t")
    # 先頭列名を sample に統一（5_number.py は chara_value という名前で出力する）
    df = df.rename(columns={df.columns[0]: "sample"})

    # 2. クラスタ情報の読み込み
    cluster_df, cluster_labels = load_cluster_file(cluster_path)

    # 3. マージとNaN除去
    merged_df = df.merge(cluster_df, on="sample", how="inner")
    merged_df = merged_df.dropna(subset=["cluster"])

    # 4. PCAの実行
    X = merged_df.iloc[:, 1:-1].values
    n_feat = X.shape[1]
    n_samp = X.shape[0]

    # 特徴量が不足している場合（データなし）はプレースホルダー画像を出力して終了
    if n_feat == 0 or n_samp == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No valid data for PCA\n(0 features or 0 samples)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("PCA — No Data")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved placeholder figure to: {output_file}")
        with open(variance_output, "w") as f:
            f.write("No data\n")
        return

    # n_components を利用可能な上限に丸める
    max_components = min(n_samp, n_feat)
    if n_components > max_components:
        print(f"[WARN] Changing n_components={n_components} to available maximum {max_components}.")
        n_components = max_components

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_

    # 5. PCペアの取得
    if len(plot_pc) % 2 != 0:
        raise ValueError("Please provide an even number of PC values (pairs).")
    pc_pairs = [(plot_pc[i] - 1, plot_pc[i + 1] - 1) for i in range(0, len(plot_pc), 2)]

    # 6. 色の割り当て
    cluster_colors = {
        0: "red", 1: "blue", 2: "gold", 3: "green", 4: "gray",
        5: "purple", 6: "deepskyblue", 7: "hotpink", 8: "saddlebrown", 9: "darkorange"
    }
    colors = merged_df["cluster"].map(cluster_colors)

    # 7. 凡例パッチ（任意ラベル対応）
    valid_clusters = [c for c in sorted(merged_df["cluster"].dropna().unique()) if c in cluster_colors]
    patches = [
        mpatches.Patch(color=cluster_colors[c], label=cluster_labels.get(c, f"Cluster {c}"))
        for c in valid_clusters
    ]

    # 8. サブプロットの準備
    num_plots = len(pc_pairs)
    cols = min(num_plots, 3)
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (pc_x, pc_y) in enumerate(pc_pairs):
        ax = axes[idx]
        if max(pc_x, pc_y) >= reduced.shape[1]:
            print(f"Skipping PC{pc_x+1} vs PC{pc_y+1}: exceeds available components")
            continue

        ax.scatter(reduced[:, pc_x], reduced[:, pc_y], c=colors, edgecolor='k')
        for i, label in enumerate(merged_df["sample"]):
            ax.text(reduced[i, pc_x], reduced[i, pc_y], case_map.get(label, label), fontsize=8)

        ax.set_xlabel(f"PC{pc_x + 1} ({var_ratio[pc_x]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y + 1} ({var_ratio[pc_y]*100:.1f}%)")
        ax.set_title(f"PC{pc_x + 1} vs PC{pc_y + 1}")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 9. 凡例（軸領域と被らないよう、先にレイアウトを整えてから右側に専用の余白を
    # 確保して配置する -- tight_layout() は fig.legend() の存在を考慮しないため、
    # 順序を誤ると凡例がプロット領域に重なる）
    plt.tight_layout()
    fig.subplots_adjust(right=0.70)
    fig.legend(handles=patches, title="Clusters", loc='upper left', bbox_to_anchor=(1.03, 1))

    # 10. 出力
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined figure to: {output_file}")

    # 11. 寄与率の書き出し
    with open(variance_output, "w") as f:
        for i, ratio in enumerate(var_ratio):
            f.write(f"PC{i + 1}: {ratio * 100:.4f}%\n")
    print(f"Saved explained variance ratios to: {variance_output}")

    # 12. PCAモデルの保存（--pca-model-out 指定時）
    if pca_model_out:
        joblib.dump(pca, pca_model_out)
        print(f"Saved PCA model to: {pca_model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA-based cluster visualization with flexible legend labels.")
    parser.add_argument("--input", type=str, required=True, help="Path to feature .tsv file (same data used for k-means)")
    parser.add_argument("--cluster", type=str, required=True, help="Path to cluster .txt file")
    parser.add_argument("--output", type=str, default="combined_plot.png", help="Output PNG image file name")
    parser.add_argument("--components", type=int, default=2, help="Number of PCA components to compute")
    parser.add_argument("--plot_pc", nargs="+", type=int, default=[1, 2],
                        help="PCs to plot as pairs (e.g., --plot_pc 1 2 1 3 2 3)")
    parser.add_argument("--variance_output", type=str, default="variance.txt", help="Output text file for variance ratios")
    parser.add_argument("--pca-model-out", default=None, help="PCAモデルをjoblib形式で保存するパス（step3で再利用）")
    parser.add_argument("--case-map", default=None,
                        help="2列TSV（大文字化サンプル名<TAB>元の表記）。指定すると点ラベルを元の表記で描画する。")

    args = parser.parse_args()
    main(args.input, args.cluster, args.output, args.components, args.plot_pc, args.variance_output,
         pca_model_out=args.pca_model_out, case_map=load_case_map(args.case_map))

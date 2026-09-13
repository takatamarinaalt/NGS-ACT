#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description="KMeans clustering with user-defined parameters (joblib version).")
parser.add_argument("--input", type=str, required=True, help="Path to the input TSV file.")
parser.add_argument("--n_clusters", type=int, default=2, help="Number of clusters for KMeans.")
parser.add_argument("--output", type=str, default="cluster_result.txt", help="Output file for cluster labels.")
parser.add_argument("--model", type=str, default="kmeans_model.joblib", help="Output file for saving the model.")
parser.add_argument("--region", type=str, default=None,
                    help="遺伝子領域（例: chr01:38382000-38390000）。joblib に保存され、"
                         "step3/1 のパイプラインで VCF 作成領域として自動参照される。")
parser.add_argument("--start_number", type=int, default=1,
                    help="クラスタ番号の開始値（デフォルト: 1）。'cluster {start_number}', "
                         "'cluster {start_number+1}', ... の形式で採番する。"
                         "二段階クラスタリングのstage1で、一次クラスタをまたいで"
                         "全体を通し番号にする（例: 1つ目の一次クラスタが cluster 1, cluster 2 を"
                         "使ったら、2つ目の一次クラスタは cluster 3 から開始する）ために使用する。"
                         "同じ番号が異なるSNPパターンの群を指してしまう誤読を避けるため、"
                         "一次クラスタ番号と二次クラスタ番号を混在させた表記（例: C1-a）は用いない。")
args = parser.parse_args()

df = pd.read_csv(args.input, sep="\t")

# 学習に使った特徴量列名（列順がKMeansにとって重要なので保存する）
feature_cols = df.columns[1:].tolist()

# 学習データ行列
data_matrix = df.loc[:, feature_cols].values

kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_matrix)

cluster_labels = {}  # cluster_id(0始まり) -> 表示ラベル文字列

with open(args.output, "w", encoding="utf-8") as f:
    for idx, cluster_id in enumerate(sorted(set(clusters))):
        label = f"cluster {args.start_number + idx}"
        cluster_labels[cluster_id] = label

        f.write(f"# {label}\n")
        cluster_varieties = df.iloc[:, 0][clusters == cluster_id]
        for variety in cluster_varieties:
            f.write(f"{variety}\n")
        f.write("\n")

print(f"[OK] Saved cluster results to {args.output}. "
      f"(cluster {args.start_number} - cluster {args.start_number + len(cluster_labels) - 1})")

bundle = {
    "model": kmeans,
    "samples": df.iloc[:, 0].tolist(),
    "labels_top": (clusters + 1).tolist(),  # 1始まり（数値。従来のダウンストリーム互換のため維持）
    "cluster_labels": [cluster_labels[c] for c in clusters],  # 表示ラベル文字列（サンプルごと）
    "method": "kmeans",
    "n_clusters": args.n_clusters,
    "start_number": args.start_number,
    "feature_cols": feature_cols,
    "gene_region": args.region,
}
joblib.dump(bundle, args.model)

print(f"[OK] Saved the trained KMeans model and sample info to {args.model}.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _project_point_to_line(p, a, b):
    """
    点 p を直線 a-b に正射影した点を返す。
    p, a, b: np.array([x, y])
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return a.copy()
    t = float(np.dot(p - a, ab) / denom)
    return a + t * ab


def find_elbow_point_normalized(x, y):
    """
    端点 (x[0],y[0]) と (x[-1],y[-1]) を結ぶ直線に対して、
    「正規化座標での垂線距離」が最大の点を elbow とする。
    環境に依存せず、常に同じ結果になる。
    戻り値: (best_k, best_idx)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 正規化（0-1スケール）
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    # ゼロ除算を防ぐ
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0
    
    x_norm = (x - x.min()) / x_range
    y_norm = (y - y.min()) / y_range

    pts = np.column_stack([x_norm, y_norm])
    a = pts[0]
    b = pts[-1]

    distances = []
    for i in range(len(pts)):
        p = pts[i]
        proj = _project_point_to_line(p, a, b)
        dist = float(np.linalg.norm(p - proj))
        distances.append(dist)

    # 端点は距離0になりやすいので除外（点が3つ以上ある場合）
    if len(distances) >= 3:
        idx = int(1 + np.argmax(distances[1:-1]))
    else:
        idx = int(np.argmax(distances))

    return int(x[idx]), idx


# 引数
parser = argparse.ArgumentParser(description="正規化座標の垂線距離最大でエルボーkを選択")
parser.add_argument("--input", required=True, help="数値データが入ったTSVファイルのパス")
parser.add_argument("--max_k", type=int, default=10, help="最大クラスタ数")
parser.add_argument("--output_fig", default="elbow_plot.png", help="出力画像ファイル名")
parser.add_argument("--print_elbow", action="store_true", help="最適クラスタ数を表示する")
args = parser.parse_args()

import sys

# データ読み込み
df = pd.read_csv(args.input, sep="\t")
X = df.iloc[:, 1:].values

# 特徴量が0列（ポジションなし）の場合は k=1 として処理
if X.shape[1] == 0:
    print(f"[WARN] Feature count is 0 (no valid positions remain after filtering). Processing as k=1.")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, "No valid positions after filtering\n(k=1 assigned)",
            ha="center", va="center", fontsize=14, transform=ax.transAxes)
    ax.set_title("Elbow Method — No Data")
    plt.tight_layout()
    plt.savefig(args.output_fig)
    if args.print_elbow:
        print(f"Final selected optimal cluster count: k = 1")
    sys.exit(0)

# SSE計算
inertias = []
cluster_range = range(1, args.max_k + 1)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(float(kmeans.inertia_))

# 最適クラスタ数の決定（正規化座標の垂線距離最大）
final_k, final_idx = find_elbow_point_normalized(list(cluster_range), inertias)

# プロット
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(cluster_range, inertias, marker='o', label="SSE")

ax.set_xlabel("Number of Clusters (k)", fontsize=14)
ax.set_ylabel("Inertia (SSE)", fontsize=14)
ax.set_title("Elbow Method with SSE", fontsize=16)

# grid線をオフにする
ax.grid(False)

# 上と右の枠線を消す
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 軸の線を太くする
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(2)

# 軸目盛りの数字を太く＆大きくする
ax.tick_params(axis='both', which='major', labelsize=12, width=2)

# x軸の目盛り設定
ax.set_xticks(list(cluster_range))

# 選択kの表示
ax.axvline(x=final_k, color='red', linestyle='--', label=f"Selected k = {final_k}")
ax.scatter(final_k, inertias[final_idx], color='red')

# 凡例
ax.legend()

# レイアウト調整して保存
plt.tight_layout()
plt.savefig(args.output_fig)

# 最終出力
if args.print_elbow:
    print(f"Final selected optimal cluster count: k = {final_k}")
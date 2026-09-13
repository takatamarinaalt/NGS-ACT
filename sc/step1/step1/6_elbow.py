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
    点 p を直線 a-b に正射影した点を返す（同一座標系）。
    p, a, b: np.array([x, y])
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return a.copy()
    t = float(np.dot(p - a, ab) / denom)
    return a + t * ab


def find_elbow_point_display(ax, x, y):
    """
    端点 (x[0],y[0]) と (x[-1],y[-1]) を結ぶ直線に対して、
    「画面上（表示座標＝ピクセル座標）での垂線距離」が最大の点を elbow とする。
    戻り値: (best_k, best_idx)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    pts_data = np.column_stack([x, y])
    pts_disp = ax.transData.transform(pts_data)  # データ座標 -> 表示座標

    a = pts_disp[0]
    b = pts_disp[-1]

    distances = []
    for i in range(len(pts_disp)):
        p = pts_disp[i]
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
parser = argparse.ArgumentParser(description="表示座標の垂線距離最大でエルボーkを選択（図の見た目は元スクリプト準拠）")
parser.add_argument("--input", required=True, help="数値データが入ったTSVファイルのパス")
parser.add_argument("--max_k", type=int, default=10, help="最大クラスタ数")
parser.add_argument("--output_fig", default="elbow_plot.png", help="出力画像ファイル名")
parser.add_argument("--print_elbow", action="store_true", help="最適クラスタ数を表示する")
args = parser.parse_args()

# データ読み込み
df = pd.read_csv(args.input, sep="\t")
X = df.iloc[:, 1:].values

# SSE計算
inertias = []
cluster_range = range(1, args.max_k + 1)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(float(kmeans.inertia_))

# プロット（見た目はあなたのスクリプトと同じ）
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

# transform を安定させるため、limits を固定して draw
ax.set_xlim(min(cluster_range) - 0.5, max(cluster_range) + 0.5)
y_min, y_max = min(inertias), max(inertias)
margin = (y_max - y_min) * 0.05
if margin == 0:
    margin = 1.0
ax.set_ylim(y_min - margin, y_max + margin)
fig.canvas.draw()

# 最適クラスタ数の決定（表示座標の垂線距離最大）
final_k, final_idx = find_elbow_point_display(ax, list(cluster_range), inertias)

# 選択kの表示（これもあなたのスクリプトと同じ見た目）
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

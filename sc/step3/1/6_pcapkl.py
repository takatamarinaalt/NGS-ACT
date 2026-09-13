#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import joblib
from sklearn.decomposition import PCA

def save_pca_model(input_tsv: str, output_pca_model: str, n_components: int):
    # 1. データ読み込み
    df = pd.read_csv(input_tsv, sep="\t")

    # 2. 数値データ部分だけ抽出（1列目はsample名だと仮定）
    X = df.iloc[:, 1:].values

    # 3. n_componentsの自動調整
    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    
    if n_components > max_components:
        print(f"[WARN] n_components={n_components} exceeds max({n_samples} samples, {n_features} features)={max_components}")
        print(f"[WARN] Auto-adjusting n_components to {max_components}")
        n_components = max_components

    # 4. PCAを実行（fitだけして座標軸を決める）
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # 5. PCAモデルを joblib で保存
    joblib.dump(pca, output_pca_model)
    print(f"[DONE] Saved PCA model (joblib format): {output_pca_model}")
    print(f"[INFO] n_components={n_components}, explained_variance_ratio={pca.explained_variance_ratio_}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCAモデルをjoblib形式で保存するスクリプト")
    parser.add_argument("--input_tsv", required=True, help="数値データTSV（1列目=sample、2列目以降=数値特徴量）")
    parser.add_argument("--output_pca_model", default="pca_model.joblib",
                        help="保存するPCAモデルファイル名（.joblib 推奨）")
    parser.add_argument("--n_components", type=int, default=2, help="PCAの主成分数（デフォルト2）")
    args = parser.parse_args()

    save_pca_model(args.input_tsv, args.output_pca_model, args.n_components)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from collections import defaultdict


def parse_cluster_file(path):
    """
    '# cluster <label>' ブロックで区切られたサンプル名一覧を読み込み、
    サンプル名(大文字正規化) -> クラスタラベル(str) の dict を返す。
    """
    mapping = {}
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                m = re.search(r"cluster\s+(\S+)", line, re.IGNORECASE)
                current = m.group(1) if m else line.lstrip("#").strip()
                continue
            sample = line.strip()
            mapping[sample.upper()] = current
    return mapping


def _sort_key(label):
    # 数字ラベルは数値順、それ以外は文字列順
    return (0, int(label)) if label.isdigit() else (1, label)


def combine_clusters(cluster_specs, output_path):
    """
    cluster_specs: [(label_key, path), ...] の順序付きリスト（結合ラベルの並び順もこの順になる）。
    3つ（snp/del/ins）でも4つ（snp/del/instp/insnottp）でも、任意の個数に対応する。
    """
    maps = {key: parse_cluster_file(path) for key, path in cluster_specs}
    all_samples = sorted(set().union(*(set(m) for m in maps.values())))

    groups = defaultdict(list)
    missing = []
    for sample in all_samples:
        vals = [maps[key].get(sample) for key, _ in cluster_specs]
        if any(v is None for v in vals):
            missing.append((sample, dict(zip((k for k, _ in cluster_specs), vals))))
            continue
        # GENE_ABSENT（DEL/INS_TP/INS_notTPのいずれかで、走査した全座位の
        # リード深度が0だったサンプル）は、SNP側の実クラスタ番号と組み合わせた
        # "3-GENE_ABSENT-GENE_ABSENT-GENE_ABSENT" のような複合ラベルにはせず、
        # 単に "GENE_ABSENT" 一語にまとめる。
        if "GENE_ABSENT" in vals:
            groups[("GENE_ABSENT",)].append(sample)
        else:
            groups[tuple(vals)].append(sample)

    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: tuple(_sort_key(x) for x in k),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for key in sorted_keys:
            label = "-".join(key)
            # GENE_ABSENTは実際のクラスタ番号の組み合わせではないので、他の
            # 複合ラベル（"HapA-2-3"等、既存のアリル名混在ケース）と違い
            # "Allele"を付けない -- 個別のDEL/INS cluster.txtの見出し
            # （write_cluster_txt()の"# GENE_ABSENT"）と同じ書式に揃える。
            header = f"# {label}" if key == ("GENE_ABSENT",) else f"# Allele {label}"
            f.write(header + "\n")
            for sample in sorted(groups[key]):
                f.write(f"{sample}\n")
            f.write("\n")

    keys_str = "/".join(k for k, _ in cluster_specs)
    print(f"[OK] Saved combined cluster result: {output_path}")
    if missing:
        print(f"[WARN] {len(missing)} sample(s) missing from one or more of {keys_str}:")
        for sample, vals in missing:
            detail = ", ".join(f"{k}={v}" for k, v in vals.items())
            print(f"       {sample}: {detail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="snp/DEL/INS（またはsnp/DEL/INS_TP/INS_notTP）の各クラスタ結果txtを読み込み、"
                     "cluster番号をハイフンで結合して出力する"
    )
    parser.add_argument("--snp_cluster", required=True,
                        help="snpクラスタ結果txt (step1出力)")
    parser.add_argument("--del_cluster", default=None,
                        help="DELクラスタ結果txt")
    parser.add_argument("--ins_cluster", default=None,
                        help="INSクラスタ結果txt（--instp_cluster/--insnottp_clusterとは併用不可）")
    parser.add_argument("--instp_cluster", default=None,
                        help="INS_TPクラスタ結果txt（--ins_clusterの代わりに指定。--insnottp_clusterとセットで使う）")
    parser.add_argument("--insnottp_cluster", default=None,
                        help="INS_notTPクラスタ結果txt（--instp_clusterとセットで使う）")
    parser.add_argument("--output", required=True,
                        help="出力txt")
    args = parser.parse_args()

    if args.ins_cluster and (args.instp_cluster or args.insnottp_cluster):
        parser.error("--ins_cluster cannot be used together with --instp_cluster/--insnottp_cluster.")
    if bool(args.instp_cluster) != bool(args.insnottp_cluster):
        parser.error("--instp_cluster and --insnottp_cluster must both be specified.")

    cluster_specs = [("snp", args.snp_cluster)]
    if args.del_cluster:
        cluster_specs.append(("del", args.del_cluster))
    if args.ins_cluster:
        cluster_specs.append(("ins", args.ins_cluster))
    elif args.instp_cluster:
        cluster_specs.append(("instp", args.instp_cluster))
        cluster_specs.append(("insnottp", args.insnottp_cluster))

    combine_clusters(cluster_specs, args.output)

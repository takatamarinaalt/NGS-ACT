#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import urllib.parse


def parse_region(region_str):
    """chr:start-end 形式から (chrom, start, end) タプルを返す（1-based, closed interval）"""
    chrom, positions = region_str.rsplit(":", 1)
    start, end = map(int, positions.split("-"))
    return (chrom, start, end)


def _merge_intervals_by_chrom(by_chrom):
    """染色体ごとの区間リスト {chrom: [(s,e), ...]} をマージして (chrom,s,e) のリストで返す"""
    merged = []
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((chrom, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((chrom, cur_s, cur_e))
    return merged


def _parse_gff_gene_features(gff_path, gene_query):
    """
    GFF3 から対象遺伝子の mRNA 区間と、そのサブフィーチャ(exon/CDS/UTR)を抽出する共通パーサ。
    parse_gff_exons（エキソン抽出）と parse_gff_introns（イントロン抽出）の両方から呼ばれる。

    【対応フォーマット】
    (1) 標準 GFF3:
        gene → mRNA/transcript → exon
    (2) RAP-DB 形式 (O.sativa IRGSP-1.0):
        gene フィーチャなし / mRNA が最上位
        exon フィーチャなし / CDS + UTR をエキソンとして扱う
        遺伝子名は Locus_id= または RAP-DB Gene Symbol Synonym(s)= (URL エンコード) に格納

    gene_query: 大文字小文字を区別しない部分一致で以下の属性を検索
      - gene / mRNA の ID= または Name=
      - Locus_id= (RAP-DB)
      - RAP-DB Gene Symbol Synonym(s)= の各シンボル
      - CGSNL Gene Symbol=
      - Oryzabase Gene Symbol Synonym(s)= の各シンボル

    戻り値: (mrna_spans, sub_feats)
      mrna_spans: {mrna_id: (chrom, start, end)}  対象 mRNA 自身の区間
      sub_feats:  [(ftype, chrom, start, end, parent_mrna_id), ...] 対象 mRNA 配下の exon/CDS/UTR
    """
    # ID を持つフィーチャ (gene / mRNA など) は辞書に格納
    id_feats  = {}   # id -> {type, chrom, start, end, attrs}
    # ID を持たないサブフィーチャ (CDS, UTR, exon) はリストに格納
    sub_feats = []   # list of (type, chrom, start, end, parent_id)

    EXON_TYPES = frozenset({'exon', 'CDS', 'five_prime_UTR', 'three_prime_UTR'})

    with open(gff_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 9:
                continue
            chrom, _, ftype = cols[0], cols[1], cols[2]
            start, end = int(cols[3]), int(cols[4])
            attrs = {}
            for item in cols[8].split(';'):
                item = item.strip()
                if '=' in item:
                    k, v = item.split('=', 1)
                    attrs[k.strip()] = v.strip()
            fid     = attrs.get('ID', '')
            parents = [p.strip() for p in attrs.get('Parent', '').split(',') if p.strip()]

            if fid:
                id_feats[fid] = {
                    'type': ftype, 'chrom': chrom,
                    'start': start, 'end': end,
                    'attrs': attrs, 'parents': parents,
                }
                # ID 付きでも exon/CDS/UTR なら sub_feats にも追加
                if ftype in EXON_TYPES:
                    for p in parents:
                        sub_feats.append((ftype, chrom, start, end, p))
            else:
                # ID なしのサブフィーチャ (CDS, UTR など) は Parent だけ使う
                if ftype in EXON_TYPES:
                    for p in parents:
                        sub_feats.append((ftype, chrom, start, end, p))

    def attr_symbols(attrs, key):
        """URL エンコードされた複数シンボルをリストで返す（RAP-DB 形式対応）"""
        raw = attrs.get(key, '')
        if not raw:
            return []
        decoded = urllib.parse.unquote(raw)
        return [s.strip() for s in decoded.split(',') if s.strip()]

    def matches(feat_attrs, query):
        """フィーチャの属性辞書が gene_query にマッチするか判定"""
        q = query.lower()
        a = feat_attrs

        # ID / Name (部分一致: Os06t0275000-01 を Os06g0275000 などで検索できるよう)
        for key in ('ID', 'Name'):
            if q in a.get(key, '').lower():
                return True

        # Locus_id (RAP-DB: 完全一致)
        if a.get('Locus_id', '').lower() == q:
            return True

        # 遺伝子シンボルは完全一致のみ（部分一致だと "Hd1 Binding Protein 1" 等を誤ヒット）
        for key in ('RAP-DB Gene Symbol Synonym(s)',
                    'CGSNL Gene Symbol',
                    'Oryzabase Gene Symbol Synonym(s)'):
            for sym in attr_symbols(a, key):
                if q == sym.lower():
                    return True

        return False

    # Step 1: 標準 GFF3 の gene フィーチャでマッチを試みる
    gene_ids = {fid for fid, feat in id_feats.items()
                if feat['type'] == 'gene' and matches(feat['attrs'], gene_query)}

    if gene_ids:
        # 標準 GFF3: gene → mRNA → exon
        mrna_ids = {fid for fid, feat in id_feats.items()
                    if feat['type'] in ('mRNA', 'transcript', 'pseudogenic_transcript')
                    and any(p in gene_ids for p in feat['parents'])}
    else:
        # RAP-DB 形式: mRNA を直接マッチング
        mrna_ids = {fid for fid, feat in id_feats.items()
                    if feat['type'] == 'mRNA' and matches(feat['attrs'], gene_query)}

    if not mrna_ids:
        return {}, []

    mrna_spans = {
        mid: (id_feats[mid]['chrom'], id_feats[mid]['start'], id_feats[mid]['end'])
        for mid in mrna_ids
    }
    relevant_sub_feats = [sf for sf in sub_feats if sf[4] in mrna_ids]

    return mrna_spans, relevant_sub_feats


def parse_gff_exons(gff_path, gene_query):
    """
    GFF3 から指定遺伝子のエキソン座標リストを返す。
    他の植物種の GFF3 でも、gene/mRNA/exon の階層構造があれば動作する。

    戻り値: list of (chrom, start, end)  ※1-based, closed interval
            マージ済み・ソート済み
    """
    mrna_spans, sub_feats = _parse_gff_gene_features(gff_path, gene_query)
    if not mrna_spans:
        return []

    # 対象 mRNA の exon / CDS / UTR 座標を収集
    regions = []
    seen = set()
    for ftype, chrom, start, end, parent in sub_feats:
        key = (chrom, start, end)
        if key not in seen:
            seen.add(key)
            regions.append(key)

    if not regions:
        return []

    # 染色体ごとに区間をマージ（複数トランスクリプト間の重複を吸収）
    by_chrom = {}
    for chrom, s, e in regions:
        by_chrom.setdefault(chrom, []).append((s, e))

    merged = _merge_intervals_by_chrom(by_chrom)
    return sorted(merged, key=lambda x: (x[0], x[1]))


def parse_gff_introns(gff_path, gene_query):
    """
    GFF3 から指定遺伝子のイントロン座標リストを返す。

    各 mRNA（トランスクリプト）ごとに、mRNA 自身の始点〜終点区間から
    exon/CDS/UTR（エキソン）区間を差し引いた残りをイントロンとする。
    複数トランスクリプトが存在する場合は、各トランスクリプトのイントロンの
    和集合（union）を最終的なイントロン領域として返す。

    シングルエキソン遺伝子（イントロンが存在しない遺伝子）の場合は空リストを返す。

    戻り値: list of (chrom, start, end)  ※1-based, closed interval
            マージ済み・ソート済み
    """
    mrna_spans, sub_feats = _parse_gff_gene_features(gff_path, gene_query)
    if not mrna_spans:
        return []

    exons_by_mrna = {}
    for ftype, chrom, start, end, parent in sub_feats:
        exons_by_mrna.setdefault(parent, []).append((start, end))

    intron_regions = []
    for mrna_id, (chrom, m_start, m_end) in mrna_spans.items():
        exon_ivs = sorted(exons_by_mrna.get(mrna_id, []))

        # 同一 mRNA 内のエキソン区間をマージ
        merged_exons = []
        for s, e in exon_ivs:
            if merged_exons and s <= merged_exons[-1][1] + 1:
                merged_exons[-1] = (merged_exons[-1][0], max(merged_exons[-1][1], e))
            else:
                merged_exons.append((s, e))

        # mRNA 区間からエキソン区間を除いた残り（隙間）がイントロン
        cursor = m_start
        for s, e in merged_exons:
            if s > cursor:
                intron_regions.append((chrom, cursor, s - 1))
            cursor = max(cursor, e + 1)
        if cursor <= m_end:
            intron_regions.append((chrom, cursor, m_end))

    if not intron_regions:
        return []

    # 複数トランスクリプトのイントロンを染色体ごとにマージ（和集合）
    by_chrom = {}
    for chrom, s, e in intron_regions:
        by_chrom.setdefault(chrom, []).append((s, e))

    merged = _merge_intervals_by_chrom(by_chrom)
    return sorted(merged, key=lambda x: (x[0], x[1]))


def extract_vcf_regions(input_vcf, output_vcf, target_regions):
    """target_regions に含まれる VCF 行のみを output_vcf に書き出す"""
    with open(input_vcf, 'r') as infile, open(output_vcf, 'w') as outfile:
        for line in infile:
            if line.startswith('#'):
                outfile.write(line)
                continue
            cols  = line.strip().split('\t')
            chrom = cols[0]
            pos   = int(cols[1])
            for target_chrom, start, end in target_regions:
                if chrom == target_chrom and start <= pos <= end:
                    outfile.write(line)
                    break


def main():
    parser = argparse.ArgumentParser(
        description="VCF から指定領域の多型を抽出する。"
                    " --mode all: 遺伝子領域全体を対象。"
                    " --mode exon: GFF3 のエキソン領域のみを対象。"
                    " --mode intron: GFF3 のイントロン領域のみを対象。"
    )
    parser.add_argument("--input_vcf",  required=True, help="入力 VCF ファイル")
    parser.add_argument("--output_vcf", required=True, help="出力 VCF ファイル")
    parser.add_argument(
        "--mode", choices=["all", "exon", "intron"], default="all",
        help="all: --regions で指定した遺伝子領域全体 (デフォルト) / "
             "exon: --gff と --gene で指定したエキソン領域のみ / "
             "intron: --gff と --gene で指定したイントロン領域のみ"
             "（亜種間の大きな違いに埋もれがちな品種内の細かいSNPの違いを"
             "捉えるための一次クラスタリング等に利用）"
    )
    # --- mode=all 用 ---
    parser.add_argument(
        "--regions", nargs='+',
        help="[mode=all 必須] 抽出領域 (例: chr06:9336376-9338569)"
    )
    # --- mode=exon / mode=intron 用 ---
    parser.add_argument(
        "--gff",
        help="[mode=exon/intron 必須] GFF3 ファイルのパス"
    )
    parser.add_argument(
        "--gene",
        help="[mode=exon/intron 必須] 対象遺伝子の ID / Name / Locus_id / 遺伝子シンボル"
             " (大文字小文字を区別しない部分一致で検索)"
    )

    args = parser.parse_args()

    # --- モード別に対象領域を決定 ---
    if args.mode == "all":
        if not args.regions:
            parser.error("--mode all requires --regions")
        target_regions = [parse_region(r) for r in args.regions]
        print(f"[INFO] mode=all: targeting {len(target_regions)} region(s)")

    else:  # mode=exon / mode=intron
        if not args.gff or not args.gene:
            parser.error(f"--mode {args.mode} requires --gff and --gene")

        if args.mode == "exon":
            target_regions = parse_gff_exons(args.gff, args.gene)
            region_kind = "exon"
        else:  # intron
            target_regions = parse_gff_introns(args.gff, args.gene)
            region_kind = "intron"

        if not target_regions:
            hint = ""
            if args.mode == "intron":
                hint = "\n        (single-exon genes have no introns)"
            print(
                f"[ERROR] No {region_kind} regions found in the GFF3 for gene '{args.gene}'.\n"
                f"        Please check ID= / Name= / Locus_id= / the gene symbol.{hint}",
                file=sys.stderr
            )
            sys.exit(1)

        print(f"[INFO] mode={args.mode}: targeting {len(target_regions)} {region_kind} region(s) for gene '{args.gene}'")
        for chrom, start, end in target_regions:
            print(f"       {chrom}:{start}-{end}  (length {end - start + 1} bp)")

    # --- VCF フィルタリング実行 ---
    extract_vcf_regions(args.input_vcf, args.output_vcf, target_regions)
    print(f"[OK] Extraction complete: {args.output_vcf}")


if __name__ == "__main__":
    main()

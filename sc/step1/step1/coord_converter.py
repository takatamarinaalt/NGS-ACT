#!/usr/bin/env python3
"""
coord_converter.py
アミノ酸座標・CDS座標 → ゲノム座標変換スクリプト

座標系:
  aa  : アミノ酸座標 (開始Metを1とする)
  cds : CDS座標 (開始コドンのAを1とする、イントロン除く)
  gdna: ゲノムDNA連番 (開始コドンのAを1とする、イントロン含む)

使用例:
  # CDS座標 → ゲノム座標
  python coord_converter.py --gff gff/transcripts.gff --gene Os01g0100100 --coord-type cds --pos 500

  # アミノ酸座標 → ゲノム座標
  python coord_converter.py --gff gff/transcripts.gff --gene Os01g0100100 --coord-type aa --pos 100

  # バッチ処理（TSVファイルから）
  python coord_converter.py --gff gff/transcripts.gff --batch input.tsv --output results.tsv

  # 他種（例: シロイヌナズナ）は --gff に対応するGFFを指定するだけ
  python coord_converter.py --gff /path/to/TAIR10.gff --gene AT1G01010 --coord-type cds --pos 300

バッチ入力TSVの形式 (ヘッダー行必須):
  gene_id         coord_type  pos   note
  Os01g0100100    cds         500   example_variant
  Os01g0100300    aa          50    another_variant
"""

import argparse
import sys
from collections import defaultdict


# ============================================================
#  GFF3 / GTF パーサー
# ============================================================

def parse_attrs(attrs_str):
    """
    GFF3またはGTFのattributes列をdictに変換する。

    GFF3形式: key=value;key=value
    GTF形式:  key "value"; key "value"

    判定方法: 先頭の区切り要素に '=' が含まれるかどうかで判別する。
    値の中に '"' が含まれる GFF3（RAP-DB 等）でも誤検出しない。
    """
    result = {}
    stripped = attrs_str.strip()
    if not stripped:
        return result

    # 先頭のセミコロン区切り要素に '=' があれば GFF3 形式とみなす
    first_item = stripped.split(";")[0].strip()
    is_gff3 = "=" in first_item

    if is_gff3:
        for item in stripped.split(";"):
            item = item.strip()
            if "=" in item:
                key, _, val = item.partition("=")
                result[key.strip()] = val.strip()
    else:
        # GTF 形式: gene_id "value"; transcript_id "value"
        for item in stripped.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(None, 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().strip('"')
                result[key] = val
    return result


def _extract_gene_symbols(attrs):
    """
    GFF3のattributesから遺伝子シンボル一覧を抽出する。

    対象属性:
      RAP-DB Gene Symbol Synonym(s): コンマ区切り（%2C でエンコード）
      CGSNL Gene Symbol:             単一値
      Oryzabase Gene Symbol Synonym(s): コンマ区切り

    Returns:
        list[str]: 大文字小文字を保持したシンボル名のリスト
    """
    symbols = []
    symbol_keys = [
        "RAP-DB Gene Symbol Synonym(s)",
        "CGSNL Gene Symbol",
        "Oryzabase Gene Symbol Synonym(s)",
    ]
    for key in symbol_keys:
        val = attrs.get(key, "")
        if not val:
            continue
        # %2C はURLエンコードされたカンマ
        for sym in val.replace("%2C", ",").split(","):
            sym = sym.strip()
            if sym:
                symbols.append(sym)
    return symbols


def parse_gff(gff_path, feature_type="CDS"):
    """
    GFF3またはGTFファイルを読み込み、遺伝子IDごとにセグメント情報を返す。

    対応するattribute形式:
      - RAP-DB/IRGSP: mRNA行に Locus_id=遺伝子ID
      - 標準GFF3:     mRNA行に Parent=遺伝子ID
      - GTF:          各行に gene_id "遺伝子ID"

    複数トランスクリプトがある遺伝子は、指定フィーチャーの合計塩基数が
    最大のトランスクリプトを代表として選択する。

    Returns:
        gene_data : dict  gene_id -> {
            "chrom": str,
            "strand": "+" or "-",
            "segments": [(start, end), ...],  # 1-indexed inclusive、昇順ソート済み
            "transcript_id": str,             # 選択されたトランスクリプトID
        }
        name_to_id : dict  遺伝子シンボル(小文字) -> gene_id のマッピング
    """
    # Pass 1: transcript_id -> gene_id のマッピングと遺伝子名辞書を構築
    transcript_to_gene = {}
    name_to_id = {}  # 遺伝子シンボル(小文字) -> gene_id
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] not in ("mRNA", "transcript"):
                continue
            attrs = parse_attrs(cols[8])
            tid = attrs.get("ID", "")
            if not tid:
                continue
            # 形式ごとに遺伝子IDを取得（優先順位: Locus_id > Parent > gene_id）
            gid = attrs.get("Locus_id") or attrs.get("Parent") or attrs.get("gene_id") or ""
            if gid:
                transcript_to_gene[tid] = gid
                # 遺伝子シンボルを登録（同一遺伝子の重複登録は上書きで問題なし）
                for sym in _extract_gene_symbols(attrs):
                    name_to_id[sym.lower()] = gid

    # Pass 2: 指定フィーチャーのセグメントをトランスクリプトごとに収集
    transcript_segments = {}  # transcript_id -> {"chrom", "strand", "segments"}
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != feature_type:
                continue
            seqid, _, _, start, end, _, strand, _, attrs_str = cols
            attrs = parse_attrs(attrs_str)

            # このフィーチャーが属するトランスクリプトIDを取得
            # GFF3: Parent属性、GTF: transcript_id属性
            tid = attrs.get("Parent") or attrs.get("transcript_id") or ""
            if not tid:
                # transcript_idが直接書かれていてParentがない場合（GTFのCDS行等）
                tid = attrs.get("gene_id") or ""
            if not tid:
                continue

            if tid not in transcript_segments:
                transcript_segments[tid] = {
                    "chrom": seqid,
                    "strand": strand,
                    "segments": [],
                }
            transcript_segments[tid]["segments"].append((int(start), int(end)))

    # セグメントをゲノム座標の昇順にソート
    for tid, data in transcript_segments.items():
        data["segments"].sort(key=lambda x: x[0])

    # 遺伝子ごとに代表トランスクリプトを選択（指定フィーチャーの合計塩基数が最大のもの）
    gene_to_transcripts = defaultdict(list)
    for tid in transcript_segments:
        gid = transcript_to_gene.get(tid, tid)  # マッピングがなければtidをそのまま使用
        gene_to_transcripts[gid].append(tid)

    gene_data = {}
    for gid, tids in gene_to_transcripts.items():
        best_tid = max(
            tids,
            key=lambda t: sum(e - s + 1 for s, e in transcript_segments[t]["segments"])
        )
        data = transcript_segments[best_tid]
        gene_data[gid] = {
            "chrom": data["chrom"],
            "strand": data["strand"],
            "segments": data["segments"],
            "transcript_id": best_tid,
        }

    return gene_data, name_to_id


# ============================================================
#  座標変換ロジック
# ============================================================

def aa_to_cds(aa_pos):
    """アミノ酸座標 → CDS座標（コドン先頭）。aa N番目 → CDS (N-1)*3+1"""
    return (aa_pos - 1) * 3 + 1


def cds_to_genome(cds_pos, segments, strand):
    """
    CDS座標（イントロン除く）→ ゲノム座標に変換。

    Args:
        cds_pos : 1-indexed CDS座標
        segments: [(start, end), ...] ゲノム座標、昇順ソート済み
        strand  : "+" or "-"

    Returns:
        genome_pos (int) or None（範囲外の場合）

    + strand: セグメントを5'→3'（座標小→大）の順に累積
    - strand: セグメントを3'→5'（座標大→小）の順に累積
    """
    if strand == "+":
        ordered = segments
        cumulative = 0
        for seg_start, seg_end in ordered:
            seg_len = seg_end - seg_start + 1
            if cumulative + seg_len >= cds_pos:
                offset = cds_pos - cumulative - 1  # 0-indexed
                return seg_start + offset
            cumulative += seg_len

    else:  # "-"
        ordered = list(reversed(segments))
        cumulative = 0
        for seg_start, seg_end in ordered:
            seg_len = seg_end - seg_start + 1
            if cumulative + seg_len >= cds_pos:
                offset = cds_pos - cumulative - 1
                return seg_end - offset
            cumulative += seg_len

    return None  # 範囲外


def gdna_to_genome(gdna_pos, segments, strand):
    """
    ゲノムDNA連番（開始コドンA=1、イントロン含む）→ ゲノム座標に変換。

    + strand: ATGはsegments[0][0]、gdna_posが増えるにつれゲノム座標も増加
    - strand: ATGはsegments[-1][1]、gdna_posが増えるにつれゲノム座標は減少
    """
    if strand == "+":
        return segments[0][0] + (gdna_pos - 1)
    else:
        return segments[-1][1] - (gdna_pos - 1)


# ============================================================
#  メイン変換関数
# ============================================================

def convert(gene_id, coord_type, pos, gene_data):
    """
    座標変換のメイン関数。

    Returns:
        dict: gene_id, coord_type, input_pos, cds_pos, chrom, genome_pos, strand, status, message
    """
    result = {
        "gene_id": gene_id,
        "coord_type": coord_type,
        "input_pos": pos,
        "cds_pos": None,
        "chrom": None,
        "genome_pos": None,
        "strand": None,
        "status": "ERROR",
        "message": "",
    }

    if gene_id not in gene_data:
        result["message"] = f"Gene ID '{gene_id}' not found in GFF"
        return result

    data = gene_data[gene_id]
    segments = data["segments"]
    strand = data["strand"]
    result["chrom"] = data["chrom"]
    result["strand"] = strand

    if not segments:
        result["message"] = "No CDS/exon information in the GFF"
        return result

    if coord_type == "aa":
        cds_pos = aa_to_cds(pos)
        genome_pos = cds_to_genome(cds_pos, segments, strand)
        result["cds_pos"] = cds_pos

    elif coord_type == "cds":
        cds_pos = pos
        genome_pos = cds_to_genome(cds_pos, segments, strand)
        result["cds_pos"] = cds_pos

    elif coord_type == "gdna":
        genome_pos = gdna_to_genome(pos, segments, strand)
        # gdnaはイントロン含みのため厳密なCDS座標への逆算は困難

    else:
        result["message"] = f"Unsupported coord_type: '{coord_type}'. Specify one of 'aa', 'cds', 'gdna'"
        return result

    if genome_pos is None:
        total_cds = sum(e - s + 1 for s, e in segments)
        result["message"] = f"Position {pos} ({coord_type}) is out of range (total CDS length: {total_cds} bp)"
        return result

    result["genome_pos"] = genome_pos
    result["status"] = "OK"
    result["message"] = "Conversion successful"
    return result


# ============================================================
#  出力フォーマット
# ============================================================

def format_result(r):
    """変換結果を表示用文字列にフォーマット。"""
    if r["status"] == "ERROR":
        return f"[ERROR] {r['gene_id']} | {r['coord_type']}:{r['input_pos']} → {r['message']}"
    coord_str = f"{r['coord_type']}:{r['input_pos']}"
    if r["cds_pos"]:
        coord_str += f" (CDS:{r['cds_pos']})"
    return f"[OK] {r['gene_id']} | {coord_str} → {r['chrom']}:{r['genome_pos']} ({r['strand']} strand)"


def result_to_tsv(r, note=""):
    """変換結果をTSV行に変換。"""
    cols = [
        r["gene_id"],
        r["coord_type"],
        str(r["input_pos"]),
        str(r["cds_pos"]) if r["cds_pos"] is not None else "NA",
        str(r["chrom"]) if r["chrom"] else "NA",
        str(r["genome_pos"]) if r["genome_pos"] is not None else "NA",
        str(r["strand"]) if r["strand"] else "NA",
        r["status"],
        r["message"],
        note,
    ]
    return "\t".join(cols)


# ============================================================
#  バッチ処理
# ============================================================

def resolve_gene_id(query, gene_data, name_to_id):
    """
    ユーザー入力（遺伝子ID or 遺伝子名）を gene_data のキーに解決する。

    優先順位:
      1. gene_data に完全一致するID（例: Os06g0275000）
      2. 大文字小文字を無視した name_to_id への一致（例: Hd1, hd1, HD1）

    Returns:
        解決された gene_id (str)、見つからなければ query をそのまま返す
    """
    if query in gene_data:
        return query
    resolved = name_to_id.get(query.lower())
    if resolved:
        return resolved
    return query  # 見つからない場合はそのまま（convert()でERRORになる）


def process_batch(tsv_path, gene_data, name_to_id, output_path=None):
    """
    TSVまたはスペース区切りファイルをバッチ変換する。

    入力ファイル形式 (ヘッダー行必須、#始まりはコメント扱い):
        gene_id  coord_type  pos  [note]
    区切り文字はタブまたはスペース（ヘッダー行で自動判別）。
    gene_idは遺伝子ID（Os06g0275000）でも遺伝子名（Hd1）でも可。
    """
    results = []
    with open(tsv_path) as f:
        header = None
        delimiter = None  # ヘッダー行で自動判別
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            if header is None:
                # タブが含まれていればTSV、なければ空白区切りとみなす
                delimiter = "\t" if "\t" in line else None
                cols = line.split(delimiter) if delimiter else line.split()
                header = [c.lower() for c in cols]
                continue
            cols = line.split(delimiter) if delimiter else line.split()
            if len(cols) < 3:
                continue
            try:
                row = dict(zip(header, cols))
                gene_query = row.get("gene_id", cols[0])
                coord_type = row.get("coord_type", cols[1]).lower()
                pos = int(row.get("pos", cols[2]))
                note = row.get("note", cols[3] if len(cols) > 3 else "")
            except (ValueError, IndexError) as e:
                print(f"[WARN] Skipping (parse error): {line} -> {e}", file=sys.stderr)
                continue
            gene_id = resolve_gene_id(gene_query, gene_data, name_to_id)
            r = convert(gene_id, coord_type, pos, gene_data)
            # 遺伝子名で入力された場合、出力のgene_idを解決済みIDに統一
            if gene_query != gene_id:
                r["gene_id"] = f"{gene_id} ({gene_query})"
            results.append((r, note))

    header_cols = ["gene_id", "coord_type", "input_pos", "cds_pos",
                   "chrom", "genome_pos", "strand", "status", "message", "note"]
    out = open(output_path, "w") if output_path else sys.stdout
    try:
        out.write("\t".join(header_cols) + "\n")
        for r, note in results:
            out.write(result_to_tsv(r, note) + "\n")
    finally:
        if output_path:
            out.close()
            print(f"Results saved: {output_path}", file=sys.stderr)

    return results


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="アミノ酸座標・CDS座標 → ゲノム座標変換ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gff", required=True,
                        help="GFF3またはGTFアノテーションファイル（種ごとに指定）")
    parser.add_argument("--feature", default="CDS",
                        help="GFFで使用するフィーチャータイプ（デフォルト: CDS）")
    parser.add_argument("--gene", default=None,
                        help="遺伝子ID（例: Os01g0100100）※単一変換時に使用")
    parser.add_argument("--coord-type", default=None, choices=["aa", "cds", "gdna"],
                        help="座標の種類: aa（アミノ酸）/ cds（CDS）/ gdna（ゲノムDNA連番）")
    parser.add_argument("--pos", type=int, default=None,
                        help="変換する座標値（整数）")
    parser.add_argument("--batch", default=None,
                        help="バッチ処理用TSVファイル（gene_id, coord_type, pos, [note]）")
    parser.add_argument("--output", default=None,
                        help="バッチ処理結果の出力先TSVファイル")
    parser.add_argument("--list-genes", action="store_true",
                        help="GFFに含まれる遺伝子ID一覧を表示して終了")

    args = parser.parse_args()

    print(f"Loading GFF: {args.gff}", file=sys.stderr)
    gene_data, name_to_id = parse_gff(args.gff, feature_type=args.feature)
    print(f"  -> Loaded {len(gene_data)} gene(s)", file=sys.stderr)

    if args.list_genes:
        for gid in sorted(gene_data.keys()):
            d = gene_data[gid]
            n_seg = len(d["segments"])
            total_len = sum(e - s + 1 for s, e in d["segments"])
            print(f"{gid}\t{d['chrom']}\t{d['strand']}\t"
                  f"segments:{n_seg}\ttotal:{total_len}bp\ttranscript:{d['transcript_id']}")
        return

    if args.batch:
        process_batch(args.batch, gene_data, name_to_id, output_path=args.output)
        return

    if not args.gene or not args.coord_type or args.pos is None:
        parser.error("A single conversion requires --gene / --coord-type / --pos (use --batch for batch processing)")

    # 遺伝子名（Hd1等）でも遺伝子IDでも受け付ける
    gene_id = resolve_gene_id(args.gene, gene_data, name_to_id)
    if gene_id != args.gene:
        print(f"  Resolved gene name '{args.gene}' -> {gene_id}", file=sys.stderr)

    result = convert(gene_id, args.coord_type, args.pos, gene_data)
    print(format_result(result))

    if result["status"] == "OK":
        d = gene_data[gene_id]
        print(f"\n  Gene info:")
        print(f"    ID          : {gene_id}")
        if args.gene != gene_id:
            print(f"    input name  : {args.gene}")
        print(f"    transcript  : {d['transcript_id']}")
        print(f"    chromosome  : {result['chrom']}")
        print(f"    strand      : {result['strand']}")
        print(f"    segments    : {len(d['segments'])}")
        for i, (s, e) in enumerate(d["segments"]):
            print(f"    {args.feature} {i+1:2d}: {s:>10} – {e:>10} ({e-s+1:>5} bp)")
        if result["cds_pos"]:
            print(f"\n  Conversion detail:")
            print(f"    input       : {args.coord_type} = {args.pos}")
            print(f"    CDS position: {result['cds_pos']}")
            print(f"    genome pos  : {result['chrom']}:{result['genome_pos']}")


if __name__ == "__main__":
    main()

import pysam
# BAMより.baiが古い場合に htslib が出す "The index file is older than the
# data file" 警告を抑える（実害は無い。GUIログにエラーのように出るのを防ぐ）。
pysam.set_verbosity(0)
import argparse
import os

def is_bam_sorted(bam_path):
    """BAMファイルが座標順にソートされているかを判定します。"""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return bam.header.get('HD', {}).get('SO') == 'coordinate'

def sort_bam(input_bam, sorted_bam):
    """pysam.sort（内蔵htslib）を使ってBAMファイルを座標順にソートします。外部samtoolsは不要。"""
    print(f"[INFO] Sorting {input_bam}...")
    pysam.sort("-o", sorted_bam, input_bam)
    print(f"[INFO] Sorted BAM: {sorted_bam}")

def extend_regions(regions, flank=10000):
    """領域リストを前後flank塩基拡張した形式に変換。"""
    extended = []
    for region in regions:
        # rsplit on the LAST colon: tolerates a contig name that itself contains
        # a colon (some assemblies use that), and fails clearly (still exactly
        # 2 parts) instead of "too many values to unpack" on any other stray colon.
        chrom, pos = region.rsplit(":", 1)
        start, end = map(int, pos.split("-"))
        # Region strings passed to pysam's fetch(region=...) are 1-based
        # (samtools CLI convention), so the minimum valid start is 1, not 0 --
        # clamping to 0 produces an internally-invalid 0-based -1 and pysam
        # raises "start out of range (-1)". This only ever showed up for a
        # gene close to the start of its chromosome (start - flank < 0),
        # which real gene coordinates rarely are, but test/small-reference
        # data can easily hit.
        new_start = max(start - flank, 1)
        new_end = end + flank
        extended.append(f"{chrom}:{new_start}-{new_end}")
    return extended

def extract_bam_regions(input_bam, output_bam, regions, flank=10000):
    """
    BAMファイルから特定の領域（±flank塩基拡張）を抽出 → ソート → インデックス。
    """
    try:
        extended_regions = extend_regions(regions, flank)
        temp_unsorted_bam = output_bam + ".unsorted.bam"

        with pysam.AlignmentFile(input_bam, "rb") as bam_file:
            with pysam.AlignmentFile(temp_unsorted_bam, "wb", header=bam_file.header) as out_file:
                for region in extended_regions:
                    for read in bam_file.fetch(region=region):
                        out_file.write(read)

        # 抽出したBAMをソートして保存
        pysam.sort("-o", output_bam, temp_unsorted_bam)

        # インデックス作成
        pysam.index(output_bam)

        # 一時ファイル削除
        os.remove(temp_unsorted_bam)

    except Exception as e:
        print(f"[ERROR] An error occurred during region extraction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAMファイルから特定の領域（±10kb拡張）を抽出し、インデックスを作成します。")
    parser.add_argument("--input_bams", nargs='+', help="入力BAMファイルのパスのリスト。", required=True)
    parser.add_argument("--regions", nargs='+', help="抽出する領域のリスト（chr:start-end 形式で指定）。", required=True)
    parser.add_argument("--output_dir", help="出力BAMファイルを保存するディレクトリ。", required=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        for input_bam in args.input_bams:
            bam_basename = os.path.basename(input_bam)
            name_root = bam_basename.replace(".bam", "")
            output_bam = os.path.join(args.output_dir, f"{name_root}.bam")

            # 必要に応じて一時的にソートしたファイルを用意
            if not is_bam_sorted(input_bam):
                print(f"[INFO] {input_bam} is not sorted; running a temporary sort.")
                temp_sorted_bam = os.path.join(args.output_dir, f"{name_root}.tmp.sorted.bam")
                sort_bam(input_bam, temp_sorted_bam)
                sorted_bam_path = temp_sorted_bam
            else:
                print(f"[INFO] {input_bam} is already sorted.")
                sorted_bam_path = input_bam

            # 抽出＋ソート＋インデックス作成
            extract_bam_regions(sorted_bam_path, output_bam, args.regions, flank=10000)

            # 一時ソートファイルがある場合は削除
            if sorted_bam_path != input_bam and os.path.exists(sorted_bam_path):
                os.remove(sorted_bam_path)

    except Exception as e:
        print(f"[ERROR] An error occurred while running the script: {e}")
        exit(1)

# NGS-ACT

NGS-ACT is a desktop GUI tool for clustering-based allele determination in plant
(re)sequencing data. It groups accessions/varieties by their genotype at a
target gene and, once a reference panel has been clustered, classifies new
samples against that panel. Nothing in the pipeline is hardcoded to a
particular species -- it works from whatever reference FASTA / GFF3 / BAM
files you point it at.

It has two linked analyses, reachable from the same launcher:

- **Clustering Alleles** -- given BAM files for several accessions, cluster a
  gene's alleles (SNPs and large INDELs) into groups, based on read-alignment
  evidence rather than raw variant calls alone. Supports batch mode (many
  genes in one run) and an optional two-stage mode (a first pass on
  intron-only SNPs to separate broad population structure, then a second pass
  per group across the whole gene region).
- **Determining Alleles** -- given a *new* sample's BAM file(s), classify it
  into one of the groups a prior Clustering Alleles run already produced.
  Also supports batch mode against many genes' cluster results at once.

## Requirements

- macOS 11 (Big Sur) or later
- [bcftools](https://samtools.github.io/bcftools/) available somewhere on
  disk (its path is entered in the GUI, or passed via `--bcftools` on the
  command line) -- this is the only thing you need to install yourself;
  everything else (a full Python environment, all required packages) is
  bundled inside the app

## Installation and Execution

**Step 1. Download**

Go to the [Releases](../../releases) page and download the latest
`NGS-ACT_vX.X.X_forMac.zip`.

**Step 2. Unzip**

Double-click the downloaded ZIP file in Finder (or run `unzip
NGS-ACT_vX.X.X_forMac.zip` in a terminal). This creates `NGS-ACT.app`.

**Step 3. Run**

Double-click `NGS-ACT.app`.

Since this app isn't signed/notarized by Apple, macOS will very likely
refuse to open it the first time, with a message like "NGS-ACT can't be
opened because Apple cannot check it for malicious software" (or similar).
To open it anyway: **right-click** (or Control-click) `NGS-ACT.app` ->
**Open** -> confirm **Open** in the dialog that appears. You only need to do
this once -- after that, double-clicking works normally.

No separate Python installation, `pip install`, or terminal command is
needed -- the app bundles its own complete, self-contained Python
environment with every required package already installed at the exact
versions the bundled DEL/INS classifier models were trained with.

### After launch

Launching `NGS-ACT.app` opens a small launcher window with two buttons,
**Clustering Alleles** and **Determining Alleles**. Every field in either
screen is editable and is validated when you click Run. Leaving optional
fields blank is fine -- you'll just fill each field in by hand.

### About sleep prevention during long batch runs

This build automatically prevents the Mac from going to sleep for the
duration of a Clustering/Determining Alleles batch run (the display can
still turn off normally -- only idle *system* sleep is held off), using the
standard macOS `caffeinate -i` mechanism. This matters because macOS system
sleep freezes every process -- including any bcftools/Python subprocess this
app has launched -- mid-execution until the Mac wakes up again, and an
external drive can also drop/unmount during that time.

If you see `[INFO] Sleep prevention active for the duration of this batch
(caffeinate).` in the log, it's working. `caffeinate` ships with macOS, so
there's nothing extra to install.

## Files you need to prepare

NGS-ACT doesn't ship with any sequencing data -- these come from your own
project and need to be ready before you click Run.

### Both analyses

- **Reference FASTA** -- the same reference the BAMs were aligned to, indexed
  with `samtools faidx`:

  ```bash
  samtools faidx reference.fasta
  ```

  This creates `reference.fasta.fai`. It's required and NGS-ACT does not
  create it for you.
- **BAM file(s)** -- aligned, sorted, and indexed, one per sample:

  ```bash
  samtools index sample.bam
  ```

  This creates `sample.bam.bai`. It's required -- NGS-ACT reads each gene
  region directly out of the BAM index rather than scanning the whole file.
- **GFF3 annotation file** -- only needed for **two-stage clustering**
  (Clustering Alleles) or **allele matching** (either analysis); the field
  doesn't even appear on screen otherwise.

### Clustering Alleles only

- **Gene region list** -- a `.txt` file you write yourself, one gene per
  line (format below).
- **Alleles folder or list** (optional) -- only if you want the resulting
  clusters matched against previously reported alleles.

### Determining Alleles only

- **Bundle file(s)** (`{gene}_bundle.joblib`) -- produced automatically by a
  prior Clustering Alleles run; not something you write by hand.
- **New sample's BAM(s)** -- same requirements as above (aligned, sorted,
  indexed).
- **Alleles folder or list** (optional) -- only if you turn on **Check for
  new alleles**, to test the new sample against alleles that weren't yet
  known when the reference panel was clustered (a bundle already built with
  Clustering Alleles' own allele-matching on usually doesn't need this).

## Input files

A few inputs accept either a single value or a batch list. List files follow
one convention throughout: one entry per line, an optional leading
`<index><TAB>` (ignored if present), blank lines and `#`-comments skipped.

- **Gene region list** (Clustering Alleles, batch mode): `<index><TAB
  >GeneName:chr:start-end` per line.
- **BAM directory or list**: either a directory of `.bam` files, or a `.txt`
  file listing absolute BAM paths (same convention as above).
- **Alleles folder or list** (optional, for matching against previously
  reported alleles): either a directory of `{gene}_cds.tsv`-style files, or a
  `.txt` list of such files. A single TSV can also define multiple genes'
  alleles at once via its `gene_id` column.
- **Bundle (Determining Alleles)**: a single `{gene}_bundle.joblib` produced
  by a prior Clustering Alleles run, a folder of them, or a `.txt` list of
  their paths.

## Output

Each gene's results land in their own subfolder, split into:

```
{outdir}/{gene}/
  _results/            -- the deliverables: combined cluster labels + a
                           combined PCA-plot montage (and, for Clustering
                           Alleles, the Determining_model/ bundle used to
                           classify future samples against this gene)
  rawdata/              -- everything else: per-position calls, per-cluster
                           PCA plots, cluster assignment files, etc.
```

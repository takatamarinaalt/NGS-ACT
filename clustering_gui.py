#!/usr/bin/env python3
"""Clustering Alleles GUI frame: wires sc/step1/step1 (SNP) -> sc/step1/step2_v2 (INDEL).

Batch layout: instead of one gene/region per run, this reads a gene-region
list file (index-TAB-"GeneName:chr:start-end" per line, see gene_ragion.txt)
and an optional folder of per-gene haplotype TSVs (matched to gene names by
filename), then runs the full single-gene Clustering Alleles pipeline once
per gene, each into its own subfolder under the chosen output folder. One
gene failing does not abort the rest of the batch. A gene with no matching
haplotype file in the folder is still processed -- just without haplotype
matching for that gene.

The BAM directory field also accepts a .txt file listing absolute BAM paths
(index-TAB-path per line, see bam.txt), as an alternative to a plain
directory -- useful when BAMs are scattered across multiple locations.

Step1 exposes only Max-K; everything else in Step1 uses fixed defaults
(single-stage runs always use VCF mode "all"). Step2 is not user-configurable
at all: DEL/INS models and predict thresholds are fixed constants, everything
else uses the Step2 orchestrator's own defaults. DEL/INS sub-clustering always
runs across the full sample population in one pass, independent of any
SNP-side two-stage grouping.
"""

import glob
import joblib
import os
import re
import shutil
import sys
import threading
import tkinter as tk

from gui_common import (
    COLOR0, COLOR5, FieldBuilderMixin, FieldSpec, SubprocessRunnerMixin,
    build_pca_montage, collect_clustering_pca_panels,
    draw_rounded_button, start_caffeinate, stop_caffeinate, which_default,
    _pca_has_haplotype_match,
)

# This folder is self-contained: sc/ and model/ live INSIDE it (copies, not
# symlinks -- see CLAUDE.md's GUI folder family tree), so the whole folder
# can be zipped and sent to someone else without the rest of the repo.
# REPO_ROOT is therefore this folder itself, not its parent.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
STEP1_SCRIPT = os.path.join(REPO_ROOT, "sc", "step1", "step1", "_run_pipeline.py")
STEP2_SCRIPT = os.path.join(REPO_ROOT, "sc", "step1", "step2_newfea2", "_run_pipeline.py")
MEAN_DEPTH_SCRIPT = os.path.join(REPO_ROOT, "sc", "step1", "step2_newfea2", "estimate_mean_depth.py")

# --- Pre-filled field defaults. Purely a convenience starting point shown in
# the entry boxes -- every field stays editable and gets re-validated
# (existence, format, etc.) on Run. Left blank here (no machine-specific
# paths); set these to your own environment's paths if you'd like the fields
# pre-filled when the GUI opens, e.g.:
#   DEFAULT_REF = "/path/to/reference_genome.fasta"
# DEFAULT_BCFTOOLS left blank falls back to whatever "bcftools" resolves to
# on your PATH (see which_default() below).
DEFAULT_OUTDIR = ""
DEFAULT_BCFTOOLS = ""
DEFAULT_REF = ""
DEFAULT_GFF = ""
DEFAULT_BAM_DIR = ""
DEFAULT_GENE_REGION_FILE = ""
DEFAULT_HAPLOTYPE_DIR = ""

# --- Step1 fixed defaults (not exposed in the GUI) ---
S1_THREADS = 8
S1_MIN_DP = 10
S1_PCA_COMPONENTS = 2
S1_PLOT_PC = [1, 2]
S1_MIN_CLUSTER_SIZE = 3
# Elbow-method upper bound (was a user-editable "Max-K" field). Both
# _run_pipeline.py orchestrators already cap this down to (n_samples - 1)
# whenever the sample count is too small for it (see their own
# _capped_max_k()), so a fixed value here is safe for any sample count --
# it only ever gets reduced automatically, never causes a crash.
S1_MAX_K = 10

# --- Step2 fixed defaults / constants (not exposed in the GUI) ---
MODEL_DIR = os.path.join(REPO_ROOT, "model")
MODEL_DEL_PATH = os.path.join(MODEL_DIR, "DEL_3class_best_15.joblib")
# CLEAN_END_RATIO/INSERT_SIZE_DIFF を含む新INS特徴量セット（INS_features_v2.py）で
# 学習したモデル（MQ0側 = 全MQ閾値を0にして計算した特徴量で学習＝実質MQフィルタなし）。
MODEL_INS_PATH = os.path.join(MODEL_DIR, "INS_3class_best_15.joblib")
# 上のモデルに合わせて、INS特徴量計算のMQ閾値も揃える（MQ>=この値のリードのみ対象）。
INS_MQ = 0
PREDICT_THRESHOLD_DEL = 0.99
PREDICT_THRESHOLD_INS = 0.98

# 0 = use all CPU cores. Mean-depth estimation is one independent, mostly I/O-bound
# call per BAM file, so it parallelizes very well (unlike DEL/INS prediction, which
# is CPU-heavy per position and already limited to S2_JOBS to avoid oversubscription).
MEAN_DEPTH_JOBS = 0

S2_PREDICT_STEP = 1
S2_JOBS = 2
S2_FILL = "N"
S2_INS_MODE = "split"
S2_N_FILL = "zero"
S2_INDEL_WEIGHT = 1.0
S2_SUB_N_INIT = 10
S2_COMPONENTS = 2
S2_PLOT_PC = [1, 2]
S2_CLUSTER_KEY = "sub"

_REGION_RE = re.compile(r"^\S+:\d+-\d+$")


def parse_gene_region_file(path):
    """Parse a gene-region list file (see gene_ragion.txt): one gene per line,
    "<index><TAB>GeneName:chr:start-end" (the leading index column is optional
    and discarded if present). Blank lines and lines starting with '#' are
    skipped. Returns (entries, warnings) where entries is an ordered list of
    (gene_name, region_str) tuples and warnings describes any skipped lines
    (malformed lines are skipped, not fatal, so one bad line doesn't block
    every other gene in the file)."""
    entries = []
    warnings = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                line = line.split("\t", 1)[1].strip()
            if ":" not in line:
                warnings.append(f"line {lineno}: no ':' found, skipping: '{raw_line.strip()}'")
                continue
            gene_name, region_str = line.split(":", 1)
            gene_name = gene_name.strip()
            region_str = region_str.strip()
            if not gene_name:
                warnings.append(f"line {lineno}: empty gene name, skipping: '{raw_line.strip()}'")
                continue
            if not _REGION_RE.match(region_str):
                warnings.append(
                    f"line {lineno}: invalid region format for gene '{gene_name}' "
                    f"(expected chr:start-end), skipping: '{raw_line.strip()}'")
                continue
            entries.append((gene_name, region_str))
    return entries, warnings


def _tsv_contains_gene_id(path, gene_lower):
    """True if any data row's gene_id column (case-insensitive, exact match)
    equals gene_lower. Lets one combined TSV -- multiple genes' allele
    definitions in a single file, distinguished by the gene_id column, same
    cds.tsv format 10_haplotype_match.py already reads -- be discovered for
    every gene it actually defines, not just a per-gene file whose name
    happens to match. Mirrors 10_haplotype_match.py's own parse_haplotype_tsv()
    header handling (lower-cased header, tab-separated). 10_haplotype_match.py
    itself needs no change for this: it already resolves each row's genome
    position from that row's own gene_id and just harmlessly fails to match
    (a WARNING, not an error) any row whose position isn't in the current
    gene's step4 TSV -- which is every row belonging to a different gene."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = None
            gene_id_idx = None
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if header is None:
                    header = [c.lower().strip() for c in cols]
                    if "gene_id" not in header:
                        return False
                    gene_id_idx = header.index("gene_id")
                    continue
                if gene_id_idx < len(cols) and cols[gene_id_idx].strip().lower() == gene_lower:
                    return True
    except OSError:
        return False
    return False


def match_haplotype_file(gene_name, haplotype_input):
    """Find the haplotype TSV that defines gene_name's alleles, from either a
    folder of TSVs or a .txt list of TSV paths -- same "folder or .txt list"
    duality as the BAM field; reuses parse_bam_list_file()'s parsing since
    the list-file convention (optional leading "<index><TAB>", blank/'#'
    lines skipped) is identical, just applied to TSV paths instead of BAM
    paths here. A candidate file matches if EITHER its filename (stem) is an
    exact case-insensitive match, or starts with gene_name followed by a
    non-alphanumeric separator (e.g. "Hd1_haplotype.tsv", "Hd1-x.tsv" --
    deliberately not a plain substring/prefix match, since gene lists like
    gene_ragion.txt contain names such as Hd1/Hd16/Hd17/Hd18 where naive
    prefix matching would wrongly match "Hd1" against "Hd16.tsv") OR its
    gene_id column contains an exact case-insensitive match for gene_name
    (see _tsv_contains_gene_id() -- this is what lets one combined TSV with
    multiple genes' alleles in it be found for each gene it defines, not just
    per-gene files). Returns the matched path, or None if there's no match or
    the match is ambiguous (multiple candidates -- logged by the caller,
    treated as "no match" rather than guessing wrong). A gene with no match
    is still processed normally by the caller, just without haplotype
    matching for that gene."""
    if not haplotype_input:
        return None, []
    if haplotype_input.endswith(".txt") and os.path.isfile(haplotype_input):
        all_paths = [p for p in parse_bam_list_file(haplotype_input) if p.lower().endswith(".tsv")]
    elif os.path.isdir(haplotype_input):
        all_paths = [os.path.join(haplotype_input, fname)
                     for fname in sorted(os.listdir(haplotype_input))
                     if fname.lower().endswith(".tsv")]
    else:
        return None, []

    gene_lower = gene_name.lower()
    candidates = []
    for p in all_paths:
        stem_lower = os.path.splitext(os.path.basename(p))[0].lower()
        name_match = stem_lower == gene_lower or (
            stem_lower.startswith(gene_lower) and len(stem_lower) > len(gene_lower)
            and not stem_lower[len(gene_lower)].isalnum())
        if name_match or _tsv_contains_gene_id(p, gene_lower):
            candidates.append(p)
    if not candidates:
        return None, []
    if len(candidates) > 1:
        return None, [os.path.basename(c) for c in candidates]
    return candidates[0], [os.path.basename(candidates[0])]


def parse_bam_list_file(path):
    """Parse a BAM-list file (see bam.txt): one absolute BAM path per line,
    "<index><TAB>path" (the leading index column is optional and discarded if
    present, same convention as parse_gene_region_file()). Blank lines and
    lines starting with '#' are skipped. Returns a plain list of path strings
    -- no existence/extension validation here, since expand_bam_inputs() /
    collect_bam_files() on the sc/ side already warn-and-skip bad entries
    gracefully rather than crashing."""
    paths = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                line = line.split("\t", 1)[1].strip()
            if line:
                paths.append(line)
    return paths


def sample_basename(bam_path):
    """Derive a sample's display name from its BAM path, stripping the same
    suffixes sc/'s own sample_basename() (estimate_mean_depth.py,
    1_predict_combined.py) strips, so the result matches what those scripts
    treat as "the sample name" for the same BAM. Case is preserved here (only
    sc/'s internal processing upper-cases it for matching purposes)."""
    name = os.path.basename(bam_path)
    for suf in (".rg.bam", ".sort.bam", ".bam"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def resolve_bam_paths(bam_input):
    """Expand the BAM field (a directory, or a .txt list of absolute paths)
    into a sorted list of real .bam file paths, skipping macOS AppleDouble
    sidecar files (e.g. "._sample.bam") -- mirrors sc/'s own
    expand_bam_inputs(), used here only to build the original-casing map
    below (the actual --bam-dir/--bams argument passed to sc/ scripts is
    still built separately in build_step1_cmd()/build_mean_depth_cmd())."""
    if bam_input.endswith(".txt") and os.path.isfile(bam_input):
        candidates = parse_bam_list_file(bam_input)
    else:
        candidates = glob.glob(os.path.join(bam_input, "*.bam"))
    return sorted(p for p in candidates if not os.path.basename(p).startswith("._"))


def build_original_case_map(bam_paths):
    """{UPPER(sample_name): sample_name_as_written} from the resolved BAM
    paths -- used to restore each sample's original casing (as given in the
    BAM dir/list) in the final cluster/label output files, which sc/'s own
    processing upper-cases internally for case-insensitive matching."""
    return {sample_basename(p).upper(): sample_basename(p) for p in bam_paths}


def restore_case_in_cluster_file(path, case_map):
    """Rewrite a "# cluster ..." formatted label file (step7_*_cluster.txt,
    sub.txt, top.txt, nested.txt, *_combined.txt -- all the same format: a
    "# cluster ..." header line, then one sample name per line, blank lines
    separating groups) in place, replacing each sample-name line with its
    original casing from case_map (case-insensitive lookup). A name with no
    match in case_map (shouldn't normally happen -- every sample that made it
    into this file came from the same BAM set the map was built from) is left
    untouched rather than dropped, so a mismatch never loses data."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    changed = False
    out_lines = []
    for raw_line in lines:
        stripped = raw_line.rstrip("\n")
        if not stripped or stripped.startswith("#"):
            out_lines.append(raw_line)
            continue
        restored = case_map.get(stripped.strip().upper())
        if restored is not None and restored != stripped:
            out_lines.append(raw_line.replace(stripped, restored, 1))
            changed = True
        else:
            out_lines.append(raw_line)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)


class ClusteringOptionsFrame(tk.Frame):
    """First screen shown after picking "Clustering Alleles": just the two
    method checkboxes (two-stage clustering / match against reported
    alleles), decided before any output-folder/BAM/GFF fields are shown.
    The next screen (ClusteringAllelesFrame) is built based on this choice --
    e.g. the GFF field is only needed (and only shown) when at least one of
    these is on, since GFF is used for intron extraction (two-stage) and for
    CDS/AA coordinate conversion (allele matching), not for anything else."""

    def __init__(self, master, on_back=None, on_next=None):
        super().__init__(master, bg=COLOR0)
        self.on_back = on_back
        self.on_next = on_next
        self.create_widgets()

    def create_widgets(self):
        top = tk.Frame(self, bg=COLOR0)
        top.pack(fill="x", padx=14, pady=(10, 0))
        if self.on_back:
            back_canvas = tk.Canvas(top, width=90, height=32, bg=COLOR0, highlightthickness=0)
            back_canvas.pack(side="left")
            draw_rounded_button(back_canvas, 0, 0, 90, 32, 12, "#9FB8AD", "Back", self.on_back)
        tk.Label(top, text="Clustering Alleles", font=("Helvetica Neue", 20, "bold"),
                fg="#264653", bg=COLOR0).pack(side="left", padx=16)

        container = tk.Frame(self, bg=COLOR0)
        container.place(relx=0.5, rely=0.42, anchor="center")

        tk.Label(container, text="Clustering method", font=("Arial", 20, "bold"),
                 bg=COLOR0, fg="#264653").pack(pady=(0, 16))

        self.two_stage_var = tk.BooleanVar(value=False)
        tk.Checkbutton(container, text="Two-stage clustering", variable=self.two_stage_var,
                       bg=COLOR0, fg="#264653", font=("Arial", 18),
                       activebackground=COLOR0).pack(anchor="w", pady=6)

        self.use_alleles_match_var = tk.BooleanVar(value=False)
        tk.Checkbutton(container, text="Match against reported alleles", variable=self.use_alleles_match_var,
                       bg=COLOR0, fg="#264653", font=("Arial", 18),
                       activebackground=COLOR0).pack(anchor="w", pady=6)

        next_canvas = tk.Canvas(container, width=140, height=50, bg=COLOR0, highlightthickness=0)
        next_canvas.pack(pady=(24, 0))
        draw_rounded_button(next_canvas, 0, 0, 140, 50, 18, COLOR5, "Next", self._on_next_clicked)

    def _on_next_clicked(self):
        if self.on_next:
            self.on_next(bool(self.two_stage_var.get()), bool(self.use_alleles_match_var.get()))


class ClusteringAllelesFrame(tk.Frame, FieldBuilderMixin, SubprocessRunnerMixin):
    """Clusters gene alleles across multiple varieties' BAMs into groups.

    two_stage / use_alleles_match are decided up front on ClusteringOptionsFrame
    (not user-editable here) -- they control which fields this screen shows:
    the GFF field only appears when at least one of them is on (needed for
    intron extraction and/or CDS/AA coordinate conversion), and the Alleles
    folder field only appears when use_alleles_match is on."""

    def __init__(self, master, on_back=None, two_stage=False, use_alleles_match=False):
        super().__init__(master, bg=COLOR0)
        self.on_back = on_back
        self.two_stage = two_stage
        self.use_alleles_match = use_alleles_match
        self.widgets = {}
        self.is_running = False
        self.create_widgets()

    # ---------------- widget construction ----------------

    def create_widgets(self):
        top = tk.Frame(self, bg=COLOR0)
        top.pack(fill="x", padx=14, pady=(10, 0))
        if self.on_back:
            back_canvas = tk.Canvas(top, width=90, height=32, bg=COLOR0, highlightthickness=0)
            back_canvas.pack(side="left")
            draw_rounded_button(back_canvas, 0, 0, 90, 32, 12, "#9FB8AD", "Back", self.on_back)
        tk.Label(top, text="Clustering Alleles", font=("Helvetica Neue", 20, "bold"),
                fg="#264653", bg=COLOR0).pack(side="left", padx=16)
        summary = (f"Two-stage: {'On' if self.two_stage else 'Off'}"
                   f"  |  Reported-allele matching: {'On' if self.use_alleles_match else 'Off'}")
        tk.Label(top, text=summary, font=("Arial", 11), fg="#5A7A70", bg=COLOR0).pack(side="left", padx=16)

        outer, inner = self.make_scrollable_frame(self)
        outer.pack(fill="both", expand=True, padx=14, pady=8)
        self.build_fields(inner)

        run_frame = tk.Frame(self, bg=COLOR0)
        run_frame.pack(fill="x", padx=20, pady=(4, 4))

        self.dry_run_var = tk.BooleanVar(value=False)
        tk.Checkbutton(run_frame, text="Dry run (build commands only, don't execute)",
                       variable=self.dry_run_var, bg=COLOR0, fg="#264653",
                       font=("Arial", 12), activebackground=COLOR0).pack(side="left", padx=5)
        self.widgets["dry_run"] = self.dry_run_var

        self.status_label = tk.Label(run_frame, text="Idle", bg=COLOR0, fg="#264653", font=("Arial", 12))
        self.status_label.pack(side="left", padx=15)

        run_canvas = tk.Canvas(run_frame, width=140, height=50, bg=COLOR0, highlightthickness=0)
        run_canvas.pack(side="right")
        draw_rounded_button(run_canvas, 0, 0, 140, 50, 18, COLOR5, "Run", self.run_script)

        log_frame = tk.Frame(self, bg=COLOR0)
        log_frame.pack(fill="both", expand=False, padx=20, pady=(0, 12))
        tk.Label(log_frame, text="Log", font=("Arial", 13, "bold"), bg=COLOR0, fg="#264653").pack(anchor="w")
        from tkinter import scrolledtext
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=14, bg="#1E1E1E", fg="#D4D4D4",
            insertbackground="#D4D4D4", font=("Menlo", 11), state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def build_fields(self, parent):
        row = 0
        needs_gff = self.two_stage or self.use_alleles_match

        self.section_label(parent, row, "Output"); row += 1
        for spec in [
            FieldSpec("outdir", "Output folder", "dir", DEFAULT_OUTDIR),
            FieldSpec("bcftools", "bcftools path", "file", DEFAULT_BCFTOOLS or which_default("bcftools")),
        ]:
            self.build_field(parent, row, spec); row += 1

        self.section_label(parent, row, "Reference & Gene regions"); row += 1
        ref_specs = [FieldSpec("ref", "Reference FASTA", "file", DEFAULT_REF)]
        if needs_gff:
            ref_specs.append(FieldSpec("gff", "GFF file", "file", DEFAULT_GFF))
        ref_specs.append(FieldSpec("gene_region_file", "Gene region list", "file", DEFAULT_GENE_REGION_FILE))
        for spec in ref_specs:
            self.build_field(parent, row, spec); row += 1

        self.section_label(parent, row, "BAM / Alleles"); row += 1
        bam_specs = [FieldSpec("bam_dir", "BAM dir or list", "dir", DEFAULT_BAM_DIR)]
        if self.use_alleles_match:
            bam_specs.append(FieldSpec("haplotype_dir", "Alleles folder or list", "dir", DEFAULT_HAPLOTYPE_DIR))
        for spec in bam_specs:
            self.build_field(parent, row, spec); row += 1

    # ---------------- input collection & validation ----------------

    def collect_inputs(self):
        errors = []
        g = lambda k: self.widgets[k].get()

        inputs = {}
        # Decided up front on ClusteringOptionsFrame, not editable on this screen.
        inputs["two_stage"] = self.two_stage
        inputs["outdir"] = g("outdir").strip()
        inputs["bcftools"] = g("bcftools").strip()
        inputs["ref"] = g("ref").strip()
        # GFF / Alleles folder fields only exist on this screen when needed
        # (see build_fields()) -- default to "" when the field wasn't shown.
        inputs["gff"] = g("gff").strip() if "gff" in self.widgets else ""
        inputs["gene_region_file"] = g("gene_region_file").strip()
        inputs["bam_dir"] = g("bam_dir").strip()
        inputs["haplotype_dir"] = g("haplotype_dir").strip() if "haplotype_dir" in self.widgets else ""
        # VCF mode field removed from the GUI (only mattered for single-stage
        # clustering, since --two-stage always ignores it anyway). Single-stage
        # runs now always use "all", matching the field's old default value.
        inputs["vcf_mode"] = "all"
        inputs["dry_run"] = bool(g("dry_run"))

        return inputs, errors

    def validate_inputs(self, inputs):
        errors = []
        if not inputs["outdir"]:
            errors.append("Output folder is required.")

        if not inputs["bam_dir"]:
            errors.append("BAM directory or list (txt) is required.")
        elif not (os.path.isdir(inputs["bam_dir"]) or
                  (os.path.isfile(inputs["bam_dir"]) and inputs["bam_dir"].endswith(".txt"))):
            errors.append(f"BAM directory or .txt list not found: {inputs['bam_dir']}")

        if not inputs["ref"]:
            errors.append("Reference FASTA is required.")
        elif not os.path.isfile(inputs["ref"]):
            errors.append(f"Reference FASTA not found: {inputs['ref']}")
        elif not os.path.exists(inputs["ref"] + ".fai"):
            errors.append(f"Reference FASTA .fai index not found: {inputs['ref']}.fai")

        if not inputs["gene_region_file"]:
            errors.append("Gene region list (txt) is required.")
        elif not os.path.isfile(inputs["gene_region_file"]):
            errors.append(f"Gene region list file not found: {inputs['gene_region_file']}")

        if inputs["haplotype_dir"] and not (os.path.isdir(inputs["haplotype_dir"]) or
                (os.path.isfile(inputs["haplotype_dir"]) and inputs["haplotype_dir"].endswith(".txt"))):
            errors.append(f"Alleles folder or .txt list not found: {inputs['haplotype_dir']}")

        if inputs["two_stage"] and not inputs["gff"]:
            errors.append("GFF file is required for two-stage clustering.")

        if self.use_alleles_match and not inputs["gff"]:
            errors.append("GFF file is required for matching against reported alleles.")

        for label, path in [("DEL model", MODEL_DEL_PATH), ("INS model", MODEL_INS_PATH)]:
            if not os.path.isfile(path):
                errors.append(f"{label} not found at fixed path: {path}")

        return errors

    def log_soft_warnings(self, inputs):
        if inputs["bcftools"] and not os.path.isfile(inputs["bcftools"]):
            self.log(f"[WARN] bcftools path not found (may be valid on a remote/mounted environment): {inputs['bcftools']}")
        elif not inputs["bcftools"]:
            self.log("[WARN] bcftools path not set; the orchestrator's own default will be used.")
        if inputs["gff"] and not os.path.isfile(inputs["gff"]):
            self.log(f"[WARN] GFF file not found: {inputs['gff']}")

    # ---------------- command construction ----------------

    def _is_bam_list_file(self, bam_input):
        """True if the BAM field holds a .txt list of paths rather than a directory."""
        return bam_input.endswith(".txt") and os.path.isfile(bam_input)

    def build_mean_depth_cmd(self, inputs, mean_depth_tsv):
        """Must run against the original full-length BAM directory, before Step1's
        region extraction -- estimate_mean_depth() divides total mapped reads by the
        genome length declared in the BAM header, which stays full-size even after
        region extraction, so computing it on an already-trimmed BAM would silently
        produce a near-zero depth and corrupt every DEPTH_RATIO-derived feature.
        --bam accepts nargs="+", and expand_bam_inputs() on that side already handles
        a mix of directories and individual .bam file paths, so a parsed BAM-list
        file's paths can be passed through exactly like a single directory would be."""
        if self._is_bam_list_file(inputs["bam_dir"]):
            bam_args = parse_bam_list_file(inputs["bam_dir"])
        else:
            bam_args = [inputs["bam_dir"]]
        return [
            sys.executable, MEAN_DEPTH_SCRIPT,
            "--bam", *bam_args,
            "-o", mean_depth_tsv,
            "--jobs", str(MEAN_DEPTH_JOBS),
        ]

    def build_step1_cmd(self, inputs, step1_outdir, case_map_tsv):
        cmd = [
            sys.executable, STEP1_SCRIPT,
            "--outdir", step1_outdir,
            "--prefix", inputs["prefix"],
            "--case-map", case_map_tsv,
        ]
        # --bams is a full alternative to --bam-dir in _run_pipeline.py, accepting
        # individual file paths (nargs="+") -- used when the BAM field holds a .txt
        # list of absolute paths instead of a single directory.
        if self._is_bam_list_file(inputs["bam_dir"]):
            cmd += ["--bams", *parse_bam_list_file(inputs["bam_dir"])]
        else:
            cmd += ["--bam-dir", inputs["bam_dir"]]
        cmd += [
            "--ref", inputs["ref"],
            "--regions", *inputs["regions"],
            "--threads", str(S1_THREADS),
            "--min-dp", str(S1_MIN_DP),
            "--max-k", str(S1_MAX_K),
            "--pca-components", str(S1_PCA_COMPONENTS),
            "--plot-pc", *[str(p) for p in S1_PLOT_PC],
        ]
        if inputs["bcftools"]:
            cmd += ["--bcftools", inputs["bcftools"]]
        if inputs["gff"]:
            cmd += ["--gff", inputs["gff"]]
        if inputs["gene"]:
            cmd += ["--gene", inputs["gene"]]
        if inputs["haplotype_tsv"]:
            cmd += ["--haplotype-tsv", inputs["haplotype_tsv"]]
        if inputs["two_stage"]:
            cmd.append("--two-stage")
            cmd += ["--min-cluster-size", str(S1_MIN_CLUSTER_SIZE)]
        else:
            cmd += ["--vcf-mode", inputs["vcf_mode"]]
            # --end-step defaults to 9 in _run_pipeline.py, which means step10
            # (haplotype matching) never runs in single-stage mode unless
            # explicitly extended to 10 -- without this, setting Haplotype TSV
            # silently did nothing when Two-stage clustering was off.
            if inputs["haplotype_tsv"]:
                cmd += ["--end-step", "10"]
        if inputs["dry_run"]:
            cmd.append("--dry-run")
        return cmd

    def build_step2_cmd(self, inputs, step2_outdir, hclust_model, step2_bam_dir, mean_depth_tsv,
                        case_map_tsv, snp_cluster_txt):
        region = inputs["regions"][0] if inputs["regions"] else None
        cmd = [
            sys.executable, STEP2_SCRIPT,
            "--outdir", step2_outdir,
            "--prefix", inputs["prefix"],
            "--bam-dir", step2_bam_dir,
            "--mean-depth-tsv", mean_depth_tsv,
            "--case-map", case_map_tsv,
            "--hclust-model", hclust_model,
            "--snp-cluster-txt", snp_cluster_txt,
            "--model-del", MODEL_DEL_PATH,
            "--model-ins", MODEL_INS_PATH,
            "--ins-mq", str(INS_MQ),
            "--predict-step", str(S2_PREDICT_STEP),
            "--predict-threshold-del", str(PREDICT_THRESHOLD_DEL),
            "--predict-threshold-ins", str(PREDICT_THRESHOLD_INS),
            "--jobs", str(S2_JOBS),
            "--fill", S2_FILL,
            "--ins-mode", S2_INS_MODE,
            "--n-fill", S2_N_FILL,
            "--max-k", str(S1_MAX_K),
            "--indel-weight", str(S2_INDEL_WEIGHT),
            "--sub-n-init", str(S2_SUB_N_INIT),
            "--components", str(S2_COMPONENTS),
            "--cluster-key", S2_CLUSTER_KEY,
            "--plot-pc", *[str(p) for p in S2_PLOT_PC],
        ]
        if region:
            cmd += ["--region", region]
        if inputs["haplotype_tsv"]:
            cmd += ["--haplotype-tsv", inputs["haplotype_tsv"]]
        if inputs["gff"]:
            cmd += ["--gff", inputs["gff"]]
        if inputs["dry_run"]:
            cmd.append("--dry-run")
        return cmd

    def locate_step1_joblib(self, step1_outdir, prefix, two_stage):
        """Path to Step1's output joblib, which feeds Step2's --hclust-model."""
        if two_stage:
            return os.path.join(step1_outdir, "combined", f"step7_{prefix}_model.joblib")
        return os.path.join(step1_outdir, f"step7_{prefix}_model.joblib")

    # ---------------- execution ----------------

    def run_script(self):
        if self.is_running:
            from tkinter import messagebox
            messagebox.showinfo("Running", "An analysis is already running. Please wait for it to finish.")
            return

        inputs, parse_errors = self.collect_inputs()
        errors = parse_errors + self.validate_inputs(inputs)
        if errors:
            from tkinter import messagebox
            messagebox.showwarning("Input error", "\n".join(errors))
            return

        self.log_soft_warnings(inputs)
        self.is_running = True
        threading.Thread(target=self.run_main_work, args=(inputs,), daemon=True).start()

    def run_main_work(self, inputs):
        """Batch orchestrator: parses the gene-region list file, resolves each
        gene's haplotype TSV (if a folder was given -- a gene with no match is
        still processed, just without haplotype matching), computes mean depth
        once (shared across every gene -- it only depends on --bam-dir/--bams,
        not the gene region), then runs the full single-gene pipeline once per
        gene. One gene failing is logged and does not stop the rest of the batch."""
        caffeinate_proc = start_caffeinate()
        try:
            self.log("========== Batch run started ==========")
            if caffeinate_proc is not None:
                self.log("[INFO] Sleep prevention active for the duration of this batch (caffeinate).")
            self.set_status("Running...")

            outdir = inputs["outdir"]
            os.makedirs(outdir, exist_ok=True)

            entries, parse_warnings = parse_gene_region_file(inputs["gene_region_file"])
            for w in parse_warnings:
                self.log(f"[WARN] gene region file: {w}")
            if not entries:
                self.log("[ERROR] No valid gene entries found in the gene region list file.")
                self.show_error("Error", "No valid gene entries found in the gene region list file. See the log.")
                return
            self.log(f"[INFO] Parsed {len(entries)} gene(s) from {inputs['gene_region_file']}")

            # Resolve haplotype TSV matches up front so mismatches are visible
            # immediately, before any gene starts processing. A gene with no
            # match is still processed normally, just without haplotype matching.
            haplotype_by_gene = {}
            if inputs["haplotype_dir"]:
                self.log(f"----- Matching haplotype TSVs in {inputs['haplotype_dir']} -----")
                for gene_name, _ in entries:
                    matched_path, candidates = match_haplotype_file(gene_name, inputs["haplotype_dir"])
                    haplotype_by_gene[gene_name] = matched_path
                    if matched_path:
                        self.log(f"  {gene_name} -> {os.path.basename(matched_path)}")
                    elif candidates:
                        self.log(f"  {gene_name} -> [WARN] ambiguous match ({', '.join(candidates)}), skipping")
                    else:
                        self.log(f"  {gene_name} -> (no Alleles TSV found -- will process without one)")

            # sc/'s own processing upper-cases every sample name internally (for
            # case-insensitive matching across VCF/BAM/wide-table joins); this map
            # lets run_one_gene() restore each sample's original casing (as written
            # in the BAM dir/list) in the final cluster/label output files. Built
            # once here since the BAM field is shared across the whole batch.
            case_map = build_original_case_map(resolve_bam_paths(inputs["bam_dir"]))
            # Also written out as a TSV and passed to Step1/Step2 (--case-map) so
            # 9_plot.py/10_haplotype_match.py can draw each PCA plot's per-sample
            # point labels in the same original casing, not just the text output
            # files -- these scripts run inside the Step1/Step2 subprocess itself,
            # so this is the only point in the pipeline where that's possible.
            case_map_tsv = os.path.join(outdir, "case_map.tsv")
            with open(case_map_tsv, "w", encoding="utf-8") as f:
                for upper, original in case_map.items():
                    f.write(f"{upper}\t{original}\n")

            # Mean depth only depends on --bam-dir/--bams (not the gene region), and
            # the BAM field is shared across the whole batch, so compute it once
            # here instead of once per gene -- see build_mean_depth_cmd() for why
            # it must run against the ORIGINAL full-length BAMs.
            mean_depth_tsv = os.path.join(outdir, "mean_depth.tsv")
            self.log("----- Estimating mean depth (full-length BAMs, shared across all genes) -----")
            mean_depth_cmd = self.build_mean_depth_cmd(inputs, mean_depth_tsv)
            if inputs["dry_run"]:
                self.log("[DryRun] $ " + " ".join(str(c) for c in mean_depth_cmd))
            else:
                rc = self.stream_subprocess(mean_depth_cmd, "[MeanDepth]")
                if rc != 0:
                    self.show_error("Mean depth estimation failed", f"Exited with code {rc}. See the log for details.")
                    return

            results = []  # (gene_name, success, message)
            for gene_name, region_str in entries:
                self.log(f"\n========== Gene: {gene_name} ({region_str}) ==========")
                gene_outdir = os.path.join(outdir, re.sub(r"[/\\]", "_", gene_name))
                try:
                    success, message = self.run_one_gene(
                        inputs, gene_name, region_str, haplotype_by_gene.get(gene_name),
                        gene_outdir, mean_depth_tsv, case_map, case_map_tsv)
                except Exception:
                    import traceback
                    success = False
                    message = "Unexpected error:\n" + traceback.format_exc()
                self.log(message)
                results.append((gene_name, success, message))

            # mean_depth.tsv and case_map.tsv are both shared across every gene's
            # Step1/Step2 calls in the loop above -- only safe to delete once the
            # whole batch is done with them, never per-gene. Neither is a
            # deliverable itself (Step1/Step2 already consumed them).
            for tmp_path in (mean_depth_tsv, case_map_tsv):
                if os.path.isfile(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            n_ok = sum(1 for _, ok, _ in results if ok)
            summary_lines = [f"Batch complete: {n_ok}/{len(results)} gene(s) succeeded.", ""]
            for gene_name, ok, message in results:
                status = "OK" if ok else "FAILED"
                summary_lines.append(f"[{status}] {gene_name}")
            summary = "\n".join(summary_lines)

            self.log("\n" + summary)
            if n_ok == 0:
                self.show_error("Batch failed", summary)
            else:
                self.show_info("Batch complete", summary)
        except Exception:
            import traceback
            err_text = "Unexpected error:\n" + traceback.format_exc()
            self.log(err_text)
            self.show_error("Error", err_text)
        finally:
            stop_caffeinate(caffeinate_proc)
            self.is_running = False
            self.set_status("Idle")

    def run_one_gene(self, inputs, gene_name, region_str, haplotype_path, gene_outdir, mean_depth_tsv, case_map, case_map_tsv):
        """Runs the full single-gene Clustering Alleles pipeline (Step1 -> Step2
        -> bundle) for one gene, under its own gene_outdir. Returns (success,
        message) instead of raising/showing a dialog, so the caller can log the
        result and continue on to the next gene in the batch."""
        gene_inputs = dict(inputs)
        gene_inputs["prefix"] = gene_name
        gene_inputs["gene"] = gene_name
        gene_inputs["regions"] = [region_str]
        gene_inputs["haplotype_tsv"] = haplotype_path or ""

        os.makedirs(gene_outdir, exist_ok=True)
        step1_outdir = os.path.join(gene_outdir, "step1")
        step2_outdir = os.path.join(gene_outdir, "step2")

        self.log(f"----- [{gene_name}] Step1 (SNP clustering) -----")
        step1_cmd = self.build_step1_cmd(gene_inputs, step1_outdir, case_map_tsv)
        rc = self.stream_subprocess(step1_cmd, f"[Step1:{gene_name}]")
        if rc != 0:
            return False, f"[{gene_name}] Step1 failed (exit code {rc}). See the log for details."

        if inputs["dry_run"]:
            return True, f"[{gene_name}] --dry-run: Step1 command verified, Step2 skipped."

        # sc/'s own processing upper-cases every sample name internally; restore
        # each one's original casing (as written in the BAM dir/list) in the final
        # cluster-label output file before anything downstream (build_bundle()) reads
        # it. Note this can't reach the sample-name labels burned into the PCA plot
        # PNGs themselves -- those are rendered inside this same Step1 subprocess,
        # before control returns here.
        snp_cluster_base = os.path.join(step1_outdir, "combined") if gene_inputs["two_stage"] else step1_outdir
        snp_cluster_txt = os.path.join(snp_cluster_base, f"step7_{gene_name}_cluster.txt")
        restore_case_in_cluster_file(snp_cluster_txt, case_map)

        joblib_path = self.locate_step1_joblib(step1_outdir, gene_name, gene_inputs["two_stage"])
        if not os.path.exists(joblib_path):
            return False, f"[{gene_name}] Step1 output joblib not found: {joblib_path}"
        self.log(f"[INFO] [{gene_name}] Found Step1 output joblib: {joblib_path}")

        # Step2 uses the SAME region-restricted BAMs Step1 already extracted (±10kb
        # around the gene region), not the original full-length BAMs -- combined with
        # the precomputed mean_depth above, this is numerically identical to running
        # Step2 against the full-length BAMs, but much faster/smaller.
        step2_bam_dir = os.path.join(step1_outdir, "step1_bams")
        if not glob.glob(os.path.join(step2_bam_dir, "*.bam")):
            return False, f"[{gene_name}] Step1's extracted BAM folder is empty or missing: {step2_bam_dir}"

        self.log(f"----- [{gene_name}] Step2 (INDEL clustering) -----")
        step2_cmd = self.build_step2_cmd(gene_inputs, step2_outdir, joblib_path, step2_bam_dir, mean_depth_tsv,
                                          case_map_tsv, snp_cluster_txt)
        rc = self.stream_subprocess(step2_cmd, f"[Step2:{gene_name}]")
        if rc != 0:
            return False, f"[{gene_name}] Step2 failed (exit code {rc}). See the log for details."

        # Same case restoration as Step1's cluster.txt, for every DEL/INS_TP/INS_notTP
        # cluster-label file plus the combined label file. Only sub.txt (not top.txt/
        # nested.txt) needs it -- cleanup_and_reorganize() below discards the other two.
        kmeans_dir = os.path.join(step2_outdir, "step7_kmeans")
        for svtype in ("DEL", "INS_TP", "INS_notTP"):
            svtype_dir = os.path.join(kmeans_dir, f"{gene_name}_{svtype}_out")
            restore_case_in_cluster_file(os.path.join(svtype_dir, "sub.txt"), case_map)
        restore_case_in_cluster_file(os.path.join(kmeans_dir, f"{gene_name}_combined.txt"), case_map)

        self.log(f"----- [{gene_name}] Combining PCA plots -----")
        panels = collect_clustering_pca_panels(step1_outdir, step2_outdir, gene_name)
        montage_path = build_pca_montage(
            panels, os.path.join(gene_outdir, f"{gene_name}_pca_montage.png"),
            title=f"{gene_name} - PCA plots")
        if montage_path:
            self.log(f"[OK] [{gene_name}] Combined PCA plot ({len(panels)} panel(s)): {montage_path}")
        else:
            self.log(f"[WARN] [{gene_name}] No PCA plots found to combine.")

        self.log(f"----- [{gene_name}] Bundling for Determining Alleles -----")
        bundle_path, bundle_errors = self.build_bundle(
            step1_outdir, step2_outdir, gene_outdir, gene_name, gene_inputs["two_stage"], region_str)
        if bundle_errors:
            self.log(f"[WARN] [{gene_name}] Bundle could not be created (Step1/Step2 outputs are unaffected):")
            for e in bundle_errors:
                self.log(f"  - {e}")
            bundle_path = None

        self.log(f"----- [{gene_name}] Reorganizing output (_results/ + rawdata/) -----")
        results_dir, snp_indel_dir, indel_dir = self.cleanup_and_reorganize(
            gene_outdir, step1_outdir, step2_outdir,
            gene_name, bundle_path, montage_path)

        summary = self.build_summary(gene_name, results_dir, snp_indel_dir, indel_dir, bundle_path, montage_path)
        return True, f"[{gene_name}] completed.\n{summary}"

    def _rm_glob(self, pattern):
        """Delete every file matching a glob pattern (supports ** with recursive=True)."""
        for p in glob.glob(pattern, recursive=True):
            try:
                os.remove(p)
            except OSError:
                pass

    def _prune_step1_vcfs(self, step1_outdir):
        """Delete every VCF under step1_outdir except ones literally named
        step3_selected.vcf (drops step2_raw.vcf, stage0's step3_intron.vcf,
        stage1's step3_0_sample_subset.vcf)."""
        for p in glob.glob(os.path.join(step1_outdir, "**", "*.vcf"), recursive=True):
            if os.path.basename(p) != "step3_selected.vcf":
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _dedupe_and_rename_allele_pca(self, step1_outdir):
        """For every step10_*_haplotype_match.txt (one per SNP-clustering stage
        that had --haplotype-tsv on), delete its sibling step10_*_haplotype_pca.png
        when _pca_has_haplotype_match() says nothing actually matched -- in that
        case it's pixel-identical to the plain step9 PCA plot, just a redundant
        copy. Whatever remains is then renamed from "haplotype" to "allele" in the
        filename (the GUI/tool itself calls this "alleles", not "haplotypes" --
        see the legend-text rename done earlier for the same reason)."""
        for match_txt in glob.glob(os.path.join(step1_outdir, "**", "*_haplotype_match.txt"), recursive=True):
            pca_png = match_txt[: -len("_haplotype_match.txt")] + "_haplotype_pca.png"
            if not _pca_has_haplotype_match(match_txt) and os.path.isfile(pca_png):
                try:
                    os.remove(pca_png)
                except OSError:
                    pass
            new_match = match_txt[: -len("_haplotype_match.txt")] + "_allele_match.txt"
            try:
                os.rename(match_txt, new_match)
            except OSError:
                pass
            if os.path.isfile(pca_png):
                new_pca = pca_png[: -len("_haplotype_pca.png")] + "_allele_pca.png"
                try:
                    os.rename(pca_png, new_pca)
                except OSError:
                    pass

    def cleanup_and_reorganize(self, gene_outdir, step1_outdir, step2_outdir,
                                gene_name, bundle_path, montage_path):
        """Restructure one gene's raw Step1/Step2 output, in place inside
        gene_outdir (== "{outdir}/{gene}/"), into:
          gene_outdir/
            _results/
              {gene}_combined.txt
              {gene}_pca_montage.png
              Determining_model/{gene}_bundle.joblib   (nothing else in this folder)
            rawdata/
              SNP_Short_InDel/  (former step1/, no further {gene} nesting -- gene_outdir
                                 is already gene-specific)
              Large_InDel/      (former step2/, likewise)
        Also deletes/renames, before the move:
          - the ±10kb region-extracted BAMs (only existed to feed Step2)
          - every intermediate *.joblib (already embedded in {gene}_bundle.joblib)
          - the one-hot/binary-encoded "step5" feature tables (already embedded
            in the bundle as each stage's "old_input") -- step4_*.tsv (the raw
            VCF->TSV polymorphism table, one step before this binarization) is
            kept in rawdata/SNP_Short_InDel/ as a real deliverable
          - every VCF except step3_selected.vcf
          - PCA explained-variance text files
          - two_stage_cluster_summary.txt
          - step1_predictions/ (step2's raw per-sample DEL/INS prediction TSVs,
            already folded into step3_wide/)
          - step10's haplotype-match PCA plot when it's identical to step9's
            (no sample actually matched), and "haplotype" -> "allele" renamed
            in whatever filenames remain
          - step7_kmeans/{gene}_{svtype}_out/ folders collapsed to a single
            {svtype}.txt (DEL.txt / INS_TP.txt / INS_notTP.txt) each, since
            sub.txt was the only file being kept in there anyway
        Runs regardless of whether bundling succeeded -- the rawdata/
        restructuring doesn't depend on it, and every _results/ move below
        already guards on the source file actually existing.
        """
        results_dir = os.path.join(gene_outdir, "_results")
        model_dir = os.path.join(results_dir, "Determining_model")
        rawdata_dir = os.path.join(gene_outdir, "rawdata")
        os.makedirs(model_dir, exist_ok=True)  # also creates results_dir
        os.makedirs(rawdata_dir, exist_ok=True)

        # 1. Delete the ±10kb region-extracted BAMs (only existed to feed Step2).
        step1_bams_dir = os.path.join(step1_outdir, "step1_bams")
        if os.path.isdir(step1_bams_dir):
            shutil.rmtree(step1_bams_dir, ignore_errors=True)

        # 2. Delete every intermediate joblib -- already embedded in {gene}_bundle.joblib.
        self._rm_glob(os.path.join(step1_outdir, "**", "*.joblib"))
        self._rm_glob(os.path.join(step2_outdir, "**", "*.joblib"))

        # 3. Delete the one-hot/binary-encoded ("バイナリ特徴量") feature tables --
        # already embedded in the bundle (as each stage's "old_input"). step4_*.tsv
        # (the VCF->TSV conversion, i.e. the raw per-position/per-sample genotype
        # table one step before this binarization) is kept in rawdata/SNP_Short_InDel/ --
        # it's a real deliverable (the collected polymorphism calls), not just an
        # intermediate, even though it's also embedded in the bundle as "targets".
        self._rm_glob(os.path.join(step1_outdir, "**", "step5_*.tsv"))
        step5_number_dir = os.path.join(step2_outdir, "step5_number")
        if os.path.isdir(step5_number_dir):
            shutil.rmtree(step5_number_dir, ignore_errors=True)

        # 4. VCFs: keep only step3_selected.vcf.
        self._prune_step1_vcfs(step1_outdir)

        # 5. Delete PCA explained-variance text files (plotting byproduct, not a deliverable).
        self._rm_glob(os.path.join(step1_outdir, "**", "*_variance.txt"))
        self._rm_glob(os.path.join(step2_outdir, "**", "*_variance.txt"))

        # 6. Delete the two-stage cluster summary and step2's raw per-sample
        # prediction TSVs (both superseded by other outputs).
        summary_txt = os.path.join(step1_outdir, "two_stage_cluster_summary.txt")
        if os.path.isfile(summary_txt):
            os.remove(summary_txt)
        step1_predictions_dir = os.path.join(step2_outdir, "step1_predictions")
        if os.path.isdir(step1_predictions_dir):
            shutil.rmtree(step1_predictions_dir, ignore_errors=True)

        # 7. step10 haplotype/allele PCA dedup + "haplotype" -> "allele" filename rename.
        self._dedupe_and_rename_allele_pca(step1_outdir)

        # 8. Collapse step7_kmeans/{gene}_{svtype}_out/sub.txt -> step7_kmeans/{svtype}.txt
        # (the folder existed only to hold what's now just this one file).
        kmeans_dir = os.path.join(step2_outdir, "step7_kmeans")
        for svtype in ("DEL", "INS_TP", "INS_notTP"):
            svtype_dir = os.path.join(kmeans_dir, f"{gene_name}_{svtype}_out")
            sub_txt = os.path.join(svtype_dir, "sub.txt")
            if os.path.isfile(sub_txt):
                shutil.move(sub_txt, os.path.join(kmeans_dir, f"{svtype}.txt"))
            if os.path.isdir(svtype_dir):
                shutil.rmtree(svtype_dir, ignore_errors=True)

        # 9. Populate _results/: a copy of the combined label file (the original
        # stays in rawdata/Large_InDel/ as part of step7_kmeans), plus the PCA
        # montage and bundle moved in (they have no other home).
        combined_txt = os.path.join(kmeans_dir, f"{gene_name}_combined.txt")
        if os.path.isfile(combined_txt):
            shutil.copy2(combined_txt, os.path.join(results_dir, f"{gene_name}_combined.txt"))
        if montage_path and os.path.isfile(montage_path):
            shutil.move(montage_path, os.path.join(results_dir, os.path.basename(montage_path)))
        if bundle_path and os.path.isfile(bundle_path):
            shutil.move(bundle_path, os.path.join(model_dir, os.path.basename(bundle_path)))

        # 10. Move the now-cleaned Step1/Step2 working directories into their final
        # rawdata/ home (gene_outdir is already gene-specific, so no further {gene}
        # nesting is needed here).
        snp_indel_dest = os.path.join(rawdata_dir, "SNP_Short_InDel")
        indel_dest = os.path.join(rawdata_dir, "Large_InDel")
        if os.path.isdir(snp_indel_dest):
            shutil.rmtree(snp_indel_dest, ignore_errors=True)
        if os.path.isdir(indel_dest):
            shutil.rmtree(indel_dest, ignore_errors=True)
        if os.path.isdir(step1_outdir):
            shutil.move(step1_outdir, snp_indel_dest)
        if os.path.isdir(step2_outdir):
            shutil.move(step2_outdir, indel_dest)

        return results_dir, snp_indel_dest, indel_dest

    def build_summary(self, gene_name, results_dir, snp_indel_dir, indel_dir, bundle_path=None, montage_path=None):
        lines = ["All steps completed.", "", "[_results/]"]
        combined_txt = os.path.join(results_dir, f"{gene_name}_combined.txt")
        lines.append(f"  Combined label: {combined_txt}" if os.path.isfile(combined_txt)
                     else "  Combined label: not created -- see the log above.")
        if bundle_path:
            final_bundle = os.path.join(results_dir, "Determining_model", os.path.basename(bundle_path))
            lines.append(f"  Bundle (Determining Alleles): {final_bundle}")
        else:
            lines.append("  Bundle (Determining Alleles): not created -- see the log above for missing artifacts.")
        if montage_path:
            final_montage = os.path.join(results_dir, os.path.basename(montage_path))
            lines.append(f"  Combined PCA plot: {final_montage}")
        else:
            lines.append("  Combined PCA plot: not created -- no PCA plots were found.")

        lines.append("")
        lines.append("[rawdata/]")
        lines.append(f"  SNP_Short_InDel: {snp_indel_dir}")
        lines.append(f"  Large_InDel: {indel_dir}")
        return "\n".join(lines)

    # ---------------- bundling (for Determining Alleles) ----------------

    def _snp_artifact_paths(self, step1_outdir, prefix, two_stage):
        base = os.path.join(step1_outdir, "combined") if two_stage else step1_outdir
        return {
            "model": os.path.join(base, f"step7_{prefix}_model.joblib"),
            "old_input": os.path.join(base, f"step5_{prefix}.tsv"),
            "old_cluster": os.path.join(base, f"step7_{prefix}_cluster.txt"),
            "targets": os.path.join(base, f"step4_{prefix}.tsv"),
            "pca_model": os.path.join(base, f"step9_{prefix}_pca_model.joblib"),
        }

    def _indel_artifact_paths(self, step2_outdir, prefix, svtype):
        kmeans_dir = os.path.join(step2_outdir, "step7_kmeans", f"{prefix}_{svtype}_out")
        return {
            "kmeans": os.path.join(kmeans_dir, f"{prefix}_{svtype}.joblib"),
            "old_cluster": os.path.join(kmeans_dir, "nested.txt"),
            "old_input": os.path.join(step2_outdir, "step5_number", f"{prefix}_{svtype}_number.tsv"),
            "targets": os.path.join(step2_outdir, "step3_wide", f"{prefix}_{svtype}.tsv"),
            "pca_model": os.path.join(step2_outdir, "step9_plot", f"{prefix}_{svtype}_pca_model.joblib"),
        }

    def _load_joblib_artifact(self, path, label, errors, required=True):
        if not os.path.isfile(path):
            msg = f"{label} not found: {path}"
            if required:
                errors.append(msg)
            else:
                self.log(f"[WARN] {msg} (optional, skipping)")
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            msg = f"{label} could not be loaded ({path}): {e}"
            if required:
                errors.append(msg)
            else:
                self.log(f"[WARN] {msg} (optional, skipping)")
            return None

    def _load_text_artifact(self, path, label, errors, required=True):
        if not os.path.isfile(path):
            msg = f"{label} not found: {path}"
            if required:
                errors.append(msg)
            else:
                self.log(f"[WARN] {msg} (optional, skipping)")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            msg = f"{label} could not be read ({path}): {e}"
            if required:
                errors.append(msg)
            else:
                self.log(f"[WARN] {msg} (optional, skipping)")
            return None

    def build_bundle(self, step1_outdir, step2_outdir, outdir, prefix, two_stage, gene_region):
        """Embed every artifact Determining Alleles needs into a single joblib
        (actual file contents/objects, not paths -- stays valid even if this
        run's output folder is later moved or deleted)."""
        errors = []

        snp_paths = self._snp_artifact_paths(step1_outdir, prefix, two_stage)
        snp = {
            "model": self._load_joblib_artifact(snp_paths["model"], "SNP model", errors),
            "old_input": self._load_text_artifact(snp_paths["old_input"], "SNP old-input features TSV", errors),
            "old_cluster": self._load_text_artifact(snp_paths["old_cluster"], "SNP old-cluster txt", errors),
            "targets": self._load_text_artifact(snp_paths["targets"], "SNP targets TSV", errors),
            "pca_model": self._load_joblib_artifact(snp_paths["pca_model"], "SNP PCA model", errors, required=False),
        }

        indel = {}
        for svtype, key in [("DEL", "del"), ("INS_TP", "instp"), ("INS_notTP", "insnottp")]:
            paths = self._indel_artifact_paths(step2_outdir, prefix, svtype)
            indel[key] = {
                "kmeans": self._load_joblib_artifact(paths["kmeans"], f"{svtype} KMeans model", errors),
                "old_input": self._load_text_artifact(paths["old_input"], f"{svtype} old-input features TSV", errors),
                "old_cluster": self._load_text_artifact(paths["old_cluster"], f"{svtype} old-cluster txt (nested.txt)", errors),
                "targets": self._load_text_artifact(paths["targets"], f"{svtype} targets TSV", errors),
                "pca_model": self._load_joblib_artifact(paths["pca_model"], f"{svtype} PCA model", errors, required=False),
            }

        if errors:
            return None, errors

        bundle = {
            "prefix": prefix,
            "gene_region": gene_region,
            "two_stage": two_stage,
            "snp": snp,
            "indel": indel,
        }
        bundle_path = os.path.join(outdir, f"{prefix}_bundle.joblib")
        joblib.dump(bundle, bundle_path)
        return bundle_path, []

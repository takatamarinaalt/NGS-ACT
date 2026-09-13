#!/usr/bin/env python3
"""Determining Alleles GUI frame: classifies new-sample BAMs against one or more
bundles produced by Clustering Alleles (sc/step3/1 = SNP, sc/step3/2 = INDEL,
sc/step3/3 = combine).

Batch layout, mirroring Clustering Alleles' batch design: the Bundle field
accepts a single .joblib file (classic single-gene use), a folder of
"{gene}_bundle.joblib" files, or a .txt list of absolute bundle paths (see
bam.txt/gene_ragion.txt for the same index-TAB-path convention) -- in the
folder/list cases, every bundle is processed against the same new-sample
BAM(s), each into its own "{outdir}/{gene_name}/" subfolder, using the gene
name recorded inside each bundle (bundle["prefix"]) to also look up a matching
haplotype TSV from the Haplotype folder, if one was given. One bundle failing
does not abort the rest of the batch.

Each bundle .joblib (produced automatically at the end of a Clustering Alleles
run) carries everything that used to require ~13 separate file paths.
Everything else (targets/model/kmeans/old-input/old-clus/pca-model for SNP,
DEL, and INS) is unpacked from the bundle at run time into a temp folder under
that gene's own output subfolder, then fed to the exact same sc/step3
orchestrators as file paths, unmodified.

The BAM field also accepts a .txt file listing absolute BAM paths, same as
Clustering Alleles' BAM field.
"""

import glob
import joblib
import os
import shutil
import sys
import threading
import tkinter as tk

from gui_common import (
    COLOR0, COLOR5, FieldBuilderMixin, FieldSpec, SubprocessRunnerMixin,
    build_pca_montage, collect_determining_pca_panels,
    draw_rounded_button, start_caffeinate, stop_caffeinate, which_default,
)

# This folder is self-contained: sc/ and model/ live INSIDE it (copies, not
# symlinks -- see CLAUDE.md's GUI folder family tree), so the whole folder
# can be zipped and sent to someone else without the rest of the repo.
# REPO_ROOT is therefore this folder itself, not its parent.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
STEP3_1_SCRIPT = os.path.join(REPO_ROOT, "sc", "step3", "1", "_run_pipeline.py")
STEP3_2_SCRIPT = os.path.join(REPO_ROOT, "sc", "step3", "2_newfea2", "___run_pipeline.py")
STEP3_3_SCRIPT = os.path.join(REPO_ROOT, "sc", "step3", "3", "combine_cluster.py")
# Same scripts Clustering Alleles uses for region extraction (±10kb around the
# gene region) and mean-depth precompute -- reused here so Determining Alleles
# follows the identical "extract once, compute mean depth from the original
# full-length BAM first" pattern instead of each step doing its own thing.
BAM_CHOICE_SCRIPT = os.path.join(REPO_ROOT, "sc", "step1", "step1", "1_bam_choice.py")
MEAN_DEPTH_SCRIPT = os.path.join(REPO_ROOT, "sc", "step1", "step2_newfea2", "estimate_mean_depth.py")
MEAN_DEPTH_JOBS = 0

# Output file prefix (was a user-editable "Prefix" field; fixed now since there
# was no real need to vary it -- every output filename is still prefix_*).
PREFIX = "new"

# Same fixed DEL/INS predict models used by Clustering Alleles (sc/step1/step2).
MODEL_DIR = os.path.join(REPO_ROOT, "model")
MODEL_DEL_PATH = os.path.join(MODEL_DIR, "DEL_3class_best_15.joblib")
# CLEAN_END_RATIO/INSERT_SIZE_DIFF を含む新INS特徴量セット（INS_features_v2.py）で
# 学習したモデル（MQ0側 = 全MQ閾値を0にして計算した特徴量で学習＝実質MQフィルタなし）。
MODEL_INS_PATH = os.path.join(MODEL_DIR, "INS_3class_best_15.joblib")
# 上のモデルに合わせて、INS特徴量計算のMQ閾値も揃える（MQ>=この値のリードのみ対象）。
INS_MQ = 0

# SNP側のジェノタイプ判定（bcftoolsでtargets座標のVCFを作成して照合）と、
# 新規多型クラスタ検出（4b/4c、bcftoolsで遺伝子領域全体をスキャンして
# 学習時のtargets外の新規多型を探す）の両方に --ref として渡す（必須）。
# ご自身の環境の参照FASTAパスに変更すると、GUI起動時にこの欄が自動で埋まります。
DEFAULT_REF = ""

REQUIRED_SNP_KEYS = ("model", "old_input", "old_cluster", "targets")
REQUIRED_INDEL_KEYS = ("kmeans", "old_input", "old_cluster", "targets")


def parse_bam_list_file(path):
    """Parse a BAM-list file (see bam.txt): one absolute BAM path per line,
    "<index><TAB>path" (the leading index column is optional and discarded if
    present). Blank lines and lines starting with '#' are skipped. Identical
    to clustering_gui.py's helper of the same name (duplicated rather than
    cross-imported, matching this project's existing pattern of independent
    per-GUI-file copies)."""
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


def parse_bundle_list_file(path):
    """Parse a bundle-list file (e.g. bundle.txt): one absolute .joblib path
    per line, same "<index><TAB>path" convention as parse_bam_list_file()."""
    return parse_bam_list_file(path)


def discover_bundles(bundle_input):
    """Resolve the Bundle field into an ordered list of bundle .joblib paths.
    Accepts a single file (classic single-gene use -> [path]), a directory
    (globs *.joblib, sorted), or a .txt list of absolute paths."""
    if os.path.isdir(bundle_input):
        return sorted(glob.glob(os.path.join(bundle_input, "*.joblib")))
    if bundle_input.endswith(".txt") and os.path.isfile(bundle_input):
        return parse_bundle_list_file(bundle_input)
    return [bundle_input]


def _tsv_contains_gene_id(path, gene_lower):
    """True if any data row's gene_id column (case-insensitive, exact match)
    equals gene_lower. Lets one combined TSV -- multiple genes' allele
    definitions in a single file, distinguished by the gene_id column, same
    cds.tsv format sc/'s haplotype-loading code already reads -- be
    discovered for every gene it actually defines, not just a per-gene file
    whose name happens to match. sc/ itself needs no change for this: both
    sc/step1/step1/10_haplotype_match.py (Clustering Alleles) and
    sc/step3/1/4_cluster.py's check_haplotype() (Determining Alleles) already
    iterate every haplotype in the TSV regardless of which gene it belongs
    to, and harmlessly fail to match (empty result, not an error) any row
    whose position isn't in the current gene's data -- which is every row
    belonging to a different gene."""
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
    the list-file convention is identical, just applied to TSV paths here. A
    candidate file matches if EITHER its filename (stem) is an exact
    case-insensitive match, or starts with gene_name followed by a
    non-alphanumeric separator (e.g. "Hd1_haplotype.tsv", "Hd1-x.tsv" --
    deliberately not a plain substring/prefix match, to avoid names like
    Hd1/Hd16/Hd17/Hd18 colliding) OR its gene_id column contains an exact
    case-insensitive match for gene_name (see _tsv_contains_gene_id() -- this
    is what lets one combined TSV with multiple genes' alleles in it be found
    for each gene it defines, not just per-gene files). Returns the matched
    path, or None if there's no match or the match is ambiguous (logged by
    the caller, treated as "no match" rather than guessing wrong). Identical
    to clustering_gui.py's helper of the same name. A gene with no match is
    still processed, just without haplotype matching for it."""
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


def sample_basename(bam_path):
    """Derive a sample's display name from its BAM path, stripping the same
    suffixes sc/'s own sample_basename()/guess_sample_name() strip, so the
    result matches what those scripts treat as "the sample name" for the same
    BAM. Case is preserved here (only sc/'s internal DEL/INS-side processing
    upper-cases it for matching purposes -- see dedup_sample_names_upper() in
    sc/step3/2_newfea2/1_predict_combined.py). Identical to clustering_gui.py's
    helper of the same name."""
    name = os.path.basename(bam_path)
    for suf in (".rg.bam", ".sort.bam", ".bam"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def build_original_case_map(bam_paths):
    """{UPPER(sample_name): sample_name_as_written} from the resolved BAM
    paths -- used to restore each sample's original casing (as given in the
    BAM dir/list) in the final cluster/label output files. Identical to
    clustering_gui.py's helper of the same name."""
    return {sample_basename(p).upper(): sample_basename(p) for p in bam_paths}


def restore_case_in_cluster_file(path, case_map):
    """Rewrite a "# cluster ..." formatted label file ({prefix}_cluster.txt,
    {prefix}_DEL_cluster.txt, {prefix}_INS_cluster.txt, {prefix}_combined.txt
    -- all the same format: a "# cluster ..."/"# Allele ..." header line, then
    one sample name per line, blank lines separating groups) in place,
    replacing each sample-name line with its original casing from case_map
    (case-insensitive lookup). A name with no match in case_map (shouldn't
    normally happen -- every sample that made it into this file came from the
    same BAM set the map was built from) is left untouched rather than
    dropped, so a mismatch never loses data. Identical to clustering_gui.py's
    helper of the same name.

    Deliberately only ever called AFTER every sc/ subprocess for this bundle
    has already finished (SNP, both INDEL calls, and the final combine step) --
    this is a one-shot cosmetic rewrite of an already-final text deliverable,
    never a file some later step in the same run still reads for
    case-sensitive sample-name matching. That ordering is what keeps this safe
    from the case-sensitivity bug found earlier in
    _apply_allele_match_to_combined_cluster() (sc/step1/step1/_run_pipeline.py),
    where two files with independently-uncertain casing were compared against
    each other mid-pipeline; nothing downstream of these files does any such
    comparison, so there is nothing left for a case mismatch to corrupt."""
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


class DeterminingAllelesFrame(tk.Frame, FieldBuilderMixin, SubprocessRunnerMixin):
    """Classifies a new sample's BAM(s) into an existing clustering-alleles group."""

    def __init__(self, master, on_back=None):
        super().__init__(master, bg=COLOR0)
        self.on_back = on_back
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
        tk.Label(top, text="Determining Alleles", font=("Helvetica Neue", 20, "bold"),
                fg="#264653", bg=COLOR0).pack(side="left", padx=16)

        outer, inner = self.make_scrollable_frame(self)
        outer.pack(fill="both", expand=True, padx=14, pady=8)
        self.build_fields(inner)

        run_frame = tk.Frame(self, bg=COLOR0)
        run_frame.pack(fill="x", padx=20, pady=(4, 4))

        self.status_label = tk.Label(run_frame, text="Idle", bg=COLOR0, fg="#264653", font=("Arial", 12))
        self.status_label.pack(side="left", padx=5)

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
        self.section_label(parent, row, "Output"); row += 1
        for spec in [
            FieldSpec("outdir", "Output folder", "dir", os.path.expanduser("~/Desktop/ngsact_new_sample")),
            FieldSpec("bcftools", "bcftools path", "file", which_default("bcftools")),
        ]:
            self.build_field(parent, row, spec); row += 1

        self.section_label(parent, row, "Input"); row += 1
        for spec in [
            FieldSpec("bundle", "Bundle (.joblib/folder/list)", "file", ""),
            FieldSpec("bam_dir", "BAM dir or list", "dir", ""),
            FieldSpec("ref", "Reference FASTA", "file", DEFAULT_REF),
        ]:
            self.build_field(parent, row, spec); row += 1

        # GFF / Alleles folder or list are only needed when checking for a
        # NEW allele not yet known at Clustering Alleles time (a bundle built
        # with "Match against reported alleles" on already embeds
        # haplotype_info, so these fields are dead weight in the common case)
        # -- hidden behind this checkbox, off by default, mirroring
        # Clustering Alleles' own ClusteringOptionsFrame checkbox.
        self.build_field(parent, row, FieldSpec(
            "use_alleles_match", "Check for new alleles", "check", False,
            command=self._toggle_alleles_match_fields))
        row += 1
        self._alleles_match_row_widgets = []
        for spec in [
            FieldSpec("gff", "GFF file", "file", ""),
            FieldSpec("haplotype_dir", "Alleles folder or list", "dir", ""),
        ]:
            self._alleles_match_row_widgets += self.build_field(parent, row, spec)
            row += 1
        self.set_row_visible(self._alleles_match_row_widgets, False)

    def _toggle_alleles_match_fields(self):
        self.set_row_visible(self._alleles_match_row_widgets,
                             bool(self.widgets["use_alleles_match"].get()))

    # ---------------- input collection & validation ----------------

    def collect_inputs(self):
        errors = []
        g = lambda k: self.widgets[k].get()
        use_alleles_match = bool(self.widgets["use_alleles_match"].get())
        inputs = {
            "outdir": g("outdir").strip(),
            "prefix": PREFIX,
            "bcftools": g("bcftools").strip(),
            "bundle": g("bundle").strip(),
            "bam_dir": g("bam_dir").strip(),
            "ref": g("ref").strip(),
            # Blank unless the checkbox is on, regardless of stale text left over
            # from a previous check/uncheck -- an unchecked box means "off", full stop.
            "gff": g("gff").strip() if use_alleles_match else "",
            "haplotype_dir": g("haplotype_dir").strip() if use_alleles_match else "",
        }
        return inputs, errors

    def validate_inputs(self, inputs):
        errors = []
        if not inputs["outdir"]:
            errors.append("Output folder is required.")
        if not inputs["bundle"]:
            errors.append("Bundle (.joblib/folder/list) is required.")
        elif not (os.path.isfile(inputs["bundle"]) or os.path.isdir(inputs["bundle"])):
            errors.append(f"Bundle file/folder/list not found: {inputs['bundle']}")
        if not inputs["bam_dir"]:
            errors.append("BAM directory or list (txt) is required.")
        elif not (os.path.isdir(inputs["bam_dir"]) or
                  (os.path.isfile(inputs["bam_dir"]) and inputs["bam_dir"].endswith(".txt"))):
            errors.append(f"BAM directory or .txt list not found: {inputs['bam_dir']}")
        if not inputs["ref"]:
            errors.append("Reference FASTA is required (used to call a VCF for the new sample "
                          "against the known target positions, and for new-allele/novel-cluster detection).")
        elif not os.path.isfile(inputs["ref"]):
            errors.append(f"Reference FASTA not found: {inputs['ref']}")
        if inputs["haplotype_dir"] and not (os.path.isdir(inputs["haplotype_dir"]) or
                (os.path.isfile(inputs["haplotype_dir"]) and inputs["haplotype_dir"].endswith(".txt"))):
            errors.append(f"Alleles folder or .txt list not found: {inputs['haplotype_dir']}")
        return errors

    def log_soft_warnings(self, inputs):
        if inputs["bcftools"] and not os.path.isfile(inputs["bcftools"]):
            self.log(f"[WARN] bcftools path not found (may be valid on a remote/mounted environment): {inputs['bcftools']}")
        if inputs["gff"] and not os.path.isfile(inputs["gff"]):
            self.log(f"[WARN] GFF file not found: {inputs['gff']}")

    # ---------------- bundle loading / unpacking ----------------

    def load_bundle(self, bundle_path):
        """Load and sanity-check the bundle. Returns (bundle_dict_or_None, errors)."""
        try:
            bundle = joblib.load(bundle_path)
        except Exception as e:
            return None, [f"Could not load bundle: {e}"]

        if not isinstance(bundle, dict):
            return None, ["Bundle is not a valid dict."]

        errors = []
        snp = bundle.get("snp")
        if not isinstance(snp, dict) or not all(snp.get(k) is not None for k in REQUIRED_SNP_KEYS):
            errors.append("Bundle is missing required SNP data (model/old_input/old_cluster/targets).")

        indel = bundle.get("indel")
        if not isinstance(indel, dict) or "del" not in indel or "instp" not in indel or "insnottp" not in indel:
            errors.append("Bundle is missing DEL/INS_TP/INS_notTP data (this bundle may be from an older "
                          "Clustering Alleles run, before INS split clustering).")
        else:
            for key in ("del", "instp", "insnottp"):
                section = indel.get(key) or {}
                if not all(section.get(k) is not None for k in REQUIRED_INDEL_KEYS):
                    errors.append(f"Bundle is missing required {key.upper()} data (kmeans/old_input/old_cluster/targets).")

        if errors:
            return None, errors
        return bundle, []

    def unpack_bundle(self, bundle, extract_dir):
        """Write the bundle's embedded content back out to real files under extract_dir.
        Returns a dict of the resulting file paths."""
        os.makedirs(extract_dir, exist_ok=True)
        paths = {}

        def write_text(key, filename, content):
            path = os.path.join(extract_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            paths[key] = path

        def write_joblib(key, filename, obj):
            path = os.path.join(extract_dir, filename)
            joblib.dump(obj, path)
            paths[key] = path

        snp = bundle["snp"]
        write_joblib("snp_model", "model.joblib", snp["model"])
        write_text("snp_old_input", "old_input.tsv", snp["old_input"])
        write_text("snp_old_cluster", "old_cluster.txt", snp["old_cluster"])
        write_text("snp_targets", "targets.tsv", snp["targets"])
        if snp.get("pca_model") is not None:
            write_joblib("snp_pca_model", "pca_model.joblib", snp["pca_model"])

        for svtype in ("del", "instp", "insnottp"):
            section = bundle["indel"][svtype]
            write_joblib(f"{svtype}_kmeans", f"kmeans_{svtype}.joblib", section["kmeans"])
            write_text(f"{svtype}_old_input", f"old_input_{svtype}.tsv", section["old_input"])
            write_text(f"{svtype}_old_cluster", f"old_clus_{svtype}.txt", section["old_cluster"])
            write_text(f"{svtype}_targets", f"targets_{svtype}.tsv", section["targets"])
            if section.get("pca_model") is not None:
                write_joblib(f"{svtype}_pca_model", f"pca_model_{svtype}.joblib", section["pca_model"])

        return paths

    # ---------------- command construction ----------------

    def build_snp_cmd(self, inputs, paths, snp_outdir, bam_files, case_map_tsv=None):
        cmd = [
            sys.executable, STEP3_1_SCRIPT,
            "--bam", *bam_files,
            "--targets", paths["snp_targets"],
            "--model", paths["snp_model"],
            "--old-input", paths["snp_old_input"],
            "--old-cluster", paths["snp_old_cluster"],
            "-o", snp_outdir,
            "--prefix", inputs["prefix"],
            "--ref", inputs["ref"],
        ]
        if inputs["bcftools"]:
            cmd += ["--bcftools", inputs["bcftools"]]
        if "snp_pca_model" in paths:
            cmd += ["--pca-model", paths["snp_pca_model"]]
        if inputs["haplotype_tsv"]:
            cmd += ["--haplotype-tsv", inputs["haplotype_tsv"]]
        if inputs["gff"]:
            cmd += ["--gff", inputs["gff"]]
        if case_map_tsv:
            cmd += ["--case-map", case_map_tsv]
        return cmd

    def build_indel_del_instp_cmd(self, inputs, paths, indel_outdir, bam_files, case_map_tsv=None):
        """DEL + INS_TP in one orchestrator call (sc/step3/2 has no ins-mode/split
        concept -- it just treats whatever --*-ins data it's given as "the INS
        clustering", so INS_TP data goes in via the ordinary --*-ins args)."""
        cmd = [
            sys.executable, STEP3_2_SCRIPT,
            "-o", indel_outdir,
            "--prefix", inputs["prefix"],
            "--bam", *bam_files,
            "--model-del", MODEL_DEL_PATH,
            "--model-ins", MODEL_INS_PATH,
            "--ins-mq", str(INS_MQ),
            "--targets-del", paths["del_targets"],
            "--kmeans-del", paths["del_kmeans"],
            "--old-input-del", paths["del_old_input"],
            "--old-clus-del", paths["del_old_cluster"],
            "--targets-ins", paths["instp_targets"],
            "--kmeans-ins", paths["instp_kmeans"],
            "--old-input-ins", paths["instp_old_input"],
            "--old-clus-ins", paths["instp_old_cluster"],
        ]
        if "del_pca_model" in paths:
            cmd += ["--pca-model-del", paths["del_pca_model"]]
        if "instp_pca_model" in paths:
            cmd += ["--pca-model-ins", paths["instp_pca_model"]]
        if inputs["gff"]:
            cmd += ["--gff", inputs["gff"]]
        if inputs["haplotype_tsv"]:
            cmd += ["--haplotype-tsv", inputs["haplotype_tsv"]]
        if inputs.get("skip_extraction"):
            cmd += ["--skip-extraction"]
        if inputs.get("region_mean_depth_tsv"):
            cmd += ["--mean-depth-tsv", inputs["region_mean_depth_tsv"]]
        if case_map_tsv:
            cmd += ["--case-map", case_map_tsv]
        return cmd

    def build_indel_insnottp_cmd(self, inputs, paths, indel_outdir2, bam_files, case_map_tsv=None):
        """INS_notTP alone, no --*-del args at all. Must write to a DIFFERENT
        output folder than build_indel_del_instp_cmd()'s -- both calls produce a
        file literally named "{prefix}_INS_cluster.txt" (the orchestrator has no
        idea it's being fed an INS_TP vs INS_notTP subset), so they'd collide if
        pointed at the same -o."""
        cmd = [
            sys.executable, STEP3_2_SCRIPT,
            "-o", indel_outdir2,
            "--prefix", inputs["prefix"],
            "--bam", *bam_files,
            "--model-ins", MODEL_INS_PATH,
            "--ins-mq", str(INS_MQ),
            "--targets-ins", paths["insnottp_targets"],
            "--kmeans-ins", paths["insnottp_kmeans"],
            "--old-input-ins", paths["insnottp_old_input"],
            "--old-clus-ins", paths["insnottp_old_cluster"],
        ]
        if "insnottp_pca_model" in paths:
            cmd += ["--pca-model-ins", paths["insnottp_pca_model"]]
        if inputs["gff"]:
            cmd += ["--gff", inputs["gff"]]
        if inputs["haplotype_tsv"]:
            cmd += ["--haplotype-tsv", inputs["haplotype_tsv"]]
        if inputs.get("skip_extraction"):
            cmd += ["--skip-extraction"]
        if inputs.get("region_mean_depth_tsv"):
            cmd += ["--mean-depth-tsv", inputs["region_mean_depth_tsv"]]
        if case_map_tsv:
            cmd += ["--case-map", case_map_tsv]
        return cmd

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

    def _resolve_bam_files(self, bam_input):
        """Resolve the BAM field (a directory, or a .txt list of absolute paths)
        into a sorted list of real .bam file paths, skipping macOS AppleDouble
        sidecar files (e.g. "._sample.bam") that show up on exFAT/FAT32 external
        drives / network shares -- they match "*.bam" but aren't real BAMs and
        make pysam fail with "Exec format error"."""
        if bam_input.endswith(".txt") and os.path.isfile(bam_input):
            candidates = parse_bam_list_file(bam_input)
        else:
            candidates = glob.glob(os.path.join(bam_input, "*.bam"))
        return sorted(p for p in candidates if not os.path.basename(p).startswith("._"))

    def run_main_work(self, inputs):
        """Batch orchestrator: resolves the Bundle field into one or more bundle
        paths (single file, folder of *.joblib, or .txt list), resolves the BAM
        field once (shared -- the same new samples are classified against every
        bundle/gene), then runs the full single-bundle Determining Alleles
        pipeline once per bundle, each into its own "{outdir}/{gene_name}/"
        subfolder. One bundle failing is logged and does not stop the batch."""
        caffeinate_proc = start_caffeinate()
        try:
            self.log("========== Batch run started ==========")
            if caffeinate_proc is not None:
                self.log("[INFO] Sleep prevention active for the duration of this batch (caffeinate).")
            self.set_status("Running...")

            outdir = inputs["outdir"]
            os.makedirs(outdir, exist_ok=True)

            bundle_paths = discover_bundles(inputs["bundle"])
            if not bundle_paths:
                self.log(f"[ERROR] No bundle(s) found at: {inputs['bundle']}")
                self.show_error("Error", f"No bundle(s) found at: {inputs['bundle']}")
                return
            self.log(f"[INFO] Found {len(bundle_paths)} bundle(s) to process.")

            bam_files = self._resolve_bam_files(inputs["bam_dir"])
            if not bam_files:
                self.show_error("Error", f"No .bam files found for: {inputs['bam_dir']}")
                return
            self.log(f"[INFO] Using {len(bam_files)} BAM file(s), shared across every bundle.")

            # Sample names get upper-cased internally by the DEL/INS-side
            # processing (and by sc/step3/3/combine_cluster.py's own
            # cross-file matching, regardless of the per-svtype files'
            # casing) -- built once here from the new sample's own BAM(s),
            # same "shared across the whole batch" pattern as mean_depth_tsv,
            # and used to restore original casing in each bundle's final
            # cluster/combined text output. Mirrors clustering_gui.py's
            # identical build_original_case_map() call.
            case_map = build_original_case_map(bam_files)
            # Also written out as a TSV and passed via --case-map into the
            # SNP/INDEL orchestrators' own 7_newplot.py subprocess calls, so
            # the new sample's PCA-plot point label shows the original BAM
            # casing too (restore_case_in_cluster_file() above only fixes the
            # text cluster/combined files, not pixels already baked into a
            # PNG by a plotting subprocess that ran earlier in the pipeline).
            case_map_tsv = os.path.join(outdir, "case_map.tsv")
            with open(case_map_tsv, "w", encoding="utf-8") as f:
                for upper, original in case_map.items():
                    f.write(f"{upper}\t{original}\n")

            # Mean depth only depends on the new sample's BAM(s) (not the gene
            # region), and the same BAM(s) are used for every bundle in this batch,
            # so compute it once here -- against the ORIGINAL full-length BAMs,
            # before any per-bundle region extraction -- instead of once per
            # bundle. Mirrors clustering_gui.py's identical batch-level
            # build_mean_depth_cmd() precompute.
            mean_depth_tsv = os.path.join(outdir, "mean_depth.tsv")
            self.log("----- Estimating mean depth (full-length BAMs, shared across all bundles) -----")
            mean_depth_cmd = [
                sys.executable, MEAN_DEPTH_SCRIPT,
                "--bam", *bam_files,
                "-o", mean_depth_tsv,
                "--jobs", str(MEAN_DEPTH_JOBS),
            ]
            rc = self.stream_subprocess(mean_depth_cmd, "[MeanDepth]")
            if rc != 0:
                self.show_error("Mean depth estimation failed", f"Exited with code {rc}. See the log for details.")
                return

            results = []  # (label, success, message)
            for bundle_path in bundle_paths:
                self.log(f"\n========== Bundle: {bundle_path} ==========")
                try:
                    success, label, message = self.run_one_bundle(
                        inputs, bundle_path, bam_files, outdir, mean_depth_tsv, case_map, case_map_tsv)
                except Exception:
                    import traceback
                    success = False
                    label = os.path.basename(bundle_path)
                    message = "Unexpected error:\n" + traceback.format_exc()
                self.log(message)
                results.append((label, success, message))

            # mean_depth.tsv and case_map.tsv are both shared across every
            # bundle in the loop above -- only safe to delete once the whole
            # batch is done with them.
            for tmp_path in (mean_depth_tsv, case_map_tsv):
                if os.path.isfile(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            n_ok = sum(1 for _, ok, _ in results if ok)
            summary_lines = [f"Batch complete: {n_ok}/{len(results)} bundle(s) succeeded.", ""]
            for label, ok, _ in results:
                status = "OK" if ok else "FAILED"
                summary_lines.append(f"[{status}] {label}")
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

    def run_one_bundle(self, inputs, bundle_path, bam_files, base_outdir, mean_depth_tsv=None, case_map=None, case_map_tsv=None):
        """Runs the full single-bundle Determining Alleles pipeline (SNP -> INDEL
        -> combine) for one bundle, under its own "{gene_name}/" subfolder.
        Returns (success, label, message) instead of raising/showing a dialog,
        so the caller can log the result and continue to the next bundle."""
        self.log(f"----- Loading bundle: {bundle_path} -----")
        bundle, bundle_errors = self.load_bundle(bundle_path)
        if bundle_errors:
            label = os.path.basename(bundle_path)
            return False, label, f"[{label}] Invalid bundle:\n" + "\n".join(bundle_errors)

        gene_name = bundle.get("prefix") or os.path.splitext(os.path.basename(bundle_path))[0]
        gene_region = bundle.get("gene_region")
        self.log(f"[INFO] Bundle loaded (prefix='{gene_name}', gene_region={gene_region}).")

        haplotype_path = None
        if inputs["haplotype_dir"]:
            haplotype_path, candidates = match_haplotype_file(gene_name, inputs["haplotype_dir"])
            if haplotype_path:
                self.log(f"[INFO] [{gene_name}] Alleles match: {os.path.basename(haplotype_path)}")
            elif candidates:
                self.log(f"[WARN] [{gene_name}] Ambiguous Alleles match ({', '.join(candidates)}), skipping")
            else:
                self.log(f"[INFO] [{gene_name}] No Alleles TSV found -- processing without one")

        bundle_inputs = dict(inputs)
        bundle_inputs["haplotype_tsv"] = haplotype_path or ""

        gene_outdir = os.path.join(base_outdir, os.path.basename(gene_name).replace("/", "_").replace("\\", "_"))
        os.makedirs(gene_outdir, exist_ok=True)
        prefix = inputs["prefix"]

        # Extract the new sample's BAM(s) to ±10kb around the gene region ONCE
        # here (same script/flank Clustering Alleles uses, sc/step1/step1/1_bam_choice.py),
        # and reuse the extracted BAMs for both the SNP and INDEL steps below --
        # mirrors Clustering Alleles' own architecture, instead of the SNP step
        # querying the full-length BAM directly while the INDEL step extracts
        # (and recomputes mean depth) independently on its own. Falls back to
        # the original BAM(s) untouched if the bundle has no recorded gene_region
        # (older bundle) or extraction fails -- the INDEL orchestrator still
        # works standalone in that case, it just does its own extraction again.
        region_bam_files = bam_files
        if gene_region:
            region_bams_dir = os.path.join(gene_outdir, "_region_bams")
            extract_cmd = [
                sys.executable, BAM_CHOICE_SCRIPT,
                "--input_bams", *bam_files,
                "--regions", gene_region,
                "--output_dir", region_bams_dir,
            ]
            self.log(f"----- [{gene_name}] Extracting BAM(s) to {gene_region} (±10kb) -----")
            rc = self.stream_subprocess(extract_cmd, f"[Extract:{gene_name}]")
            if rc == 0:
                region_bam_files = [os.path.join(region_bams_dir, os.path.basename(b)) for b in bam_files]
                bundle_inputs["skip_extraction"] = True
                if mean_depth_tsv:
                    bundle_inputs["region_mean_depth_tsv"] = mean_depth_tsv
            else:
                self.log(f"[WARN] [{gene_name}] BAM region extraction failed (exit code {rc}); "
                         f"falling back to the original BAM(s) (INDEL step will extract on its own).")
        else:
            self.log(f"[INFO] [{gene_name}] Bundle has no recorded gene_region -- "
                     f"skipping shared extraction (INDEL step will extract on its own if needed).")

        extract_dir = os.path.join(gene_outdir, "_bundle_extract")
        paths = self.unpack_bundle(bundle, extract_dir)
        self.log(f"[INFO] [{gene_name}] Unpacked bundle into: {extract_dir}")

        snp_outdir = os.path.join(gene_outdir, "step3_1")
        indel_outdir = os.path.join(gene_outdir, "step3_2")
        indel_outdir_insnottp = os.path.join(gene_outdir, "step3_2_insnottp")
        combine_outdir = os.path.join(gene_outdir, "step3_3")

        self.log(f"----- [{gene_name}] SNP determining (step3/1) -----")
        snp_cmd = self.build_snp_cmd(bundle_inputs, paths, snp_outdir, region_bam_files, case_map_tsv)
        rc = self.stream_subprocess(snp_cmd, f"[SNP:{gene_name}]")
        if rc != 0:
            return False, gene_name, f"[{gene_name}] SNP step failed (exit code {rc}). See the log for details."

        self.log(f"----- [{gene_name}] INDEL determining: DEL + INS_TP (step3/2) -----")
        indel_cmd = self.build_indel_del_instp_cmd(bundle_inputs, paths, indel_outdir, region_bam_files, case_map_tsv)
        rc = self.stream_subprocess(indel_cmd, f"[INDEL:{gene_name}]")
        if rc != 0:
            return False, gene_name, f"[{gene_name}] INDEL step failed (exit code {rc}). See the log for details."

        self.log(f"----- [{gene_name}] INDEL determining: INS_notTP (step3/2) -----")
        indel_insnottp_cmd = self.build_indel_insnottp_cmd(bundle_inputs, paths, indel_outdir_insnottp, region_bam_files, case_map_tsv)
        rc = self.stream_subprocess(indel_insnottp_cmd, f"[INDEL:{gene_name}]")
        if rc != 0:
            return False, gene_name, f"[{gene_name}] INDEL step failed (exit code {rc}). See the log for details."

        snp_cluster = os.path.join(snp_outdir, f"{prefix}_cluster.txt")
        del_cluster = os.path.join(indel_outdir, f"{prefix}_DEL_cluster.txt")
        instp_cluster = os.path.join(indel_outdir, f"{prefix}_INS_cluster.txt")
        insnottp_cluster = os.path.join(indel_outdir_insnottp, f"{prefix}_INS_cluster.txt")

        # Restore original BAM-derived casing in each svtype's own cluster.txt
        # -- safe to do now since every subprocess that could still read these
        # files for case-sensitive matching (SNP/INDEL predict, this run's own
        # PCA plotting) has already finished; only the upcoming combine step
        # and the human reading these files afterward touch them from here on,
        # and combine_cluster.py already normalizes casing internally for its
        # own cross-file matching (see restore_case_in_cluster_file()'s
        # docstring), so restoring case on its inputs doesn't affect it.
        if case_map:
            for cluster_txt in (snp_cluster, del_cluster, instp_cluster, insnottp_cluster):
                restore_case_in_cluster_file(cluster_txt, case_map)

        combined_path = None
        if all(os.path.exists(p) for p in (snp_cluster, del_cluster, instp_cluster, insnottp_cluster)):
            os.makedirs(combine_outdir, exist_ok=True)
            combined_path = os.path.join(combine_outdir, f"{prefix}_combined.txt")
            combine_cmd = [
                sys.executable, STEP3_3_SCRIPT,
                "--snp_cluster", snp_cluster,
                "--del_cluster", del_cluster,
                "--instp_cluster", instp_cluster,
                "--insnottp_cluster", insnottp_cluster,
                "--output", combined_path,
            ]
            self.log(f"----- [{gene_name}] Combine (step3/3) -----")
            rc = self.stream_subprocess(combine_cmd, f"[Combine:{gene_name}]")
            if rc != 0:
                return False, gene_name, f"[{gene_name}] Combine step failed (exit code {rc}). See the log for details."
            # combine_cluster.py's own cross-file sample matching upper-cases
            # every sample name internally (see parse_cluster_file()), so its
            # output is always all-uppercase regardless of the input files'
            # casing -- restore it here too, same as the per-svtype files above.
            if case_map:
                restore_case_in_cluster_file(combined_path, case_map)
        else:
            self.log(f"[WARN] [{gene_name}] Combine step skipped: one or more cluster result files were not produced.")

        self.log(f"----- [{gene_name}] Combining PCA plots -----")
        panels = collect_determining_pca_panels(snp_outdir, indel_outdir, indel_outdir_insnottp, prefix)
        montage_path = build_pca_montage(
            panels, os.path.join(gene_outdir, f"{prefix}_pca_montage.png"),
            title=f"{gene_name} - PCA plots")
        if montage_path:
            self.log(f"[OK] [{gene_name}] Combined PCA plot ({len(panels)} panel(s)): {montage_path}")
        else:
            self.log(f"[WARN] [{gene_name}] No PCA plots found to combine.")

        results_dir, rawdata_dir = self.cleanup_and_reorganize(
            gene_outdir, snp_outdir, indel_outdir, indel_outdir_insnottp,
            combined_path, montage_path, prefix)

        summary = self.build_summary(results_dir, rawdata_dir, prefix)
        return True, gene_name, f"[{gene_name}] completed.\n{summary}"

    # ---------------- output cleanup / reorganization ----------------

    def cleanup_and_reorganize(self, gene_outdir, snp_outdir, indel_outdir, indel_outdir_insnottp,
                                combined_path, montage_path, prefix):
        """Restructure one bundle's raw step3_1/step3_2/step3_2_insnottp/step3_3
        output, in place inside gene_outdir (== "{outdir}/{gene}/"), into:
          gene_outdir/
            _results/
              {prefix}_combined.txt
              {prefix}_pca_montage.png
            rawdata/
              step3_1/
              step3_2/
              step3_2_insnottp/
        Also deletes, before the move (all intermediate/bulky files already
        consumed by earlier steps, not deliverables in their own right):
          - _bundle_extract/ (the bundle unpacked to real files -- already
            embedded in the bundle.joblib itself, not needed once every step
            that reads it has run)
          - step3_1's {prefix}_raw.tsv / {prefix}_features.tsv (superseded by
            cluster.txt / simplified.tsv)
          - the ±10kb region-extracted BAMs (_region_bams/, only existed to
            feed 1_predict_combined.py)
          - mean_depth.tsv (Step1's input, not a deliverable)
          - gene_absent_samples.txt (1_predict_combined.py's GENE_ABSENT
            detection marker -- already reflected in cluster.txt/combined.txt)
          - the DEL/INS one-hot feature TSVs ({prefix}_DEL_features.tsv /
            {prefix}_INS_features.tsv, PCA-prep byproduct)
          - features/ and predictions/ (per-position/per-sample raw
            prediction output, already folded into {prefix}_{svtype}_merged.tsv)
        Runs regardless of whether the combine step succeeded -- the rawdata/
        restructuring doesn't depend on it, and the _results/ move guards on
        combined_path actually existing.
        """
        results_dir = os.path.join(gene_outdir, "_results")
        rawdata_dir = os.path.join(gene_outdir, "rawdata")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(rawdata_dir, exist_ok=True)

        # 1. Delete the unpacked bundle extraction dir -- already embedded in the bundle.
        extract_dir = os.path.join(gene_outdir, "_bundle_extract")
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)

        # 1b. Delete the shared ±10kb region-extracted BAMs (run_one_bundle()'s own
        # gene_outdir/_region_bams/, used by both the SNP and INDEL steps) -- only
        # existed to feed those steps, not a deliverable.
        shared_region_bams_dir = os.path.join(gene_outdir, "_region_bams")
        if os.path.isdir(shared_region_bams_dir):
            shutil.rmtree(shared_region_bams_dir, ignore_errors=True)

        # 2. step3_1: drop the raw/features TSVs (superseded by cluster.txt/simplified.tsv).
        for name in (f"{prefix}_raw.tsv", f"{prefix}_features.tsv"):
            p = os.path.join(snp_outdir, name)
            if os.path.isfile(p):
                os.remove(p)

        # 3-5. step3_2 / step3_2_insnottp: region-extracted BAMs, mean_depth.tsv,
        # one-hot feature TSVs, and the per-position/per-sample features/predictions dirs.
        for indel_dir in (indel_outdir, indel_outdir_insnottp):
            region_bams_dir = os.path.join(indel_dir, "_region_bams")
            if os.path.isdir(region_bams_dir):
                shutil.rmtree(region_bams_dir, ignore_errors=True)
            mean_depth_tsv = os.path.join(indel_dir, "mean_depth.tsv")
            if os.path.isfile(mean_depth_tsv):
                os.remove(mean_depth_tsv)
            # 1_predict_combined.py's GENE_ABSENT detection marker -- already
            # embedded in the final cluster.txt/combined.txt labels, not a
            # deliverable in its own right.
            gene_absent_tsv = os.path.join(indel_dir, "gene_absent_samples.txt")
            if os.path.isfile(gene_absent_tsv):
                os.remove(gene_absent_tsv)
            for name in (f"{prefix}_DEL_features.tsv", f"{prefix}_INS_features.tsv"):
                p = os.path.join(indel_dir, name)
                if os.path.isfile(p):
                    os.remove(p)
            for sub in ("features", "predictions"):
                d = os.path.join(indel_dir, sub)
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)

        # 6. Populate _results/: the combined label + PCA montage (they have no other home).
        # The now-empty step3_3/ (its only file just moved out) is removed rather than
        # left behind as a dangling empty directory.
        if combined_path and os.path.isfile(combined_path):
            combine_dir = os.path.dirname(combined_path)
            shutil.move(combined_path, os.path.join(results_dir, os.path.basename(combined_path)))
            if os.path.isdir(combine_dir) and not os.listdir(combine_dir):
                os.rmdir(combine_dir)
        if montage_path and os.path.isfile(montage_path):
            shutil.move(montage_path, os.path.join(results_dir, os.path.basename(montage_path)))

        # 7. Move the now-cleaned step3_1/step3_2/step3_2_insnottp working
        # directories into their final rawdata/ home, named after the
        # polymorphism type rather than the internal step-folder name
        # (matching Clustering Alleles' own SNP_Short_InDel/Large_InDel naming).
        rawdata_names = {
            snp_outdir: "SNP_Short_InDel",
            indel_outdir: "Large_InDel",
            indel_outdir_insnottp: "Large_InDel_notTP",
        }
        for src, dest_name in rawdata_names.items():
            if os.path.isdir(src):
                dest = os.path.join(rawdata_dir, dest_name)
                if os.path.isdir(dest):
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(src, dest)

        return results_dir, rawdata_dir

    def build_summary(self, results_dir, rawdata_dir, prefix):
        lines = ["All steps completed.", "", "[_results/]"]
        combined_txt = os.path.join(results_dir, f"{prefix}_combined.txt")
        lines.append(f"  Combined label: {combined_txt}" if os.path.isfile(combined_txt)
                     else "  Combined label: not created -- see the log above for missing artifacts.")
        montage = os.path.join(results_dir, f"{prefix}_pca_montage.png")
        lines.append(f"  Combined PCA plot: {montage}" if os.path.isfile(montage)
                     else "  Combined PCA plot: not created -- no PCA plots were found.")
        lines.append("")
        lines.append("[rawdata/]")
        lines.append(f"  {rawdata_dir}")
        return "\n".join(lines)

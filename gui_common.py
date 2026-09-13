#!/usr/bin/env python3
"""Shared widgets, helpers, and subprocess-streaming mixins for the NGS-ACT GUI."""

import glob
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_FILES = None
    DND_AVAILABLE = False

COLOR0 = "#FAF8EF"   # background
COLOR4 = "#2C6E49"   # section headings
COLOR5 = "#D84F4F"   # primary action button (Run)
ENTRY_BG = "#DDE7E2"
ENTRY_FOCUS_BG = "#CFE6E0"
BROWSE_BTN_COLOR = "#FFB38A"


def start_caffeinate():
    """Spawn `caffeinate -i` to hold a "prevent idle system sleep" assertion
    for as long as this Popen handle is kept alive. macOS system sleep halts
    the CPU entirely -- every process (including any bcftools/Python
    subprocess this app has launched) is frozen mid-execution until the Mac
    wakes up again, and (per the sc/ gotcha this mirrors) an external drive
    can also drop/unmount during that time. `caffeinate` is the standard,
    documented macOS mechanism for preventing that. Returns None (instead of
    raising) if `caffeinate` isn't available -- e.g. running on a non-macOS
    system -- so callers can treat "no assertion held" as a safe no-op rather
    than crashing the run."""
    try:
        return subprocess.Popen(["caffeinate", "-i"])
    except OSError:
        return None


def stop_caffeinate(proc):
    """Release the assertion started by start_caffeinate(). Safe to call with
    None (e.g. if start_caffeinate() couldn't launch caffeinate)."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass


def _pca_has_haplotype_match(match_txt_path):
    """True if a step10 haplotype-match TSV (columns: haplotype, sample,
    cluster) shows at least one row where a sample actually matched --
    "(一致なし)"/blank means no match. Used to decide whether the combined
    PCA montage should show that stage's haplotype-overlay plot instead of
    the plain one."""
    if not os.path.isfile(match_txt_path):
        return False
    with open(match_txt_path, "r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip() not in ("", "(一致なし)"):
                return True
    return False


def _pca_pick_snp_panel(dir_path, plain_name, hap_match_name, hap_png_name):
    """Pick which PNG represents a given step1 SNP stage: the haplotype-match
    overlay plot (step10) if 10_haplotype_match.py actually found a matching
    sample there, otherwise the plain PCA plot (step9). Returns (path, used_
    haplotype_plot) or (None, False) if neither file exists."""
    hap_png = os.path.join(dir_path, hap_png_name)
    if _pca_has_haplotype_match(os.path.join(dir_path, hap_match_name)) and os.path.isfile(hap_png):
        return hap_png, True
    plain_png = os.path.join(dir_path, plain_name)
    if os.path.isfile(plain_png):
        return plain_png, False
    return None, False


def collect_clustering_pca_panels(step1_outdir, step2_outdir, prefix):
    """All PCA plots Clustering Alleles' step1 (SNP -- just the unified
    combined-stage plot, not the per-primary-cluster ones) and step2 (DEL /
    INS_TP / INS_notTP) produced for one gene, as an ordered list of (label,
    png_path) tuples left-to-right. Handles both two-stage (combined/) and
    single-stage (flat step1_outdir) layouts. A stage missing its plot (e.g.
    PCA skipped due to 0 features) is simply omitted, not an error."""
    panels = []
    stage1_dirs = [d for d in glob.glob(os.path.join(step1_outdir, "stage1_cluster*")) if os.path.isdir(d)]
    if stage1_dirs:
        combined_dir = os.path.join(step1_outdir, "combined")
        png, used_hap = _pca_pick_snp_panel(
            combined_dir, f"step9_{prefix}_pca.png",
            f"step10_{prefix}_haplotype_match.txt",
            f"step10_{prefix}_haplotype_pca.png")
        if png:
            panels.append(("combined (SNP_Short_InDel)" + (" [allele match]" if used_hap else ""), png))
    else:
        png, used_hap = _pca_pick_snp_panel(
            step1_outdir, f"step9_{prefix}_pca.png",
            f"step10_{prefix}_haplotype_match.txt",
            f"step10_{prefix}_haplotype_pca.png")
        if png:
            panels.append(("SNP_Short_InDel" + (" [allele match]" if used_hap else ""), png))

    for svtype, label in [("DEL", "DEL"), ("INS_TP", "INS_TP"), ("INS_notTP", "INS_notTP")]:
        png = os.path.join(step2_outdir, "step9_plot", f"{prefix}_{svtype}_plot.png")
        if os.path.isfile(png):
            panels.append((label, png))
    return panels


def collect_determining_pca_panels(snp_outdir, indel_outdir, indel_outdir_insnottp, prefix):
    """All PCA plots Determining Alleles' SNP (step3/1) and INDEL (step3/2,
    run twice -- DEL+INS_TP then INS_notTP) steps produced for one bundle, as
    an ordered list of (label, png_path) tuples left-to-right. Unlike
    Clustering Alleles, step3/1's --haplotype-tsv only annotates the model's
    haplotype_info (used for cluster-label matching) rather than producing a
    separate overlay plot, so there is no plain-vs-haplotype substitution
    here -- just one plot per stage."""
    panels = []
    for label, png in [
        ("SNP_Short_InDel", os.path.join(snp_outdir, f"{prefix}_plot.png")),
        ("DEL", os.path.join(indel_outdir, f"{prefix}_DEL_plot.png")),
        ("INS_TP", os.path.join(indel_outdir, f"{prefix}_INS_plot.png")),
        ("INS_notTP", os.path.join(indel_outdir_insnottp, f"{prefix}_INS_plot.png")),
    ]:
        if os.path.isfile(png):
            panels.append((label, png))
    return panels


def build_pca_montage(panels, out_path, title=None, cols=2):
    """Combine PCA plot PNGs (a list of (label, path) tuples, already in the
    desired reading order) into one grid-arranged image (2 columns by default,
    wrapping to as many rows as needed -- e.g. 4 panels become a 2x2 square
    instead of one wide horizontal strip), each panel labeled above it.
    Returns out_path, or None if there were no panels to combine (not every
    gene/bundle has produced every stage's plot)."""
    if not panels:
        return None

    import math
    from PIL import Image, ImageDraw, ImageFont

    cols = max(1, min(cols, len(panels)))
    rows = math.ceil(len(panels) / cols)
    label_h = 80
    gap = 16
    margin = 20
    title_h = 46 if title else 0

    imgs = [Image.open(p).convert("RGB") for _, p in panels]
    # Every cell in the grid is the same size (the largest panel's dimensions)
    # so columns/rows line up evenly; smaller panels are scaled up to fill
    # their cell (aspect ratio preserved) rather than left at their own size,
    # since the source plots are already near-identical in shape.
    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)
    resized = []
    for im in imgs:
        if im.width != cell_w or im.height != cell_h:
            scale = min(cell_w / im.width, cell_h / im.height)
            new_w, new_h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)
        resized.append(im)

    total_w = cell_w * cols + gap * (cols - 1) + margin * 2
    total_h = (cell_h + label_h) * rows + gap * (rows - 1) + title_h + margin * 2

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    def _load_font(paths, size):
        for path in paths:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    font = _load_font(
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Helvetica.ttc"], 60)
    title_font = _load_font(["/System/Library/Fonts/Helvetica.ttc"], 28)

    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, margin // 2), title, fill="black", font=title_font)

    for idx, ((label, _), im) in enumerate(zip(panels, resized)):
        col, row = idx % cols, idx // cols
        cell_x = margin + col * (cell_w + gap)
        cell_y = margin + title_h + row * (cell_h + label_h + gap)

        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cell_x + (cell_w - tw) // 2, cell_y), label, fill="black", font=font)
        img_x = cell_x + (cell_w - im.width) // 2
        canvas.paste(im, (img_x, cell_y + label_h))

    canvas.save(out_path)
    return out_path


def draw_rounded_button(canvas, x, y, width, height, radius, bg_color, text, command, font_size=12):
    """Draw a rounded button on a Canvas."""
    parts = []
    parts.append(canvas.create_arc(x, y, x + 2 * radius, y + 2 * radius, start=90, extent=90, fill=bg_color, outline=bg_color))
    parts.append(canvas.create_arc(x + width - 2 * radius, y, x + width, y + 2 * radius, start=0, extent=90, fill=bg_color, outline=bg_color))
    parts.append(canvas.create_arc(x, y + height - 2 * radius, x + 2 * radius, y + height, start=180, extent=90, fill=bg_color, outline=bg_color))
    parts.append(canvas.create_arc(x + width - 2 * radius, y + height - 2 * radius, x + width, y + height, start=270, extent=90, fill=bg_color, outline=bg_color))
    parts.append(canvas.create_rectangle(x + radius, y, x + width - radius, y + height, fill=bg_color, outline=bg_color))
    parts.append(canvas.create_rectangle(x, y + radius, x + width, y + height - radius, fill=bg_color, outline=bg_color))

    text_id = canvas.create_text(x + width // 2, y + height // 2, text=text, fill="white", font=("Arial", font_size, "bold"))

    for part in parts:
        canvas.addtag_withtag("btn_part", part)
    canvas.addtag_withtag("btn_text", text_id)

    def on_enter(event):
        for part in parts:
            canvas.itemconfig(part, fill="#FF8866", outline="#FF8866")

    def on_leave(event):
        for part in parts:
            canvas.itemconfig(part, fill=bg_color, outline=bg_color)

    canvas.tag_bind("btn_part", "<Enter>", on_enter)
    canvas.tag_bind("btn_text", "<Enter>", on_enter)
    canvas.tag_bind("btn_part", "<Leave>", on_leave)
    canvas.tag_bind("btn_text", "<Leave>", on_leave)

    canvas.tag_bind("btn_part", "<Button-1>", lambda e: command())
    canvas.tag_bind("btn_text", "<Button-1>", lambda e: command())


class FieldSpec:
    """Definition of a single input field. kind: entry / file / dir / combo / check"""

    def __init__(self, key, label, kind="entry", default="", options=None, width=48, command=None):
        self.key = key
        self.label = label
        self.kind = kind
        self.default = default
        self.options = options or []
        self.width = width
        self.command = command  # kind=="check" only: called (no args) on toggle


class FieldBuilderMixin:
    """Widget-building helpers shared by every GUI frame. Expects self.widgets to be a dict."""

    def section_label(self, parent, row, text):
        tk.Label(parent, text=text, font=("Arial", 14, "bold"), bg=COLOR0, fg=COLOR4).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(16, 4))

    def build_field(self, parent, row, spec: FieldSpec):
        """Grids one field's widgets (label + input [+ Browse button]) at `row`.
        Returns the list of widgets placed, so a caller can later toggle the
        whole row's visibility via grid_remove()/grid() (e.g. a checkbox that
        shows/hides an optional field further down the same screen)."""
        label = tk.Label(parent, text=spec.label, font=("Arial", 13, "bold"), bg=COLOR0,
                fg="#264653", width=22, anchor="e")
        label.grid(row=row, column=0, padx=5, pady=3, sticky="ne")
        row_widgets = [label]

        if spec.kind == "check":
            var = tk.BooleanVar(value=bool(spec.default))
            chk = tk.Checkbutton(parent, variable=var, bg=COLOR0, activebackground=COLOR0,
                                 command=spec.command)
            chk.grid(row=row, column=1, padx=10, pady=3, sticky="w")
            self.widgets[spec.key] = var
            row_widgets.append(chk)
            return row_widgets

        if spec.kind == "combo":
            # Keep a reference to the StringVar on self.widgets (not just as textvariable) --
            # a local-only StringVar can be garbage collected once this method returns,
            # which silently blanks the Combobox's value (a known tkinter pitfall).
            var = tk.StringVar(value=spec.default)
            combo = ttk.Combobox(parent, textvariable=var, values=spec.options,
                                 width=spec.width - 2, state="readonly", font=("Arial", 12))
            combo.grid(row=row, column=1, padx=10, pady=3, sticky="w")
            self.widgets[spec.key] = var
            row_widgets.append(combo)
            return row_widgets

        entry = tk.Entry(parent, fg="black", bg=ENTRY_BG, insertbackground="black",
                         width=spec.width, relief="flat", highlightthickness=0, font=("Arial", 12))
        entry.insert(0, spec.default)
        entry.grid(row=row, column=1, padx=10, pady=3, ipady=2, sticky="w")
        entry.bind("<FocusIn>", self.on_focus_in)
        entry.bind("<FocusOut>", self.on_focus_out)
        self.widgets[spec.key] = entry
        row_widgets.append(entry)

        if spec.kind in ("file", "dir"):
            btn_canvas = tk.Canvas(parent, width=80, height=30, bg=COLOR0, highlightthickness=0)
            btn_canvas.grid(row=row, column=2, padx=5)
            if spec.kind == "file":
                cmd = (lambda e=entry: self.browse_file(e))
            else:
                cmd = (lambda e=entry: self.browse_dir(e))
            draw_rounded_button(btn_canvas, 0, 0, 80, 30, 10, BROWSE_BTN_COLOR, "Browse", cmd)
            self.enable_drop(entry)
            row_widgets.append(btn_canvas)

        return row_widgets

    def set_row_visible(self, row_widgets, visible):
        """Show/hide a row's widgets (as returned by build_field) via
        grid()/grid_remove() -- a hidden row takes up no space, so rows
        below it are unaffected."""
        for w in row_widgets:
            if visible:
                w.grid()
            else:
                w.grid_remove()

    def browse_file(self, entry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def browse_dir(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def enable_drop(self, entry):
        """Let the user drag a file/folder from Finder straight onto the entry,
        as an alternative to the Browse button. No-op if tkinterdnd2 isn't
        available, or if this widget's root wasn't created with TkinterDnD.Tk()
        (the entry just stays Browse-only in either case)."""
        if not DND_AVAILABLE:
            return

        try:
            entry.drop_target_register(DND_FILES)
        except tk.TclError:
            return

        def on_drop_enter(event):
            entry.configure(bg=ENTRY_FOCUS_BG)
            return event.action

        def on_drop_leave(event):
            entry.configure(bg=ENTRY_BG)
            return event.action

        def on_drop(event):
            entry.configure(bg=ENTRY_BG)
            paths = entry.tk.splitlist(event.data)
            if paths:
                entry.delete(0, tk.END)
                entry.insert(0, paths[0])
            return event.action

        entry.dnd_bind("<<DropEnter>>", on_drop_enter)
        entry.dnd_bind("<<DropLeave>>", on_drop_leave)
        entry.dnd_bind("<<Drop>>", on_drop)

    def on_focus_in(self, e):
        e.widget.configure(bg=ENTRY_FOCUS_BG)

    def on_focus_out(self, e):
        e.widget.configure(bg=ENTRY_BG)

    def make_scrollable_frame(self, parent):
        """Build a scrollable canvas+frame inside parent. Returns (outer, inner);
        pack/grid `outer` into parent, place fields into `inner`."""
        outer = tk.Frame(parent, bg=COLOR0)

        canvas = tk.Canvas(outer, bg=COLOR0, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        inner = tk.Frame(canvas, bg=COLOR0)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _wheel, add="+")

        return outer, inner

    def make_scrollable_tab(self, notebook, title):
        outer, inner = self.make_scrollable_frame(notebook)
        notebook.add(outer, text=title)
        return inner

    # ---------------- input parsing helpers ----------------

    def _parse_int(self, raw, label, errors, allow_blank=False):
        raw = (raw or "").strip()
        if raw == "":
            if allow_blank:
                return None
            errors.append(f"{label} is required.")
            return None
        try:
            return int(raw)
        except ValueError:
            errors.append(f"{label} must be an integer (got '{raw}').")
            return None

    def _parse_float(self, raw, label, errors, allow_blank=False):
        raw = (raw or "").strip()
        if raw == "":
            if allow_blank:
                return None
            errors.append(f"{label} is required.")
            return None
        try:
            return float(raw)
        except ValueError:
            errors.append(f"{label} must be a number (got '{raw}').")
            return None

    def _parse_int_list(self, raw, label, errors):
        raw = (raw or "").strip()
        if not raw:
            return []
        try:
            return [int(tok) for tok in raw.split()]
        except ValueError:
            errors.append(f"{label} must be space-separated integers (e.g. '1 2', got '{raw}').")
            return []


class SubprocessRunnerMixin:
    """Streaming subprocess execution + thread-safe UI updates. Expects self.log_text, self.status_label."""

    def stream_subprocess(self, cmd, tag):
        self.log(f"{tag} $ " + " ".join(str(c) for c in cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
        except OSError as e:
            self.log(f"{tag} [ERROR] Failed to launch command: {e}")
            return 1
        for line in proc.stdout:
            self.log(f"{tag} {line.rstrip()}")
        proc.wait()
        self.log(f"{tag} [exit code] {proc.returncode}")
        return proc.returncode

    def log(self, text):
        self.after(0, self._append_log, text)

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def show_error(self, title, msg):
        self.after(0, lambda: messagebox.showerror(title, msg))

    def show_info(self, title, msg):
        self.after(0, lambda: messagebox.showinfo(title, msg))

    def set_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))


def which_default(name):
    import shutil as _shutil
    return _shutil.which(name) or ""

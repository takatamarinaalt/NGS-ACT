#!/usr/bin/env python3

import os
import sys
import tkinter as tk

# htslib（pysam と bcftools が共有するログレベル環境変数）が、コピー等で
# .bai/.fai がBAM/FASTA本体より古くなった時に出す "The index file is older
# than the data file" 警告を抑える。実害の無い警告で、GUIのログにエラーの
# ように紛れて表示されるのを防ぐ。ERRORレベル以上（実際のエラー）は引き続き
# 表示される。子プロセス（sc/ 配下のPythonスクリプト・bcftools）は環境変数を
# 継承するため、ここ一箇所の設定で両方に効く。
os.environ.setdefault("HTS_LOG_LEVEL", "ERROR")

try:
    from tkinterdnd2 import TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    TkinterDnD = None
    DND_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Current working directory:", os.getcwd())

if hasattr(sys, "_MEIPASS"):
    os.chdir(os.path.dirname(sys.executable))  # PyInstaller runtime
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # normal runtime

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

from gui_common import COLOR0, draw_rounded_button
from clustering_gui import ClusteringAllelesFrame, ClusteringOptionsFrame
from determining_gui import DeterminingAllelesFrame


class SelectorFrame(tk.Frame):
    """Landing screen: choose which analysis to run."""

    def __init__(self, master, on_select):
        super().__init__(master, bg=COLOR0)
        self.on_select = on_select
        self.create_widgets()

    def create_widgets(self):
        container = tk.Frame(self, bg=COLOR0)
        container.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(container, text="Select GUI", font=("Helvetica Neue", 22, "bold"),
                fg="#264653", bg=COLOR0).pack(pady=(0, 30))

        btn1 = tk.Canvas(container, width=300, height=70, bg=COLOR0, highlightthickness=0)
        btn1.pack(pady=10)
        draw_rounded_button(btn1, 0, 0, 300, 70, 24, "#FFB38A", "Clustering Alleles",
                            lambda: self.on_select("clustering"), font_size=18)

        btn2 = tk.Canvas(container, width=300, height=70, bg=COLOR0, highlightthickness=0)
        btn2.pack(pady=10)
        draw_rounded_button(btn2, 0, 0, 300, 70, 24, "#FFB38A", "Determining Alleles",
                            lambda: self.on_select("determining"), font_size=18)


class App:
    """Single-root frame router: selector <-> clustering-alleles <-> determining-alleles."""

    def __init__(self, root):
        self.root = root
        self.root.title("NGS_ACT")
        self.root.geometry("1040x880")
        self.root.configure(bg=COLOR0)
        self.current_frame = None
        self.show_selector()

    def _swap(self, frame):
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = frame
        self.current_frame.pack(fill="both", expand=True)

    def show_selector(self):
        self._swap(SelectorFrame(self.root, on_select=self.on_select))

    def on_select(self, choice):
        if choice == "clustering":
            self.show_clustering_options()
        elif choice == "determining":
            self._swap(DeterminingAllelesFrame(self.root, on_back=self.show_selector))

    def show_clustering_options(self):
        self._swap(ClusteringOptionsFrame(self.root, on_back=self.show_selector,
                                          on_next=self.on_clustering_options_chosen))

    def on_clustering_options_chosen(self, two_stage, use_alleles_match):
        self._swap(ClusteringAllelesFrame(self.root, on_back=self.show_clustering_options,
                                          two_stage=two_stage, use_alleles_match=use_alleles_match))


def main():
    # TkinterDnD.Tk() is a drop-in replacement for tk.Tk() that additionally lets
    # Entry widgets accept OS-level drag-and-drop (see gui_common.enable_drop()).
    # Falls back to a plain Tk root -- Browse-button-only -- if tkinterdnd2 isn't installed.
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    if not DND_AVAILABLE:
        print("[WARN] tkinterdnd2 not installed -- drag & drop disabled, use the Browse buttons instead.")
    try:
        root.iconbitmap("icon.ico")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

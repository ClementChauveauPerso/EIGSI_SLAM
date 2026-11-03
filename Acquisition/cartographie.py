# cartographie.py — Aligne des scans LD06 et construit une carte globale.
#
# Interface :
#   - « Charger un scan… » : choisit un ou plusieurs CSV ;
#   - « Ouvrir l'assemblage… » : recharge un JSON (scans + poses) ;
#   - panneau de gauche : liste des scans chargés (dossier/fichier), clic pour sélectionner ;
#   - glisser (gauche) : déplacer le scan sélectionné ;
#   - glisser (droit), curseur θ, A/D ou flèches : faire tourner le scan ;
#   - curseurs x / y : réglage fin de la position ;
#   - à droite : carte d'occupation recalculée en direct (résolution réglable) ;
#   - « Sauver l'assemblage… » : JSON listant chaque scan avec sa position/rotation
#     (x, y en mètres, θ en degrés) — sans fusionner les points ;
#   - « Exporter la carte… » : fusionne les scans dans Data\Cartes\<nom>\
#     (carte.csv, occupation.csv) puis appelle visualisation.py pour les rendus PNG.
#
# Commandes clavier :
#   ← → ↑ ↓  déplacer (Maj ×10, Ctrl ÷4)      A / D  tourner (Maj ×10, Ctrl ÷10)
#   N / P, 1–9  changer de scan               Suppr  retirer le scan
#   E  exporter   S  sauver   O  ouvrir       F  recadrer     Q  quitter
#
# Usage : python cartographie.py [dossier]
# Dépendances : matplotlib, numpy

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.widgets import Button, Slider

try:  # module voisin (même dossier)
    import occupation as occ
except ImportError:  # importé comme paquet
    from . import occupation as occ

DEFAULT_ROOT = Path(r"C:\Users\Clement\Downloads\SLAM\Data\Scans")
CARTES = DEFAULT_ROOT.parent / "Cartes"
STEP_XY, STEP_TH = 0.02, 1.0  # pas fins : 2 cm et 1°
COLORS = plt.get_cmap("tab20").colors
BG, PANEL, GRID = "#ffffff", "#fbfbfd", "#e7e9ef"
TEXT, MUTED, AXIS = "#1f2430", "#6b7280", "#9aa1ad"
ACCENT, HILITE = "#3b5bdb", "#e9eefb"
OPEN_AFTER_EXPORT = True  # ouvre carte.png dans la visionneuse après l'export


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_scan(path):
    """(x, y) en mètres et intensité des points d'un scan, sans les distances nulles."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3),
                      ndmin=2, encoding="utf-8-sig")
    angle, dist, inten = data.T
    keep = dist > 0
    r, th = dist[keep] / 1000.0, np.deg2rad(angle[keep])
    return np.column_stack((r * np.cos(th), r * np.sin(th))), inten[keep]


def transform(pts, pose):
    """Points du repère lidar vers le repère global (rotation θ puis translation)."""
    if not len(pts):
        return pts
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pts @ np.array(((c, s), (-s, c))) + pose[:2]


def wrap_angle(angle):
    """Ramène un angle dans ]-180°, 180°] (pour rester dans la plage du curseur)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(eq=False)  # identité : évite toute comparaison de tableaux numpy
class Scan:
    path: Path
    pts: np.ndarray
    inten: np.ndarray
    pose: np.ndarray  # x (m), y (m), θ (rad)
    color: tuple
    cloud: object = None
    label: object = None


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class Carte:
    def __init__(self, root):
        self.root = Path(root)
        self.cartes = CARTES
        self.visualisation = Path(__file__).resolve().with_name("visualisation.py")
        self.scans = []
        self.active = -1
        self.scroll = 0
        self.rows = []
        self.visible_rows = 10
        self.drag = None
        self.naming = None
        self.notice = None
        self.syncing = False
        self.grid = None
        self.grid_res = 0.10
        self.grid_dirty = True
        self.grid_stamp = 0.0
        self.grid_timer = None
        self.last_dir = self.root if self.root.is_dir() else Path.home()

        self._build_ui()
        self.refresh()
        self.update_grid()

    # -- interface ---------------------------------------------------------- #
    def _build_ui(self):
        for name in ("save", "fullscreen", "home", "back", "forward", "pan",
                     "zoom", "xscale", "yscale", "all_axes", "grid_major",
                     "grid_minor"):
            try:
                plt.rcParams[f"keymap.{name}"] = []
            except KeyError:  # paramètre absent selon la version de matplotlib
                pass

        self.fig = plt.figure(figsize=(14.2, 9.8))
        self.fig.canvas.manager.set_window_title("Cartographie — assemblage de scans")
        self.fig.patch.set_facecolor(BG)

        # Vue d'assemblage (centre) et carte d'occupation (droite) : même taille.
        self.ax = self.fig.add_axes((0.200, 0.415, 0.365, 0.515))
        self.ax.set_facecolor(PANEL)
        self.ax.set_aspect("equal")
        self.ax.grid(True, color=GRID, lw=0.8, zorder=0)
        self.ax.tick_params(colors=AXIS, labelsize=8, length=3)
        for spine in self.ax.spines.values():
            spine.set_color(GRID)
        self.ax.set_xlabel("x (m)", color=MUTED, fontsize=9)
        self.ax.set_ylabel("y (m)", color=MUTED, fontsize=9)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.hint = self.ax.text(0.5, 0.5, "Aucun scan chargé\n\nCliquez sur « Charger un scan… »",
                                 transform=self.ax.transAxes, ha="center", va="center",
                                 fontsize=11, color=MUTED, linespacing=1.6)
        self.title = self.ax.set_title("Carte", fontsize=12, fontweight="bold",
                                       color=TEXT, pad=10)
        self.status = self.fig.text(0.3825, 0.378, "", ha="center", va="center",
                                    fontsize=9, color=TEXT)
        self.fig.text(0.5, 0.055,
                      "glisser G : déplacer     glisser D : tourner     curseurs : réglage fin     "
                      "flèches / A-D : pas fins (Maj ×10, Ctrl ÷10)     Suppr : retirer     Q : quitter",
                      ha="center", va="center", fontsize=8, color=MUTED)

        # Origines et caps des scans (le trait épais = scan sélectionné).
        self.origins = self.ax.scatter([], [], marker="o", s=46, linewidths=1.4,
                                       facecolors="none", edgecolors=AXIS, zorder=5)
        self.heads, = self.ax.plot([], [], color=AXIS, lw=0.9, alpha=0.8, zorder=2)
        self.heading, = self.ax.plot([], [], color=ACCENT, lw=1.8, zorder=6)

        # Panneau de gauche : liste des scans.
        self.panel = self.fig.add_axes((0.010, 0.365, 0.175, 0.565))
        self.panel.set_axis_off()
        self.panel.set_xlim(0, 1)
        self.panel.set_ylim(0, 1)

        # Boutons.
        def make_button(rect, label, callback):
            ax = self.fig.add_axes(rect)
            btn = Button(ax, label, color="#f2f4f8", hovercolor=HILITE)
            btn.label.set_fontsize(9)
            btn.label.set_color(TEXT)
            btn.on_clicked(callback)
            return btn

        self.btn_load = make_button((0.010, 0.300, 0.175, 0.040),
                                    "Charger un scan…", lambda _: self.ask_load())
        self.btn_open = make_button((0.010, 0.250, 0.175, 0.040),
                                    "Ouvrir l'assemblage…", lambda _: self.ask_assembly())
        self.btn_save = make_button((0.010, 0.200, 0.175, 0.040),
                                    "Sauver l'assemblage…", lambda _: self.save_assembly())
        self.btn_export = make_button((0.010, 0.150, 0.175, 0.040),
                                      "Exporter la carte…", lambda _: self.begin_export())
        self.btn_fit = make_button((0.010, 0.100, 0.175, 0.040),
                                   "Recadrer la vue", lambda _: self.fit())

        # Curseurs du scan sélectionné (sous la vue d'assemblage uniquement).
        self.sliders = {}
        specs = (("x", "x (m)", -40.0, 40.0, 0.01, "%+.2f"),
                 ("y", "y (m)", -40.0, 40.0, 0.01, "%+.2f"),
                 ("th", "θ (°)", -180.0, 180.0, 0.1, "%+.1f"))
        for i, (key, label, lo, hi, step, fmt) in enumerate(specs):
            ax = self.fig.add_axes((0.255, 0.300 - i * 0.06, 0.30, 0.024))
            slider = Slider(ax, label, lo, hi, valinit=0, valstep=step,
                            valfmt=fmt, color=AXIS)
            slider.label.set_fontsize(9)
            slider.label.set_color(MUTED)
            slider.valtext.set_fontsize(9)
            slider.valtext.set_color(TEXT)
            slider.on_changed(lambda v, k=key: self.on_slider(k, v))
            self.sliders[key] = slider

        # Carte d'occupation : même cadre que la vue d'assemblage.
        self.grid_ax = self.fig.add_axes((0.615, 0.415, 0.365, 0.515))
        self.grid_ax.set_facecolor("#f3f4f7")
        self.grid_ax.set_aspect("equal")
        self.grid_ax.set_xticks([])
        self.grid_ax.set_yticks([])
        for spine in self.grid_ax.spines.values():
            spine.set_color(GRID)
        self.grid_img = self.grid_ax.imshow(np.zeros((2, 2, 4)), origin="lower",
                                            extent=(-1, 1, -1, 1),
                                            interpolation="nearest", zorder=2)
        self.grid_src, = self.grid_ax.plot([], [], "x", ms=6, mew=1.3,
                                           color="#e11d48", zorder=4)
        self.grid_ax.set_title("Carte d'occupation", fontsize=12,
                               fontweight="bold", color=TEXT, pad=10)
        handles = [
            Patch(facecolor="#ffffff", edgecolor=GRID, label="libre"),
            Patch(facecolor=tuple(float(c) for c in occ.DARK), label="occupé"),
            Patch(facecolor=tuple(float(c) for c in occ.UNKNOWN_RGB),
                  edgecolor=GRID, label="inconnu"),
            Line2D([], [], marker="x", ls="", color="#e11d48", label="lidar"),
        ]
        self.grid_ax.legend(handles=handles, loc="upper center",
                            bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False,
                            fontsize=8, handlelength=1.1, columnspacing=1.4,
                            handletextpad=0.4, borderaxespad=0.0)
        self.grid_info = self.fig.text(0.7975, 0.300, "aucun scan chargé",
                                       ha="center", va="center", fontsize=8,
                                       color=MUTED)
        res_ax = self.fig.add_axes((0.685, 0.180, 0.195, 0.024))
        self.res_slider = Slider(res_ax, "résolution (m)", 0.02, 0.30,
                                 valinit=self.grid_res, valstep=0.01,
                                 valfmt="%.2f", color=AXIS)
        self.res_slider.label.set_fontsize(9)
        self.res_slider.label.set_color(MUTED)
        self.res_slider.valtext.set_fontsize(9)
        self.res_slider.valtext.set_color(TEXT)
        self.res_slider.on_changed(self.on_grid_res)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        # Recalcul de la carte d'occupation ~5 fois/s (jamais pendant un rendu).
        try:
            if plt.get_backend().lower() != "agg":
                self.grid_timer = self.fig.canvas.new_timer(interval=200)
                self.grid_timer.add_callback(self.update_grid)
                self.grid_timer.start()
        except Exception:
            self.grid_timer = None

    # -- chargement --------------------------------------------------------- #
    def ask_load(self):
        """Ouvre un sélecteur de fichiers (un ou plusieurs CSV)."""
        files = self._dialog("open", title="Charger un ou plusieurs scans",
                             initialdir=str(self.last_dir),
                             filetypes=[("Scans lidar", "*.csv"),
                                        ("Tous les fichiers", "*.*")])
        if files:
            self.load_paths(files)

    def ask_assembly(self):
        """Ouvre un assemblage JSON (scans + positions/rotations)."""
        files = self._dialog("open", title="Ouvrir un assemblage",
                             initialdir=str(self.last_dir),
                             filetypes=[("Assemblage", "*.json"),
                                        ("Tous les fichiers", "*.*")])
        if files:
            self.load_paths(files)

    def _dialog(self, mode, **kw):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            print(f"Sélecteur de fichiers indisponible : {exc}")
            return [] if mode == "open" else ""
        # Sous TkAgg on réutilise la fenêtre de la figure (évite un 2e interpréteur Tk).
        root, own = None, True
        try:
            win = self.fig.canvas.manager.window
            if win is not None and hasattr(win, "winfo_exists") and win.winfo_exists():
                root, own = win, False
        except Exception:
            root = None
        if own:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
        try:
            if mode == "open":
                return list(filedialog.askopenfilenames(parent=root, **kw))
            return filedialog.asksaveasfilename(parent=root, **kw)
        finally:
            if own:
                root.destroy()

    def load_paths(self, paths):
        """Charge des CSV (un par scan) ou un assemblage JSON."""
        before = len(self.scans)
        for item in paths:
            path = Path(item)
            if not path.exists():
                print(f"  introuvable : {path}")
                continue
            self.last_dir = path.parent
            if path.suffix.lower() == ".json":
                self.load_assembly(path)
            else:
                if any(s.path == path for s in self.scans):
                    print(f"  déjà chargé : {path.name}")
                    continue
                try:
                    self.add_scan(path)
                except Exception as exc:
                    print(f"  échec {path.name} : {exc}")
        if len(self.scans) > before:
            self.fit()
            self.refresh()

    def add_scan(self, path, pose=None):
        pts, inten = load_scan(path)
        if pose is None:  # nouveau scan : posé à côté du précédent
            pose = (1.5 * len(self.scans), 0.0, 0.0)
        pose = np.array([pose[0], pose[1], math.radians(pose[2])], dtype=float)
        color = COLORS[len(self.scans) % len(COLORS)]
        cloud = self.ax.scatter([], [], s=10, c=[color], linewidths=0, zorder=3)
        label = self.ax.text(0, 0, "", fontsize=9, color=color, ha="left",
                             va="bottom", zorder=6)
        self.scans.append(Scan(path, pts, inten, pose, color, cloud, label))
        self.active = len(self.scans) - 1
        self.keep_visible()
        print(f"  + {path.name} ({len(pts)} points)")
        self.draw_list()
        self.refresh()

    def load_assembly(self, path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            entries = data.get("scans", data if isinstance(data, list) else [])
        except Exception as exc:
            print(f"  assemblage illisible {path.name} : {exc}")
            return
        loaded = 0
        for entry in entries:
            file = Path(str(entry.get("file", "")))
            if not file.exists():
                print(f"  scan absent : {file}")
                continue
            if any(s.path == file for s in self.scans):
                continue
            pose = (float(entry.get("x", 0.0)), float(entry.get("y", 0.0)),
                    float(entry.get("theta_deg", 0.0)))
            self.add_scan(file, pose)
            loaded += 1
        if loaded:
            self.fit()
            self.notice = f"Assemblage chargé : {loaded} scan(s) depuis {path.name}"

    # -- liste des scans ---------------------------------------------------- #
    @staticmethod
    def short_name(scan):
        """« bas/20261103_120727.csv » : dossier parent + fichier."""
        return f"{scan.path.parent.name}/{scan.path.name}"

    def draw_list(self):
        ax = self.panel
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.axvline(1.0, color=GRID, lw=1)
        ax.text(0.02, 0.985, "Scans chargés", fontsize=10, fontweight="bold",
                color=TEXT, va="top")
        ax.text(0.02, 0.935, f"{len(self.scans)} scan(s)", fontsize=8,
                color=MUTED, va="top")
        self.rows = []
        top, height = 0.885, 0.058
        self.visible_rows = max(int((top - 0.02) / height), 1)
        if not self.scans:
            ax.text(0.02, top, "aucun", fontsize=8.5, color=MUTED, va="center")
            return
        end = min(len(self.scans), self.scroll + self.visible_rows)
        for k, i in enumerate(range(self.scroll, end)):
            scan = self.scans[i]
            y = top - k * height
            if i == self.active:
                ax.add_patch(Rectangle((0.0, y - 0.030), 1.0, 0.056,
                                       color=HILITE, lw=0))
            ax.add_patch(Rectangle((0.03, y - 0.014), 0.045, 0.030,
                                   color=scan.color, lw=0))
            name = self.short_name(scan)
            if len(name) > 32:
                name = name[:29] + "…"
            ax.text(0.10, y, f"{i + 1}. {name}", fontsize=8, va="center", color=TEXT)
            self.rows.append((y - 0.030, y + 0.030, i))
        if len(self.scans) > self.visible_rows:  # ascenseur
            frac = self.visible_rows / len(self.scans)
            y0 = 0.02 + (1 - frac) * (1 - self.scroll / max(len(self.scans) - self.visible_rows, 1))
            ax.add_patch(Rectangle((0.975, y0), 0.02, frac * 0.9,
                                   color=GRID, lw=0))

    def keep_visible(self):
        if self.active < self.scroll:
            self.scroll = self.active
        elif self.active >= self.scroll + self.visible_rows:
            self.scroll = self.active - self.visible_rows + 1

    # -- affichage ---------------------------------------------------------- #
    def draw(self, scan):
        scan.cloud.set_offsets(transform(scan.pts, scan.pose))
        active = 0 <= self.active < len(self.scans) and self.scans[self.active] is scan
        scan.cloud.set_sizes([16 if active else 10])
        scan.cloud.set_alpha(0.95 if active else 0.38)
        scan.cloud.set_zorder(4 if active else 3)
        # numéro du scan à côté de son point d'origine
        scan.label.set_position((scan.pose[0] + 0.12, scan.pose[1] + 0.12))
        scan.label.set_text(str(self.scans.index(scan) + 1))
        scan.label.set_color(scan.color)
        scan.label.set_fontsize(10 if active else 8)
        scan.label.set_fontweight("bold" if active else "normal")

    def refresh(self):
        for scan in self.scans:
            self.draw(scan)
        if self.scans:  # origine de chaque scan + petit cap
            self.origins.set_offsets(np.array([s.pose[:2] for s in self.scans]))
            self.origins.set_edgecolors([s.color for s in self.scans])
            self.origins.set_sizes([80 if i == self.active else 44
                                    for i in range(len(self.scans))])
            xs, ys = [], []
            for s in self.scans:
                x, y, th = s.pose
                xs += [x, x + 0.35 * math.cos(th), float("nan")]
                ys += [y, y + 0.35 * math.sin(th), float("nan")]
            self.heads.set_data(xs, ys)
        else:
            self.origins.set_offsets(np.empty((0, 2)))
            self.heads.set_data([], [])
        if 0 <= self.active < len(self.scans):
            s = self.scans[self.active]
            x, y = s.pose[0], s.pose[1]
            self.heading.set_data((x, x + 0.6 * math.cos(s.pose[2])),
                                  (y, y + 0.6 * math.sin(s.pose[2])))
            self.status.set_text(
                f"{self.active + 1}/{len(self.scans)}   {self.short_name(s)}   "
                f"x = {s.pose[0]:+.2f} m   y = {s.pose[1]:+.2f} m   "
                f"θ = {np.rad2deg(s.pose[2]):+.1f}°   ({len(s.pts)} points)")
        else:
            self.heading.set_data([], [])
            self.status.set_text(
                self.notice or "Chargez un scan avec « Charger un scan… » "
                "pour commencer l'assemblage.")
        if self.notice and self.scans:
            self.status.set_text(self.notice)
        total = sum(len(s.pts) for s in self.scans)
        self.title.set_text(f"Carte — {len(self.scans)} scan(s), {total} points")
        self.hint.set_visible(not self.scans)
        self.update_sliders()
        self.grid_dirty = True
        self.grid_stamp = time.monotonic()
        self.fig.canvas.draw_idle()

    def update_sliders(self):
        scan = self.scans[self.active] if 0 <= self.active < len(self.scans) else None
        values = ((scan.pose[0] if scan else 0.0), (scan.pose[1] if scan else 0.0),
                  (np.rad2deg(scan.pose[2]) if scan else 0.0))
        self.syncing = True
        for key, value in zip(("x", "y", "th"), values):
            slider = self.sliders[key]
            slider.set_val(min(max(value, slider.valmin), slider.valmax))
        self.syncing = False

    def fit(self):
        pts = [transform(s.pts, s.pose) for s in self.scans if len(s.pts)]
        if not pts:
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.fig.canvas.draw_idle()
            return
        allp = np.vstack(pts)
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
        center = (lo + hi) / 2
        span = max(float((hi - lo).max()), 1.0) * 1.25
        self.ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        self.ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        self.fig.canvas.draw_idle()

    # -- carte d'occupation ------------------------------------------------- #
    def on_grid_res(self, value):
        self.grid_res = float(value)
        self.grid_dirty = True
        if self.grid_timer is None:  # pas de boucle d'événements (mode Agg) : direct
            self.update_grid()

    def update_grid(self, *_args, force=False):
        """(Re)construit la grille d'occupation à partir des scans assemblés."""
        if not self.grid_dirty:
            return
        # Pendant un geste (glisser, curseur) on attend une courte pause : la
        # grille se met à jour ~0,35 s après la dernière modification.
        if (not force and self.grid_timer is not None
                and time.monotonic() - self.grid_stamp < 0.35):
            return
        self.grid_dirty = False
        rays = [(transform(s.pts, s.pose), s.pose[:2])
                for s in self.scans if len(s.pts)]
        grille = occ.build(rays, self.grid_res) if rays else None
        self.grid = grille
        if grille is None:
            self.grid_img.set_data(np.zeros((2, 2, 4)))
            self.grid_img.set_extent((-1.0, 1.0, -1.0, 1.0))
            self.grid_ax.set_xlim(-1, 1)
            self.grid_ax.set_ylim(-1, 1)
            self.grid_src.set_data([], [])
            self.grid_info.set_text("aucun scan chargé")
        else:
            self.grid_img.set_data(occ.to_rgb(grille))
            x0, x1, y0, y1 = grille.extent()
            self.grid_img.set_extent((x0, x1, y0, y1))
            self.grid_ax.set_xlim(x0, x1)
            self.grid_ax.set_ylim(y0, y1)
            if len(grille.sources):
                self.grid_src.set_data(grille.sources[:, 0], grille.sources[:, 1])
            else:
                self.grid_src.set_data([], [])
            self.grid_info.set_text(f"{grille.nx} × {grille.ny} cases  •  "
                                    f"{grille.res * 100:.0f} cm")
        self.fig.canvas.draw_idle()

    def on_close(self, _event):
        if self.grid_timer is not None:
            try:
                self.grid_timer.stop()
            except Exception:
                pass

    # -- édition ------------------------------------------------------------ #
    def on_slider(self, key, value):
        if self.syncing or not (0 <= self.active < len(self.scans)):
            return
        scan = self.scans[self.active]
        self.notice = None
        if key == "x":
            scan.pose[0] = float(value)
        elif key == "y":
            scan.pose[1] = float(value)
        else:
            scan.pose[2] = math.radians(float(value))
        self.refresh()

    def move(self, dx, dy):
        if not (0 <= self.active < len(self.scans)):
            return
        self.notice = None
        self.scans[self.active].pose[:2] += (dx, dy)
        self.refresh()

    def rotate(self, deg):
        if not (0 <= self.active < len(self.scans)):
            return
        self.notice = None
        scan = self.scans[self.active]
        scan.pose[2] = wrap_angle(scan.pose[2] + math.radians(deg))
        self.refresh()

    def select(self, index):
        if not self.scans:
            return
        self.notice = None
        self.active = max(0, min(index, len(self.scans) - 1))
        self.keep_visible()
        self.draw_list()
        self.refresh()

    def remove_active(self):
        if not (0 <= self.active < len(self.scans)):
            return
        scan = self.scans.pop(self.active)
        scan.cloud.remove()
        scan.label.remove()
        self.active = min(self.active, len(self.scans) - 1)
        self.keep_visible()
        self.notice = None
        self.draw_list()
        self.refresh()

    # -- sauvegarde de l'assemblage ----------------------------------------- #
    def save_assembly(self, path=None):
        if not self.scans:
            self.notice = "Rien à sauvegarder : aucun scan chargé."
            self.refresh()
            return
        if path is None:
            path = self._dialog("save", title="Sauver l'assemblage",
                                initialdir=str(self.last_dir),
                                initialfile="assemblage.json",
                                defaultextension=".json",
                                filetypes=[("Assemblage", "*.json")])
            if not path:
                return
        data = {"version": 1, "scans": [
            {"file": s.path.as_posix(),
             "x": round(float(s.pose[0]), 4),
             "y": round(float(s.pose[1]), 4),
             "theta_deg": round(float(np.rad2deg(s.pose[2])), 3)}
            for s in self.scans]}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
        print(f"Assemblage sauvegardé : {path}")
        self.notice = f"Assemblage sauvegardé : {Path(path).name}"
        self.refresh()

    # -- export de la carte fusionnée --------------------------------------- #
    @staticmethod
    def open_image(path):
        """Ouvre le rendu dans la visionneuse par défaut (Windows / macOS / Linux)."""
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            print(f"  ouverture automatique impossible : {exc}")

    def next_name(self):
        used = set()
        if self.cartes.is_dir():
            used = {int(p.name) for p in self.cartes.iterdir()
                    if p.is_dir() and p.name.isdigit()}
        return str(next(i for i in range(1, len(used) + 2) if i not in used))

    @staticmethod
    def clean_name(name):
        name = "".join(c for c in name.strip() if c not in '\\/:*?"<>|')
        return name.strip(" .")

    def begin_export(self):
        self.naming = ""
        self.show_naming()

    def show_naming(self):
        hint = self.next_name()
        self.sliders["th"].valtext.set_text("")
        self.status.set_text(
            f"Nom de la carte : {self.naming}_     (Entrée = exporter"
            f"{f' « {hint} »' if not self.naming else ''}, Échap = annuler)")
        self.fig.canvas.draw_idle()

    def export(self, name):
        """Écrit Data\\Cartes\\<nom>\\carte.csv puis appelle visualisation.py."""
        usable = [s for s in self.scans if len(s.pts)]
        if not usable:
            self.notice = "Rien à exporter : aucun scan chargé."
            self.refresh()
            return
        name = self.clean_name(name) or self.next_name()
        folder = self.cartes / name
        folder.mkdir(parents=True, exist_ok=True)
        pts = np.vstack([transform(s.pts, s.pose) for s in usable])
        inten = np.concatenate([s.inten for s in usable])
        table = np.column_stack((np.arange(len(pts)),
                                 np.round(pts * 1000),
                                 np.clip(np.round(inten), 0, 255))).astype(np.int64)
        csv = folder / "carte.csv"
        with open(csv, "w", encoding="utf-8", newline="") as f:
            f.write("N,x(mm),y(mm),Intensity\n")
            np.savetxt(f, table, fmt="%d", delimiter=",")
        print(f"Carte « {name} » ({len(pts)} points) : {csv}")

        # Carte d'occupation : même dossier, rendue en PNG par visualisation.py.
        grille = occ.build([(transform(s.pts, s.pose), s.pose[:2]) for s in usable],
                           self.grid_res)
        occ_csv = None
        if grille is not None:
            occ_csv = folder / "occupation.csv"
            occ.save_csv(grille, occ_csv)
            print(f"Occupation ({grille.nx} × {grille.ny} cases à "
                  f"{grille.res * 100:.0f} cm) : {occ_csv}")

        png = folder / "carte.png"
        occ_png = folder / "occupation.png"
        self.notice = f"Carte « {name} » exportée — rendu en cours…"
        self.refresh()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            done = subprocess.run(
                [sys.executable, str(self.visualisation), str(folder), "--force"],
                capture_output=True, text=True, timeout=180)
            detail = (done.stderr or done.stdout).strip()
        except Exception as exc:
            detail = str(exc)
        if png.exists():
            print(f"Rendu : {png}")
            if occ_csv is not None and occ_png.exists():
                print(f"Rendu : {occ_png}")
                self.notice = (f"Carte « {name} » exportée : carte.csv, "
                               "occupation.csv et rendus PNG")
            else:
                self.notice = f"Carte « {name} » exportée : carte.csv + carte.png"
            if OPEN_AFTER_EXPORT:
                self.open_image(png)
        else:
            print(f"  échec du rendu : {detail}")
            self.notice = (f"Carte « {name} » exportée, mais le rendu a échoué "
                           "(voir la console).")
        self.refresh()

    # -- événements --------------------------------------------------------- #
    def on_key(self, event):
        if event.key is None:
            return
        if self.naming is not None:  # saisie du nom de carte
            self.on_name_key(event)
            return
        key = event.key
        parts = key.split("+")
        name = parts[-1].lower()
        shift = "shift" in parts or (len(key) == 1 and key.isupper())
        ctrl = "ctrl" in parts or "control" in parts
        step = STEP_XY * (10 if shift else 0.25 if ctrl else 1)
        turn = STEP_TH * (10 if shift else 0.1 if ctrl else 1)

        if name in ("q", "escape"):
            plt.close(self.fig)
            return
        moves = {"left": (-step, 0), "right": (step, 0),
                 "down": (0, -step), "up": (0, step)}
        if name in moves:
            self.move(*moves[name])
        elif name in ("a", ","):
            self.rotate(-turn)
        elif name in ("d", "."):
            self.rotate(turn)
        elif name in ("n", "pagedown"):
            self.select(self.active + 1)
        elif name in ("p", "pageup"):
            self.select(self.active - 1)
        elif len(name) == 1 and name in "123456789" and int(name) <= len(self.scans):
            self.select(int(name) - 1)
        elif name == "delete":
            self.remove_active()
        elif name == "f":
            self.fit()
        elif name == "e":
            self.begin_export()
        elif name == "s":
            self.save_assembly()
        elif name == "o":
            self.ask_assembly()

    def on_name_key(self, event):
        key = event.key
        if key in ("enter", "return"):
            typed, self.naming = self.naming, None
            self.update_sliders()
            self.export(typed)
        elif key == "escape":
            self.naming = None
            self.refresh()
        elif key == "backspace":
            self.naming = self.naming[:-1]
            self.show_naming()
        elif len(key) == 1 and key.isprintable():
            self.naming += key
            self.show_naming()

    def on_press(self, event):
        if event.inaxes is self.panel and event.button == 1:
            self.click_list(event)
            return
        if self.naming is not None or event.inaxes is not self.ax:
            return
        if not (0 <= self.active < len(self.scans)):
            return
        self.drag = {"button": event.button, "x": event.xdata, "y": event.ydata,
                     "pix": event.x, "pixy": event.y,
                     "pose": self.scans[self.active].pose.copy(),
                     "xlim": self.ax.get_xlim(), "ylim": self.ax.get_ylim()}

    def click_list(self, event):
        if event.ydata is None:
            return
        for y0, y1, index in self.rows:
            if y0 <= event.ydata <= y1:
                self.select(index)
                return

    def on_release(self, _event):
        if self.drag is not None:
            self.notice = None
            self.refresh()
        self.drag = None

    def on_motion(self, event):
        if self.drag is None or not (0 <= self.active < len(self.scans)):
            return
        button = self.drag["button"]
        pose = self.drag["pose"].copy()
        scan = self.scans[self.active]
        if button == 1 and event.xdata is not None:
            pose[0] += event.xdata - self.drag["x"]
            pose[1] += event.ydata - self.drag["y"]
            scan.pose = pose
            self.notice = None
            self.refresh()
        elif button == 3:  # rotation : 0,15°/pixel (Maj = 0,03°/pixel)
            speed = 0.03 if "shift" in (event.key or "") else 0.15
            pose[2] = wrap_angle(pose[2] - math.radians(speed * (event.x - self.drag["pix"])))
            scan.pose = pose
            self.notice = None
            self.refresh()
        elif button == 2:  # panoramique de la vue
            w, h = self.ax.bbox.width, self.ax.bbox.height
            kx = (self.drag["xlim"][1] - self.drag["xlim"][0]) / max(w, 1)
            ky = (self.drag["ylim"][1] - self.drag["ylim"][0]) / max(h, 1)
            dx = (event.x - self.drag["pix"]) * kx
            dy = (event.y - self.drag["pixy"]) * ky
            self.ax.set_xlim(self.drag["xlim"][0] - dx, self.drag["xlim"][1] - dx)
            self.ax.set_ylim(self.drag["ylim"][0] - dy, self.drag["ylim"][1] - dy)
            self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes is self.panel:  # défilement de la liste
            step = -1 if event.button == "up" else 1
            limit = max(len(self.scans) - self.visible_rows, 0)
            self.scroll = max(0, min(self.scroll + step, limit))
            self.draw_list()
            self.fig.canvas.draw_idle()
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 0.85 if event.button == "up" else 1 / 0.85
        x, y = event.xdata, event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(x + (x0 - x) * factor, x + (x1 - x) * factor)
        self.ax.set_ylim(y + (y0 - y) * factor, y + (y1 - y) * factor)
        self.fig.canvas.draw_idle()

    def run(self):
        print("Choisissez « Charger un scan… », puis sélectionnez un scan dans la liste.")
        plt.show()


def main():
    try:  # console Windows : remplace les caractères non affichables (θ…) au lieu de planter
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    Carte(root).run()


if __name__ == "__main__":
    main()

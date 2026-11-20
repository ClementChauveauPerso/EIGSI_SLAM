# localisation.py — Interface de localisation par scan matching (TP jour 2).
#
# Interface (même style que cartographie.py) :
#   - « Charger une carte… » : un dossier exporté par cartographie
#     (assemblage.json -> reconstruit la grille d'occupation et la carte
#     vectorielle, comme les poses de l'assemblage servent de vérité terrain) ;
#   - scan à localiser : un des scans de la carte (curseur « scan # »), sa pose
#     d'assemblage = vérité terrain ;
#   - « Perturber la pose » : estimation initiale = vérité + bruit réglable
#   - référence : grille / vectorielle / scan (nuage) — deux sources possibles :
#     « Charger une carte… » (dossier exporté : assemblage.json -> grille +
#     vectorielle, poses d'assemblage = vérité terrain) ou « Charger un scan
#     référence… » (un CSV devient la référence : mini-carte + nuage posés à
#     l'identité, puis « Charger les scans à localiser… » pour les CSV cibles ;
#     la vérité terrain est la pose relative retrouvée dans les assemblages
#     connus si possible, sinon identité sans vérité) ;
#   - « Perturber la pose » : estimation initiale = vérité + bruit réglable
#     (m, °) — l'algorithme doit la ramener vers la vérité ;
#   - « Lancer » puis vue 2 bascule score / probabilité : la carte de
#     probabilité d'emplacement P(x, y) (softmax du score, curseur « croyance »,
#     ou densité des particules pour la méthode probabiliste) et l'incertitude
#     σx, σy dans les diagnostics ;
#   - vue 1 : la carte, le scan à la vérité (gris), le scan à l'estimation
#     (rouge), la trajectoire des itérations, les particules finales ;
#   - vue 2 : paysage du score (x, y) à θ* — marqueurs vérité / initial /
#     résultat, chemin des itérations ;
#   - vue 3 : score et erreur par itération + chiffres (pose, erreurs, évals,
#     temps) — l'évaluation chiffrée du TP ;
#   - « Benchmark » : N essais (mêmes perturbations pour toutes les méthodes)
#     -> tableau erreurs / temps par méthode et référence.
#
# Commandes clavier : C charger  L lancer  P perturber  B benchmark
#                     F recadrer  S sauvegarder scans  Q quitter
# Usage : python localisation.py [dossier_de_carte]
# Dépendances : matplotlib, numpy

import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import Button, RadioButtons, Slider

try:  # modules voisins (même dossier)
    import cartographie as carto
    import scan_matching as sm
    from algo_base import (charger_carte, pose_relative, score_grille,
                           score_nuage, score_segments, VERITES_RELATIVES)
    import occupation as occ
    import vectorielle as vec
except ImportError:  # importé comme paquet
    from . import cartographie as carto
    from . import scan_matching as sm
    from .algo_base import (charger_carte, pose_relative, score_grille,
                            score_nuage, score_segments, VERITES_RELATIVES)
    from . import occupation as occ
    from . import vectorielle as vec

CARTES = carto.CARTES
BG, PANEL, GRID = carto.BG, carto.PANEL, carto.GRID
TEXT, MUTED, AXIS = carto.TEXT, carto.MUTED, carto.AXIS
ACCENT, HILITE = carto.ACCENT, carto.HILITE
ROUGE, VERT, GRIS = "#e11d48", "#15803d", "#9aa1ad"
SEUIL_ERR, SEUIL_DTH = 0.05, 2.0        # « convergé » : 5 cm et 2°


def _flou(a, passes=2):
    """Petit flou 3×3 (lissage de la densité de particules)."""
    for _ in range(passes):
        p = np.pad(a, 1, mode="constant")
        a = sum(p[i:i + a.shape[0], j:j + a.shape[1]]
                for i in range(3) for j in range(3)) / 9.0
    return a


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class Localisation:
    def __init__(self, dossier=None):
        self.essais = 8                 # nb d'essais du benchmark
        self.dossier = None
        self.scans = []                 # [(chemin, points, pose vérité), …]
        self.grille = None
        self.vect = None
        self.nuage = None               # scan référence (mode scan-contre-scan)
        self._verites = None            # cache {chemin: pose} des assemblages connus
        self._verites_carte = {}        # poses de la carte chargée (mode carte)
        self.scan_idx = 0
        self.resultat = None
        self.hist = []                  # historique du dernier run
        self.paysage = None             # données affichées dans la vue 2
        self.sigma = None               # incertitude de la croyance (probabilité)
        self.verite_ok = []             # par scan : vérité terrain connue ?
        self.poses_trouvees = {}        # scan idx -> pose du dernier run réussi
        self._deltas = {}               # scan idx -> Δ scan-contre-scan (chaînage)
        self.benchmark_txt = ""
        self.syncing = False
        self.drag = None
        self.last_dir = CARTES if CARTES.is_dir() else Path.home()

        self._build_ui()
        self.affiche_carte()
        if dossier is not None:
            self.charger(dossier)

    # -- interface ---------------------------------------------------------- #
    def _build_ui(self):
        for name in ("save", "fullscreen", "home", "back", "forward", "pan",
                     "zoom", "xscale", "yscale", "all_axes", "grid_major",
                     "grid_minor"):
            try:
                plt.rcParams[f"keymap.{name}"] = []
            except KeyError:
                pass

        self.fig = plt.figure(figsize=(14.2, 9.8))
        self.fig.canvas.manager.set_window_title("Localisation — scan matching")
        self.fig.patch.set_facecolor(BG)

        # Vue 1 : la carte de référence et le scan à localiser.
        self.ax = self.fig.add_axes((0.195, 0.415, 0.360, 0.515))
        self.ax.set_facecolor(PANEL)
        self.ax.set_aspect("equal")
        self.ax.grid(True, color=GRID, lw=0.8, zorder=0)
        self.ax.tick_params(colors=AXIS, labelsize=8, length=3)
        for spine in self.ax.spines.values():
            spine.set_color(GRID)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.hint = self.ax.text(0.5, 0.5, "Aucune carte chargée\n\n"
                                 "Cliquez sur « Charger une carte… »",
                                 transform=self.ax.transAxes, ha="center",
                                 va="center", fontsize=11, color=MUTED,
                                 linespacing=1.6)
        self.ax.set_title("Carte de référence", fontsize=12, fontweight="bold",
                          color=TEXT, pad=10)
        self.img = self.ax.imshow(np.zeros((2, 2, 4)), origin="lower",
                                  extent=(-1, 1, -1, 1),
                                  interpolation="nearest", zorder=1)
        self.segs = LineCollection([], colors=[tuple(float(c) for c in vec.DARK)],
                                   linewidths=2.0, zorder=3)
        self.ax.add_collection(self.segs)
        self.nuage_pts, = self.ax.plot([], [], ".", ms=2.5, color=ACCENT,
                                       alpha=0.35, zorder=2)
        self.ghost, = self.ax.plot([], [], ".", ms=4, color=GRIS, alpha=0.55,
                                   zorder=4)
        self.scan_est, = self.ax.plot([], [], ".", ms=5, color=ROUGE, zorder=6)
        self.traj, = self.ax.plot([], [], "-", lw=1.0, color=ACCENT, alpha=0.8,
                                  zorder=5)
        self.traj_pts, = self.ax.plot([], [], ".", ms=3, color=ACCENT, zorder=5)
        self.orig_v, = self.ax.plot([], [], marker="^", ms=8, ls="", color=VERT,
                                    zorder=7)
        self.orig_e, = self.ax.plot([], [], marker="^", ms=8, ls="", color=ROUGE,
                                    zorder=7)
        self.part_sc = self.ax.scatter([], [], s=6, color="#f59e0b", alpha=0.35,
                                       linewidths=0, zorder=5)
        self.status = self.fig.text(0.375, 0.333, "", ha="center", va="center",
                                    fontsize=9, color=TEXT)

        # Vue 2 : paysage du score score(x, y) à θ*.
        self.pays_ax = self.fig.add_axes((0.580, 0.415, 0.190, 0.515))
        self.pays_ax.set_facecolor(PANEL)
        self.pays_ax.set_xticks([])
        self.pays_ax.set_yticks([])
        for spine in self.pays_ax.spines.values():
            spine.set_color(GRID)
        self.pays_img = self.pays_ax.imshow(np.zeros((2, 2)), origin="lower",
                                            extent=(-1, 1, -1, 1), cmap="viridis",
                                            zorder=1)
        self.pays_v, = self.pays_ax.plot([], [], marker="*", ms=13, ls="",
                                         color=VERT, zorder=4)
        self.pays_i, = self.pays_ax.plot([], [], marker="o", ms=9, ls="",
                                         mfc="none", mec="w", mew=1.4, zorder=4)
        self.pays_r, = self.pays_ax.plot([], [], marker="X", ms=9, ls="",
                                         color=ROUGE, zorder=4)
        self.pays_chemin = None
        self.pays_ax.set_title("score (x, y)", fontsize=12, fontweight="bold",
                               color=TEXT, pad=10)

        # Vue 3a : courbes score / erreur par itération.
        self.courb_ax = self.fig.add_axes((0.795, 0.640, 0.190, 0.290))
        self.courb_ax.set_facecolor(PANEL)
        self.courb_ax.grid(True, color=GRID, lw=0.7, zorder=0)
        self.courb_ax.tick_params(colors=AXIS, labelsize=7, length=3)
        for spine in self.courb_ax.spines.values():
            spine.set_color(GRID)
        self.courb_ax2 = self.courb_ax.twinx()
        self.courb_ax2.tick_params(colors=ROUGE, labelsize=7, length=3)
        self.courb_ax.set_title("convergence", fontsize=12, fontweight="bold",
                                color=TEXT, pad=10)
        self.vline = self.courb_ax.axvline(np.nan, color=AXIS, lw=1.0, zorder=3)

        # Vue 3b : chiffres du run (ou tableau du benchmark).
        self.txt_ax = self.fig.add_axes((0.795, 0.415, 0.190, 0.205))
        self.txt_ax.set_axis_off()
        self.txt = self.txt_ax.text(0.0, 1.0, "aucun run", family="monospace",
                                    fontsize=7.5, color=TEXT, va="top",
                                    linespacing=1.4)

        self.fig.text(0.5, 0.018,
                      "clic : placer l'estimation     molette : zoom     "
                      "glisser : panoramique     double-clic : recadrer     "
                      "C charger  L lancer  P perturber  B benchmark  "
                      "F recadrer  Q quitter",
                      ha="center", va="center", fontsize=8, color=MUTED)

        # Colonne de gauche : carte + choix + actions.
        self.lbl_carte = self.fig.text(0.012, 0.972, "aucune carte",
                                       fontsize=9, color=TEXT)
        self.lbl_fichier = self.fig.text(0.012, 0.947, "", fontsize=7.5,
                                         color=MUTED)

        def make_button(rect, label, callback):
            ax = self.fig.add_axes(rect)
            btn = Button(ax, label, color="#f2f4f8", hovercolor=HILITE)
            btn.label.set_fontsize(9)
            btn.label.set_color(TEXT)
            btn.on_clicked(callback)
            return btn

        self.btn_charger = make_button((0.010, 0.880, 0.175, 0.042),
                                       "Charger une carte…", lambda _: self.ask_load())
        self.btn_ref_scan = make_button((0.010, 0.832, 0.175, 0.042),
                                        "Charger un scan référence…",
                                        lambda _: self.ask_scan_reference())
        self.btn_scans = make_button((0.010, 0.784, 0.175, 0.042),
                                     "Charger les scans à localiser…",
                                     lambda _: self.ask_scans_cibles())
        self.btn_lancer = make_button((0.010, 0.448, 0.175, 0.042),
                                      "Lancer le matching", lambda _: self.lancer())
        self.btn_perturber = make_button((0.010, 0.400, 0.175, 0.042),
                                         "Perturber la pose", lambda _: self.perturber())
        self.btn_bench = make_button((0.010, 0.352, 0.175, 0.042),
                                     f"Benchmark ({self.essais} essais)", lambda _: self.benchmark())
        self.btn_fit = make_button((0.010, 0.304, 0.175, 0.042),
                                   "Recadrer la vue", lambda _: self.fit())
        self.btn_save_scans = make_button((0.010, 0.256, 0.175, 0.042),
                                          "Sauvegarder scans",
                                          lambda _: self.sauvegarder_scans())

        rad_ref = self.fig.add_axes((0.012, 0.690, 0.16, 0.078))
        rad_ref.set_facecolor(PANEL)
        self.rad_ref = RadioButtons(rad_ref, ("grille", "vectorielle", "scan (nuage)"),
                                    activecolor=ACCENT)
        for lbl in self.rad_ref.labels:
            lbl.set_fontsize(9)
        self.rad_ref.on_clicked(lambda _r: self.affiche_carte())

        rad_met = self.fig.add_axes((0.012, 0.490, 0.16, 0.185))
        rad_met.set_facecolor(PANEL)
        self.rad_met = RadioButtons(rad_met, tuple(a.nom for a in sm.ALGORITHMES),
                                    activecolor=ACCENT)
        for lbl in self.rad_met.labels:
            lbl.set_fontsize(9)

        # Vue 2 : score brut ou carte de probabilité d'emplacement (croyance).
        rad_vue = self.fig.add_axes((0.584, 0.372, 0.10, 0.036))
        rad_vue.set_facecolor(PANEL)
        self.rad_vue = RadioButtons(rad_vue, ("score", "probabilité"),
                                    activecolor=ACCENT)
        for lbl in self.rad_vue.labels:
            lbl.set_fontsize(8)
        self.rad_vue.on_clicked(lambda _r: (self.affiche_paysage(),
                                            self.affiche_texte()))

        # Curseurs : estimation initiale (gauche) et réglages (droite).
        self.sliders = {}
        specs = ((0.195, 0.300, 0.290, "x", "x (m)", -25.0, 25.0, 0.01, "%+.2f"),
                 (0.195, 0.252, 0.290, "y", "y (m)", -25.0, 25.0, 0.01, "%+.2f"),
                 (0.195, 0.204, 0.290, "th", "θ (°)", -180.0, 180.0, 0.1, "%+.1f"),
                 (0.575, 0.300, 0.200, "scan", "scan #", 0.0, 1.0, 1, "%d"),
                 (0.575, 0.252, 0.200, "bruit_m", "bruit (m)", 0.0, 0.60, 0.01, "%.2f"),
                 (0.575, 0.204, 0.200, "bruit_d", "bruit (°)", 0.0, 20.0, 0.5, "%.1f"),
                 (0.575, 0.156, 0.200, "iter", "itération", 0.0, 1.0, 1, "%d"),
                 (0.575, 0.108, 0.200, "k", "croyance ×", 1.0, 10.0, 0.5, "%.1f"))
        for x, y, w, key, label, lo, hi, step, fmt in specs:
            ax = self.fig.add_axes((x, y, w, 0.022))
            slider = Slider(ax, label, lo, hi, valinit=0, valstep=step,
                            valfmt=fmt, color=AXIS)
            slider.label.set_fontsize(9)
            slider.label.set_color(MUTED)
            slider.valtext.set_fontsize(9)
            slider.valtext.set_color(TEXT)
            slider.on_changed(lambda v, k=key: self.on_slider(k, v))
            self.sliders[key] = slider
        self.sliders["bruit_m"].set_val(0.25)
        self.sliders["bruit_d"].set_val(6.0)
        self.sliders["k"].set_val(4.0)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    # -- chargement --------------------------------------------------------- #
    def ask_load(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                dossier = filedialog.askdirectory(parent=root,
                                                  initialdir=str(self.last_dir),
                                                  title="Choisir un dossier de carte")
            finally:
                root.destroy()
        except Exception as exc:
            print(f"Sélecteur indisponible : {exc}")
            return
        if dossier:
            self.charger(dossier)

    def charger(self, dossier):
        """Charge un dossier de carte et reconstruit grille + carte vectorielle."""
        t0 = time.perf_counter()
        self.status.set_text("chargement de la carte…")
        self.fig.canvas.draw_idle()
        try:
            echantillons, grille, vect = charger_carte(dossier)
        except Exception as exc:
            self.status.set_text(f"échec : {exc}")
            return
        self.dossier = Path(dossier)
        self.scans = echantillons
        self.verite_ok = [True] * len(self.scans)
        self.grille, self.vect = grille, vect
        self.nuage = None                    # une carte remplace le scan référence
        self._verites_carte = {f.resolve(): pose for f, _p, pose in echantillons}
        self.last_dir = self.dossier
        self.lbl_carte.set_text(f"carte : {self.dossier.name}")
        self.lbl_fichier.set_text(str(self.dossier))
        n = max(len(self.scans) - 1, 0)
        self.sliders["scan"].valmax = n
        self.sliders["scan"].set_val(0)
        self.on_scan(0)                      # estimation initiale = vérité
        self.hint.set_visible(False)
        self.fit()
        self.status.set_text(
            f"carte chargée : {len(self.scans)} scans, {len(vect.segments)} segments "
            f"({time.perf_counter() - t0:.1f} s)")

    # -- mode scan-contre-scan ---------------------------------------------- #
    def ask_scan_reference(self):
        chemin = self._dialog_csv("Choisir le scan de référence")
        if chemin:
            self.charger_scan_reference(chemin)

    def charger_scan_reference(self, chemin):
        """Un scan CSV devient la référence : mini-carte (grille + segments)
        posée à l'identité + nuage brut pour la référence « scan (nuage) »."""
        chemin = Path(chemin)
        pts, _inten = carto.load_scan(chemin)
        rays = [(pts, (0.0, 0.0))]
        self.nuage = pts
        self.grille = occ.build(rays, 0.10)
        self.vect = vec.build(rays)
        self._verites_carte = {}
        self.dossier = chemin.parent
        self.scans = []
        self.verite_ok = []
        self.resultat, self.hist, self.benchmark_txt = None, [], ""
        self.poses_trouvees, self._deltas = {}, {}
        self.lbl_carte.set_text(f"référence : {chemin.name}")
        self.lbl_fichier.set_text(str(chemin))
        self.hint.set_visible(False)
        self.last_dir = chemin.parent
        self.affiche_carte()
        self.fit()
        self.status.set_text(f"scan référence chargé ({len(pts)} points) — "
                             "chargez maintenant les scans à localiser")

    def ask_scans_cibles(self):
        if self.nuage is None and self.grille is None:
            # ni carte ni scan référence : demander la référence d'abord
            chemin = self._dialog_csv("Choisir d'abord le scan de référence")
            if not chemin:
                self.status.set_text("chargement annulé")
                return
            self.charger_scan_reference(chemin)
        chemins = self._dialog_csv("Choisir les scans à localiser", multiple=True)
        if chemins:
            self.charger_scans(chemins)
        else:
            self.status.set_text("chargement annulé")

    def charger_scans(self, chemins):
        """Charge les scans à localiser.

        Mode carte : la référence est la carte chargée, la vérité terrain est la
        pose du scan dans l'assemblage de la carte (s'il y figure). Mode scan :
        vérité relative au scan référence (paire connue ou assemblage), sinon
        identité — sans vérité, les erreurs ne sont pas significatives.
        """
        verites = self._verites_connues()
        ref_pose = verites.get(self._scan_reference_resolu()) if self.nuage is not None else None
        self.scans = []
        self.verite_ok = []
        for chemin in chemins:
            chemin = Path(chemin)
            pts, _inten = carto.load_scan(chemin)
            pose_cible = verites.get(chemin.resolve())
            verite, source, ok = np.zeros(3), "identité (pas de vérité terrain)", False
            if self.nuage is not None:
                if ref_pose is not None and pose_cible is not None:
                    verite = self._pose_relative(ref_pose, pose_cible)
                    source, ok = "vérité relative (assemblage)", True
                else:
                    nom_ref = self._scan_reference_resolu().name
                    v = VERITES_RELATIVES.get((nom_ref, chemin.name))
                    if v is not None:
                        verite = np.asarray(v, float)
                        source, ok = "vérité relative (paire connue)", True
                    else:
                        v = VERITES_RELATIVES.get((chemin.name, nom_ref))
                        if v is not None:
                            verite = self._inverse_pose(np.asarray(v, float))
                            source, ok = "vérité relative (paire connue, inversée)", True
            elif pose_cible is not None:
                verite, source, ok = pose_cible, "vérité carte (assemblage)", True
            self.scans.append((chemin, pts, verite))
            self.verite_ok.append(ok)
            print(f"  + {chemin.name} : {source}")
        self.resultat, self.hist, self.benchmark_txt = None, [], ""
        self.poses_trouvees, self._deltas = {}, {}
        n = max(len(self.scans) - 1, 0)
        self.sliders["scan"].valmax = n
        self.sliders["scan"].set_val(0)
        self.on_scan(0)
        self.fit()
        self.status.set_text(f"{len(self.scans)} scan(s) à localiser chargé(s)")

    def sauvegarder_scans(self):
        """Copie chaque fichier CSV des scans chargés dans le sous-dossier
        « scans » du dossier de carte courant (dossier/scans/), pour que les
        scans d'un live restent disponibles avec sa carte et son assemblage."""
        if self.dossier is None:
            self.status.set_text("charge d'abord une carte")
            return
        if not self.scans:
            self.status.set_text("aucun scan à sauvegarder")
            return
        cible = self.dossier / "scans"
        cible.mkdir(parents=True, exist_ok=True)
        n = 0
        for chemin, _pts, _verite in self.scans:
            src = Path(chemin)
            if not src.exists():
                self.status.set_text(f"introuvable : {src.name}")
                continue
            shutil.copy2(src, cible / src.name)
            n += 1
        self.status.set_text(f"{n} scan(s) sauvegardé(s) dans {cible}")
        print(f"scans sauvegardés dans {cible}")

    def _dialog_csv(self, titre, multiple=False):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                kw = dict(parent=root, initialdir=str(self.last_dir),
                          title=titre,
                          filetypes=[("Scans lidar", "*.csv"),
                                     ("Tous les fichiers", "*.*")])
                if multiple:
                    return list(filedialog.askopenfilenames(**kw))
                return filedialog.askopenfilename(**kw)
            finally:
                root.destroy()
        except Exception as exc:
            print(f"Sélecteur indisponible : {exc}")
            return [] if multiple else ""

    def _scan_reference_resolu(self):
        return Path(self.lbl_fichier.get_text()).resolve()

    def _verites_connues(self):
        """Toutes les poses connues des assemblages de Data\\Cartes\\*."""
        if self._verites is None:
            self._verites = {}
            for asm in CARTES.glob("*/assemblage.json"):
                try:
                    data = json.loads(asm.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for e in data.get("scans", []):
                    f = Path(str(e.get("file", "")))
                    if f.exists():
                        self._verites[f.resolve()] = np.array(
                            (e.get("x", 0.0), e.get("y", 0.0),
                             math.radians(e.get("theta_deg", 0.0))), float)
        return self._verites

    @staticmethod
    def _pose_relative(pose_ref, pose_cible):
        """Pose de la cible exprimée dans le repère du scan référence."""
        c, s = math.cos(pose_ref[2]), math.sin(pose_ref[2])
        d = pose_cible[:2] - pose_ref[:2]
        return np.array((c * d[0] + s * d[1],          # R(θr)^T · (p_t − p_r)
                         -s * d[0] + c * d[1],
                         carto.wrap_angle(pose_cible[2] - pose_ref[2])))

    @staticmethod
    def _inverse_pose(pose):
        """Pose inverse : la référence vue du repère de la cible."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.array((-(c * pose[0] + s * pose[1]),
                         -(-s * pose[0] + c * pose[1]),
                         carto.wrap_angle(-pose[2])))
        self.hint.set_visible(False)
        self.fit()
        self.status.set_text(
            f"carte chargée : {len(self.scans)} scans, {len(vect.segments)} segments "
            f"({time.perf_counter() - t0:.1f} s)")

    # -- données courantes --------------------------------------------------- #
    def scan_courant(self):
        if not self.scans:
            return None
        return self.scans[min(self.scan_idx, len(self.scans) - 1)]

    def verite_connue(self):
        """Vérité terrain connue pour le scan courant ?"""
        i = self.scan_idx
        return 0 <= i < len(self.verite_ok) and self.verite_ok[i]

    def reference(self):
        return self.rad_ref.value_selected

    def algo_ok(self, algo, ref):
        """Le calcul direct (ICP point-vers-segment) n'a pas de sens sur le
        nuage brut ; tout le reste passe."""
        if ref == "scan (nuage)":
            return algo.nom != "direct"
        return algo.disponible(ref)

    def algo_courant(self):
        nom = self.rad_met.value_selected
        return next(a for a in sm.ALGORITHMES if a.nom == nom)

    def contexte(self):
        _chemin, pts, verite = self.scan_courant()
        return {"pts": pts, "grille": self.grille,
                "segments": self.vect.segments, "verite": verite,
                "nuage": self.nuage}

    def score_fn(self):
        pts = self.scan_courant()[1]
        ref = self.reference()
        if ref == "grille":
            return score_grille(self.grille, pts)
        if ref == "vectorielle":
            return score_segments(self.vect.segments, pts)
        return score_nuage(self.nuage, pts)

    # -- actions ------------------------------------------------------------- #
    @staticmethod
    def _compose(pose, delta):
        """Composition pose ∘ delta : la pose de `pose` suivie de `delta`."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.array((pose[0] + c * delta[0] - s * delta[1],
                         pose[1] + s * delta[0] + c * delta[1],
                         carto.wrap_angle(pose[2] + delta[2])))

    def _estimation_chainee(self):
        """Estimation initiale par chaînage (mode scan-contre-scan, scan i > 0) :
        pose du scan 0 ∘ Δ(0→i) où Δ = pose_relative (recherche globale
        générique scan-contre-scan). Δ est calculé une fois par scan ; la
        composition rejouée si la pose du scan 0 a été affinée depuis."""
        if self.nuage is None or not (0 < self.scan_idx < len(self.scans)):
            return None
        i = self.scan_idx
        delta = self._deltas.get(i)
        if delta is None:
            self.status.set_text(
                f"chaînage scan-contre-scan : Δ(scan0 → scan{i})…")
            self.fig.canvas.draw_idle()
            delta = pose_relative(self.scans[0][1], self.scans[i][1])
            self._deltas[i] = delta
        est = self._compose(self.poses_trouvees.get(0, self.scans[0][2]), delta)
        self.status.set_text(
            f"scan : {self.scans[i][0].name} — estimation chaînée "
            f"Δ={delta[0]:+.2f},{delta[1]:+.2f},"
            f"{math.degrees(delta[2]):+.1f}°")
        return est

    def on_scan(self, value):
        self.scan_idx = min(int(round(value)), max(len(self.scans) - 1, 0))
        est = self._estimation_chainee()
        if est is None:
            _chemin, _pts, verite = self.scan_courant()
            self.set_estimation(verite)      # estimation initiale = vérité
            self.status.set_text(f"scan : {self.scans[self.scan_idx][0].name}")
        else:
            self.set_estimation(est)

    def set_estimation(self, pose):
        self.syncing = True
        self.sliders["x"].set_val(float(pose[0]))
        self.sliders["y"].set_val(float(pose[1]))
        self.sliders["th"].set_val(math.degrees(float(pose[2])))
        self.syncing = False
        self.affiche_carte()

    def estimation(self):
        return np.array((self.sliders["x"].val, self.sliders["y"].val,
                         math.radians(self.sliders["th"].val)))

    def perturber(self):
        scan = self.scan_courant()
        if scan is None:
            return
        _chemin, _pts, verite = scan
        # avec vérité : bruit autour d'elle ; sans vérité (scans externes) :
        # jitter autour de l'estimation placée au clic
        base_pose = verite if self.verite_connue() else self.estimation()
        rng = np.random.default_rng()
        m = self.sliders["bruit_m"].val
        d = math.radians(self.sliders["bruit_d"].val)
        bruit = rng.uniform(-1.0, 1.0, 3) * np.array((m, m, d))
        pose = base_pose + bruit
        pose[2] = carto.wrap_angle(pose[2])
        self.set_estimation(pose)
        autour = "la vérité" if self.verite_connue() else "l'estimation placée"
        self.status.set_text(
            f"estimation perturbée (autour de {autour}) : dx={bruit[0]:+.2f} "
            f"dy={bruit[1]:+.2f} dθ={math.degrees(bruit[2]):+.1f}°")

    def lancer(self):
        if self.grille is None or self.vect is None:
            self.status.set_text("charge d'abord une carte ou un scan référence")
            return
        if not self.scans:
            self.status.set_text("aucun scan à localiser")
            return
        algo = self.algo_courant()
        if not self.algo_ok(algo, self.reference()):
            self.status.set_text(f"« {algo.nom} » indisponible sur "
                                 f"la référence {self.reference()}")
            return
        self.status.set_text("calcul…")
        self.fig.canvas.draw_idle()
        try:
            res = algo.run(self.score_fn(), self.estimation(), self.contexte())
        except Exception as exc:
            self.status.set_text(f"échec : {exc}")
            return
        self.resultat = res
        self.poses_trouvees[self.scan_idx] = res.pose   # sert au chaînage
        self.hist = list(res.historique) or [(res.pose, res.score)]
        self.benchmark_txt = ""
        self.affiche_carte()
        self.affiche_paysage()
        self.affiche_courbes()
        self.sliders["iter"].valmax = max(len(self.hist) - 1, 0)
        self.sliders["iter"].set_val(float(len(self.hist) - 1))   # -> affiche tout

    def benchmark(self):
        if self.grille is None or self.vect is None:
            self.status.set_text("charge d'abord une carte")
            return
        scan = self.scan_courant()
        if scan is None:
            self.status.set_text("aucun scan à localiser")
            return
        _chemin, pts, verite = scan
        ctx = self.contexte()
        m, d = self.sliders["bruit_m"].val, math.radians(self.sliders["bruit_d"].val)
        rng = np.random.default_rng(42)      # mêmes perturbations pour tous
        bruits = rng.uniform(-1.0, 1.0, (self.essais, 3)) * np.array((m, m, d))
        refs = ["grille", "vectorielle"] + (["scan (nuage)"] if self.nuage is not None else [])
        nom_ref = {"vectorielle": "vector.", "scan (nuage)": "nuage"}
        lignes = ["benchmark — scan : " + scan[0].name,
                  f"{self.essais} essais, bruit ±{m:.2f} m / ±{math.degrees(d):.0f}°",
                  f"{'algo':9s} {'réf':8s} {'moy':>6s} {'max':>6s} {'dθ':>6s} "
                  f"{'ms':>6s} {'éch':>4s}"]
        print("\n".join(lignes))
        for ref in refs:
            fn = (score_grille(self.grille, pts) if ref == "grille"
                  else score_segments(self.vect.segments, pts) if ref == "vectorielle"
                  else score_nuage(self.nuage, pts))
            for algo in sm.ALGORITHMES:
                if not self.algo_ok(algo, ref):
                    continue
                erreurs, dths, temps, echecs = [], [], [], 0
                for bruit in bruits:
                    pose0 = verite + bruit
                    res = algo.run(fn, pose0, ctx)
                    err = float(np.hypot(*(res.pose[:2] - verite[:2])))
                    dth = abs(math.degrees(carto.wrap_angle(res.pose[2] - verite[2])))
                    erreurs.append(err)
                    dths.append(dth)
                    temps.append(res.temps)
                    echecs += err > 0.10 or dth > 5.0
                    self.status.set_text(f"benchmark {algo.nom}/{ref}…")
                    self.fig.canvas.draw_idle()
                lignes.append(
                    f"{algo.nom:9s} {nom_ref.get(ref, ref):8s} "
                    f"{np.mean(erreurs) * 100:5.1f}cm {np.max(erreurs) * 100:5.1f}cm "
                    f"{np.mean(dths):5.1f}° {np.mean(temps) * 1000:5.0f} "
                    f"{echecs:3d}/{self.essais}")
        self.benchmark_txt = "\n".join(lignes)
        print(self.benchmark_txt)
        self.txt.set_text(self.benchmark_txt)
        self.status.set_text("benchmark terminé (tableau à droite)")

    # -- affichage ------------------------------------------------------------ #
    def _fenetre(self):
        if self.nuage is not None:           # mode scan : cadrer sur le contenu
            x0, x1 = self.nuage[:, 0].min(), self.nuage[:, 0].max()
            y0, y1 = self.nuage[:, 1].min(), self.nuage[:, 1].max()
        elif self.grille is not None:
            x0, x1, y0, y1 = self.grille.extent()
        elif self.vect is not None:
            x0, x1, y0, y1 = self.vect.extent()
        else:
            return (-5, 5, -5, 5)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        span = max(x1 - x0, y1 - y0) * 1.1 + 0.5
        return (cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2)

    def fit(self, *_args):
        wx0, wx1, wy0, wy1 = self._fenetre()
        self.ax.set_xlim(wx0, wx1)
        self.ax.set_ylim(wy0, wy1)
        self.fig.canvas.draw_idle()

    def affiche_carte(self):
        """Redessine la vue 1 (carte + scans + trajectoire) selon la référence."""
        ref = self.reference()
        a_grille = ref == "grille" and self.grille is not None
        a_nuage = ref == "scan (nuage)" and self.nuage is not None
        self.img.set_visible(a_grille)
        self.segs.set_visible(ref == "vectorielle" and self.vect is not None)
        self.nuage_pts.set_visible(a_nuage)
        if a_grille:
            self.img.set_data(occ.to_rgb(self.grille))
            self.img.set_extent(self.grille.extent())
        elif ref == "vectorielle" and self.vect is not None:
            self.segs.set_segments(self.vect.segments.reshape(-1, 2, 2))
            poids = np.clip(self.vect.hits - 2, 0, 58)
            self.segs.set_linewidths(0.7 + 2.3 * poids / 58.0)
        elif a_nuage:
            self.nuage_pts.set_data(self.nuage[:, 0], self.nuage[:, 1])
        scan = self.scan_courant()
        if scan is None:
            self.ghost.set_data([], [])
            self.scan_est.set_data([], [])
            self.orig_v.set_data([], [])
            self.orig_e.set_data([], [])
        else:
            _chemin, pts, verite = scan
            if self.verite_connue():
                monde_v = carto.transform(pts, verite)
                self.ghost.set_data(monde_v[:, 0], monde_v[:, 1])
                self.orig_v.set_data([verite[0]], [verite[1]])
            else:                        # pas de vérité : pas de fantôme trompeur
                self.ghost.set_data([], [])
                self.orig_v.set_data([], [])
            monde_e = carto.transform(pts, self.estimation())
            self.scan_est.set_data(monde_e[:, 0], monde_e[:, 1])
            est = self.estimation()
            self.orig_e.set_data([est[0]], [est[1]])
        if not self.hist:
            self.traj.set_data([], [])
            self.traj_pts.set_data([], [])
            self.part_sc.set_offsets(np.empty((0, 2)))
        else:
            poses = np.array([p for p, _s in self.hist])
            self.traj.set_data(poses[:, 0], poses[:, 1])
            self.traj_pts.set_data(poses[:, 0], poses[:, 1])
            if self.resultat is not None and self.resultat.particules is not None:
                part = np.asarray(self.resultat.particules)
                self.part_sc.set_offsets(part[:, :2])
        if a_grille:
            handles = [Patch(facecolor="#ffffff", edgecolor=GRID, label="libre"),
                       Patch(facecolor=tuple(float(c) for c in occ.DARK), label="occupé"),
                       Patch(facecolor=tuple(float(c) for c in occ.UNKNOWN_RGB),
                             edgecolor=GRID, label="inconnu")]
        else:
            handles = [Line2D([], [], color=tuple(float(c) for c in vec.DARK), lw=2.2,
                              label="mur")]
            if a_nuage:
                handles = [Line2D([], [], marker=".", ls="", color=ACCENT,
                                  label="scan référence")]
        handles += [Line2D([], [], marker=".", ls="", color=GRIS, label="scan vérité"),
                    Line2D([], [], marker=".", ls="", color=ROUGE, label="scan estimé"),
                    Line2D([], [], marker="^", ls="", color=VERT, label="pose vérité"),
                    Line2D([], [], marker="^", ls="", color=ROUGE, label="pose estimée")]
        if self.resultat is not None and self.resultat.particules is not None:
            handles.append(Line2D([], [], marker=".", ls="", color="#f59e0b",
                                  label="particules"))
        self.ax.legend(handles=handles, loc="upper center",
                       bbox_to_anchor=(0.5, -0.085), ncol=4, frameon=False,
                       fontsize=7, handlelength=1.1, columnspacing=1.0,
                       handletextpad=0.3, borderaxespad=0.0)
        self.fig.canvas.draw_idle()

    def affiche_paysage(self):
        """Redessine la vue 2 : score(x, y) ou carte de probabilité d'emplacement
        à θ*, avec les marqueurs vérité / initial / résultat.

        Mode « probabilité » : P(x, y) ∝ exp(k · score/score_range) — la
        croyance sur la position. Pour la méthode à particules, la carte est la
        densité des particules (la vraie croyance Monte-Carlo), et l'incertitude
        σx, σy en découle (affichée dans les diagnostics).
        """
        self.sigma = None
        if self.resultat is None:
            self.pays_img.set_visible(False)
            for art in (self.pays_v, self.pays_i, self.pays_r):
                art.set_data([], [])
            return
        self.pays_img.set_visible(True)
        res = self.resultat
        probabilite = self.rad_vue.value_selected == "probabilité"

        if probabilite and res.particules is not None:
            # croyance Monte-Carlo : densité du nuage de particules
            part = np.asarray(res.particules)
            n = 61
            etendue = max(3.0 * float(np.max(np.std(part[:, :2], axis=0))),
                          0.35,                              # un minimum lisible
                          float(np.hypot(*(self.scan_courant()[2][:2] - res.pose[:2]))))
            xs = res.pose[0] + np.linspace(-etendue, etendue, n)
            ys = res.pose[1] + np.linspace(-etendue, etendue, n)
            vals = _flou(np.histogram2d(part[:, 0], part[:, 1],
                                        bins=(xs, ys))[0].T, 2)
            th = res.pose[2]
            titre = "P(emplacement) — particules"
            self.sigma = np.std(part[:, :2], axis=0)
        else:
            if res.paysage is not None:
                xs, ys, th, vals = res.paysage
            else:  # méthode sans paysage : léger échantillonnage local
                fn, centre = self.score_fn(), res.pose
                n, span = 41, 1.2
                xs = centre[0] + np.linspace(-span / 2, span / 2, n)
                ys = centre[1] + np.linspace(-span / 2, span / 2, n)
                th = centre[2]
                vals = np.full((n, n), -np.inf)
                for j, y in enumerate(ys):
                    for i, x in enumerate(xs):
                        vals[j, i] = fn(np.array((x, y, th)))
            if probabilite:   # croyance déduite du score : softmax normalisée
                ptp = max(float(np.ptp(vals)), 1e-9)
                vals = np.exp(float(self.sliders["k"].val) * (vals - np.max(vals)) / ptp)
                vals /= vals.max()
                titre = f"P(emplacement)  à  θ={math.degrees(th):+.1f}°"
                if np.sum(vals) > 0:
                    X, Y = np.meshgrid(xs, ys)          # vals[ligne, colonne] = (y, x)
                    w = vals / vals.sum()
                    cx, cy = float((X * w).sum()), float((Y * w).sum())
                    self.sigma = np.sqrt([float((w * (X - cx) ** 2).sum()),
                                          float((w * (Y - cy) ** 2).sum())])
            else:
                titre = (f"score (x, y)  à  θ={math.degrees(th):+.1f}°\n"
                         f"min {np.min(vals):.2f} – max {np.max(vals):.2f}")

        self.pays_img.set_data(vals)
        self.pays_img.set_extent((xs[0], xs[-1], ys[0], ys[-1]))
        if probabilite:
            self.pays_img.set_cmap("magma")
            self.pays_img.set_norm(None)
        elif np.min(vals) > 0:                 # grande dynamique : échelle log
            self.pays_img.set_cmap("viridis")
            self.pays_img.set_norm(mat_colors.LogNorm(vmin=max(np.min(vals), 1e-3),
                                                      vmax=np.max(vals)))
        else:
            self.pays_img.set_cmap("viridis")
            self.pays_img.set_norm(None)       # score vectoriel : valeurs négatives
        self.pays_ax.set_xlim(xs[0], xs[-1])
        self.pays_ax.set_ylim(ys[0], ys[-1])
        self.pays_ax.set_title(titre, fontsize=10, fontweight="bold",
                               color=TEXT, pad=8)
        verite = self.scan_courant()[2]
        init = self.hist[0][0]
        self.pays_v.set_visible(self.verite_connue())
        self.pays_v.set_data([verite[0]], [verite[1]])
        self.pays_i.set_data([init[0]], [init[1]])
        self.pays_r.set_data([res.pose[0]], [res.pose[1]])
        if self.pays_chemin is not None:     # chemin de l'optimisation précédent
            self.pays_chemin.remove()
            self.pays_chemin = None
        if len(self.hist) > 1:               # chemin de l'optimisation
            poses = np.array([p for p, _s in self.hist])
            self.pays_chemin, = self.pays_ax.plot(poses[:, 0], poses[:, 1],
                                                  "-o", ms=2.5, lw=1.0,
                                                  color="w", alpha=0.9, zorder=3)

    def affiche_courbes(self):
        self.courb_ax.clear()
        self.courb_ax2.clear()
        self.vline = self.courb_ax.axvline(np.nan, color=AXIS, lw=1.0, zorder=3)
        if not self.hist:
            self.courb_ax.set_xticks([])
            self.courb_ax.set_yticks([])
            return
        scores = [s for _p, s in self.hist]
        verite = self.scan_courant()[2]
        err = [float(np.hypot(p[0] - verite[0], p[1] - verite[1]))
               for p, _s in self.hist]
        it = np.arange(len(self.hist))
        l1, = self.courb_ax.plot(it, scores, color=ACCENT, lw=1.4, label="score")
        self.courb_ax.set_xlabel("itération", color=MUTED, fontsize=8)
        self.courb_ax.set_ylabel("score", color=ACCENT, fontsize=8)
        l2, = self.courb_ax2.plot(it, err, color=ROUGE, lw=1.4, label="erreur (m)")
        self.courb_ax2.set_ylabel("erreur (m)", color=ROUGE, fontsize=8)
        self.courb_ax.tick_params(colors=AXIS, labelsize=7)
        self.courb_ax2.tick_params(colors=ROUGE, labelsize=7)
        self.courb_ax.grid(True, color=GRID, lw=0.7, zorder=0)
        self.courb_ax.legend(handles=[l1, l2], loc="center right", frameon=False,
                             fontsize=7)

    @staticmethod
    def _extra_txt(res):
        """Lignes de diagnostic de l'exhaustif : incohérence des rayons de la
        pose finale + les bassins concurrents arbitrés (combo = score −
        incohérence ; quasi-égalités départagées par l'estimation initiale)."""
        if not res.extra:
            return ""
        lignes = ""
        inco = res.extra.get("incoherence")
        if inco is not None:
            lignes += f"incohérence : {inco:.3f}\n"
        bassins = res.extra.get("bassins")
        if bassins:
            for combo, _f, _ic, pb in sorted(bassins, key=lambda b: -b[0])[:3]:
                lignes += (f"  bassin ({pb[0]:+.2f},{pb[1]:+.2f},"
                           f"{math.degrees(pb[2]) % 360:.0f}°) combo {combo:.3f}\n")
        return lignes

    def affiche_texte(self, i=None):
        if self.benchmark_txt:
            return                            # le benchmark occupe le panneau
        if self.resultat is None:
            self.txt.set_text("aucun run")
            return
        res = self.resultat
        scan = self.scan_courant()
        _chemin, _pts, verite = scan
        pose = res.pose if i is None else self.hist[min(i, len(self.hist) - 1)][0]
        if not self.verite_connue():
            self.txt.set_text(
                f"méthode   : {self.algo_courant().nom}\n"
                f"référence : {self.reference()}\n"
                f"évals     : {res.n_evals}\n"
                f"temps     : {res.temps * 1000:.0f} ms\n"
                f"score     : {res.score:.3f}\n"
                f"pose      : x={pose[0]:+.2f} y={pose[1]:+.2f} "
                f"θ={math.degrees(pose[2]):+.1f}°\n"
                + self._extra_txt(res)
                + f"verdict   : sans vérité terrain — évaluer visuellement")
            return
        dx, dy = pose[0] - verite[0], pose[1] - verite[1]
        dth = math.degrees(carto.wrap_angle(pose[2] - verite[2]))
        err = math.hypot(dx, dy)
        conv = err < SEUIL_ERR and abs(dth) < SEUIL_DTH
        self.txt.set_text(
            f"méthode   : {self.algo_courant().nom}\n"
            f"référence : {self.reference()}\n"
            f"évals     : {res.n_evals}\n"
            f"temps     : {res.temps * 1000:.0f} ms\n"
            f"score     : {res.score:.3f}\n"
            f"pose      : x={pose[0]:+.2f} y={pose[1]:+.2f} "
            f"θ={math.degrees(pose[2]):+.1f}°\n"
            f"erreur    : {err * 100:.1f} cm  {dth:+.1f}°\n"
            + self._extra_txt(res)
            + (f"croyance : σx=±{self.sigma[0] * 100:.1f} cm  "
               f"σy=±{self.sigma[1] * 100:.1f} cm\n" if self.sigma is not None else "")
            + f"verdict   : {'convergé' if conv else 'HORS seuil'} "
            f"({SEUIL_ERR * 100:.0f} cm / {SEUIL_DTH:.0f}°)")

    def affiche_iteration(self, i):
        """Rejoue une itération de l'historique dans la vue 1 + repères."""
        if not self.hist:
            return
        i = min(int(i), len(self.hist) - 1)
        pose, _s = self.hist[i]
        _chemin, pts, _verite = self.scan_courant()
        monde = carto.transform(pts, pose)
        self.scan_est.set_data(monde[:, 0], monde[:, 1])
        self.orig_e.set_data([pose[0]], [pose[1]])
        self.vline.set_xdata([i])
        self.affiche_texte(i)
        self.status.set_text(f"itération {i + 1}/{len(self.hist)}")
        self.fig.canvas.draw_idle()

    # -- curseurs ------------------------------------------------------------- #
    def on_slider(self, key, value):
        if self.syncing:
            return
        if key in ("x", "y", "th"):
            self.affiche_carte()
            self.affiche_texte()
        elif key == "scan":
            self.on_scan(value)
        elif key == "iter":
            self.affiche_iteration(value)
        elif key == "k":
            if (self.resultat is not None
                    and self.rad_vue.value_selected == "probabilité"
                    and self.resultat.particules is None):
                self.affiche_paysage()

    # -- navigation ------------------------------------------------------------- #
    def on_key(self, event):
        touche = (event.key or "").lower()
        if touche == "q":
            plt.close(self.fig)
        elif touche == "c":
            self.ask_load()
        elif touche == "l":
            self.lancer()
        elif touche == "p":
            self.perturber()
        elif touche == "b":
            self.benchmark()
        elif touche == "f":
            self.fit()
        elif touche == "s":
            self.sauvegarder_scans()

    def on_press(self, event):
        if event.inaxes is not self.ax:
            return
        if event.dblclick:
            self.fit()
            return
        self.drag = {"pix": event.x, "pixy": event.y,
                     "xdata": event.xdata, "ydata": event.ydata,
                     "button": event.button,
                     "xlim": self.ax.get_xlim(), "ylim": self.ax.get_ylim()}

    def on_release(self, event):
        # un clic simple (sans glissement) place l'estimation initiale
        if (self.drag is not None and event.inaxes is self.ax
                and event.button == 1 and event.xdata is not None
                and math.hypot(event.x - self.drag["pix"],
                               event.y - self.drag["pixy"]) < 4):
            est = self.estimation()
            self.set_estimation((event.xdata, event.ydata, est[2]))
            self.status.set_text(
                f"estimation placée : x={event.xdata:+.2f} y={event.ydata:+.2f} "
                "(règle θ au curseur, puis Lance)")
        self.drag = None

    def on_motion(self, event):
        if self.drag is None or event.x is None:
            return
        ax = self.ax
        kx = (self.drag["xlim"][1] - self.drag["xlim"][0]) / max(ax.bbox.width, 1)
        ky = (self.drag["ylim"][1] - self.drag["ylim"][0]) / max(ax.bbox.height, 1)
        dx = (event.x - self.drag["pix"]) * kx
        dy = (event.y - self.drag["pixy"]) * ky
        ax.set_xlim(self.drag["xlim"][0] - dx, self.drag["xlim"][1] - dx)
        ax.set_ylim(self.drag["ylim"][0] - dy, self.drag["ylim"][1] - dy)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
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
        plt.show()


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    dossier = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    Localisation(dossier).run()


if __name__ == "__main__":
    main()

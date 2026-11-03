# cartographie.py — Superpose plusieurs scans LD06 pour construire une carte globale.
#
# Charge tous les CSV de Data\Scans (récursivement) et permet de déplacer/aligner
# chaque scan (x, y, θ) à la souris ou au clavier. Les poses sont sauvegardées
# dans Data\Scans\poses.json et rechargées au lancement ; les nouveaux CSV
# apparaissent automatiquement (acquisition en cours).
#
# Usage : python cartographie.py [dossier]
# Défaut : C:\Users\Clement\Downloads\SLAM\Data\Scans
# Dépendances : matplotlib, numpy
#
# Commandes (fenêtre active) :
#   clic gauche + glisser   déplacer le scan courant
#   clic droit + glisser    tourner le scan courant
#   ← → ↑ ↓                 déplacer (Maj = ×10)
#   A / D                   tourner (Maj = ×10)
#   N / P, 1–9              changer de scan
#   F                       recadrer la vue      molette = zoomer
#   E                       exporter la carte (Data\Cartes\<nom>\carte.csv + rendu PNG)
#   R                       remettre le scan à zéro     Maj+R = tous
#   S                       sauvegarder les poses       U = recharger les scans
#   T                       afficher/masquer la trace   Q = quitter

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ROOT = Path(r"C:\Users\Clement\Downloads\SLAM\Data\Scans")
STEP_XY, STEP_TH = 0.02, 1.0  # pas fins : 2 cm et 1°
COLORS = plt.get_cmap("tab20").colors
BG, PANEL, GRID = "#ffffff", "#fbfbfd", "#e7e9ef"
TEXT, MUTED, AXIS = "#1f2430", "#6b7280", "#9aa1ad"


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
        self.root = root
        self.poses_file = root / "poses.json"
        self.cartes = root.parent / "Cartes"  # Data\Scans -> Data\Cartes
        self.visualisation = Path(__file__).resolve().with_name("visualisation.py")
        self.saved = {}
        if self.poses_file.exists():
            try:
                self.saved = json.loads(self.poses_file.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"poses.json illisible ({exc}) — poses remises à zéro")
        self.scans = []
        self.active = 0
        self.drag = None
        self.naming = None  # saisie du nom de carte en cours
        self.saved_on_exit = False

        self._build_ui()
        self.reload()
        if not self.scans:
            sys.exit(f"Aucun scan .csv dans {root}")

        # Nouveaux scans pris en compte pendant que la fenêtre est ouverte.
        self.timer = self.fig.canvas.new_timer(interval=2000)
        self.timer.add_callback(self.poll)
        self.timer.start()

    # -- interface ---------------------------------------------------------- #
    def _build_ui(self):
        # Libère les raccourcis matplotlib que l'on utilise.
        for name in ("save", "fullscreen", "home", "back", "forward", "pan",
                     "zoom", "xscale", "yscale", "all_axes", "grid_major",
                     "grid_minor"):
            try:
                plt.rcParams[f"keymap.{name}"] = []
            except KeyError:  # paramètre absent selon la version de matplotlib
                pass

        self.fig, self.ax = plt.subplots(figsize=(11, 9.2))
        self.fig.canvas.manager.set_window_title("Cartographie — superposition de scans")
        self.fig.patch.set_facecolor(BG)
        self.ax.set_facecolor(PANEL)
        self.ax.set_aspect("equal")
        self.ax.grid(True, color=GRID, lw=0.8, zorder=0)
        self.ax.tick_params(colors=AXIS, labelsize=8, length=3)
        for spine in self.ax.spines.values():
            spine.set_color(GRID)
        self.ax.set_xlabel("x (m)", color=MUTED, fontsize=9)
        self.ax.set_ylabel("y (m)", color=MUTED, fontsize=9)

        self.suptitle = self.fig.suptitle("", fontsize=14, fontweight="bold", color=TEXT)
        self.status = self.fig.text(0.5, 0.045, "", ha="center", va="center",
                                    fontsize=10, color=TEXT)
        self.fig.text(0.5, 0.012,
                      "glisser G : déplacer     glisser D : tourner     flèches / A-D : "
                      "pas fins (+Maj×10)     N-P : scan     E : exporter     F : recadrer     "
                      "S : sauver     Q : quitter",
                      ha="center", va="center", fontsize=8, color=MUTED)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.08)

        self.trail, = self.ax.plot([], [], ls=":", lw=1.1, color="#b9bfca",
                                   marker="o", ms=3, zorder=1)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    # -- chargement --------------------------------------------------------- #
    def reload(self):
        """Relit les CSV du dossier en conservant les poses déjà réglées."""
        for s in self.scans:
            self.saved[self.key(s)] = self.pose_list(s)

        files = sorted(self.root.rglob("*.csv"))
        if not self.scans:
            self.active = 0
        current = self.key(self.scans[self.active]) if self.scans else None

        for s in self.scans:
            s.cloud.remove()
            if s.label is not None:
                s.label.remove()
        self.scans = [self.make_scan(p) for p in files if p.stat().st_size > 0]

        if not self.scans:
            self.fig.canvas.draw_idle()
            return
        if current is not None:
            keys = [self.key(s) for s in self.scans]
            self.active = keys.index(current) if current in keys else min(self.active, len(self.scans) - 1)
        self.active = min(self.active, len(self.scans) - 1)
        self.refresh()
        self.fit()

    def make_scan(self, path):
        try:
            pts, inten = load_scan(path)
        except Exception as exc:
            print(f"  ignore {path.name} : {exc}")
            pts, inten = np.empty((0, 2)), np.empty(0)
        key = self.key(path)
        pose = np.array(self.saved.get(key, (0.0, 0.0, 0.0)), dtype=float)
        pose[2] = math.radians(pose[2])
        color = COLORS[len(self.scans) % len(COLORS)]
        cloud = self.ax.scatter([], [], s=10, c=[color], linewidths=0, zorder=3)
        label = self.ax.text(0, 0, "", fontsize=8, color=color, ha="left",
                             va="bottom", zorder=5)
        scan = Scan(path, pts, inten, pose, color, cloud, label)
        self.scans.append(scan)
        self.draw(scan)
        return scan

    def poll(self):
        """Ajoute les CSV apparus dans le dossier depuis le dernier passage."""
        known = {self.key(s) for s in self.scans}
        new = [p for p in sorted(self.root.rglob("*.csv"))
               if p.stat().st_size > 0 and self.key(p) not in known]
        for path in new:
            self.make_scan(path)
        if new:
            print(f"  + {len(new)} nouveau(x) scan(s)")
            self.refresh()

    # -- affichage ---------------------------------------------------------- #
    def key(self, scan_or_path):
        path = scan_or_path.path if isinstance(scan_or_path, Scan) else scan_or_path
        return path.relative_to(self.root).as_posix()

    def pose_list(self, scan):
        return [round(float(scan.pose[0]), 4), round(float(scan.pose[1]), 4),
                round(float(np.rad2deg(scan.pose[2])), 3)]

    def draw(self, scan):
        """Place les points et l'étiquette d'un scan d'après sa pose."""
        scan.cloud.set_offsets(transform(scan.pts, scan.pose))
        active = 0 <= self.active < len(self.scans) and self.scans[self.active] is scan
        scan.cloud.set_sizes([15 if active else 10])
        scan.cloud.set_alpha(0.95 if active else 0.42)
        scan.cloud.set_zorder(4 if active else 3)
        scan.label.set_position((scan.pose[0], scan.pose[1]))
        scan.label.set_text(f"{self.scans.index(scan) + 1}" if active else "")
        scan.label.set_visible(active)

    def refresh(self):
        for scan in self.scans:
            self.draw(scan)
        if self.scans:
            org = np.array([s.pose[:2] for s in self.scans])
            self.trail.set_data(org[:, 0], org[:, 1])
        else:
            self.trail.set_data([], [])

        total = sum(len(s.pts) for s in self.scans)
        self.suptitle.set_text(f"Carte globale — {len(self.scans)} scan(s), {total} points")
        s = self.scans[self.active]
        self.status.set_text(
            f"{self.active + 1}/{len(self.scans)}   {self.key(s)}   "
            f"x = {s.pose[0]:+.2f} m   y = {s.pose[1]:+.2f} m   "
            f"θ = {np.rad2deg(s.pose[2]):+.1f}°   ({len(s.pts)} points)")
        self.fig.canvas.draw_idle()

    def fit(self):
        """Cadre la vue sur l'ensemble des points, avec une marge."""
        pts = [transform(s.pts, s.pose) for s in self.scans if len(s.pts)]
        if not pts:
            return
        allp = np.vstack(pts)
        lo = np.percentile(allp, 1, axis=0)
        hi = np.percentile(allp, 99, axis=0)
        center = (lo + hi) / 2
        span = max(float((hi - lo).max()), 1.0) * 1.2
        self.ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        self.ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        self.fig.canvas.draw_idle()

    # -- édition ------------------------------------------------------------ #
    def move(self, dx, dy):
        s = self.scans[self.active]
        s.pose[0] += dx
        s.pose[1] += dy
        self.refresh()
        self.save()

    def rotate(self, deg):
        self.scans[self.active].pose[2] += math.radians(deg)
        self.refresh()
        self.save()

    def select(self, index):
        if not self.scans:
            return
        self.active = index % len(self.scans)
        self.refresh()

    def save(self, quiet=True):
        if not self.scans:
            return
        data = {self.key(s): self.pose_list(s) for s in self.scans}
        self.poses_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        if not quiet:
            print(f"Poses sauvegardées : {self.poses_file}")

    # -- export ------------------------------------------------------------- #
    def next_name(self):
        """Plus petit numéro libre dans Data\\Cartes (1, 2, 3…)."""
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
        self.status.set_text(
            f"Nom de la carte : {self.naming}_     (Entrée = exporter"
            f"{f' « {hint} »' if not self.naming else ''}, Échap = annuler)")
        self.fig.canvas.draw_idle()

    def export(self, name):
        """Écrit Data\\Cartes\\<nom>\\carte.csv puis appelle visualisation.py."""
        usable = [s for s in self.scans if len(s.pts)]
        if not usable:
            self.status.set_text("Rien à exporter : aucun point dans les scans.")
            self.fig.canvas.draw_idle()
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

        self.status.set_text(f"Carte « {name} » exportée — rendu en cours…")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        png = folder / "carte.png"
        try:
            done = subprocess.run(
                [sys.executable, str(self.visualisation), str(folder), "--force"],
                capture_output=True, text=True, timeout=180)
            detail = (done.stderr or done.stdout).strip()
        except Exception as exc:
            detail = str(exc)
        if png.exists():
            print(f"Rendu : {png}")
            self.status.set_text(f"Carte « {name} » exportée : carte.csv + carte.png")
        else:
            print(f"  échec du rendu : {detail}")
            self.status.set_text(f"Carte « {name} » exportée, mais le rendu a échoué "
                                 "(voir la console).")
        self.fig.canvas.draw_idle()

    # -- événements --------------------------------------------------------- #
    def on_key(self, event):
        if not self.scans or event.key is None:
            return
        if self.naming is not None:  # saisie du nom de carte
            self.on_name_key(event)
            return
        key = event.key
        parts = key.split("+")
        name = parts[-1].lower()
        coarse = "shift" in parts or (len(key) == 1 and key.isupper())
        step = STEP_XY * (10 if coarse else 1)
        turn = STEP_TH * (10 if coarse else 1)

        if name in ("q", "escape"):
            self.timer.stop()
            self.save()
            self.saved_on_exit = True
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
        elif name in ("n", "pagedown", "tab"):
            self.select(self.active + 1)
        elif name in ("p", "pageup"):
            self.select(self.active - 1)
        elif len(name) == 1 and name in "123456789" and int(name) <= len(self.scans):
            self.select(int(name) - 1)
        elif name == "f":
            self.fit()
        elif name == "e":
            self.begin_export()
        elif name == "r":
            if coarse:
                for s in self.scans:
                    s.pose[:] = 0.0
            else:
                self.scans[self.active].pose[:] = 0.0
            self.refresh()
            self.save()
        elif name == "s":
            self.save(quiet=False)
        elif name == "u":
            self.reload()
        elif name == "t":
            self.trail.set_visible(not self.trail.get_visible())
            self.fig.canvas.draw_idle()

    def on_name_key(self, event):
        key = event.key
        if key in ("enter", "return"):
            typed, self.naming = self.naming, None
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
        if self.naming is not None or event.inaxes is not self.ax or not self.scans:
            return
        self.drag = {"button": event.button, "x": event.xdata, "y": event.ydata,
                     "pix": event.x, "pixy": event.y,
                     "pose": self.scans[self.active].pose.copy(),
                     "xlim": self.ax.get_xlim(), "ylim": self.ax.get_ylim()}

    def on_release(self, _event):
        if self.drag is not None:
            self.save()
        self.drag = None

    def on_close(self, _event):
        if not self.saved_on_exit:
            self.save(quiet=False)
            self.saved_on_exit = True

    def on_motion(self, event):
        if self.drag is None:
            return
        button = self.drag["button"]
        pose = self.drag["pose"].copy()
        if button == 1 and event.xdata is not None:
            pose[0] += event.xdata - self.drag["x"]
            pose[1] += event.ydata - self.drag["y"]
            self.scans[self.active].pose = pose
            self.refresh()
        elif button == 3:  # rotation : ~0,4° par pixel
            pose[2] -= math.radians(0.4 * (event.x - self.drag["pix"]))
            self.scans[self.active].pose = pose
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
        print("Fenêtre ouverte — glisser pour aligner les scans, S pour sauvegarder, Q pour quitter.")
        plt.show()


def main():
    try:  # évite un plantage d'affichage sous une console Windows non UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    if not root.is_dir():
        sys.exit(f"Dossier introuvable : {root}\nLancez d'abord acquisition.py.")
    if not any(root.rglob("*.csv")):
        sys.exit(f"Aucun scan .csv dans {root}\nLancez d'abord acquisition.py.")
    Carte(root).run()


if __name__ == "__main__":
    main()

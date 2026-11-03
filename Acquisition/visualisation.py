# visualisation.py — Rend un scan CSV du LD06, une carte globale ou une carte
# d'occupation en image PNG.
#
# Parcourt récursivement un dossier, ignore les fichiers déjà rendus et génère les
# autres à côté du CSV :
#     Data\Scans\1\2\20260101_120000.csv  ->  Data\Scans\1\2\20260101_120000.png
#     Data\Cartes\1\carte.csv              ->  Data\Cartes\1\carte.png
#
# Deux formats sont reconnus d'après l'en-tête :
#   - scan lidar  : N, Angle(°), Distance(mm), Intensity, x, y   -> rendu polaire
#   - carte       : N, x(mm), y(mm), Intensity                   -> rendu cartésien
#   - occupation  : en-tête « # occupation;res=… »               -> grille d'occupation
#     (occupation.csv, libre / occupé / inconnu, généré à l'export de la carte).
#
# Rendu : nuage de points en mètres, axes gradués, grille et points colorés
# selon leur intensité (0–255).
#
# Usage : python visualisation.py [dossier|fichier.csv] [--force]
#         dossier par défaut : Data\Scans ; --force refait les rendus existants.
# Dépendances : matplotlib, numpy

import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch

try:  # module voisin (même dossier)
    import occupation
except ImportError:  # rendu sans le module : la carte d'occupation est ignorée
    occupation = None

ROOT = Path(r"C:\Users\Clement\Downloads\SLAM\Data\Scans")

warnings.filterwarnings("ignore", "loadtxt")  # fichier vide = scan sans point

BG, PANEL, GRID, SPOKE = "#ffffff", "#fbfbfd", "#d5d8de", "#ebedf1"
AXIS, TEXT, MUTED = "#6b7280", "#1f2430", "#6b7280"
STEPS = (0.1, 0.2, 0.5, 1.0, 2.0, 2.5, 5.0)


def header(path):
    """Noms de colonnes du CSV, en minuscules."""
    with open(path, encoding="utf-8-sig") as f:
        return [c.strip().lower() for c in f.readline().strip().split(",")]


def column(cols, name, default=None):
    for i, c in enumerate(cols):
        if c.startswith(name):
            return i
    return default


def load(path):
    """(x, y) en mètres, intensité et booléen polaire (scan lidar ou carte)."""
    cols = header(path)
    i_int = column(cols, "intensity", 3)
    i_angle = column(cols, "angle")

    if i_angle is not None:  # scan lidar : N, Angle(°), Distance(mm), Intensity, x, y
        # x,y ne sont pas réutilisés : la librairie LD06 les calcule en miroir
        # (y = -r·sin θ), on repasse par l'angle mathématique du CSV.
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2,
                          usecols=(i_angle, column(cols, "distance", 2), i_int),
                          encoding="utf-8-sig")
        angle, dist, inten = data.T
        keep = dist > 0  # distance nulle = pas de retour
        r, th = dist[keep] / 1000.0, np.deg2rad(angle[keep])
        return r * np.cos(th), r * np.sin(th), inten[keep], True

    # carte globale exportée : N, x(mm), y(mm), Intensity
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2,
                      usecols=(column(cols, "x(mm)", 1), column(cols, "y(mm)", 2), i_int),
                      encoding="utf-8-sig")
    x, y, inten = data.T
    return x / 1000.0, y / 1000.0, inten, False


def frame(ax, limit, step, cx=0.0, cy=0.0):
    """Fond polaire (scan) : 12 rayons et un anneau par graduation."""
    for a in range(0, 360, 30):
        t = math.radians(a)
        ax.plot([0, limit * math.cos(t)], [0, limit * math.sin(t)],
                color=SPOKE, lw=0.8, zorder=0)
    for v in np.arange(1, round(limit / step) + 1) * step:
        ax.add_patch(Circle((0, 0), v, fill=False, ec=GRID, lw=0.8, zorder=1))
        ax.text(v / math.sqrt(2), v / math.sqrt(2), f"{v:g}" + (" m" if abs(v - limit) < 1e-9 else ""),
                fontsize=7, color=MUTED, ha="center", va="center", zorder=4,
                bbox=dict(fc=PANEL, ec="none", alpha=0.9, pad=0.9))

    # Axes X/Y gradués (valeurs en mètres).
    for dx, dy in ((1.05 * limit, 0), (0, 1.05 * limit)):
        ax.annotate("", xy=(dx, dy), xytext=(-dx, -dy), zorder=2,
                    arrowprops=dict(arrowstyle="<|-|>", color=AXIS, lw=1,
                                    mutation_scale=11, shrinkA=0, shrinkB=0))
    t = 0.014 * limit
    for v in np.arange(1, round(limit / step) + 1) * step:
        for s in (-1, 1):
            ax.plot([s * v, s * v], [-t, t], color=AXIS, lw=0.9, zorder=2)
            ax.text(s * v, -2.5 * t, f"{s * v:g}", fontsize=7.5, color=MUTED,
                    ha="center", va="top", zorder=2)
            ax.plot([-t, t], [s * v, s * v], color=AXIS, lw=0.9, zorder=2)
            ax.text(-2.7 * t, s * v, f"{s * v:g}", fontsize=7.5, color=MUTED,
                    ha="right", va="center", zorder=2)
    ax.text(1.10 * limit, 0, "x (m)", fontsize=9, color=AXIS,
            ha="left", va="center", style="italic")
    ax.text(0, 1.10 * limit, "y (m)", fontsize=9, color=AXIS,
            ha="right", va="center", style="italic")

    # Repères angulaires du lidar.
    for a, ha, va in ((0, "right", "center"), (90, "center", "top"),
                      (180, "left", "center"), (270, "center", "bottom")):
        t = math.radians(a)
        ax.text(0.93 * limit * math.cos(t), 0.93 * limit * math.sin(t), f"{a}°",
                fontsize=8, color=MUTED, ha=ha, va=va, zorder=2)


def grid(ax, limit, step, cx, cy):
    """Fond cartésien (carte) : grille carrée graduée en mètres."""
    ax.axis("on")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(np.arange(math.ceil((cx - limit) / step) * step, cx + limit + step / 2, step))
    ax.set_yticks(np.arange(math.ceil((cy - limit) / step) * step, cy + limit + step / 2, step))
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlabel("x (m)", fontsize=9, color=MUTED)
    ax.set_ylabel("y (m)", fontsize=9, color=MUTED)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(GRID)


def render(path, x, y, inten, name, polar=True):
    rad = np.hypot(x, y)
    if polar:
        ref = float(np.percentile(rad, 99.5))  # ignore les rares points aberrants
        step = next(s for s in STEPS if ref / s <= 6)
        limit = max(math.ceil(ref / step) * step, step)
        cx = cy = 0.0
    else:
        lo, hi = np.percentile(np.column_stack((x, y)), 1, axis=0), \
            np.percentile(np.column_stack((x, y)), 99, axis=0)
        cx, cy = (lo + hi) / 2
        ref = max(float((hi - lo).max()), 0.5)
        step = next(s for s in STEPS if ref / s <= 7)
        limit = max(math.ceil(ref / 2 / step) * step, step)

    fig = plt.figure(figsize=(9, 8.6), dpi=160)
    ax = fig.add_axes((0.02, 0.05, 0.82, 0.85))
    ax.set_facecolor(PANEL)
    ax.set_xlim(cx - 1.22 * limit, cx + 1.22 * limit)
    ax.set_ylim(cy - 1.22 * limit, cy + 1.22 * limit)
    ax.set_aspect("equal")
    ax.axis("off")

    if polar:
        frame(ax, limit, step)
    else:
        grid(ax, limit, step, cx, cy)

    # Points colorés par l'intensité.
    sc = ax.scatter(x, y, c=inten, cmap="viridis", vmin=0, vmax=255,
                    s=12, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, cax=fig.add_axes((0.865, 0.18, 0.02, 0.62)))
    cb.set_label("Intensité (0–255)", fontsize=9)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=3)
    cb.outline.set_edgecolor(GRID)

    if polar:  # position du lidar
        ax.scatter([0], [0], marker=(3, 0, -90), s=80, color="#111827",
                   edgecolors=BG, linewidths=0.9, zorder=5)
        ax.text(0.045 * limit, 0.055 * limit, "lidar", fontsize=8, color=MUTED,
                ha="left", va="bottom", zorder=4,
                bbox=dict(fc=PANEL, ec="none", alpha=0.85, pad=0.8))

    fig.suptitle(f"{'Scan' if polar else 'Carte'} {name}",
                 fontsize=14, fontweight="bold", color=TEXT)
    detail = (f"{len(x)} points  •  portée {rad.min():.2f} – {rad.max():.2f} m"
              if polar else
              f"{len(x)} points  •  étendue {np.ptp(x):.2f} × {np.ptp(y):.2f} m")
    ax.set_title(detail, fontsize=9, color=MUTED, pad=10)

    # Écriture atomique : on rend dans un fichier temporaire puis on remplace, pour
    # qu'une visionneuse ne tombe jamais sur un PNG à moitié écrit.
    png = path.with_suffix(".png")
    tmp = png.with_name(f".{png.stem}.tmp.png")
    fig.savefig(tmp, dpi=160, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    os.replace(tmp, png)


def render_occupation(path, name):
    """Rend une carte d'occupation (occupation.csv) : cases libres/occupées/inconnues."""
    grille = occupation.load_csv(path)
    x0, x1, y0, y1 = grille.extent()

    fig = plt.figure(figsize=(9, 8.6), dpi=160)
    ax = fig.add_axes((0.07, 0.10, 0.88, 0.78))
    ax.set_facecolor(PANEL)
    ax.imshow(occupation.to_rgb(grille), origin="lower", extent=(x0, x1, y0, y1),
              interpolation="nearest", zorder=2)

    span = max(x1 - x0, y1 - y0)
    step = next(s for s in STEPS if span / s <= 7)
    ax.set_xticks(np.arange(math.ceil(x0 / step) * step, x1 + step / 2, step))
    ax.set_yticks(np.arange(math.ceil(y0 / step) * step, y1 + step / 2, step))
    ax.grid(True, color=GRID, lw=0.7, zorder=3)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlabel("x (m)", fontsize=9, color=MUTED)
    ax.set_ylabel("y (m)", fontsize=9, color=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    if len(grille.sources):  # positions des lidars
        ax.scatter(grille.sources[:, 0], grille.sources[:, 1], marker="x", s=46,
                   linewidths=1.3, color="#e11d48", zorder=4)

    handles = [
        Patch(facecolor="#ffffff", edgecolor=GRID, label="libre"),
        Patch(facecolor=tuple(float(c) for c in occupation.DARK), label="occupé"),
        Patch(facecolor=tuple(float(c) for c in occupation.UNKNOWN_RGB),
              edgecolor=GRID, label="inconnu"),
        plt.Line2D([], [], marker="x", ls="", color="#e11d48", label="lidar"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.51, 0.0))

    fig.suptitle(f"Carte d'occupation {name}", fontsize=14, fontweight="bold",
                 color=TEXT)
    ax.set_title(f"{grille.nx} × {grille.ny} cases  •  résolution "
                 f"{grille.res * 100:.0f} cm", fontsize=9, color=MUTED, pad=10)

    png = path.with_suffix(".png")
    tmp = png.with_name(f".{png.stem}.tmp.png")
    fig.savefig(tmp, dpi=160, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    os.replace(tmp, png)


def main():
    args = sys.argv[1:]
    force = any(a in ("-f", "--force") for a in args)
    paths = [a for a in args if not a.startswith("-")]
    target = Path(paths[0]) if paths else ROOT

    if target.is_file():
        files, base = [target], target.parent
    elif target.is_dir():
        files, base = sorted(target.rglob("*.csv")), target
    else:
        sys.exit(f"Dossier introuvable : {target}")

    todo = [p for p in files if force or not p.with_suffix(".png").exists()]
    print(f"{len(todo)} fichier(s) à rendre sur {len(files)}")
    for path in todo:
        try:
            if occupation and occupation.is_occupation(path):
                name = (path.parent.name if path.stem == "occupation"
                        else path.relative_to(base).with_suffix("").as_posix())
                render_occupation(path, name)
                print(f"  {path} (carte d'occupation)")
                continue
            x, y, inten, polar = load(path)
            if x.size < 2:
                raise ValueError("aucun point exploitable")
            name = path.relative_to(base).with_suffix("").as_posix()
            if path.stem == "carte":  # Data\Cartes\1\carte.csv -> titre « 1 »
                name = path.parent.relative_to(base.parent).as_posix()
            render(path, x, y, inten, name, polar)
            print(f"  {path}")
        except Exception as exc:  # un fichier cassé ne bloque pas les suivants
            print(f"  échec {path} : {exc}")


if __name__ == "__main__":
    main()

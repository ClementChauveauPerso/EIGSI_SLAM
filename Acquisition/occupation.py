# occupation.py — Carte d'occupation (grille) construite à partir des scans assemblés.
#
# Méthode (voir Ressources\Cartographie\Construction dune grid map 2D...) :
#   1. grille régulière de résolution res (m) couvrant les points observés ;
#   2. chaque rayon lidar marque comme *libres* les cases traversées ;
#   3. la case d'impact est marquée *occupée* ;
#   4. probabilité = impacts / (impacts + passages) ; une case jamais traversée
#      reste *inconnue* (127). L'impact pèse double pour garder des murs nets.
#
# Encodage : uint8, 0 = libre, 255 = occupé, 127 = inconnu.
#
# Fichier CSV : une ligne d'en-tête puis la matrice (une ligne par y, en partant
# de y0 vers le haut) :
#   # occupation;res=0.1000;x0=-2.3000;y0=-1.4000;nx=150;ny=140;
#     0=libre,255=occupe,127=inconnu;sources=-1.2,0.4|3.6,0.1

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNKNOWN = 127
MIN_RANGE = 0.05   # en deçà, le point est dans le boîtier du lidar
MAX_RANGE = 12.0   # portée utile d'un LD06
MAX_CELLS = 1_500_000  # garde-fou mémoire (~1,5 Mo en uint8)

DARK = np.array((0.13, 0.16, 0.24), dtype=np.float32)     # occupé
UNKNOWN_RGB = np.array((0.90, 0.91, 0.94), dtype=np.float32)  # inconnu


@dataclass
class Grille:
    x0: float
    y0: float
    res: float
    proba: np.ndarray            # (ny, nx) uint8
    sources: np.ndarray          # (m, 2) positions des lidars

    @property
    def nx(self):
        return self.proba.shape[1]

    @property
    def ny(self):
        return self.proba.shape[0]

    def extent(self):
        """(xmin, xmax, ymin, ymax) pour imshow."""
        return (self.x0, self.x0 + self.nx * self.res,
                self.y0, self.y0 + self.ny * self.res)

    def proba_en(self, pts):
        """Probabilité d'occupation (uint8) aux points monde (N, 2).

        Hors grille : inconnu (127).
        """
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        ci = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
        cj = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
        dedans = (ci >= 0) & (ci < self.nx) & (cj >= 0) & (cj < self.ny)
        out = np.full(len(pts), UNKNOWN, dtype=np.uint8)
        out[dedans] = self.proba[cj[dedans], ci[dedans]]
        return out


def build(rays, res=0.10, max_range=MAX_RANGE):
    """Construit la grille.

    rays : liste de (points globaux (N, 2), origine du lidar (2,)).
    Retourne une Grille, ou None si aucun rayon exploitable.
    """
    res = max(float(res), 0.01)
    ends, srcs = [], []
    for pts, origin in rays:
        pts = np.asarray(pts, dtype=float)
        if not len(pts):
            continue
        origin = np.asarray(origin, dtype=float)[:2]
        dist = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
        keep = (dist >= MIN_RANGE) & (dist <= max_range)
        if keep.any():
            ends.append(pts[keep])
            srcs.append(np.broadcast_to(origin, (int(keep.sum()), 2)))
    if not ends:
        return None
    ends = np.vstack(ends)
    srcs = np.vstack(srcs)

    # Cadre : 1–99 % des impacts + les lidars (ignore les points aberrants).
    ref = np.vstack((ends, srcs))
    lo = np.percentile(ref, 1, axis=0) - 2 * res
    hi = np.percentile(ref, 99, axis=0) + 2 * res
    while ((hi - lo) / res).prod() > MAX_CELLS:  # grille trop fine pour la zone
        res *= 1.25
    nx = max(int(math.ceil((hi[0] - lo[0]) / res)), 2)
    ny = max(int(math.ceil((hi[1] - lo[1]) / res)), 2)

    hits = np.zeros(nx * ny, dtype=np.float32)
    free = np.zeros(nx * ny, dtype=np.float32)
    _raster(ends, srcs, lo, res, nx, ny, hits, free)

    total = hits + free
    proba = np.full(nx * ny, UNKNOWN, dtype=np.uint8)
    known = total > 0
    # Un impact pèse double : un mur touché puis « traversé » par un rayon voisin
    # reste nettement occupé au lieu de virer au gris.
    proba[known] = np.clip(np.round(2 * hits[known] / (2 * hits[known] + free[known]) * 255),
                           0, 255).astype(np.uint8)
    sources = np.unique(np.round(srcs, 4), axis=0)
    return Grille(float(lo[0]), float(lo[1]), res, proba.reshape(ny, nx), sources)


def _raster(ends, srcs, lo, res, nx, ny, hits, free, chunk=600):
    """Marque les cases traversées (libres) et d'impact (occupées)."""
    for start in range(0, len(ends), chunk):
        e = ends[start:start + chunk]
        o = srcs[start:start + chunk]
        d = e - o
        dist = np.hypot(d[:, 0], d[:, 1])
        steps = np.clip(np.ceil(dist / res), 2, 900).astype(np.int32)
        idx = np.arange(int(steps.max()), dtype=np.int32)[None, :]
        valid = idx < steps[:, None]
        t = idx / np.maximum(steps[:, None] - 1, 1)  # 0 = lidar, 1 = impact
        xs = o[:, 0, None] + d[:, 0, None] * t
        ys = o[:, 1, None] + d[:, 1, None] * t
        ci = np.floor((xs - lo[0]) / res).astype(np.int32)
        cj = np.floor((ys - lo[1]) / res).astype(np.int32)
        inside = valid & (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
        lin = cj * nx + ci
        free += np.bincount(lin[inside & (idx < steps[:, None] - 1)],
                            minlength=nx * ny)
        hits += np.bincount(lin[inside & (idx == steps[:, None] - 1)],
                            minlength=nx * ny)


def to_rgb(grille):
    """Image RGBA (ny, nx, 4) : blanc = libre, sombre = occupé, gris = inconnu."""
    p = grille.proba.astype(np.float32) / 255.0
    known = grille.proba != UNKNOWN
    img = np.empty((grille.ny, grille.nx, 4), dtype=np.float32)
    for k in range(3):
        img[..., k] = np.where(known, 1.0 - p * (1.0 - DARK[k]), UNKNOWN_RGB[k])
    img[..., 3] = 1.0
    return img


def save_csv(grille, path):
    """Enregistre la grille (en-tête + matrice), écriture atomique."""
    src = "|".join(f"{x:.3f},{y:.3f}" for x, y in grille.sources)
    lines = [f"# occupation;res={grille.res:.4f};x0={grille.x0:.4f};"
             f"y0={grille.y0:.4f};nx={grille.nx};ny={grille.ny};"
             f"0=libre,255=occupe,127=inconnu;sources={src}"]
    lines += [",".join(str(int(v)) for v in row) for row in grille.proba]
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_csv(path):
    """Relit une grille enregistrée par save_csv (sources optionnelles)."""
    path = Path(path)
    with open(path, encoding="utf-8-sig") as f:
        head = f.readline().strip()
    meta = {}
    if head.startswith("#"):
        for part in head.lstrip("#").split(";"):
            key, sep, value = part.partition("=")
            if sep:
                meta[key.strip().lower()] = value.strip()
    proba = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2,
                       dtype=np.uint8, encoding="utf-8-sig")
    sources = np.empty((0, 2), dtype=float)
    raw = meta.get("sources", "").strip()
    if raw:
        pairs = [p.split(",") for p in raw.split("|") if "," in p]
        sources = np.array([[float(a), float(b)] for a, b in pairs], dtype=float)
    return Grille(float(meta.get("x0", 0.0)), float(meta.get("y0", 0.0)),
                  float(meta.get("res", 0.1)), proba, sources)


def is_occupation(path):
    """Vrai si le CSV est une carte d'occupation (en-tête « # occupation »)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            return f.readline().lstrip().startswith("# occupation")
    except OSError:
        return False

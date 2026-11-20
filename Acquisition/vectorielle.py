# vectorielle.py — Carte vectorielle : nuage de points -> segments.
#
# Méthode (voir Ressources\Cartographie\Construction dune carte vectorielle 2D…) :
#   0. prétraitement : points hors portée retirés, et — si une grille
#      d'occupation est fournie — points gardés seulement dans les cases bien
#      occupées (la grille filtre les artefacts, le nuage garde la précision) ;
#   1. transformer le nuage de points en segments : clusters aux discontinuités de
#      distance, droite ACP par cluster, coupée récursivement au pire point ;
#   2. fusionner les segments « proches » (angle + position + écart : proches(),
#      fusionner()) — même mur vu de deux scans ;
#   3. dédoublonner : une corde inclinée redondante avec un mur mieux soutenu
#      (dérive d'assemblage) est retirée (_dedoubler) ;
#   4. découper les segments traversés par un autre (intersection(), decouper()) :
#      chaque segment garde son plus long côté, les dépassements disparaissent ;
#   5. retirer les fragments trop courts et les segments peu soutenus.
#
# Primitives géométriques exposées : angle_between, proches, fusionner,
# intersection, decouper. Elles opèrent sur des segments (x1, y1, x2, y2) en
# mètres, sans orientation : un segment et son inverse sont équivalents.
#
# Fichier CSV : en-tête « # vectoriel;… » puis une ligne par segment :
#   Type,x1(mm),y1(mm),x2(mm),y2(mm),Hits
#   segment,x1,y1,x2,y2,hits

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOL = 1e-6             # tolérance numérique (m, et sinus pour le parallélisme)
MIN_RANGE = 0.05       # en deçà, le point est dans le boîtier du lidar
MAX_RANGE = 12.0       # portée utile d'un LD06
DEFAULT_TOL = 0.04     # écart max point/droite (m)
MIN_LEN = 0.12         # segment plus court = bruit
GAP_MIN = 0.10         # discontinuité mini entre deux points d'un scan (m)
GAP_RATIO = 0.06       # … et proportionnelle à la portée (1 point ≈ 1°)
MIN_PROBA = 128        # case « plus occupée que traversée » pour valider un point

ANG_TOL = 10.0         # écart d'angle max pour « proches » (degrés)
DIST_TOL = 0.20        # écart perpendiculaire max entre deux segments (m)
GAP_TOL = 0.30         # écart max entre extrémités le long du mur (m)
MIN_HITS = 4           # points mini pour garder un segment (sinon = bruit)
CLEAN_LEN = 0.30       # nettoyage : un segment plus court…
CLEAN_HITS = 12        # …ne reste que s'il est soutenu par ce nombre de points

REDUND_ANGLE = 30.0    # dédoublonnage : écart d'angle max avec le mur dominant
REDUND_OFF = 0.60      # … distance perpendiculaire max au mur dominant
REDUND_RATIO = 1.5     # … et le mur doit être 1,5× mieux soutenu
CUT_RATIO = 4          # découpage : un cutter 4× plus faible ne débite pas un mur

DARK = np.array((0.13, 0.16, 0.24), dtype=np.float32)


@dataclass
class Vectorielle:
    segments: np.ndarray   # (n, 4) : x1, y1, x2, y2 en mètres
    sources: np.ndarray    # (k, 2) : positions des lidars
    hits: np.ndarray       # (n,)   : points ayant soutenu chaque segment

    def __len__(self):
        return len(self.segments)

    def extent(self):
        """(xmin, xmax, ymin, ymax) des segments."""
        if not len(self.segments):
            pts = self.sources if len(self.sources) else np.zeros((1, 2))
        else:
            pts = np.vstack((self.segments[:, :2], self.segments[:, 2:]))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


# --------------------------------------------------------------------------- #
# Primitives géométriques
# --------------------------------------------------------------------------- #
def _vector(seg):
    return np.array((seg[2] - seg[0], seg[3] - seg[1]), dtype=float)


def _unit(seg):
    v = _vector(seg)
    n = math.hypot(*v)
    return v / n if n > TOL else None


def _longueur(seg):
    return math.hypot(seg[2] - seg[0], seg[3] - seg[1])


def angle_between(a, b):
    """Angle non orienté entre deux segments, en degrés dans [0, 90]."""
    u, v = _unit(a), _unit(b)
    if u is None or v is None:
        return 0.0
    return math.degrees(math.acos(min(abs(float(u @ v)), 1.0)))


def proches(a, b, ang_tol=ANG_TOL, dist_tol=DIST_TOL, gap_tol=GAP_TOL):
    """Vrai si deux segments sont alignés, presque sur la même droite et jointifs.

    ang_tol en degrés, dist_tol et gap_tol en mètres.
    """
    if angle_between(a, b) > ang_tol:
        return False
    u = _unit(a)
    if u is None:
        return False
    n = np.array((-u[1], u[0]))
    o = np.array(a[:2], dtype=float)
    # écart perpendiculaire des extrémités de b à la droite de a
    if max(abs((np.array(q, dtype=float) - o) @ n) for q in (b[:2], b[2:4])) > dist_tol:
        return False
    # recouvrement (ou petit écart) le long du mur
    proj = sorted(float((np.array(q, dtype=float) - o) @ u) for q in (b[:2], b[2:4]))
    return proj[0] <= _longueur(a) + gap_tol and proj[1] >= -gap_tol


def fusionner(a, b):
    """Segment unique couvrant a et b (droite des moindres carrés des 4 extrémités).

    La droite passe par une ACP des extrémités : deux vues d'un même mur légèrement
    décalées (dérive d'assemblage) donnent un mur bien droit et non une corde.
    """
    pts = np.array([a[:2], a[2:4], b[:2], b[2:4]], dtype=float)
    c = pts.mean(axis=0)
    d = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    t = (pts - c) @ d
    i, j = int(np.argmin(t)), int(np.argmax(t))
    return (float(c[0] + t[i] * d[0]), float(c[1] + t[i] * d[1]),
            float(c[0] + t[j] * d[0]), float(c[1] + t[j] * d[1]))


def _params(a, b, tol=TOL):
    """Paramètres (t, u, la, lb) du croisement des droites, ou None.

    t et u situent le point sur a et b (0 = premier point, 1 = second).
    None si une longueur est nulle ou si les droites sont parallèles/colinéaires.
    """
    p = np.array(a[:2], dtype=float)
    r = _vector(a)
    q = np.array(b[:2], dtype=float)
    s = _vector(b)
    la, lb = math.hypot(*r), math.hypot(*s)
    if la <= tol or lb <= tol:
        return None
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) <= tol * la * lb:
        return None
    d = q - p
    return ((d[0] * s[1] - d[1] * s[0]) / den,
            (d[0] * r[1] - d[1] * r[0]) / den, la, lb)


def intersection(a, b, tol=TOL):
    """Point d'intersection des deux segments, ou None.

    None si les droites sont parallèles ou colinéaires, ou si le point tombe hors
    de l'un des deux segments (à tol près).
    """
    params = _params(a, b, tol)
    if params is None:
        return None
    t, u, la, lb = params
    if not (-tol / la <= t <= 1 + tol / la):
        return None
    if not (-tol / lb <= u <= 1 + tol / lb):
        return None
    return np.array(a[:2], dtype=float) + t * _vector(a)


def decouper(seg, coupe, tol=TOL):
    """Coupe seg au croisement avec coupe et garde son plus long côté.

    Rien n'est changé si : pas d'intersection, segments colinéaires/parallèles,
    ou intersection sur une extrémité (les deux segments doivent se croiser
    franchement, à tol près). À longueurs égales (à tol près), le côté conservé
    est celui qui contient le premier point (x1, y1) du segment.
    """
    p = np.array(seg[:2], dtype=float)
    q = np.array(seg[2:4], dtype=float)
    long = math.hypot(*(q - p))
    if long <= tol:
        return tuple(seg[:4])
    params = _params(seg, coupe, tol)
    if params is None:  # pas de croisement (parallèles, colinéaires)
        return tuple(seg[:4])
    t, u, la, lb = params
    # il faut un vrai croisement : strictement à l'intérieur des deux segments
    if not (tol / la < t < 1 - tol / la):
        return tuple(seg[:4])
    if not (tol / lb < u < 1 - tol / lb):
        return tuple(seg[:4])
    pt = p + t * (q - p)
    d1, d2 = t * long, (1 - t) * long
    if abs(d1 - d2) <= tol or d1 > d2:  # égalité : côté du premier point
        return (float(p[0]), float(p[1]), float(pt[0]), float(pt[1]))
    return (float(pt[0]), float(pt[1]), float(q[0]), float(q[1]))


# --------------------------------------------------------------------------- #
# Nuage de points -> segments
# --------------------------------------------------------------------------- #
def build(rays, tol=DEFAULT_TOL, max_range=MAX_RANGE, grille=None,
          proba_min=MIN_PROBA):
    """Construit la carte vectorielle.

    rays : liste de (points globaux (N, 2) dans l'ordre du scan, origine lidar (2,)).
    grille : Grille d'occupation (optionnelle). Si fournie, un point n'est gardé
        que dans une case d'occupation >= proba_min : la grille filtre les
        artefacts (sa force), les points gardent la mesure précise (la sienne).
    """
    tol = max(float(tol), 0.01)
    scans, srcs = [], []
    for pts, origin in rays:
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            continue
        o = np.asarray(origin, dtype=float)[:2]
        dist = np.hypot(*(pts - o).T)
        p = pts[(dist >= MIN_RANGE) & (dist <= max_range)]
        if grille is not None and len(p):
            p = p[grille.proba_en(p) >= proba_min]
        if len(p) >= 2:
            scans.append(p)
            srcs.append(o)
    if not scans:
        return Vectorielle(np.empty((0, 4)), np.empty((0, 2)),
                           np.empty(0, dtype=int))
    # Cadre : 1–99 % des impacts + les lidars (ignore les échos aberrants),
    # comme pour la carte d'occupation.
    ref = np.vstack(scans + srcs)
    lo = np.percentile(ref, 1, axis=0) - 0.5
    hi = np.percentile(ref, 99, axis=0) + 0.5

    segs = []
    for p, o in zip(scans, srcs):
        dedans = ((p >= lo) & (p <= hi)).all(axis=1)
        p = p[dedans]
        if len(p) < 2:
            continue
        dist = np.hypot(*(p - o).T)
        saut = np.hypot(*np.diff(p, axis=0).T)
        coupe = saut > np.maximum(GAP_MIN, GAP_RATIO * dist[1:])
        for cluster in np.split(p, np.flatnonzero(coupe) + 1):
            if len(cluster) >= 2:
                _extract(cluster, tol, segs)

    segs = _merge(segs)
    segs = _dedoubler(segs)
    segs = _decouper_tout(segs)
    segs = [s for s in segs if _longueur(s) >= MIN_LEN]
    # nettoyage : hors les trop petits fragments isolés (miettes de bruit)
    segs = [s for s in segs
            if _longueur(s) >= CLEAN_LEN or s[4] >= CLEAN_HITS]
    sources = np.unique(np.round(np.array(srcs), 4), axis=0)
    return Vectorielle(
        np.array([s[:4] for s in segs], dtype=float).reshape(-1, 4),
        np.asarray(sources, dtype=float).reshape(-1, 2),
        np.array([s[4] for s in segs], dtype=int),
    )


def _extract(p, tol, segs, depth=0):
    """Segments d'un cluster : droite ACP, coupée récursivement au pire point."""
    if len(p) < 2:
        return
    (a, b), resid = _line(p)
    erreur = float(resid.max()) if len(resid) else 0.0
    if erreur <= tol or depth >= 8 or len(p) == 2:
        segs.append((a[0], a[1], b[0], b[1], len(p)))
        return
    k = min(max(int(np.argmax(resid)), 1), len(p) - 2)
    _extract(p[:k + 1], tol, segs, depth + 1)
    _extract(p[k + 1:], tol, segs, depth + 1)


def _line(p):
    """Droite des moindres carrés (ACP) : extrémités réelles et résidus."""
    c = p.mean(axis=0)
    q = p - c
    d = np.linalg.svd(q, full_matrices=False)[2][0]
    t = q @ d
    i, j = int(np.argmin(t)), int(np.argmax(t))
    return (p[i], p[j]), np.abs(q @ np.array((-d[1], d[0])))


def _merge(segs):
    """Fusionne les segments proches (même mur vu de deux scans)."""
    segs = [s for s in segs if _longueur(s) >= MIN_LEN]
    changed = True
    while changed:
        changed, out = False, []
        while segs:
            a = segs.pop()
            for k, b in enumerate(segs):
                if proches(a, b):
                    segs[k] = (*fusionner(a, b), a[4] + b[4])
                    changed = True
                    break
            else:
                out.append(a)
        segs = out
    return [s for s in segs if s[4] >= MIN_HITS]


def _dedoubler(segs):
    """Absorbe les segments redondants dans le mur dominant qui les recouvre.

    Un même mur vu par des scans légèrement mal alignés (dérive) forme un arc :
    l'extraction en tire une ligne dominante et des cordes parallèles inclinées.
    Une corde est absorbée si un segment au moins REDUND_RATIO× mieux soutenu
    lui est REDUND_ANGLE° près, à REDUND_OFF au plus, et la recouvre à moitié ;
    fusionner() redresse alors le mur et cumule les points de soutien.
    """
    out = []
    for s in sorted(segs, key=lambda s: -s[4]):
        for k, q in enumerate(out):
            if (angle_between(s, q) <= REDUND_ANGLE
                    and s[4] * REDUND_RATIO <= q[4]
                    and _distance_droite(s, q) <= REDUND_OFF
                    and _chevauchement(s, q) >= 0.5):
                out[k] = (*fusionner(s, q), q[4] + s[4])
                break
        else:
            out.append(s)
    return out


def _distance_droite(s, q):
    """Distance perpendiculaire du milieu de s à la droite portée par q."""
    u = _unit(q)
    if u is None:
        return math.inf
    n = np.array((-u[1], u[0]))
    m = np.array(((s[0] + s[2]) / 2, (s[1] + s[3]) / 2), dtype=float)
    return abs(float((m - np.array(q[:2], dtype=float)) @ n))


def _chevauchement(s, q):
    """Fraction de s recouverte par l'étendue de q, le long de la droite de q."""
    u = _unit(q)
    if u is None:
        return 0.0
    o = np.array(q[:2], dtype=float)
    t = sorted(float((np.array(p, dtype=float) - o) @ u) for p in (s[:2], s[2:4]))
    commun = min(t[1], _longueur(q)) - max(t[0], 0.0)
    return max(commun, 0.0) / max(t[1] - t[0], TOL)


def _decouper_tout(segs, passages=4, tol=TOL):
    """Découpe chaque segment traversé par un autre : il garde son plus long côté.

    Passe sur toutes les paires ordonnées, répétée jusqu'à stabilisation (une
    coupe ne fait que raccourcir, ça converge vite). Tolérance numérique TOL,
    pas la tolérance géométrique de la carte. Un segment bien plus faible que sa
    cible (bruit) ne coupe pas : il ne faut pas qu'une corde débite un mur.
    """
    out = list(segs)
    for _ in range(passages):
        avant = [s[:4] for s in out]
        for i in range(len(out)):
            for j in range(len(out)):
                if i == j or out[j][4] * CUT_RATIO < out[i][4]:
                    continue
                c = decouper(out[i], out[j], tol)
                if c[:4] != out[i][:4]:
                    out[i] = (*c, out[i][4])
        if [s[:4] for s in out] == avant:
            break
    return out


# --------------------------------------------------------------------------- #
# Fichier
# --------------------------------------------------------------------------- #
def save_csv(carte, path):
    """Enregistre les segments (en-tête + lignes), écriture atomique."""
    src = "|".join(f"{x:.3f},{y:.3f}" for x, y in carte.sources)
    lines = [f"# vectoriel;segments={len(carte.segments)};sources={src}",
             "Type,x1(mm),y1(mm),x2(mm),y2(mm),Hits"]
    for (x1, y1, x2, y2), h in zip(carte.segments, carte.hits):
        lines.append(f"segment,{round(x1 * 1000)},{round(y1 * 1000)},"
                     f"{round(x2 * 1000)},{round(y2 * 1000)},{h}")
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_csv(path):
    """Relit une carte enregistrée par save_csv (sources optionnelles)."""
    rows = Path(path).read_text(encoding="utf-8-sig").splitlines()
    segs, hits = [], []
    for line in rows[1:]:
        parts = line.split(",")
        try:
            v = [float(x) for x in parts[1:]]
        except ValueError:  # ligne d'en-tête
            continue
        if len(v) < 5:
            continue
        segs.append((v[0] / 1000.0, v[1] / 1000.0, v[2] / 1000.0, v[3] / 1000.0))
        hits.append(int(v[4]))
    src = ""
    head = rows[0] if rows else ""
    if head.startswith("#"):
        for part in head.lstrip("#").split(";"):
            key, sep, value = part.partition("=")
            if sep and key.strip().lower() == "sources":
                src = value.strip()
    sources = np.empty((0, 2))
    if src:
        pairs = [p.split(",") for p in src.split("|") if "," in p]
        sources = np.array([[float(a), float(b)] for a, b in pairs], dtype=float)
    return Vectorielle(
        np.array(segs, dtype=float).reshape(-1, 4),
        sources,
        np.array(hits, dtype=int),
    )


def is_vector(path):
    """Vrai si le CSV est une carte vectorielle (en-tête « # vectoriel »)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            return f.readline().lstrip().startswith("# vectoriel")
    except OSError:
        return False


if __name__ == "__main__":  # vérification rapide : python vectorielle.py [assemblage.json]
    import json
    import sys

    import cartographie as carto

    default = (Path(__file__).resolve().parents[1] / "Data" / "Cartes" / "salle"
               / "assemblage.json")
    asm = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    scans = json.loads(asm.read_text(encoding="utf-8"))["scans"]
    rays = []
    for e in scans:
        pts, _ = carto.load_scan(Path(e["file"]))
        pose = np.array((e["x"], e["y"], math.radians(e["theta_deg"])), dtype=float)
        rays.append((carto.transform(pts, pose), pose[:2]))
    carte = build(rays)
    print(f"{len(carte.segments)} segments, {len(carte.sources)} lidars")
    assert len(carte.segments) >= 4, "aucun mur extrait"

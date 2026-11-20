# algo_base.py — Socle commun des algorithmes de localisation (scan matching).
#
# Chaque algorithme hérite de Algorithme et implémente run() :
#     resultat = algo.run(score_fn, pose0, contexte)
#   - score_fn : fonction score(pose) à MAXIMISER, construite pour la référence
#     choisie par score_grille() ou score_segments() (ci-dessous) ;
#   - pose0 : estimation initiale (x, y, θ en rad), par exemple la vérité
#     perturbée d'un bruit connu ;
#   - contexte : dict(grille=…, segments=…, pts=…, verite=…) — les données
#     brutes pour les algorithmes qui veulent aller au-delà du score boîte
#     noire (ex. multi-résolution sur une pyramide de grilles, calcul direct
#     sur les segments) ;
#   - Resultat : pose trouvée, historique [(pose, score), …] pour l'interface,
#     nb d'évaluations du score, temps mesuré par run() lui-même (cible < 1 s),
#     et éventuellement paysage (scores 2D à θ* pour l'affichage) et particules
#     (nuage final de la méthode probabiliste).
#
# Deux références possibles :
#   - grille d'occupation : score = occupation moyenne sous les points du scan
#     (interpolation bilinéaire : lisse, donc la descente de gradient marche) ;
#   - carte vectorielle   : score = -d² moyenne aux segments les plus proches
#     (distances plafonnées à CAP_DIST² pour borner l'influence des échos loin
#     des murs).

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as _ndi
except ImportError:      # environnement sans scipy : pas de filtre de bruit
    _ndi = None


def _murs_surs(proba, min_cellules=8, seuil=200):
    """Cellules occupéeS fiables : composantes connectées (8-connexité) de
    ≥ min_cellules. Les taches de bruit d'assemblage (échos parasites isolés,
    typiquement 1-4 cellules) créent des faux attracteurs de score — un
    fragment du scan peut s'y « poser » et battre la vraie pose. Sans scipy,
    tout est gardé (le champ reste calculable, juste moins net)."""
    murs = proba >= seuil
    if _ndi is None:
        return murs
    lab, _n = _ndi.label(murs, structure=np.ones((3, 3), dtype=bool))
    if lab.max() == 0:
        return murs
    tailles = np.bincount(lab.ravel())
    garder = np.zeros(tailles.size, dtype=bool)
    garder[1:] = tailles[1:] >= min_cellules
    return garder[lab]

try:  # modules voisins (même dossier)
    import cartographie as carto
    import occupation as occ
    import vectorielle as vec
except ImportError:  # importé comme paquet
    from . import cartographie as carto
    from . import occupation as occ
    from . import vectorielle as vec

CAP_DIST = 0.50        # distance (m) au-delà de laquelle un point ne compte plus
MAX_PTS = 180          # sous-échantillonnage du scan pour le score (suffisant)
TOL = 1e-9


@dataclass
class Resultat:
    pose: np.ndarray           # pose trouvée (x, y, θ en rad)
    score: float
    historique: list           # [(pose (3,), score), …] une entrée par itération
    n_evals: int               # nb de poses évaluées (coût de calcul)
    temps: float               # durée (s) mesurée par run()
    paysage: object = None     # (xs, ys, θ*, scores 2D) si la méthode le produit
    particules: object = None  # nuage final des particules (méthode probabiliste)
    extra: dict = None         # infos libres pour l'interface


class Algorithme:
    """Interface commune de toutes les méthodes de localisation."""
    nom = "?"
    references = ("grille", "vectorielle")   # références supportées

    def run(self, score_fn, pose0, contexte):
        raise NotImplementedError

    def disponible(self, reference):
        return reference in self.references


# --------------------------------------------------------------------------- #
# Scores (boîtes noires partagées par toutes les méthodes)
# --------------------------------------------------------------------------- #
def _echantillon(pts, n=MAX_PTS):
    """Sous-échantillonne uniformément : le score n'a pas besoin de tous les points."""
    pts = np.asarray(pts, dtype=float)
    return pts[::max(1, math.ceil(len(pts) / n))]


def champ_distance(grille, cap_m=1.0):
    """Champ de distance (m) à la cellule occupée la plus proche.

    Likelihood field (Thrun) : d(p) = distance au mur le plus proche, plafonnée
    à cap_m. Propagation de coûts (chamfer 8-connexe) depuis les cellules
    clairement occupées ; les cases libres et inconnues ont leur distance au
    mur — un point qui flotte hors de la pièce coûte le plafond au lieu d'être
    éliminé par une troncature (un alignement partiel ne peut plus gagner
    contre un alignement complet). Mémoïsé sur l'objet grille (calculé une
    fois par carte, pas une fois par run/algorithme).
    """
    cache = getattr(grille, "_champ_cache", None)
    if cache is not None and cache[0] == cap_m:
        return cache[1]
    proba = np.asarray(grille.proba)
    murs = _murs_surs(proba)
    cap = cap_m / grille.res
    pad = int(cap) + 2
    d = np.where(murs, 0.0, float(cap) + 1.0)
    d = np.pad(d, pad, mode="constant", constant_values=float(cap) + 1.0)
    couts = ((0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
    for _ in range(int(cap) + 2):
        nouveau = d.copy()
        for dy, dx, c in couts:
            source = np.roll(np.roll(d, dy, axis=0), dx, axis=1)
            # le wrap de roll retombe dans le bourrelet neutre (cap + 1) : sans effet
            nouveau = np.minimum(nouveau, source + c)
        if np.array_equal(nouveau, d):
            break
        d = nouveau
    champ = d[pad:-pad, pad:-pad] * grille.res
    try:
        grille._champ_cache = (cap_m, champ)
    except AttributeError:
        pass
    return champ


def image_score_grille(grille, sigma=0.20):
    """(img, x0, y0, res) de la métrique grille : exp(-d²/2σ²) sur le champ de
    likelihood. σ large (0,20 m) pour la recherche : bassins larges, tolère un
    léger flottement ; σ fin (0,06-0,10 m) pour le raffinage final : colle les
    structures. Les cases hors carte valent 0. Les noyaux rapides des
    algorithmes répliquent exactement score_grille."""
    d = champ_distance(grille)
    img = np.exp(-(np.minimum(d, 1.0) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    return img, grille.x0, grille.y0, grille.res


def score_grille(grille, pts_local, sigma=0.20):
    """Fonction score(pose) -> champ de likelihood (max) : moyenne de l'image
    -d² plafonné (champ_distance) — tous les points comptés, un point hors
    pièce coûte ~0 (noyau gaussien saturé au-delà de 1 m). Lisse
    (bilineaire + champ), donc utilisable par la descente de gradient, et
    insensible aux échos hors pièce."""
    img, x0, y0, res = image_score_grille(grille, sigma)
    pts = _echantillon(pts_local)

    def score(pose):
        monde = carto.transform(pts, pose)
        return float(_bilineaire(img, x0, y0, res, monde, hors=0.0).mean())

    return score


def _coh_cache(grille):
    """(murs, libre) pour la cohérence des rayons : murs fiables (_murs_surs)
    et cellules carvées libres — mémoïsé sur l'objet grille."""
    cache = getattr(grille, "_coh_cache", None)
    if cache is None:
        murs = _murs_surs(grille.proba)
        libre = grille.proba <= 50          # complément symétrique du seuil 200
        cache = (murs, libre)
        try:
            grille._coh_cache = cache
        except AttributeError:
            pass
    return cache


def incoherence(grille, pts_local, pose, n=220):
    """Fraction de rayons physiquement impossibles pour la pose donnée (0 = tout
    est cohérent, à minimiser).

    Un scan raconte deux choses qu'une distance aux murs ignore : un rayon qui
    TRAVERSE une cellule murale cartographiée est impossible (le mur l'aurait
    arrêté), et un écho qui atterrit sur une cellule LIBRE (carvée par
    l'assemblage) est impossible (aucune surface y a jamais été vue). Ces deux
    contradictions éliminent les poses où le scan « se pose » sur des
    fragments de carte en travers de la géométrie réelle — indépendamment de
    la complétude de la carte. L'origine du repère du scan est le lidar :
    rayon = segment (0,0) -> écho."""
    murs, libre = _coh_cache(grille)
    q = _echantillon(pts_local, n)
    d = np.hypot(q[:, 0], q[:, 1])
    ok = d > 2.5 * grille.res
    if not ok.any():
        return 1.0
    q, d = q[ok], d[ok]
    tmax = 1.0 - 1.5 * grille.res / d          # l'impact lui-même n'est pas une traversée
    ts = np.linspace(0.02, 1.0, 36)[None, :] * tmax[:, None]
    loc = q[:, None, :] * ts[:, :, None]       # (R, nep, 2)
    monde = carto.transform(loc.reshape(-1, 2), pose).reshape(len(q), -1, 2)
    ci = np.floor((monde[..., 0] - grille.x0) / grille.res).astype(np.int64)
    cj = np.floor((monde[..., 1] - grille.y0) / grille.res).astype(np.int64)
    dedans = (ci >= 0) & (ci < grille.nx) & (cj >= 0) & (cj < grille.ny)
    ci = np.clip(ci, 0, grille.nx - 1)
    cj = np.clip(cj, 0, grille.ny - 1)
    traverse = (murs[cj, ci] & dedans).any(1)
    impact = carto.transform(q, pose)
    ei = np.floor((impact[:, 0] - grille.x0) / grille.res).astype(np.int64)
    ej = np.floor((impact[:, 1] - grille.y0) / grille.res).astype(np.int64)
    edans = (ei >= 0) & (ei < grille.nx) & (ej >= 0) & (ej < grille.ny)
    ei = np.clip(ei, 0, grille.nx - 1)
    ej = np.clip(ej, 0, grille.ny - 1)
    sur_libre = np.zeros(len(q), dtype=bool)
    sur_libre[edans] = libre[ej[edans], ei[edans]]
    return float((traverse | sur_libre).mean())


def _bilineaire(img, x0, y0, res, pts, hors=0.0):
    """Image interpolée bilinéairement ; `hors` hors carte (0 par défaut, pour
    la métrique likelihood : -1, un point hors pièce coûte le plafond)."""
    gx = (pts[:, 0] - x0) / res - 0.5       # position continue dans la grille
    gy = (pts[:, 1] - y0) / res - 0.5
    ny, nx = img.shape
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - np.floor(gx), gy - np.floor(gy)
    ok = (ix >= 0) & (ix < nx - 1) & (iy >= 0) & (iy < ny - 1)
    ic, jc = np.clip(ix, 0, nx - 2), np.clip(iy, 0, ny - 2)
    v = (img[jc, ic] * (1 - fx) * (1 - fy) + img[jc, ic + 1] * fx * (1 - fy)
         + img[jc + 1, ic] * (1 - fx) * fy + img[jc + 1, ic + 1] * fx * fy)
    return np.where(ok, v, hors)


def score_segments(segments, pts_local):
    """Fonction score(pose) -> -d² moyenne aux segments les plus proches (max).

    La géométrie des segments (extrémités, directions, longueurs²) est
    précalculée une fois : un appel de score ne fait que transformer les
    points et évaluer les distances, sans réallouer les projections (N, M, 2).
    Moyenne tronquée sur les paires (point, segment) : les 25 % de paires les
    plus mal appariées sont écartées, pour que les échos hors carte ne
    saturent pas le score. (Limite assumée : un point proche de deux murs
    compte dans deux paires — à passer à une troncature par point si besoin.)
    """
    segs = np.asarray(segments, dtype=float)[:, :4]
    a = segs[:, :2]
    ab = segs[:, 2:4] - a
    l2 = np.maximum((ab * ab).sum(1), TOL)
    cap2 = CAP_DIST * CAP_DIST
    pts = _echantillon(pts_local)
    garde = max(int(pts.shape[0] * 0.75), 1)   # 75 % des points gardés

    def score(pose):
        monde = carto.transform(pts, pose)
        ap = monde[:, None, :] - a[None, :, :]          # (N, M, 2)
        u = np.einsum("nmk,mk->nm", ap, ab) / l2        # paramètre non borné
        t = np.clip(u, 0.0, 1.0)
        d2 = (ap * ap).sum(-1) - l2 * (2.0 * t * u - t * t)
        pire = np.partition(np.minimum(np.maximum(d2, 0.0), cap2).ravel(),
                            garde - 1)[:garde]
        return -float(pire.mean())

    return score


# alias publics : les algorithmes réutilisent la même sous-sélection du scan
# et (au besoin) la même interpolation que le score
echantillon = _echantillon
bilineaire = _bilineaire


# --------------------------------------------------------------------------- #
# Scénario réel de test scan-contre-scan (Data\Scans\j2\salle\test1|test2)
# --------------------------------------------------------------------------- #
# Deux scans pris à quelques centimètres l'un de l'autre, orientations
# différentes (~53° dans le repère des CSV). Pas d'assemblage connu : la vérité
# relative vient d'une recherche globale (lacet ±180°) concordante sur les 3
# références (grille = segments = nuage), confirmée par la corrélation croisée
# des profils de portée (53°, r = 0,87) et par l'utilisateur.
SCENARIO_J2 = {
    "reference": Path(r"C:\Users\Clement\Downloads\SLAM\Data\Scans\j2\salle\test1\20261103_160948.csv"),
    "cible": Path(r"C:\Users\Clement\Downloads\SLAM\Data\Scans\j2\salle\test2\20261103_161012.csv"),
    "verite": np.array((0.0, 0.0, math.radians(53.5))),  # cible dans repère référence
    "tolerance": (0.10, math.radians(4.0)),              # (m, rad)
}
# (nom fichier référence, nom fichier cible) -> pose relative de la cible dans
# le repère de la référence ; consulté par l'interface pour les scans hors assemblage.
VERITES_RELATIVES = {
    (SCENARIO_J2["reference"].name, SCENARIO_J2["cible"].name): SCENARIO_J2["verite"],
}


def _dist_seg(pts, segs):
    """Pour chaque point : d² au segment le plus proche, son indice, sa projection.

    pts : (N, 2) ; segs : (M, 4) -> (N,), (N,), (N, 2).
    """
    a = segs[:, :2]
    ab = segs[:, 2:4] - a
    ap = pts[:, None, :] - a[None, :, :]                    # (N, M, 2)
    l2 = np.maximum((ab * ab).sum(1), TOL)                  # (M,)
    t = np.clip((ap * ab[None, :, :]).sum(-1) / l2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]    # (N, M, 2)
    d2 = ((pts[:, None, :] - proj) ** 2).sum(-1)            # (N, M)
    idx = d2.argmin(1)
    ordre = np.arange(len(pts))
    return d2[ordre, idx], idx, proj[ordre, idx]


def score_nuage(nuage, pts_local):
    """Fonction score(pose) -> -d² moyenne aux points les plus proches du nuage
    de référence (localisation scan-contre-scan, max).

    Même politique que score_segments : distances plafonnées (CAP_DIST²) et
    25 % des points les plus mal appariés écartés (moyenne tronquée), pour que
    les zones vues par un seul des deux scans ne biaisent pas la pose.
    """
    nu = np.asarray(nuage, dtype=float)
    pts = _echantillon(pts_local)
    cap2 = CAP_DIST * CAP_DIST
    garde = max(int(pts.shape[0] * 0.75), 1)

    def score(pose):
        monde = carto.transform(pts, pose)
        d2 = ((monde[:, None, :] - nu[None, :, :]) ** 2).sum(-1)   # (N, M)
        pire = np.partition(np.minimum(d2, cap2).ravel(), garde - 1)[:garde]
        return -float(pire.mean())

    return score


def balayage_correlation(img, x0, y0, res, pts, pas_th=math.radians(7.5), stride=3):
    """Scores (nθ, ny, nx) de toutes les poses translation pan-carte × plein
    cercle en θ — la recherche globale, sans aucune hypothèse de départ.

    Corrélation image : à θ fixé, le score en une translation est la somme des
    valeurs de l'image aux points tournés+translatés, soit l'image additionnée
    une fois par point décalée de sa position grille (coût nθ × npts ajouts de
    tableaux, résolution = res, indépendant d'un pas de translation). Un point
    hors carte contribue 0, comme la métrique likelihood (`hors` de
    _bilineaire). Case (iy, ix) à θ : pose (x0 + (ix+0.5)*res,
    y0 + (iy+0.5)*res, θ).
    """
    q = np.asarray(pts, dtype=float)[::stride]
    ny, nx = img.shape
    ths = np.arange(0.0, 2.0 * math.pi, pas_th)
    out = np.zeros((len(ths), ny, nx), dtype=np.float32)
    for ith, th in enumerate(ths):
        c, s = math.cos(th), math.sin(th)
        rot = q @ np.array(((c, s), (-s, c)))
        cols = np.rint(rot[:, 0] / res).astype(np.int64)
        rang = np.rint(rot[:, 1] / res).astype(np.int64)
        for ac, br in zip(cols, rang):
            if not (-nx < ac < nx and -ny < br < ny):
                continue           # jamais dans la carte, quelle que soit la pose
            ly0, ly1 = max(0, -br), min(ny, ny - br)
            lx0, lx1 = max(0, -ac), min(nx, nx - ac)
            out[ith, ly0:ly1, lx0:lx1] += img[ly0 + br:ly1 + br, lx0 + ac:lx1 + ac]
    return out, ths


def _maxima_locaux(scores, k):
    """Indices (n, 3) des k meilleurs maxima locaux 3D (voisinage 3×3×3)."""
    pad = np.pad(scores, 1, constant_values=np.float32(-np.inf))
    loc = scores == np.lib.stride_tricks.sliding_window_view(
        pad, (3, 3, 3)).max(axis=(3, 4, 5))
    cand = np.argwhere(loc)
    if len(cand) > k:
        cand = cand[np.argsort(scores[tuple(cand.T)])[::-1][:k]]
    return cand


def pose_relative(nuage, pts):
    """Δ = pose de `pts` dans le repère du `nuage` (scan-contre-scan).

    Recherche globale générique au score_nuage : plein cercle en θ (pas 10°),
    translations ±0,5 m (pas 12,5 cm — les scans d'une même prise), puis
    raffinage à deux étages des 3 meilleurs bassins. Sert à poser les autres
    scans d'une même prise relativement au premier (chaînage de l'interface)
    — aucune hypothèse sur la scène ni sur le cap.
    """
    nu = _echantillon(nuage, 150)
    q = _echantillon(pts, 90)
    cap2 = CAP_DIST * CAP_DIST
    garde = max(int(q.shape[0] * 0.75), 1)

    def sc(pose):
        monde = carto.transform(q, pose)
        d2 = ((monde[:, None, :] - nu[None, :, :]) ** 2).sum(-1)
        return -float(np.partition(np.minimum(d2, cap2).ravel(),
                                   garde - 1)[:garde].mean())

    xs = np.arange(-0.5, 0.5 + 1e-9, 0.125)
    ths = np.arange(0.0, 2.0 * math.pi, math.radians(10.0))
    grues = [(sc(np.array((x, y, th))), np.array((x, y, th)))
             for th in ths for x in xs for y in xs]
    grues.sort(key=lambda e: -e[0])
    meil = None
    for _s, c0 in grues[:3]:
        for dth in np.arange(-math.radians(6.0), math.radians(6.0) + 1e-9,
                             math.radians(2.0)):
            for dx in np.arange(-0.14, 0.14 + 1e-9, 0.04):
                for dy in np.arange(-0.14, 0.14 + 1e-9, 0.04):
                    p = c0 + np.array((dx, dy, dth))
                    s = sc(p)
                    if meil is None or s > meil[0]:
                        meil = (s, p)
    c0 = meil[1]
    fn = score_nuage(nuage, pts)
    for dth in np.arange(-math.radians(2.0), math.radians(2.0) + 1e-9,
                         math.radians(1.0)):
        for dx in np.arange(-0.06, 0.06 + 1e-9, 0.02):
            for dy in np.arange(-0.06, 0.06 + 1e-9, 0.02):
                p = c0 + np.array((dx, dy, dth))
                s = fn(p)
                if s > meil[0]:
                    meil = (s, p)
    return meil[1]


# --------------------------------------------------------------------------- #
# Chargement d'une carte exportée (dossier avec assemblage.json)
# --------------------------------------------------------------------------- #
def charger_carte(dossier):
    """Charge une carte : (echantillons, grille, vect).

    echantillons : [(chemin, points locaux, pose vérité), …] — les poses de
    l'assemblage servent de vérité terrain pour mesurer les erreurs.
    """
    dossier = Path(dossier)
    echantillons, rays = [], []
    for e in json.loads((dossier / "assemblage.json").read_text(encoding="utf-8"))["scans"]:
        f = Path(e["file"])
        if not f.exists():
            continue
        pts, _inten = carto.load_scan(f)
        pose = np.array((e["x"], e["y"], math.radians(e["theta_deg"])), float)
        echantillons.append((f, pts, pose))
        rays.append((carto.transform(pts, pose), pose[:2]))
    grille = occ.build(rays, 0.10) if rays else None
    vect = vec.build(rays, grille=grille) if rays else None
    return echantillons, grille, vect

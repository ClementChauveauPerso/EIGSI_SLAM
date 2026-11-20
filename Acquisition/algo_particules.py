# algo_particules.py — Localisation probabiliste (Monte-Carlo / filtre à particules).
#
# n poses (particules) échantillonnées autour de l'estimation initiale,
# pondérées par le score, ré-échantillonnées ; le bruit décroît à chaque
# génération, le nuage se concentre vers le maximum du score. Ne suppose pas
# un seul bassin : gère l'ambiguïté (plusieurs zones plausibles).
#
# Perf : TOUTES les particules d'une génération sont scorées EN UN LOT
# (transformations + interpolation vectorisées, float32, produits matriciels)
# — ~6 gros appels numpy au lieu de n × générations appels python. La
# référence incarnée par score_fn (grille ou vectorielle) étant une boîte
# noire, elle est identifiée en comparant sa valeur à la pose initiale avec
# les versions en lot :
#   - grille       : bilinéaire en lot sur l'image du score (extraite des
#     cellules de fermeture de score_fn, donc toujours synchrone avec le
#     socle) ;
#   - vectorielle  : le filtre est piloté par un champ de d² pré-calculé
#     (score interne en lot, permis par le contrat), puis les finalistes sont
#     polis sur un paysage interne complet ;
#   - nuage (score_nuage) : même schéma, le nuage étant traité comme des
#     segments dégénérés (points) — champ allégé pour le filtre, rappel
#     complet sur un nuage sous-échantillonné ;
#   - aucun ne colle : repli sur score_fn une pose à la fois.
# Resultat.score reste mesuré avec le score_fn fourni, à la pose finale.

import math
import time

import numpy as np

try:  # modules voisins (même dossier)
    import occupation as occ
except ImportError:  # importé comme paquet
    from . import occupation as occ

from algo_base import (CAP_DIST, MAX_PTS, TOL, Algorithme, Resultat,
                       echantillon, image_score_grille, incoherence)

GARDE = 0.75           # part des paires gardées (moyenne tronquée du socle)


# --------------------------------------------------------------------------- #
# Noyaux en lot (toutes les particules d'une génération d'un coup, float32)
# --------------------------------------------------------------------------- #
def _transforme(pts, part):
    """Points du scan (P, 2) transformés par toutes les poses (n, 3) : (n, P)."""
    p = part.astype(np.float32)
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    px = pts[:, 0].astype(np.float32)
    py = pts[:, 1].astype(np.float32)
    wx = (px[None, :] * c[:, None] - py[None, :] * s[:, None]) + p[:, None, 0]
    wy = (px[None, :] * s[:, None] + py[None, :] * c[:, None]) + p[:, None, 1]
    return wx, wy


def _bilineaire_lot(img, x0, y0, res, pts, hors):
    """Valeurs bilinéaires de img sous les points de chaque pose : (n, P).

    Même interpolation que algo_base._bilineaire ; `hors` = valeur hors carte
    (0 : un point qui sort de la carte doit coûter cher ; cap² pour le champ
    de distances : un écho loin de tout mur est déjà plafonné).
    """
    ny, nx = img.shape
    z = np.float32(hors)

    def echantillonne(part):
        wx, wy = _transforme(pts, part)
        gx = (wx - np.float32(x0)) / np.float32(res) - np.float32(0.5)
        gy = (wy - np.float32(y0)) / np.float32(res) - np.float32(0.5)
        flx, fly = np.floor(gx), np.floor(gy)
        ix, iy = flx.astype(np.int32), fly.astype(np.int32)
        fx, fy = gx - flx, gy - fly
        ok = (ix >= 0) & (ix < nx - 1) & (iy >= 0) & (iy < ny - 1)
        ic, jc = np.clip(ix, 0, nx - 2), np.clip(iy, 0, ny - 2)
        v = (img[jc, ic] * (1 - fx) * (1 - fy) + img[jc, ic + 1] * fx * (1 - fy)
             + img[jc + 1, ic] * (1 - fx) * fy + img[jc + 1, ic + 1] * fx * fy)
        return np.where(ok, v, z)

    return echantillonne


def _prep_segments(segs):
    """Précalculs float32 des segments : (aT, a_dot_ab, a2, abT, l2, inv_l2, cap2)."""
    a = np.asarray(segs[:, :2], dtype=np.float32)
    ab = np.asarray(segs[:, 2:4], dtype=np.float32) - a
    l2 = np.maximum((ab * ab).sum(1), np.float32(TOL))
    inv_l2 = np.float32(1.0) / l2
    return (np.ascontiguousarray(a.T), (a * ab).sum(1), (a * a).sum(1),
            np.ascontiguousarray(ab.T), l2, inv_l2, np.float32(CAP_DIST * CAP_DIST))


def _d2_seg_points(pre, monde):
    """d² plafonné de chaque point (N,) à chaque segment (M,) : (N, M) float32.

    Même distance que le score du socle (d² = |p-a|² - l2·(2·t·u - t²), avec
    u = (p-a)·ab/|ab|² non borné et t = u borné à [0, 1]), développée sans
    allouer de projections (N, M, 2) : produits matriciels, chunks sur les
    points pour borner la mémoire.
    shortcut: tableaux (k × M) intermédiaires, à chunker plus fin si la carte
    vectorielle grossit beaucoup.
    """
    aT, a_dot_ab, a2, abT, l2, inv_l2, cap2 = pre
    out = np.empty((len(monde), len(l2)), dtype=np.float32)
    pas = max(1, 2_000_000 // max(len(l2), 1))     # ~8 Mo par tableau
    for i in range(0, len(monde), pas):
        blk = monde[i:i + pas]
        p2 = np.einsum('ij,ij->i', blk, blk)                  # |p|²       (k,)
        dot = blk @ abT - a_dot_ab                            # (p-a)·ab   (k, M)
        u = dot * inv_l2                                      # t non borné
        t = np.clip(u, np.float32(0.0), np.float32(1.0))
        d2 = t * u
        d2 *= np.float32(-2.0)
        d2 += t * t
        d2 *= l2                                              # -l2·(2·t·u - t²)
        ap2 = blk @ aT                                        # p·a        (k, M)
        ap2 *= np.float32(-2.0)
        ap2 += p2[:, None]
        ap2 += a2[None, :]                                    # = |p-a|²
        d2 += ap2
        np.maximum(d2, np.float32(0.0), out=d2)   # arrondis près de zéro
        np.minimum(d2, cap2, out=d2)              # échos lointains plafonnés
        out[i:i + pas] = d2
    return out


# --------------------------------------------------------------------------- #
# Scores en lot exacts (identification de la référence, rappel des finalistes)
# --------------------------------------------------------------------------- #
def _tronquee(d2, garde):
    """-moyenne tronquée du socle : les `garde` meilleures paires par pose."""
    meilleures = np.partition(d2, garde - 1, axis=1)[:, :garde]
    return -meilleures.mean(axis=1, dtype=np.float64)


def _score_segments_lot(segs, pts, garde):
    """part (n, 3) -> score exact du socle pour chaque pose (n,), en lot."""
    pre = _prep_segments(segs)
    P = len(pts)

    def score(part):
        wx, wy = _transforme(pts, part)
        n = len(part)
        monde = np.empty((n * P, 2), dtype=np.float32)
        monde[:, 0] = wx.ravel()
        monde[:, 1] = wy.ravel()
        d2 = _d2_seg_points(pre, monde).reshape(n, -1)        # (n, P·M)
        return _tronquee(d2, garde)

    return score


def _score_segments_lot_plein(segs, pts):
    """part (n, 3) -> -d² moyen COMPLETE par point (n,), en lot (score interne).

    La moyenne tronquée du socle (meilleures paires point-segment) récompense
    les alignements qui ne cale qu'un seul mur quand la référence est
    partielle : le filtre interne utilise la moyenne complète, sensible à
    l'alignement de tous les points.
    """
    pre = _prep_segments(segs)
    P = len(pts)

    def score(part):
        wx, wy = _transforme(pts, part)
        n = len(part)
        monde = np.empty((n * P, 2), dtype=np.float32)
        monde[:, 0] = wx.ravel()
        monde[:, 1] = wy.ravel()
        d2 = _d2_seg_points(pre, monde).min(axis=1)           # segment le plus proche
        return -d2.reshape(n, -1).mean(axis=1, dtype=np.float64)

    return score


def _score_grille_lot(img, x0, y0, res, pts):
    """part (n, 3) -> occupation moyenne sous les points du scan (n,), en lot."""
    echant = _bilineaire_lot(img, x0, y0, res, pts, hors=0.0)

    def score(part):
        return echant(part).mean(axis=1, dtype=np.float64)

    return score


# --------------------------------------------------------------------------- #
# Champ de distances (score interne en lot pour piloter le filtre)
# --------------------------------------------------------------------------- #
_CHAMPS = {}   # id(segments) -> (segs, champ) : la réf gardée stabilise l'id


def _champ_segments(segs, res=0.04):
    """Champ des d² plafonnés aux segments, lu comme une grille d'occupation.

    Score interne en lot pour la référence vectorielle : une bilinéaire par
    génération au lieu d'un calcul exact par paire point-segment. Résolution
    fine (erreur d'interpolation ≪ bruit du filtre), bords à cap² (un écho
    loin de tout mur est déjà plafonné) ; run() re-mesure de toute façon les
    finalistes avec le score exact.
    """
    entree = _CHAMPS.get(id(segs))
    if entree is not None:
        return entree[1]
    segs = np.asarray(segs, dtype=float)[:, :4]
    lo = np.minimum(segs[:, :2].min(0), segs[:, 2:4].min(0)) - CAP_DIST
    hi = np.maximum(segs[:, :2].max(0), segs[:, 2:4].max(0)) + CAP_DIST
    nx = int(math.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(math.ceil((hi[1] - lo[1]) / res)) + 1
    pas_xy = np.float32(res)
    xs = np.float32(lo[0]) + (np.arange(nx, dtype=np.float32) + np.float32(0.5)) * pas_xy
    ys = np.float32(lo[1]) + (np.arange(ny, dtype=np.float32) + np.float32(0.5)) * pas_xy
    gx, gy = np.meshgrid(xs, ys)                           # centres des cases
    monde = np.stack((gx.ravel(), gy.ravel()), axis=1)
    d2 = _d2_seg_points(_prep_segments(segs), monde).min(axis=1)
    champ = (d2.reshape(ny, nx), float(lo[0]), float(lo[1]), res)
    _CHAMPS[id(segs)] = (segs, champ)
    return champ


def _score_champ_lot(champ, pts, plein=False):
    """part (n, 3) -> score lu dans le champ pré-calculé (n,), en lot.

    Par défaut moyenne complète des d² échantillonnés (`plein`) ; c'est le
    score interne qui pilote le filtre, la moyenne tronquée du socle ne sert
    qu'aux mesures rapportées.
    """
    img, x0, y0, res = champ
    echant = _bilineaire_lot(img, x0, y0, res, pts, hors=CAP_DIST * CAP_DIST)

    def score(part):
        return -echant(part).mean(axis=1, dtype=np.float64)

    return score


# --------------------------------------------------------------------------- #
# Nuage de points (référence « scan-contre-scan », algo_base.score_nuage)
# --------------------------------------------------------------------------- #
def _score_nuage_lot(nu, pts, garde):
    """part (n, 3) -> score exact du socle (score_nuage) par pose (n,), en lot.

    Chaque point du nuage est un « segment » dégénéré (extrémités confondues,
    ab nul) : la distance au point le plus proche tombe de la même primitive
    que les segments (u = 0, d² = |p-a|²). Charge mémoire en P·K paires :
    utilisé tel quel pour le sondage (une pose).
    """
    nu = np.asarray(nu, dtype=float)
    pre = _prep_segments(np.column_stack((nu, nu)))
    P = len(pts)

    def score(part):
        wx, wy = _transforme(pts, part)
        n = len(part)
        monde = np.empty((n * P, 2), dtype=np.float32)
        monde[:, 0] = wx.ravel()
        monde[:, 1] = wy.ravel()
        d2 = _d2_seg_points(pre, monde).reshape(n, -1)        # (n, P·K)
        return _tronquee(d2, garde)

    return score


def _score_nuage_lot_plein(nu, pts):
    """part (n, 3) -> -d² moyen COMPLETE aux points du nuage (n,), en lot.

    Score interne (moyenne complète, cf. _score_segments_lot_plein) : sert de
    rappel et de polissage, sur un nuage sous-échantillonné pour la mémoire.
    """
    nu = np.asarray(nu, dtype=float)
    pre = _prep_segments(np.column_stack((nu, nu)))
    P = len(pts)

    def score(part):
        wx, wy = _transforme(pts, part)
        n = len(part)
        monde = np.empty((n * P, 2), dtype=np.float32)
        monde[:, 0] = wx.ravel()
        monde[:, 1] = wy.ravel()
        d2 = _d2_seg_points(pre, monde).min(axis=1)           # point le plus proche
        return -d2.reshape(n, -1).mean(axis=1, dtype=np.float64)

    return score


def _champ_nuage(nuage, res=0.05, points_max=600):
    """Champ des d² aux points du nuage, lu comme une grille (score interne).

    Nuage sous-échantillonné pour borner la construction : le maximum du
    paysage est conservé (l'espacement des échantillons décale d² de qq cm)
    et run() re-mesure les finalistes avec le score exact en lot.
    """
    entree = _CHAMPS.get(id(nuage))
    if entree is not None:
        return entree[1]
    nu = np.asarray(nuage, dtype=float)
    nu = nu[::max(1, int(math.ceil(len(nu) / points_max)))]
    lo = nu.min(0) - CAP_DIST
    hi = nu.max(0) + CAP_DIST
    nx = int(math.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(math.ceil((hi[1] - lo[1]) / res)) + 1
    pas_xy = np.float32(res)
    xs = np.float32(lo[0]) + (np.arange(nx, dtype=np.float32) + np.float32(0.5)) * pas_xy
    ys = np.float32(lo[1]) + (np.arange(ny, dtype=np.float32) + np.float32(0.5)) * pas_xy
    gx, gy = np.meshgrid(xs, ys)
    monde = np.stack((gx.ravel(), gy.ravel()), axis=1)
    d2 = _d2_seg_points(_prep_segments(np.column_stack((nu, nu))), monde).min(axis=1)
    champ = (d2.reshape(ny, nx), float(lo[0]), float(lo[1]), res)
    _CHAMPS[id(nuage)] = (nu, champ)
    return champ


# --------------------------------------------------------------------------- #
# Identification de la référence incarnée par score_fn
# --------------------------------------------------------------------------- #
def _image_grille(score_fn, grille):
    """L'image (float32, inconnu -> 0) que le score du socle interpole.

    Extraite des cellules de fermeture de score_fn : toujours synchrone avec
    le socle, même s'il change. Sinon reconstruite comme algo_base.score_grille.
    """
    for cell in getattr(score_fn, "__closure__", None) or ():
        v = cell.cell_contents
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype == np.float32:
            return v                       # l'image du score (le scan est float64)
    proba = np.asarray(grille.proba)
    return np.where(proba == occ.UNKNOWN, 0, proba).astype(np.float32)


def _polisse(scoreur, meilleure, pas):
    """Descente de coordonnées autour de la meilleure pose (maximisation).

    Après l'amorçage global, le jitter décroissant du filtre ne remonte plus
    les derniers degrés/centimètres vers le maximum local : cette recherche
    de motif (6 évaluations en lot par itération) finit le rapprochement,
    pour un coût négligeable devant le filtre.
    """
    m = meilleure.copy()
    val = float(scoreur(m[None, :])[0])
    for _ in range(48):
        ameliore = False
        for axe in range(3):
            for sens in (1.0, -1.0):
                cand = m.copy()
                cand[axe] += sens * pas[axe]
                v = float(scoreur(cand[None, :])[0])
                if v > val:
                    m, val = cand, v
                    ameliore = True
        if not ameliore:
            pas = pas * 0.5
            if pas[0] < 2e-3 and pas[2] < 1e-3:      # 2 mm / 0,06° : fini
                break
    return m, val


def _scoreur_lot(score_fn, pose, contexte):
    """(scoreur en lot, scoreur exact de rappel, tag) ou None.

    Identifie la référence de score_fn en comparant sa valeur à la pose
    initiale avec les versions en lot exactes (moyenne tronquée du socle) :
    - grille      : bilinéaire en lot, identique au score partagé ;
    - vectorielle : champ de distances pour le filtre + paysage interne
      complet (sans troncature) en rappel/polissage ;
    - nuage       : champ de points pour le filtre + paysage interne complet
      sur un nuage allégé en rappel/polissage ;
    - aucun ne colle (contexte sans carte, score exotique) : None, et run()
      retombe sur les appels score_fn un à un.
    """
    pts = contexte.get("pts")
    if pts is None:
        return None
    # même sous-sélection du scan que le socle (alias public)
    pts = echantillon(pts)
    ref = float(score_fn(pose))
    tol = 1e-3 * max(1.0, abs(ref))
    grille = contexte.get("grille")
    if grille is not None:
        lot = _score_grille_lot(_image_grille(score_fn, grille),
                                grille.x0, grille.y0, grille.res, pts)
        if abs(float(lot(pose[None, :])[0]) - ref) <= tol:
            return lot, None, "grille"
    segments = contexte.get("segments")
    if segments is not None and len(np.asarray(segments)):
        garde = max(int(len(pts) * GARDE), 1)
        exact = _score_segments_lot(np.asarray(segments, float)[:, :4], pts, garde)
        if abs(float(exact(pose[None, :])[0]) - ref) <= tol:
            champ = _score_champ_lot(_champ_segments(segments), pts)
            return champ, _score_segments_lot_plein(
                np.asarray(segments, float)[:, :4], pts), "champ+plein"
    nuage = contexte.get("nuage")
    if nuage is not None and len(np.asarray(nuage)):
        garde = max(int(len(pts) * GARDE), 1)
        sondage = _score_nuage_lot(nuage, pts, garde)     # nuage complet, une pose
        if abs(float(sondage(pose[None, :])[0]) - ref) <= tol:
            champ = _score_champ_lot(_champ_nuage(nuage), pts)
            nu = np.asarray(nuage, dtype=float)           # nuage allégé pour le rappel
            exact = _score_nuage_lot_plein(
                nu[::max(1, int(math.ceil(len(nu) / 800.0)))], pts)
            return champ, exact, "nuage+plein"
    return None


class Particules(Algorithme):
    nom = "particules"
    references = ("grille", "vectorielle")
    n = 400
    sigma = (0.15, 0.15, math.radians(8.0))  # bruit initial d'échantillonnage
    generations = 6
    decroissance = 0.55                      # le bruit décroît par génération
    graine = 0                               # reproductible (tests, benchmark)
    exploration = 0.25   # part des particules globales (voir run) : départ dur
    portee = 0.50        # dispersion xy (m) de ces particules globales

    def run(self, score_fn, pose0, contexte):
        debut = time.perf_counter()
        rng = np.random.default_rng(self.graine)
        origine = np.asarray(pose0, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        part = origine + rng.normal(0.0, sigma, (self.n, 3))
        # amorçage global : une localisation scan-contre-scan peut partir à
        # ~180° de la pose cherchée, loin du bassin de départ (sigma ±8° ne
        # suffirait pas). Une fraction des particules balaie tous les caps
        # avec une large dispersion xy, le reste affine localement.
        n_glob = int(self.n * self.exploration)
        if n_glob > 0:
            part[:n_glob, 0:2] = origine[:2] + rng.normal(0.0, self.portee, (n_glob, 2))
            part[:n_glob, 2] = origine[2] + rng.uniform(-math.pi, math.pi, n_glob)
        choisie = _scoreur_lot(score_fn, origine, contexte)
        lot, exact, tag = choisie if choisie else (None, None, "boucle")
        histo, evals = [], 0
        meilleure, m_score = origine.copy(), -np.inf
        bassins = [origine.copy()]   # la pose de départ est toujours candidate
        for _ in range(self.generations):
            if lot is None:  # repli : score boîte noir, une pose à la fois
                s = np.array([float(score_fn(q)) for q in part])
            else:
                s = lot(part)               # toutes les particules d'un coup
            evals += self.n
            i = int(s.argmax())
            if s[i] > m_score:
                m_score, meilleure = s[i], part[i].copy()
            histo.append((meilleure.copy(), m_score))
            # trace des hypothèses : toute pose à ≤0,10 du max de la génération,
            # dédoublonnée (1 m / 25°). Le likelihood seul préfère parfois un
            # parasite (carte peu informative) — on garde les candidats pour les
            # départager à la fin par l'incohérence des rayons
            for i2 in np.argsort(-s):
                if s[i2] < s.max() - 0.10:
                    break
                p2 = part[i2]
                if all(np.hypot(*(p2[:2] - b[:2])) > 1.0
                       or abs((p2[2] - b[2] + math.pi) % (2 * math.pi) - math.pi)
                       > math.radians(25.0) for b in bassins):
                    bassins.append(p2.copy())
            del bassins[16:]
            w = (s - s.min()) / max(float(np.ptp(s)), 1e-9)
            w = w ** 4                        # accentue les meilleures poses
            w /= w.sum()
            choix = np.searchsorted(np.cumsum(w), rng.random(self.n), side="right")
            part = part[choix] + rng.normal(0.0, sigma * 0.3, (self.n, 3))
            sigma = sigma * self.decroissance
        if exact is not None:
            # rappel interne : la pose rapportée doit maximiser le paysage
            # complet (sensible à l'alignement de tous les points), pas celui
            # du champ interpolé qui a piloté le filtre
            k = min(32, self.n)
            finaux = part[np.argpartition(lot(part), len(part) - k)[-k:]]
            s_ex = exact(finaux)
            j = int(s_ex.argmax())
            if s_ex[j] > float(exact(meilleure[None, :])[0]):
                meilleure = finaux[j].copy()
            evals += k + 1
        if lot is not None:
            # polissage de chaque bassin distinct, sur le paysage en lot (le
            # champ interpolé garde le maximum ; le paysage exact complet
            # coûterait ~1 ms PAR évaluation × 16 candidats) : une pose mal
            # polie serait arbitée sur une incohérence surestimée
            pas_pol = np.array((0.06, 0.06, math.radians(5.0)))
            candidats = ([_polisse(lot, b, pas_pol)[0] for b in bassins]
                         if bassins else [meilleure])
        else:
            candidats = [meilleure]
        grille = contexte.get("grille")
        if grille is not None and candidats:
            if len(candidats) > 1:
                # arbitrage au score fiable (f - incohérence) ; quasi-égalités
                # (±0,08 — le bruit de l'incohérence, une zone aveugle de la
                # carte) départagées par le prior pose0 : même règle que
                # l'exhaustif. Un écart décisif ignore le prior.
                pts_in = contexte["pts"]

                def _combo(p):
                    return float(score_fn(p)) - incoherence(grille, pts_in, p)

                def _d0(p):
                    dth = abs((p[2] - origine[2] + math.pi) % (2 * math.pi)
                              - math.pi)
                    return dth + float(np.hypot(*(p[:2] - origine[:2])))

                comb = [_combo(c) for c in candidats]
                seuil = max(comb) - 0.08
                meilleure = min((c for c, v in zip(candidats, comb) if v >= seuil),
                                key=_d0)
            else:
                meilleure = candidats[0]
            # polissage multi-σ : le noyau large tolère ~10 cm de flottement
            # sur une carte fragmentée ; σ décroissant colle la géométrie
            pts_lot = echantillon(contexte["pts"])
            for sigma_f in (0.10, 0.06):
                img_f, ox, oy, ores = image_score_grille(grille, sigma_f)
                fin = _score_grille_lot(img_f, ox, oy, ores, pts_lot)
                meilleure, _ = _polisse(fin, meilleure,
                                        np.array((0.04, 0.04, math.radians(3.0))))
        return Resultat(meilleure, float(score_fn(meilleure)), histo, evals,
                        time.perf_counter() - debut,
                        particules=part, extra={"scoreur": tag})

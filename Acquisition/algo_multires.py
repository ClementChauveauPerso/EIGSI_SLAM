# algo_multires.py — Localisation multi-résolution.
#
# Pyramide de grilles d'occupation construite par max-pool depuis
# contexte["grille"] (pool 4×4 puis 2×2) : le niveau grossier balaye le lacet
# en GLOBAL (±180°, localisation sans indice d'orientation) sur la grille la
# plus grossière (coût par pose ÷16), chaque niveau affine sur une grille
# deux fois plus fine. Toutes les poses
# d'un niveau sont évaluées en numpy vectorisé : le scan est pré-tourné une
# fois par θ, puis toutes les translations sont évaluées d'un coup (plus
# aucune boucle python par pose). Le niveau fin affine sur les segments avec
# un score interne (d² plafonné + moyenne tronquée) qui corrige la saturation
# du score partagé au-delà de CAP_DIST (faux résultats sur la salle réelle) ;
# Resultat.score reste mesuré avec le score_fn fourni, à la pose finale.

import math
import time

import numpy as np

from algo_base import Algorithme, Resultat, _bilineaire, _echantillon

CAP2 = 0.50 ** 2    # plafond d² identique au score partagé (m²)
GARDE = 0.75        # moyenne tronquée : fraction de points les mieux appariés


def _pool(img, f):
    """Max-pool f×f (bord complété par 0 = libre) : le mur survit au pool."""
    py, px = (-img.shape[0]) % f, (-img.shape[1]) % f
    if py or px:
        img = np.pad(img, ((0, py), (0, px)))
    return img.reshape(img.shape[0] // f, f, img.shape[1] // f, f).max(axis=(1, 3))


def _scorer_grille(img, x0, y0, res):
    """Toutes les translations d'un coup : occupation bilinéaire moyenne."""
    def scorer(rot, trans, _th):
        monde = trans[:, None, :] + rot[None, :, :]
        v = _bilineaire(img, x0, y0, res, monde.reshape(-1, 2))
        return v.reshape(len(trans), -1).mean(1)
    return scorer


def _scorer_segments(segs):
    """Score interne : -d² moyenne tronquée aux segments les plus proches.

    La moyenne tronquée lâche les points aberrants et le plafond d² borne les
    autres : sans elle le score partagé sature au-delà de CAP_DIST et la
    localisation dérive (~25 cm) sur la salle réelle.
    """
    segs = np.asarray(segs, dtype=np.float32)[:, :4]
    a, ab = segs[:, :2], segs[:, 2:4] - segs[:, :2]
    l2 = np.maximum((ab * ab).sum(1), 1e-9)

    def scorer(rot, trans, _th):
        rot, trans = rot.astype(np.float32), trans.astype(np.float32)
        monde = trans[:, None, :] + rot[None, :, :]                 # (P, K, 2)
        ap = monde[:, :, None, :] - a                               # (P, K, M, 2)
        u = np.einsum("pkmi,mi->pkm", ap, ab)
        t = np.clip(u / l2, 0.0, 1.0)
        d2 = (ap * ap).sum(-1) - t * (2 * u - t * l2)               # (P, K, M)
        d2 = d2.min(-1)                                             # (P, K)
        k = max(1, min(int(d2.shape[1] * GARDE), d2.shape[1] - 1))
        d2 = np.partition(d2, k, axis=1)[:, :k]                     # meilleurs pts
        return -np.minimum(d2, CAP2).mean(1)
    return scorer


def _scorer_fn(score_fn):
    """Repli : ni grille ni segments dans le contexte -> score boîte noire."""
    def scorer(_rot, trans, th):
        return np.array([score_fn(np.array((tx, ty, th))) for tx, ty in trans],
                        dtype=np.float32)
    return scorer


class Multires(Algorithme):
    nom = "multi-rés."
    references = ("grille", "vectorielle")
    rayons = (0.6, 0.6, math.pi)               # fenêtre du niveau grossier :
    pas = (0.20, 0.20, math.radians(4.0))      # lacet GLOBAL (±180°), trans locale
    niveaux = 3
    rapport = 4.0                              # fenêtre/pas ÷ ~4 par niveau
    # étages (fenêtre, pas, facteur de pool de la grille, points du scan)
    ETAGES = (
        (rayons, pas, 4, 90),
        ((0.20, 0.20, math.radians(6.0)), (0.025, 0.025, math.radians(1.0)), 2, 72),
        ((0.10, 0.10, math.radians(2.0)), (0.0125, 0.0125, math.radians(0.5)), 1, 180),
    )
    ETAGE_SEG = ((0.10, 0.10, math.radians(3.0)),      # étage final sur segments
                 (0.025, 0.025, math.radians(1.5)), 0, 75)

    @staticmethod
    def _niveau(pts, base, rayons, pas, scorer):
        """Exhaustif vectorisé d'un étage : (pose, score, n_évals, paysage)."""
        axes = [base[k] + np.arange(-int(round(rayons[k] / pas[k])),
                                    int(round(rayons[k] / pas[k])) + 1) * pas[k]
                for k in range(3)]
        sc3d = np.empty((len(axes[0]), len(axes[1]), len(axes[2])), dtype=np.float32)
        trans = np.stack(np.meshgrid(axes[0], axes[1], indexing="ij"),
                         axis=-1).reshape(-1, 2)
        for it, th in enumerate(axes[2]):
            c, s = math.cos(th), math.sin(th)
            rot = pts @ np.array(((c, s), (-s, c)))
            sc3d[:, :, it] = scorer(rot, trans, th).reshape(sc3d.shape[:2])
        i = np.unravel_index(int(sc3d.argmax()), sc3d.shape)
        pose = np.array((axes[0][i[0]], axes[1][i[1]], axes[2][i[2]]))
        paysage = (axes[0], axes[1], axes[2][i[2]], sc3d[:, :, i[2]])
        return pose, float(sc3d[i]), int(sc3d.size), paysage

    def run(self, score_fn, pose0, contexte):
        debut = time.perf_counter()
        pose = np.asarray(pose0, dtype=float).copy()
        pts = np.asarray(contexte["pts"], dtype=float)
        grille = contexte.get("grille")
        segs = contexte.get("segments")
        if grille is not None:
            fine = np.asarray(grille.proba, dtype=np.float32)
            pyr = [fine, _pool(fine, 2), _pool(_pool(fine, 2), 2)]   # ÷1, ÷2, ÷4

        def preparer(etage):
            _fen, _pas, f, _n = etage
            if f and grille is not None:
                img = pyr[int(math.log2(f))]
                return _scorer_grille(img, grille.x0, grille.y0, grille.res * f)
            if segs is not None:
                return _scorer_segments(segs)
            return _scorer_fn(score_fn)

        etages = (self.ETAGES[0], self.ETAGES[1],
                  self.ETAGE_SEG if segs is not None else self.ETAGES[2])
        histo, evals, paysage = [], 0, None
        for etage in etages:
            fen, pas, _f, n = etage
            pose, s, ne, paysage = self._niveau(_echantillon(pts, n), pose,
                                                fen, pas, preparer(etage))
            evals += ne
            histo.append((pose.copy(), s))
        score = float(score_fn(pose))
        histo.append((pose.copy(), score))
        return Resultat(pose, score, histo, evals + 1,
                        time.perf_counter() - debut, paysage=paysage)

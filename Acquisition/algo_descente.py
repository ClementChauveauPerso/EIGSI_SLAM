# algo_descente.py — Localisation par descente de gradient (numérique).
#
# Suit le gradient du score depuis l'estimation initiale. Rapide (quelques
# dizaines d'évaluations) mais ne voit que son bassin d'attraction : il faut
# démarrer près de la bonne pose. Le pas se fait selon le SIGNe du gradient :
# insensible aux échelles très différentes de x, y (m) et θ (rad).
#
# Robustesse (le score brut a des plateaux qui bloquent la descente : cases
# inconnues codées 127 = constante entre 0 et 255, murs fins sur une grille de
# 10 cm, distances plafonnées à CAP_DIST) :
#   1. phase grossière : descente sur un score interne LISSÉ construit depuis
#      contexte["grille"] (inconnu ramené à 0, box blur 2×(fenêtre 7), 60
#      points) — les murs émettent du signal autour d'eux, le gradient est non
#      nul même à ~25 cm ;
#   2. phase de finition : descente sur un score interne des segments (mêmes
#      maths que score_segments, 60 points -> ~3× moins cher que le score
#      boîte noire) qui converge au mm près vers le même optimum.
#      Régime scan-contre-scan (contexte["nuage"] présent) : les scores d'un
#      scan unique sont plats ou biaisés (nuage tronqué, quelques segments
#      partiels) -> finition sur une réplique allégée de score_nuage, puis
#      recalage angulaire + mini-translation sur la grille lissée dont le
#      paysage est propre ; toutes les références convergent alors au même
#      point (le meilleur des contrôles réels n'est rendu que sur carte
#      assemblée, où score_fn est un juge fiable).
# Gradient central, pas adaptatifs PAR AXE (×1.4 si le pas est accepté,
# ×0.6 sinon), re-décollage quand bloqué, arrêt sur déplacement marginal ou
# après quelques échecs. Le score RÉEL (score_fn) n'est mesuré qu'aux points
# de contrôle (départ, fin de chaque phase) et Resultat.pose/.score sont
# ceux du meilleur de ces contrôles : jamais pire que le départ. (Les scores
# de l'historique sont ceux du proxy suivi à chaque itération.)
# Budget ≤ ~300 évaluations de poses.

import math
import time

import numpy as np

try:  # modules voisins (même dossier)
    import cartographie as carto
    import occupation as occ
    from algo_base import (Algorithme, CAP_DIST, Resultat, _bilineaire,
                           _echantillon)
except ImportError:  # importé comme paquet
    from . import cartographie as carto
    from . import occupation as occ
    from .algo_base import (Algorithme, CAP_DIST, Resultat, _bilineaire,
                            _echantillon)


def _lisse(img, r=3, passes=2):
    """Box blur séparable (fenêtre 2r+1, padding nul) ; 2 passes ≈ gaussien."""
    img = np.asarray(img, dtype=np.float32)
    for _ in range(passes):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r, r)
            p = np.pad(img, pad)
            n = img.shape[axis]
            sl = [slice(None), slice(None)]
            acc = np.zeros_like(img)
            for k in range(-r, r + 1):
                sl[axis] = slice(r + k, r + k + n)
                acc += p[tuple(sl)]
            img = acc / float(2 * r + 1)
    return img


def _score_grille_lisse(grille, pts_local, r=3, passes=2):
    """Score interne de la phase grossière : occupation (inconnu -> 0) floutée.
    Proxy lisse et SANS plateau du score_grille : le gradient tire le scan vers
    les murs même à ~25 cm. Resultat.score reste mesuré avec score_fn.
    (r/passes petits : peu lissée, pour une finition fidèle.)"""
    proba = grille.proba
    img = np.where(proba == occ.UNKNOWN, 0, proba).astype(np.float32) / 255.0
    img = _lisse(img, r=r, passes=passes)
    x0, y0, res = grille.x0, grille.y0, grille.res
    pts = _echantillon(pts_local, 60)

    def fn(pose):
        monde = carto.transform(pts, pose)
        return float(_bilineaire(img, x0, y0, res, monde).mean())

    return fn


def _score_segments_lisse(segments, pts_local):
    """Score interne de la phase de finition : -d² moyenne aux segments les
    plus proches (mêmes maths que score_segments, mais sur 60 points -> ~3×
    moins cher par évaluation, même optimum géométrique)."""
    segs = np.asarray(segments, dtype=float)[:, :4]
    pts = _echantillon(pts_local, 60)
    a, ab = segs[:, :2], segs[:, 2:4] - segs[:, :2]
    l2 = np.maximum((ab * ab).sum(1), 1e-9)
    cap2 = CAP_DIST * CAP_DIST

    def fn(pose):
        monde = carto.transform(pts, pose)
        ap = monde[:, None, :] - a[None, :, :]                  # (P, M, 2)
        t = np.clip((ap * ab[None, :, :]).sum(-1) / l2[None, :], 0.0, 1.0)
        proj = a[None, :, :] + t[..., None] * ab[None, :, :]
        d2 = ((monde[:, None, :] - proj) ** 2).sum(-1)
        return -float(np.minimum(d2, cap2).mean())

    return fn


class Descente(Algorithme):
    nom = "gradient"
    references = ("grille", "vectorielle")
    pas = (0.05, 0.05, math.radians(2.0))   # pas initial (x, y, θ)
    iters = 80
    decroissance = 0.6                      # décroissance du pas sur échec
    budget = 300                            # garde-fou : nb max d'évaluations

    def _fn_interne(self, contexte, grossier):
        """Score interne pour la phase grossière (grille lissée) ou de
        finition (segments légers). (None, False) -> score_fn réel."""
        grille, segments = contexte.get("grille"), contexte.get("segments")
        if grossier:
            # pas de grille -> phase 1 sur le score réel (il converge de loin)
            return ((_score_grille_lisse(grille, contexte["pts"]), True)
                    if grille is not None else (None, False))
        nuage = contexte.get("nuage")        # scan-contre-scan : finition sur
        if nuage is not None and len(nuage):
            # une réplique du score_nuage du socle (mêmes 180 points, même
            # troncature 75 %) avec le nuage sous-échantillonné d'un sur deux
            # -> même optimum géométrique pour toutes les références (celui
            # du score_fn peut être biaisé, ex. 7 segments partiels), ~3×
            # moins cher par évaluation.
            # shortcut: nuage réduit d'un sur deux, à revoir si des scans
            # très denses décalent l'optimum.
            pts = _echantillon(contexte["pts"])
            nu = np.asarray(nuage, dtype=float)[::2]
            cap2 = CAP_DIST * CAP_DIST
            garde = max(int(pts.shape[0] * 0.75), 1)

            def fn(pose):
                monde = carto.transform(pts, pose)
                d2 = ((monde[:, None, :] - nu[None, :, :]) ** 2).sum(-1)
                pire = np.partition(np.minimum(d2, cap2).ravel(),
                                    garde - 1)[:garde]
                return -float(pire.mean())

            return fn, True
        if segments is not None and len(segments):
            return _score_segments_lisse(segments, contexte["pts"]), True
        if grille is not None:               # finition sur grille peu floutée
            return _score_grille_lisse(grille, contexte["pts"]), True
        return None, False

    def run(self, score_fn, pose0, contexte):
        debut = time.perf_counter()
        pose0 = np.asarray(pose0, dtype=float)
        s0 = float(score_fn(pose0))
        histo = [(pose0.copy(), s0)]
        evals = [1]                          # boîte mutable partagée

        # Phase 1 — grossière : score lissé sans plateau, plancher ~1 cm.
        # Itérations bornées : près du sommet plat du score lissé, la descente
        # ping-pong sans progrès ; le meilleur état vu est déjà le bon.
        fn1, _ = self._fn_interne(contexte, grossier=True)
        if fn1 is None:
            fn1 = score_fn
        p1, _ = self._descente(fn1, pose0, histo, evals,
                               np.asarray(self.pas, float) * 1.2,
                               np.array((0.008, 0.008, math.radians(0.4))),
                               evals[0] + 150, red_max=1, iters_max=16)

        # Phase 2 — finition : score_nuage répliqué si présent (scan-contre-
        # scan : même optimum géométrique quelle que soit la référence du
        # score_fn, dont le max peut être biaisé — ex. 7 segments partiels),
        # sinon segments (carte assemblée), plancher ~2 mm. Petits pas : la
        # phase 1 amène déjà dans le bon bassin, ici on finit sans dériver.
        fn2, _ = self._fn_interne(contexte, grossier=False)
        if fn2 is None:
            fn2 = score_fn
        p2, _ = self._descente(fn2, p1, histo, evals,
                               np.asarray(self.pas, float) * 0.25,
                               np.array((0.002, 0.002, math.radians(0.1))),
                               evals[0] + 80, red_max=0, iters_max=8)
        s2 = float(score_fn(p2))
        evals[0] += 1
        # Régime scan-contre-scan (nuage présent) : le score_fn d'un scan
        # unique est plat ou biaisé (nuage tronqué, 7 segments partiels), on
        # recale donc l'angle sur la grille lissée (paysage θ propre), puis
        # mini-finition (x, y) sur une grille peu lissée et fidèle.
        # ~25 évaluations. Sans nuage (carte assemblée), la phase 2 est
        # déjà fidèle : on ne touche à rien.
        grille = contexte.get("grille")
        if grille is not None and contexte.get("nuage") is not None:
            fn_a = _score_grille_lisse(grille, contexte["pts"])
            p2 = self._recalage_angle(fn_a, p2, evals)
            fn_tr = _score_grille_lisse(grille, contexte["pts"], r=1, passes=1)
            s_a = float(fn_tr(p2))
            for _ in range(2):
                g = np.zeros(2)
                for d in range(2):           # gradient central (x, y) seuls
                    a, b = p2.copy(), p2.copy()
                    a[d] += 0.0075
                    b[d] -= 0.0075
                    g[d] = fn_tr(a) - fn_tr(b)
                evals[0] += 4
                if np.any(g):
                    essai = p2.copy()
                    essai[:2] += 0.015 * np.sign(g)
                    s_e = float(fn_tr(essai))
                    evals[0] += 1
                    if s_e > s_a:
                        p2, s_a = essai, s_e
            s2 = float(score_fn(p2))
            evals[0] += 1
        else:
            # Carte assemblée : score_fn est un juge fiable entre la phase 1
            # (grille lissée) et la phase 2 (segments) -> on rend le meilleur.
            s1 = float(score_fn(p1))
            evals[0] += 1
            if s1 > s2:
                p2, s2 = p1, s1
        return Resultat(p2, s2, histo, evals[0], time.perf_counter() - debut)

    def _recalage_angle(self, fn, pose, evals):
        """Descente ANGLE SEUL sur fn (grille lissée) : gradient central,
        pas 3° puis 1.5°, 2 pas par échelle. Ne touche ni x ni y. Retourne la
        pose avec l'angle recalé (inchangée si le gradient est nul)."""
        s = float(fn(pose))
        evals[0] += 1
        for pas_a in (math.radians(3.0), math.radians(1.5)):
            for _ in range(2):
                a, b = pose.copy(), pose.copy()
                a[2] = carto.wrap_angle(a[2] + pas_a)
                b[2] = carto.wrap_angle(b[2] - pas_a)
                g = float(fn(a)) - float(fn(b))
                evals[0] += 2
                if g:
                    essai = pose.copy()
                    essai[2] = carto.wrap_angle(
                        pose[2] + math.copysign(pas_a, g))
                    s_e = float(fn(essai))
                    evals[0] += 1
                    if s_e > s:
                        pose, s = essai, s_e
        return pose

    def _descente(self, fn, pose0, histo, evals, pas, pas_min, budget_max,
                  red_max, iters_max, croissance=1.4, plafond=1.5):
        """Descente sur fn depuis pose0. Gradient numérique central par axe,
        pas par axe (×croissance si accepté, ×decroissance sinon, plafonnés à
        plafond×pas initial), re-décollage (≤ red_max) après 3 échecs, arrêt
        après quelques échecs pas réduit ou sur déplacement marginal.
        Retourne (pose, score) du MEILLEUR état vu."""
        pose = np.asarray(pose0, float).copy()
        s = float(fn(pose))
        evals[0] += 1
        meilleur = (pose.copy(), s)
        pas0 = pas.copy()
        bloques, redecolles = 0, 0
        seuil = 3 if red_max else 2          # échecs consécutifs -> arrêt
        for _ in range(min(self.iters, iters_max)):
            if evals[0] >= budget_max:
                break
            grad = np.zeros(3)
            for d in range(3):               # gradient numérique central
                h = 0.5 * pas[d]
                a, b = pose.copy(), pose.copy()
                a[d] += h
                b[d] -= h
                grad[d] = fn(a) - fn(b)
            evals[0] += 6
            dir_ = np.sign(grad)
            ameliore = False
            if np.any(dir_):
                essai = pose + pas * dir_
                essai[2] = carto.wrap_angle(essai[2])
                s2 = float(fn(essai))
                evals[0] += 1
                if s2 > s:
                    depl = np.abs(essai - pose)
                    pose, s = essai, s2
                    ameliore = True
                    pas = np.minimum(pas * croissance, pas0 * plafond)
                    bloques = 0
                    if s > meilleur[1]:
                        meilleur = (pose.copy(), s)
                    if np.all(depl < 0.5 * pas_min):
                        break                # déplacement marginal : convergé
            if not ameliore:
                bloques += 1
                pas = np.maximum(pas * self.decroissance, pas_min)
                if red_max and bloques >= 3 and redecolles < red_max \
                        and np.any(pas < pas0):
                    pas = np.minimum(pas0, pas * 3.0)   # re-décollage
                    redecolles += 1
                    bloques = 0
                elif bloques >= seuil and np.all(pas <= 0.7 * pas0):
                    break                    # pas réduit sans succès : fini
            histo.append((pose.copy(), s))
        return meilleur

# algo_direct.py — Localisation par calcul direct sur la carte vectorielle.
#
# ICP point-vers-ligne EN FORME FERMÉE : à chaque itération, chaque point du
# scan est apparié à son segment le plus proche (projection orthogonale), puis
# la pose qui minimise la somme des distances aux droites porteuses s'obtient
# exactement par moindres carrés linéaires (système 3×3 en δx, δy, δθ) — pas
# de recherche, une pose exacte par itération. Contrairement au Procrustes
# point-vers-point, la contrainte est uniquement perpendiculaire aux murs :
# les murs partiels ou décalés (dérive d'assemblage de la carte réelle) ne
# tirent plus la solution le long des murs. Robustesse : le rayon
# d'appariement (gate) décroît d'un facteur `decay` par itération, de `cap`
# (large, rattrape les 25 cm / 7° du départ) jusqu'à `cap_min` (serré : rejette
# les échos loin des murs et fige la pose) ; garde-fous : minimum de
# correspondances, limite de pas (région de confiance), arrêt sur déplacement
# < 1e-4 et au plus `iters` itérations.
#
# Mode scan-contre-scan (si contexte["nuage"] fourni) : une carte vectorielle
# réduite à quelques segments est quasi-symétrique (salle ~rectangulaire) et
# crée des bassins parasites tournés de ~90°, indiscernables par le score
# segments. Si le prior (pose0 raffiné) colle mal au nuage de référence
# (médiane des distances au plus proche voisin > `med_max`), on explore les
# 12 hypothèses d'orientation (pas 30°) par chaînes ICP à gate lent, on
# sélectionne celle qui colle au NUAGE (la médiane au plus proche voisin
# discrimine largement les bassins), puis on affine le gagnant par Procrustes
# point-à-point nuage-à-nuage (forme fermée, SVD). Sans "nuage" (contrat
# carte), rien de tout cela ne s'exécute.
# Fonctionne uniquement sur la carte vectorielle.

import math
import time

import numpy as np

try:
    import cartographie as carto
except ImportError:
    from . import cartographie as carto

from algo_base import Algorithme, Resultat, CAP_DIST, _echantillon


class Direct(Algorithme):
    nom = "direct"
    references = ("vectorielle",)
    iters = 30
    cap = CAP_DIST          # gate initial (m), décroît d'un facteur decay/itération
    cap_min = 0.08          # gate final : correspondances quasi sur les murs
    decay = 0.7
    # mode scan-contre-scan :
    med_max = 0.12          # médiane nuage (m) tolérée avant de suspecter le prior
    pas_theta = math.radians(30.0)   # pas de la grille d'orientations explorées
    iters_large = 50        # chaînes d'exploration : gate lent (voyage plus loin)
    decay_large = 0.95
    cap_min_large = 0.15

    @staticmethod
    def _geometrie(segs):
        """(a, ab, l2, normales) précalculés pour toutes les itérations."""
        a = segs[:, :2]
        ab = segs[:, 2:4] - a
        l2 = np.maximum((ab * ab).sum(1), 1e-12)
        n = np.column_stack((-ab[:, 1], ab[:, 0])) / np.sqrt(l2)[:, None]
        return a, ab, l2, n

    def _chaine(self, pts, geo, pose0, cap0, decay, cap_min, iters):
        """Une chaîne ICP point-vers-ligne ; retourne (pose, historique, évals)."""
        a, ab, l2, n = geo
        pose = np.asarray(pose0, dtype=float).copy()
        ordre = np.arange(len(pts))
        histo, evals = [], 0
        for k in range(iters):
            cap = max(cap0 * decay ** k, cap_min)
            # transform (rotation puis translation, convention row-vector)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            monde = pts @ np.array(((c, s), (-s, c))) + pose[:2]
            # d² de chaque point à chaque segment (sans tableau (N, M, 2)) :
            ap = monde[:, None, :] - a[None, :, :]
            u = (ap * ab[None, :, :]).sum(-1) / l2[None, :]     # t non borné
            tn = np.clip(u, 0.0, 1.0)
            d2 = (ap * ap).sum(-1) - l2[None, :] * (2.0 * tn * u - tn * tn)
            idx = d2.argmin(1)
            tsel = tn[ordre, idx]           # t de la projection retenue, par point
            # score interne : même formule que score_segments sur ces points
            score = -float(np.minimum(d2, CAP_DIST * CAP_DIST).mean())
            histo.append((pose.copy(), score))
            evals += 1
            d = np.sqrt(np.maximum(d2[ordre, idx], 0.0))
            ok = d < cap
            if int(ok.sum()) < 5:
                break                       # correspondances insuffisantes
            i = idx[ok]
            p = monde[ok]
            ni = n[i]
            # résidu e = n·(p + δt + δθ·J(p − t) − proj) ; MC linéaire → 3×3
            b = (ni * (p - (a[i] + tsel[ok, None] * ab[i]))).sum(1)
            j = np.column_stack((-(p[:, 1] - pose[1]), p[:, 0] - pose[0]))
            aa = np.column_stack((ni, (ni * j).sum(1)))         # (K, 3)
            try:
                delta = np.linalg.solve(aa.T @ aa, -(b @ aa))
            except np.linalg.LinAlgError:                       # système singulier
                break
            if np.hypot(delta[0], delta[1]) > cap or abs(delta[2]) > 0.35:
                break                       # région de confiance : pas aberrant
            pose = np.array((pose[0] + delta[0], pose[1] + delta[1],
                             carto.wrap_angle(pose[2] + delta[2])))
            if max(float(np.hypot(delta[0], delta[1])), abs(delta[2])) < 1e-4:
                break
        return pose, histo, evals

    @staticmethod
    def _med_nuage(nuage, pts, pose):
        """Médiane des distances au plus proche voisin du nuage de référence."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        monde = pts @ np.array(((c, s), (-s, c))) + pose[:2]
        d2 = ((monde[:, None, :] - nuage[None, :, :]) ** 2).sum(-1)
        return float(np.median(np.sqrt(d2.min(1))))

    def _icp_nuage(self, nuage, pts, pose0):
        """Finition Procrustes point-à-point (forme fermée, SVD) vers le nuage."""
        pose = np.asarray(pose0, dtype=float).copy()
        for k in range(self.iters):
            gate = max(0.5 * 0.85 ** k, 0.10)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            m = pts @ np.array(((c, s), (-s, c))) + pose[:2]
            ap = m[:, None, :] - nuage[None, :, :]
            d2 = (ap * ap).sum(-1)
            j = d2.argmin(1)
            d = np.sqrt(np.maximum(d2[np.arange(len(m)), j], 0.0))
            ok = d < gate
            if int(ok.sum()) < 20:
                break
            p, q = m[ok], nuage[j[ok]]
            w = (1.0 - d[ok] / gate) ** 2
            sw = w.sum()
            c_p, c_q = w @ p / sw, w @ q / sw
            u, _sv, vt = np.linalg.svd((p - c_p).T @ (w[:, None] * (q - c_q)))
            corr = np.diag((1.0, np.sign(np.linalg.det(vt.T @ u.T))))
            r = vt.T @ corr @ u.T
            phi = math.atan2(r[1, 0], r[0, 0])
            t_nouv = r @ pose[:2] + (c_q - r @ c_p)
            depl = max(float(np.hypot(*(t_nouv - pose[:2]))), abs(phi))
            pose = np.array((t_nouv[0], t_nouv[1], carto.wrap_angle(pose[2] + phi)))
            if depl < 1e-4:
                break
        return pose

    def run(self, score_fn, pose0, contexte):
        debut = time.perf_counter()
        segs = np.asarray(contexte["segments"], dtype=float)[:, :4]
        pts_bruts = np.asarray(contexte["pts"], dtype=float)
        geo = self._geometrie(segs)
        pose0 = np.asarray(pose0, dtype=float).copy()

        # chaîne principale (échantillon léger, paramètres serrés) — contrat carte
        pts = _echantillon(pts_bruts)
        pose, histo, evals = self._chaine(pts, geo, pose0,
                                          self.cap, self.decay, self.cap_min, self.iters)

        nuage = contexte.get("nuage")
        if nuage is not None:
            nuage = _echantillon(np.asarray(nuage, dtype=float), 300)
            pts_med = _echantillon(pts_bruts, 300)
            med = self._med_nuage(nuage, pts_med, pose)
            evals += 1
            if med > self.med_max:
                # le prior colle mal au nuage : salle quasi-symétrique, mauvais
                # bassin -> exploration des orientations, gate lent, tous points
                pts_tous = pts_bruts
                meilleur, histo, med_min = pose, histo, med
                for k in range(1, 12):
                    d0 = pose0 + np.array((0.0, 0.0, k * self.pas_theta))
                    cand, h, ev = self._chaine(pts_tous, geo, d0, self.cap,
                                               self.decay_large,
                                               self.cap_min_large, self.iters_large)
                    evals += ev
                    m = self._med_nuage(nuage, pts_med, cand)
                    evals += 1
                    if m < med_min:
                        meilleur, histo, med_min = cand, h, m
                    if m < self.med_max * 0.5:
                        break               # correspondance nuage franchement bonne
                pose = meilleur
            # finition Procrustes nuage-à-nuage (précision sub-cm)
            pose = self._icp_nuage(nuage, pts_med, pose)

        # score final mesuré avec le score_fn fourni (comparabilité inter-méthodes)
        score = float(score_fn(pose))
        evals += 1
        if not histo or not np.array_equal(histo[-1][0], pose):   # pose déjà notée si sortie sans update
            histo.append((pose.copy(), score))
        return Resultat(pose, score, histo, evals, time.perf_counter() - debut)

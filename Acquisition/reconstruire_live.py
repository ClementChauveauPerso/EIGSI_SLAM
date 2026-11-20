# reconstruire_live.py — Rejoue un dossier de scans bruts (exportés par temps_reel
# dans <live>/scans/) et régénère l'occupation + le vectoriel complet, comme si
# c'était un nouveau live. Usage :
#   python reconstruire_live.py Data\Cartes\Live_20261104_190647\scans
#
# La chaîne est identique à temps_reel : consolidation par scan, localisation
# scan-contre-scan (exhaustif via le registre), mise à jour de CarteLive
# (oubli exponentiel), puis export carte.csv / occupation.csv / vectoriel.csv.
import sys, math, time
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import cartographie as carto
import algo_base as base
import occupation as occ
import vectorielle as vec
import scan_matching as sm
from temps_reel import (CarteLive, RES, COTE, RHO, MAX_DELTA_SCORE_BON,
                        CARTE_SCORE_MIN, QUALITE_MIN, QUALITE_REL,
                        SEUIL_DECAY, ALGO, SIGMA)

# Les scans d'un replay sont consécutifs : le robot ne peut pas « sauter ».
PAS_MAX_D_M = 0.5        # déplacement max entre deux scans acceptés (m)
PAS_MAX_DEG = 40.0       # rotation max entre deux scans acceptés (°)
RECAL_MAX_D_M = 0.30     # passe 2 : correction max autorisée par scan (m)
RECAL_MAX_DEG = 10.0     # passe 2 : correction max autorisée par scan (°)

from algo_base import _echantillon, _bilineaire
from scipy.ndimage import distance_transform_edt


def charger_scan_csv(chemin):
    """Lit un CSV scan_XXXX.csv (N,Angle,Distance,Intensity,x,y) -> (pts, inten)."""
    data = np.loadtxt(chemin, delimiter=",", skiprows=1, usecols=(1, 2, 3),
                      ndmin=2, encoding="utf-8-sig")
    angle, dist, inten = data.T
    keep = dist > 0
    r, th = dist[keep] / 1000.0, np.deg2rad(angle[keep])
    return np.column_stack((r * np.cos(th), r * np.sin(th))), inten[keep]


def qualite(pts, grille):
    """Jauge d'appariement 0..1 (même logique que Vue._qualite)."""
    occupe = np.asarray(grille.proba) >= 200
    if not occupe.any():
        return lambda pose: 1.0
    d = distance_transform_edt(~occupe) * grille.res
    img = np.exp(-(np.minimum(d, 1.0) ** 2) / (2.0 * SIGMA ** 2))
    pts = _echantillon(pts)

    def score(pose):
        monde = carto.transform(pts, pose)
        valeurs = _bilineaire(img, grille.x0, grille.y0, grille.res,
                              monde, hors=0.0)
        connus = valeurs > 0.0
        if not connus.any():
            return 1.0
        return float(valeurs[connus].mean())
    return score


def ecart(a, b):
    """(distance en m, écart d'angle en °) entre deux poses."""
    dp = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return (math.hypot(dp[0], dp[1]),
            abs(math.degrees(carto.wrap_angle(dp[2]))))


def main():
    if len(sys.argv) < 2:
        print("Usage : python reconstruire_live.py <dossier_de_scans>")
        sys.exit(1)

    scans_dir = Path(sys.argv[1])
    if not scans_dir.is_dir():
        print(f"Dossier introuvable : {scans_dir}")
        sys.exit(1)

    fichiers = sorted(scans_dir.glob("scan_*.csv"))
    if not fichiers:
        print(f"Aucun scan_*.csv dans {scans_dir}")
        sys.exit(1)

    print(f"Replay de {len(fichiers)} scans depuis {scans_dir}")

    carte = CarteLive()
    algo = next(a for a in sm.ALGORITHMES if a.nom == ALGO)            # global
    local = next(a for a in sm.ALGORITHMES if a.nom == "gradient")     # continu
    pose = np.zeros(3)
    archives = deque()
    scores, echecs, refuses = [], 0, 0
    t0 = time.perf_counter()

    for k, f in enumerate(fichiers):
        pts, inten = charger_scan_csv(f)
        if not len(pts):
            continue

        score = None
        if carte.grille is not None:
            fn = qualite(pts, carte.grille)
            ctx = {"pts": pts, "grille": carte.grille}
            # 1) suivi continu : on repart de la dernière pose (le scan est
            #    pris à quelques cm / degrés du précédent).
            res = local.run(fn, pose.copy(), ctx)
            cand, score = res.pose, float(fn(res.pose))
            seuil = max(QUALITE_MIN, QUALITE_REL * (float(np.median(scores))
                        if scores else 0.0) * SEUIL_DECAY ** echecs)
            # 2) seulement si le suivi est mauvais : recherche globale. Elle
            #    balaye ±180° de cap et peut tomber sur un faux optimum
            #    (salle rectangulaire = symétrique) : son résultat reste soumis
            #    au verrou de saut ci-dessous.
            if score < max(seuil, CARTE_SCORE_MIN):
                res = algo.run(fn, pose.copy(), ctx)
                s2 = float(fn(res.pose))
                if s2 > score:
                    cand, score = res.pose, s2
            d_dist, d_deg = ecart(cand, pose)
            saut = d_dist > PAS_MAX_D_M or d_deg > PAS_MAX_DEG
            if (not np.isfinite(score) or not np.isfinite(cand).all()
                    or score < max(seuil, CARTE_SCORE_MIN)
                    or (saut and score < MAX_DELTA_SCORE_BON)):
                # scan rejeté : ni la pose ni la carte ne bougent
                refuses += 1
                echecs += 1
                scores.append(score if np.isfinite(score) else 0.0)
                print(f"  scan {k+1} rejeté : score {score:.2f}, "
                      f"saut {d_dist:.2f} m / {d_deg:.0f}°")
                continue
            echecs = 0
            pose = np.asarray(cand, dtype=float)
            scores.append(score)

        pose = np.asarray(pose, dtype=float).copy()
        pose[2] = carto.wrap_angle(pose[2])
        glob = carto.transform(pts, pose)
        carte.update([(glob, pose[:2])])
        archives.append((pts, inten, pose.copy()))

        if (k + 1) % 20 == 0 or k == len(fichiers) - 1:
            sc = "—" if score is None else f"{score:.2f}"
            print(f"  scan {k+1}/{len(fichiers)} : pose "
                  f"({pose[0]:+.2f}, {pose[1]:+.2f}, {math.degrees(pose[2]):+.1f}°) "
                  f"score {sc}")

    t_total = time.perf_counter() - t0
    print(f"Localisation terminée : {len(archives)} scans gardés, "
          f"{refuses} rejetés, en {t_total:.1f}s")

    # --- passe 2 : recalage global sur la carte finale --------------------- #
    # La carte de la passe 1 est complète : on affine chaque scan sur cette
    # carte, depuis SA pose de la passe 1 et avec une correction bornée.
    print("Recalage global sur la carte finale…")
    poses_finaux, corriges = [], 0
    for k, (pts, inten, pose1) in enumerate(archives):
        fn = qualite(pts, carte.grille)
        ctx = {"pts": pts, "grille": carte.grille}
        score1 = float(fn(pose1))
        # départ = pose de la passe 1 (jamais l'origine) : correction locale
        res = local.run(fn, pose1.copy(), ctx)
        pose2, score2 = res.pose, float(fn(res.pose))
        d_dist, d_deg = ecart(pose2, pose1)
        if (score2 > score1 and d_dist <= RECAL_MAX_D_M
                and d_deg <= RECAL_MAX_DEG):
            poses_finaux.append(np.asarray(pose2, dtype=float).copy())
            corriges += 1
        else:                       # sinon on garde la pose de la passe 1
            poses_finaux.append(pose1.copy())
        if (k + 1) % 20 == 0 or k == len(archives) - 1:
            print(f"  recal {k+1}/{len(archives)} : score {score1:.2f} -> "
                  f"{max(score1, score2):.2f}")
    print(f"{corriges}/{len(archives)} poses corrigées")

    # remplacer les poses archives par les poses recalées
    archives = deque((pts, inten, p) for (pts, inten, _old), p
                     in zip(archives, poses_finaux))

    # reconstruire la carte avec les poses corrigées (grille fraîche)
    carte = CarteLive()
    for pts, _inten, pose in archives:
        glob = carto.transform(pts, pose)
        carte.update([(glob, pose[:2])])

    nom = "Live_reconstruit_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier = carto.CARTES / nom
    dossier.mkdir(parents=True, exist_ok=True)

    pts_all = np.vstack([carto.transform(p, pose) for p, _, pose in archives])
    inten_all = np.concatenate([i for _, i, _ in archives])
    table = np.column_stack((np.arange(len(pts_all)),
                             np.round(pts_all * 1000),
                             np.clip(np.round(inten_all), 0, 255))).astype(np.int64)
    with open(dossier / "carte.csv", "w", encoding="utf-8", newline="") as f:
        f.write("N,x(mm),y(mm),Intensity\n")
        np.savetxt(f, table, fmt="%d", delimiter=",")
    print(f"carte.csv ({len(pts_all)} points) : {dossier / 'carte.csv'}")

    if carte.grille is not None:
        occ.save_csv(carte.grille, dossier / "occupation.csv")
        print(f"occupation.csv : {dossier / 'occupation.csv'}")

    rays = [(carto.transform(p, pose), pose[:2]) for p, _, pose in archives]
    vect = vec.build(rays, grille=carte.grille)
    if vect is not None:
        vec.save_csv(vect, dossier / "vectoriel.csv")
        print(f"vectoriel.csv ({len(vect)} segments) : {dossier / 'vectoriel.csv'}")

    print(f"\nNouveau live reconstruit : {dossier}")


if __name__ == "__main__":
    main()

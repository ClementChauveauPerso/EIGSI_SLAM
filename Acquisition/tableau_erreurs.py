# tableau_erreurs.py — Évaluation chiffrée des méthodes de localisation (Jour 2).
#
# Protocole (cf. consigne) : mêmes scans et mêmes poses initiales pour toutes
# les méthodes. Pour chaque écart initial (0,15 m/5°, 0,30 m/10°, 0,50 m/15°)
# et chaque tirage (direction aléatoire, générateurseedé par le tirage : tous
# les algorithmes reçoivent EXACTEMENT la même pose de départ), on mesure :
#   - erreur de position (m) et d'orientation (°) contre la vérité terrain,
#   - temps de calcul (s), taux de réussite : err < 0,10 m ET dθ < 5,0°.
#
# Deux scénarios réels :
#   1. carte « salle » : chaque scan de l'assemblage (poses connues dans
#      assemblage.json = vérité) est localisé sur la carte complète ;
#   2. scan contre scan : la paire j2 (vérité officialisée dans
#      algo_base.SCENARIO_J2), aux mêmes niveaux d'écart initial.
#
# Lancer : python tableau_erreurs.py   (quelques minutes : l'exhaustif domine)

import math
import sys
import time
from pathlib import Path

import numpy as np

import algo_base as base
import cartographie as carto
import occupation as occ
import scan_matching as sm
import vectorielle as vec

SEUIL_ERR = 0.10     # m : réussite si err < SEUIL_ERR
SEUIL_DTH = 5.0      # °  et dθ < SEUIL_DTH
NIVEAUX = ((0.15, 5.0), (0.30, 10.0), (0.50, 15.0))   # écart initial (m, °)
N_TIRAGES = 3


def perturbation(rng, dist_m, dist_deg):
    """Écart initial : direction aléatoire uniforme, signe d'θ aléatoire."""
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return np.array((dist_m * math.cos(phi), dist_m * math.sin(phi),
                     math.radians(dist_deg if rng.uniform() < 0.5 else -dist_deg)))


def mesure(algo, fn, ctx, verite, pose0):
    """(err m, dθ °, temps s, réussite) d'un run contre la vérité."""
    res = algo.run(fn, pose0, ctx)
    err = float(np.hypot(*(res.pose[:2] - verite[:2])))
    dth = abs(math.degrees(carto.wrap_angle(res.pose[2] - verite[2])))
    return err, dth, res.temps, err < SEUIL_ERR and dth < SEUIL_DTH


def evaluation(titre, runs):
    """runs : liste de (algo, ref, niveau, tirage, fn, ctx, verite, seed0).
    Une même graine par (niveau, tirage) : poses initiales identiques pour
    tous les algorithmes. Affiche le tableau markdown et le renvoie."""
    stats = {}                       # (algo, ref, niveau) -> [(err, dth, t, ok)]
    total = len(runs)
    faits = 0
    t0 = time.perf_counter()
    for algo, ref, niveau, tirage, fn, ctx, verite, seed in runs:
        rng = np.random.default_rng(seed)
        dep = verite + perturbation(rng, *niveau)
        stats.setdefault((algo.nom, ref, niveau), []).append(mesure(algo, fn, ctx, verite, dep))
        faits += 1
        if faits % 10 == 0 or faits == total:
            print(f"  [{faits}/{total}] {time.perf_counter() - t0:5.1f} s", flush=True)

    print(f"\n## {titre}\n")
    print(f"Réussite : err < {SEUIL_ERR:.2f} m et dθ < {SEUIL_DTH:.1f}° ; "
          f"{N_TIRAGES} tirages par niveau, pose initiale identique entre méthodes.\n")
    md = ["| algo | réf | " + " | ".join(f"±{d:.2f} m / ±{a:.0f}°" for d, a in NIVEAUX) + " |",
          "|---|---|" + "---|" * len(NIVEAUX)]
    for algo in sm.ALGORITHMES:
        for ref in sorted({r for (n, r, _n) in stats if n == algo.nom}):
            cells = []
            for niveau in NIVEAUX:
                essais = stats.get((algo.nom, ref, niveau))
                if not essais:
                    continue
                ok = sum(e[3] for e in essais)
                err = sorted(e[0] for e in essais)[len(essais) // 2]
                dth = sorted(e[1] for e in essais)[len(essais) // 2]
                t = sorted(e[2] for e in essais)[len(essais) // 2]
                cells.append(f"{ok}/{len(essais)} — {err * 100:.1f} cm, "
                             f"{dth:.1f}°, {t * 1000:.0f} ms")
            md.append(f"| {algo.nom} | {ref} | " + " | ".join(cells) + " |")
    print("\n".join(md) + "\n", flush=True)
    return md


def scenario_carte():
    """Section 1 : chaque scan de l'assemblage localisé sur la carte salle."""
    dossier = Path(__file__).resolve().parents[1] / "Data" / "Cartes" / "salle"
    echantillons, grille, vect = base.charger_carte(dossier)
    runs = []
    for i, (_f, pts, verite) in enumerate(echantillons):
        ctx = {"pts": pts, "grille": grille, "segments": vect.segments, "verite": verite}
        for algo in sm.ALGORITHMES:
            for ref in [r for r in ("grille", "vectorielle") if algo.disponible(r)]:
                fn = (base.score_grille(grille, pts) if ref == "grille"
                      else base.score_segments(vect.segments, pts))
                for tirage in range(N_TIRAGES):
                    for j, niveau in enumerate(NIVEAUX):
                        runs.append((algo, ref, niveau, tirage, fn, ctx,
                                     verite, 1000 * j + 100 * i + tirage))
    return ("Localisation sur la carte « salle » (8 scans de l'assemblage, "
            "poses connues = vérité)"), runs


def scenario_j2():
    """Section 2 : scan contre scan, paire j2 (vérité 0/0/+53,5°)."""
    scen = base.SCENARIO_J2
    pts_ref, _ = carto.load_scan(scen["reference"])
    pts_cible, _ = carto.load_scan(scen["cible"])
    verite = scen["verite"]
    grille = occ.build([(pts_ref, (0.0, 0.0))], 0.10)
    segments = vec.build([(pts_ref, (0.0, 0.0))]).segments
    ctx = {"pts": pts_cible, "grille": grille, "segments": segments,
           "nuage": pts_ref, "verite": verite}
    fns = {"grille": base.score_grille(grille, pts_cible),
           "vectorielle": base.score_segments(segments, pts_cible),
           "nuage": base.score_nuage(pts_ref, pts_cible)}
    runs = []
    for algo in sm.ALGORITHMES:
        for ref, fn in fns.items():
            if not (algo.disponible(ref) or (ref == "nuage" and algo.nom != "direct")):
                continue
            for tirage in range(N_TIRAGES):
                for j, niveau in enumerate(NIVEAUX):
                    runs.append((algo, ref, niveau, tirage, fn, ctx,
                                 verite, 5000 + 100 * j + tirage))
    return ("Localisation scan contre scan (paire j2 : test2 dans le repère "
            "de test1, vérité 0/0/+53,5°)"), runs


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    entetes = []
    for titre, runs in (scenario_carte(), scenario_j2()):
        t0 = time.perf_counter()
        entetes.append(evaluation(titre, runs))
        print(f"({time.perf_counter() - t0:.0f} s)", flush=True)

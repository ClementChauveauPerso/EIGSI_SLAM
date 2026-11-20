# scan_matching.py — Registre des algorithmes de localisation (scan matching).
#
# Chaque méthode vit dans son propre fichier (algo_*.py), hérite de
# algo_base.Algorithme et expose run(score_fn, pose0, contexte) -> Resultat.
# L'interface (localisation.py) et les tests itèrent sur ALGORITHMES.

import math
import sys

import numpy as np

try:  # modules voisins (même dossier)
    import cartographie as carto
except ImportError:
    from . import cartographie as carto

from algo_base import score_grille, score_segments
from algo_exhaustif import Exhaustif
from algo_descente import Descente
from algo_multires import Multires
from algo_particules import Particules
from algo_direct import Direct

ALGORITHMES = [Exhaustif(), Descente(), Multires(), Particules(), Direct()]


if __name__ == "__main__":
    # auto-vérification : petite salle synthétique (rectangle 4 × 3 m),
    # vérité perturbée -> chaque algorithme doit retrouver la pose.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    import occupation as occ
    import vectorielle as vec

    t = np.linspace(0.0, 1.0, 120)
    murs = np.vstack([
        np.column_stack((-2 + 4 * t, np.full_like(t, -1.5))),
        np.column_stack((-2 + 4 * t, np.full_like(t, 1.5))),
        np.column_stack((np.full_like(t, -2), -1.5 + 3 * t)),
        np.column_stack((np.full_like(t, 2), -1.5 + 3 * t)),
    ])
    verite = np.array((0.12, -0.08, math.radians(4.0)))
    rays = [(carto.transform(murs, verite), verite[:2])]
    grille = occ.build(rays, 0.10)
    vect = vec.build(rays)
    ctx = {"pts": murs, "grille": grille, "segments": vect.segments, "verite": verite}

    print(f"{'algo':12s} {'réf':12s} {'err (cm)':>9s} {'dθ (°)':>7s} "
          f"{'évals':>7s} {'temps':>9s}")
    for algo in ALGORITHMES:
        refs = [r for r in ("grille", "vectorielle") if algo.disponible(r)]
        for ref in refs:
            fn = (score_grille(grille, murs) if ref == "grille"
                  else score_segments(vect.segments, murs))
            pose0 = verite + np.array((0.25, -0.20, math.radians(-7.0)))
            res = algo.run(fn, pose0, ctx)
            err = float(np.hypot(*(res.pose[:2] - verite[:2]))) * 100
            dth = abs(math.degrees(carto.wrap_angle(res.pose[2] - verite[2])))
            print(f"{algo.nom:12s} {ref:12s} {err:9.1f} {dth:7.1f} "
                  f"{res.n_evals:7d} {res.temps * 1000:8.0f} ms")

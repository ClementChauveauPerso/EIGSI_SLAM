# test_scan_matching.py — Contrat des algorithmes de localisation.
#
# 1) Justesse : chaque algorithme doit retrouver la vérité (posée perturbée)
#    sur une salle synthétique, puis sur la carte réelle de Data\Cartes\salle.
# 2) Performance : chaque run sur la carte réelle doit durer < 1 s
#    (quasi temps réel).
#
# Chaque algorithme ne modifie QUE son fichier algo_*.py ; ce fichier est le
# contrat, il ne doit pas bouger. Lancer : python test_scan_matching.py

import math
import time

import numpy as np

import algo_base as base
import cartographie as carto
import scan_matching as sm

LIMITE_S = 1.0          # temps max par run sur la carte réelle (quasi temps réel)


# --------------------------------------------------------------------------- #
# Banque d'essai : salle synthétique 4 × 3 m scannée depuis la pose VERITE
# --------------------------------------------------------------------------- #
VERITE = np.array((0.12, -0.08, math.radians(4.0)))


def salle_synthetique():
    t = np.linspace(0.0, 1.0, 120)
    murs = np.vstack([
        np.column_stack((-2 + 4 * t, np.full_like(t, -1.5))),
        np.column_stack((-2 + 4 * t, np.full_like(t, 1.5))),
        np.column_stack((np.full_like(t, -2), -1.5 + 3 * t)),
        np.column_stack((np.full_like(t, 2), -1.5 + 3 * t)),
    ])
    rays = [(carto.transform(murs, VERITE), VERITE[:2])]
    import occupation as occ
    import vectorielle as vec
    return murs, occ.build(rays, 0.10), vec.build(rays)


MURS, GRILLE, VECT = salle_synthetique()
CTX = {"pts": MURS, "grille": GRILLE, "segments": VECT.segments, "verite": VERITE}


def score_pour(ref):
    return (base.score_grille(GRILLE, MURS) if ref == "grille"
            else base.score_segments(VECT.segments, MURS))


def erreur(res):
    """(erreur translation m, erreur angle °) entre la pose trouvée et la vérité."""
    return (float(np.hypot(*(res.pose[:2] - VERITE[:2]))),
            abs(math.degrees(carto.wrap_angle(res.pose[2] - VERITE[2]))))


def algo_par_nom(nom):
    return next(a for a in sm.ALGORITHMES if a.nom == nom)


# --------------------------------------------------------------------------- #
# Socle partagé (scores) — ne dépend pas des algorithmes
# --------------------------------------------------------------------------- #
def test_scores():
    # occupation : max à la vérité, interpolation bilinéaire exacte au centre d'une case
    fn = base.score_grille(GRILLE, MURS)
    assert fn(VERITE) > fn(VERITE + np.array((0.3, 0.3, math.radians(6))))
    img = np.ones((10, 10), dtype=np.float32)
    val = base._bilineaire(img, 0.0, 0.0, 0.1, np.array(((0.55, 0.75),)))
    assert abs(float(val[0]) - 1.0) < 1e-6
    assert float(base._bilineaire(img, 0.0, 0.0, 0.1, np.array(((9.0, 5.0),)))[0]) == 0.0

    # segments : un point posé sur un segment est à distance nulle, le score max est à la vérité
    fn = base.score_segments(VECT.segments, MURS)
    d2, _i, _p = base._dist_seg(np.array(((1.0, -1.5),)), np.array(((0.0, -1.5, 2.0, -1.5),)))
    assert d2[0] < 1e-12
    assert fn(VERITE) > fn(VERITE + np.array((0.2, -0.2, math.radians(5))))


# --------------------------------------------------------------------------- #
# Justesse de chaque algorithme (salle synthétique)
# --------------------------------------------------------------------------- #
def test_exhaustif():
    algo = algo_par_nom("exhaustif")
    for ref in algo.references:
        res = algo.run(score_pour(ref),
                       VERITE + np.array((0.3, -0.25, math.radians(-8))), CTX)
        err, dth = erreur(res)
        assert err < 0.05 and dth < 2.5, (ref, err, dth)   # ≤ un demi-pas de grille
        assert res.paysage is not None                     # pour l'affichage


def test_descente():
    algo = algo_par_nom("gradient")
    for ref in algo.references:
        res = algo.run(score_pour(ref),
                       VERITE + np.array((0.06, -0.05, math.radians(2))), CTX)
        err, dth = erreur(res)
        assert err < 0.025 and dth < 1.5, (ref, err, dth)


def test_multires():
    algo = algo_par_nom("multi-rés.")
    for ref in algo.references:
        res = algo.run(score_pour(ref),
                       VERITE + np.array((0.4, 0.3, math.radians(10))), CTX)
        err, dth = erreur(res)
        assert err < 0.05 and dth < 2.5, (ref, err, dth)
        assert res.paysage is not None


def test_particules():
    algo = algo_par_nom("particules")
    for ref in algo.references:
        res = algo.run(score_pour(ref), VERITE, CTX)
        err, dth = erreur(res)
        assert err < 0.07 and dth < 3.5, (ref, err, dth)
        assert res.particules is not None


def test_direct():
    algo = algo_par_nom("direct")
    assert not algo.disponible("grille")                # vectorielle seulement
    res = algo.run(score_pour("vectorielle"),
                   VERITE + np.array((0.10, 0.08, math.radians(3))), CTX)
    err, dth = erreur(res)
    assert err < 0.025 and dth < 1.5, (err, dth)


# --------------------------------------------------------------------------- #
# Carte réelle : Data\Cartes\salle (assemblage 8 scans) + performance
# --------------------------------------------------------------------------- #
SALLE = None


def charger_salle():
    global SALLE
    if SALLE is None:
        from pathlib import Path
        dossier = Path(__file__).resolve().parents[1] / "Data" / "Cartes" / "salle"
        SALLE = base.charger_carte(dossier)
    return SALLE


def test_salle():
    """Chaque algorithme localise un vrai scan de la salle : justesse + < 1 s."""
    for algo in sm.ALGORITHMES:
        test_salle_algo(algo.nom)


def test_salle_algo(nom):
    """test_salle limité à un algorithme (validation isolée : un agent par algo)."""
    algo = algo_par_nom(nom)
    echantillons, grille, vect = charger_salle()
    pts, verite = echantillons[0][1], echantillons[0][2]
    ctx = {"pts": pts, "grille": grille, "segments": vect.segments, "verite": verite}
    bruit = np.array((0.25, -0.20, math.radians(-7.0)))
    print(f"\n{'algo':12s} {'réf':12s} {'err (cm)':>9s} {'dθ (°)':>7s} "
          f"{'évals':>7s} {'temps':>10s}  (limite {LIMITE_S:.1f} s)")
    for ref in [r for r in ("grille", "vectorielle") if algo.disponible(r)]:
        fn = (base.score_grille(grille, pts) if ref == "grille"
              else base.score_segments(vect.segments, pts))
        res = algo.run(fn, verite + bruit, ctx)
        err = float(np.hypot(*(res.pose[:2] - verite[:2])))
        dth = abs(math.degrees(carto.wrap_angle(res.pose[2] - verite[2])))
        print(f"{algo.nom:12s} {ref:12s} {err * 100:9.1f} {dth:7.1f} "
              f"{res.n_evals:7d} {res.temps * 1000:9.0f} ms")
        assert err < 0.10 and dth < 5.0, (algo.nom, ref, err, dth)   # juste
        assert res.temps < LIMITE_S, \
            f"{algo.nom}/{ref} trop lent : {res.temps:.2f} s (cible < {LIMITE_S} s)"


def test_scenario_j2():
    """Scénario réel scan-contre-scan : deux scans j2 pris à quelques
    centimètres l'un de l'autre, orientations différentes (~53° dans le repère
    des CSV). Vérité officialisée dans algo_base.SCENARIO_J2 (recherche globale
    concordante sur les 3 références, corrélation croisée des profils de portée,
    confirmation utilisateur). Critères : erreur ≤ tolérance du scénario,
    < 1 s, et cohérence inter-références. Les méthodes globales doivent aussi
    réussir le départ dur (0, 0, 0) ; le gradient est local (départ bruité seul)."""
    import occupation as occ
    import vectorielle as vec
    scen = base.SCENARIO_J2
    pts_ref, _ = carto.load_scan(scen["reference"])
    pts_cible, _ = carto.load_scan(scen["cible"])
    verite = scen["verite"]
    err_max, dth_max = scen["tolerance"]
    dth_max = math.degrees(dth_max)
    grille = occ.build([(pts_ref, (0.0, 0.0))], 0.10)
    segments = vec.build([(pts_ref, (0.0, 0.0))]).segments
    ctx = {"pts": pts_cible, "grille": grille, "segments": segments,
           "nuage": pts_ref, "verite": verite}
    fns = {"grille": base.score_grille(grille, pts_cible),
           "vectorielle": base.score_segments(segments, pts_cible),
           "nuage": base.score_nuage(pts_ref, pts_cible)}
    rng = np.random.default_rng(3)
    bruite = verite + rng.uniform(-1.0, 1.0, 3) * np.array((0.25, 0.25, math.radians(7.0)))
    globaux = {"exhaustif", "multi-rés.", "particules", "direct"}
    print(f"\nscénario j2 (scan-contre-scan, vérité 0/0/+53,5°) :")
    for algo in sm.ALGORITHMES:
        poses_bruitees = []
        for ref_nom, fn in fns.items():
            if not (algo.disponible(ref_nom)
                    or (ref_nom == "nuage" and algo.nom != "direct")):
                continue
            for nom_dep, dep in (("bruité", bruite), ("dur", np.zeros(3))):
                if nom_dep == "dur" and algo.nom not in globaux:
                    continue                    # méthodes locales : hors spec
                res = algo.run(fn, np.asarray(dep, float), ctx)
                err = float(np.hypot(*(res.pose[:2] - verite[:2])))
                dth = abs(math.degrees(carto.wrap_angle(res.pose[2] - verite[2])))
                print(f"  {algo.nom:12s} {ref_nom:12s} départ {nom_dep:7s} "
                      f"err={err * 100:6.1f} cm dθ={dth:5.1f}° "
                      f"{res.temps * 1000:6.0f} ms")
                assert err <= err_max and dth <= dth_max, \
                    (algo.nom, ref_nom, nom_dep, err, dth)
                assert res.temps < LIMITE_S, (algo.nom, ref_nom, res.temps)
                if nom_dep == "bruité":
                    poses_bruitees.append(res.pose)
        if len(poses_bruitees) >= 2:            # cohérence inter-références
            poses = np.array(poses_bruitees)
            eco = max(float(np.hypot(*(p[:2] - poses[0][:2]))) for p in poses[1:])
            eth = max(abs(math.degrees(carto.wrap_angle(p[2] - poses[0][2])))
                      for p in poses[1:])
            assert eco <= 0.07 and eth <= 2.0, (algo.nom, eco, eth)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        import sys
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    tests = (test_scores, test_exhaustif, test_descente, test_multires,
             test_particules, test_direct, test_scenario_j2, test_salle)
    echoues = 0
    for test in tests:
        try:
            test()
            print(f"  ok  {test.__name__}")
        except AssertionError as exc:
            echoues += 1
            print(f"  ÉCHEC  {test.__name__} : {exc}")
    total = len(tests)
    if echoues:
        print(f"{total - echoues}/{total} tests passés — voir les échecs ci-dessus")
        raise SystemExit(1)
    print(f"{total}/{total} tests passés")

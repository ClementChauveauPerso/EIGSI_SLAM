# algo_exhaustif.py — Localisation par recherche exhaustive (brute force).
#
# Évalue toutes les poses d'une grille régulière (x, y, θ) et garde la
# meilleure. Simple, robuste (pas de minimum local : il voit tout le bassin).
#
# Version vectorisée numpy : le scan est pré-tourné une fois par θ, puis
# toutes les translations sont évaluées d'un coup (corrélation d'image pour la
# grille, distances aux segments par lots pour le vectoriel).
#
# Recherche en quatre temps :
#   0. balayage global : pan-carte en translation (corrélation de l'image de
#      likelihood avec les points du scan — coût nθ × npts ajouts de tableaux,
#      résolution = res de la grille) × plein cercle en θ. Aucun prior sur le
#      départ : c'est l'exhaustif qui voit tout le bassin. Sur une carte
#      fragmentée, plusieurs bassins se valent au seul critère de distance —
#      chaque bassin est donc arbitré par le score FIABLE (score_fn −
#      incohérence des rayons, voir algo_base.incoherence) ;
#   1. grossier : balayage complet autour du meilleur bassin — grille
#      d'occupation si disponible, sinon d²/(d²+σ²) sur les segments ;
#   2. fin : balayage local (±6 cm, ±2°, x/y puis θ) au pas fin — noyau
#      grille si le score_fn est la métrique grille (reconnue par sonde),
#      sinon segments plafonnés ;
#   3. polissage multi-σ (grille seulement) : le noyau large (σ = 0,20 m)
#      tolère ~10 cm de flottement ; re-raffiner à σ décroissant (0,10 puis
#      0,06 m) colle la géométrie fine sans changer de bassin.
# Le score final reste mesuré avec le score_fn fourni (comparabilité
# inter-méthodes) ; le paysage est évalué avec le noyau fin interne (identique
# au score_fn quand la métrique grille est reconnue, sinon paysage local net).

import math
import time

import numpy as np

try:  # modules voisins (même dossier)
    import occupation as occ
except ImportError:  # importé comme paquet
    from . import occupation as occ

from algo_base import (Algorithme, Resultat, _bilineaire, _echantillon,
                       CAP_DIST, balayage_correlation, image_score_grille,
                       incoherence, _maxima_locaux)

SIG2 = 1.0             # σ² (m²) du noyau large d²/(d²+σ²) : portée ~1 m, sans plateau
N_BASSINS = 25         # maxima locaux du balayage global arbitres par le score fiable


def _d2_seg(pts, segs):
    """d² de chaque point (Q, 2) à chaque segment (M, 4) -> (Q, M).

    Formule algébrique |ap - t·ab|² = |ap|² - l2(2tu - t²) : aucun tableau
    (Q, M, 2) intermédiaire au-delà de ap, et einsum (produits scalaires sans
    tableau temporaire).
    """
    a = segs[:, :2]
    ab = segs[:, 2:] - a
    l2 = np.maximum((ab * ab).sum(1), segs.dtype.type(1e-12))
    ap = pts[:, None, :] - a                                 # (Q, M, 2)
    uu = np.einsum("qmk,mk->qm", ap, ab) / l2                # (Q, M)
    t = np.clip(uu, 0.0, 1.0)
    return np.einsum("qmk,qmk->qm", ap, ap) - l2 * (2.0 * t * uu - t * t)


def _bilineaire32(img, x0, y0, res, pts, hors=np.float32(0.0)):
    """Image interpolée (float32 partout) — noyau de recherche : même
    calcul que base._bilineaire mais sans l'upcast float64 des constantes,
    deux fois moins de bande passante. Hors carte -> `hors`, comme le socle."""
    c0 = np.float32(0.5)
    gx = (pts[:, 0] - np.float32(x0)) / np.float32(res) - c0
    gy = (pts[:, 1] - np.float32(y0)) / np.float32(res) - c0
    ny, nx = img.shape
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - np.floor(gx), gy - np.floor(gy)
    ok = (ix >= 0) & (ix < nx - 1) & (iy >= 0) & (iy < ny - 1)
    ic, jc = np.clip(ix, 0, nx - 2), np.clip(iy, 0, ny - 2)
    v = (img[jc, ic] * (1 - fx) * (1 - fy) + img[jc, ic + 1] * fx * (1 - fy)
         + img[jc + 1, ic] * (1 - fx) * fy + img[jc + 1, ic + 1] * fx * fy)
    return np.where(ok, v, hors)


class Exhaustif(Algorithme):
    nom = "exhaustif"
    references = ("grille", "vectorielle")
    rayons = (0.5, 0.5, math.radians(20.0))   # fenêtre de recherche autour de pose0
    pas = (0.05, 0.05, math.radians(2.0))     # pas d'échantillonnage des poses
    fin_rayons = (0.06, 0.06, math.radians(2.0))   # fenêtre du balayage fin
    fin_pas = (0.02, 0.02, math.radians(0.5))      # pas du balayage fin
    sigmas_fin = (0.10, 0.06)                 # polissage multi-σ (grille)

    def run(self, score_fn, pose0, contexte):
        debut = time.perf_counter()
        base = np.asarray(pose0, dtype=float)
        axx = axy = axth = None
        pts = _echantillon(contexte["pts"])
        segs = np.asarray(contexte["segments"], dtype=float)[:, :4]
        if not segs.shape[0]:                  # carte sans murs : segment dégénéré
            segs = np.zeros((1, 4))            # (distance à l'origine, jamais planté)
        grille = contexte.get("grille")

        # -- noyaux de score, par point (Q,) ; moyenne par pose faite dans _lot --
        # (pré-traitement grille identique à score_grille : champ de likelihood
        # -d² plafonné, bordure 0 hors carte — via base.image_score_grille)
        img = gx0 = gy0 = res = None
        if grille is not None:
            img, gx0, gy0, res = image_score_grille(grille)
        k_grille = (lambda w: _bilineaire(img, gx0, gy0, res, w, hors=0.0)
                    ) if grille is not None else None   # sonde : exact (float64)
        k_grille32 = (lambda w: _bilineaire32(img, gx0, gy0, res, w, hors=0.0)
                      ) if grille is not None else None  # recherche (float32)

        # noyaux segments en float32 (recherche seulement : le score final
        # reste mesuré avec score_fn) — divise la bande passante par 2
        segs32 = segs.astype(np.float32)

        def k_seg_cap(w):      # noyau fin maison : -d² plafonné (à maximiser)
            return -np.minimum(_d2_seg(w, segs32).min(1),
                               np.float32(CAP_DIST * CAP_DIST))

        def k_seg_large(w):    # noyau large : décroît encore au-delà de CAP_DIST
            d2 = _d2_seg(w, segs32).min(1)
            return -d2 / (d2 + SIG2)

        def _lot(k, rot, trans, stride=1, chunk=128):
            """Score de toutes les poses (rot + translation), par lots -> (P,).

            stride > 1 : sous-échantillonne les points (noyaux segments —
            la moyenne sur un point sur deux suffit pour chercher) et passe
            en float32.
            """
            out = np.empty(len(trans))
            for i in range(0, len(trans), chunk):
                bloc = trans[i:i + chunk]      # lots : borne la mémoire de (q, N, M)
                w = (rot[None, :, :] + bloc[:, None, :])[:, ::stride, :].reshape(-1, 2)
                if stride > 1:
                    w = w.astype(np.float32)
                out[i:i + chunk] = k(w).reshape(len(bloc), -1).mean(1)
            return out

        def _balayage(k, axs, stride=1):
            """Scores (nθ, nx, ny) sur la fenêtre axs, + les translations (P, 2)."""
            X, Y = np.meshgrid(axs[0], axs[1], indexing="ij")
            trans = np.column_stack((X.ravel(), Y.ravel()))
            out = np.empty((len(axs[2]), len(axs[0]), len(axs[1])))
            for ith, th in enumerate(axs[2]):
                c, s = math.cos(th), math.sin(th)
                rot = pts @ np.array(((c, s), (-s, c)))
                out[ith] = _lot(k, rot, trans, stride).reshape(len(axs[0]), len(axs[1]))
            return out, trans

        # 0) recherche globale. Grille : balayage pan-carte par corrélation
        #    (aucune hypothèse de départ) + arbitrage des bassins au score
        #    fiable. Segments seuls : anneau large autour de pose0 (pas
        #    d'image à corréler — ex. pose relative de deux scans).
        k_coarse = k_grille32 if k_grille is not None else k_seg_large
        evals_ring = 0
        centre = base
        if grille is not None:
            out, ths = balayage_correlation(img, gx0, gy0, res, pts, stride=3)
            evals_ring = int(out.size)
            # bassins candidats : maxima locaux 3D dédoublonnés (deux cellules
            # voisines du même bassin ne comptent qu'une fois)
            bassins = []
            for ith, iy, ix in _maxima_locaux(out, N_BASSINS * 3):
                p = np.array((gx0 + (ix + 0.5) * res, gy0 + (iy + 0.5) * res,
                              float(ths[ith])))
                if any(abs((p[2] - q[2] + math.pi) % (2 * math.pi) - math.pi)
                       < math.radians(12.0)
                       and np.hypot(*(p[:2] - q[:2])) < 0.5 for q in bassins):
                    continue
                bassins.append(p)
                if len(bassins) >= N_BASSINS:
                    break
            # la pose de départ est toujours candidate : le balayage par
            # corrélation peut la rater (auto-coïncidence d'un scan sur sa
            # propre carte : la crête y est plate, pas un maximum local) alors
            # que son score fiable peut être imbattable. L'arbitrage tranche.
            bassins.append(base.copy())
            # raffinage grossier de chaque bassin (xy, θ, xy — pleine
            # résolution : une pose mal posée surestime son incohérence) et
            # arbitrage au score fiable : la distance seule ne voit pas les
            # poses géométriquement impossibles (rayons traversant un mur,
            # échos posés dans le vide cartographié)
            arbitres = []
            for c0 in bassins:
                pb = c0.copy()
                for _passe in range(2):
                    bx = pb[0] + np.arange(-4, 5) * self.pas[0]
                    by = pb[1] + np.arange(-4, 5) * self.pas[0]
                    Xb, Yb = np.meshgrid(bx, by, indexing="ij")
                    tr_b = np.column_stack((Xb.ravel(), Yb.ravel()))
                    c, s = math.cos(pb[2]), math.sin(pb[2])
                    s_b = _lot(k_grille32, pts @ np.array(((c, s), (-s, c))), tr_b, 1)
                    j = int(s_b.argmax())
                    pb = np.array((tr_b[j][0], tr_b[j][1], pb[2]))
                    ths_f = pb[2] + np.arange(-3, 4) * math.radians(1.0)
                    s_th = []
                    for th in ths_f:
                        c, s = math.cos(th), math.sin(th)
                        s_th.append(float(k_grille32(
                            pts @ np.array(((c, s), (-s, c))) + pb[:2]).mean()))
                    it = int(np.argmax(s_th))
                    pb = np.array((pb[0], pb[1], ths_f[it]))
                f = float(score_fn(pb))
                inco = incoherence(grille, contexte["pts"], pb)
                arbitres.append((f - inco, f, inco, pb))
                evals_ring += 2 * (len(tr_b) + len(ths_f))
            # arbitrage au score fiable (f - incohérence) ; les quasi-égalités
            # (bruit de l'incohérence : 220 rayons -> ~0,03 ; zone aveugle de la
            # carte où rien ne contredit) sont départagées par le prior pose0 —
            # quand la carte ne tranche pas (salle symétrique -> ambiguïté à
            # 180°), c'est la seule information restante. Un écart décisif
            # (carte informative) ignore donc complètement le prior.
            EPS_COMBO = 0.08
            meilleur = max(a[0] for a in arbitres)

            def _d0(pb):
                dth = abs((pb[2] - base[2] + math.pi) % (2 * math.pi) - math.pi)
                return dth + float(np.hypot(*(pb[:2] - base[:2])))

            centre = min((a for a in arbitres if a[0] >= meilleur - EPS_COMBO),
                         key=lambda a: _d0(a[3]))[3]
        elif segs.shape[0]:
            # anneau : plein cercle en θ au pas épais, fenêtre xy large autour
            # de pose0 ; bassins candidats = maxima locaux du profil θ,
            # arbitres par le score_fn réel (à score égal, le plus proche du
            # prior l'emporte)
            pr = self.pas[0] * 3.0                       # 0,15 m
            nr = int(round(self.rayons[0] / pr))
            pth = math.radians(9.0)
            xr = base[0] + np.arange(-nr, nr + 1) * pr
            yr = base[1] + np.arange(-nr, nr + 1) * pr
            thr = np.arange(0.0, 2.0 * math.pi, pth)
            Xr, Yr = np.meshgrid(xr, yr, indexing="ij")
            trans_r = np.column_stack((Xr.ravel(), Yr.ravel()))
            ring = np.empty((len(thr), len(trans_r)))
            for ith, th in enumerate(thr):
                c, s = math.cos(th), math.sin(th)
                rot = pts @ np.array(((c, s), (-s, c)))
                ring[ith] = _lot(k_coarse, rot, trans_r, 4)
            evals_ring = int(ring.size)
            cand = []
            for i in np.argsort(ring.max(1))[::-1]:
                if all(min(abs(thr[i] - t), 2 * math.pi - abs(thr[i] - t))
                       >= 3 * pth for t in cand):
                    cand.append(thr[i])
                if len(cand) >= 4:
                    break
            bas = []
            for th in cand:
                m = int(ring[int(np.argmin(abs(thr - th)))].argmax())
                c0 = np.array((trans_r[m, 0], trans_r[m, 1], th))
                bx = c0[0] + np.arange(-nr * 2, nr * 2 + 1) * self.pas[0]
                by = c0[1] + np.arange(-nr * 2, nr * 2 + 1) * self.pas[0]
                Xb, Yb = np.meshgrid(bx, by, indexing="ij")
                tr_b = np.column_stack((Xb.ravel(), Yb.ravel()))
                c, s = math.cos(th), math.sin(th)
                s_b = _lot(k_coarse, pts @ np.array(((c, s), (-s, c))), tr_b, 2)
                j = int(s_b.argmax())
                pose_b = np.array((tr_b[j, 0], tr_b[j, 1], th))
                bas.append((float(score_fn(pose_b)), pose_b))
                evals_ring += len(tr_b)
            s_max = max(s for s, _ in bas)
            gardes = [b for b in bas if b[0] >= s_max - 1e-6 * max(1.0, abs(s_max))]
            def _dist0(pb):
                dth = abs((pb[2] - base[2] + math.pi) % (2 * math.pi) - math.pi)
                return dth + float(np.hypot(*(pb[:2] - base[:2])))
            centre = min(gardes, key=lambda b: _dist0(b[1]))[1]

        # 1) balayage grossier autour du meilleur bassin — le paysage sert à
        #    l'affichage ; en mode grille la pose du bassin arbitré fait foi
        #    (le likelihood seul peut glisser vers un fragment voisin)
        axes = [centre[k] + np.arange(-int(round(self.rayons[k] / self.pas[k])),
                                      int(round(self.rayons[k] / self.pas[k])) + 1)
                * self.pas[k] for k in range(3)]
        axx, axy, axth = axes
        scores, trans = _balayage(k_coarse, axes, stride=2)
        if grille is not None:
            pose_c = centre
        else:
            ith, ix, iy = np.unravel_index(int(scores.argmax()), scores.shape)
            pose_c = np.array((axx[ix], axy[iy], axth[ith]))

        # sonde (2 poses) : le score_fn est-il la métrique grille ? (sinon noyau maison)
        s_fn0 = float(score_fn(pose_c))
        c, s = math.cos(pose_c[2]), math.sin(pose_c[2])
        rot0 = pts @ np.array(((c, s), (-s, c)))
        pose_d = pose_c + np.array((0.17, -0.13, math.radians(5.0)))
        c, s = math.cos(pose_d[2]), math.sin(pose_d[2])
        w1 = pts @ np.array(((c, s), (-s, c))) + pose_d[:2]
        grille_ok = k_grille is not None and all(
            abs(sf - float(k_grille(w).mean())) <= 1e-6 * max(1.0, abs(sf))
            for sf, w in ((s_fn0, rot0 + pose_c[:2]),
                          (float(score_fn(pose_d)), w1)))
        k_fine = k_grille32 if grille_ok else k_seg_cap

        # 2) balayage fin local avec la métrique reconnue (ou segments plafonnés) :
        #    (x, y) puis θ — dans un voisinage de ±6 cm le couplage translation/
        #    rotation est négligeable, et 7×7 + 9 poses coûtent ~8× moins que 7×7×9.
        fr, fp = self.fin_rayons, self.fin_pas

        def polis(pose_c, k, stride):
            fx = pose_c[0] + np.arange(-int(round(fr[0] / fp[0])),
                                       int(round(fr[0] / fp[0])) + 1) * fp[0]
            fy = pose_c[1] + np.arange(-int(round(fr[1] / fp[1])),
                                       int(round(fr[1] / fp[1])) + 1) * fp[1]
            fth = pose_c[2] + np.arange(-int(round(fr[2] / fp[2])),
                                        int(round(fr[2] / fp[2])) + 1) * fp[2]
            Xf, Yf = np.meshgrid(fx, fy, indexing="ij")
            tr_f = np.column_stack((Xf.ravel(), Yf.ravel()))
            c, s = math.cos(pose_c[2]), math.sin(pose_c[2])
            rot_c = pts @ np.array(((c, s), (-s, c)))
            s_xy = _lot(k, rot_c, tr_f, stride).reshape(len(fx), len(fy))
            ix, iy = np.unravel_index(int(s_xy.argmax()), s_xy.shape)
            xy = np.array((fx[ix], fy[iy]))
            s_th = np.array([_lot(k, pts @ np.array(((math.cos(th), math.sin(th)),
                                                     (-math.sin(th), math.cos(th)))),
                                   xy[None, :], stride)[0] for th in fth])
            it = int(s_th.argmax())
            return np.array((xy[0], xy[1], fth[it])), float(s_th[it])

        stride = 1 if k_fine is k_grille32 else 3
        evals_fin = [0]

        def polis(pose_c, k, stride):
            fx = pose_c[0] + np.arange(-int(round(fr[0] / fp[0])),
                                       int(round(fr[0] / fp[0])) + 1) * fp[0]
            fy = pose_c[1] + np.arange(-int(round(fr[1] / fp[1])),
                                       int(round(fr[1] / fp[1])) + 1) * fp[1]
            fth = pose_c[2] + np.arange(-int(round(fr[2] / fp[2])),
                                        int(round(fr[2] / fp[2])) + 1) * fp[2]
            evals_fin[0] += len(fx) * len(fy) + len(fth)
            Xf, Yf = np.meshgrid(fx, fy, indexing="ij")
            tr_f = np.column_stack((Xf.ravel(), Yf.ravel()))
            c, s = math.cos(pose_c[2]), math.sin(pose_c[2])
            rot_c = pts @ np.array(((c, s), (-s, c)))
            s_xy = _lot(k, rot_c, tr_f, stride).reshape(len(fx), len(fy))
            ix, iy = np.unravel_index(int(s_xy.argmax()), s_xy.shape)
            xy = np.array((fx[ix], fy[iy]))
            s_th = np.array([_lot(k, pts @ np.array(((math.cos(th), math.sin(th)),
                                                     (-math.sin(th), math.cos(th)))),
                                   xy[None, :], stride)[0] for th in fth])
            it = int(s_th.argmax())
            return np.array((xy[0], xy[1], fth[it])), float(s_th[it])

        stride = 1 if k_fine is k_grille32 else 3
        pose, s_fin = polis(pose_c, k_fine, stride)

        # 3) polissage multi-σ : le noyau large tolère ~10 cm de flottement, le
        #    noyau fin colle la géométrie (le bassin, lui, ne bouge plus)
        if grille is not None:
            for sigma in self.sigmas_fin:
                img_f, ox, oy, ores = image_score_grille(grille, sigma)
                pose, s_fin = polis(
                    pose,
                    (lambda w, im=img_f, a=ox, b=oy, r=ores:
                     _bilineaire32(im, a, b, r, w, hors=0.0)), 1)

        # paysage 2D à θ* : évaluation interne vectorisée (identique au score_fn
        # pour la référence grille ; paysage local net sinon, pour l'affichage)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rot = pts @ np.array(((c, s), (-s, c)))
        pay = _lot(k_fine, rot, trans, stride).reshape(len(axx), len(axy))

        # historique « meilleur jusqu'ici » dans l'ordre du balayage (θ, y, x)
        plat = scores.transpose(0, 2, 1).ravel()
        cum = np.maximum.accumulate(plat)
        idx = np.flatnonzero(np.concatenate(([True], plat[1:] > cum[:-1])))
        if len(idx) > 100:     # allégé pour l'interface (la courbe garde sa forme)
            idx = np.unique(np.concatenate((idx[::max(1, len(idx) // 100)], idx[-1:])))
        histo = []
        for m in idx:
            ith, r = divmod(int(m), len(axy) * len(axx))
            iy_, ix_ = divmod(r, len(axx))
            histo.append((np.array((axx[ix_], axy[iy_], axth[ith])), float(plat[m])))
        if k_fine is k_coarse and s_fin > histo[-1][1]:
            histo.append((pose, s_fin))

        n_evals = int(evals_ring + scores.size + evals_fin[0] + pay.size + 2)
        extra = None
        if grille is not None:
            extra = {"incoherence": incoherence(grille, contexte["pts"], pose)}
            try:      # bassins arbitres (combo, score, incoherence, pose) —
                extra["bassins"] = arbitres   # diagnostic pour l'interface
            except NameError:
                pass
        return Resultat(pose, float(score_fn(pose)), histo, n_evals,
                        time.perf_counter() - debut,
                        paysage=(axx, axy, pose[2], pay), extra=extra)

# temps_reel.py — Cartographie LIVE : acquisition continue → localisation → mapping.
#
# Architecture (tous les modules voisins sont réutilisés, aucun n'est modifié) :
#   - FluxSerie (thread) : lit le port série EN CONTINU (acquisition.read_scan)
#     et empile chaque scan 360° complet dans un tampon. Jamais d'attente de
#     N tours : les scans reçus PENDANT un cycle de traitement sont moyennés
#     (par secteur d'1°) au début du cycle suivant — le moyennage suit le
#     traitement au lieu de le bloquer.
#   - Vue (boucle matplotlib ~10 Hz), un cycle = :
#       1. vider le tampon (au plus MAX_BATCH scans, les plus vieux sont jetés) ;
#       2. LOCALISATION : partie en cours, NON MODIFIÉE — l'app appelle
#          uniquement algo.run(score_fn, pose0, contexte) du registre
#          scan_matching. Le score passé EST notre jauge d'appariement 0..1
#          (si le scan tombe sur les murs de la carte, cf. _qualite), pas la
#          métrique interne de Resultat.score : sa sémantique a déjà changé
#          trois fois (-d², occupation 0..255, vraisemblance 0..1) et une
#          échelle inconnue rendait tout seuil absurde (tous les scans
#          rejetés). Sous le seuil (max d'un plancher absolu et d'une fraction
#          de la médiane récente, décroissante à chaque refus consécutif) le
#          scan est rejeté : carte et pose ne bougent pas ;
#       3. MAPPING : CarteLive, grille persistante réutilisant
#          occupation._raster avec oubli exponentiel (ρ) : un mur touché en
#          continu reste occupé, un objet disparu n'est plus que traversé →
#          ses cases retombent vers « libre » ;
#       4. rendu des 3 vues : nuage courant + trajectoire, carte d'occupation
#          (occupé = dégradé rouge proche du robot → noir loin), carte
#          vectorielle (reconstruite toutes les VEC_EVERY secondes).
#   - « E » exporte la carte au format de cartographie.py dans
#     Data\Cartes\Live_<date>\ (carte.csv, occupation.csv, vectoriel.csv +
#     rendus PNG via visualisation.py).
#   - Mode --replay : rejoue des scans enregistrés (assemblage.json de
#     cartographie.py, ou un dossier de CSV) dans la même chaîne, sans
#     matériel. Pose initiale = vérité de l'assemblage bruitée (protocole
#     « départ bruité » de test_scan_matching) et erreur par scan affichée :
#     le replay mesure la dérive de la chaîne live complète.
#
# Touches : Espace pause/reprise · E exporter · Q quitter
# Usage   : python temps_reel.py [COMx]
#           python temps_reel.py --replay Data\Cartes\salle\assemblage.json
# Dépendances : pyserial, matplotlib, numpy, scipy

import json
import math
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.collections import LineCollection
from scipy.ndimage import distance_transform_edt

try:  # modules voisins (même dossier)
    import acquisition as acq
    import algo_base as base
    import cartographie as carto
    import occupation as occ
    import scan_matching as sm
    import vectorielle as vec
except ImportError:  # importé comme paquet
    from . import acquisition as acq
    from . import algo_base as base
    from . import cartographie as carto
    from . import occupation as occ
    from . import scan_matching as sm
    from . import vectorielle as vec

RES = 0.10            # résolution de la grille live (m)
COTE = 20.0           # carte fixe carrée centrée sur l'origine (m)
RHO = 0.90            # oubli exponentiel par cycle : fenêtre ~1/(1-ρ) cycles ;
                      # un mur touché en continu reste solide, un objet qui ne
                      # renvoie plus d'impact fond vers « libre » en ~1 s à 10 Hz
PERIODE_MS = 100      # période du cycle de traitement
MAX_BATCH = 12        # scans moyennés au max par cycle (les plus vieux sont jetés)
TAMPON_MAX = 120      # scans gardés en attente (~12 s à 10 Hz, sinon les vieux tombent)
MAX_ARCHIVE = 1800    # scans consolidés gardés pour l'export / le vectoriel
VEC_EVERY = 8.0        # secondes entre deux reconstructions de la carte vectorielle
VEC_FENETRE = 30       # nb de derniers scans consolidés nourrissant le vectoriel
REDRAW_EVERY_S = 2.0   # redessin des 3 vues : pas à chaque cycle (matplotlib domine),
                       # mais toutes les 3–5 s ; un refus ou une touche redessine à la demande
QUALITE_REL = 0.30    # rejet si qualité < 30 % de la médiane des qualités récentes
QUALITE_MIN = 0.05    # plancher absolu très bas : accepter presque tous les scans
                      # (tenu à la main = champ obstrué, scores bas mais utiles)
                      # l'oubli exponentiel (RHO=0.90) estompe les données médiocres
SIGMA = 0.20          # largeur (m) du noyau gaussien de la jauge d'appariement
MAX_DELTA_D_M = 1.5           # avance max entre deux scans acceptés (m)
MAX_DELTA_THETA_DEG = 120.0   # rotation max entre deux scans acceptés (deg)
                              # au-delà, il faut un score excellent (>= 0.50)
                              # pour accepter le saut (sinon c'est une fausse
                              # localisation qui crée la dérive / le "L")
MAX_DELTA_SCORE_BON = 0.50    # score mini pour accepter un saut de pose
                              # important (> MAX_DELTA_D_M ou > MAX_DELTA_THETA_DEG)
MAX_DELTA_COMBINE_D_M = 1.0   # si on avance plus de 1,0 m, rotation limitée
                              # à 60° (mouvement combiné suspect)
MAX_DELTA_COMBINE_THETA_DEG = 60.0
CARTE_SCORE_MIN = 0.25        # score mini pour mettre la carte à jour. En
                              # dessous, on accepte la pose pour le suivi mais
                              # on n'intègre pas le scan dans la grille
RELOC_ECHECS = 8              # après ce nb de refus consécutifs, on tente une
                              # relocalisation sur plusieurs poses candidates
RELOC_POSES_MAX = 6           # nb de poses candidates testées (origine +
                              # dernières positions des lidars)
RELOC_MAX_D_M = 4.0           # borne de saut pour la relocalisation de secours
RELOC_MAX_THETA_DEG = 180.0   # idem pour l'angle
RELOC_SCORE_MIN = 0.20        # score mini pour accepter une relocalisation
                              # (plus bas que CARTE_SCORE_MIN : on accepte la
                              # pose mais on met à jour la carte avec prudence)
MODE_RECUP_ECHECS = 5         # au-delà de ce nb de refus, on passe en mode
                              # récupération : bornes élargies, seuil carte baissé
POSE_LISS_ALPHA = 0.3         # lissage de la pose acceptée (0 = aucun, 1 = figé)
                              # réduit le bruit de la localisation scan à scan
MIN_PREUVE = 0.3              # preuves cumulées (hits+free) mini pour qu'une
                              # case soit considérée connue. En dessous, un
                              # impact isolé (point fantôme) s'estompe en
                              # ~11 cycles à 10 Hz (oubli RHO=0.90) et redevient
                              # « inconnu » au lieu de rester figé en « occupé »;
                              # un mur rescané en continu maintient ses preuves
                              # au-dessus de ce seuil
SEUIL_DECAY = 0.90    # le seuil décroît à chaque refus consécutif (anti-blocage :
                      # la médiane des acceptés seul gèle le seuil en zone à
                      # qualités basses et tue le suivi)
PROFILE_SEUIL_S = 0.5 # alerte si un cycle dépasse ce temps (s)
ALGO = "multi-rés."   # algorithme du registre scan_matching utilisé en live
PROXIMITE = 5.0       # distance (m) à partir de laquelle le rouge devient noir
REPLAY_PERIODE = 0.25 # secondes entre deux scans rejoués (cadence simulée)


# --------------------------------------------------------------------------- #
# Acquisition continue
# --------------------------------------------------------------------------- #
class FluxSerie(threading.Thread):
    """Lit le port série en boucle et empile chaque scan 360° complet dans le
    tampon. Ne bloque jamais le traitement : le cycle videra le tampon quand
    il le pourra, et moyennera les scans arrivés entre-temps."""

    def __init__(self, port, tampon, arret):
        super().__init__(daemon=True)
        self.port, self.tampon, self.arret = port, tampon, arret

    def run(self):
        try:
            with serial.Serial(self.port, acq.BAUD, timeout=5) as ser:
                print(f"Connecté sur {self.port} — acquisition continue.", flush=True)
                rates = 0
                while not self.arret.is_set():
                    lignes = acq.read_scan(ser)
                    if lignes is None:  # 3 dumps consécutifs illisibles
                        rates += 1
                        if rates >= 3:
                            print("Flux série interrompu — thread arrêté.", flush=True)
                            break
                        continue
                    rates = 0
                    scan = self._parser(lignes)
                    if len(scan):
                        self.tampon.append(scan)
        except serial.SerialException as exc:
            print(f"Port série indisponible : {exc}", flush=True)
            self.arret.set()

    @staticmethod
    def _parser(lignes):
        """Lignes CSV « N,Angle,Distance,Intensity,x,y » → tableau (M, 3)."""
        vals = []
        for l in lignes:
            parts = l.split(",")
            try:
                vals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except (ValueError, IndexError):  # ligne bruitée
                continue
        return np.array(vals, dtype=float).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Replay (mêmes tuyaux que le port série, sans matériel)
# --------------------------------------------------------------------------- #
def charger_assemblage(chemin):
    """Replay : (fichiers CSV, poses vérités relatives au premier scan).

    chemin est un assemblage.json (poses de cartographie.py) ou un dossier de
    CSV (alors aucune vérité). Le premier scan posant l'origine de la carte
    live, la vérité du scan k est sa pose transformée dans le repère du
    premier scan — c'est ce que l'app doit retrouver.
    """
    chemin = Path(chemin)
    if chemin.suffix.lower() != ".json":
        return sorted(chemin.glob("*.csv")), None
    scans = json.loads(chemin.read_text(encoding="utf-8")).get("scans", [])
    fichiers = [Path(e["file"]) for e in scans]
    poses = [np.array((e["x"], e["y"], math.radians(e["theta_deg"]))) for e in scans]
    if not poses:
        return fichiers, None

    def inverse(pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.array((-(c * pose[0] + s * pose[1]),
                         s * pose[0] - c * pose[1], -pose[2]))

    def composer(a, b):  # a ∘ b
        ca, sa = math.cos(a[2]), math.sin(a[2])
        return np.array((ca * b[0] - sa * b[1] + a[0],
                         sa * b[0] + ca * b[1] + a[1], a[2] + b[2]))

    inv0 = inverse(poses[0])
    return fichiers, [composer(inv0, p) for p in poses]


class FluxReplay(threading.Thread):
    """Rejoue des scans enregistrés dans le tampon, au rythme simulé. Les CSV
    passent par le même analyseur que le port série : la chaîne est identique."""

    def __init__(self, fichiers, tampon, arret):
        super().__init__(daemon=True)
        self.fichiers, self.tampon, self.arret = fichiers, tampon, arret

    def run(self):
        for f in self.fichiers:
            if self.arret.is_set():
                break
            try:
                lignes = Path(f).read_text(encoding="utf-8-sig").splitlines()
            except OSError as exc:
                print(f"  illisible : {f} ({exc})", flush=True)
                continue
            scan = FluxSerie._parser(lignes)
            if len(scan):
                self.tampon.append(scan)
                print(f"  replay : {Path(f).name} ({len(scan)} points)", flush=True)
                time.sleep(REPLAY_PERIODE)
        print("Replay terminé — la fenêtre reste ouverte (E : export, Q : quitter).",
              flush=True)


# --------------------------------------------------------------------------- #
# Consolidation (moyenne + filtre)
# --------------------------------------------------------------------------- #
def consolider(scans, step=1.0):
    """Moyenne par secteur d'angle des scans reçus pendant un cycle.

    scans : liste de tableaux (angle°, dist mm, inten).
    Retourne (points (N, 2) dans le repère lidar, intensité (N,)).
    Convention +sin (comme cartographie.load_scan) : cohérente avec toute la
    pile de localisation et les cartes déjà exportées.
    """
    bins = {}
    for s in scans:
        for angle, dist, inten in s:
            if dist <= 0:  # pas de retour : exclu
                continue
            b = bins.setdefault(int(angle // step), [0.0, 0.0, 0.0, 0])
            b[0] += angle
            b[1] += dist
            b[2] += inten
            b[3] += 1
    if not bins:
        return np.empty((0, 2)), np.empty(0)
    cols = np.array([bins[k] for k in sorted(bins)], dtype=float)
    ang, dist, inten = cols[:, 0] / cols[:, 3], cols[:, 1] / cols[:, 3], cols[:, 2] / cols[:, 3]
    garde = (dist >= occ.MIN_RANGE * 1000) & (dist <= occ.MAX_RANGE * 1000)
    ang, dist, inten = ang[garde], dist[garde], inten[garde]
    th = np.deg2rad(ang)
    pts = np.column_stack((dist * np.cos(th), dist * np.sin(th))) / 1000.0
    return pts, inten


# --------------------------------------------------------------------------- #
# Carte d'occupation live (compteurs bornés → la carte s'agrémente et oublie)
# --------------------------------------------------------------------------- #
class CarteLive:
    """Grille glissante (fenêtre COTE × COTE à RES) mise à jour incrémentalement.

    Même trame que occupation.build (cases libres traversées / case d'impact),
    mais les compteurs persistent d'un cycle à l'autre avec un oubli
    exponentiel (ρ par cycle) : l'état stationnaire d'une case reflète sa
    balance ACTUELLE impacts/traversées. Un mur touché à chaque cycle reste
    solide même si des rayons le traversent parfois (bruit, ouvertures) ; un
    objet disparu n'est plus que traversé et fond vers « libre » ; un mur hors
    de vue ne reçoit aucun rayon et ne bouge pas. La proba est symétrique
    hits/free (le facteur ×2 de occupation.build ne sert qu'aux
    reconstructions complètes hors-ligne).
    """

    def __init__(self, res=RES, cote=COTE):
        self.res = float(res)
        self.x0 = self.y0 = -cote / 2
        self.n = max(int(round(cote / self.res)), 2)
        self.hits = np.zeros(self.n * self.n, dtype=np.float32)
        self.free = np.zeros(self.n * self.n, dtype=np.float32)
        self.proba = np.full(self.n * self.n, occ.UNKNOWN, dtype=np.uint8)
        self.sources = deque(maxlen=256)
        self.grille = None  # vue occ.Grille pour le rendu / la localisation

    def update(self, rays):
        """rays : liste de (points globaux (N, 2), origine lidar (2,))."""
        ends, srcs = [], []
        for pts, origin in rays:
            if not len(pts):
                continue
            origin = np.asarray(origin, dtype=float)[:2]
            d = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
            self.sources.append(origin)
            keep = (d >= occ.MIN_RANGE) & (d <= occ.MAX_RANGE)
            if keep.any():
                ends.append(pts[keep])
                srcs.append(np.broadcast_to(origin, (int(keep.sum()), 2)))
        if not ends:
            return
        self.hits *= RHO  # oubli exponentiel AVANT l'ajout : h ← ρ·h + nouveau
        self.free *= RHO
        occ._raster(np.vstack(ends), np.vstack(srcs),
                    np.array((self.x0, self.y0)), self.res, self.n, self.n,
                    self.hits, self.free)
        total = self.hits + self.free
        connu = total >= MIN_PREUVE          # assez de preuves pour être connue
        self.proba = np.full(self.n * self.n, occ.UNKNOWN, dtype=np.uint8)
        self.proba[connu] = np.clip(np.round(self.hits[connu] / total[connu] * 255),
                                    0, 255).astype(np.uint8)
        self._vue()

    def _vue(self):
        """(Re)construit la vue occ.Grille consommée par le rendu et la
        localisation (sources dédupliquées pour l'export)."""
        src = (np.unique(np.round(np.array(self.sources), 4), axis=0)
               if self.sources else np.empty((0, 2)))
        self.grille = occ.Grille(self.x0, self.y0, self.res,
                                 self.proba.reshape(self.n, self.n), src)

    def recentrer(self, pose):
        """Recentre la carte sur le robot quand il sort du quart central.

        La fenêtre reste bornée (COTE), mais elle glisse : décalage de cases
        entières dans les trois tableaux, x0/y0 suivent — les coordonnées
        monde restent absolues, seuls les murs sortis de la fenêtre sont
        abandonnés. Sans cela, un trajet de plus de ~10 m dans une direction
        sort de la grille et la localisation perd ses murs (l'export réel
        atteignait déjà 17 m de large sur 20).
        """
        i = (pose[0] - self.x0) / self.res
        j = (pose[1] - self.y0) / self.res
        marge = self.n * 0.20          # réagit dès que le robot sort du cinquième central
        if marge <= i <= self.n - marge and marge <= j <= self.n - marge:
            return
        di = int(round(i - self.n / 2))
        dj = int(round(j - self.n / 2))
        self.hits = self._decale(self.hits, di, dj, 0.0)
        self.free = self._decale(self.free, di, dj, 0.0)
        self.proba = self._decale(self.proba, di, dj, occ.UNKNOWN)
        self.x0 += di * self.res
        self.y0 += dj * self.res
        self._vue()

    def _decale(self, plat, di, dj, vide):
        """Tableau plat (n×n, ligne = y, colonne = x) décalé de (di, dj) cases."""
        a = plat.reshape(self.n, self.n)
        out = np.full_like(a, vide)
        r0, r1 = max(0, -dj), min(self.n, self.n - dj)
        c0, c1 = max(0, -di), min(self.n, self.n - di)
        if r1 > r0 and c1 > c0:
            out[r0:r1, c0:c1] = a[r0 + dj:r1 + dj, c0 + di:c1 + di]
        return out.ravel()


# --------------------------------------------------------------------------- #
# Rendu de la carte d'occupation live
# --------------------------------------------------------------------------- #
def to_rgb_live(grille, pose, portee=PROXIMITE):
    """Image RGBA de la grille pour la vue live : les cases occupées sont
    colorées selon leur distance au robot — rouge = proche, noir = loin
    (dégradé complet à `portee` mètres). Libre = blanc, inconnu = gris."""
    p = grille.proba.astype(np.float32) / 255.0
    connu = grille.proba != occ.UNKNOWN
    xs = grille.x0 + (np.arange(grille.nx) + 0.5) * grille.res
    ys = grille.y0 + (np.arange(grille.ny) + 0.5) * grille.res
    d = np.hypot(xs[None, :] - pose[0], ys[:, None] - pose[1])
    t = np.clip(d / max(portee, 1e-6), 0.0, 1.0)[..., None]  # 0 = sur le robot
    rouge = np.array((0.85, 0.12, 0.12), dtype=np.float32)
    coul = rouge * (1.0 - t) + occ.DARK * t                  # occupé : rouge → sombre
    img = np.empty((grille.ny, grille.nx, 4), dtype=np.float32)
    for k in range(3):
        img[..., k] = np.where(connu, 1.0 - p * (1.0 - coul[..., k]),
                               occ.UNKNOWN_RGB[k])
    img[..., 3] = 1.0
    return img


# --------------------------------------------------------------------------- #
# Application (3 vues live)
# --------------------------------------------------------------------------- #
def choisir_algo(nom):
    """Sélection dans le registre de la localisation (partie non modifiée)."""
    algos = [a for a in sm.ALGORITHMES if a.disponible("grille")]
    for a in algos:
        if a.nom == nom:
            return a
    return algos[0]


class Vue:
    def __init__(self, tampon, arret, verites=None):
        self.tampon = tampon
        self.arret = arret
        self.carte = CarteLive()
        self.pose = np.zeros(3)                       # x, y, θ (rad) — origine au 1er scan
        self.traj = deque(maxlen=20000)               # positions successives
        self.archives = deque(maxlen=MAX_ARCHIVE)     # (pts local, inten, pose)
        self.scores = deque(maxlen=20)                # scores récents, acceptés ET
                                                      # refusés : le seuil suit le
                                                      # niveau réel, pas seulement
                                                      # les réussites
        self.echecs = 0                               # refus consécutifs
        self.algo = choisir_algo(ALGO)
        self.verites = list(verites) if verites else None  # replay : poses attendues
        self.rng = np.random.default_rng(0)           # bruit d'amorçage reproductible
        self.traites = 0                              # scans consolidés placés
        self.vect = None
        self.vec_lock = threading.Lock()          # protège self.vect (build thread vs dessin)
        self.vec_thread = None                    # thread de reconstruction vectorielle
        self.vec_stamp = 0.0
        self.dernier = None                           # points globaux du dernier scan
        self.pause = False
        self.notice = ""
        self.recus = self.acceptes = self.refuses = self.ignores = 0
        self.cadence = 0.0
        self.stamp = time.monotonic()
        self.timer = None
        self.last_draw = 0.0            # dernier redessin des vues (throttlé)
        self._construire()

    # -- interface ---------------------------------------------------------- #
    def _construire(self):
        for name in ("save", "fullscreen", "home", "back", "forward", "pan",
                     "zoom", "xscale", "yscale", "all_axes", "grid_major",
                     "grid_minor"):
            try:
                plt.rcParams[f"keymap.{name}"] = []
            except KeyError:
                pass
        self.fig = plt.figure(figsize=(14.2, 8.2))
        self.fig.canvas.manager.set_window_title("Cartographie live")
        self.fig.patch.set_facecolor("#ffffff")

        def cadre(rect, titre, face="#fbfbfd", grille=False):
            ax = self.fig.add_axes(rect)
            ax.set_facecolor(face)
            ax.set_aspect("equal")
            if grille:
                ax.grid(True, color="#e7e9ef", lw=0.8, zorder=0)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#e7e9ef")
            ax.set_title(titre, fontsize=12, fontweight="bold", color="#1f2430", pad=10)
            return ax

        # Vue 1 : nuage courant + trajectoire. Vues 2-3 : occupation, vectoriel.
        self.ax = cadre((0.030, 0.44, 0.30, 0.47), "Assemblage live", grille=True)
        self.scan_pts = self.ax.scatter([], [], s=12, c=["#3b5bdb"],
                                        linewidths=0, zorder=3)
        self.traj_ligne, = self.ax.plot([], [], color="#e8590c", lw=1.6, zorder=4)
        self.ax.scatter((0,), (0,), marker="o", s=60, facecolors="none",
                        edgecolors="#9aa1ad", linewidths=1.4, zorder=5)
        self.cap, = self.ax.plot([], [], color="#e8590c", lw=1.8, zorder=6)

        self.grid_ax = cadre((0.360, 0.44, 0.30, 0.47), "Carte d'occupation", "#f3f4f7")
        self.grid_img = self.grid_ax.imshow(np.zeros((2, 2, 4)), origin="lower",
                                            extent=(-1, 1, -1, 1),
                                            interpolation="nearest", zorder=2)
        self.grid_src, = self.grid_ax.plot([], [], "x", ms=6, mew=1.3,
                                           color="#e11d48", zorder=4)

        self.vec_ax = cadre((0.690, 0.44, 0.30, 0.47), "Carte vectorielle")
        self.vec_seg = LineCollection([], colors=[tuple(float(c) for c in vec.DARK)],
                                      linewidths=2.0, zorder=3)
        self.vec_ax.add_collection(self.vec_seg)
        self.vec_src, = self.vec_ax.plot([], [], "x", ms=6, mew=1.3,
                                         color="#e11d48", zorder=4)

        self.status = self.fig.text(0.5, 0.375, "en attente du premier scan…",
                                    ha="center", va="center", fontsize=9,
                                    color="#1f2430")
        self.fig.text(0.5, 0.06,
                      "Espace : pause/reprise     E : exporter la carte     Q : quitter",
                      ha="center", va="center", fontsize=8, color="#6b7280")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        try:  # boucle de traitement (pas de timer en mode Agg / tests)
            if plt.get_backend().lower() != "agg":
                self.timer = self.fig.canvas.new_timer(interval=PERIODE_MS)
                self.timer.add_callback(self.cycle)
                self.timer.start()
        except Exception:
            self.timer = None

    # -- cycle de traitement ------------------------------------------------ #
    def _drainer(self):
        items = []
        while self.tampon:
            try:
                items.append(self.tampon.popleft())
            except IndexError:  # le thread a tout consommé entre-temps
                break
        if len(items) > MAX_BATCH:  # traitement en retard : on ne garde que les
            self.ignores += len(items) - MAX_BATCH  # plus récents
            items = items[-MAX_BATCH:]
        self.recus += len(items)
        return items

    @staticmethod
    def _qualite(pts, grille):
        """Jauge d'appariement 0..1, indépendante de la métrique de
        Resultat.score — dont la sémantique a déjà changé trois fois (-d²,
        occupation 0..255, vraisemblance 0..1), chaque changement rendant le
        seuil absurde (dernier incident : tout rejeté à -0,0). Vraisemblance
        moyenne exp(-d²/2σ²) des points sur les murs de la carte, σ = 20 cm :
        même champ que le likelihood de algo_base, mais calculé ici car leur
        champ_distance est leur code, en cours de réécriture."""
        occupe = np.asarray(grille.proba) >= 200
        if not occupe.any():                    # aucune référence : rien à juger
            return lambda pose: 1.0
        d = distance_transform_edt(~occupe) * grille.res
        img = np.exp(-(np.minimum(d, 1.0) ** 2) / (2.0 * SIGMA ** 2))
        pts = base.echantillon(pts)

        def score(pose):
            monde = carto.transform(pts, pose)
            valeurs = base.bilineaire(img, grille.x0, grille.y0, grille.res,
                                      monde, hors=0.0)
            # Ne pénaliser que les points dans la carte connue : un scan
            # partiel (main devant le lidar) qui regarde dans l'inconnu ne
            # doit pas faire chuter la moyenne.
            connus = valeurs > 0.0
            if not connus.any():
                return 1.0
            return float(valeurs[connus].mean())
        return score

    def _seuil(self):
        """Seuil d'acceptation du scan courant (max entre plancher et relatif).

        La médiane porte sur les qualités acceptées ET refusées (sinon, après un
        passage en zone à scores hauts, elle ne peut plus descendre et le
        suivi meurt — observé en réel : 8 refus consécutifs, seuil gelé à
        65,3 pour des scores honnêtes à 19–32). Le facteur de décroissance
        par refus consécutif accélère la reprise ; le plancher absolu garantit
        qu'on n'accepte jamais du bruit, même après une longue série de refus.
        """
        if not self.scores:
            return QUALITE_MIN
        return max(QUALITE_MIN,
                   QUALITE_REL * float(np.median(self.scores))
                   * (SEUIL_DECAY ** self.echecs))

    def _tenter_relocalisation(self, pts, inten, fn, ctx):
        """Relocalise le scan sur plusieurs poses candidates.
        Teste l'origine (0,0,0), les dernières positions des lidars, et la
        dernière pose connue. Retourne True si une candidate a été acceptée."""
        candidates = [np.zeros(3)]  # origine de la carte
        # ajouter les dernières positions des lidar (poses déjà acceptées)
        if self.carte.grille is not None and len(self.carte.grille.sources):
            srcs = np.asarray(self.carte.grille.sources)
            # garder les plus récentes, dédoublonnées
            for s in srcs[::-max(1, len(srcs)//3)]:
                candidates.append(np.array((s[0], s[1], self.pose[2])))
        candidates.append(self.pose.copy())  # dernière pose connue

        meilleure_pose = None
        meilleur_score = -1.0
        for pose0 in candidates[:RELOC_POSES_MAX]:
            res = self.algo.run(fn, pose0, ctx)
            pose = res.pose
            score = float(fn(pose))
            if score > meilleur_score:
                meilleur_score = score
                meilleure_pose = pose

        if meilleure_pose is None:
            return False
        pose = meilleure_pose
        score = meilleur_score

        # pas de verrou de saut en relocalisation de secours : on se fie
        # uniquement au score qualité (si le scan matche bien la carte,
        # la pose est bonne, même si le saut par rapport à la pose figée
        # est important — le robot a pu bouger pendant les refus)
        seuil = max(RELOC_SCORE_MIN, self._seuil())
        if not np.isfinite(score) or not np.isfinite(pose).all() or score < seuil:
            return False
        # accepté : appliquer comme un scan normal
        self.pose = np.asarray(pose, dtype=float)
        self.pose[2] = carto.wrap_angle(self.pose[2])
        self._pose_lisse = self.pose.copy()  # repartir du bon pied après relocalisation
        self.scores.append(score)
        self.echecs = 0
        glob = carto.transform(pts, self.pose)
        # en relocalisation, on met à jour la carte seulement si score >= CARTE_SCORE_MIN
        if score >= CARTE_SCORE_MIN:
            self.carte.update([(glob, self.pose[:2])])
        self.dernier = glob
        self.traj.append(self.pose[:2].copy())
        self.archives.append((pts, inten, self.pose.copy()))
        self.acceptes += 1
        if self.notice.startswith("scan rejeté"):
            self.notice = ""
        self.notice = "relocalisé"
        print(f"  scan accepté : relocalisation "
              f"pose ({self.pose[0]:+.2f}, {self.pose[1]:+.2f}, "
              f"{np.rad2deg(self.pose[2]):+.1f}°) score {score:.2f}",
              flush=True)
        self._dessiner()
        return True

    def _lisser_pose(self, pose):
        """Lissage exponentiel de la pose pour réduire le bruit scan à scan.
        Alpha faible = plus de lissage, mais plus de latence.
        Ne lisse pas si le saut est important (vrai mouvement, pas du bruit)."""
        if not hasattr(self, '_pose_lisse'):
            self._pose_lisse = pose.copy()
            return pose
        dp = pose - self._pose_lisse
        dp[2] = carto.wrap_angle(dp[2])
        if math.hypot(dp[0], dp[1]) > 0.15 or abs(math.degrees(dp[2])) > 8.0:
            # vrai mouvement : pas de lissage, on prend la pose telle quelle
            self._pose_lisse = pose.copy()
            return pose
        self._pose_lisse = (1 - POSE_LISS_ALPHA) * self._pose_lisse + POSE_LISS_ALPHA * pose
        self._pose_lisse[2] = carto.wrap_angle(self._pose_lisse[2])
        return self._pose_lisse.copy()

    def cycle(self, *_args):
        t_total0 = time.perf_counter()
        t_drain = t_consol = t_loc = t_carte = t_vec = t_dessin = 0.0
        scans = self._drainer()
        t_drain = time.perf_counter() - t_total0
        now = time.monotonic()
        dt = now - self.stamp
        self.stamp = now
        if dt > 0 and self.recus:
            inst = self.recus / dt
            self.cadence = inst if not self.cadence else 0.7 * self.cadence + 0.3 * inst
            self.recus = 0
        if self.pause or not scans:
            return
        pts, inten = consolider(scans)
        t_consol = time.perf_counter() - t_total0 - t_drain
        if not len(pts):
            return
        self.traites += 1
        t0_loc = time.perf_counter()

        if self.carte.grille is None:
            # premier scan : il définit l'origine de la carte (pose 0, 0, 0)
            pose, score = np.zeros(3), None
        else:
            if self.verites is not None and self.traites <= len(self.verites):
                # replay : pose0 = vérité bruitée (protocole « départ bruité »
                # de test_scan_matching), pour mesurer la dérive scan à scan
                v = self.verites[self.traites - 1]
                pose0 = v + np.array((self.rng.uniform(-0.25, 0.25),
                                      self.rng.uniform(-0.25, 0.25),
                                      math.radians(self.rng.uniform(-7.0, 7.0))))
            else:
                pose0 = self.pose.copy()
            self.carte.recentrer(pose0)  # la fenêtre glisse avec le robot
            fn = self._qualite(pts, self.carte.grille)
            # segments volontairement hors de la boucle : la carte vectorielle
            # est reconstruite sur des scans déjà localisés (boucle de retour
            # qui amplifie la dérive, cf. replay salle) — elle reste un rendu
            # et un export, pas une référence.
            ctx = {"pts": pts, "grille": self.carte.grille}
            res = self.algo.run(fn, pose0, ctx)
            pose = res.pose
            score = float(fn(pose))  # notre jauge, jamais Resultat.score
            # verrou de cohérence : pas d'odométrie, le robot à la main ne
            # saute pas de MAX_DELTA_D_M / MAX_DELTA_THETA_DEG en un cycle ;
            # verrou de cohérence : pas d'odométrie, le robot à la main ne
            # saute pas déraisonnablement entre deux scans ;
            # un petit mouvement (< 0,5 m ET < 40°) est accepté si le score
            # dépasse le plancher ; tout saut plus important exige un score
            # excellent (>= 0,60) pour justifier le déplacement, sinon c'est
            # une fausse localisation qui crée la dérive / le "L".
            if self.acceptes > 0 and self.verites is None:
                dp = pose - self.pose
                dp[2] = carto.wrap_angle(dp[2])
                d_dist = math.hypot(dp[0], dp[1])
                d_deg = abs(math.degrees(dp[2]))
                petit_mouvement = d_dist <= 1.0 and d_deg <= 70.0
                saut_justifie = petit_mouvement or score >= MAX_DELTA_SCORE_BON
                if not saut_justifie:
                    self.refuses += 1
                    self.echecs += 1
                    self.scores.append(score)
                    suite = f" (×{self.echecs} consécutifs)" if self.echecs >= 3 else ""
                    self.notice = (f"scan rejeté : saut de pose {d_dist:.2f} m / "
                                   f"{d_deg:.1f}° (trop gros{suite}")
                    print(f"  scan rejeté : saut de pose {d_dist:.2f} m / "
                          f"{d_deg:.1f}°{suite}", flush=True)
                    # relocalisation de secours après 5 refus consécutifs
                    if self.echecs >= RELOC_ECHECS:
                        if self._tenter_relocalisation(pts, inten, fn, ctx):
                            self._dessiner()
                            return
                    self._dessiner()
                    return
            seuil = self._seuil()
            # en récupération, on abaisse le seuil pour retrouver le suivi plus vite
            if self.echecs >= MODE_RECUP_ECHECS:
                seuil = QUALITE_MIN
            if not np.isfinite(score) or not np.isfinite(pose).all() or score < seuil:
                self.refuses += 1
                self.echecs += 1
                self.scores.append(score)
                suite = f" (×{self.echecs} consécutifs)" if self.echecs >= 3 else ""
                self.notice = f"scan rejeté (score {score:.2f} < {seuil:.2f}){suite}"
                print(f"  scan rejeté : score {score:.2f} < {seuil:.2f}{suite}",
                      flush=True)
                # relocalisation de secours depuis l'origine quand on est bloqué
                if self.echecs >= RELOC_ECHECS:
                    if self._tenter_relocalisation(pts, inten, fn, ctx):
                        self._dessiner()
                        return
                self._dessiner()
                return
            self.echecs = 0
            t_loc = time.perf_counter() - t0_loc

        self.pose = np.asarray(pose, dtype=float)
        self.pose[2] = carto.wrap_angle(self.pose[2])  # θ dans ]-180°, 180°]
        self.pose = self._lisser_pose(self.pose)  # lissage pour réduire le bruit
        if score is not None:
            self.scores.append(score)
        if self.verites is not None and self.traites >= 2:
            v = self.verites[self.traites - 1]
            print(f"  replay {self.traites}/{len(self.verites)} : erreur "
                  f"{np.hypot(*(self.pose[:2] - v[:2])) * 100:.1f} cm / "
                  f"{abs(math.degrees(carto.wrap_angle(self.pose[2] - v[2]))):.1f}°",
                  flush=True)
        t0 = time.perf_counter()
        glob = carto.transform(pts, self.pose)
        # carte mise à jour seulement si le scan est bien localisé (évite
        # d'empoisonner la grille avec des scans douteux). En replay, on
        # connaît la vérité : on met toujours à jour.
        if self.verites is not None or score is None or score >= CARTE_SCORE_MIN:
            self.carte.update([(glob, self.pose[:2])])
        else:
            self.notice = (self.notice or "") + " (carte non mise à jour)"
        self.dernier = glob
        self.traj.append(self.pose[:2].copy())
        self.archives.append((pts, inten, self.pose.copy()))
        t_carte = time.perf_counter() - t0
        self.acceptes += 1
        if self.notice.startswith("scan rejeté"):  # le bandeau rouge ne survit
            self.notice = ""                       # pas au scan accepté suivant
        if now - self.vec_stamp >= VEC_EVERY and len(self.archives) >= 2 and \
           (self.vec_thread is None or not self.vec_thread.is_alive()):
            self.vec_stamp = now
            self.vec_thread = threading.Thread(target=self._vectoriel, daemon=True)
            self.vec_thread.start()
            t_vec = 0.0  # non bloquant : le cycle ne attend pas la vectorielle
        else:
            t_vec = 0.0
        t0 = time.perf_counter()
        if now - self.last_draw >= REDRAW_EVERY_S:
            self._dessiner()
            self.last_draw = now
        t_dessin = time.perf_counter() - t0
        t_total = time.perf_counter() - t_total0
        lent = "  << LENT" if t_total > PROFILE_SEUIL_S else ""
        if self.acceptes % 20 == 1 or lent:
            sc = "—" if score is None else f"{score:.2f}"
            print(f"  scan {self.acceptes} : pose ({self.pose[0]:+.2f}, "
                  f"{self.pose[1]:+.2f}, {np.rad2deg(self.pose[2]):+.1f}°) · "
                  f"score {sc} · {len(pts)} pts · {self.cadence:.1f} scans/s{lent}",
                  flush=True)
            print(f"  profil : drain {t_drain*1e3:.0f}ms consol {t_consol*1e3:.0f}ms "
                  f"loc {t_loc*1e3:.0f}ms carte {t_carte*1e3:.0f}ms "
                  f"vec {t_vec*1e3:.0f}ms dessin {t_dessin*1e3:.0f}ms "
                  f"total {t_total*1e3:.0f}ms", flush=True)

    def _vectoriel(self):
        """Carte vectorielle sur une fenêtre glissante de derniers scans.
        Exécuté dans un thread dédié pour ne pas bloquer le cycle de 9 Hz."""
        fen = list(self.archives)[-VEC_FENETRE:]
        rays = [(carto.transform(p, pose), pose[:2]) for p, _i, pose in fen]
        vect = vec.build(rays, grille=self.carte.grille)
        with self.vec_lock:
            self.vect = vect

    # -- rendu -------------------------------------------------------------- #
    def _fenetre(self):
        """Fenêtre carrée sur les cases connues de la grille (commune aux 3 vues)."""
        g = self.carte.grille
        connus = np.argwhere(g.proba != occ.UNKNOWN) if g is not None else []
        if not len(connus):
            return (-2.0, 2.0, -2.0, 2.0)
        (j0, i0), (j1, i1) = connus.min(0), connus.max(0)
        x0, y0 = g.x0 + i0 * g.res, g.y0 + j0 * g.res
        x1, y1 = g.x0 + (i1 + 1) * g.res, g.y0 + (j1 + 1) * g.res
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        span = max(x1 - x0, y1 - y0) * 1.15 + 1.0
        return (cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2)

    def _dessiner(self):
        if self.dernier is not None:
            self.scan_pts.set_offsets(self.dernier)
        if self.traj:
            t = np.array(self.traj)
            self.traj_ligne.set_data(t[:, 0], t[:, 1])
        x, y, th = self.pose
        self.cap.set_data((x, x + 0.6 * math.cos(th)), (y, y + 0.6 * math.sin(th)))
        fen = self._fenetre()
        with self.vec_lock:
            vect = self.vect
        for ax in (self.ax, self.grid_ax, self.vec_ax):
            ax.set_xlim(fen[0], fen[1])
            ax.set_ylim(fen[2], fen[3])
        g = self.carte.grille
        if g is not None:
            self.grid_img.set_data(to_rgb_live(g, self.pose))
            self.grid_img.set_extent(g.extent())
        for src_artist, sources in ((self.grid_src, g.sources if g else None),
                                    (self.vec_src, g.sources if g else None)):
            if sources is not None and len(sources):
                src_artist.set_data(sources[:, 0], sources[:, 1])
        if vect is not None and len(vect):
            self.vec_seg.set_segments(vect.segments.reshape(-1, 2, 2))
            poids = np.clip(vect.hits - 2, 0, 58)  # mur bien soutenu = plus épais
            self.vec_seg.set_linewidths(0.7 + 2.3 * poids / 58.0)
        self._statut()
        self.fig.canvas.draw_idle()

    def _statut(self):
        x, y, th = self.pose
        sc = f"{self.scores[-1]:.2f}" if self.scores else "—"
        etat = "PAUSE" if self.pause else "live"
        self.status.set_text(
            f"{etat} · pose ({x:+.2f}, {y:+.2f}, {np.rad2deg(th):+.1f}°) · score {sc} · "
            f"{self.cadence:.1f} scans/s · acceptés {self.acceptes}, rejetés "
            f"{self.refuses}, ignorés {self.ignores} · algo {self.algo.nom}"
            + (f" · {self.notice}" if self.notice else ""))
        self.status.set_color("#e11d48" if self.notice else "#1f2430")

    # -- export ------------------------------------------------------------- #
    def exporter(self, *_args):
        if not self.archives:
            self.notice = "Rien à exporter : aucun scan accepté."
            self._dessiner()
            return
        dossier = carto.CARTES / ("Live_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        dossier.mkdir(parents=True, exist_ok=True)
        # Sauvegarde des scans bruts pour pouvoir les revoir plus tard.
        scans_dir = dossier / "scans"
        scans_dir.mkdir(exist_ok=True)
        for k, (p, _i, _pose) in enumerate(self.archives):
            if not len(p):
                continue
            d = np.hypot(p[:, 0], p[:, 1]) * 1000.0        # mm
            a = np.rad2deg(np.arctan2(p[:, 1], p[:, 0])) % 360.0  # degrés
            with open(scans_dir / f"scan_{k:04d}.csv", "w", encoding="utf-8",
                      newline="") as f:
                f.write("N,Angle,Distance,Intensity,x,y\n")
                for j in range(len(p)):
                    f.write(f"{j},{a[j]:.2f},{d[j]:.1f},{int(_i[j])},"
                            f"{p[j, 0] * 1000:.1f},{p[j, 1] * 1000:.1f}\n")
        print(f"{len(self.archives)} scan(s) bruts archivés dans {scans_dir}",
              flush=True)
        pts = np.vstack([carto.transform(p, pose) for p, _i, pose in self.archives])
        inten = np.concatenate([i for _p, i, _pose in self.archives])
        table = np.column_stack((np.arange(len(pts)), np.round(pts * 1000),
                                 np.clip(np.round(inten), 0, 255))).astype(np.int64)
        with open(dossier / "carte.csv", "w", encoding="utf-8", newline="") as f:
            f.write("N,x(mm),y(mm),Intensity\n")
            np.savetxt(f, table, fmt="%d", delimiter=",")
        print(f"Carte live ({len(pts)} points) : {dossier / 'carte.csv'}", flush=True)
        if self.carte.grille is not None:
            occ.save_csv(self.carte.grille, dossier / "occupation.csv")
            print(f"Occupation ({self.carte.grille.nx} × {self.carte.grille.ny} "
                  f"cases) : {dossier / 'occupation.csv'}", flush=True)
        with self.vec_lock:
            vect = self.vect
        if vect is not None:
            vec.save_csv(vect, dossier / "vectoriel.csv")
            print(f"Vectoriel ({len(vect)} segments) : "
                  f"{dossier / 'vectoriel.csv'}", flush=True)
        self.notice = f"carte exportée → {dossier.name}"
        try:  # rendus PNG, même chaîne que cartographie.py
            subprocess.run([sys.executable,
                            str(Path(__file__).resolve().with_name("visualisation.py")),
                            str(dossier), "--force"],
                           capture_output=True, text=True, timeout=180)
        except Exception as exc:
            print(f"  rendu indisponible : {exc}", flush=True)
        png = dossier / "carte.png"
        if png.exists():
            carto.Carte.open_image(png)
        self._dessiner()

    # -- événements --------------------------------------------------------- #
    def on_key(self, event):
        k = (event.key or "").lower()
        if k in (" ", "space"):
            self.pause = not self.pause
            if not self.pause:
                self.notice = ""
            self._dessiner()
        elif k == "e":
            self.exporter()
        elif k in ("q", "escape"):
            plt.close(self.fig)

    def on_close(self, _event):
        self.arret.set()
        if self.timer is not None:
            try:
                self.timer.stop()
            except Exception:
                pass

    def run(self):
        print(f"Cartographie live — algo « {self.algo.nom} », carte {COTE:.0f} m à "
              f"{RES * 100:.0f} cm, moyennage {MAX_BATCH} scans max/cycle.", flush=True)
        print("Espace : pause · E : exporter · Q : quitter", flush=True)
        plt.show()


def main():
    try:  # console Windows : caractères non affichables remplacés au lieu de planter
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    replay = None
    if "--replay" in args:
        i = args.index("--replay")
        try:
            replay = args[i + 1]
        except IndexError:
            sys.exit("Usage : --replay <assemblage.json | dossier de CSV>")
        del args[i:i + 2]
    tampon = deque(maxlen=TAMPON_MAX)
    arret = threading.Event()
    if replay is not None:
        fichiers, verites = charger_assemblage(replay)
        if not fichiers:
            sys.exit(f"Aucun scan trouvé pour --replay {replay}")
        print(f"Replay de {len(fichiers)} scans depuis {replay}", flush=True)
        FluxReplay(fichiers, tampon, arret).start()
        Vue(tampon, arret, verites).run()
        return
    port = args.pop(0) if args and args[0].upper().startswith("COM") else None
    if not port:
        from serial.tools import list_ports
        trouves = [p.device for p in list_ports.comports()]
        if len(trouves) == 1:
            port = trouves[0]
        else:
            sys.exit(f"Port série ambigu, précisez-le : python temps_reel.py COMx. "
                     f"Détectés : {trouves or 'aucun'}")
    FluxSerie(port, tampon, arret).start()
    Vue(tampon, arret).run()


if __name__ == "__main__":
    main()

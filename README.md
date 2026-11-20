# SLAM 2D avec LiDAR

L'objectif de ce TP "Cartographie et Localisation" est de cartographier la salle de classe en 2D avec un robot équipé d'un lidar 2D et le localiser dedans.

## Travaux

**Jour 1 : cartographie**
- Acquisition des scans moyennés depuis l'ESP32 (`acquisition.py`).
- Assemblage de 8 scans de la salle en réglant la pose de chacun à la main (`cartographie.py`, `visualisation.py`).
- Carte d'occupation (`occupation.py`).
- Carte vectorielle (`vectorielle.py`).

**Jour 2 : localisation**
Cinq méthodes de scan matching, une par fichier `algo_*.py` : exhaustive, descente de gradient, multi-résolution, particules, calcul direct (ICP). Elles marchent sur la grille, la carte vectorielle et un scan de référence (sauf le calcul direct, qui ne marche pas sur une grille).

**Jour 3 : SLAM**
- `temps_reel.py` : localisation et cartographie en direct (ou en rejouant des scans enregistrés avec `--replay`).
- `reconstruire_live.py` : refait une carte à partir des scans bruts sauvegardés pendant un live.

## Résultats

Mes meilleurs essais sont dans `Data/Cartes/` :
- `Live_20261103_180507`
- `Live_20261104_083312`
- `Live_reconstruit_20261120_231726`

Dans chaque dossier : `carte.png` (nuage de points), `occupation.png`, `vectoriel.png`.

## Évaluation des méthodes (Jour 2)

Protocole (`tableau_erreurs.py`) : chaque scan est localisé depuis une pose initiale volontairement fausse, 3 tirages par niveau avec les **mêmes poses de départ pour toutes les méthodes**. Réussite : erreur < 0,10 m et dθ < 5°. err, dθ et temps sont les médianes au niveau ±0,30 m / ±10°.

**Sur la carte « salle »** :

| algo | réf | err | dθ | temps | réussite (±0,15 / ±0,30 / ±0,50 m) |
|---|---|---|---|---|---|
| exhaustif | grille | 4.0 cm | 0.7° | 440 ms | 88 / 92 / 75 % |
| exhaustif | vectorielle | 4.1 cm | 0.5° | 465 ms | 88 / 92 / 83 % |
| gradient | grille | 2.9 cm | 0.8° | 26 ms | 100 / 88 / 92 % |
| gradient | vectorielle | 2.6 cm | 0.6° | 28 ms | 100 / 96 / 96 % |
| multi-rés. | grille | 3.0 cm | 0.0° | 127 ms | 100 / 100 / 100 % |
| multi-rés. | vectorielle | 3.0 cm | 0.0° | 126 ms | 100 / 100 / 100 % |
| particules | grille | 3.8 cm | 0.5° | 92 ms | 88 / 88 / 88 % |
| particules | vectorielle | 3.9 cm | 0.5° | 275 ms | 88 / 88 / 88 % |
| direct | vectorielle | 2.3 cm | 0.1° | 6 ms | 100 / 92 / 71 % |

**Scan contre scan** :

| algo | réf | err | dθ | temps |
|---|---|---|---|---|
| exhaustif | grille | 6.3 cm | 3.5° | 1278 ms |
| exhaustif | nuage | 1.9 cm | 2.0° | 1592 ms |
| exhaustif | vectorielle | 1.9 cm | 2.0° | 1443 ms |
| gradient | grille | 5.1 cm | 0.7° | 216 ms |
| gradient | nuage | 5.1 cm | 0.7° | 239 ms |
| gradient | vectorielle | 5.1 cm | 0.7° | 177 ms |
| multi-rés. | grille | 5.3 cm | 1.0° | 321 ms |
| multi-rés. | nuage | 5.3 cm | 1.0° | 228 ms |
| multi-rés. | vectorielle | 5.3 cm | 1.0° | 209 ms |
| particules | grille | 2.6 cm | 2.5° | 378 ms |
| particules | nuage | 1.3 cm | 1.7° | 705 ms |
| particules | vectorielle | 1.2 cm | 1.7° | 901 ms |
| direct | vectorielle | 2.2 cm | 1.1° | 79 ms |

À retenir : la multi-résolution réussit 100 % des tirages à tous les écarts ; le calcul direct est le plus rapide (~6 ms) mais décroche à ±0,50 m (71 %) ; l'exhaustif est le plus lent et dépasse 1 s en scan-contre-scan. La vérité de la section « carte » vient des poses réglées à la main dans `assemblage.json` : l'erreur mesurée inclut l'incertitude de ce réglage.

## Difficultés

**Reconstruction.** Certains essais en direct donnaient une carte illisible, corrigée par un suivi depuis la dernière pose. J'ai donc appliqué des traitements après l'enregistrement pour retraiter la carte. L'accumulation d'erreurs (sauts) sur les scans a fait perdre la localisation sur la deuxième partie de la carte.

**Mesure.** Les mesures en direct (SLAM) se sont réalisées le robot tenu en main. Le LiDAR n'a donc pas pu récupérer des données exploitables et a donc perdu environ 90° de couverture. Ce qui, pour certains virages aux coins de la salle, donnait des localisations totalement fausses par manque d'informations.


## Utilisation de l'IA

J'ai travaillé avec Claude Code pour l'écriture / test des différents scripts Python, l'aide à la compréhension et à la rédaction.

## Lancer

Depuis le dossier `Acquisition/` (dépendances : `pip install numpy scipy matplotlib pyserial`) :

python acquisition.py COM3 mon/dossier --tours 5
python visualisation.py ../Data/Scans
python scan_matching.py
python test_scan_matching.py
python tableau_erreurs.py
python temps_reel.py COM3
python temps_reel.py --replay ../Data/Cartes/salle/assemblage.json
python reconstruire_live.py ../Data/Cartes/Live_20261104_190647/scans

## Ce que je retiens du module

**Ce que j'ai apprécié** : la satisfaction de voir la salle apparaître progressivement, la création en temps réel des différentes carte et de la localisation continue du robot.

**Ce que j'améliorerais** : ajouter la fusion de capteur avec un IMU et Odométrie, la fermeture de boucle pour les grands parcours.

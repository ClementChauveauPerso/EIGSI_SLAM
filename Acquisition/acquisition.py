# acquisition.py — Moyenne plusieurs tours du LD06 (firmware ESP32_LD06_Test.ino)
# puis en affiche aussitôt le rendu visuel.
# Le firmware envoie chaque scan 360° en CSV, terminé par une ligne vide.
# Les points sont moyennés PAR ANGLE (jamais par index), par secteurs de 1°.
# Usage   : python acquisition.py [COMx] [sous/chemin] [--tours N]
# Exemple : python acquisition.py 1/2 --tours 5  ->  Data\Scans\1\2\{timestamp}.csv (+ .png)
# Dépendances : pip install pyserial matplotlib numpy

import math
import os
import subprocess
import sys
from datetime import datetime

import serial
from serial.tools import list_ports

BAUD = 115200  # cf. Serial.begin(115200) dans le .ino
OUT = r"C:\Users\Clement\Downloads\SLAM\Data\Scans"
HEADER = "N,Angle(°),Distance(mm),Intensity,x(mm),y(mm)"  # cf. printScanCSV()
VISU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualisation.py")
TOURS = 3    # nombre de tours complets moyennés par défaut
STEP = 1.0   # largeur du secteur angulaire de moyennage (°)


def read_scan(ser):
    """Retourne le premier scan complet (commençant au point 0), sinon None."""
    for _ in range(3):  # au plus 3 dumps : le premier peut être tronqué ou bruité
        lines = [l for l in iter(lambda: ser.readline().decode("utf-8", "ignore").strip(), "")
                 if not l.startswith("N,")]  # ignore l'en-tête (envoyé une fois au boot)
        if lines and lines[0].startswith("0,"):
            return lines
    return None


def average(scans):
    """Moyenne les scans par secteur d'angle (et non par index du laser)."""
    bins = {}
    for scan in scans:
        for line in scan:
            try:
                _, angle, dist, inten = line.split(",")[:4]
                angle, dist, inten = float(angle), float(dist), float(inten)
            except ValueError:  # ligne bruitée
                continue
            if dist <= 0:  # pas de retour : exclu de la moyenne
                continue
            b = bins.setdefault(int(angle // STEP), [0.0, 0.0, 0.0, 0])
            b[0] += angle
            b[1] += dist
            b[2] += inten
            b[3] += 1
    lines = []
    for n, key in enumerate(sorted(bins)):
        sum_angle, sum_dist, sum_inten, count = bins[key]
        angle, dist, inten = sum_angle / count, sum_dist / count, sum_inten / count
        rad = math.radians(angle)
        x = round(dist * math.cos(rad))
        y = round(-dist * math.sin(rad))  # convention de la librairie LD06
        lines.append(f"{n},{angle:.2f},{round(dist)},{round(inten)},{x},{y}")
    return lines


def visualize(path):
    """Rend le scan en PNG via visualisation.py --force, puis l'ouvre."""
    try:
        subprocess.run([sys.executable, VISU, path, "--force"], check=True)
        png = os.path.splitext(path)[0] + ".png"
        os.startfile(png)  # ouvre l'image dans la visionneuse par défaut
        print(f"  rendu visuel : {png}", flush=True)
    except Exception as e:  # l'acquisition reste valable même sans rendu
        print(f"  rendu visuel indisponible : {e}", flush=True)


def main():
    args = sys.argv[1:]
    tours = TOURS
    if "--tours" in args:
        i = args.index("--tours")
        try:
            tours = int(args[i + 1])
        except (IndexError, ValueError):
            sys.exit("Usage : --tours N (entier ≥ 1)")
        del args[i:i + 2]
    port = args.pop(0) if args and args[0].upper().startswith("COM") else None
    sub = args[0] if args else ""
    if not port:
        found = [p.device for p in list_ports.comports()]
        if len(found) == 1:
            port = found[0]
        else:
            sys.exit(f"Port série ambigu, précisez-le : python acquisition.py COMx [sous/chemin]. Détectés : {found or 'aucun'}")
    scans = []
    with serial.Serial(port, BAUD, timeout=5) as ser:
        print(f"Connecté sur {port} — {tours} tours complets à acquérir...", flush=True)
        while len(scans) < tours:
            scan = read_scan(ser)
            if scan is None:
                sys.exit("Flux série interrompu avant la fin de l'acquisition.")
            scans.append(scan)
            print(f"  tour {len(scans)}/{tours} : {len(scan)} points", flush=True)
    scan = average(scans)
    if not scan:
        sys.exit("Aucun point exploitable reçu.")
    path = os.path.join(OUT, sub, datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER + "\n" + "\n".join(scan) + "\n")
    print(f"{len(scan)} points moyennés sur {tours} tours : {path}", flush=True)
    visualize(path)


if __name__ == "__main__":
    main()

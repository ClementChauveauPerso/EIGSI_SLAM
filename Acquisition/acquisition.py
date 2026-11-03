# acquisition.py — Enregistre UN scan complet du LD06 (firmware ESP32_LD06_Test.ino)
# et en affiche aussitôt le rendu visuel.
# Le firmware envoie chaque scan 360° en CSV, terminé par une ligne vide.
# Usage   : python acquisition.py [COMx] [sous/chemin]
# Exemple : python acquisition.py 1/2  ->  Data\Scans\1\2\{timestamp}.csv (+ .png)
# Dépendances : pip install pyserial matplotlib numpy

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


def read_scan(ser):
    """Retourne le premier scan complet (commençant au point 0), sinon None."""
    for _ in range(3):  # au plus 3 dumps : le premier peut être tronqué ou bruité
        lines = [l for l in iter(lambda: ser.readline().decode("utf-8", "ignore").strip(), "")
                 if not l.startswith("N,")]  # ignore l'en-tête (envoyé une fois au boot)
        if lines and lines[0].startswith("0,"):
            return lines
    return None


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
    port = args.pop(0) if args and args[0].upper().startswith("COM") else None
    sub = args[0] if args else ""
    if not port:
        found = [p.device for p in list_ports.comports()]
        if len(found) == 1:
            port = found[0]
        else:
            sys.exit(f"Port série ambigu, précisez-le : python acquisition.py COMx [sous/chemin]. Détectés : {found or 'aucun'}")
    with serial.Serial(port, BAUD, timeout=5) as ser:
        print(f"Connecté sur {port} — attente d'un scan complet...", flush=True)
        scan = read_scan(ser)
    if not scan:
        sys.exit("Aucun scan complet reçu (vérifiez le port et le câblage).")
    path = os.path.join(OUT, sub, datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER + "\n" + "\n".join(scan) + "\n")
    print(f"{len(scan)} points enregistrés : {path}", flush=True)
    visualize(path)


if __name__ == "__main__":
    main()

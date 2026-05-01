"""
QW1 — Helper de calibration du masque imprimante.

Le masque definit la zone d'INCLUSION dans la ROI imprimante: les pixels HORS
polygone sont exclus de la detection (typiquement la zone ou la main du
caissier passe regulierement).

Usage:

  # Lire la valeur actuelle stockee en DB
  python -m scripts.calibrer_mask_imprimante --db /opt/fraude/data/fraude.db --show

  # Definir un polygone depuis une chaine JSON (coordonnees en % de la ROI, 0.0-1.0)
  python -m scripts.calibrer_mask_imprimante \\
      --db /opt/fraude/data/fraude.db \\
      --set '[[0.0,0.0],[1.0,0.0],[1.0,0.55],[0.0,0.55]]'

  # Idem depuis un fichier JSON
  python -m scripts.calibrer_mask_imprimante --db /opt/fraude/data/fraude.db --set-file mask.json

  # Effacer le polygone (revient a la ROI complete = comportement legacy)
  python -m scripts.calibrer_mask_imprimante --db /opt/fraude/data/fraude.db --clear

  # Apercu visuel: applique le polygone sur une ROI snapshot et sauvegarde une image overlay
  python -m scripts.calibrer_mask_imprimante --preview ROI.jpg \\
      --polygon '[[0.0,0.0],[1.0,0.0],[1.0,0.55],[0.0,0.55]]' \\
      --output ROI_preview.jpg

Conseil de calibration manuelle:
  1. Recuperer un fichier *_roi.jpg dans /opt/fraude/snapshots/imprimante_obs/
  2. L'ouvrir dans un visualiseur d'images qui affiche les coordonnees pixel
     au survol (Windows Photos, GIMP, IrfanView, ds9...).
  3. Identifier la zone du ticket (a inclure) vs la zone de la main du caissier
     (a exclure). Cliquer 4-6 sommets en suivant la frontiere.
  4. Convertir chaque (x_px, y_px) en (x_px/largeur_roi, y_px/hauteur_roi).
  5. Composer le JSON et le pousser via --set.
  6. Verifier avec --preview avant restart.

Apres modif: redemarrer le service pour prendre en compte (le worker
charge le polygone au demarrage de l'AnalyseurCaisse).
  ssh fraude "cd /opt/fraude && docker compose restart fraud-detector"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


CLE_PARAM = "imprimante_mask_polygon"


def _valider_polygone(data) -> list[tuple[float, float]]:
    if not isinstance(data, list) or len(data) < 3:
        raise ValueError("Le polygone doit etre une liste d'au moins 3 points")
    out: list[tuple[float, float]] = []
    for i, p in enumerate(data):
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ValueError(f"Point #{i} mal forme: attendu [x, y], recu {p!r}")
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Point #{i}: coordonnees non numeriques ({e})")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"Point #{i}=({x},{y}) hors plage [0,1]. "
                f"Les coordonnees sont en proportions de la ROI."
            )
        out.append((x, y))
    return out


def cmd_show(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT valeur FROM parametres WHERE cle=?", (CLE_PARAM,)
        ).fetchone()
    if row is None or not row[0]:
        print(f"[{CLE_PARAM}] vide (= ROI complete, comportement legacy)")
        return 0
    print(f"[{CLE_PARAM}] valeur actuelle:")
    print(row[0])
    try:
        data = json.loads(row[0])
        print(f"  -> {len(data)} points, valide={_test_valide(data)}")
    except json.JSONDecodeError:
        print("  -> ATTENTION: JSON invalide")
    return 0


def _test_valide(data) -> bool:
    try:
        _valider_polygone(data)
        return True
    except ValueError:
        return False


def cmd_set(db_path: Path, polygone_json: str) -> int:
    try:
        data = json.loads(polygone_json)
    except json.JSONDecodeError as e:
        print(f"ERREUR: JSON invalide: {e}", file=sys.stderr)
        return 2
    try:
        polygone = _valider_polygone(data)
    except ValueError as e:
        print(f"ERREUR: {e}", file=sys.stderr)
        return 2

    valeur = json.dumps([list(p) for p in polygone])
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """INSERT INTO parametres (cle, valeur, categorie, description, type_valeur, mis_a_jour)
               VALUES (?, ?, 'caisse',
                       'Polygone (JSON liste [x,y] en %% de la ROI, 0.0-1.0) excluant la zone main du caissier. Vide=ROI complete.',
                       'str', CURRENT_TIMESTAMP)
               ON CONFLICT(cle) DO UPDATE SET
                   valeur=excluded.valeur,
                   mis_a_jour=CURRENT_TIMESTAMP""",
            (CLE_PARAM, valeur),
        )
        conn.commit()
    print(f"[{CLE_PARAM}] mis a jour: {len(polygone)} points")
    print(f"  -> {valeur}")
    print("Restart fraud-detector pour appliquer:")
    print("  ssh fraude 'cd /opt/fraude && docker compose restart fraud-detector'")
    return 0


def cmd_clear(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE parametres SET valeur='', mis_a_jour=CURRENT_TIMESTAMP WHERE cle=?",
            (CLE_PARAM,),
        )
        conn.commit()
    print(f"[{CLE_PARAM}] efface (= ROI complete, fallback legacy)")
    return 0


def cmd_preview(image_path: Path, polygone_json: str, output_path: Path) -> int:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERREUR: cv2 et numpy requis pour --preview", file=sys.stderr)
        return 3
    try:
        polygone = _valider_polygone(json.loads(polygone_json))
    except ValueError as e:
        print(f"ERREUR polygone: {e}", file=sys.stderr)
        return 2

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERREUR: impossible de lire {image_path}", file=sys.stderr)
        return 4

    h, w = img.shape[:2]
    overlay = img.copy()
    pts = np.array(
        [(int(round(x * (w - 1))), int(round(y * (h - 1)))) for x, y in polygone],
        dtype=np.int32,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype("uint8")
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (px, py) in enumerate(pts):
        cv2.circle(overlay, (int(px), int(py)), 4, (0, 0, 255), -1)
        cv2.putText(overlay, str(i), (int(px) + 6, int(py)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ratio = float(mask.mean()) / 255.0
    print(f"Preview ecrit: {output_path}")
    print(f"  ROI {w}x{h}, {ratio:.0%} pixels inclus dans la detection")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="QW1 — Calibration mask imprimante")
    parser.add_argument("--db", type=Path, help="Chemin de la DB SQLite (defaut /opt/fraude/data/fraude.db)",
                        default=Path("/opt/fraude/data/fraude.db"))
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--show", action="store_true", help="Afficher la valeur actuelle")
    grp.add_argument("--set", dest="set_value", metavar="JSON",
                     help="Definir le polygone (chaine JSON)")
    grp.add_argument("--set-file", dest="set_file", type=Path,
                     help="Definir le polygone (fichier JSON)")
    grp.add_argument("--clear", action="store_true", help="Effacer (revient a ROI complete)")
    grp.add_argument("--preview", type=Path, metavar="IMAGE",
                     help="Sauvegarder un apercu visuel sur une image ROI")
    parser.add_argument("--polygon", help="Pour --preview: polygone JSON")
    parser.add_argument("--output", type=Path, default=Path("preview_mask.jpg"),
                        help="Pour --preview: chemin de sortie (defaut preview_mask.jpg)")
    args = parser.parse_args()

    if args.show:
        return cmd_show(args.db)
    if args.set_value:
        return cmd_set(args.db, args.set_value)
    if args.set_file:
        return cmd_set(args.db, args.set_file.read_text())
    if args.clear:
        return cmd_clear(args.db)
    if args.preview:
        if not args.polygon:
            print("ERREUR: --preview requiert --polygon", file=sys.stderr)
            return 2
        return cmd_preview(args.preview, args.polygon, args.output)
    return 1


if __name__ == "__main__":
    sys.exit(main())

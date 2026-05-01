"""
Script de validation des QW1/QW2 sur snapshots imprimante existants.

Iterate sur les fichiers *_roi.jpg generes par l'OBS imprimante et compare
les ratios de detection avec/sans QW1 (mask) et avec QW2 (HSV) vs gray.

Note: ce script ne peut pas comparer le QW3 (cooldown temporel) puisque les
snapshots sont des images statiques. Il sert uniquement a verifier l'impact
des QW1/QW2 sur la classification 'pixel papier' frame-par-frame.

Usage:
  # Sur le magasin (dans le container)
  docker compose exec fraud-detector python -m scripts.valider_detection_imprimante \\
      /opt/fraude/snapshots/imprimante_obs/

  # Avec un polygone de masque a evaluer
  ... --mask-json '[[0.0,0.0],[1.0,0.0],[1.0,0.55],[0.0,0.55]]'

  # Sortie CSV au lieu de l'affichage
  ... --csv resultats.csv

L'output (par image):
  - ratio_papier_gray: ratio avec mode legacy (gris > 200)
  - ratio_papier_hsv:  ratio avec mode HSV (S < 40 ET V > 180)
  - ratio_papier_hsv_masked: idem HSV mais avec polygone applique (si --mask-json)
  - delta_hsv_vs_gray: % point d'ecart (negatif = HSV plus strict)
  - delta_mask: % point d'ecart entre HSV-masked et HSV (negatif = mask exclut)

Permet de detecter les snapshots ou:
  - HSV serait passe sous le seuil alors que gray etait au-dessus (vrai
    positif probable pour QW2 si l'image est un faux positif gray)
  - Le mask reduit fortement le ratio (vrai positif probable pour QW1 si
    la zone main couvrait les pixels suspects)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _calculer_masque_blanc_gray(roi_bgr: np.ndarray, seuil: int = 200) -> np.ndarray:
    import cv2
    roi_gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return roi_gris > seuil


def _calculer_masque_blanc_hsv(
    roi_bgr: np.ndarray, seuil_sat: int = 40, seuil_val: int = 180
) -> np.ndarray:
    import cv2
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = roi_hsv[..., 1]
    val = roi_hsv[..., 2]
    return (sat < seuil_sat) & (val > seuil_val)


def _calculer_polygone_mask(
    roi_shape: Tuple[int, int],
    polygone: List[Tuple[float, float]],
) -> np.ndarray:
    import cv2
    h, w = roi_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(
        [
            (int(round(max(0.0, min(1.0, x)) * (w - 1))),
             int(round(max(0.0, min(1.0, y)) * (h - 1))))
            for x, y in polygone
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _analyser_image(
    path: Path,
    polygone: Optional[List[Tuple[float, float]]],
    seuil_sat: int,
    seuil_val: int,
    seuil_gray: int,
) -> Optional[dict]:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None

    h, w = img.shape[:2]
    nb_pixels_total = h * w

    masque_gray = _calculer_masque_blanc_gray(img, seuil_gray)
    masque_hsv = _calculer_masque_blanc_hsv(img, seuil_sat, seuil_val)

    ratio_gray = float(np.count_nonzero(masque_gray) / nb_pixels_total)
    ratio_hsv = float(np.count_nonzero(masque_hsv) / nb_pixels_total)

    ratio_hsv_masked = None
    nb_pixels_inclus = nb_pixels_total
    if polygone is not None:
        mask_inclus = _calculer_polygone_mask((h, w), polygone)
        nb_pixels_inclus = int(np.count_nonzero(mask_inclus))
        if nb_pixels_inclus > 0:
            masque_hsv_in = masque_hsv & mask_inclus
            ratio_hsv_masked = float(
                np.count_nonzero(masque_hsv_in) / nb_pixels_inclus
            )

    return {
        "fichier": path.name,
        "shape": f"{w}x{h}",
        "ratio_blanc_gray": round(ratio_gray, 4),
        "ratio_blanc_hsv": round(ratio_hsv, 4),
        "ratio_blanc_hsv_masked": (
            round(ratio_hsv_masked, 4) if ratio_hsv_masked is not None else None
        ),
        "delta_hsv_vs_gray_pp": round((ratio_hsv - ratio_gray) * 100, 2),
        "delta_mask_pp": (
            round((ratio_hsv_masked - ratio_hsv) * 100, 2)
            if ratio_hsv_masked is not None else None
        ),
        "pct_pixels_inclus_mask": (
            round(nb_pixels_inclus / nb_pixels_total * 100, 1)
            if polygone is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validation QW1/QW2 sur snapshots imprimante existants"
    )
    parser.add_argument(
        "directory", type=Path,
        help="Repertoire contenant les snapshots *_roi.jpg "
             "(ex: /opt/fraude/snapshots/imprimante_obs/)"
    )
    parser.add_argument(
        "--mask-json", help="Polygone JSON a tester (active QW1 dans le rapport)"
    )
    parser.add_argument("--seuil-sat", type=int, default=40, help="QW2 seuil S MAX")
    parser.add_argument("--seuil-val", type=int, default=180, help="QW2 seuil V MIN")
    parser.add_argument("--seuil-gray", type=int, default=200, help="Legacy seuil gris")
    parser.add_argument(
        "--seuil-changement", type=float, default=0.15,
        help="Seuil ratio_papier au-dessus duquel detection brute (informatif)"
    )
    parser.add_argument("--csv", type=Path, help="Sauvegarder en CSV au lieu d'afficher")
    parser.add_argument("--limit", type=int, default=0, help="Limiter a N fichiers")
    parser.add_argument(
        "--pattern", default="*_roi.jpg",
        help="Pattern glob a chercher (defaut *_roi.jpg, ou *.jpg pour images full)"
    )
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"ERREUR: {args.directory} introuvable", file=sys.stderr)
        return 2

    polygone: Optional[List[Tuple[float, float]]] = None
    if args.mask_json:
        try:
            data = json.loads(args.mask_json)
            polygone = [(float(p[0]), float(p[1])) for p in data]
            if len(polygone) < 3:
                raise ValueError("polygone < 3 points")
        except (json.JSONDecodeError, TypeError, ValueError, IndexError) as e:
            print(f"ERREUR: --mask-json invalide ({e})", file=sys.stderr)
            return 2

    fichiers = sorted(args.directory.rglob(args.pattern))
    if args.limit > 0:
        fichiers = fichiers[: args.limit]

    if not fichiers:
        print(f"Aucun fichier {args.pattern} trouve sous {args.directory}",
              file=sys.stderr)
        return 1

    resultats = []
    for f in fichiers:
        r = _analyser_image(
            f, polygone, args.seuil_sat, args.seuil_val, args.seuil_gray,
        )
        if r is not None:
            resultats.append(r)

    if not resultats:
        print("Aucune image lisible", file=sys.stderr)
        return 1

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(resultats[0].keys()))
            writer.writeheader()
            writer.writerows(resultats)
        print(f"CSV ecrit: {args.csv} ({len(resultats)} lignes)")
    else:
        for r in resultats:
            print(
                f"{r['fichier']:50s} {r['shape']:>9s}  "
                f"gray={r['ratio_blanc_gray']:.3f}  "
                f"hsv={r['ratio_blanc_hsv']:.3f}  "
                f"d(hsv-gray)={r['delta_hsv_vs_gray_pp']:+.1f}pp"
                + (
                    f"  hsv_mask={r['ratio_blanc_hsv_masked']:.3f}  "
                    f"d(mask)={r['delta_mask_pp']:+.1f}pp"
                    if r["ratio_blanc_hsv_masked"] is not None else ""
                )
            )

        ratios_gray = [r["ratio_blanc_gray"] for r in resultats]
        ratios_hsv = [r["ratio_blanc_hsv"] for r in resultats]

        print()
        print(f"=== STATS sur {len(resultats)} images ===")
        print(f"ratio_blanc_gray : moy={np.mean(ratios_gray):.3f}  "
              f"med={np.median(ratios_gray):.3f}  "
              f"max={np.max(ratios_gray):.3f}")
        print(f"ratio_blanc_hsv  : moy={np.mean(ratios_hsv):.3f}  "
              f"med={np.median(ratios_hsv):.3f}  "
              f"max={np.max(ratios_hsv):.3f}")

        seuil = args.seuil_changement
        ng = sum(1 for r in ratios_gray if r > seuil)
        nh = sum(1 for r in ratios_hsv if r > seuil)
        print(f"Au-dessus du seuil_changement {seuil:.2f}:")
        print(f"  gray: {ng}/{len(resultats)} ({ng / len(resultats):.0%})")
        print(f"  hsv : {nh}/{len(resultats)} ({nh / len(resultats):.0%})")
        print(f"  ecart QW2: {ng - nh:+d} (negatif = HSV plus strict)")

        if polygone is not None:
            ratios_hsv_m = [
                r["ratio_blanc_hsv_masked"] for r in resultats
                if r["ratio_blanc_hsv_masked"] is not None
            ]
            if ratios_hsv_m:
                nm = sum(1 for r in ratios_hsv_m if r > seuil)
                print(f"  hsv_mask (QW1+QW2): {nm}/{len(ratios_hsv_m)} "
                      f"({nm / len(ratios_hsv_m):.0%})")
                print(f"  ecart QW1: {nh - nm:+d} (negatif = mask plus strict)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

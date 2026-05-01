"""
Recalibration de la reference drift imprimante.

Supprime le fichier reference.jpg et demande au worker de le recapturer au
prochain check (apres restart). A utiliser quand l'utilisateur a physiquement
remis l'imprimante en place et veut redefinir la nouvelle position comme
reference baseline.

Usage:

  # Local (script direct, non-mounte dans container) — pour info:
  python -m scripts.recalibrer_drift_imprimante --dir /opt/fraude/snapshots/imprimante_drift

  # En production sur pos-pc:
  rm /opt/fraude/snapshots/imprimante_drift/reference.jpg \\
    && cd /opt/fraude && docker compose restart fraud-detector

Le code a la meme logique: supprime reference.jpg, force le worker a
re-capturer au prochain analyser() call.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("/opt/fraude/snapshots/imprimante_drift"),
        help="Repertoire ou se trouve reference.jpg",
    )
    parser.add_argument(
        "--keep-snapshots",
        action="store_true",
        help="Ne pas supprimer les snapshots drift_*.jpg (conserve l'historique).",
    )
    args = parser.parse_args()

    ref_path = args.dir / "reference.jpg"
    if not ref_path.exists():
        print(f"[recalibrer] {ref_path} n'existe pas — rien a supprimer.")
        print("[recalibrer] La ref sera capturee au prochain restart du worker.")
        return 0

    try:
        ref_path.unlink()
        print(f"[recalibrer] supprime: {ref_path}")
    except OSError as e:
        print(f"[recalibrer] erreur suppression: {e}", file=sys.stderr)
        return 1

    if not args.keep_snapshots:
        # Supprime aussi les snapshots de comparaison historiques
        nb = 0
        for pattern in ("drift_*_current.jpg", "drift_*_ref.jpg", "drift_*_compare.jpg"):
            for f in args.dir.glob(pattern):
                try:
                    f.unlink()
                    nb += 1
                except OSError:
                    pass
        if nb > 0:
            print(f"[recalibrer] supprime {nb} snapshots historiques (--keep-snapshots pour garder)")

    print()
    print("Etape suivante: redemarrer le service fraud-detector pour que la")
    print("nouvelle reference soit capturee depuis la frame courante:")
    print("  cd /opt/fraude && docker compose restart fraud-detector")
    return 0


if __name__ == "__main__":
    sys.exit(main())

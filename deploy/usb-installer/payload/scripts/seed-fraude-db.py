"""
Seed initial pour Fraude : cree une fraude.db avec :
  - Schema complet (via app/database.py)
  - Parametres par defaut (initialiser_parametres_defaut)
  - 2 cameras du magasin :
      * Caisse (active=1, mode=caisse) - surveillance caisse principale
      * CAM5   (active=0, mode=vol)    - rayon, a activer apres validation reseau

Usage (depuis le repo, dans un container python avec deps installees) :
    python deploy/usb-installer/payload/scripts/seed-fraude-db.py \\
        --output deploy/usb-installer/payload/data/fraude.db

Le user d'install peut ensuite editer les URLs RTSP via le dashboard.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Genere une fraude.db pre-seedee")
    parser.add_argument(
        "--output", required=True,
        help="Chemin de sortie pour la DB (sera ecrasee si elle existe)",
    )
    parser.add_argument(
        "--repo-root", default=".",
        help="Racine du repo Fraude (pour importer app.database)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    output_db = Path(args.output).resolve()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
        for ext in (".db-wal", ".db-shm"):
            sib = output_db.with_suffix(output_db.suffix + ext.lstrip(".db"))
            if sib.exists():
                sib.unlink()

    # On force la DB cible avant l'import (config.py lit DATABASE_PATH)
    import os
    os.environ["DATABASE_PATH"] = str(output_db)

    from app.database import BaseDonneesFraude

    # Le constructeur initialise schema + parametres par defaut automatiquement
    db = BaseDonneesFraude(chemin_db=output_db)

    # ----- Cameras du magasin (mapping reel du dev) -----
    # 7 cameras configurees, 2 actives par defaut (Caisse + CAM5).
    # Les autres restent inactives : a activer une par une depuis le dashboard
    # selon la capacite CPU de la machine cible.
    cameras = [
        {"nom": "Caisse", "source": "rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/301",
         "zone": "caisse",        "niveau": "Niveau 0",
         "position_description": "Camera principale au-dessus du comptoir caisse",
         "mode_detection": "caisse", "active": True},

        {"nom": "CAM5",   "source": "rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/501",
         "zone": "inconnue",      "niveau": "Niveau 0",
         "position_description": "Camera rayon - active par defaut",
         "mode_detection": "vol",    "active": True},

        {"nom": "CAM1",   "source": "rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/101",
         "zone": "inconnue",      "niveau": "Niveau 0",
         "position_description": "Camera entree - inactive (a activer si CPU le permet)",
         "mode_detection": "vol",    "active": False},

        {"nom": "CAM2",   "source": "rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/201",
         "zone": "inconnue",      "niveau": "Niveau 0",
         "position_description": "Camera rayon - inactive",
         "mode_detection": "vol",    "active": False},

        {"nom": "CAM4",   "source": "rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/401",
         "zone": "allee_droite",  "niveau": "Niveau 0",
         "position_description": "Camera allee droite - inactive",
         "mode_detection": "vol",    "active": False},

        {"nom": "CAM6",   "source": "rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/601",
         "zone": "inconnue",      "niveau": "Niveau 0",
         "position_description": "Camera rayon - inactive",
         "mode_detection": "vol",    "active": False},

        {"nom": "CAM8",   "source": "rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/801",
         "zone": "inconnue",      "niveau": "Niveau 0",
         "position_description": "Camera rayon - inactive",
         "mode_detection": "vol",    "active": False},
    ]

    for cam in cameras:
        cam_id = db.ajouter_camera(
            nom=cam["nom"],
            source=cam["source"],
            zone=cam["zone"],
            niveau=cam["niveau"],
            position_description=cam["position_description"],
            mode_detection=cam["mode_detection"],
        )
        if not cam["active"]:
            db.modifier_camera(cam_id, active=False)

    # Verification
    rows = db.obtenir_cameras()
    print(f"[OK] DB seedee : {output_db}")
    print(f"[OK] {len(rows)} cameras inserees :")
    for r in rows:
        flag = "ACTIVE" if r["active"] else "inactive"
        print(f"      - {r['nom']:10s} [{flag:8s}] mode={r['mode_detection']:7s} zone={r['zone']}")

    # Force checkpoint WAL pour produire un .db autonome (sans -wal/-shm)
    import sqlite3
    with sqlite3.connect(str(output_db)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for ext in ("-wal", "-shm"):
        sib = output_db.with_name(output_db.name + ext)
        if sib.exists():
            sib.unlink()


if __name__ == "__main__":
    main()

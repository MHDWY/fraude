"""
Script de telechargement des modeles YOLO necessaires.
Telecharge les modeles ONNX pour la detection et l'estimation de pose.

Usage:
    python scripts/download_models.py
"""

import os
import sys
import shutil
from pathlib import Path

# Repertoire de destination
REPERTOIRE_MODELES = Path(__file__).parent.parent / "models"

# Modeles a telecharger avec URL directe en fallback
MODELES = {
    "yolov8n.onnx": {
        "description": "YOLOv8 Nano - Detection personnes et objets",
        "taille_attendue_mo": 6.0,
        "pt_name": "yolov8n.pt",
    },
    "yolov8n-pose.onnx": {
        "description": "YOLOv8 Nano Pose - Estimation de pose (17 keypoints)",
        "taille_attendue_mo": 7.0,
        "pt_name": "yolov8n-pose.pt",
    },
}


def afficher_progression(actuel: int, total: int, largeur: int = 50):
    """Affiche une barre de progression dans la console."""
    pourcent = actuel / max(total, 1)
    rempli = int(largeur * pourcent)
    barre = "#" * rempli + "-" * (largeur - rempli)
    taille_mo = actuel / (1024 * 1024)
    total_mo = total / (1024 * 1024)
    sys.stdout.write(f"\r  [{barre}] {pourcent:.0%} ({taille_mo:.1f}/{total_mo:.1f} Mo)")
    sys.stdout.flush()


def verifier_modele(chemin: Path, taille_min_mo: float) -> bool:
    """Verifie qu'un modele existe et a une taille raisonnable."""
    if not chemin.exists():
        return False
    taille_mo = chemin.stat().st_size / (1024 * 1024)
    if taille_mo < taille_min_mo * 0.3:
        print(f"  ATTENTION: {chemin.name} semble trop petit ({taille_mo:.1f} Mo)")
        return False
    return True


def telecharger_avec_ultralytics(pt_name: str, chemin_sortie: Path) -> bool:
    """Telecharge et exporte un modele via Ultralytics."""
    try:
        from ultralytics import YOLO

        nom_base = chemin_sortie.stem
        print(f"  Chargement du modele {pt_name}...")
        modele = YOLO(pt_name)

        print(f"  Export en ONNX (sans simplification)...")
        chemin_exporte = modele.export(format="onnx", imgsz=640)

        if chemin_exporte and Path(chemin_exporte).exists():
            chemin_exp = Path(chemin_exporte)
            if chemin_exp != chemin_sortie:
                shutil.move(str(chemin_exp), str(chemin_sortie))

            # Nettoyer le .pt
            fichier_pt = Path(pt_name)
            if fichier_pt.exists():
                fichier_pt.unlink()
            # Nettoyer aussi dans le cwd
            for p in Path(".").glob(f"{nom_base}*.pt"):
                p.unlink()

            return True
        return False

    except Exception as e:
        print(f"  Erreur Ultralytics: {e}")
        return False


def telecharger_direct(url: str, chemin_sortie: Path) -> bool:
    """Telecharge un fichier via HTTP avec barre de progression."""
    try:
        import requests
        print(f"  Telechargement depuis: {url}")
        response = requests.get(url, stream=True, timeout=120, allow_redirects=True)
        response.raise_for_status()

        taille_totale = int(response.headers.get("content-length", 0))
        taille_dl = 0

        with open(chemin_sortie, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                taille_dl += len(chunk)
                if taille_totale > 0:
                    afficher_progression(taille_dl, taille_totale)
        print()
        return True

    except Exception as e:
        print(f"\n  Erreur telechargement: {e}")
        if chemin_sortie.exists():
            chemin_sortie.unlink()
        return False


def main():
    print("\n" + "=" * 55)
    print("  TELECHARGEMENT DES MODELES YOLO")
    print("  Detection de fraude en magasin")
    print("=" * 55)

    REPERTOIRE_MODELES.mkdir(parents=True, exist_ok=True)
    print(f"\nRepertoire: {REPERTOIRE_MODELES}")

    resultats = {}

    for nom_fichier, info in MODELES.items():
        chemin_sortie = REPERTOIRE_MODELES / nom_fichier
        print(f"\n--- {info['description']} ---")

        # Verifier si deja present
        if verifier_modele(chemin_sortie, info["taille_attendue_mo"]):
            taille = chemin_sortie.stat().st_size / (1024 * 1024)
            print(f"  Deja present: {nom_fichier} ({taille:.1f} Mo)")
            resultats[nom_fichier] = True
            continue

        succes = False

        # Methode 1: Export via Ultralytics
        print(f"  Methode 1: Export via Ultralytics...")
        try:
            succes = telecharger_avec_ultralytics(info["pt_name"], chemin_sortie)
            if succes:
                succes = verifier_modele(chemin_sortie, info["taille_attendue_mo"])
        except Exception as e:
            print(f"  Echec methode 1: {e}")

        if succes:
            taille = chemin_sortie.stat().st_size / (1024 * 1024)
            print(f"  [OK] {nom_fichier} ({taille:.1f} Mo)")
            resultats[nom_fichier] = True
            continue

        # Methode 2: Telecharger le .pt et exporter manuellement
        print(f"  Methode 2: Telechargement .pt + export...")
        try:
            import requests
            pt_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{info['pt_name']}"
            pt_path = REPERTOIRE_MODELES / info["pt_name"]

            if telecharger_direct(pt_url, pt_path):
                from ultralytics import YOLO
                modele = YOLO(str(pt_path))
                chemin_exp = modele.export(format="onnx", imgsz=640)
                if chemin_exp and Path(chemin_exp).exists():
                    if Path(chemin_exp) != chemin_sortie:
                        shutil.move(str(chemin_exp), str(chemin_sortie))
                    succes = verifier_modele(chemin_sortie, info["taille_attendue_mo"])
                # Nettoyer le .pt
                if pt_path.exists():
                    pt_path.unlink()
        except Exception as e:
            print(f"  Echec methode 2: {e}")

        if succes:
            taille = chemin_sortie.stat().st_size / (1024 * 1024)
            print(f"  [OK] {nom_fichier} ({taille:.1f} Mo)")
            resultats[nom_fichier] = True
        else:
            print(f"  ECHEC: {nom_fichier}")
            print(f"\n  === TELECHARGEMENT MANUEL ===")
            print(f"  1. Telechargez {info['pt_name']} depuis:")
            print(f"     https://github.com/ultralytics/assets/releases")
            print(f"  2. Executez: yolo export model={info['pt_name']} format=onnx")
            print(f"  3. Placez {nom_fichier} dans: {REPERTOIRE_MODELES}")
            resultats[nom_fichier] = False

    # Resume
    print("\n" + "=" * 55)
    print("  RESUME")
    print("=" * 55)

    tout_ok = True
    for nom, ok in resultats.items():
        statut = "[OK]" if ok else "[MANQUANT]"
        print(f"  {statut} {nom}")
        if not ok:
            tout_ok = False

    if tout_ok:
        print("\n  Tous les modeles sont prets !")
        print("  Lancez: python -m app.main --test-webcam")
    else:
        print("\n  Certains modeles sont manquants.")
        print("  Le systeme fonctionnera en mode degrade.")

    print()
    return 0 if tout_ok else 1


if __name__ == "__main__":
    sys.exit(main())

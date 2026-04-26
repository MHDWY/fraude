"""
Script de test rapide avec la webcam du PC.
Lance le système de détection de fraude en mode test.

Usage:
    python test_webcam.py              # Webcam par défaut (index 0)
    python test_webcam.py --source 1   # Webcam index 1
    python test_webcam.py --source video.mp4  # Fichier vidéo

Contrôles:
    q : Quitter
    s : Sauvegarder une capture d'écran
    p : Pause / Reprendre
    d : Afficher/masquer les détections
    h : Afficher l'aide
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import obtenir_config


def afficher_aide():
    """Affiche les instructions dans la console."""
    print("\n" + "=" * 50)
    print("  DETECTION DE FRAUDE - MODE TEST WEBCAM")
    print("=" * 50)
    print()
    print("  Controles clavier:")
    print("    q  - Quitter")
    print("    s  - Sauvegarder une capture")
    print("    p  - Pause / Reprendre")
    print("    d  - Afficher/masquer detections")
    print("    h  - Afficher cette aide")
    print()
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Test de detection de fraude avec webcam",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Source video: index webcam (0, 1...) ou chemin fichier",
    )
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Desactiver la detection (afficher uniquement la video)",
    )
    parser.add_argument(
        "--resolution",
        default="640x480",
        help="Resolution de la webcam (ex: 640x480, 1280x720)",
    )

    args = parser.parse_args()
    afficher_aide()

    # Ouvrir la source vidéo
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print(f"\nOuverture de la source: {source}")

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"\nERREUR: Impossible d'ouvrir la source '{source}'")
        print("Verifiez que votre webcam est connectee.")
        print("Essayez: python test_webcam.py --source 0")
        sys.exit(1)

    # Appliquer la résolution
    try:
        w, h = args.resolution.split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    except ValueError:
        pass

    # Charger les modèles si la détection est activée
    detecteur = None
    estimateur_pose = None
    tracker = None
    analyseur = None

    if not args.no_detection:
        print("\nChargement des modeles...")
        config = obtenir_config()

        try:
            from app.detector import DetecteurPersonnes, EstimateurPose
            from app.tracker import ByteTracker
            from app.behavior_analyzer import AnalyseurComportements

            if config.chemin_modele_yolo.exists():
                detecteur = DetecteurPersonnes(
                    config.chemin_modele_yolo,
                    confiance_min=config.yolo_confidence,
                )
                print("  [OK] Detecteur YOLO charge")
            else:
                print(f"  [!!] Modele YOLO non trouve: {config.chemin_modele_yolo}")
                print("       Executez: python scripts/download_models.py")

            if config.chemin_modele_pose.exists():
                estimateur_pose = EstimateurPose(
                    config.chemin_modele_pose,
                    confiance_min=config.pose_confidence,
                )
                print("  [OK] Estimateur de pose charge")
            else:
                print(f"  [!!] Modele pose non trouve: {config.chemin_modele_pose}")

            tracker = ByteTracker()
            analyseur = AnalyseurComportements(seuil_alerte=config.behavior_threshold)
            print("  [OK] Tracker et analyseur initialises")

        except Exception as e:
            print(f"  [!!] Erreur chargement: {e}")
            print("       Mode video simple active")

    print(f"\nWebcam ouverte: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Appuyez sur 'q' pour quitter\n")

    # Boucle principale
    en_pause = False
    afficher_detections = True
    nb_frames = 0
    temps_debut = time.time()
    fps = 0.0

    while True:
        if not en_pause:
            ret, frame = cap.read()
            if not ret:
                print("Fin de la source video")
                break

            nb_frames += 1
            elapsed = time.time() - temps_debut
            if elapsed > 0:
                fps = nb_frames / elapsed

            # Traitement si détection activée
            if detecteur and afficher_detections:
                # Détection des personnes
                detections = detecteur.detecter_personnes(frame)
                objets = detecteur.detecter_objets(frame)

                # Suivi
                if tracker:
                    dets_tracker = [(d.bbox, d.confidence) for d in detections]
                    pistes = tracker.mettre_a_jour(dets_tracker)

                    # Dessiner les pistes
                    for piste in pistes:
                        x1, y1, x2, y2 = piste.bbox
                        score_susp = 0.0

                        # Analyse comportementale
                        if analyseur and estimateur_pose:
                            pose = estimateur_pose.estimer_pose(frame, piste.bbox)
                            resultats = analyseur.analyser(
                                piste, pose, objets, frame.shape[:2]
                            )
                            score_susp = analyseur.obtenir_score_suspicion(piste.id_piste).score_global

                            # Afficher les alertes
                            for r in resultats:
                                print(
                                    f"  [ALERTE] {r.description} "
                                    f"(piste #{r.id_piste}, confiance: {r.confiance:.0%})"
                                )
                        elif analyseur:
                            resultats = analyseur.analyser(
                                piste, None, objets, frame.shape[:2]
                            )
                            score_susp = analyseur.obtenir_score_suspicion(piste.id_piste).score_global

                        # Couleur selon suspicion
                        if score_susp >= 0.6:
                            couleur = (0, 0, 255)
                        elif score_susp >= 0.3:
                            couleur = (0, 165, 255)
                        else:
                            couleur = (0, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
                        label = f"#{piste.id_piste} [{score_susp:.0%}]"
                        cv2.putText(
                            frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 1, cv2.LINE_AA,
                        )

                        # Trajectoire
                        centres = list(piste.historique_centres)[-20:]
                        for i in range(1, len(centres)):
                            pt1 = (int(centres[i-1][0]), int(centres[i-1][1]))
                            pt2 = (int(centres[i][0]), int(centres[i][1]))
                            cv2.line(frame, pt1, pt2, couleur, 1)

                # Dessiner les objets
                for obj in objets:
                    x1, y1, x2, y2 = obj.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
                    cv2.putText(
                        frame, f"{obj.class_name} {obj.confidence:.0%}",
                        (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1,
                    )

            # Barre d'info
            h_frame, w_frame = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w_frame, 30), (50, 50, 50), -1)
            info_texte = f"FPS: {fps:.1f} | Frame: {nb_frames}"
            if detecteur:
                info_texte += " | Detection: ON" if afficher_detections else " | Detection: OFF"
            if en_pause:
                info_texte += " | PAUSE"
            cv2.putText(
                frame, info_texte, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Afficher la frame
            cv2.imshow("Test Webcam - Detection Fraude", frame)

        # Gestion des touches
        touche = cv2.waitKey(1) & 0xFF

        if touche == ord("q"):
            print("\nArret demande")
            break
        elif touche == ord("s"):
            chemin = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(chemin, frame)
            print(f"  Capture sauvegardee: {chemin}")
        elif touche == ord("p"):
            en_pause = not en_pause
            print("  PAUSE" if en_pause else "  REPRISE")
        elif touche == ord("d"):
            afficher_detections = not afficher_detections
            print(f"  Detections: {'ON' if afficher_detections else 'OFF'}")
        elif touche == ord("h"):
            afficher_aide()

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()

    duree = time.time() - temps_debut
    print(f"\nSession terminee: {nb_frames} frames en {duree:.0f}s ({fps:.1f} FPS moyen)")


if __name__ == "__main__":
    main()

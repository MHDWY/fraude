"""
Point d'entree principal du systeme de detection de fraude.
Supporte le mode mono-source (legacy) et multi-camera.

Mode multi-camera:
    - Charge les cameras actives depuis la base de donnees
    - Cree un CameraWorker par camera (thread dedie)
    - Partage les modeles YOLO et la DB entre les workers
    - Superviseur dans le thread principal (health monitoring)

Usage:
    python -m app.main                    # Mode multi-camera (sources DB)
    python -m app.main --test-webcam      # Mode test avec webcam (mono)
    python -m app.main --source chemin    # Source video specifique (mono)
    python -m app.main --dashboard-only   # Lancer uniquement le dashboard
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .camera_worker import CameraWorker
from .config import FraudeConfig, obtenir_config
from .database import BaseDonneesFraude
from .detector import DetecteurPersonnes, EstimateurPose, Detection, DetecteurApprentissage
from .tracker import ByteTracker, HistoriqueTrajectoires
from .behavior_analyzer import AnalyseurComportements, ResultatAnalyse, TypeComportement
from .caisse_analyzer import AnalyseurCaisse
from .alert_manager import GestionnaireAlertes
from .video_recorder import EnregistreurVideo

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraude.main")

# Drapeau d'arret propre
_arreter = False


def gestionnaire_signal(sig, frame):
    """Gestion propre de l'arret (Ctrl+C)."""
    global _arreter
    logger.info("Signal d'arret recu, fermeture en cours...")
    _arreter = True


signal.signal(signal.SIGINT, gestionnaire_signal)
signal.signal(signal.SIGTERM, gestionnaire_signal)


class PipelineFraude:
    """
    Pipeline principal de détection de fraude.
    Orchestre tous les composants: détection, suivi, analyse, alertes.
    """

    def __init__(self, config: FraudeConfig, source: Optional[str] = None):
        """
        Initialise le pipeline avec tous les composants.

        Args:
            config: Configuration du système
            source: Source vidéo (override la config si spécifié)
        """
        self.config = config
        self.source = source

        # Initialiser la base de données
        self.db = BaseDonneesFraude(config.chemin_base_donnees)

        # Initialiser l'enregistreur vidéo + snapshots
        self.enregistreur = EnregistreurVideo(
            repertoire_sortie=config.chemin_enregistrements,
            repertoire_snapshots=config.chemin_snapshots,
            duree_clip=config.video_clip_duration,
            fps=15,
            pre_evenement_secondes=5,
            retention_jours=config.retention_jours,
            retention_videos_jours=config.retention_videos_jours,
            retention_snapshots_jours=config.retention_snapshots_jours,
            quota_stockage_max_gb=config.quota_stockage_max_gb,
            quota_seuil_alerte_pct=config.quota_seuil_alerte_pct,
        )

        # Initialiser le détecteur (YOLO ONNX)
        self.detecteur = None
        self.estimateur_pose = None
        self._charger_modeles()

        # Initialiser le tracker ByteTrack
        self.tracker = ByteTracker(
            seuil_score_haut=config.yolo_confidence,
            seuil_score_bas=0.1,
            max_frames_perdues=30,
        )
        self.historique = HistoriqueTrajectoires(duree_max_secondes=300)

        # Initialiser l'analyseur de comportements
        self.analyseur = AnalyseurComportements(
            seuil_alerte=config.behavior_threshold,
            cooldown_secondes=config.alert_cooldown_seconds,
        )

        # Initialiser l'analyseur de caisse
        self.analyseur_caisse = AnalyseurCaisse(
            timeout_ticket_secondes=float(
                self.db.obtenir_parametre("caisse_timeout_ticket", 12.0)
            ),
            zone_caisse_y_min_pct=float(
                self.db.obtenir_parametre("caisse_zone_y_min_pct", 0.70)
            ),
            zone_caisse_x_min_pct=float(
                self.db.obtenir_parametre("caisse_zone_x_min_pct", 0.25)
            ),
            zone_caisse_x_max_pct=float(
                self.db.obtenir_parametre("caisse_zone_x_max_pct", 0.75)
            ),
            seuil_proximite_mains=float(
                self.db.obtenir_parametre("caisse_seuil_proximite_mains", 0.08)
            ),
            nb_cycles_scan_min=int(
                self.db.obtenir_parametre("caisse_nb_cycles_scan_min", 2)
            ),
            cooldown_secondes=config.alert_cooldown_seconds,
        )

        # Initialiser le gestionnaire d'alertes
        self.alertes = GestionnaireAlertes(
            config=config,
            base_donnees=self.db,
            enregistreur=self.enregistreur,
        )

        # Compteurs de performance
        self._nb_frames = 0
        self._temps_debut = time.time()
        self._derniere_mesure_fps = time.time()
        self._fps_actuel = 0.0

        logger.info("Pipeline de detection de fraude initialise")

    def _charger_modeles(self):
        """Charge les modèles YOLO et pose estimation."""
        chemin_yolo = self.config.chemin_modele_yolo
        chemin_pose = self.config.chemin_modele_pose
        taille_yolo = int(self.db.obtenir_parametre("taille_entree_yolo", 320))

        # Détecteur YOLO principal
        if chemin_yolo.exists():
            try:
                self.detecteur = DetecteurPersonnes(
                    chemin_modele=chemin_yolo,
                    confiance_min=self.config.yolo_confidence,
                    taille_entree=taille_yolo,
                )
                logger.info(f"Detecteur YOLO charge (taille={taille_yolo})")
            except Exception as e:
                logger.error(f"Erreur chargement YOLO: {e}")
        else:
            logger.warning(
                f"Modele YOLO non trouve: {chemin_yolo}\n"
                f"Executez: python scripts/download_models.py"
            )

        # Estimateur de pose
        if chemin_pose.exists():
            try:
                self.estimateur_pose = EstimateurPose(
                    chemin_modele=chemin_pose,
                    confiance_min=self.config.pose_confidence,
                    taille_entree=taille_yolo,
                )
                logger.info(f"Estimateur de pose charge (taille={taille_yolo})")
            except Exception as e:
                logger.warning(f"Pose estimation non disponible: {e}")
        else:
            logger.warning(
                f"Modele pose non trouve: {chemin_pose}\n"
                f"Le systeme fonctionnera sans estimation de pose."
            )

    def _ouvrir_source(self, source: str) -> Optional[cv2.VideoCapture]:
        """
        Ouvre une source vidéo (webcam, RTSP, fichier).
        Configure les timeouts et buffers pour les flux RTSP.
        """
        # Webcam (index numérique)
        try:
            index = int(source)
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow sur Windows
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)  # Fallback
            if cap.isOpened():
                # Optimiser la résolution pour CPU
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                logger.info(f"Webcam ouverte (index {index})")
                return cap
        except ValueError:
            pass

        # RTSP: forcer le transport TCP et configurer les timeouts
        if source.startswith("rtsp://") or source.startswith("rtsps://"):
            # CAP_FFMPEG avec option TCP pour fiabilité
            env_opts = (
                f"rtsp_transport;tcp|"
                f"stimeout;5000000|"        # socket timeout 5s
                f"probesize;1000000|"       # probe 1MB (HEVC a besoin de plus que H.264)
                f"analyzeduration;1000000|"  # 1s analyse (HEVC headers)
                f"max_delay;0|"             # zero delay
                f"reorder_queue_size;0|"    # pas de reorder
                f"fflags;nobuffer+discardcorrupt|"  # zero buffer FFmpeg
                f"flags;low_delay"          # mode basse latence
            )
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = env_opts
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)

        if cap.isOpened():
            # Buffer minimal pour réduire la latence
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"Source video ouverte: {source}")
            return cap

        logger.error(f"Impossible d'ouvrir la source: {source}")
        return None

    def traiter_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Traite une frame complète à travers tout le pipeline.

        Args:
            frame: Image BGR

        Returns:
            Frame annotée avec les détections et alertes
        """
        taille_frame = frame.shape[:2]

        # Alimenter le buffer vidéo
        self.enregistreur.alimenter_buffer(frame)

        # 1. Détection YOLO (une seule inference pour personnes + objets)
        detections_personnes = []
        detections_objets = []

        if self.detecteur is not None:
            detections_personnes, detections_objets = self.detecteur.detecter_personnes_et_objets(frame)

        # 2. Suivi ByteTrack
        detections_pour_tracker = [
            (d.bbox, d.confidence) for d in detections_personnes
        ]
        pistes_actives = self.tracker.mettre_a_jour(detections_pour_tracker)

        # Mettre à jour l'historique des trajectoires
        for piste in pistes_actives:
            self.historique.ajouter_observation(piste.id_piste, piste.bbox)

        # 3. Estimation de pose (uniquement pour les pistes actives)
        poses = {}
        if self.estimateur_pose is not None and pistes_actives:
            bboxes = [p.bbox for p in pistes_actives]
            # Limiter à 5 personnes max pour les performances CPU
            bboxes_limit = bboxes[:5]
            resultats_pose = self.estimateur_pose.estimer_poses_multiples(
                frame, bboxes_limit
            )
            for i, piste in enumerate(pistes_actives[:5]):
                if i < len(resultats_pose) and resultats_pose[i] is not None:
                    poses[piste.id_piste] = resultats_pose[i]

        # 4. Analyse comportementale
        alertes_frame: List[ResultatAnalyse] = []
        for piste in pistes_actives:
            if piste.etat != "actif":
                continue

            pose = poses.get(piste.id_piste)
            resultats = self.analyseur.analyser(
                piste=piste,
                pose=pose,
                detections_objets=detections_objets,
                taille_frame=taille_frame,
            )
            alertes_frame.extend(resultats)

        # 4b. Analyse specifique caisse (machine a etats)
        alertes_caisse = self.analyseur_caisse.analyser(
            pistes=pistes_actives,
            poses=poses,
            detections_objets=detections_objets,
            taille_frame=taille_frame,
        )
        # Convertir les alertes caisse en ResultatAnalyse pour le systeme d'alertes
        for ac in alertes_caisse:
            type_c = TypeComportement(ac.type_alerte.value)
            resultat_caisse = ResultatAnalyse(
                type_comportement=type_c,
                confiance=ac.confiance,
                id_piste=ac.id_caissier,
                description=ac.description,
            )
            alertes_frame.append(resultat_caisse)

        # 5. Traiter les alertes
        for alerte in alertes_frame:
            piste = self.tracker.obtenir_piste(alerte.id_piste)
            bbox = piste.bbox if piste else None
            self.alertes.traiter_alerte(
                resultat=alerte,
                frame=frame,
                bbox=bbox,
            )

        # 6. Annoter la frame pour l'affichage
        frame_annotee = self._annoter_frame(
            frame, pistes_actives, detections_objets, poses, alertes_frame
        )

        # Maintenance périodique (toutes les 300 frames)
        self._nb_frames += 1
        if self._nb_frames % 300 == 0:
            ids_actifs = [p.id_piste for p in pistes_actives]
            self.analyseur.nettoyer_pistes_supprimees(ids_actifs)
            self.analyseur_caisse.nettoyer(ids_actifs)
            self.historique.nettoyer()
            self.alertes.nettoyer_cooldowns()

        # Calculer les FPS
        maintenant = time.time()
        if maintenant - self._derniere_mesure_fps >= 1.0:
            self._fps_actuel = self._nb_frames / (maintenant - self._temps_debut)
            self._derniere_mesure_fps = maintenant

        return frame_annotee

    def _annoter_frame(
        self,
        frame: np.ndarray,
        pistes: list,
        objets: List[Detection],
        poses: dict,
        alertes: List[ResultatAnalyse],
    ) -> np.ndarray:
        """Dessine les annotations sur la frame."""
        annotee = frame.copy()
        h, w = annotee.shape[:2]

        # Dessiner les pistes de personnes
        for piste in pistes:
            x1, y1, x2, y2 = piste.bbox
            couleur = (0, 255, 0)  # Vert par défaut

            # Changer la couleur si suspect
            score_susp = self.analyseur.obtenir_score_suspicion(piste.id_piste)
            if score_susp.score_global >= 0.6:
                couleur = (0, 0, 255)  # Rouge
            elif score_susp.score_global >= 0.3:
                couleur = (0, 165, 255)  # Orange

            cv2.rectangle(annotee, (x1, y1), (x2, y2), couleur, 2)

            # Label avec ID et score
            label = f"#{piste.id_piste} ({score_susp.score_global:.0%})"
            cv2.putText(
                annotee, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 1, cv2.LINE_AA,
            )

            # Dessiner la trajectoire récente
            centres = list(piste.historique_centres)[-30:]
            for i in range(1, len(centres)):
                pt1 = (int(centres[i - 1][0]), int(centres[i - 1][1]))
                pt2 = (int(centres[i][0]), int(centres[i][1]))
                cv2.line(annotee, pt1, pt2, couleur, 1, cv2.LINE_AA)

            # Dessiner les points de pose si disponibles
            pose = poses.get(piste.id_piste)
            if pose is not None:
                for j in range(17):
                    kp = pose.keypoints[j]
                    if kp[2] > 0.3:
                        cv2.circle(
                            annotee,
                            (int(kp[0]), int(kp[1])),
                            3, (0, 255, 255), -1,
                        )

        # Dessiner les objets détectés
        for obj in objets:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(annotee, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(
                annotee, obj.class_name,
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1,
            )

        # Barre d'informations en haut
        cv2.rectangle(annotee, (0, 0), (w, 30), (50, 50, 50), -1)
        info = (
            f"FPS: {self._fps_actuel:.1f} | "
            f"Personnes: {len(pistes)} | "
            f"Alertes: {self.alertes.compteur_alertes} | "
            f"Objets: {len(objets)}"
        )
        cv2.putText(
            annotee, info,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Afficher les alertes actives
        y_alerte = 50
        for alerte in alertes:
            texte = f"[!] {alerte.description} (#{alerte.id_piste})"
            cv2.putText(
                annotee, texte,
                (10, y_alerte),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
            )
            y_alerte += 25

        return annotee

    def _ecrire_heartbeat(self):
        """Ecrit un fichier heartbeat pour le healthcheck Docker."""
        try:
            heartbeat = Path(self.config.chemin_base_donnees).parent / "heartbeat"
            heartbeat.write_text(f"{time.time():.0f}\n{self._nb_frames}\n{self._fps_actuel:.1f}")
        except Exception as e:
            logger.debug(f"Erreur ecriture heartbeat: {e}")

    def executer(self, afficher: bool = True):
        """
        Lance la boucle de traitement principale (mode mono-source legacy).
        Gere les reconnexions RTSP avec backoff exponentiel.

        Args:
            afficher: Si True, affiche la video annotee dans une fenetre
        """
        global _arreter

        # Determiner la source video
        if self.source:
            sources = [self.source]
        elif self.config.webcam_test:
            sources = ["0"]
        else:
            sources = self.config.sources_liste

        if not sources:
            logger.error("Aucune source video configuree")
            return

        # Verification startup: modeles charges?
        if self.detecteur is None:
            logger.error("ERREUR CRITIQUE: Aucun modele YOLO charge. Pipeline ne peut pas demarrer.")
            return

        # Ouvrir la premiere source
        source_str = sources[0]
        cap = self._ouvrir_source(source_str)

        if cap is None:
            logger.error(f"Impossible d'ouvrir la source: {source_str}")
            logger.info("Conseil: essayez --test-webcam pour utiliser la webcam")
            return

        logger.info(f"Demarrage du traitement sur: {source_str}")
        logger.info("Appuyez sur 'q' pour quitter" if afficher else "Ctrl+C pour arreter")

        reconnect_delay = 2.0
        max_reconnect_delay = 60.0
        consecutive_failures = 0
        max_consecutive_failures = 50

        try:
            while not _arreter:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"Abandon apres {max_consecutive_failures} echecs consecutifs "
                            f"sur la source: {source_str}"
                        )
                        break

                    logger.warning(
                        f"Perte de la source video (echec #{consecutive_failures}), "
                        f"reconnexion dans {reconnect_delay:.0f}s..."
                    )
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

                    cap.release()
                    cap = self._ouvrir_source(source_str)
                    if cap is None:
                        continue
                    logger.info("Reconnexion reussie")
                    continue

                if consecutive_failures > 0:
                    logger.info(f"Flux restaure apres {consecutive_failures} echec(s)")
                    consecutive_failures = 0
                    reconnect_delay = 2.0

                try:
                    frame_annotee = self.traiter_frame(frame)
                except Exception as e:
                    logger.error(f"Erreur traitement frame #{self._nb_frames}: {e}", exc_info=True)
                    continue

                if afficher:
                    try:
                        cv2.imshow("Detection de Fraude - Magasin", frame_annotee)
                        touche = cv2.waitKey(1) & 0xFF
                        if touche == ord("q"):
                            logger.info("Arret demande par l'utilisateur")
                            break
                    except Exception:
                        afficher = False
                        logger.warning("Affichage desactive (mode headless detecte)")

                if self._nb_frames % 30 == 0:
                    self._ecrire_heartbeat()

                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interruption clavier")
        except Exception as e:
            logger.critical(f"Erreur fatale dans la boucle principale: {e}", exc_info=True)
        finally:
            logger.info("Fermeture du systeme...")
            cap.release()
            if afficher:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    logger.debug("Impossible de fermer les fenetres (mode headless)")
            self.enregistreur.arreter_tout()

            try:
                self.enregistreur.nettoyer_anciens_fichiers()
                self.db.nettoyer_anciennes_donnees(
                    self.config.retention_jours,
                    retention_alertes_jours=self.config.retention_alertes_jours,
                    retention_stats_jours=self.config.retention_stats_jours,
                )
            except Exception as e:
                logger.warning(f"Erreur lors du nettoyage: {e}")

            duree_totale = time.time() - self._temps_debut
            logger.info(
                f"Session terminee: {self._nb_frames} frames traitees en "
                f"{duree_totale:.0f}s ({self._nb_frames / max(duree_totale, 1):.1f} FPS moyen)"
            )


# ============================================================================
# MULTI-CAMERA ORCHESTRATOR
# ============================================================================


class OrchestrateurMultiCamera:
    """
    Orchestre plusieurs CameraWorker en parallele.
    Charge les cameras actives depuis la DB, cree un worker par camera,
    et supervise leur sante dans le thread principal.
    Les modeles YOLO et la DB sont partages entre tous les workers.
    """

    def __init__(self, config: FraudeConfig):
        self.config = config
        self.db = BaseDonneesFraude(config.chemin_base_donnees)

        # Charger config depuis DB
        config.charger_depuis_db(self.db)

        # Modeles partages (thread-safe en lecture)
        self.detecteur: Optional[DetecteurPersonnes] = None
        self.estimateur_pose: Optional[EstimateurPose] = None
        # Detecteur Open Images V7 (~600 classes) pour l'apprentissage.
        # Lazy-load: pas de cout RAM tant qu'aucune session d'apprentissage
        # n'est declenchee.
        self.detecteur_apprentissage: Optional[DetecteurApprentissage] = None
        self._charger_modeles()

        # Semaphore pour limiter la concurrence GPU/CPU sur l'inference
        self._inference_sem = threading.Semaphore(config.inference_concurrency)

        # Gestionnaire d'alertes partage
        # On cree un enregistreur "global" pour le GestionnaireAlertes
        # (chaque worker a son propre enregistreur pour les clips,
        #  mais le manager utilise celui-ci pour les snapshots d'alerte)
        self._enregistreur_global = EnregistreurVideo(
            repertoire_sortie=config.chemin_enregistrements,
            repertoire_snapshots=config.chemin_snapshots,
            duree_clip=config.video_clip_duration,
            fps=15,
            pre_evenement_secondes=5,
            retention_jours=config.retention_jours,
            retention_videos_jours=config.retention_videos_jours,
            retention_snapshots_jours=config.retention_snapshots_jours,
            quota_stockage_max_gb=config.quota_stockage_max_gb,
            quota_seuil_alerte_pct=config.quota_seuil_alerte_pct,
        )
        self.alertes_manager = GestionnaireAlertes(
            config=config,
            base_donnees=self.db,
            enregistreur=self._enregistreur_global,
        )

        # Workers
        self._workers: Dict[int, CameraWorker] = {}
        self._temps_debut = time.time()

    def _charger_modeles(self):
        """Charge les modeles YOLO et pose estimation (une seule fois, partages)."""
        chemin_yolo = self.config.chemin_modele_yolo
        chemin_pose = self.config.chemin_modele_pose
        taille_yolo = int(self.db.obtenir_parametre("taille_entree_yolo", 320))

        if chemin_yolo.exists():
            try:
                self.detecteur = DetecteurPersonnes(
                    chemin_modele=chemin_yolo,
                    confiance_min=self.config.yolo_confidence,
                    taille_entree=taille_yolo,
                )
                logger.info(f"Detecteur YOLO charge (partage, taille={taille_yolo})")
            except Exception as e:
                logger.error(f"Erreur chargement YOLO: {e}")
        else:
            logger.warning(f"Modele YOLO non trouve: {chemin_yolo}")

        if chemin_pose.exists():
            try:
                self.estimateur_pose = EstimateurPose(
                    chemin_modele=chemin_pose,
                    confiance_min=self.config.pose_confidence,
                    taille_entree=taille_yolo,
                )
                logger.info(f"Estimateur de pose charge (partage, taille={taille_yolo})")
            except Exception as e:
                logger.warning(f"Pose estimation non disponible: {e}")

        # Detecteur OIV7 pour apprentissage (lazy-load au premier usage).
        # On l'instancie meme si le fichier est absent : il ne chargera rien
        # tant que personne ne l'appelle, et loguera un warning en cas d'appel.
        chemin_oiv7 = self.config.chemin_modele_oiv7
        try:
            self.detecteur_apprentissage = DetecteurApprentissage(
                chemin_modele=chemin_oiv7,
                confiance_min=0.25,
                taille_entree=640,
            )
            if chemin_oiv7.exists():
                logger.info(f"Detecteur OIV7 pret (lazy-load): {chemin_oiv7.name}")
            else:
                logger.info(
                    f"Modele OIV7 absent ({chemin_oiv7}). Apprentissage "
                    f"retombera sur COCO 80 classes (fallback)."
                )
        except Exception as e:
            logger.warning(f"Detecteur OIV7 non disponible: {e}")
            self.detecteur_apprentissage = None

    def _creer_workers(self) -> int:
        """
        Cree un CameraWorker pour chaque camera active dans la DB.
        Retourne le nombre de workers crees.
        """
        cameras = self.db.obtenir_cameras(actives_seulement=True)

        if not cameras:
            logger.warning("Aucune camera active dans la base de donnees")
            return 0

        max_cam = self.config.max_cameras
        if len(cameras) > max_cam:
            logger.warning(
                f"{len(cameras)} cameras actives mais max_cameras={max_cam}, "
                f"seules les {max_cam} premieres seront utilisees"
            )
            cameras = cameras[:max_cam]

        for cam in cameras:
            cam_id = cam["id"]
            try:
                worker = CameraWorker(
                    camera_id=cam_id,
                    camera_nom=cam["nom"],
                    source_url=cam["source"],
                    zone=cam.get("zone", "inconnue"),
                    config=self.config,
                    detecteur=self.detecteur,
                    estimateur_pose=self.estimateur_pose,
                    db=self.db,
                    alertes_manager=self.alertes_manager,
                    inference_semaphore=self._inference_sem,
                    mode_detection=cam.get("mode_detection", "tout"),
                    detecteur_apprentissage=self.detecteur_apprentissage,
                )
                self._workers[cam_id] = worker
                logger.info(f"Worker cree: {cam['nom']} ({cam['source']})")
            except Exception as e:
                logger.error(f"Erreur creation worker camera {cam['nom']}: {e}")

        return len(self._workers)

    def _ecrire_heartbeat(self):
        """Ecrit un fichier heartbeat agrege pour le healthcheck Docker."""
        try:
            heartbeat = Path(self.config.chemin_base_donnees).parent / "heartbeat"
            total_frames = sum(
                w.obtenir_stats()["frames"] for w in self._workers.values()
            )
            nb_actifs = sum(1 for w in self._workers.values() if w.est_actif())
            heartbeat.write_text(
                f"{time.time():.0f}\n{total_frames}\n{nb_actifs}/{len(self._workers)} cameras"
            )
        except Exception as e:
            logger.debug(f"Erreur ecriture heartbeat multi-cam: {e}")

    def executer(self):
        """
        Demarre tous les workers et lance la boucle de supervision.
        Bloque jusqu'a reception d'un signal d'arret.
        """
        global _arreter

        if self.detecteur is None:
            logger.error("ERREUR CRITIQUE: Aucun modele YOLO charge. Impossible de demarrer.")
            return

        nb = self._creer_workers()
        if nb == 0:
            logger.error(
                "Aucun worker cree. Verifiez que des cameras actives sont "
                "configurees dans Administration > Cameras."
            )
            return

        logger.info(f"Demarrage de {nb} camera worker(s)...")

        # Demarrer tous les workers
        for worker in self._workers.values():
            worker.demarrer()

        logger.info(
            f"Tous les workers demarres. Superviseur actif "
            f"(interval: {self.config.supervisor_interval_seconds}s)"
        )

        # Boucle de supervision
        try:
            while not _arreter:
                time.sleep(self.config.supervisor_interval_seconds)

                if _arreter:
                    break

                # Verifier la sante des workers
                nb_actifs = 0
                for cam_id, worker in self._workers.items():
                    stats = worker.obtenir_stats()
                    if worker.est_actif():
                        nb_actifs += 1
                    else:
                        logger.warning(
                            f"Worker {stats['camera_nom']} ({stats['status']}) "
                            f"n'est plus actif — tentative de relance"
                        )
                        try:
                            worker.demarrer()
                            logger.info(f"Worker {stats['camera_nom']} relance")
                        except Exception as e:
                            logger.error(
                                f"Impossible de relancer {stats['camera_nom']}: {e}"
                            )

                # Heartbeat
                self._ecrire_heartbeat()

                # Log periodique
                duree = time.time() - self._temps_debut
                logger.info(
                    f"[Superviseur] {nb_actifs}/{len(self._workers)} actifs | "
                    f"uptime {duree:.0f}s"
                )

                # Nettoyage periodique (toutes les heures)
                if int(duree) % 3600 < self.config.supervisor_interval_seconds:
                    try:
                        self._enregistreur_global.nettoyer_anciens_fichiers()
                        self.db.nettoyer_anciennes_donnees(
                            self.config.retention_jours,
                            retention_alertes_jours=self.config.retention_alertes_jours,
                            retention_stats_jours=self.config.retention_stats_jours,
                        )
                        logger.info("Nettoyage periodique effectue")
                    except Exception as e:
                        logger.warning(f"Erreur nettoyage periodique: {e}")

                if nb_actifs == 0:
                    logger.error("Tous les workers sont morts. Arret du superviseur.")
                    break

        except KeyboardInterrupt:
            logger.info("Interruption clavier")
        except Exception as e:
            logger.critical(f"Erreur fatale superviseur: {e}", exc_info=True)
        finally:
            self._arreter_tout()

    def _arreter_tout(self):
        """Arrete proprement tous les workers et fait le nettoyage."""
        logger.info("Arret de tous les workers...")
        for worker in self._workers.values():
            try:
                worker.arreter()
            except Exception as e:
                logger.error(f"Erreur arret worker: {e}")

        self._enregistreur_global.arreter_tout()

        try:
            self._enregistreur_global.nettoyer_anciens_fichiers()
            self.db.nettoyer_anciennes_donnees(
                self.config.retention_jours,
                retention_alertes_jours=self.config.retention_alertes_jours,
                retention_stats_jours=self.config.retention_stats_jours,
            )
        except Exception as e:
            logger.warning(f"Erreur nettoyage final: {e}")

        duree = time.time() - self._temps_debut
        total_frames = sum(
            w.obtenir_stats()["frames"] for w in self._workers.values()
        )
        logger.info(
            f"Orchestrateur arrete: {len(self._workers)} cameras, "
            f"{total_frames} frames totales en {duree:.0f}s"
        )

    def obtenir_stats(self) -> List[Dict]:
        """Retourne les stats de tous les workers (pour le dashboard)."""
        return [w.obtenir_stats() for w in self._workers.values()]


def lancer_dashboard():
    """Lance le dashboard Streamlit séparément."""
    import subprocess
    config = obtenir_config()
    chemin_dashboard = Path(__file__).parent.parent / "dashboard" / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(chemin_dashboard),
        "--server.port", str(config.dashboard_port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    logger.info(f"Lancement du dashboard sur le port {config.dashboard_port}")
    subprocess.run(cmd)


def main():
    """Point d'entree avec gestion des arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Systeme de detection de fraude en magasin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m app.main                     # Mode multi-camera (cameras actives en DB)
  python -m app.main --single            # Mode mono-source (legacy)
  python -m app.main --test-webcam       # Test avec webcam (mono)
  python -m app.main --source video.mp4  # Fichier video (mono)
  python -m app.main --dashboard-only    # Dashboard seul
  python -m app.main --no-display        # Sans affichage (mono)
        """,
    )

    parser.add_argument(
        "--single",
        action="store_true",
        help="Mode mono-source (legacy, premiere source de config)",
    )
    parser.add_argument(
        "--test-webcam",
        action="store_true",
        help="Utiliser la webcam du PC (index 0) pour les tests",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source video specifique (index webcam, URL RTSP, ou chemin fichier)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Desactiver l'affichage video (mode serveur/Docker)",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Lancer uniquement le dashboard Streamlit",
    )

    args = parser.parse_args()

    # Mode dashboard uniquement
    if args.dashboard_only:
        lancer_dashboard()
        return

    # Configuration
    config = obtenir_config()

    # Mode mono-source: --single, --test-webcam, ou --source
    if args.single or args.test_webcam or args.source:
        if args.test_webcam:
            config.webcam_test = True
        source = args.source
        if args.test_webcam and source is None:
            source = "0"
        pipeline = PipelineFraude(config=config, source=source)
        pipeline.executer(afficher=not args.no_display)
    else:
        # Mode multi-camera (defaut)
        logger.info("Demarrage en mode multi-camera")
        orchestrateur = OrchestrateurMultiCamera(config=config)
        orchestrateur.executer()


if __name__ == "__main__":
    main()

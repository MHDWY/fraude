"""
Worker de traitement par camera.
Chaque CameraWorker gere une source video dans son propre thread
avec ses propres instances de tracker, analyseur et enregistreur.
Les modeles YOLO et la base de donnees sont partages (thread-safe).
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .behavior_analyzer import AnalyseurComportements, ResultatAnalyse, TypeComportement
from .caisse_analyzer import AnalyseurCaisse
from .config import FraudeConfig
from .database import BaseDonneesFraude
from .detector import DetecteurPersonnes, EstimateurPose, DetecteurApprentissage
from .tracker import ByteTracker, HistoriqueTrajectoires
from .video_recorder import EnregistreurVideo

logger = logging.getLogger("fraude.worker")


class CameraWorker:
    """
    Pipeline de detection pour une seule camera.
    Tourne dans son propre thread avec ses propres composants stateful.
    Partage les modeles YOLO et la DB avec les autres workers.
    """

    def __init__(
        self,
        camera_id: int,
        camera_nom: str,
        source_url: str,
        zone: str,
        config: FraudeConfig,
        detecteur: DetecteurPersonnes,
        estimateur_pose: Optional[EstimateurPose],
        db: BaseDonneesFraude,
        alertes_manager,  # GestionnaireAlertes
        inference_semaphore: threading.Semaphore,
        mode_detection: str = "tout",  # "tout", "vol", "caisse"
        detecteur_apprentissage: Optional["DetecteurApprentissage"] = None,
    ):
        self.camera_id = camera_id
        self.camera_nom = camera_nom
        self.source_url = source_url
        self.zone = zone
        self.mode_detection = mode_detection
        self.config = config

        # Ressources partagees (thread-safe)
        self.detecteur = detecteur
        self.estimateur_pose = estimateur_pose
        # Detecteur OIV7 partage (~600 classes) pour l'apprentissage zones.
        # Lazy-loaded: ne consomme rien tant qu'aucune session n'est active.
        # Fallback: si None, l'apprentissage utilise le detecteur COCO.
        self.detecteur_apprentissage = detecteur_apprentissage
        self.db = db
        self.alertes_manager = alertes_manager
        self._inference_sem = inference_semaphore

        # Ressources propres a cette camera
        self.tracker = ByteTracker(
            seuil_score_haut=config.yolo_confidence,
            seuil_score_bas=0.1,
            max_frames_perdues=30,
        )
        self.historique = HistoriqueTrajectoires(duree_max_secondes=300)

        self.analyseur = AnalyseurComportements(
            seuil_alerte=config.behavior_threshold,
            cooldown_secondes=config.alert_cooldown_seconds,
            db=db,
        )

        # Charger les objets de reference pour cette camera
        self.objets_reference = db.obtenir_objets_reference(camera_id)
        self._objets_par_role = {}
        for obj in self.objets_reference:
            if obj["role"]:
                self._objets_par_role[obj["role"].lower()] = obj

        # Zones mannequins calibrees (role = "mannequin")
        self._mannequin_bboxes = []
        for obj in self.objets_reference:
            if (obj.get("role") or "").lower() == "mannequin":
                self._mannequin_bboxes.append((
                    obj["bbox_x1"], obj["bbox_y1"],
                    obj["bbox_x2"], obj["bbox_y2"],
                ))
        if self._mannequin_bboxes:
            logger.info(f"[{camera_nom}] {len(self._mannequin_bboxes)} mannequin(s) calibre(s)")

        # Filtre immobilite position-based (survit aux changements de track ID)
        self._zones_statiques: List[List] = []  # [[cx, cy, first_seen_time]]
        self._seuil_immobilite_sec = float(db.obtenir_parametre("mannequin_seuil_immobilite_sec", 30))
        self._seuil_deplacement_px = 40

        # Zone imprimante depuis les objets de reference (calibration visuelle)
        obj_imprimante = self._objets_par_role.get("imprimante")
        if obj_imprimante:
            self._imprimante_bbox = (
                obj_imprimante["bbox_x1"], obj_imprimante["bbox_y1"],
                obj_imprimante["bbox_x2"], obj_imprimante["bbox_y2"],
            )
            logger.info(f"Imprimante identifiee pour camera '{camera_nom}': "
                        f"bbox=({obj_imprimante['bbox_x1']:.0f},{obj_imprimante['bbox_y1']:.0f})-"
                        f"({obj_imprimante['bbox_x2']:.0f},{obj_imprimante['bbox_y2']:.0f})")
        else:
            self._imprimante_bbox = None

        # Zones d'exclusion (coordonnees en pourcentage 0.0-1.0)
        self._zones_exclusion_pct = []
        self._charger_zones_exclusion(db)

        # QW1: polygone de masque imprimante (JSON, ROI-relatif)
        imprimante_mask_polygon = self._charger_imprimante_mask(db)

        # Charger les params caisse depuis la DB
        self.analyseur_caisse = AnalyseurCaisse(
            timeout_ticket_secondes=float(db.obtenir_parametre("caisse_timeout_ticket", 12.0)),
            zone_caisse_y_min_pct=float(db.obtenir_parametre("caisse_zone_y_min_pct", 0.70)),
            zone_caisse_x_min_pct=float(db.obtenir_parametre("caisse_zone_x_min_pct", 0.25)),
            zone_caisse_x_max_pct=float(db.obtenir_parametre("caisse_zone_x_max_pct", 0.75)),
            seuil_proximite_mains=float(db.obtenir_parametre("caisse_seuil_proximite_mains", 0.08)),
            nb_cycles_scan_min=int(db.obtenir_parametre("caisse_nb_cycles_scan_min", 2)),
            cooldown_secondes=config.alert_cooldown_seconds,
            imprimante_seuil_blanc=int(db.obtenir_parametre("imprimante_seuil_blanc", 200)),
            imprimante_seuil_changement=float(db.obtenir_parametre("imprimante_seuil_changement", 0.15)),
            imprimante_bbox=self._imprimante_bbox,
            imprimante_mask_polygon=imprimante_mask_polygon,
            imprimante_mode_detection=str(db.obtenir_parametre("imprimante_mode_detection", "hsv")),
            imprimante_seuil_saturation=int(db.obtenir_parametre("imprimante_seuil_saturation", 40)),
            imprimante_seuil_valeur=int(db.obtenir_parametre("imprimante_seuil_valeur", 180)),
            imprimante_min_frames_consecutives=int(db.obtenir_parametre("imprimante_min_frames_consecutives", 2)),
            imprimante_cooldown_detection=float(db.obtenir_parametre("imprimante_cooldown_detection", 4.0)),
            detecter_transaction_fantome=bool(db.obtenir_parametre("caisse_detecter_transaction_fantome", True)),
        )

        # Enregistreur video specifique a cette camera
        cam_slug = camera_nom.replace(" ", "_").lower()[:20]
        self.enregistreur = EnregistreurVideo(
            repertoire_sortie=config.chemin_enregistrements / cam_slug,
            repertoire_snapshots=config.chemin_snapshots / cam_slug,
            duree_clip=config.video_clip_duration,
            fps=15,
            pre_evenement_secondes=5,
            retention_jours=config.retention_jours,
            retention_videos_jours=config.retention_videos_jours,
            retention_snapshots_jours=config.retention_snapshots_jours,
            quota_stockage_max_gb=config.quota_stockage_max_gb,
            quota_seuil_alerte_pct=config.quota_seuil_alerte_pct,
        )

        # Skip d'inference (N=0 => toutes les frames, N=2 => 1 frame sur 3)
        try:
            self._inference_skip = max(0, int(db.obtenir_parametre("inference_frame_skip", 0)))
        except (TypeError, ValueError):
            self._inference_skip = 0
        self._inference_counter = 0

        # Etat du worker
        self._arreter = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nb_frames = 0
        self._fps = 0.0
        self._status = "initialise"
        self._derniere_frame_ts = 0.0
        self._reconnections = 0
        self._lock_stats = threading.Lock()

        # Thread de grab dedie pour RTSP (evite l'accumulation de buffer)
        self._grab_thread: Optional[threading.Thread] = None
        self._derniere_frame: Optional[np.ndarray] = None
        self._frame_event = threading.Event()
        self._lock_frame = threading.Lock()
        self._grab_actif = False
        self._is_rtsp = source_url.startswith(("rtsp://", "rtsps://"))

        # Auto-apprentissage
        self._apprentissage_actif = False
        self._apprentissage_session_id: Optional[int] = None
        self._apprentissage_debut: float = 0.0
        self._apprentissage_duree_sec: float = 0.0
        self._apprentissage_observations: Dict[int, List] = {}
        self._apprentissage_next_obs_id = 0
        self._apprentissage_inference_every = 15  # 1 inference complete toutes les N frames
        self._apprentissage_max_obs = 500  # Cap dict pour eviter fuite memoire

        # Timeout du semaphore d'inference (anti-deadlock si YOLO hang)
        self._inference_timeout_sec = 30.0

    @contextmanager
    def _acquire_inference(self):
        """Context manager pour le semaphore d'inference avec timeout.
        Si acquire echoue dans le delai imparti, yield False et skip l'inference.
        Evite les deadlocks si ONNX hang sur un worker."""
        acquired = self._inference_sem.acquire(timeout=self._inference_timeout_sec)
        if not acquired:
            logger.error(
                f"[{self.camera_nom}] Timeout semaphore inference "
                f"({self._inference_timeout_sec}s). Skip frame."
            )
            yield False
            return
        try:
            yield True
        finally:
            self._inference_sem.release()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def demarrer(self):
        """Lance le worker dans un thread daemon."""
        self._thread = threading.Thread(
            target=self._boucle_principale,
            name=f"cam-{self.camera_nom}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[{self.camera_nom}] Worker demarre")

    def arreter(self):
        """Signale l'arret au worker."""
        self._arreter.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self.enregistreur.arreter_tout()
        logger.info(f"[{self.camera_nom}] Worker arrete")

    def est_actif(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def obtenir_stats(self) -> Dict:
        """Stats thread-safe pour le monitoring."""
        with self._lock_stats:
            return {
                "camera_id": self.camera_id,
                "camera_nom": self.camera_nom,
                "source": self.source_url,
                "zone": self.zone,
                "status": self._status,
                "fps": round(self._fps, 1),
                "frames": self._nb_frames,
                "reconnections": self._reconnections,
                "derniere_frame": self._derniere_frame_ts,
            }

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    @staticmethod
    def _charger_imprimante_mask(db: BaseDonneesFraude) -> Optional[List[Tuple[float, float]]]:
        """QW1: charge le polygone mask imprimante depuis la DB (JSON string).

        Format attendu: liste de [x, y] en proportions de la ROI (0.0-1.0).
        Retourne None si vide, mal forme, ou < 3 points.
        """
        raw = db.obtenir_parametre("imprimante_mask_polygon", "")
        if not raw or not isinstance(raw, str):
            return None
        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"[QW1] imprimante_mask_polygon invalide (JSON): {e}")
            return None
        if not isinstance(data, list) or len(data) < 3:
            logger.warning(f"[QW1] imprimante_mask_polygon doit etre une liste de >=3 points")
            return None
        try:
            return [(float(p[0]), float(p[1])) for p in data]
        except (TypeError, ValueError, IndexError) as e:
            logger.warning(f"[QW1] imprimante_mask_polygon: points mal formes ({e})")
            return None

    def _ouvrir_source(self) -> Optional[cv2.VideoCapture]:
        """Ouvre la source video avec config RTSP optimisee."""
        source = self.source_url

        # Webcam
        try:
            index = int(source)
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                return cap
            # Liberer le handle meme si l'ouverture a echoue (evite fuite FD)
            try:
                cap.release()
            except Exception:
                pass
            return None
        except ValueError:
            pass

        # RTSP avec transport TCP
        if source.startswith("rtsp://") or source.startswith("rtsps://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000"
                "|probesize;1000000|analyzeduration;1000000"
                "|max_delay;0|reorder_queue_size;0"
                "|fflags;nobuffer+discardcorrupt|flags;low_delay"
            )
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        # Liberer le handle pour eviter une fuite de file descriptor
        try:
            cap.release()
        except Exception:
            pass
        return None

    def _boucle_grab(self, cap: cv2.VideoCapture):
        """Thread dedie qui lit le flux RTSP en continu via read().
        Ne garde que la derniere frame decodee — vide le buffer en permanence.
        Alimente aussi le buffer video a la cadence camera native (independant
        du FPS d'inference) pour que les enregistrements restent fluides."""
        while self._grab_actif and not self._arreter.is_set():
            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock_frame:
                    self._derniere_frame = frame
                self._frame_event.set()
                try:
                    self.enregistreur.alimenter_buffer(frame)
                except Exception as e:
                    logger.debug(f"[{self.camera_nom}] alimenter_buffer (grab): {e}")
            else:
                self._frame_event.clear()
                time.sleep(0.01)
        logger.debug(f"[{self.camera_nom}] Grab thread arrete")

    def _demarrer_grab(self, cap: cv2.VideoCapture):
        """Lance le thread de grab pour les flux RTSP."""
        self._grab_actif = True
        self._derniere_frame = None
        self._frame_event.clear()
        self._grab_thread = threading.Thread(
            target=self._boucle_grab, args=(cap,),
            name=f"grab-{self.camera_nom}", daemon=True,
        )
        self._grab_thread.start()

    def _arreter_grab(self):
        """Stoppe le thread de grab. Timeout long pour eviter race avec cap.release().
        Si le thread n'exit pas proprement (read() bloque sur TCP), on loggue mais
        on attend davantage pour laisser cv2 terminer son appel en cours."""
        self._grab_actif = False
        self._frame_event.set()  # Debloque tout _lire_frame en attente
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=8)
            if self._grab_thread.is_alive():
                logger.warning(
                    f"[{self.camera_nom}] Grab thread n'a pas exit en 8s "
                    f"(read() probablement bloque). Release() risque."
                )
        self._grab_thread = None
        self._derniere_frame = None
        self._frame_event.clear()

    def _lire_frame(self, cap: cv2.VideoCapture):
        """Lit une frame. RTSP: derniere frame du grab thread. Sinon: lecture directe."""
        if self._is_rtsp:
            if not self._frame_event.wait(timeout=5.0):
                return False, None
            with self._lock_frame:
                frame = self._derniere_frame
            if frame is None:
                return False, None
            return True, frame
        else:
            return cap.read()

    def _boucle_principale(self):
        """Boucle de traitement — tourne dans le thread du worker."""
        self._maj_status("connexion")
        cap = self._ouvrir_source()

        if cap is None:
            logger.error(f"[{self.camera_nom}] Impossible d'ouvrir: {self.source_url}")
            self._maj_status("erreur_connexion")
            cap = self._reconnexion_loop()
            if cap is None:
                self._maj_status("abandonne")
                return

        # Lancer le grab thread pour RTSP
        if self._is_rtsp:
            self._demarrer_grab(cap)

        self._maj_status("actif")
        logger.info(f"[{self.camera_nom}] Flux ouvert: {self.source_url}")

        reconnect_delay = 2.0
        consecutive_failures = 0
        temps_debut = time.time()
        derniere_mesure = time.time()

        try:
            while not self._arreter.is_set():
                ret, frame = self._lire_frame(cap)

                # Garde defensive: frame None meme si ret==True
                if ret and frame is None:
                    ret = False

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 50:
                        logger.error(f"[{self.camera_nom}] Abandon apres 50 echecs")
                        break

                    self._maj_status("reconnexion")
                    logger.warning(
                        f"[{self.camera_nom}] Perte flux (#{consecutive_failures}), "
                        f"retry dans {reconnect_delay:.0f}s"
                    )
                    # Ordre critique: arreter grab AVANT release (eviter race read/release)
                    self._arreter_grab()
                    try:
                        cap.release()
                    except Exception as e:
                        logger.debug(f"[{self.camera_nom}] cap.release erreur: {e}")
                    cap = None
                    self._arreter.wait(reconnect_delay)
                    if self._arreter.is_set():
                        break
                    reconnect_delay = min(reconnect_delay * 2, 60.0)
                    cap = self._ouvrir_source()
                    if cap is None:
                        with self._lock_stats:
                            self._reconnections += 1
                        continue
                    if self._is_rtsp:
                        self._demarrer_grab(cap)
                    logger.info(f"[{self.camera_nom}] Reconnecte")
                    continue

                # Reset backoff sur succes
                if consecutive_failures > 0:
                    consecutive_failures = 0
                    reconnect_delay = 2.0
                    self._maj_status("actif")

                # Traiter la frame
                try:
                    self._traiter_frame(frame)
                except Exception as e:
                    logger.error(f"[{self.camera_nom}] Erreur frame #{self._nb_frames}: {e}")

                # Compteurs
                with self._lock_stats:
                    self._nb_frames += 1
                    self._derniere_frame_ts = time.time()

                maintenant = time.time()
                if maintenant - derniere_mesure >= 2.0:
                    with self._lock_stats:
                        self._fps = self._nb_frames / max(maintenant - temps_debut, 1)
                    derniere_mesure = maintenant

                # Poll session apprentissage frequent (cheap DB SELECT)
                if self._nb_frames % 30 == 0 and not self._apprentissage_actif:
                    session = self.db.obtenir_session_apprentissage_active(self.camera_id)
                    if session:
                        self.demarrer_apprentissage(session["id"], session["duree_minutes"])

                # Maintenance periodique
                if self._nb_frames % 300 == 0:
                    ids_actifs = [p.id_piste for p in self.tracker.pistes_actives]
                    self.analyseur.nettoyer_pistes_supprimees(ids_actifs)
                    self.analyseur_caisse.nettoyer(ids_actifs)
                    self.historique.nettoyer()
                    # Nettoyer cooldowns alertes (partages multi-cam)
                    if self.alertes_manager is not None:
                        try:
                            self.alertes_manager.nettoyer_cooldowns()
                        except Exception as e:
                            logger.debug(f"[{self.camera_nom}] Nettoyage cooldowns: {e}")
                    # Nettoyer les zones statiques trop vieilles (>5 min)
                    seuil_nettoyage = time.time() - 300
                    self._zones_statiques = [
                        z for z in self._zones_statiques if z[2] > seuil_nettoyage
                    ]
                    # Recharger les zones d'exclusion (capte les changements dashboard)
                    self._charger_zones_exclusion()
                    # Verifier si une session d'apprentissage a ete demandee
                    if not self._apprentissage_actif:
                        session = self.db.obtenir_session_apprentissage_active(self.camera_id)
                        if session:
                            self.demarrer_apprentissage(session["id"], session["duree_minutes"])

        except Exception as e:
            logger.critical(f"[{self.camera_nom}] Erreur fatale: {e}", exc_info=True)
        finally:
            # Arreter grab AVANT release, garder cap nullable-safe
            self._arreter_grab()
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    logger.debug(f"[{self.camera_nom}] cap.release erreur (finally): {e}")
            self.enregistreur.arreter_tout()
            self._maj_status("arrete")
            duree = time.time() - temps_debut
            logger.info(
                f"[{self.camera_nom}] Termine: {self._nb_frames} frames en {duree:.0f}s "
                f"({self._nb_frames / max(duree, 1):.1f} FPS)"
            )

    def _reconnexion_loop(self) -> Optional[cv2.VideoCapture]:
        """Tente des reconnexions avec backoff avant d'abandonner."""
        delay = 5.0
        for i in range(10):
            if self._arreter.is_set():
                return None
            logger.info(f"[{self.camera_nom}] Tentative de connexion {i + 1}/10...")
            self._arreter.wait(delay)
            cap = self._ouvrir_source()
            if cap is not None:
                return cap
            delay = min(delay * 1.5, 30.0)
            with self._lock_stats:
                self._reconnections += 1
        return None

    # ------------------------------------------------------------------
    # Traitement d'une frame
    # ------------------------------------------------------------------

    def _traiter_frame(self, frame: np.ndarray):
        """Pipeline complet pour une frame."""
        taille_frame = frame.shape[:2]

        # Buffer video: pour RTSP, alimente depuis _boucle_grab (cadence camera).
        # Pour webcam/fichier, on alimente ici (pas de grab thread dedie).
        if not self._is_rtsp:
            self.enregistreur.alimenter_buffer(frame)

        # Skip d'inference: on saute YOLO/pose/analyse sur N-1 frames sur N.
        # Le buffer video continue d'etre alimente (via grab thread pour RTSP).
        if self._inference_skip > 0:
            self._inference_counter += 1
            if self._inference_counter % (self._inference_skip + 1) != 0:
                return

        # 1. Detection YOLO (protegee par semaphore avec timeout anti-deadlock)
        detections_personnes = []
        detections_objets = []

        with self._acquire_inference() as acquired:
            if acquired and self.detecteur is not None:
                detections_personnes, detections_objets = (
                    self.detecteur.detecter_personnes_et_objets(frame)
                )

        # 1a. Apprentissage automatique (si actif, collecte les observations)
        # Utilise une detection large (OIV7 ~600 classes, fallback COCO 80)
        # toutes les N frames pour capturer les objets statiques
        # (caisse, imprimante, scanner, ecran, mannequin, etagere, etc.).
        if self._apprentissage_actif:
            if self._nb_frames % self._apprentissage_inference_every == 0:
                seuil_appr = float(self.db.obtenir_parametre(
                    "apprentissage_confiance_min", 0.3))
                # Priorite OIV7 (~600 classes), fallback COCO 80 classes
                if self.detecteur_apprentissage is not None:
                    detecteur_appr, methode, source_modele = (
                        self.detecteur_apprentissage, "detecter_tout", "OIV7")
                elif self.detecteur is not None:
                    detecteur_appr, methode, source_modele = (
                        self.detecteur, "detecter_tout_coco", "COCO")
                else:
                    detecteur_appr, methode, source_modele = (None, None, "aucun")

                detections_appr = []
                if detecteur_appr is not None:
                    with self._acquire_inference() as acquired:
                        if acquired:
                            detections_appr = getattr(detecteur_appr, methode)(
                                frame, confiance_min=seuil_appr)

                if detections_appr:
                    classes = [d.class_name for d in detections_appr[:5]]
                    logger.debug(f"[{self.camera_nom}] Apprentissage cycle: "
                                 f"{len(detections_appr)} det {source_modele} @ {seuil_appr} "
                                 f"(top: {classes}), obs={len(self._apprentissage_observations)}")
                    self._traiter_apprentissage(detections_appr, taille_frame)

        # 1b. Filtrer les mannequins calibres (avant tracking)
        if self._mannequin_bboxes and detections_personnes:
            detections_personnes = [
                d for d in detections_personnes
                if not self._est_mannequin_calibre(d.bbox)
            ]

        # 1c. Filtrer les zones d'exclusion (avant tracking)
        if self._zones_exclusion_pct:
            h, w = taille_frame
            detections_personnes = [
                d for d in detections_personnes
                if not self._est_dans_zone_exclusion(d.bbox, w, h)
            ]
            detections_objets = [
                d for d in detections_objets
                if not self._est_dans_zone_exclusion(d.bbox, w, h)
            ]

        # 2. Suivi ByteTrack
        detections_pour_tracker = [(d.bbox, d.confidence) for d in detections_personnes]
        pistes_actives = self.tracker.mettre_a_jour(detections_pour_tracker)

        for piste in pistes_actives:
            self.historique.ajouter_observation(piste.id_piste, piste.bbox)

        # 3. Pose estimation (aussi protegee par semaphore)
        # Skip pose en mode "caisse": la detection visuelle imprimante suffit,
        # la pose ne sert qu'au fallback "geste de prise ticket".
        poses = {}
        if (self.estimateur_pose is not None
                and pistes_actives
                and self.mode_detection != "caisse"):
            bboxes = [p.bbox for p in pistes_actives[:5]]
            resultats_pose = []
            with self._acquire_inference() as acquired:
                if acquired:
                    resultats_pose = self.estimateur_pose.estimer_poses_multiples(frame, bboxes)
            for i, piste in enumerate(pistes_actives[:5]):
                if i < len(resultats_pose) and resultats_pose[i] is not None:
                    poses[piste.id_piste] = resultats_pose[i]

        # DEBUG pipeline — activer via LOG_LEVEL=DEBUG si necessaire
        if logger.isEnabledFor(logging.DEBUG) and self._nb_frames % 30 == 0:
            nb_immobiles = sum(1 for p in pistes_actives if p.etat == "actif" and self._est_immobile(p))
            scores = self.analyseur._scores if hasattr(self.analyseur, '_scores') else {}
            top_scores = {k: f"{max(v.values()):.2f}" for k, v in scores.items() if v and max(v.values()) > 0.01}
            mean_px = float(np.mean(frame)) if frame is not None else -1
            det_confs = [f"{d.confidence:.2f}" for d in detections_personnes] if detections_personnes else []
            piste_etats = [f"{p.id_piste}:{p.etat}" for p in pistes_actives] if pistes_actives else []
            logger.debug(
                f"[{self.camera_nom}] PIPELINE: frame={frame.shape if frame is not None else None} "
                f"mean_px={mean_px:.0f} det={self.detecteur is not None} "
                f"pers={len(detections_personnes)} confs={det_confs} "
                f"pistes={len(pistes_actives)} etats={piste_etats} poses={len(poses)} "
                f"immobiles={nb_immobiles} scores={top_scores or '-'}"
            )

        # 4. Analyse comportementale (vol) — sauf si mode = "caisse" uniquement
        alertes_frame: List[ResultatAnalyse] = []
        if self.mode_detection in ("tout", "vol"):
            for piste in pistes_actives:
                if piste.etat == "supprime":
                    continue
                # Ignorer les pistes immobiles (mannequins non calibres)
                if self._est_immobile(piste):
                    continue
                pose = poses.get(piste.id_piste)
                resultats = self.analyseur.analyser(
                    piste=piste, pose=pose,
                    detections_objets=detections_objets,
                    taille_frame=taille_frame,
                )
                alertes_frame.extend(resultats)

        # 4b. Analyse caisse — sauf si mode = "vol" uniquement
        if self.mode_detection in ("tout", "caisse"):
            alertes_caisse = self.analyseur_caisse.analyser(
                pistes=pistes_actives, poses=poses,
                detections_objets=detections_objets,
                taille_frame=taille_frame,
                frame=frame,
            )
            for ac in alertes_caisse:
                type_c = TypeComportement(ac.type_alerte.value)
                alertes_frame.append(ResultatAnalyse(
                    type_comportement=type_c,
                    confiance=ac.confiance,
                    id_piste=ac.id_caissier,
                    description=ac.description,
                ))

        # 5. Traiter les alertes
        for alerte in alertes_frame:
            piste = self.tracker.obtenir_piste(alerte.id_piste)
            bbox = piste.bbox if piste else None
            self.alertes_manager.traiter_alerte(
                resultat=alerte,
                frame=frame,
                bbox=bbox,
                source_camera=self.camera_nom,
                camera_id=self.camera_id,
                enregistreur_camera=self.enregistreur,
            )

    # ------------------------------------------------------------------
    # Filtrage mannequins
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box1, box2) -> float:
        """Intersection over Union entre deux bboxes (x1,y1,x2,y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / max(area1 + area2 - inter, 1e-6)

    def _est_mannequin_calibre(self, bbox) -> bool:
        """Verifie si une detection chevauche une zone mannequin calibree."""
        for mb in self._mannequin_bboxes:
            if self._iou(bbox, mb) > 0.3:
                return True
        return False

    def _charger_zones_exclusion(self, db=None):
        """Charge les zones d'exclusion depuis la DB."""
        db = db or self.db
        zones = db.obtenir_zones_exclusion(self.camera_id)
        self._zones_exclusion_pct = [
            (z["pct_x1"], z["pct_y1"], z["pct_x2"], z["pct_y2"], z.get("label", ""))
            for z in zones
        ]
        if self._zones_exclusion_pct:
            logger.info(f"[{self.camera_nom}] {len(self._zones_exclusion_pct)} zone(s) d'exclusion chargee(s)")

    def _est_dans_zone_exclusion(self, bbox, largeur_frame: int, hauteur_frame: int) -> bool:
        """Verifie si le centre d'une detection tombe dans une zone d'exclusion."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        cx_pct = cx / largeur_frame
        cy_pct = cy / hauteur_frame
        for (px1, py1, px2, py2, _label) in self._zones_exclusion_pct:
            if px1 <= cx_pct <= px2 and py1 <= cy_pct <= py2:
                return True
        return False

    # ------------------------------------------------------------------
    # Auto-apprentissage
    # ------------------------------------------------------------------

    def demarrer_apprentissage(self, session_id: int, duree_minutes: float):
        """Active le mode apprentissage pour cette camera."""
        self._apprentissage_actif = True
        self._apprentissage_session_id = session_id
        self._apprentissage_debut = time.time()
        self._apprentissage_duree_sec = duree_minutes * 60
        self._apprentissage_observations = {}
        self._apprentissage_next_obs_id = 0
        logger.info(f"[{self.camera_nom}] Apprentissage demarre "
                     f"(session {session_id}, {duree_minutes:.0f} min)")

    def _traiter_apprentissage(self, detections, taille_frame):
        """Collecte les observations pendant l'apprentissage."""
        if not self._apprentissage_actif:
            return

        maintenant = time.time()
        elapsed = maintenant - self._apprentissage_debut

        # Fin de session ?
        if elapsed >= self._apprentissage_duree_sec:
            self._terminer_apprentissage()
            return

        h, w = taille_frame
        seuil_dist = float(self.db.obtenir_parametre(
            "apprentissage_seuil_deplacement_px", 30)) / max(w, h)
        confiance_min = float(self.db.obtenir_parametre(
            "apprentissage_confiance_min", 0.3))

        # Exclure les personnes : on cherche des objets statiques
        toutes = [d for d in detections
                  if d.confidence >= confiance_min and d.class_name != "personne"]

        for det in toutes:
            cx = (det.bbox[0] + det.bbox[2]) / 2 / w
            cy = (det.bbox[1] + det.bbox[3]) / 2 / h
            bw = (det.bbox[2] - det.bbox[0]) / w
            bh = (det.bbox[3] - det.bbox[1]) / h

            # Chercher une observation existante proche
            matched = False
            for obs in self._apprentissage_observations.values():
                dist = ((cx - obs[0]) ** 2 + (cy - obs[1]) ** 2) ** 0.5
                if dist < seuil_dist:
                    obs[7] = maintenant  # last_seen
                    obs[8] += 1          # count
                    # Mise a jour glissante du centre et taille
                    obs[0] = obs[0] * 0.9 + cx * 0.1
                    obs[1] = obs[1] * 0.9 + cy * 0.1
                    obs[2] = obs[2] * 0.9 + bw * 0.1
                    obs[3] = obs[3] * 0.9 + bh * 0.1
                    matched = True
                    break

            if not matched:
                # Cap sur la taille du dict pour eviter fuite memoire.
                # On purge les observations stale (last_seen le plus ancien) —
                # les objets statiques reels ont un last_seen recent car on
                # les match a chaque detection.
                if len(self._apprentissage_observations) >= self._apprentissage_max_obs:
                    # Supprimer les 10% plus anciens par last_seen
                    a_purger = max(1, self._apprentissage_max_obs // 10)
                    triees = sorted(
                        self._apprentissage_observations.items(),
                        key=lambda kv: kv[1][7],  # last_seen
                    )
                    for cle, _ in triees[:a_purger]:
                        del self._apprentissage_observations[cle]

                # [cx, cy, bw, bh, classe, confiance, first_seen, last_seen, count]
                self._apprentissage_observations[self._apprentissage_next_obs_id] = [
                    cx, cy, bw, bh, det.class_name, det.confidence,
                    maintenant, maintenant, 1,
                ]
                self._apprentissage_next_obs_id += 1

    def _terminer_apprentissage(self):
        """Termine l'apprentissage et ecrit les zones proposees dans la DB."""
        seuil_sec = float(self.db.obtenir_parametre(
            "apprentissage_seuil_immobilite_sec", 120.0))

        nb_proposees = 0
        for obs in self._apprentissage_observations.values():
            duree = obs[7] - obs[6]  # last_seen - first_seen
            if duree >= seuil_sec:
                cx, cy, bw, bh = obs[0], obs[1], obs[2], obs[3]
                marge = 0.02  # 2% marge
                pct_x1 = max(0.0, cx - bw / 2 - marge)
                pct_y1 = max(0.0, cy - bh / 2 - marge)
                pct_x2 = min(1.0, cx + bw / 2 + marge)
                pct_y2 = min(1.0, cy + bh / 2 + marge)

                self.db.ajouter_zone_proposee(
                    session_id=self._apprentissage_session_id,
                    camera_id=self.camera_id,
                    pct_bbox=(pct_x1, pct_y1, pct_x2, pct_y2),
                    duree_sec=duree,
                    classe=obs[4],
                    confiance=obs[5],
                )
                nb_proposees += 1

        self.db.terminer_session_apprentissage(
            self._apprentissage_session_id, nb_proposees)
        logger.info(f"[{self.camera_nom}] Apprentissage termine: "
                     f"{nb_proposees} zone(s) proposee(s) sur "
                     f"{len(self._apprentissage_observations)} observations")

        self._apprentissage_actif = False
        self._apprentissage_session_id = None
        self._apprentissage_observations = {}

    def _est_immobile(self, piste) -> bool:
        """Detecte les objets immobiles par position (survit aux changements de track ID).
        Maintient une carte de zones ou des detections persistent sans bouger."""
        cx = (piste.bbox[0] + piste.bbox[2]) / 2
        cy = (piste.bbox[1] + piste.bbox[3]) / 2
        maintenant = time.time()
        # Chercher une zone statique proche
        for zone in self._zones_statiques:
            dist = ((cx - zone[0]) ** 2 + (cy - zone[1]) ** 2) ** 0.5
            if dist < self._seuil_deplacement_px:
                # Position connue — verifier la duree
                if (maintenant - zone[2]) >= self._seuil_immobilite_sec:
                    return True
                return False
        # Nouvelle position — enregistrer
        self._zones_statiques.append([cx, cy, maintenant])
        # Limiter a 20 zones max
        if len(self._zones_statiques) > 20:
            self._zones_statiques.pop(0)
        return False

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _maj_status(self, status: str):
        with self._lock_stats:
            self._status = status

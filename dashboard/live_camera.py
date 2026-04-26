"""
Module de visualisation camera en direct pour le dashboard Streamlit.
Capture des frames, inference YOLO + Pose, annotation et affichage.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Couleurs pour l'annotation (BGR)
COULEUR_PERSONNE = (0, 255, 0)       # Vert
COULEUR_OBJET = (255, 165, 0)        # Orange
COULEUR_VETEMENT = (255, 0, 200)     # Magenta/Rose - vetements detectes
COULEUR_POSE = (0, 255, 255)         # Jaune
COULEUR_ALERTE = (0, 0, 255)         # Rouge
COULEUR_TEXTE_BG = (0, 0, 0)         # Noir
COULEUR_ZONE = (200, 200, 200)       # Gris

# Connexions squelette COCO pour le dessin de la pose
CONNEXIONS_POSE = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Tete
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Bras
    (5, 11), (6, 12), (11, 12),                # Torse
    (11, 13), (13, 15), (12, 14), (14, 16),    # Jambes
]

NOMS_KEYPOINTS = [
    "Nez", "Oeil G", "Oeil D", "Oreille G", "Oreille D",
    "Epaule G", "Epaule D", "Coude G", "Coude D",
    "Poignet G", "Poignet D", "Hanche G", "Hanche D",
    "Genou G", "Genou D", "Cheville G", "Cheville D",
]


class CameraLiveViewer:
    """
    Capture et analyse des frames camera en direct.
    Utilise les detecteurs YOLO existants du projet.
    """

    def __init__(self, chemin_modeles: str = "./models", taille_entree_yolo: int = 320):
        """
        Initialise le viewer avec les modeles YOLO.

        Args:
            chemin_modeles: Repertoire contenant les fichiers ONNX
            taille_entree_yolo: Taille d'entree YOLO (320=rapide, 640=precis)
        """
        self.chemin_modeles = Path(chemin_modeles)
        self._taille_entree_yolo = taille_entree_yolo
        self._detecteur = None
        self._estimateur_pose = None
        self._detecteur_vetements = None
        self._detecteur_apprentissage = None  # OIV7 ~600 classes (lazy)
        self._captures: Dict[str, cv2.VideoCapture] = {}
        # Thread de grab continu pour RTSP
        self._derniere_frame: Dict[str, Optional[np.ndarray]] = {}
        self._grab_actif: Dict[str, bool] = {}
        self._grab_threads: Dict[str, threading.Thread] = {}
        self._frame_ready: Dict[str, threading.Event] = {}

    def _charger_detecteur(self):
        """Charge le detecteur YOLO (lazy loading)."""
        if self._detecteur is not None:
            return

        from app.detector import DetecteurPersonnes
        chemin = self.chemin_modeles / "yolov8n.onnx"
        if chemin.exists():
            self._detecteur = DetecteurPersonnes(chemin, confiance_min=0.45, taille_entree=self._taille_entree_yolo)
            logger.info("Detecteur YOLO charge pour le live viewer")
        else:
            logger.warning(f"Modele YOLO non trouve: {chemin}")

    def _charger_estimateur_pose(self):
        """Charge l'estimateur de pose (lazy loading)."""
        if self._estimateur_pose is not None:
            return

        from app.detector import EstimateurPose
        chemin = self.chemin_modeles / "yolov8n-pose.onnx"
        if chemin.exists():
            self._estimateur_pose = EstimateurPose(chemin, confiance_min=0.5, taille_entree=self._taille_entree_yolo)
            logger.info("Estimateur de pose charge pour le live viewer")
        else:
            logger.warning(f"Modele pose non trouve: {chemin}")

    def _charger_detecteur_apprentissage(self):
        """Charge le detecteur OIV7 pour l'apprentissage (lazy, optionnel).

        Le detecteur est cree meme si le fichier est absent — il ne chargera
        rien tant que personne n'appelle `detecter_tout()`. Cela evite les
        re-tentatives sur chaque invocation.
        """
        if self._detecteur_apprentissage is not None:
            return

        from app.detector import DetecteurApprentissage
        chemin = self.chemin_modeles / "yolov8n-oiv7.onnx"
        self._detecteur_apprentissage = DetecteurApprentissage(
            chemin_modele=chemin, confiance_min=0.25, taille_entree=640,
        )
        if chemin.exists():
            logger.info(f"Detecteur OIV7 pret pour le viewer (lazy): {chemin.name}")
        else:
            logger.info(
                f"Modele OIV7 absent ({chemin}). La capture 'detecter tout' "
                f"retombera sur COCO 80 classes."
            )

    def _charger_detecteur_vetements(self):
        """Charge le detecteur de vetements (lazy loading, optionnel)."""
        if self._detecteur_vetements is not None:
            return

        from app.detector import DetecteurVetements
        # Chercher le modele fashion dans le repertoire modeles
        for nom in ["yolov8n-fashion.onnx", "yolov8-fashion.onnx", "fashion_detector.onnx"]:
            chemin = self.chemin_modeles / nom
            if chemin.exists():
                self._detecteur_vetements = DetecteurVetements(chemin, confiance_min=0.40)
                logger.info(f"Detecteur vetements charge: {nom}")
                return
        # Creer un detecteur inactif (evite de re-chercher a chaque appel)
        self._detecteur_vetements = DetecteurVetements(
            self.chemin_modeles / "yolov8n-fashion.onnx", confiance_min=0.40
        )

    def ouvrir_camera(self, source: str) -> bool:
        """
        Ouvre une source video (RTSP, fichier, webcam).
        Pour RTSP, lance un thread de grab continu avec params optimises.
        """
        self._fermer_camera(source)

        try:
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            elif source.startswith(("rtsp://", "rtsps://")):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|stimeout;5000000"
                    "|probesize;1000000|analyzeduration;1000000"
                    "|max_delay;0|reorder_queue_size;0"
                    "|fflags;nobuffer+discardcorrupt|flags;low_delay"
                )
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                logger.warning(f"Impossible d'ouvrir la source: {source}")
                return False

            self._captures[source] = cap

            # RTSP : thread de grab continu (read + garde derniere frame)
            if source.startswith(("rtsp://", "rtsps://")):
                self._frame_ready[source] = threading.Event()
                self._derniere_frame[source] = None
                self._grab_actif[source] = True
                t = threading.Thread(target=self._boucle_grab, args=(source,), daemon=True)
                t.start()
                self._grab_threads[source] = t
                logger.info(f"Camera RTSP ouverte: {source}")
            else:
                logger.info(f"Camera ouverte: {source}")

            return True

        except Exception as e:
            logger.error(f"Erreur ouverture camera {source}: {e}")
            return False

    def _fermer_camera(self, source: str):
        """Ferme proprement une source camera et son thread de grab."""
        if source in self._grab_actif:
            self._grab_actif[source] = False
        if source in self._grab_threads:
            self._grab_threads[source].join(timeout=3)
            del self._grab_threads[source]
        self._derniere_frame.pop(source, None)
        self._grab_actif.pop(source, None)
        self._frame_ready.pop(source, None)
        if source in self._captures:
            self._captures[source].release()
            del self._captures[source]

    def _boucle_grab(self, source: str):
        """Thread qui lit le flux RTSP en continu, ne garde que la derniere frame."""
        cap = self._captures.get(source)
        if cap is None:
            return
        evt = self._frame_ready.get(source)
        while self._grab_actif.get(source, False):
            ret, frame = cap.read()
            if ret and frame is not None:
                self._derniere_frame[source] = frame
                if evt:
                    evt.set()
            else:
                time.sleep(0.01)

    def capturer_frame(self, source: str) -> Optional[np.ndarray]:
        """
        Capture une frame depuis la source.
        RTSP : retrieve() decode uniquement la derniere frame grabbed (la plus recente).
        """
        # Ouvrir si pas encore fait (premiere fois seulement, reste ouvert ensuite)
        if source not in self._captures:
            if not self.ouvrir_camera(source):
                return None

        # RTSP : derniere frame du grab thread
        if source in self._frame_ready:
            frame = self._derniere_frame.get(source)
            if frame is not None:
                return frame.copy()
            evt = self._frame_ready[source]
            if evt.wait(timeout=3.0):
                frame = self._derniere_frame.get(source)
                return frame.copy() if frame is not None else None
            logger.warning(f"Timeout capture RTSP: {source}")
            return None

        # Non-RTSP : lecture directe
        cap = self._captures[source]
        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        return frame

    def analyser_frame(
        self,
        frame: np.ndarray,
        detecter_objets: bool = True,
        estimer_pose: bool = True,
        confiance_min: float = 0.45,
        mode_tout_coco: bool = False,
        detecter_vetements: bool = False,
    ) -> Dict:
        """
        Analyse une frame avec YOLO, estimation de pose, et detection vetements.

        Args:
            frame: Image BGR
            detecter_objets: Activer la detection d'objets
            estimer_pose: Activer l'estimation de pose
            confiance_min: Seuil de confiance YOLO
            mode_tout_coco: Si True, detecte les 80 classes COCO (mode test)
            detecter_vetements: Si True, lance aussi le detecteur fashion

        Returns:
            Dictionnaire avec les resultats d'analyse
        """
        resultat = {
            "personnes": [],
            "objets": [],
            "vetements": [],
            "poses": [],
            "objets_par_personne": [],
            "temps_inference_ms": 0,
        }

        if not detecter_objets and not estimer_pose:
            return resultat

        self._charger_detecteur()
        if self._detecteur is None:
            return resultat

        seuil_original = self._detecteur.confiance_min
        self._detecteur.confiance_min = confiance_min

        t_start = time.time()

        if mode_tout_coco:
            # Priorite OIV7 (~600 classes, inclut caisse/imprimante/scanner).
            # Fallback COCO uniquement si OIV7 indisponible (fichier absent ou
            # echec de chargement), PAS si OIV7 a tourne et trouve 0 objet.
            self._charger_detecteur_apprentissage()
            toutes = []
            oiv7_utilise = False
            if self._detecteur_apprentissage is not None:
                toutes = self._detecteur_apprentissage.detecter_tout(
                    frame, confiance_min=confiance_min)
                oiv7_utilise = self._detecteur_apprentissage.actif
            if not oiv7_utilise:
                toutes = self._detecteur.detecter_tout_coco(frame)
            # OIV7 "Person" est traduit en "personne" (cf. TRADUCTIONS_OIV7_FR)
            # donc filtrer par class_name marche pour les 2 modeles.
            resultat["personnes"] = [d for d in toutes if d.class_name == "personne"]
            resultat["objets"] = [d for d in toutes if d.class_name != "personne"]
        else:
            personnes, objets = self._detecteur.detecter_personnes_et_objets(frame)
            resultat["personnes"] = personnes
            resultat["objets"] = objets

        # Detection de vetements (modele fashion optionnel)
        if detecter_vetements:
            self._charger_detecteur_vetements()
            if self._detecteur_vetements and self._detecteur_vetements.actif:
                vetements = self._detecteur_vetements.detecter(frame)
                resultat["vetements"] = vetements

        # Estimation de pose
        if estimer_pose and resultat["personnes"]:
            self._charger_estimateur_pose()
            if self._estimateur_pose is not None:
                bboxes = [p.bbox for p in resultat["personnes"][:5]]
                poses = self._estimateur_pose.estimer_poses_multiples(frame, bboxes)
                resultat["poses"] = poses

        # Associer les objets + vetements aux personnes
        tous_objets = resultat["objets"] + resultat["vetements"]
        if resultat["personnes"] and tous_objets:
            resultat["objets_par_personne"] = self._associer_objets_personnes(
                resultat["personnes"], tous_objets
            )

        resultat["temps_inference_ms"] = round((time.time() - t_start) * 1000, 1)

        self._detecteur.confiance_min = seuil_original

        return resultat

    @staticmethod
    def _calculer_iou(bbox1: Tuple, bbox2: Tuple) -> float:
        """Calcule l'Intersection over Union entre deux bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        aire1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        aire2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = aire1 + aire2 - intersection

        return intersection / max(union, 1)

    @staticmethod
    def _calculer_contenance(bbox_objet: Tuple, bbox_personne: Tuple) -> float:
        """Calcule le % de l'objet contenu dans la bbox de la personne."""
        x1 = max(bbox_objet[0], bbox_personne[0])
        y1 = max(bbox_objet[1], bbox_personne[1])
        x2 = min(bbox_objet[2], bbox_personne[2])
        y2 = min(bbox_objet[3], bbox_personne[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        aire_objet = (bbox_objet[2] - bbox_objet[0]) * (bbox_objet[3] - bbox_objet[1])

        return intersection / max(aire_objet, 1)

    def _associer_objets_personnes(
        self,
        personnes: List,
        objets: List,
    ) -> List[Dict]:
        """
        Associe chaque objet a la personne la plus proche.
        Un objet est 'porte' si >30% de sa bbox est contenu dans la bbox personne.

        Returns:
            Liste de dicts: [{personne_idx, personne_bbox, objets_portes: [...]}]
        """
        associations = []

        for i, pers in enumerate(personnes):
            objets_portes = []
            px1, py1, px2, py2 = pers.bbox
            # Elargir la bbox personne de 15% pour capter les objets proches
            marge_x = int((px2 - px1) * 0.15)
            marge_y = int((py2 - py1) * 0.10)
            bbox_elargie = (px1 - marge_x, py1 - marge_y, px2 + marge_x, py2 + marge_y)

            for obj in objets:
                contenance = self._calculer_contenance(obj.bbox, bbox_elargie)
                if contenance > 0.30:  # >30% de l'objet est dans/sur la personne
                    # Determiner la position relative sur le corps
                    obj_cy = (obj.bbox[1] + obj.bbox[3]) / 2
                    pers_h = py2 - py1
                    ratio_y = (obj_cy - py1) / max(pers_h, 1)

                    if ratio_y < 0.3:
                        position = "tete/epaules"
                    elif ratio_y < 0.6:
                        position = "torse/bras"
                    elif ratio_y < 0.85:
                        position = "hanches/jambes"
                    else:
                        position = "pieds"

                    objets_portes.append({
                        "nom": obj.class_name,
                        "confiance": obj.confidence,
                        "position": position,
                        "contenance": contenance,
                        "bbox": obj.bbox,
                    })

            associations.append({
                "personne_idx": i + 1,
                "personne_bbox": pers.bbox,
                "personne_confiance": pers.confidence,
                "objets_portes": objets_portes,
            })

        return associations

    def annoter_frame(
        self,
        frame: np.ndarray,
        resultat: Dict,
        afficher_boites: bool = True,
        afficher_poses: bool = True,
        afficher_labels: bool = True,
        afficher_confiance: bool = True,
        afficher_zones: bool = False,
        afficher_objets_portes: bool = False,
    ) -> np.ndarray:
        """
        Annote une frame avec les resultats de detection.

        Args:
            frame: Image BGR originale
            resultat: Resultats de analyser_frame()
            afficher_boites: Dessiner les boites englobantes
            afficher_poses: Dessiner le squelette de pose
            afficher_labels: Afficher les noms des classes
            afficher_confiance: Afficher les scores de confiance
            afficher_zones: Dessiner la grille des 9 zones du magasin

        Returns:
            Frame annotee (copie)
        """
        annotee = frame.copy()
        h, w = annotee.shape[:2]

        # Dessiner les zones du magasin
        if afficher_zones:
            self._dessiner_zones(annotee, w, h)

        # Dessiner les personnes
        if afficher_boites:
            for i, det in enumerate(resultat.get("personnes", [])):
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotee, (x1, y1), (x2, y2), COULEUR_PERSONNE, 2)

                if afficher_labels:
                    label = det.class_name
                    if afficher_confiance:
                        label += f" {det.confidence:.0%}"
                    self._dessiner_label(annotee, label, x1, y1 - 5, COULEUR_PERSONNE)

            # Dessiner les objets
            for det in resultat.get("objets", []):
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotee, (x1, y1), (x2, y2), COULEUR_OBJET, 2)

                if afficher_labels:
                    label = det.class_name
                    if afficher_confiance:
                        label += f" {det.confidence:.0%}"
                    self._dessiner_label(annotee, label, x1, y1 - 5, COULEUR_OBJET)

            # Dessiner les vetements detectes
            for det in resultat.get("vetements", []):
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotee, (x1, y1), (x2, y2), COULEUR_VETEMENT, 2)
                # Ligne pointillee pour distinguer des objets COCO
                for px in range(x1, x2, 8):
                    cv2.line(annotee, (px, y1), (min(px + 4, x2), y1), COULEUR_VETEMENT, 2)
                    cv2.line(annotee, (px, y2), (min(px + 4, x2), y2), COULEUR_VETEMENT, 2)

                if afficher_labels:
                    label = f"👕 {det.class_name}"
                    if afficher_confiance:
                        label += f" {det.confidence:.0%}"
                    self._dessiner_label(annotee, label, x1, y1 - 5, COULEUR_VETEMENT)

        # Dessiner les poses
        if afficher_poses:
            for pose in resultat.get("poses", []):
                if pose is not None:
                    self._dessiner_pose(annotee, pose)

        # Dessiner les lignes d'association objet-personne
        if afficher_objets_portes:
            for assoc in resultat.get("objets_par_personne", []):
                pidx = assoc["personne_idx"]
                px1, py1, px2, py2 = assoc["personne_bbox"]
                pers_cx = (px1 + px2) // 2

                for obj_info in assoc["objets_portes"]:
                    ox1, oy1, ox2, oy2 = obj_info["bbox"]
                    obj_cx = (ox1 + ox2) // 2
                    obj_cy = (oy1 + oy2) // 2

                    # Ligne reliant l'objet a la personne
                    cv2.line(annotee, (pers_cx, py1), (obj_cx, obj_cy),
                             (255, 0, 255), 2, cv2.LINE_AA)

                    # Label de l'objet porte
                    label_porte = f"{obj_info['nom']} ({obj_info['position']})"
                    self._dessiner_label(annotee, label_porte, ox1, oy1 - 18, (255, 0, 255))

        # Barre d'information en haut
        nb_personnes = len(resultat.get("personnes", []))
        nb_objets = len(resultat.get("objets", []))
        temps = resultat.get("temps_inference_ms", 0)
        info = f"Personnes: {nb_personnes} | Objets: {nb_objets} | Inference: {temps}ms"

        cv2.rectangle(annotee, (0, 0), (w, 30), COULEUR_TEXTE_BG, -1)
        cv2.putText(annotee, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return annotee

    def _dessiner_label(self, frame: np.ndarray, texte: str, x: int, y: int, couleur: tuple):
        """Dessine un label avec fond sur la frame."""
        (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 4, y)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y), couleur, -1)
        cv2.putText(frame, texte, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _dessiner_pose(self, frame: np.ndarray, pose):
        """Dessine le squelette de pose sur la frame."""
        kp = pose.keypoints

        # Dessiner les connexions
        for (i, j) in CONNEXIONS_POSE:
            if kp[i, 2] > 0.3 and kp[j, 2] > 0.3:
                pt1 = (int(kp[i, 0]), int(kp[i, 1]))
                pt2 = (int(kp[j, 0]), int(kp[j, 1]))
                cv2.line(frame, pt1, pt2, COULEUR_POSE, 2)

        # Dessiner les points cles
        for idx in range(17):
            if kp[idx, 2] > 0.3:
                cx, cy = int(kp[idx, 0]), int(kp[idx, 1])
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 4, COULEUR_POSE, 1)

    def _dessiner_zones(self, frame: np.ndarray, w: int, h: int):
        """Dessine la grille des 9 zones du magasin."""
        # Lignes horizontales (30%, 70%)
        y1 = int(h * 0.30)
        y2 = int(h * 0.70)
        cv2.line(frame, (0, y1), (w, y1), COULEUR_ZONE, 1, cv2.LINE_AA)
        cv2.line(frame, (0, y2), (w, y2), COULEUR_ZONE, 1, cv2.LINE_AA)

        # Lignes verticales (33%, 66%)
        x1 = int(w * 0.33)
        x2 = int(w * 0.66)
        cv2.line(frame, (x1, 0), (x1, h), COULEUR_ZONE, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, 0), (x2, h), COULEUR_ZONE, 1, cv2.LINE_AA)

        # Labels des zones
        noms_zones = [
            ("Rayon A", 0.16, 0.15), ("Rayon B", 0.50, 0.15), ("Rayon C", 0.83, 0.15),
            ("Allee G", 0.16, 0.50), ("Allee C", 0.50, 0.50), ("Allee D", 0.83, 0.50),
            ("Entree", 0.16, 0.85), ("Caisse", 0.50, 0.85), ("Sortie", 0.83, 0.85),
        ]
        for nom, fx, fy in noms_zones:
            cx, cy = int(w * fx), int(h * fy)
            (tw, th), _ = cv2.getTextSize(nom, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, nom, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, COULEUR_ZONE, 1, cv2.LINE_AA)

    def generer_rapport_detections(self, resultat: Dict) -> List[Dict]:
        """
        Genere un rapport structure des detections pour l'affichage tableau.

        Args:
            resultat: Resultats de analyser_frame()

        Returns:
            Liste de dictionnaires pour affichage DataFrame
        """
        rapport = []

        for det in resultat.get("personnes", []):
            x1, y1, x2, y2 = det.bbox
            rapport.append({
                "Type": "Personne",
                "Classe": det.class_name,
                "Confiance": f"{det.confidence:.0%}",
                "Position": f"({x1},{y1})-({x2},{y2})",
                "Taille": f"{x2-x1}x{y2-y1} px",
            })

        for det in resultat.get("objets", []):
            x1, y1, x2, y2 = det.bbox
            rapport.append({
                "Type": "Objet",
                "Classe": det.class_name,
                "Confiance": f"{det.confidence:.0%}",
                "Position": f"({x1},{y1})-({x2},{y2})",
                "Taille": f"{x2-x1}x{y2-y1} px",
            })

        for det in resultat.get("vetements", []):
            x1, y1, x2, y2 = det.bbox
            rapport.append({
                "Type": "Vetement",
                "Classe": det.class_name,
                "Confiance": f"{det.confidence:.0%}",
                "Position": f"({x1},{y1})-({x2},{y2})",
                "Taille": f"{x2-x1}x{y2-y1} px",
            })

        return rapport

    def generer_rapport_pose(self, resultat: Dict) -> List[Dict]:
        """
        Genere un rapport des keypoints de pose.

        Returns:
            Liste de dicts pour affichage DataFrame
        """
        rapport = []

        for i, pose in enumerate(resultat.get("poses", [])):
            if pose is None:
                continue
            for idx in range(17):
                if pose.keypoints[idx, 2] > 0.3:
                    rapport.append({
                        "Personne": i + 1,
                        "Keypoint": NOMS_KEYPOINTS[idx],
                        "X": int(pose.keypoints[idx, 0]),
                        "Y": int(pose.keypoints[idx, 1]),
                        "Confiance": f"{pose.keypoints[idx, 2]:.0%}",
                    })

        return rapport

    def analyser_comportement_suspect(self, resultat: Dict) -> List[Dict]:
        """
        Analyse les comportements suspects sur une photo unique.
        Combine pose (position des mains) + objets detectes pour identifier :
        - Main vers le torse/poche (dissimulation dans vetement)
        - Main dans/vers un sac (transfert d'article)
        - Objet cache sous vetement (main tenant un objet pres du corps)
        - Posture de dissimulation (bras replies, corps penche)

        Args:
            resultat: Resultats de analyser_frame() (avec poses et objets)

        Returns:
            Liste d'alertes suspectes avec score, description et type
        """
        alertes = []
        personnes = resultat.get("personnes", [])
        objets = resultat.get("objets", [])
        poses = resultat.get("poses", [])
        objets_par_personne = resultat.get("objets_par_personne", [])

        for i, pers in enumerate(personnes):
            pose = poses[i] if i < len(poses) and poses[i] is not None else None
            px1, py1, px2, py2 = pers.bbox
            pers_w = px2 - px1
            pers_h = py2 - py1

            # Objets portes par cette personne
            objets_pers = []
            if i < len(objets_par_personne):
                objets_pers = objets_par_personne[i].get("objets_portes", [])

            # Noms des sacs portes
            sacs = [o for o in objets_pers if o["nom"] in (
                "sac_a_main", "sac_a_dos", "valise", "sac_a_main"
            )]

            if pose is None:
                # Sans pose, analyse basique par objets
                if sacs:
                    for sac in sacs:
                        if sac["position"] in ("torse/bras", "hanches/jambes"):
                            alertes.append({
                                "personne": i + 1,
                                "type": "Sac en position suspecte",
                                "description": (
                                    f"{sac['nom'].replace('_', ' ')} au niveau "
                                    f"{sac['position']} (recouvrement {sac['contenance']:.0%})"
                                ),
                                "score": 0.5,
                                "severite": "MOYENNE",
                            })
                continue

            kp = pose.keypoints  # (17, 3) - x, y, confiance

            # --- ANALYSE 1 : Main proche du torse (dissimulation dans vetement/poche) ---
            centre_torse = pose.obtenir_centre_torse()
            if centre_torse is not None:
                main_g, main_d = pose.obtenir_position_mains()

                for main, cote in [(main_g, "gauche"), (main_d, "droite")]:
                    if main is None:
                        continue

                    # Distance main-torse normalisee par la taille de la personne
                    dist = np.linalg.norm(main - centre_torse) / max(pers_h, 1)

                    if dist < 0.12:
                        # Main tres proche du torse = tres suspect
                        alertes.append({
                            "personne": i + 1,
                            "type": "Main contre le torse",
                            "description": (
                                f"Main {cote} collee au torse (distance: {dist:.2f}). "
                                f"Possible dissimulation dans vetement ou poche."
                            ),
                            "score": 0.85,
                            "severite": "HAUTE",
                        })
                    elif dist < 0.20:
                        alertes.append({
                            "personne": i + 1,
                            "type": "Main proche du torse",
                            "description": (
                                f"Main {cote} pres du torse (distance: {dist:.2f}). "
                                f"Geste potentiel de dissimulation."
                            ),
                            "score": 0.6,
                            "severite": "MOYENNE",
                        })

            # --- ANALYSE 2 : Main dans un sac (transfert d'article) ---
            for sac in sacs:
                sx1, sy1, sx2, sy2 = sac["bbox"]
                sac_cx = (sx1 + sx2) / 2
                sac_cy = (sy1 + sy2) / 2

                for main, cote in [(main_g, "gauche"), (main_d, "droite")]:
                    if main is None:
                        continue

                    # Distance main - centre du sac, normalisee
                    dist_sac = np.sqrt(
                        (main[0] - sac_cx) ** 2 + (main[1] - sac_cy) ** 2
                    ) / max(pers_h, 1)

                    # Main dans le sac (la main est a l'interieur de la bbox du sac)
                    main_dans_sac = (
                        sx1 <= main[0] <= sx2 and sy1 <= main[1] <= sy2
                    )

                    if main_dans_sac:
                        alertes.append({
                            "personne": i + 1,
                            "type": "Main dans le sac",
                            "description": (
                                f"Main {cote} a l'interieur du {sac['nom'].replace('_', ' ')}. "
                                f"Possible transfert d'article vole."
                            ),
                            "score": 0.90,
                            "severite": "HAUTE",
                        })
                    elif dist_sac < 0.15:
                        alertes.append({
                            "personne": i + 1,
                            "type": "Main proche du sac",
                            "description": (
                                f"Main {cote} tres proche du {sac['nom'].replace('_', ' ')} "
                                f"(distance: {dist_sac:.2f})."
                            ),
                            "score": 0.65,
                            "severite": "MOYENNE",
                        })

            # --- ANALYSE 3 : Bras replies / posture fermee (dissimulation) ---
            coude_g = kp[7] if kp[7, 2] > 0.3 else None  # Coude gauche
            coude_d = kp[8] if kp[8, 2] > 0.3 else None  # Coude droit
            epaule_g = kp[5] if kp[5, 2] > 0.3 else None
            epaule_d = kp[6] if kp[6, 2] > 0.3 else None

            bras_replies = 0
            for coude, epaule, main, cote in [
                (coude_g, epaule_g, main_g, "gauche"),
                (coude_d, epaule_d, main_d, "droite"),
            ]:
                if coude is None or epaule is None or main is None:
                    continue

                # Angle du bras : si le poignet est plus haut que le coude
                # et le coude est plie (main remonte vers l'epaule)
                ep = epaule[:2]
                co = coude[:2]

                # Vecteur bras superieur et avant-bras
                v_sup = co - ep
                v_inf = main - co

                # Angle entre les deux vecteurs
                cos_angle = np.dot(v_sup, v_inf) / (
                    max(np.linalg.norm(v_sup), 0.01) * max(np.linalg.norm(v_inf), 0.01)
                )
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                if angle_deg < 60:  # Bras tres replie
                    bras_replies += 1

            if bras_replies >= 2:
                alertes.append({
                    "personne": i + 1,
                    "type": "Posture de dissimulation",
                    "description": (
                        "Les deux bras sont replies contre le corps. "
                        "Posture typique de dissimulation d'article."
                    ),
                    "score": 0.75,
                    "severite": "HAUTE",
                })
            elif bras_replies == 1:
                alertes.append({
                    "personne": i + 1,
                    "type": "Bras replie suspect",
                    "description": (
                        "Un bras replie contre le corps. "
                        "Possible dissimulation d'article sous le vetement."
                    ),
                    "score": 0.50,
                    "severite": "MOYENNE",
                })

            # --- ANALYSE 4 : Main sous le vetement (ouverture veste) ---
            # Si le poignet est entre les epaules (horizontalement) et au niveau du torse
            if centre_torse is not None:
                for main, cote in [(main_g, "gauche"), (main_d, "droite")]:
                    if main is None:
                        continue

                    # Verifier si la main est dans la zone "interieur veste"
                    # = entre les 2 epaules en X, entre epaules et hanches en Y
                    ep_g_x = kp[5, 0] if kp[5, 2] > 0.3 else px1
                    ep_d_x = kp[6, 0] if kp[6, 2] > 0.3 else px2
                    ep_y = min(kp[5, 1], kp[6, 1]) if kp[5, 2] > 0.3 and kp[6, 2] > 0.3 else py1 + pers_h * 0.25
                    hanche_y = max(kp[11, 1], kp[12, 1]) if kp[11, 2] > 0.3 and kp[12, 2] > 0.3 else py1 + pers_h * 0.65

                    main_entre_epaules = min(ep_g_x, ep_d_x) < main[0] < max(ep_g_x, ep_d_x)
                    main_zone_torse = ep_y < main[1] < hanche_y

                    if main_entre_epaules and main_zone_torse:
                        dist_centre = np.linalg.norm(main - centre_torse) / max(pers_h, 1)
                        if dist_centre < 0.15:
                            alertes.append({
                                "personne": i + 1,
                                "type": "Main sous le vetement",
                                "description": (
                                    f"Main {cote} positionnee a l'interieur de la zone du torse, "
                                    f"entre les epaules. Typique de cacher un article dans la veste."
                                ),
                                "score": 0.80,
                                "severite": "HAUTE",
                            })

        # Dedoublonner : garder le score le plus haut par (personne, type)
        seen = {}
        alertes_uniques = []
        for a in sorted(alertes, key=lambda x: -x["score"]):
            cle = (a["personne"], a["type"])
            if cle not in seen:
                seen[cle] = True
                alertes_uniques.append(a)

        return alertes_uniques

    def annoter_alertes_suspect(
        self, frame: np.ndarray, alertes: List[Dict], personnes: List
    ) -> np.ndarray:
        """Annote la frame avec les alertes de comportement suspect."""
        annotee = frame.copy()
        h, w = annotee.shape[:2]

        for alerte in alertes:
            pidx = alerte["personne"] - 1
            if pidx >= len(personnes):
                continue

            px1, py1, px2, py2 = personnes[pidx].bbox
            score = alerte["score"]
            sev = alerte["severite"]

            # Couleur selon la severite
            couleur = (0, 0, 255) if sev == "HAUTE" else (0, 165, 255)

            # Bordure d'alerte autour de la personne
            epaisseur = 3 if sev == "HAUTE" else 2
            cv2.rectangle(annotee, (px1, py1), (px2, py2), couleur, epaisseur)

            # Icone alerte en haut a gauche de la bbox
            cv2.circle(annotee, (px1 + 12, py1 + 12), 12, couleur, -1)
            cv2.putText(annotee, "!", (px1 + 7, py1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Label de l'alerte sous la bbox
            label = f"{alerte['type']} ({score:.0%})"
            self._dessiner_label(annotee, label, px1, py2 + 15, couleur)

        # Banniere rouge si alertes haute severite
        alertes_hautes = [a for a in alertes if a["severite"] == "HAUTE"]
        if alertes_hautes:
            cv2.rectangle(annotee, (0, h - 35), (w, h), (0, 0, 200), -1)
            texte = f"ALERTE: {len(alertes_hautes)} comportement(s) suspect(s) detecte(s)"
            cv2.putText(annotee, texte, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotee

    def fermer_camera(self, source: str):
        """Ferme une capture specifique."""
        if source in self._captures:
            self._captures[source].release()
            del self._captures[source]

    def fermer_tout(self):
        """Ferme toutes les captures ouvertes."""
        # Arreter tous les threads de grab
        for source in list(self._grab_actif.keys()):
            self._grab_actif[source] = False
        for source, t in list(self._grab_threads.items()):
            t.join(timeout=2)
        self._grab_threads.clear()
        self._grab_actif.clear()
        self._derniere_frame.clear()
        # Arreter le serveur MJPEG
        self.arreter_mjpeg()
        # Liberer les captures
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()

    # ================================================================
    # Serveur MJPEG pour flux live zero-lag
    # ================================================================
    _mjpeg_serveur = None
    _mjpeg_thread = None
    _mjpeg_source = None
    _mjpeg_params = None
    _mjpeg_stats = {"frame": 0, "fps": 0.0, "inference_ms": 0, "personnes": 0, "objets": 0, "vetements": 0}

    def demarrer_mjpeg(self, source: str, port: int = 8555, params: dict = None):
        """
        Demarre un serveur MJPEG sur le port specifie.
        Le navigateur se connecte directement au flux — zero lag Streamlit.
        """
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler

        # Arreter le serveur precedent si actif
        self.arreter_mjpeg()

        # S'assurer que la camera est ouverte
        if source not in self._captures:
            if not self.ouvrir_camera(source):
                logger.error(f"MJPEG: impossible d'ouvrir {source}")
                return False

        self._mjpeg_source = source
        self._mjpeg_params = params or {}
        self._mjpeg_stats = {"frame": 0, "fps": 0.0, "inference_ms": 0, "personnes": 0, "objets": 0, "vetements": 0}
        viewer_ref = self

        class MJPEGHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/stream":
                    self._handle_stream()
                elif self.path == "/stats":
                    self._handle_stats()
                else:
                    self.send_error(404)

            def _handle_stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                params = viewer_ref._mjpeg_params
                src = viewer_ref._mjpeg_source
                compteur = 0
                t_debut = time.time()
                dernier_resultat = None  # Cache des detections YOLO
                inference_chaque_n = 3   # YOLO toutes les N frames (les autres = passthrough)

                while viewer_ref._mjpeg_serveur is not None:
                    try:
                        frame = viewer_ref.capturer_frame(src)
                        if frame is None:
                            time.sleep(0.03)
                            continue

                        compteur += 1
                        nb_pers = nb_obj = nb_vet = 0
                        t_inf = 0

                        if params.get("detections", True):
                            # Inference YOLO uniquement toutes les N frames
                            if compteur % inference_chaque_n == 1 or dernier_resultat is None:
                                resultat = viewer_ref.analyser_frame(
                                    frame, detecter_objets=True,
                                    estimer_pose=params.get("pose", False),
                                    confiance_min=params.get("confiance", 0.3),
                                    mode_tout_coco=params.get("tout_coco", True),
                                    detecter_vetements=params.get("vetements", False),
                                )
                                dernier_resultat = resultat
                                t_inf = resultat["temps_inference_ms"]
                            else:
                                resultat = dernier_resultat

                            frame = viewer_ref.annoter_frame(
                                frame, resultat, afficher_boites=True,
                                afficher_poses=params.get("pose", False),
                                afficher_labels=True, afficher_confiance=True,
                                afficher_objets_portes=params.get("objets_portes", False),
                            )
                            nb_pers = len(resultat["personnes"])
                            nb_obj = len(resultat["objets"])
                            nb_vet = len(resultat.get("vetements", []))

                        # Overlay stats
                        fps = compteur / max(time.time() - t_debut, 0.001)
                        txt = f"FPS:{fps:.1f} | Inf:{t_inf}ms | P:{nb_pers} O:{nb_obj}"
                        if nb_vet > 0:
                            txt += f" V:{nb_vet}"
                        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        viewer_ref._mjpeg_stats.update({
                            "frame": compteur, "fps": round(fps, 1),
                            "inference_ms": t_inf, "personnes": nb_pers,
                            "objets": nb_obj, "vetements": nb_vet,
                        })

                        # JPEG qualite reduite pour streaming rapide
                        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(buf.tobytes())
                        self.wfile.write(b"\r\n")

                    except (BrokenPipeError, ConnectionResetError):
                        break
                    except Exception as e:
                        logger.error(f"MJPEG stream error: {e}")
                        break

            def _handle_stats(self):
                import json
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(viewer_ref._mjpeg_stats).encode())

            def log_message(self, format, *args):
                pass  # Silence les logs HTTP

        try:
            self._mjpeg_serveur = HTTPServer(("0.0.0.0", port), MJPEGHandler)
            self._mjpeg_thread = threading.Thread(target=self._mjpeg_serveur.serve_forever, daemon=True)
            self._mjpeg_thread.start()
            logger.info(f"Serveur MJPEG demarre sur le port {port} pour {source}")
            return True
        except Exception as e:
            logger.error(f"Impossible de demarrer le serveur MJPEG: {e}")
            self._mjpeg_serveur = None
            return False

    def arreter_mjpeg(self):
        """Arrete le serveur MJPEG."""
        if self._mjpeg_serveur is not None:
            self._mjpeg_serveur.shutdown()
            self._mjpeg_serveur = None
            self._mjpeg_source = None
            logger.info("Serveur MJPEG arrete")

    @property
    def classes_detectables(self) -> Dict[int, str]:
        """Retourne les classes COCO detectables."""
        from app.detector import DetecteurPersonnes
        return DetecteurPersonnes.CLASSES_PERTINENTES.copy()

    @property
    def classes_vetements(self) -> Dict[int, str]:
        """Retourne les classes de vetements detectables."""
        from app.detector import DetecteurVetements
        return DetecteurVetements.CLASSES_VETEMENTS.copy()

    @property
    def detecteur_vetements_actif(self) -> bool:
        """Verifie si le detecteur de vetements est disponible."""
        self._charger_detecteur_vetements()
        return self._detecteur_vetements is not None and self._detecteur_vetements.actif

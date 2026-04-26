"""
Détecteur de personnes et d'objets basé sur YOLO (ONNX).
Supporte OpenVINO (Intel GPU/CPU) avec fallback ONNX CPU.
Inclut l'estimation de pose pour l'analyse comportementale.
Inclut la détection de vêtements (modèle fashion optionnel).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def creer_session_onnx(chemin_modele: str, nom_modele: str = "modele") -> "ort.InferenceSession":
    """Cree une session ONNX en essayant OpenVINO d'abord, puis CPU.

    Detecte automatiquement le nombre de threads optimal.
    """
    import onnxruntime as ort

    # Threads auto: P-cores sont les plus efficaces pour l'inference
    nb_cpus = os.cpu_count() or 4
    intra_threads = max(2, nb_cpus // 2)  # moitie des threads logiques
    inter_threads = 2

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = inter_threads
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True

    # Essayer OpenVINO d'abord (Intel GPU/CPU), puis CPU seul
    providers_disponibles = ort.get_available_providers()

    if "OpenVINOExecutionProvider" in providers_disponibles:
        try:
            session = ort.InferenceSession(
                chemin_modele,
                sess_options=options,
                providers=[
                    ("OpenVINOExecutionProvider", {"device_type": "GPU_FP16"}),
                    "CPUExecutionProvider",
                ],
            )
            logger.info(f"{nom_modele}: OpenVINO GPU actif (threads={intra_threads})")
            return session
        except Exception:
            # GPU pas dispo, essayer CPU OpenVINO
            try:
                session = ort.InferenceSession(
                    chemin_modele,
                    sess_options=options,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "CPU_FP32"}),
                        "CPUExecutionProvider",
                    ],
                )
                logger.info(f"{nom_modele}: OpenVINO CPU actif (threads={intra_threads})")
                return session
            except Exception as e:
                logger.debug(f"{nom_modele}: OpenVINO CPU echec: {e}")

    # Fallback: CPU pur
    session = ort.InferenceSession(
        chemin_modele,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    logger.info(f"{nom_modele}: ONNX CPU (threads={intra_threads})")
    return session


def _letterbox_yolo(
    frame: np.ndarray, taille_entree: int
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox YOLO: redimensionne en preservant le ratio et padde a taille_entree.

    Convention: fond gris 114, normalise float32 [0,1], transpose BCHW.

    Returns:
        (image_norm_bchw, ratio, (pad_x, pad_y))
    """
    hauteur, largeur = frame.shape[:2]
    ratio = min(taille_entree / hauteur, taille_entree / largeur)
    nw = int(largeur * ratio)
    nh = int(hauteur * ratio)
    image_redim = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image_pad = np.full((taille_entree, taille_entree, 3), 114, dtype=np.uint8)
    pad_x = (taille_entree - nw) // 2
    pad_y = (taille_entree - nh) // 2
    image_pad[pad_y:pad_y + nh, pad_x:pad_x + nw] = image_redim
    # Fusion normalize + transpose en un seul tampon CHW contigu (1 alloc au lieu de 2).
    # Multiplication par reciproque plus rapide que la division par 255.0.
    image_norm = np.ascontiguousarray(
        image_pad.transpose(2, 0, 1)
    ).astype(np.float32) * (1.0 / 255.0)
    return image_norm[None], ratio, (pad_x, pad_y)


def _decoder_sortie_yolo(
    sortie: np.ndarray,
    ratio: float,
    padding: Tuple[int, int],
    taille_originale: Tuple[int, int],
    seuil_confiance: float,
    seuil_nms: float = 0.45,
    nb_classes: Optional[int] = None,
    filtre_classes: Optional[set] = None,
    taille_bbox_min: int = 10,
) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
    """Decode sortie YOLO (format ligne = cx,cy,w,h,scores...) + applique NMS.

    Args:
        sortie: Sortie brute du modele (1,C,N) ou (N,C).
        ratio, padding, taille_originale: Parametres letterbox inverse.
        seuil_confiance: Filtre les detections sous ce score (et seuil NMS).
        seuil_nms: IoU threshold pour cv2.dnn.NMSBoxes.
        nb_classes: Si fourni, ne regarde que ligne[4:4+nb_classes] (utile pour
            modeles fashion ou un modele a moins de classes que la sortie).
        filtre_classes: Si fourni, rejette les class_ids absents du set.
        taille_bbox_min: Rejette les boites < ce nombre de pixels (largeur ou hauteur).

    Returns:
        Liste de tuples ((x1, y1, x2, y2), score, class_id) apres NMS, en
        coordonnees image originale. Le caller construit les Detection avec
        ses propres class_name/categorie.
    """
    if sortie.ndim == 3:
        sortie = sortie[0]
    if sortie.shape[0] < sortie.shape[1]:
        sortie = sortie.T

    hauteur_orig, largeur_orig = taille_originale
    pad_x, pad_y = padding

    # Vectorise: une seule passe sur les ~8400 lignes au lieu d'une boucle Python.
    scores_classes = sortie[:, 4:4 + nb_classes] if nb_classes is not None else sortie[:, 4:]
    if scores_classes.shape[1] == 0:
        return []
    score_max = scores_classes.max(axis=1)
    masque_conf = score_max >= seuil_confiance
    if not masque_conf.any():
        return []

    rows = sortie[masque_conf]
    score_max = score_max[masque_conf]
    class_ids = scores_classes[masque_conf].argmax(axis=1)

    if filtre_classes is not None:
        masque_classe = np.isin(class_ids, np.fromiter(filtre_classes, dtype=class_ids.dtype))
        if not masque_classe.any():
            return []
        rows = rows[masque_classe]
        score_max = score_max[masque_classe]
        class_ids = class_ids[masque_classe]

    cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    x1 = np.clip(((cx - w / 2 - pad_x) / ratio).astype(np.int32), 0, largeur_orig)
    y1 = np.clip(((cy - h / 2 - pad_y) / ratio).astype(np.int32), 0, hauteur_orig)
    x2 = np.clip(((cx + w / 2 - pad_x) / ratio).astype(np.int32), 0, largeur_orig)
    y2 = np.clip(((cy + h / 2 - pad_y) / ratio).astype(np.int32), 0, hauteur_orig)
    bw = x2 - x1
    bh = y2 - y1
    masque_taille = (bw >= taille_bbox_min) & (bh >= taille_bbox_min)
    if not masque_taille.any():
        return []

    x1, y1, bw, bh = x1[masque_taille], y1[masque_taille], bw[masque_taille], bh[masque_taille]
    score_max = score_max[masque_taille]
    class_ids = class_ids[masque_taille]

    # cv2.dnn.NMSBoxes attend des listes Python (boites en xywh).
    boites = np.column_stack((x1, y1, bw, bh)).tolist()
    scores = score_max.astype(float).tolist()

    indices = cv2.dnn.NMSBoxes(boites, scores, seuil_confiance, seuil_nms)
    if len(indices) == 0:
        return []

    indices = indices.flatten()
    resultats: List[Tuple[Tuple[int, int, int, int], float, int]] = [
        ((int(x1[i]), int(y1[i]), int(x1[i] + bw[i]), int(y1[i] + bh[i])),
         float(score_max[i]), int(class_ids[i]))
        for i in indices
    ]
    return resultats


@dataclass
class Detection:
    """Représente une détection (personne ou objet)."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    categorie: str = "objet"  # "personne", "objet", "vetement"


@dataclass
class PoseKeypoints:
    """Points clés de pose COCO (17 points)."""
    # Format: tableau numpy (17, 3) -> x, y, confiance pour chaque point
    keypoints: np.ndarray
    confidence: float

    # Indices des points clés COCO
    NEZ = 0
    OEIL_GAUCHE = 1
    OEIL_DROIT = 2
    OREILLE_GAUCHE = 3
    OREILLE_DROIT = 4
    EPAULE_GAUCHE = 5
    EPAULE_DROITE = 6
    COUDE_GAUCHE = 7
    COUDE_DROIT = 8
    POIGNET_GAUCHE = 9
    POIGNET_DROIT = 10
    HANCHE_GAUCHE = 11
    HANCHE_DROITE = 12
    GENOU_GAUCHE = 13
    GENOU_DROIT = 14
    CHEVILLE_GAUCHE = 15
    CHEVILLE_DROITE = 16

    def obtenir_position_mains(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retourne les positions des deux poignets (main gauche, main droite)."""
        main_gauche = None
        main_droite = None

        # Vérifier confiance du poignet gauche
        if self.keypoints[self.POIGNET_GAUCHE, 2] > 0.3:
            main_gauche = self.keypoints[self.POIGNET_GAUCHE, :2]

        # Vérifier confiance du poignet droit
        if self.keypoints[self.POIGNET_DROIT, 2] > 0.3:
            main_droite = self.keypoints[self.POIGNET_DROIT, :2]

        return main_gauche, main_droite

    def obtenir_centre_torse(self) -> Optional[np.ndarray]:
        """Retourne le centre du torse (milieu des épaules et hanches)."""
        points = [
            self.keypoints[self.EPAULE_GAUCHE],
            self.keypoints[self.EPAULE_DROITE],
            self.keypoints[self.HANCHE_GAUCHE],
            self.keypoints[self.HANCHE_DROITE],
        ]

        # Vérifier que suffisamment de points sont visibles
        points_valides = [p[:2] for p in points if p[2] > 0.3]
        if len(points_valides) < 2:
            return None

        return np.mean(points_valides, axis=0)

    def obtenir_centre_hanches(self) -> Optional[np.ndarray]:
        """Retourne le centre des hanches (zone de dissimulation sous vetements)."""
        points = [
            self.keypoints[self.HANCHE_GAUCHE],
            self.keypoints[self.HANCHE_DROITE],
        ]
        points_valides = [p[:2] for p in points if p[2] > 0.3]
        if not points_valides:
            return None
        return np.mean(points_valides, axis=0)

    def obtenir_coudes(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retourne les positions des coudes (pour detecter bras plies)."""
        coude_g = None
        coude_d = None
        if self.keypoints[self.COUDE_GAUCHE, 2] > 0.3:
            coude_g = self.keypoints[self.COUDE_GAUCHE, :2]
        if self.keypoints[self.COUDE_DROIT, 2] > 0.3:
            coude_d = self.keypoints[self.COUDE_DROIT, :2]
        return coude_g, coude_d

    def obtenir_epaules(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retourne les positions des epaules."""
        ep_g = None
        ep_d = None
        if self.keypoints[self.EPAULE_GAUCHE, 2] > 0.3:
            ep_g = self.keypoints[self.EPAULE_GAUCHE, :2]
        if self.keypoints[self.EPAULE_DROITE, 2] > 0.3:
            ep_d = self.keypoints[self.EPAULE_DROITE, :2]
        return ep_g, ep_d


class DetecteurPersonnes:
    """
    Détecteur de personnes et objets utilisant YOLO en format ONNX.
    Optimisé pour l'inférence CPU.
    """

    # Les 80 classes COCO completes
    CLASSES_COCO = {
        0: "personne", 1: "velo", 2: "voiture", 3: "moto", 4: "avion",
        5: "bus", 6: "train", 7: "camion", 8: "bateau", 9: "feu_tricolore",
        10: "bouche_incendie", 11: "panneau_stop", 12: "parcmetre", 13: "banc",
        14: "oiseau", 15: "chat", 16: "chien", 17: "cheval", 18: "mouton",
        19: "vache", 20: "elephant", 21: "ours", 22: "zebre", 23: "girafe",
        24: "sac_a_dos", 25: "parapluie", 26: "sac_a_main", 27: "cravate",
        28: "valise", 29: "frisbee", 30: "skis", 31: "snowboard",
        32: "ballon", 33: "cerf_volant", 34: "batte_baseball", 35: "gant_baseball",
        36: "skateboard", 37: "planche_surf", 38: "raquette_tennis",
        39: "bouteille", 40: "verre_vin", 41: "tasse", 42: "fourchette",
        43: "couteau", 44: "cuillere", 45: "bol", 46: "banane", 47: "pomme",
        48: "sandwich", 49: "orange", 50: "brocoli", 51: "carotte",
        52: "hot_dog", 53: "pizza", 54: "donut", 55: "gateau",
        56: "chaise", 57: "canape", 58: "plante_en_pot", 59: "lit",
        60: "table", 61: "toilettes", 62: "television", 63: "ordinateur_portable",
        64: "souris", 65: "telecommande", 66: "clavier", 67: "telephone",
        68: "micro_ondes", 69: "four", 70: "grille_pain", 71: "evier",
        72: "refrigerateur", 73: "livre", 74: "horloge", 75: "vase",
        76: "ciseaux", 77: "ours_en_peluche", 78: "seche_cheveux", 79: "brosse_a_dents",
    }

    # Classes pertinentes pour la detection de fraude/vol en magasin
    CLASSES_PERTINENTES = {
        0: "personne",
        24: "sac_a_dos",     # dissimulation d'articles
        26: "sac_a_main",    # dissimulation d'articles
        28: "valise",        # dissimulation d'articles
        67: "telephone",     # contexte comportemental
    }

    # Classes d'objets portables (pour l'analyse "objets portes" / dissimulation)
    CLASSES_PORTABLES = {
        24: "sac_a_dos",
        26: "sac_a_main",
        28: "valise",
    }

    # Categorie pour chaque class_id (personne vs objet)
    @staticmethod
    def categorie_pour(class_id: int) -> str:
        return "personne" if class_id == 0 else "objet"

    def __init__(self, chemin_modele: Path, confiance_min: float = 0.45, taille_entree: int = 320):
        """
        Initialise le détecteur YOLO ONNX.

        Args:
            chemin_modele: Chemin vers le fichier .onnx
            confiance_min: Seuil de confiance minimum
            taille_entree: Taille d'entree du modele (320 = rapide, 640 = precis)
        """
        self.confiance_min = confiance_min
        self.taille_entree = taille_entree
        self.session = None

        self._charger_modele(chemin_modele)

    def _charger_modele(self, chemin_modele: Path):
        """Charge le modèle ONNX avec OpenVINO ou CPU."""
        try:
            self.session = creer_session_onnx(str(chemin_modele), "YOLO detection")

            self.nom_entree = self.session.get_inputs()[0].name
            self.noms_sortie = [o.name for o in self.session.get_outputs()]

            logger.info(f"Modele YOLO charge: {chemin_modele.name}")
            logger.info(f"Entree: {self.nom_entree}, Sorties: {self.noms_sortie}")

        except Exception as e:
            logger.error(f"Erreur chargement modele: {e}")
            raise RuntimeError(
                f"Impossible de charger le modele YOLO: {chemin_modele}. "
                f"Executez scripts/download_models.py d'abord."
            ) from e

    def _preprocesser(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Pretraitement YOLO: letterbox a taille_entree avec padding gris 114."""
        return _letterbox_yolo(frame, self.taille_entree)

    def _postprocesser(
        self,
        sortie: np.ndarray,
        ratio: float,
        padding: Tuple[int, int],
        taille_originale: Tuple[int, int],
    ) -> List[Detection]:
        """Post-traitement YOLO + NMS + filtre CLASSES_PERTINENTES."""
        brut = _decoder_sortie_yolo(
            sortie, ratio, padding, taille_originale,
            seuil_confiance=self.confiance_min,
            filtre_classes=set(self.CLASSES_PERTINENTES.keys()),
        )
        return [
            Detection(
                bbox=bbox, confidence=score, class_id=id_c,
                class_name=self.CLASSES_PERTINENTES.get(id_c, "inconnu"),
                categorie=self.categorie_pour(id_c),
            )
            for bbox, score, id_c in brut
        ]

    def detecter(self, frame: np.ndarray) -> List[Detection]:
        """
        Détecte toutes les entités pertinentes dans une frame.

        Args:
            frame: Image BGR (numpy array)

        Returns:
            Liste de détections (personnes et objets)
        """
        if self.session is None:
            return []

        taille_originale = frame.shape[:2]

        # Prétraitement
        image_prep, ratio, padding = self._preprocesser(frame)

        # Inférence ONNX (protegee contre les erreurs)
        try:
            sorties = self.session.run(self.noms_sortie, {self.nom_entree: image_prep})
        except Exception as e:
            logger.error(f"Erreur inference ONNX: {e}")
            return []

        # Post-traitement
        detections = self._postprocesser(sorties[0], ratio, padding, taille_originale)

        return detections

    def detecter_personnes_et_objets(self, frame: np.ndarray) -> tuple:
        """Detecte et separe personnes et objets en une seule inference."""
        toutes = self.detecter(frame)
        personnes = [d for d in toutes if d.class_id == 0]
        objets = [d for d in toutes if d.class_id != 0]
        return personnes, objets

    def detecter_personnes(self, frame: np.ndarray) -> List[Detection]:
        """Détecte uniquement les personnes dans la frame."""
        toutes = self.detecter(frame)
        return [d for d in toutes if d.class_id == 0]

    def detecter_objets(self, frame: np.ndarray) -> List[Detection]:
        """Détecte uniquement les objets (sacs, articles) dans la frame."""
        toutes = self.detecter(frame)
        return [d for d in toutes if d.class_id != 0]

    def detecter_tout_coco(self, frame: np.ndarray, confiance_min: Optional[float] = None) -> List[Detection]:
        """
        Détecte TOUTES les 80 classes COCO (mode test/dashboard).
        Utilisé pour afficher tous les objets visibles dans une image.
        """
        if self.session is None:
            return []

        taille_originale = frame.shape[:2]
        image_prep, ratio, padding = self._preprocesser(frame)
        sorties = self.session.run(self.noms_sortie, {self.nom_entree: image_prep})

        seuil = confiance_min if confiance_min is not None else self.confiance_min
        return self._postprocesser_tout_coco(sorties[0], ratio, padding, taille_originale, seuil)

    def _postprocesser_tout_coco(
        self,
        sortie: np.ndarray,
        ratio: float,
        padding: Tuple[int, int],
        taille_originale: Tuple[int, int],
        seuil_confiance: Optional[float] = None,
    ) -> List[Detection]:
        """Post-traitement sans filtre de classes — retourne les 80 classes COCO."""
        seuil = seuil_confiance if seuil_confiance is not None else self.confiance_min
        brut = _decoder_sortie_yolo(
            sortie, ratio, padding, taille_originale,
            seuil_confiance=seuil,
        )
        return [
            Detection(
                bbox=bbox, confidence=score, class_id=id_c,
                class_name=self.CLASSES_COCO.get(id_c, f"classe_{id_c}"),
                categorie=self.categorie_pour(id_c),
            )
            for bbox, score, id_c in brut
        ]


class EstimateurPose:
    """
    Estimateur de pose utilisant YOLOv8-pose en ONNX.
    Retourne 17 points clés COCO pour chaque personne détectée.
    """

    def __init__(self, chemin_modele: Path, confiance_min: float = 0.5, taille_entree: int = 320):
        """
        Initialise l'estimateur de pose.

        Args:
            chemin_modele: Chemin vers yolov8n-pose.onnx
            confiance_min: Seuil minimum de confiance
            taille_entree: Taille d'entree du modele (320 = rapide, 640 = precis)
        """
        self.confiance_min = confiance_min
        self.taille_entree = taille_entree
        self.session = None

        self._charger_modele(chemin_modele)

    def _charger_modele(self, chemin_modele: Path):
        """Charge le modèle de pose ONNX."""
        try:
            self.session = creer_session_onnx(str(chemin_modele), "YOLO pose")

            self.nom_entree = self.session.get_inputs()[0].name
            self.noms_sortie = [o.name for o in self.session.get_outputs()]

            logger.info(f"Modele pose charge: {chemin_modele.name}")

        except Exception as e:
            logger.warning(f"Impossible de charger le modele pose: {e}")
            self.session = None

    def estimer_pose(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[PoseKeypoints]:
        """
        Estime la pose d'une personne dans la frame.

        Args:
            frame: Image BGR complète
            bbox: Boîte englobante optionnelle (x1, y1, x2, y2) pour recadrer

        Returns:
            PoseKeypoints ou None si échec
        """
        if self.session is None:
            return None

        # Recadrer si bbox fourni (avec marge)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            # Ajouter une marge de 10%
            marge_x = int((x2 - x1) * 0.1)
            marge_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - marge_x)
            y1 = max(0, y1 - marge_y)
            x2 = min(w, x2 + marge_x)
            y2 = min(h, y2 + marge_y)
            region = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            region = frame
            offset = (0, 0)

        if region.size == 0:
            return None

        # Pretraitement letterbox (meme convention que DetecteurPersonnes)
        image_norm, ratio, (pad_x, pad_y) = _letterbox_yolo(region, self.taille_entree)

        # Inférence
        try:
            sorties = self.session.run(self.noms_sortie, {self.nom_entree: image_norm})
        except Exception as e:
            logger.error(f"Erreur inference pose: {e}")
            return None

        # Extraire les points clés
        return self._extraire_keypoints(sorties[0], ratio, (pad_x, pad_y), offset)

    def _extraire_keypoints(
        self,
        sortie: np.ndarray,
        ratio: float,
        padding: Tuple[int, int],
        offset: Tuple[int, int],
    ) -> Optional[PoseKeypoints]:
        """Extrait les points clés de la sortie du modèle pose."""
        # Sortie pose YOLO: (1, 56, N) -> 4 bbox + 1 conf + 51 keypoints (17*3)
        if sortie.ndim == 3:
            sortie = sortie[0]
        if sortie.shape[0] < sortie.shape[1]:
            sortie = sortie.T

        if len(sortie) == 0:
            return None

        pad_x, pad_y = padding
        off_x, off_y = offset

        # Trouver la détection avec la meilleure confiance
        meilleur_score = 0.0
        meilleur_idx = -1

        for i, ligne in enumerate(sortie):
            score = float(ligne[4])
            if score > meilleur_score:
                meilleur_score = score
                meilleur_idx = i

        if meilleur_idx < 0 or meilleur_score < self.confiance_min:
            return None

        ligne = sortie[meilleur_idx]

        # Extraire les 17 points clés (après les 5 premières valeurs: cx, cy, w, h, conf)
        # Chaque point: x, y, confiance
        keypoints = np.zeros((17, 3), dtype=np.float32)

        for j in range(17):
            idx_base = 5 + j * 3
            if idx_base + 2 < len(ligne):
                kp_x = (ligne[idx_base] - pad_x) / ratio + off_x
                kp_y = (ligne[idx_base + 1] - pad_y) / ratio + off_y
                kp_conf = ligne[idx_base + 2]
                keypoints[j] = [kp_x, kp_y, kp_conf]

        return PoseKeypoints(keypoints=keypoints, confidence=meilleur_score)

    def estimer_poses_multiples(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[PoseKeypoints]]:
        """
        Estime la pose pour plusieurs personnes.
        Traite chaque personne individuellement pour économiser la mémoire.
        """
        resultats = []
        for bbox in bboxes:
            pose = self.estimer_pose(frame, bbox)
            resultats.append(pose)
        return resultats


# ============================================================
# Detecteur Open Images V7 (apprentissage zones statiques)
# ============================================================

# Traductions FR pour les classes retail pertinentes (nom_anglais -> nom_francais).
# Les classes non listees gardent leur nom original (anglais).
# Utilise lors de l'apprentissage pour afficher des noms parlants dans le dashboard.
TRADUCTIONS_OIV7_FR = {
    "Cash register": "caisse_enregistreuse",
    "Printer": "imprimante",
    "Scanner": "scanner",
    "Computer monitor": "ecran",
    "Computer keyboard": "clavier",
    "Computer mouse": "souris",
    "Laptop": "ordinateur_portable",
    "Telephone": "telephone",
    "Corded phone": "telephone_fixe",
    "Mobile phone": "telephone_mobile",
    "Television": "television",
    "Mannequin": "mannequin",
    "Person": "personne",
    "Human face": "visage",
    "Human body": "corps",
    "Backpack": "sac_a_dos",
    "Handbag": "sac_a_main",
    "Briefcase": "mallette",
    "Suitcase": "valise",
    "Luggage and bags": "bagages",
    "Plastic bag": "sac_plastique",
    "Shopping cart": "chariot",
    "Box": "boite",
    "Chair": "chaise",
    "Table": "table",
    "Desk": "bureau",
    "Couch": "canape",
    "Shelf": "etagere",
    "Clothing": "vetement",
    "Coat": "manteau",
    "Dress": "robe",
    "Jacket": "veste",
    "Jeans": "jean",
    "Trousers": "pantalon",
    "Shirt": "chemise",
    "Skirt": "jupe",
    "Suit": "costume",
    "Glasses": "lunettes",
    "Hat": "chapeau",
    "Sunglasses": "lunettes_soleil",
    "Door": "porte",
    "Window": "fenetre",
    "Book": "livre",
    "Clock": "horloge",
    "Poster": "affiche",
    "Flag": "drapeau",
    "Picture frame": "cadre_photo",
    "Lamp": "lampe",
    "Light bulb": "ampoule",
    "Fan": "ventilateur",
    "Air conditioner": "climatiseur",
    "Camera": "camera",
    "Refrigerator": "refrigerateur",
    "Microwave oven": "micro_ondes",
    "Coffeemaker": "machine_a_cafe",
    "Bottle": "bouteille",
    "Cup": "tasse",
    "Vase": "vase",
    "Mirror": "miroir",
    "Stairs": "escalier",
    "Fire hydrant": "borne_incendie",
    "Billboard": "panneau_publicitaire",
    "Signage": "panneau",
    "Traffic sign": "panneau_circulation",
}


def _lire_noms_classes_onnx(session) -> dict:
    """Extrait le mapping {id: nom} depuis les metadata ONNX (Ultralytics).

    Ultralytics stocke les noms de classes dans session.get_modelmeta()
    .custom_metadata_map['names'] sous forme de repr Python d'un dict.
    """
    import ast
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names")
        if not raw:
            logger.warning("Pas de metadata 'names' dans le modele ONNX")
            return {}
        noms_dict = ast.literal_eval(raw)
        return {int(k): str(v) for k, v in noms_dict.items()}
    except Exception as e:
        logger.warning(f"Impossible de lire les noms de classes ONNX: {e}")
        return {}


class DetecteurApprentissage:
    """
    Detecteur Open Images V7 (~600 classes) utilise UNIQUEMENT pendant
    les sessions d'apprentissage pour identifier les objets statiques
    retail (caisse, imprimante, scanner, ecran, mannequin, etc.).

    Lazy-load: la session ONNX n'est creee qu'au premier appel. Peut etre
    liberee via `liberer()` une fois l'apprentissage termine pour rendre
    la RAM au systeme.
    """

    def __init__(self, chemin_modele: Path, confiance_min: float = 0.25,
                 taille_entree: int = 640):
        self.chemin_modele = chemin_modele
        self.confiance_min = confiance_min
        self.taille_entree = taille_entree
        self.session = None
        self.actif = False
        self.classes: dict = {}
        self.nom_entree = None
        self.noms_sortie = None

    def _charger_si_necessaire(self) -> bool:
        """Charge le modele ONNX a la premiere utilisation."""
        if self.session is not None:
            return True
        if not self.chemin_modele.exists():
            logger.warning(
                f"Modele apprentissage (OIV7) absent: {self.chemin_modele}. "
                f"L'apprentissage zones retombera sur COCO 80 classes."
            )
            self.actif = False
            return False
        try:
            self.session = creer_session_onnx(
                str(self.chemin_modele), "YOLO OIV7 apprentissage")
            self.nom_entree = self.session.get_inputs()[0].name
            self.noms_sortie = [o.name for o in self.session.get_outputs()]
            self.classes = _lire_noms_classes_onnx(self.session)
            self.actif = True
            logger.info(
                f"Modele OIV7 charge: {self.chemin_modele.name} "
                f"({len(self.classes)} classes)"
            )
            return True
        except Exception as e:
            logger.error(f"Erreur chargement modele OIV7: {e}")
            self.session = None
            self.actif = False
            return False

    def liberer(self):
        """Libere la session ONNX (appelle apres fin d'apprentissage)."""
        if self.session is not None:
            try:
                del self.session
            except Exception as e:
                logger.debug(f"Erreur liberation session OIV7 (non bloquant): {e}")
            self.session = None
            self.actif = False
            logger.info("Session OIV7 liberee")

    def _preprocesser(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox preprocessing (meme que DetecteurPersonnes)."""
        return _letterbox_yolo(frame, self.taille_entree)

    def detecter_tout(self, frame: np.ndarray,
                      confiance_min: Optional[float] = None) -> List[Detection]:
        """Detecte toutes les classes OIV7 dans la frame.

        Retourne les Detection avec class_name traduit en FR quand dispo
        (sinon nom anglais original).
        """
        if not self._charger_si_necessaire():
            return []
        if self.session is None:
            return []

        taille_originale = frame.shape[:2]
        image_prep, ratio, padding = self._preprocesser(frame)
        try:
            sorties = self.session.run(self.noms_sortie, {self.nom_entree: image_prep})
        except Exception as e:
            logger.error(f"Erreur inference OIV7: {e}")
            return []

        seuil = confiance_min if confiance_min is not None else self.confiance_min
        return self._postprocesser(sorties[0], ratio, padding, taille_originale, seuil)

    def _postprocesser(
        self,
        sortie: np.ndarray,
        ratio: float,
        padding: Tuple[int, int],
        taille_originale: Tuple[int, int],
        seuil_confiance: float,
    ) -> List[Detection]:
        """Decode OIV7 avec traduction FR. class_id offset par 2000 pour eviter
        collision avec COCO (0-79) et Vetements (1000+)."""
        brut = _decoder_sortie_yolo(
            sortie, ratio, padding, taille_originale,
            seuil_confiance=seuil_confiance,
        )
        detections: List[Detection] = []
        for bbox, score, id_c in brut:
            nom_en = self.classes.get(id_c, f"classe_{id_c}")
            nom_fr = TRADUCTIONS_OIV7_FR.get(nom_en, nom_en.lower().replace(" ", "_"))
            detections.append(Detection(
                bbox=bbox, confidence=score, class_id=2000 + id_c,
                class_name=nom_fr,
                categorie="personne" if nom_en == "Person" else "objet",
            ))
        return detections


class DetecteurVetements:
    """
    Détecteur de vêtements basé sur un modèle YOLO fine-tuné sur DeepFashion2.
    Détecte 13 types de vêtements dans les images.
    Le modèle est optionnel — si absent, le détecteur est inactif.
    """

    # 13 classes de vêtements (DeepFashion2)
    CLASSES_VETEMENTS = {
        0: "t-shirt",
        1: "pull",
        2: "veste_courte",
        3: "manteau",
        4: "gilet",
        5: "jupe",
        6: "short",
        7: "pantalon",
        8: "robe_courte",
        9: "robe_longue",
        10: "robe_sans_manches",
        11: "jupe_longue",
        12: "combinaison",
    }

    # Classes volables (articles de valeur dans un magasin textile)
    CLASSES_VOLABLES = {
        0: "t-shirt",
        1: "pull",
        2: "veste_courte",
        3: "manteau",
        4: "gilet",
        5: "jupe",
        6: "short",
        7: "pantalon",
        8: "robe_courte",
        9: "robe_longue",
        10: "robe_sans_manches",
        11: "jupe_longue",
        12: "combinaison",
    }

    def __init__(self, chemin_modele: Path, confiance_min: float = 0.40):
        self.confiance_min = confiance_min
        self.taille_entree = 640
        self.session = None
        self.actif = False

        if chemin_modele.exists():
            self._charger_modele(chemin_modele)
        else:
            logger.info(f"Modele vetements non trouve: {chemin_modele} — detection vetements inactive")

    def _charger_modele(self, chemin_modele: Path):
        try:
            self.session = creer_session_onnx(str(chemin_modele), "YOLO vetements")

            self.nom_entree = self.session.get_inputs()[0].name
            self.noms_sortie = [o.name for o in self.session.get_outputs()]
            self.actif = True

            logger.info(f"Modele vetements charge: {chemin_modele.name}")

        except Exception as e:
            logger.warning(f"Impossible de charger le modele vetements: {e}")
            self.session = None
            self.actif = False

    def _preprocesser(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocessing identique YOLO (letterbox a taille_entree)."""
        return _letterbox_yolo(frame, self.taille_entree)

    def detecter(self, frame: np.ndarray) -> List[Detection]:
        """
        Détecte les vêtements dans la frame.

        Returns:
            Liste de Detection avec categorie='vetement'
        """
        if not self.actif or self.session is None:
            return []

        taille_originale = frame.shape[:2]
        image_prep, ratio, padding = self._preprocesser(frame)

        try:
            sorties = self.session.run(self.noms_sortie, {self.nom_entree: image_prep})
        except Exception as e:
            logger.error(f"Erreur inference vetements: {e}")
            return []

        return self._postprocesser(sorties[0], ratio, padding, taille_originale)

    def _postprocesser(
        self,
        sortie: np.ndarray,
        ratio: float,
        padding: Tuple[int, int],
        taille_originale: Tuple[int, int],
    ) -> List[Detection]:
        """Post-traitement du modele fashion. class_id offset par 1000 pour
        eviter collision avec COCO."""
        brut = _decoder_sortie_yolo(
            sortie, ratio, padding, taille_originale,
            seuil_confiance=self.confiance_min,
            nb_classes=len(self.CLASSES_VETEMENTS),
        )
        detections = [
            Detection(
                bbox=bbox, confidence=score, class_id=1000 + id_c,
                class_name=self.CLASSES_VETEMENTS.get(id_c, "vetement"),
                categorie="vetement",
            )
            for bbox, score, id_c in brut
        ]

        return detections


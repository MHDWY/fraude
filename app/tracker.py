"""
Suivi de personnes avec ByteTrack.
Association IoU en deux étapes (scores hauts puis bas).
Historique des trajectoires pour l'analyse comportementale.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class PisteSuivi:
    """Représente une piste de suivi d'une personne."""
    id_piste: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) courant
    score: float
    # Historique des boîtes englobantes (pour trajectoire)
    historique_bbox: deque = field(default_factory=lambda: deque(maxlen=300))
    # Historique des positions centrales
    historique_centres: deque = field(default_factory=lambda: deque(maxlen=300))
    # Horodatages des observations
    historique_temps: deque = field(default_factory=lambda: deque(maxlen=300))
    # Nombre de frames sans correspondance
    frames_perdues: int = 0
    # État de la piste
    etat: str = "nouveau"  # nouveau, actif, perdu, supprime
    # Timestamp de première détection
    temps_debut: float = 0.0
    # Timestamp de dernière mise à jour
    derniere_maj: float = 0.0
    # Compteur de mises à jour
    nb_mises_a_jour: int = 0

    def __post_init__(self):
        if self.temps_debut == 0.0:
            self.temps_debut = time.time()
        self.derniere_maj = time.time()
        self._ajouter_observation()

    def _ajouter_observation(self):
        """Enregistre l'observation courante dans l'historique."""
        maintenant = time.time()
        self.historique_bbox.append(self.bbox)
        centre = self._calculer_centre(self.bbox)
        self.historique_centres.append(centre)
        self.historique_temps.append(maintenant)

    def _calculer_centre(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calcule le centre d'une boîte englobante."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def mettre_a_jour(self, bbox: Tuple[int, int, int, int], score: float):
        """Met à jour la piste avec une nouvelle détection."""
        self.bbox = bbox
        self.score = score
        self.frames_perdues = 0
        self.derniere_maj = time.time()
        self.nb_mises_a_jour += 1
        self._ajouter_observation()

        # Passer à actif immediatement (CCTV: pas besoin d'attendre)
        if self.etat == "nouveau" and self.nb_mises_a_jour >= 1:
            self.etat = "actif"

    def marquer_perdue(self):
        """Marque la piste comme perdue pour une frame."""
        self.frames_perdues += 1

    @property
    def centre(self) -> Tuple[float, float]:
        """Centre actuel de la boîte englobante."""
        return self._calculer_centre(self.bbox)

    @property
    def duree_presence(self) -> float:
        """Durée de présence en secondes."""
        return time.time() - self.temps_debut

    @property
    def vitesse_moyenne(self) -> float:
        """Vitesse moyenne en pixels/seconde sur les dernières observations."""
        if len(self.historique_centres) < 2:
            return 0.0

        centres = list(self.historique_centres)
        temps = list(self.historique_temps)

        if len(centres) < 2:
            return 0.0

        distances = []
        for i in range(1, min(len(centres), 30)):  # Dernières 30 observations
            dx = centres[i][0] - centres[i - 1][0]
            dy = centres[i][1] - centres[i - 1][1]
            dt = temps[i] - temps[i - 1]
            if dt > 0:
                distances.append(np.sqrt(dx ** 2 + dy ** 2) / dt)

        return float(np.mean(distances)) if distances else 0.0



def calculer_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calcule l'Intersection over Union entre deux boîtes englobantes.

    Args:
        bbox1, bbox2: (x1, y1, x2, y2)

    Returns:
        Valeur IoU entre 0 et 1
    """
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

    return intersection / max(union, 1e-6)


def matrice_cout_iou(
    pistes: List[PisteSuivi],
    detections: List[Tuple[Tuple[int, int, int, int], float]],
) -> np.ndarray:
    """
    Calcule la matrice de coût IoU entre pistes et détections.

    Returns:
        Matrice (nb_pistes, nb_detections) de coûts (1 - IoU)
    """
    nb_pistes = len(pistes)
    nb_detections = len(detections)

    if nb_pistes == 0 or nb_detections == 0:
        return np.empty((nb_pistes, nb_detections))

    matrice = np.zeros((nb_pistes, nb_detections))
    for i, piste in enumerate(pistes):
        for j, (bbox, _) in enumerate(detections):
            matrice[i, j] = 1.0 - calculer_iou(piste.bbox, bbox)

    return matrice


class ByteTracker:
    """
    Implémentation de ByteTrack pour le suivi multi-personnes.
    Association en deux étapes: scores hauts puis scores bas.
    """

    def __init__(
        self,
        seuil_score_haut: float = 0.3,
        seuil_score_bas: float = 0.1,
        max_frames_perdues: int = 30,
        seuil_iou: float = 0.3,
    ):
        """
        Args:
            seuil_score_haut: Seuil pour les détections haute confiance
            seuil_score_bas: Seuil pour les détections basse confiance
            max_frames_perdues: Nombre max de frames sans correspondance avant suppression
            seuil_iou: Seuil IoU minimum pour l'association
        """
        self.seuil_score_haut = seuil_score_haut
        self.seuil_score_bas = seuil_score_bas
        self.max_frames_perdues = max_frames_perdues
        self.seuil_iou = seuil_iou

        self.pistes_actives: List[PisteSuivi] = []
        self.pistes_perdues: List[PisteSuivi] = []
        self._pistes_supprimees_derniere_frame: List[PisteSuivi] = []
        self._prochain_id: int = 1

    def _obtenir_nouvel_id(self) -> int:
        """Génère un nouvel identifiant unique de piste."""
        id_piste = self._prochain_id
        self._prochain_id += 1
        return id_piste

    def _associer(
        self,
        pistes: List[PisteSuivi],
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        seuil_iou: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Association entre pistes et détections par IoU.

        Returns:
            (correspondances, pistes_non_associees, detections_non_associees)
        """
        if len(pistes) == 0 or len(detections) == 0:
            return [], list(range(len(pistes))), list(range(len(detections)))

        # Calculer la matrice de coût
        matrice = matrice_cout_iou(pistes, detections)

        # Résoudre l'affectation optimale (algorithme hongrois)
        indices_pistes, indices_dets = linear_sum_assignment(matrice)

        correspondances = []
        pistes_non_assoc = set(range(len(pistes)))
        dets_non_assoc = set(range(len(detections)))

        for ip, id_det in zip(indices_pistes, indices_dets):
            cout = matrice[ip, id_det]
            if cout < (1.0 - seuil_iou):
                correspondances.append((ip, id_det))
                pistes_non_assoc.discard(ip)
                dets_non_assoc.discard(id_det)

        return correspondances, list(pistes_non_assoc), list(dets_non_assoc)

    def mettre_a_jour(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
    ) -> List[PisteSuivi]:
        """
        Met à jour le tracker avec les nouvelles détections.
        Implémentation ByteTrack en deux étapes.

        Args:
            detections: Liste de (bbox, score)

        Returns:
            Liste des pistes actives mises à jour
        """
        # Séparer les détections haute et basse confiance
        dets_hautes = [(bbox, s) for bbox, s in detections if s >= self.seuil_score_haut]
        dets_basses = [(bbox, s) for bbox, s in detections if self.seuil_score_bas <= s < self.seuil_score_haut]

        # === ÉTAPE 1: Associer détections hautes avec pistes actives ===
        correspondances_1, pistes_restantes_1, dets_restantes_1 = self._associer(
            self.pistes_actives, dets_hautes, self.seuil_iou
        )

        # Mettre à jour les pistes associées
        for ip, id_det in correspondances_1:
            bbox, score = dets_hautes[id_det]
            self.pistes_actives[ip].mettre_a_jour(bbox, score)

        # === ÉTAPE 2: Associer détections basses avec pistes restantes ===
        pistes_non_associees = [self.pistes_actives[i] for i in pistes_restantes_1]

        correspondances_2, pistes_restantes_2, _ = self._associer(
            pistes_non_associees, dets_basses, self.seuil_iou
        )

        # Mettre à jour les pistes associées avec les détections basses
        for ip, id_det in correspondances_2:
            bbox, score = dets_basses[id_det]
            pistes_non_associees[ip].mettre_a_jour(bbox, score)

        # === ÉTAPE 3: Traiter les pistes non associées ===
        pistes_encore_restantes = [pistes_non_associees[i] for i in pistes_restantes_2]
        for piste in pistes_encore_restantes:
            piste.marquer_perdue()
            if piste.frames_perdues > self.max_frames_perdues:
                piste.etat = "supprime"
                logger.debug(f"Piste {piste.id_piste} supprimee (perdue trop longtemps)")
            else:
                piste.etat = "perdu"

        # === ÉTAPE 4: Créer de nouvelles pistes pour les détections hautes non associées ===
        for idx in dets_restantes_1:
            bbox, score = dets_hautes[idx]
            nouvelle_piste = PisteSuivi(
                id_piste=self._obtenir_nouvel_id(),
                bbox=bbox,
                score=score,
            )
            self.pistes_actives.append(nouvelle_piste)
            logger.debug(f"Nouvelle piste creee: {nouvelle_piste.id_piste}")

        # Capturer les pistes supprimees avant de les filtrer
        self._pistes_supprimees_derniere_frame = [
            p for p in self.pistes_actives if p.etat == "supprime"
        ]

        # Mettre a jour les listes de pistes
        self.pistes_actives = [
            p for p in self.pistes_actives if p.etat != "supprime"
        ]

        # Retourner les pistes actives, nouvelles et recemment perdues (<10 frames)
        return [p for p in self.pistes_actives if p.etat in ("actif", "nouveau") or (p.etat == "perdu" and p.frames_perdues <= 10)]

    @property
    def pistes_recemment_supprimees(self) -> List[PisteSuivi]:
        """Retourne les pistes supprimees lors de la derniere mise a jour."""
        return self._pistes_supprimees_derniere_frame

    def obtenir_piste(self, id_piste: int) -> Optional[PisteSuivi]:
        """Retourne une piste par son identifiant."""
        for piste in self.pistes_actives:
            if piste.id_piste == id_piste:
                return piste
        return None

    def obtenir_toutes_pistes(self) -> List[PisteSuivi]:
        """Retourne toutes les pistes (actives et perdues)."""
        return list(self.pistes_actives)

    def reinitialiser(self):
        """Réinitialise complètement le tracker."""
        self.pistes_actives.clear()
        self.pistes_perdues.clear()
        self._prochain_id = 1
        logger.info("Tracker reinitialise")


class HistoriqueTrajectoires:
    """
    Stocke et analyse les trajectoires de toutes les personnes détectées.
    Utilisé par l'analyseur de comportements.
    """

    def __init__(self, duree_max_secondes: int = 300):
        """
        Args:
            duree_max_secondes: Durée max de conservation des trajectoires (5 min par défaut)
        """
        self.duree_max = duree_max_secondes
        # Dictionnaire id_piste -> liste de (timestamp, centre_x, centre_y, bbox)
        self.trajectoires: Dict[int, deque] = {}

    def ajouter_observation(
        self,
        id_piste: int,
        bbox: Tuple[int, int, int, int],
        timestamp: Optional[float] = None,
    ):
        """Ajoute une observation de position pour une piste."""
        if timestamp is None:
            timestamp = time.time()

        if id_piste not in self.trajectoires:
            self.trajectoires[id_piste] = deque(maxlen=1000)

        centre_x = (bbox[0] + bbox[2]) / 2.0
        centre_y = (bbox[1] + bbox[3]) / 2.0
        self.trajectoires[id_piste].append((timestamp, centre_x, centre_y, bbox))

    def obtenir_trajectoire(self, id_piste: int) -> List[Tuple[float, float, float]]:
        """Retourne la trajectoire (timestamp, x, y) d'une piste."""
        if id_piste not in self.trajectoires:
            return []
        return [(t, x, y) for t, x, y, _ in self.trajectoires[id_piste]]

    def obtenir_derniere_position(self, id_piste: int) -> Optional[Tuple[float, float]]:
        """Retourne la dernière position connue d'une piste."""
        if id_piste not in self.trajectoires or len(self.trajectoires[id_piste]) == 0:
            return None
        _, x, y, _ = self.trajectoires[id_piste][-1]
        return (x, y)

    def calculer_zone_presence(self, id_piste: int, derniers_n_seconds: float = 30.0) -> Optional[Tuple[float, float, float, float]]:
        """
        Calcule la zone de présence d'une piste sur les N dernières secondes.
        Retourne (x_min, y_min, x_max, y_max) ou None.
        """
        if id_piste not in self.trajectoires:
            return None

        maintenant = time.time()
        seuil = maintenant - derniers_n_seconds

        points = [(x, y) for t, x, y, _ in self.trajectoires[id_piste] if t >= seuil]
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def nettoyer(self):
        """Supprime les trajectoires anciennes pour libérer la mémoire."""
        maintenant = time.time()
        seuil = maintenant - self.duree_max

        ids_a_supprimer = []
        for id_piste, traj in self.trajectoires.items():
            # Supprimer les observations anciennes
            while traj and traj[0][0] < seuil:
                traj.popleft()
            # Supprimer les trajectoires vides
            if not traj:
                ids_a_supprimer.append(id_piste)

        for id_piste in ids_a_supprimer:
            del self.trajectoires[id_piste]

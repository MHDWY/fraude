"""
Analyseur specifique de fraude a la caisse.

Detecte les anomalies dans le processus de paiement:
1. Passage article devant le scanner (mouvement de scan)
2. Impression du ticket (mouvement papier pres de l'imprimante)
3. Remise du ticket au client (mains qui se rapprochent)
4. Alerte si article scanne + paiement sans ticket dans 10-12s
5. Alerte si client quitte la caisse sans ticket apres paiement

Fonctionne comme une machine a etats par transaction,
en combinant les pistes (tracker) et les poses (keypoints).
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection, PoseKeypoints
from .tracker import PisteSuivi

logger = logging.getLogger(__name__)


# ============================================================================
# ETATS ET TYPES
# ============================================================================

class EtatTransaction(Enum):
    """Etats possibles d'une transaction a la caisse."""
    INACTIF = "inactif"                      # Pas d'activite detectee
    SCAN_DETECTE = "scan_detecte"            # Article passe devant scanner
    PAIEMENT_DETECTE = "paiement_detecte"    # Interaction avec le terminal
    ATTENTE_TICKET = "attente_ticket"        # En attente d'impression ticket
    TICKET_IMPRIME = "ticket_imprime"        # Ticket sorti de l'imprimante
    TICKET_REMIS = "ticket_remis"            # Ticket remis au client
    TRANSACTION_OK = "transaction_ok"        # Transaction complete
    ALERTE_PAS_TICKET = "alerte_pas_ticket"  # Timeout: pas de ticket
    ALERTE_DEPART_SANS_TICKET = "alerte_depart_sans_ticket"  # Client parti sans ticket


class TypeAlerteCaisse(Enum):
    """Types d'alertes specifiques a la caisse."""
    SCAN_SANS_TICKET = "scan_sans_ticket"
    PAIEMENT_SANS_TICKET = "paiement_sans_ticket"
    DEPART_SANS_TICKET = "depart_sans_ticket"
    TICKET_SANS_CLIENT = "ticket_sans_client"


DESCRIPTIONS_ALERTES_CAISSE = {
    TypeAlerteCaisse.SCAN_SANS_TICKET: "Article scanne mais pas de ticket imprime",
    TypeAlerteCaisse.PAIEMENT_SANS_TICKET: "Paiement effectue sans remise de ticket",
    TypeAlerteCaisse.DEPART_SANS_TICKET: "Client quitte la caisse sans ticket",
    TypeAlerteCaisse.TICKET_SANS_CLIENT: "Transaction fantome: ticket imprime sans client present",
}


@dataclass
class TransactionCaisse:
    """Etat d'une transaction en cours a la caisse."""
    id_caissier: int                          # ID piste du caissier
    id_client: Optional[int] = None           # ID piste du client
    etat: EtatTransaction = EtatTransaction.INACTIF
    timestamp_debut: float = field(default_factory=time.time)
    timestamp_scan: Optional[float] = None
    timestamp_paiement: Optional[float] = None
    timestamp_ticket: Optional[float] = None
    timestamp_remise: Optional[float] = None
    nb_scans: int = 0
    client_vu: bool = False                   # Un client a-t-il ete detecte pendant la transaction?
    alertes_emises: List[TypeAlerteCaisse] = field(default_factory=list)


@dataclass
class AlerteCaisse:
    """Resultat d'une alerte caisse."""
    type_alerte: TypeAlerteCaisse
    confiance: float
    id_caissier: int
    id_client: Optional[int]
    description: str
    horodatage: float = field(default_factory=time.time)


# ============================================================================
# ANALYSEUR PRINCIPAL
# ============================================================================

class AnalyseurCaisse:
    """
    Analyseur de fraude specifique a la zone caisse.

    Utilise une machine a etats pour suivre chaque transaction:
    INACTIF -> SCAN -> PAIEMENT -> ATTENTE_TICKET -> TICKET_IMPRIME -> REMIS -> OK

    Detecte les anomalies:
    - Timeout ticket (scan/paiement sans ticket dans le delai)
    - Depart client sans ticket
    """

    def __init__(
        self,
        timeout_ticket_secondes: float = 12.0,
        zone_caisse_y_min_pct: float = 0.70,
        zone_caisse_y_max_pct: float = 1.0,
        zone_caisse_x_min_pct: float = 0.25,
        zone_caisse_x_max_pct: float = 0.75,
        seuil_proximite_mains: float = 0.08,
        nb_cycles_scan_min: int = 2,
        cooldown_secondes: int = 30,
        imprimante_seuil_blanc: int = 200,
        imprimante_seuil_changement: float = 0.15,
        imprimante_bbox: Optional[tuple] = None,
        imprimante_mask_polygon: Optional[List[Tuple[float, float]]] = None,
        detecter_transaction_fantome: bool = True,
    ):
        """
        Args:
            timeout_ticket_secondes: Delai max apres scan/paiement pour le ticket
            zone_caisse_*: Proportions de la frame definissant la zone caisse
            seuil_proximite_mains: Distance relative (par rapport a la diag frame)
                                   pour considerer les mains comme proches
            nb_cycles_scan_min: Nombre min de cycles extension-retraction de la
                                douchette pour detecter un scan
            cooldown_secondes: Delai entre deux alertes du meme type
            imprimante_seuil_blanc: Valeur min de luminosite pour considerer un pixel comme blanc
            imprimante_seuil_changement: % min de pixels changes pour detecter du papier
            imprimante_bbox: Bbox absolue (x1,y1,x2,y2) depuis objets_reference (calibration visuelle)
            imprimante_mask_polygon: Polygone (liste de (x,y) en proportions de la ROI, 0.0-1.0)
                                     definissant les pixels A INCLURE dans la detection.
                                     Permet d'exclure la zone ou la main du caissier passe regulierement.
                                     None ou vide = ROI complete (comportement legacy).
        """
        self.timeout_ticket = timeout_ticket_secondes
        self.zone_caisse_y_min_pct = zone_caisse_y_min_pct
        self.zone_caisse_y_max_pct = zone_caisse_y_max_pct
        self.zone_caisse_x_min_pct = zone_caisse_x_min_pct
        self.zone_caisse_x_max_pct = zone_caisse_x_max_pct
        self.seuil_proximite_mains = seuil_proximite_mains

        # Zone imprimante pour detection visuelle du ticket (via objets_reference)
        self._imprimante_bbox_abs = imprimante_bbox
        self.imprimante_seuil_blanc = imprimante_seuil_blanc
        self.imprimante_seuil_changement = imprimante_seuil_changement
        self._imprimante_configuree = imprimante_bbox is not None
        if imprimante_bbox:
            logger.info(f"Imprimante configuree via objets_reference: bbox={imprimante_bbox}")

        # QW1: polygone de masque (ROI-relatif) pour exclure la zone main du caissier.
        # Stocke comme liste de tuples (x_pct, y_pct), 0.0-1.0. Vide = ROI complete.
        self._imprimante_mask_polygon: Optional[List[Tuple[float, float]]] = (
            list(imprimante_mask_polygon) if imprimante_mask_polygon else None
        )
        # Cache du masque binaire (uint8) recalcule si la shape de ROI change.
        self._imprimante_mask: Optional[np.ndarray] = None
        self._imprimante_mask_shape: Optional[Tuple[int, int]] = None
        if self._imprimante_mask_polygon:
            logger.info(
                f"[QW1] Mask polygon configure: {len(self._imprimante_mask_polygon)} points"
            )

        # Frame de reference de la zone imprimante (sans papier)
        self._imprimante_ref: Optional[np.ndarray] = None
        self._imprimante_ref_ts: float = 0.0
        # Compteur de detections positives consecutives pour filtrer les glitches HEVC
        # (un seul frame qui declenche peut etre une corruption de decode RTSP).
        # On ne valide qu'apres N frames consecutives.
        self._imprimante_compteur_positif: int = 0
        # Min frames consecutives requises avant de declencher (1 = pas de filtre,
        # 2 = strict mais peut rater des tickets brefs, vu que YOLO tourne ~1-2fps).
        # On garde 1 + sanity HEVC qui suffit pour filtrer les glitches.
        self._imprimante_min_frames_consecutives: int = 1
        # Timestamp du dernier snapshot OBS sauvegarde (cooldown anti-spam)
        self._imprimante_obs_last_snapshot_ts: float = 0.0
        self._imprimante_obs_snapshot_cooldown: float = 10.0  # secondes
        self.nb_cycles_scan_min = nb_cycles_scan_min
        self.cooldown = cooldown_secondes
        self.detecter_transaction_fantome = detecter_transaction_fantome

        # Cache du resultat de detection visuelle pour la frame courante.
        # Calcule une seule fois par appel a analyser(), reutilise par OBS + state machine.
        self._ticket_visuel_cette_frame: Optional[bool] = None

        # Transactions actives par ID caissier
        self._transactions: Dict[int, TransactionCaisse] = {}

        # Historiques de positions de mains pour detecter les mouvements
        # id_piste -> deque[(timestamp, main_g, main_d)]
        self._historique_mains: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=60)
        )

        # Historique de la zone imprimante (mouvement detecte)
        # id_piste -> deque[(timestamp, mouvement_detecte)]
        self._historique_imprimante: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )

        # Pistes identifiees comme caissiers (restent dans la zone caisse longtemps)
        self._caissiers_potentiels: Dict[int, float] = {}  # id -> temps premiere detection

        # Cooldowns pour les alertes
        self._derniere_alerte: Dict[Tuple[int, str], float] = {}

    def _est_dans_zone_caisse(
        self, centre: Tuple[float, float], taille_frame: Tuple[int, int]
    ) -> bool:
        """Verifie si un point est dans la zone caisse."""
        h, w = taille_frame
        x, y = centre
        return (
            self.zone_caisse_y_min_pct * h <= y <= self.zone_caisse_y_max_pct * h
            and self.zone_caisse_x_min_pct * w <= x <= self.zone_caisse_x_max_pct * w
        )

    def _main_est_etendue(
        self, main: np.ndarray, centre_torse: np.ndarray, bbox_h: float
    ) -> bool:
        """Verifie si la main est etendue loin du torse (geste de douchette)."""
        if bbox_h <= 0:
            return False
        dist = np.linalg.norm(main - centre_torse) / bbox_h
        return dist > 0.35  # Main etendue a plus de 35% de la hauteur bbox

    def _identifier_caissiers(
        self,
        pistes: List[PisteSuivi],
        taille_frame: Tuple[int, int],
    ) -> List[int]:
        """
        Identifie les caissiers: personnes stationnaires dans la zone caisse.
        Un caissier reste dans la zone caisse pendant au moins 30 secondes.
        """
        caissiers = []
        maintenant = time.time()

        for piste in pistes:
            if piste.etat != "actif":
                continue

            dans_caisse = self._est_dans_zone_caisse(piste.centre, taille_frame)

            if dans_caisse:
                if piste.id_piste not in self._caissiers_potentiels:
                    self._caissiers_potentiels[piste.id_piste] = maintenant
                elif maintenant - self._caissiers_potentiels[piste.id_piste] > 30:
                    caissiers.append(piste.id_piste)
            else:
                # Si la personne quitte la zone caisse, elle n'est plus caissier
                self._caissiers_potentiels.pop(piste.id_piste, None)

        return caissiers

    def _identifier_client_caisse(
        self,
        pistes: List[PisteSuivi],
        id_caissier: int,
        taille_frame: Tuple[int, int],
    ) -> Optional[int]:
        """
        Identifie le client en face du caissier.
        Le client est la personne la plus proche du caissier dans la zone caisse
        qui n'est pas un autre caissier.
        """
        caissier = None
        for p in pistes:
            if p.id_piste == id_caissier:
                caissier = p
                break
        if caissier is None:
            return None

        meilleur_dist = float("inf")
        meilleur_id = None

        for piste in pistes:
            if piste.id_piste == id_caissier or piste.etat != "actif":
                continue
            # Le client doit etre a proximite de la zone caisse
            h, w = taille_frame
            cy = piste.centre[1]
            if cy < self.zone_caisse_y_min_pct * h * 0.85:
                continue  # Trop loin de la caisse

            dist = np.sqrt(
                (piste.centre[0] - caissier.centre[0]) ** 2
                + (piste.centre[1] - caissier.centre[1]) ** 2
            )
            if dist < meilleur_dist:
                meilleur_dist = dist
                meilleur_id = piste.id_piste

        return meilleur_id

    def _detecter_mouvement_scan(
        self,
        id_piste: int,
        pose: Optional[PoseKeypoints],
        taille_frame: Tuple[int, int],
    ) -> bool:
        """
        Detecte un mouvement de scan avec douchette (scanner a main).

        Pattern: le caissier prend la douchette, etend la main vers l'article
        (vetement pret-a-porter), puis ramene la main vers lui. Ce mouvement
        d'extension-retraction se repete pour chaque article.

        Detection: alternance main etendue / main proche du corps dans la zone caisse.
        """
        if pose is None:
            return False

        main_g, main_d = pose.obtenir_position_mains()
        centre_torse = pose.obtenir_centre_torse()
        maintenant = time.time()

        # Enregistrer la position actuelle
        self._historique_mains[id_piste].append((maintenant, main_g, main_d))

        if centre_torse is None:
            return False

        hist = list(self._historique_mains[id_piste])
        if len(hist) < 8:
            return False

        # Calculer la hauteur de la bbox du caissier pour normaliser
        # On cherche dans les pistes mais on n'a pas acces ici,
        # donc on utilise la distance epaule-hanche comme reference
        ep_g = pose.keypoints[pose.EPAULE_GAUCHE]
        ha_g = pose.keypoints[pose.HANCHE_GAUCHE]
        if ep_g[2] > 0.3 and ha_g[2] > 0.3:
            ref_h = abs(ha_g[1] - ep_g[1]) * 2  # Approximation hauteur personne
        else:
            ref_h = taille_frame[0] * 0.3  # Fallback

        # Analyser chaque main pour le pattern extension-retraction
        for cle_main in range(2):  # 0=gauche, 1=droite
            distances = []
            for _, mg, md in hist[-15:]:
                m = mg if cle_main == 0 else md
                if m is not None:
                    dist = np.linalg.norm(m - centre_torse) / max(ref_h, 1)
                    distances.append(dist)

            if len(distances) < 6:
                continue

            # Compter les alternances proche/etendu
            # Proche du corps: dist < 0.20 | Etendu: dist > 0.35
            seuil_proche = 0.20
            seuil_etendu = 0.35
            alternances = 0
            etait_etendu = False

            for d in distances:
                if d > seuil_etendu and not etait_etendu:
                    etait_etendu = True
                elif d < seuil_proche and etait_etendu:
                    alternances += 1
                    etait_etendu = False

            # Cycles extension-retraction = scan avec douchette
            if alternances >= self.nb_cycles_scan_min:
                # Verifier la vitesse du mouvement (doit etre assez rapide)
                vitesses = []
                for i in range(1, len(distances)):
                    vitesses.append(abs(distances[i] - distances[i - 1]))
                vitesse_moy = np.mean(vitesses) if vitesses else 0

                if vitesse_moy > 0.02:  # Mouvement significatif
                    return True

        return False

    def _detecter_impression_ticket(
        self,
        id_caissier: int,
        pose: Optional[PoseKeypoints],
        taille_frame: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Detecte l'impression du ticket via 2 methodes combinees:

        1. Detection visuelle (prioritaire si zone imprimante configuree):
           Utilise le resultat cache de _detecter_papier_imprimante
           (calcule une seule fois par frame dans analyser()).

        2. Analyse des mains du caissier (fallback):
           Detecte un geste de prise (descente + montee de la main).
        """
        if self._imprimante_configuree and self._ticket_visuel_cette_frame is not None:
            return self._ticket_visuel_cette_frame

        # Methode 2: Analyse du mouvement des mains (fallback)
        return self._detecter_prise_ticket_mains(id_caissier, pose)

    def _detecter_papier_imprimante(
        self,
        frame: np.ndarray,
        taille_frame: Tuple[int, int],
    ) -> Optional[bool]:
        """
        Detection visuelle du papier ticket dans la zone imprimante.

        Principe:
        - On extrait la ROI de l'imprimante
        - On la convertit en niveaux de gris
        - On compare avec la frame de reference (imprimante vide)
        - Si beaucoup de pixels deviennent blancs/clairs = papier qui sort

        Returns:
            True si papier detecte, False si pas de papier, None si indetermine
        """
        # Sanity check : si la frame entiere est anormalement claire (mean > 200),
        # c'est presque toujours une corruption HEVC du flux RTSP (decode error
        # type "Could not find ref with POC X"). On ignore pour eviter un faux
        # positif geant qui declenche une fausse alerte ticket.
        frame_mean_global = float(frame.mean())
        if frame_mean_global > 200.0:
            logger.debug(
                f"_detecter_papier_imprimante: frame ignoree, mean global "
                f"{frame_mean_global:.0f} > 200 (corruption HEVC probable)"
            )
            self._imprimante_compteur_positif = 0  # reset au premier glitch
            return None

        # Extraire la ROI de l'imprimante
        x1, y1, x2, y2 = self._obtenir_roi_imprimante(taille_frame)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        roi = frame[y1:y2, x1:x2]
        import cv2
        roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        maintenant = time.time()

        # Initialisation de la reference
        if self._imprimante_ref is None:
            self._imprimante_ref = roi_gris.copy()
            self._imprimante_ref_ts = maintenant
            return None

        # Reprise si la forme change (changement de resolution)
        if self._imprimante_ref.shape != roi_gris.shape:
            self._imprimante_ref = roi_gris.copy()
            self._imprimante_ref_ts = maintenant
            return None

        # Difference absolue
        diff = cv2.absdiff(roi_gris, self._imprimante_ref)

        # Refresh intelligent: ne rafraichit la reference que si la scene est stable
        # (ROI identique a la ref actuelle) ET que le timeout (180s) est depasse.
        # Evite de capturer un ticket/main dans la reference.
        if (maintenant - self._imprimante_ref_ts) > 180.0:
            diff_moyenne = float(np.mean(diff))
            if diff_moyenne < 8.0:  # Scene quasi-identique => safe de refresh
                self._imprimante_ref = roi_gris.copy()
                self._imprimante_ref_ts = maintenant
                return None

        # Pixels qui ont significativement change ET sont devenus clairs (papier blanc)
        masque_change = diff > 30  # Changement significatif
        masque_blanc = roi_gris > self.imprimante_seuil_blanc  # Pixel clair
        masque_papier = masque_change & masque_blanc

        # QW1: appliquer le masque de polygone (exclure zone main caissier).
        # On sauvegarde le ratio non-masque pour logger l'impact du QW1 a posteriori.
        ratio_papier_unmasked = float(
            np.count_nonzero(masque_papier) / max(roi_gris.size, 1)
        )
        mask_inclus = self._obtenir_mask_imprimante(roi_gris.shape)
        if mask_inclus is not None:
            mask_bool = mask_inclus.astype(bool)
            masque_change = masque_change & mask_bool
            masque_blanc = masque_blanc & mask_bool
            masque_papier = masque_papier & mask_bool
            nb_pixels = int(np.count_nonzero(mask_bool))
        else:
            nb_pixels = roi_gris.size
        if nb_pixels == 0:
            return None
        ratio_papier = np.count_nonzero(masque_papier) / nb_pixels

        # QW1: si le mask filtre une detection qui serait positive sans lui, on log
        if (
            mask_inclus is not None
            and ratio_papier_unmasked > self.imprimante_seuil_changement
            and ratio_papier <= self.imprimante_seuil_changement
        ):
            logger.info(
                f"[QW1] Detection filtree par mask: ratio non-masque "
                f"{ratio_papier_unmasked:.1%} > seuil, ratio masque "
                f"{ratio_papier:.1%} <= seuil"
            )

        # Pre-detection brute (1 seule frame). On confirmera avec le compteur
        # consecutif ci-dessous pour eviter les declenchements sur 1 frame
        # corrompue HEVC ou un flash de luminosite isole.
        detecte_brut = False
        raison = ""

        if ratio_papier > self.imprimante_seuil_changement:
            detecte_brut = True
            raison = f"{ratio_papier:.1%} pixels blancs nouveaux"

        # Methode complementaire: detecter un objet blanc allonge (ticket)
        # Chercher des pixels blancs en bande verticale (ticket = rectangle blanc etroit).
        # masque_blanc deja masque par mask_inclus ci-dessus si QW1 actif.
        if not detecte_brut:
            nb_blancs = np.count_nonzero(masque_blanc)
            ratio_blanc = nb_blancs / nb_pixels
            if ratio_blanc > 0.3:
                coords_blanc = np.where(masque_blanc)
                if len(coords_blanc[0]) > 0:
                    y_range = coords_blanc[0].max() - coords_blanc[0].min()
                    x_range = coords_blanc[1].max() - coords_blanc[1].min()
                    if y_range > x_range * 1.5:  # Plus haut que large = ticket
                        detecte_brut = True
                        raison = f"forme allongee {ratio_blanc:.1%} blanc, H/W={y_range}/{x_range}"

        # Confirmation par N frames consecutives positives
        if detecte_brut:
            self._imprimante_compteur_positif += 1
            if self._imprimante_compteur_positif >= self._imprimante_min_frames_consecutives:
                logger.info(
                    f"Papier detecte dans l'imprimante (confirme apres "
                    f"{self._imprimante_compteur_positif} frames): {raison}"
                )
                self._imprimante_compteur_positif = 0  # reset pour la prochaine detection
                return True
            else:
                logger.debug(
                    f"Papier candidat (frame {self._imprimante_compteur_positif}/"
                    f"{self._imprimante_min_frames_consecutives}): {raison}"
                )
                return None  # En attente de confirmation
        else:
            self._imprimante_compteur_positif = 0  # reset si frame normale

        return False

    def _detecter_prise_ticket_mains(
        self,
        id_caissier: int,
        pose: Optional[PoseKeypoints],
    ) -> bool:
        """
        Detecte la prise du ticket par le mouvement des mains.
        Fallback quand la zone imprimante n'est pas configuree.

        Pattern: main qui descend vers le comptoir puis remonte (prise).
        """
        if pose is None:
            return False

        main_g, main_d = pose.obtenir_position_mains()
        centre_torse = pose.obtenir_centre_torse()

        if centre_torse is None:
            return False

        hist = list(self._historique_mains.get(id_caissier, []))
        if len(hist) < 8:
            return False

        # Analyser chaque main pour un mouvement de prise
        for cle_main, main in enumerate([main_g, main_d]):
            if main is None:
                continue

            # Extraire les positions Y recentes de cette main
            y_positions = []
            x_positions = []
            for _, mg, md in hist[-12:]:
                m = mg if cle_main == 0 else md
                if m is not None:
                    y_positions.append(m[1])
                    x_positions.append(m[0])

            if len(y_positions) < 6:
                continue

            # Mouvement localise: faible deplacement horizontal
            dx = max(x_positions) - min(x_positions)
            dy = max(y_positions) - min(y_positions)

            if dx > dy * 2:
                continue  # Trop de mouvement lateral, pas une prise de ticket

            if dy < 10:
                continue  # Pas assez de mouvement vertical

            # Pattern descente puis montee (prise du ticket)
            mid = len(y_positions) // 3
            phase1 = y_positions[:mid]  # Debut: main en position haute
            phase2 = y_positions[mid:2 * mid]  # Milieu: main descend
            phase3 = y_positions[2 * mid:]  # Fin: main remonte

            if not phase1 or not phase2 or not phase3:
                continue

            moy1 = np.mean(phase1)
            moy2 = np.mean(phase2)
            moy3 = np.mean(phase3)

            # Descente puis montee: moy2 > moy1 et moy3 < moy2
            # (Y augmente vers le bas dans l'image)
            if moy2 > moy1 + 5 and moy3 < moy2 - 5:
                return True

        return False

    def _obtenir_mask_imprimante(self, roi_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        QW1: Retourne le masque binaire (uint8, 0/1) de la zone d'inclusion dans
        la ROI. Les pixels a 1 sont pris en compte dans la detection, ceux a 0
        sont exclus (zone ou la main du caissier passe regulierement).

        Cache le masque tant que la shape de ROI ne change pas. None si pas de
        polygone configure.
        """
        if not self._imprimante_mask_polygon:
            return None
        if self._imprimante_mask is not None and self._imprimante_mask_shape == roi_shape:
            return self._imprimante_mask

        import cv2
        h, w = roi_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            pts = np.array(
                [
                    (int(round(max(0.0, min(1.0, x)) * (w - 1))),
                     int(round(max(0.0, min(1.0, y)) * (h - 1))))
                    for x, y in self._imprimante_mask_polygon
                ],
                dtype=np.int32,
            )
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
        except (TypeError, ValueError) as e:
            logger.warning(f"[QW1] Polygone invalide ({e}), fallback ROI complete")
            mask[:, :] = 1
        self._imprimante_mask = mask
        self._imprimante_mask_shape = roi_shape
        ratio_inclus = float(mask.mean())
        logger.info(
            f"[QW1] Mask binaire calcule pour ROI {w}x{h}: "
            f"{ratio_inclus:.0%} des pixels inclus dans la detection"
        )
        return mask

    def _obtenir_roi_imprimante(self, taille_frame: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Retourne les coordonnees (x1,y1,x2,y2) de la zone imprimante."""
        h, w = taille_frame
        if self._imprimante_bbox_abs is None:
            return 0, 0, 0, 0
        x1 = max(0, int(self._imprimante_bbox_abs[0]))
        y1 = max(0, int(self._imprimante_bbox_abs[1]))
        x2 = min(w, int(self._imprimante_bbox_abs[2]))
        y2 = min(h, int(self._imprimante_bbox_abs[3]))
        return x1, y1, x2, y2

    def mettre_a_jour_ref_imprimante(self, frame: np.ndarray, taille_frame: Tuple[int, int]):
        """Force la mise a jour de la reference imprimante (imprimante vide)."""
        if not self._imprimante_configuree:
            return
        x1, y1, x2, y2 = self._obtenir_roi_imprimante(taille_frame)
        import cv2
        self._imprimante_ref = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        self._imprimante_ref_ts = time.time()
        logger.info("Reference imprimante mise a jour")

    def _detecter_remise_ticket(
        self,
        pistes: List[PisteSuivi],
        poses: Dict[int, PoseKeypoints],
        id_caissier: int,
        id_client: Optional[int],
        taille_frame: Tuple[int, int],
    ) -> bool:
        """
        Detecte la remise du ticket: les mains du caissier et du client
        se rapprochent l'une de l'autre (echange physique).
        """
        if id_client is None:
            return False

        pose_caissier = poses.get(id_caissier)
        pose_client = poses.get(id_client)

        if pose_caissier is None or pose_client is None:
            return False

        main_caissier_g, main_caissier_d = pose_caissier.obtenir_position_mains()
        main_client_g, main_client_d = pose_client.obtenir_position_mains()

        h, w = taille_frame
        diag = np.sqrt(h ** 2 + w ** 2)
        seuil = diag * self.seuil_proximite_mains

        # Verifier si une main du caissier est proche d'une main du client
        mains_caissier = [m for m in [main_caissier_g, main_caissier_d] if m is not None]
        mains_client = [m for m in [main_client_g, main_client_d] if m is not None]

        for mc in mains_caissier:
            for mcl in mains_client:
                dist = np.linalg.norm(mc - mcl)
                if dist < seuil:
                    return True

        return False

    def _client_quitte_caisse(
        self,
        pistes: List[PisteSuivi],
        id_client: Optional[int],
        taille_frame: Tuple[int, int],
    ) -> bool:
        """Detecte si le client s'eloigne de la zone caisse."""
        if id_client is None:
            return False

        for piste in pistes:
            if piste.id_piste == id_client:
                h, w = taille_frame
                # Le client quitte: il est sorti de la zone caisse vers le haut
                # (vers la sortie du magasin)
                if piste.centre[1] < self.zone_caisse_y_min_pct * h * 0.8:
                    return True
                # Ou il se deplace rapidement loin de la caisse
                if piste.vitesse_moyenne > 30 and not self._est_dans_zone_caisse(
                    piste.centre, taille_frame
                ):
                    return True
                break

        return False

    def _verifier_cooldown(self, id_caissier: int, type_a: TypeAlerteCaisse) -> bool:
        """Verifie le cooldown avant d'emettre une alerte."""
        cle = (id_caissier, type_a.value)
        if cle in self._derniere_alerte:
            if time.time() - self._derniere_alerte[cle] < self.cooldown:
                return False
        return True

    def _emettre_alerte(
        self, type_a: TypeAlerteCaisse, transaction: TransactionCaisse, confiance: float
    ) -> Optional[AlerteCaisse]:
        """Emet une alerte si le cooldown est respecte."""
        if type_a in transaction.alertes_emises:
            return None

        if not self._verifier_cooldown(transaction.id_caissier, type_a):
            return None

        self._derniere_alerte[(transaction.id_caissier, type_a.value)] = time.time()
        transaction.alertes_emises.append(type_a)

        return AlerteCaisse(
            type_alerte=type_a,
            confiance=confiance,
            id_caissier=transaction.id_caissier,
            id_client=transaction.id_client,
            description=DESCRIPTIONS_ALERTES_CAISSE[type_a],
        )

    def analyser(
        self,
        pistes: List[PisteSuivi],
        poses: Dict[int, PoseKeypoints],
        detections_objets: List[Detection],
        taille_frame: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
    ) -> List[AlerteCaisse]:
        """
        Analyse principale: detecte les fraudes a la caisse.

        Args:
            pistes: Toutes les pistes actives du tracker
            poses: Poses estimees par ID de piste
            detections_objets: Objets detectes dans la frame
            taille_frame: (hauteur, largeur)

        Returns:
            Liste des alertes caisse generees
        """
        alertes = []
        maintenant = time.time()

        # === DETECTION VISUELLE IMPRIMANTE (une seule fois par frame) ===
        # Resultat cache dans _ticket_visuel_cette_frame, reutilise par
        # l'observation OBS et par la machine a etats (via _detecter_impression_ticket).
        self._ticket_visuel_cette_frame = None
        if self._imprimante_configuree and frame is not None:
            self._ticket_visuel_cette_frame = self._detecter_papier_imprimante(frame, taille_frame)

        # === OBSERVATION IMPRIMANTE INDEPENDANTE DU STATE MACHINE ===
        if self._ticket_visuel_cette_frame:
            logger.info(
                f"[IMPRIMANTE OBS] Ticket detecte (mode observation, "
                f"hors state machine) — {len(pistes)} pistes actives"
            )
            # Sauvegarder un snapshot annote (avec cooldown anti-spam)
            if (maintenant - self._imprimante_obs_last_snapshot_ts) >= self._imprimante_obs_snapshot_cooldown:
                try:
                    import os, cv2
                    from datetime import datetime
                    dt = datetime.now()
                    out_dir = f"/opt/fraude/snapshots/imprimante_obs/{dt.strftime('%Y-%m-%d')}"
                    os.makedirs(out_dir, exist_ok=True)
                    ts = dt.strftime("%H%M%S")
                    x1, y1, x2, y2 = self._obtenir_roi_imprimante(taille_frame)
                    annot = frame.copy()
                    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annot, "TICKET DETECTE", (x1, max(15, y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annot, dt.strftime("%Y-%m-%d %H:%M:%S"),
                                (10, annot.shape[0]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    path_full = f"{out_dir}/{ts}_full.jpg"
                    cv2.imwrite(path_full, annot, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    roi = frame[y1:y2, x1:x2]
                    cv2.imwrite(f"{out_dir}/{ts}_roi.jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    self._imprimante_obs_last_snapshot_ts = maintenant
                    logger.info(f"[IMPRIMANTE OBS] Snapshot sauvegarde: {path_full}")
                except Exception as e:
                    logger.warning(f"[IMPRIMANTE OBS] Echec sauvegarde snapshot: {e}")

            # Note: TICKET_SANS_CLIENT n'est PAS genere ici depuis l'OBS.
            # YOLO ne detecte pas fiablement les personnes depuis l'angle caisse
            # (souvent 0 pistes actives alors que des clients sont presents),
            # ce qui causait des faux positifs toutes les 30 secondes.
            # L'alerte TICKET_SANS_CLIENT est geree uniquement par la state machine
            # (quand un caissier EST identifie mais qu'aucun client n'a ete vu).

        # 1. Identifier les caissiers
        ids_caissiers = self._identifier_caissiers(pistes, taille_frame)

        for id_caissier in ids_caissiers:
            # Obtenir ou creer la transaction
            if id_caissier not in self._transactions:
                self._transactions[id_caissier] = TransactionCaisse(
                    id_caissier=id_caissier
                )

            tx = self._transactions[id_caissier]

            # Identifier le client
            id_client = self._identifier_client_caisse(
                pistes, id_caissier, taille_frame
            )
            if id_client is not None:
                tx.id_client = id_client
                tx.client_vu = True

            pose_caissier = poses.get(id_caissier)

            # --- MACHINE A ETATS ---

            # INACTIF -> SCAN_DETECTE (pose) ou ATTENTE_TICKET (mode visuel)
            if tx.etat in (EtatTransaction.INACTIF, EtatTransaction.TRANSACTION_OK):
                if self._detecter_mouvement_scan(id_caissier, pose_caissier, taille_frame):
                    tx.etat = EtatTransaction.SCAN_DETECTE
                    tx.timestamp_scan = maintenant
                    tx.nb_scans += 1
                    tx.alertes_emises = []
                    logger.debug(f"Scan detecte - caissier #{id_caissier}")
                elif self._imprimante_configuree and id_client is not None:
                    tx.etat = EtatTransaction.ATTENTE_TICKET
                    tx.timestamp_paiement = maintenant
                    tx.alertes_emises = []
                    tx.id_client = id_client
                    tx.client_vu = True
                    logger.info(
                        f"Mode visuel: client #{id_client} au comptoir "
                        f"- caissier #{id_caissier}, attente ticket"
                    )

            # SCAN_DETECTE -> PAIEMENT_DETECTE (apres un delai court, on considere
            # que le paiement a eu lieu si le client est toujours la)
            elif tx.etat == EtatTransaction.SCAN_DETECTE:
                # Detecter d'eventuels scans supplementaires
                if self._detecter_mouvement_scan(id_caissier, pose_caissier, taille_frame):
                    tx.nb_scans += 1
                    tx.timestamp_scan = maintenant

                # Apres 3 secondes de scan, considerer le paiement en cours
                if tx.timestamp_scan and maintenant - tx.timestamp_scan > 3.0:
                    tx.etat = EtatTransaction.PAIEMENT_DETECTE
                    tx.timestamp_paiement = maintenant
                    logger.debug(f"Paiement detecte - caissier #{id_caissier}")

            # PAIEMENT_DETECTE -> ATTENTE_TICKET
            elif tx.etat == EtatTransaction.PAIEMENT_DETECTE:
                tx.etat = EtatTransaction.ATTENTE_TICKET
                logger.debug(
                    f"Attente ticket - caissier #{id_caissier}, "
                    f"timeout dans {self.timeout_ticket}s"
                )

            # ATTENTE_TICKET -> TICKET_IMPRIME ou ALERTE
            elif tx.etat == EtatTransaction.ATTENTE_TICKET:
                # Verifier si le ticket est imprime
                if self._detecter_impression_ticket(
                    id_caissier, pose_caissier, taille_frame, frame=frame
                ):
                    tx.etat = EtatTransaction.TICKET_IMPRIME
                    tx.timestamp_ticket = maintenant
                    logger.info(f"Ticket imprime - caissier #{id_caissier}")

                    if self.detecter_transaction_fantome and not tx.client_vu:
                        alerte = self._emettre_alerte(
                            TypeAlerteCaisse.TICKET_SANS_CLIENT, tx, 0.85
                        )
                        if alerte:
                            alertes.append(alerte)
                            logger.warning(
                                f"ALERTE: Transaction fantome - caissier #{id_caissier} "
                                f"ticket imprime sans client"
                            )

                elif self._imprimante_configuree:
                    # Mode visuel: pas de timeout scan/paiement (pas fiable sans pose).
                    # Alerte uniquement si le client quitte sans ticket.
                    if self._client_quitte_caisse(pistes, tx.id_client, taille_frame):
                        tx.etat = EtatTransaction.ALERTE_DEPART_SANS_TICKET
                        alerte = self._emettre_alerte(
                            TypeAlerteCaisse.DEPART_SANS_TICKET, tx, 0.85
                        )
                        if alerte:
                            alertes.append(alerte)
                            logger.warning(
                                f"ALERTE: Client #{tx.id_client} quitte sans ticket - "
                                f"caissier #{id_caissier} (mode visuel)"
                            )
                    # Cleanup: reset si le client est parti depuis longtemps (120s)
                    elif (
                        tx.timestamp_paiement
                        and maintenant - tx.timestamp_paiement > 120.0
                    ):
                        logger.debug(f"Mode visuel: timeout cleanup 120s - caissier #{id_caissier}")
                        self._transactions[id_caissier] = TransactionCaisse(
                            id_caissier=id_caissier
                        )

                else:
                    # Mode pose: timeout apres delai configurable
                    if (
                        tx.timestamp_paiement
                        and maintenant - tx.timestamp_paiement > self.timeout_ticket
                    ):
                        tx.etat = EtatTransaction.ALERTE_PAS_TICKET
                        alerte = self._emettre_alerte(
                            TypeAlerteCaisse.PAIEMENT_SANS_TICKET, tx, 0.8
                        )
                        if alerte:
                            alertes.append(alerte)
                            logger.warning(
                                f"ALERTE: Paiement sans ticket - caissier #{id_caissier}"
                            )

                    # Verifier si le client part sans ticket
                    if self._client_quitte_caisse(pistes, tx.id_client, taille_frame):
                        tx.etat = EtatTransaction.ALERTE_DEPART_SANS_TICKET
                        alerte = self._emettre_alerte(
                            TypeAlerteCaisse.DEPART_SANS_TICKET, tx, 0.85
                        )
                        if alerte:
                            alertes.append(alerte)
                            logger.warning(
                                f"ALERTE: Client #{tx.id_client} quitte sans ticket - "
                                f"caissier #{id_caissier}"
                            )

            # TICKET_IMPRIME -> TICKET_REMIS ou ALERTE (client part sans prendre)
            elif tx.etat == EtatTransaction.TICKET_IMPRIME:
                if self._detecter_remise_ticket(
                    pistes, poses, id_caissier, tx.id_client, taille_frame
                ):
                    tx.etat = EtatTransaction.TICKET_REMIS
                    tx.timestamp_remise = maintenant
                    logger.debug(
                        f"Ticket remis au client #{tx.id_client} - "
                        f"caissier #{id_caissier}"
                    )

                elif self._imprimante_configuree:
                    # Mode visuel: auto-complete apres 5s (pas de pose pour
                    # detecter la remise physique du ticket)
                    if (
                        tx.timestamp_ticket
                        and maintenant - tx.timestamp_ticket > 5.0
                    ):
                        tx.etat = EtatTransaction.TRANSACTION_OK
                        logger.info(
                            f"Transaction complete (mode visuel) - "
                            f"caissier #{id_caissier}, client #{tx.id_client}"
                        )
                        self._transactions[id_caissier] = TransactionCaisse(
                            id_caissier=id_caissier
                        )

                # Client part avant que le ticket soit remis
                elif self._client_quitte_caisse(pistes, tx.id_client, taille_frame):
                    tx.etat = EtatTransaction.ALERTE_DEPART_SANS_TICKET
                    alerte = self._emettre_alerte(
                        TypeAlerteCaisse.DEPART_SANS_TICKET, tx, 0.75
                    )
                    if alerte:
                        alertes.append(alerte)

                # Timeout: le ticket est imprime mais pas remis
                elif (
                    tx.timestamp_ticket
                    and maintenant - tx.timestamp_ticket > self.timeout_ticket
                ):
                    alerte = self._emettre_alerte(
                        TypeAlerteCaisse.PAIEMENT_SANS_TICKET, tx, 0.7
                    )
                    if alerte:
                        alertes.append(alerte)
                    # Reset pour la prochaine transaction
                    tx.etat = EtatTransaction.INACTIF

            # TICKET_REMIS -> TRANSACTION_OK
            elif tx.etat == EtatTransaction.TICKET_REMIS:
                tx.etat = EtatTransaction.TRANSACTION_OK
                logger.info(
                    f"Transaction complete - caissier #{id_caissier}, "
                    f"client #{tx.id_client}, {tx.nb_scans} scans"
                )
                # Reset pour la prochaine transaction
                self._transactions[id_caissier] = TransactionCaisse(
                    id_caissier=id_caissier
                )

            # ALERTE -> reset quand nouveau client ou inactivite prolongee
            elif tx.etat in (
                EtatTransaction.ALERTE_PAS_TICKET,
                EtatTransaction.ALERTE_DEPART_SANS_TICKET,
            ):
                # Reset si nouveau client detecte (ID different)
                nouveau_client = self._identifier_client_caisse(
                    pistes, id_caissier, taille_frame
                )
                ancien_client = tx.id_client
                client_change = (
                    nouveau_client is not None
                    and ancien_client is not None
                    and nouveau_client != ancien_client
                )
                temps_depuis_alerte = maintenant - (
                    tx.timestamp_paiement or tx.timestamp_scan or maintenant
                )
                if client_change or temps_depuis_alerte > 30:
                    self._transactions[id_caissier] = TransactionCaisse(
                        id_caissier=id_caissier
                    )

        return alertes

    def obtenir_etat_transactions(self) -> Dict[int, Dict]:
        """Retourne l'etat actuel de toutes les transactions (pour le dashboard)."""
        etats = {}
        for id_caissier, tx in self._transactions.items():
            etats[id_caissier] = {
                "id_caissier": tx.id_caissier,
                "id_client": tx.id_client,
                "etat": tx.etat.value,
                "nb_scans": tx.nb_scans,
                "client_vu": tx.client_vu,
                "timestamp_scan": tx.timestamp_scan,
                "timestamp_paiement": tx.timestamp_paiement,
                "timestamp_ticket": tx.timestamp_ticket,
                "timestamp_remise": tx.timestamp_remise,
                "alertes": [a.value for a in tx.alertes_emises],
            }
        return etats

    def nettoyer(self, ids_actifs: List[int]):
        """Nettoie les donnees des pistes disparues."""
        ids_set = set(ids_actifs)
        for stockage in [self._historique_mains, self._historique_imprimante]:
            for k in list(stockage.keys()):
                if k not in ids_set:
                    del stockage[k]

        for k in list(self._caissiers_potentiels.keys()):
            if k not in ids_set:
                del self._caissiers_potentiels[k]

        for k in list(self._transactions.keys()):
            if k not in ids_set:
                del self._transactions[k]

"""
Analyseur de comportements suspects en magasin.
Detecte 2 types de vol (cacher article sous vetements, dissimuler dans sac)
en combinant la detection d'objets, le suivi et l'estimation de pose.

Logique de detection:
- CACHER_ARTICLE: main dans la zone de dissimulation (hanches/taille) +
  mouvement rentrant (main s'eloigne puis revient vers le corps)
- DISSIMULER_SAC: main pres d'un sac detecte + alternance main-etagere-sac

Tous les seuils sont configurables via la table parametres (categorie "vol").
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


class TypeComportement(Enum):
    """Les 6 comportements suspects détectés par le système."""
    CACHER_ARTICLE = "cacher_article"
    DISSIMULER_SAC = "dissimuler_sac"
    # Fraude caisse
    SCAN_SANS_TICKET = "scan_sans_ticket"
    PAIEMENT_SANS_TICKET = "paiement_sans_ticket"
    DEPART_SANS_TICKET = "depart_sans_ticket"
    TICKET_SANS_CLIENT = "ticket_sans_client"


# Description en français pour les alertes et le dashboard
DESCRIPTIONS_COMPORTEMENTS = {
    TypeComportement.CACHER_ARTICLE: "Cache un article sous ses vetements",
    TypeComportement.DISSIMULER_SAC: "Dissimule un article dans un sac",
    TypeComportement.SCAN_SANS_TICKET: "Article scanne mais pas de ticket imprime",
    TypeComportement.PAIEMENT_SANS_TICKET: "Paiement sans remise de ticket",
    TypeComportement.DEPART_SANS_TICKET: "Client quitte la caisse sans ticket",
    TypeComportement.TICKET_SANS_CLIENT: "Transaction fantome: ticket imprime sans client present",
}


@dataclass
class ResultatAnalyse:
    """Résultat de l'analyse d'un comportement suspect."""
    type_comportement: TypeComportement
    confiance: float  # 0.0 à 1.0
    id_piste: int
    description: str
    horodatage: float = field(default_factory=time.time)


@dataclass
class ScoreSuspicion:
    """Score de suspicion agrégé pour une personne suivie."""
    id_piste: int
    score_global: float = 0.0
    scores_par_type: Dict[TypeComportement, float] = field(default_factory=dict)
    alertes_declenchees: List[TypeComportement] = field(default_factory=list)

    def est_suspect(self, seuil: float = 0.6) -> bool:
        return self.score_global >= seuil


class AnalyseurComportements:
    """
    Analyse les comportements suspects en combinant :
    - Positions des mains et corps (pose estimation)
    - Detection d'objets (YOLO: sacs)
    - Mouvement directionnel (main qui revient vers le corps)

    Detecte 2 types de vol avec accumulation temporelle.
    Tous les seuils sont charges depuis la DB (categorie "vol").
    """

    POIDS_COMPORTEMENTS = {
        TypeComportement.CACHER_ARTICLE: 0.9,
        TypeComportement.DISSIMULER_SAC: 0.95,
    }

    def __init__(
        self,
        seuil_alerte: float = 0.6,
        cooldown_secondes: int = 30,
        db=None,
    ):
        self.seuil_alerte = seuil_alerte
        self.cooldown = cooldown_secondes

        # Charger les parametres depuis la DB (avec defauts)
        self._charger_parametres(db)

        # id_piste -> historique des observations de mains
        self._historique_mains: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=60)
        )

        # Cooldown par piste et type de comportement
        self._derniere_alerte: Dict[Tuple[int, str], float] = {}

        # Scores de suspicion accumules
        self._scores: Dict[int, Dict[TypeComportement, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Timestamps pour normaliser le decay au temps reel
        self._derniers_timestamps: Dict[tuple, float] = {}

    def _charger_parametres(self, db):
        """Charge les parametres de detection depuis la DB."""
        def _p(cle, defaut):
            if db is None:
                return defaut
            try:
                return type(defaut)(db.obtenir_parametre(cle, defaut))
            except (ValueError, TypeError):
                return defaut

        # Cacher article sous vetements
        self.dist_main_corps = _p("vol_distance_main_corps", 0.25)
        self.zone_dissim_haut = _p("vol_zone_dissimulation_haut", 0.3)
        self.zone_dissim_bas = _p("vol_zone_dissimulation_bas", 0.85)
        self.incr_main_corps = _p("vol_increment_main_corps", 0.12)
        self.incr_mouvement = _p("vol_increment_mouvement_rentrant", 0.15)
        self.incr_objet = _p("vol_increment_objet_proche", 0.10)
        self.ratio_rapprochement = _p("vol_ratio_rapprochement", 0.6)
        self.hist_min_frames = _p("vol_historique_min_frames", 5)

        # Dissimuler dans sac
        self.sac_incr_base = _p("vol_sac_increment_base", 0.10)
        self.sac_incr_alternance = _p("vol_sac_increment_alternance", 0.15)
        self.sac_alternances_min = _p("vol_sac_alternances_min", 2)
        self.sac_distance_ratio = _p("vol_sac_distance_ratio", 0.8)

        # Decay
        self.decay_rate = _p("vol_decay_rate", 0.95)
        self.fps_ref = _p("vol_fps_ref", 2.0)

        logger.info(
            f"Params vol charges: dist={self.dist_main_corps} "
            f"zone=[{self.zone_dissim_haut}-{self.zone_dissim_bas}] "
            f"incr_corps={self.incr_main_corps} incr_mvt={self.incr_mouvement} "
            f"decay={self.decay_rate} fps_ref={self.fps_ref}"
        )

    # =============================================
    # Interface publique
    # =============================================

    def analyser(
        self,
        piste: PisteSuivi,
        pose: Optional[PoseKeypoints],
        detections_objets: List[Detection],
        taille_frame: Tuple[int, int],
    ) -> List[ResultatAnalyse]:
        """Analyse complete d'une personne suivie pour tous les comportements."""
        resultats = []

        r = self._analyser_cacher_article(piste, pose, detections_objets)
        if r:
            resultats.append(r)

        r = self._analyser_dissimulation_sac(piste, pose, detections_objets)
        if r:
            resultats.append(r)

        return resultats

    def obtenir_score_suspicion(self, id_piste: int) -> ScoreSuspicion:
        """Score de suspicion agrege. MAX(score x poids) + bonus multi-comportements."""
        scores_types = dict(self._scores.get(id_piste, {}))

        if not scores_types:
            return ScoreSuspicion(id_piste=id_piste)

        scores_ponderes = {
            t: s * self.POIDS_COMPORTEMENTS.get(t, 0.5)
            for t, s in scores_types.items()
        }
        score_global = max(scores_ponderes.values()) if scores_ponderes else 0.0

        nb_actifs = sum(1 for sp in scores_ponderes.values() if sp > 0.25)
        if nb_actifs >= 2:
            score_global = min(1.0, score_global * (1.0 + 0.15 * (nb_actifs - 1)))

        score_global = min(1.0, max(0.0, score_global))

        alertes = [t for t, s in scores_types.items() if s >= self.seuil_alerte]

        return ScoreSuspicion(
            id_piste=id_piste,
            score_global=score_global,
            scores_par_type=scores_types,
            alertes_declenchees=alertes,
        )

    # =============================================
    # Cooldown & Score accumulation
    # =============================================

    def _verifier_cooldown(self, id_piste: int, type_c: TypeComportement) -> bool:
        cle = (id_piste, type_c.value)
        if cle in self._derniere_alerte:
            elapsed = time.time() - self._derniere_alerte[cle]
            if elapsed < self.cooldown:
                return False
        return True

    def _marquer_alerte(self, id_piste: int, type_c: TypeComportement):
        self._derniere_alerte[(id_piste, type_c.value)] = time.time()

    def _accumuler_score(
        self,
        id_piste: int,
        type_c: TypeComportement,
        increment: float,
    ):
        """
        Accumule le score avec decroissance temporelle normalisee.
        La decroissance est basee sur le temps reel pour etre FPS-independante.
        decay^(dt * fps_ref): a fps_ref=2, 1 seconde = decay^2.
        """
        score_actuel = self._scores[id_piste][type_c]

        cle_ts = (id_piste, type_c)
        maintenant = time.time()
        dernier_ts = self._derniers_timestamps.get(cle_ts, maintenant)
        dt = maintenant - dernier_ts
        self._derniers_timestamps[cle_ts] = maintenant

        exposant = dt * self.fps_ref
        if exposant > 0:
            score_actuel *= self.decay_rate ** exposant

        score_actuel = min(1.0, score_actuel + increment)
        self._scores[id_piste][type_c] = score_actuel

    # =============================================
    # COMPORTEMENT 1: Cacher article sous vetements
    # =============================================

    def _analyser_cacher_article(
        self,
        piste: PisteSuivi,
        pose: Optional[PoseKeypoints],
        detections_objets: List[Detection],
    ) -> Optional[ResultatAnalyse]:
        """
        Detecte quand une personne cache un article sous ses vetements.

        Indicateurs (tous bases sur la pose, pas sur YOLO article):
        1. Main dans la ZONE DE DISSIMULATION (entre hanches et taille, pas juste "pres du torse")
           → Zone = entre zone_dissim_haut (30%) et zone_dissim_bas (85%) de la bbox
        2. MOUVEMENT RENTRANT: la main etait eloignee du corps puis se rapproche
           → Detecte le geste "prendre sur etagere → cacher sur soi"
        3. BONUS objet: un objet YOLO est detecte pres de la main
        """
        if pose is None:
            self._accumuler_score(piste.id_piste, TypeComportement.CACHER_ARTICLE, 0.0)
            return None

        main_g, main_d = pose.obtenir_position_mains()
        centre_hanches = pose.obtenir_centre_hanches()
        centre_torse = pose.obtenir_centre_torse()

        # Besoin d'au moins un point de reference corps
        ref_corps = centre_hanches if centre_hanches is not None else centre_torse
        if ref_corps is None:
            self._accumuler_score(piste.id_piste, TypeComportement.CACHER_ARTICLE, 0.0)
            return None

        bbox_h = piste.bbox[3] - piste.bbox[1]
        bbox_y_top = piste.bbox[1]
        if bbox_h <= 0:
            return None

        score_increment = 0.0

        for main in [main_g, main_d]:
            if main is None:
                continue

            # Position verticale de la main dans la bbox (0=haut, 1=bas)
            main_y_rel = (main[1] - bbox_y_top) / bbox_h

            # Distance horizontale main-corps normalisee
            dist_corps = np.linalg.norm(main - ref_corps) / bbox_h

            # === INDICATEUR 1: Main dans la zone de dissimulation ===
            # La main doit etre:
            # - Dans la zone verticale (entre epaules et genoux)
            # - Proche du corps horizontalement
            dans_zone_verticale = self.zone_dissim_haut <= main_y_rel <= self.zone_dissim_bas
            proche_corps = dist_corps < self.dist_main_corps

            if dans_zone_verticale and proche_corps:
                score_increment += self.incr_main_corps

                # === INDICATEUR 3: Objet YOLO pres de la main ===
                for obj in detections_objets:
                    if obj.class_name in ("sac_a_main", "sac_a_dos", "valise"):
                        continue  # Les sacs sont traites par dissimuler_sac
                    ox = (obj.bbox[0] + obj.bbox[2]) / 2
                    oy = (obj.bbox[1] + obj.bbox[3]) / 2
                    dist_obj = np.sqrt((main[0] - ox) ** 2 + (main[1] - oy) ** 2)
                    if dist_obj < bbox_h * 0.3:
                        score_increment += self.incr_objet

        # Enregistrer position des mains pour analyse temporelle
        self._historique_mains[piste.id_piste].append({
            "temps": time.time(),
            "main_g": main_g.tolist() if main_g is not None else None,
            "main_d": main_d.tolist() if main_d is not None else None,
            "ref_corps": ref_corps.tolist(),
        })

        # === INDICATEUR 2: Mouvement rentrant ===
        # Main qui etait loin du corps et se rapproche = geste de dissimulation
        hist = list(self._historique_mains[piste.id_piste])
        if len(hist) >= self.hist_min_frames:
            nb_recents = max(2, self.hist_min_frames // 2)
            nb_anciens = nb_recents
            for cle_main in ["main_g", "main_d"]:
                # Positions anciennes (debut de l'historique recent)
                anciennes = [h[cle_main] for h in hist[-(nb_anciens + nb_recents):-nb_recents]
                             if h[cle_main] is not None]
                # Positions recentes (fin)
                recentes = [h[cle_main] for h in hist[-nb_recents:]
                            if h[cle_main] is not None]
                refs = [h["ref_corps"] for h in hist[-(nb_anciens + nb_recents):]
                        if h["ref_corps"] is not None]

                if anciennes and recentes and refs:
                    ref_moy = np.mean(refs, axis=0)
                    dist_avant = np.mean([np.linalg.norm(np.array(p) - ref_moy) for p in anciennes])
                    dist_apres = np.mean([np.linalg.norm(np.array(p) - ref_moy) for p in recentes])

                    # La main s'est rapprochee du corps significativement
                    if dist_avant > 0 and (dist_apres / max(dist_avant, 1)) < self.ratio_rapprochement:
                        score_increment += self.incr_mouvement

        self._accumuler_score(piste.id_piste, TypeComportement.CACHER_ARTICLE, score_increment)

        score = self._scores[piste.id_piste][TypeComportement.CACHER_ARTICLE]
        if score >= self.seuil_alerte and self._verifier_cooldown(piste.id_piste, TypeComportement.CACHER_ARTICLE):
            self._marquer_alerte(piste.id_piste, TypeComportement.CACHER_ARTICLE)
            return ResultatAnalyse(
                type_comportement=TypeComportement.CACHER_ARTICLE,
                confiance=score,
                id_piste=piste.id_piste,
                description=DESCRIPTIONS_COMPORTEMENTS[TypeComportement.CACHER_ARTICLE],
            )
        return None

    # =============================================
    # COMPORTEMENT 2: Dissimuler dans un sac
    # =============================================

    def _analyser_dissimulation_sac(
        self,
        piste: PisteSuivi,
        pose: Optional[PoseKeypoints],
        detections_objets: List[Detection],
    ) -> Optional[ResultatAnalyse]:
        """
        Detecte quand une personne met des articles dans un sac.

        Indicateurs:
        1. Main proche d'un sac detecte par YOLO (sac_a_main, sac_a_dos, valise)
        2. BONUS: alternance main loin/pres du sac (mouvement repetitif = multiple articles)
        """
        if pose is None:
            self._accumuler_score(piste.id_piste, TypeComportement.DISSIMULER_SAC, 0.0)
            return None

        main_g, main_d = pose.obtenir_position_mains()

        # Trouver les sacs dans les detections
        sacs = [d for d in detections_objets
                if d.class_name in ("sac_a_main", "sac_a_dos", "valise")]

        if not sacs:
            self._accumuler_score(piste.id_piste, TypeComportement.DISSIMULER_SAC, 0.0)
            return None

        score_increment = 0.0
        bbox_h = piste.bbox[3] - piste.bbox[1]
        if bbox_h <= 0:
            return None

        for main in [main_g, main_d]:
            if main is None:
                continue

            for sac in sacs:
                sac_cx = (sac.bbox[0] + sac.bbox[2]) / 2
                sac_cy = (sac.bbox[1] + sac.bbox[3]) / 2
                sac_taille = max(sac.bbox[2] - sac.bbox[0], sac.bbox[3] - sac.bbox[1])
                dist_main_sac = np.sqrt((main[0] - sac_cx) ** 2 + (main[1] - sac_cy) ** 2)

                # Main proche du sac
                if dist_main_sac < sac_taille * self.sac_distance_ratio:
                    # Verifier que le sac est dans la bbox de la personne
                    sac_dans_bbox = (
                        piste.bbox[0] <= sac_cx <= piste.bbox[2]
                        and piste.bbox[1] <= sac_cy <= piste.bbox[3]
                    )
                    if sac_dans_bbox:
                        score_increment += self.sac_incr_base

                        # Bonus alternance main-etagere-sac
                        hist = list(self._historique_mains.get(piste.id_piste, []))
                        if len(hist) >= self.hist_min_frames:
                            distances_sac = []
                            for h in hist[-20:]:
                                for cle in ["main_g", "main_d"]:
                                    if h.get(cle) is not None:
                                        pos = np.array(h[cle])
                                        d = np.linalg.norm(pos - np.array([sac_cx, sac_cy]))
                                        distances_sac.append(d)

                            if len(distances_sac) >= 4:
                                seuil_dist = sac_taille * 1.5
                                alternances = 0
                                for i in range(1, len(distances_sac)):
                                    if ((distances_sac[i - 1] > seuil_dist and distances_sac[i] < seuil_dist)
                                            or (distances_sac[i - 1] < seuil_dist and distances_sac[i] > seuil_dist)):
                                        alternances += 1

                                if alternances >= self.sac_alternances_min:
                                    score_increment += self.sac_incr_alternance

        self._accumuler_score(piste.id_piste, TypeComportement.DISSIMULER_SAC, score_increment)

        score = self._scores[piste.id_piste][TypeComportement.DISSIMULER_SAC]
        if score >= self.seuil_alerte and self._verifier_cooldown(piste.id_piste, TypeComportement.DISSIMULER_SAC):
            self._marquer_alerte(piste.id_piste, TypeComportement.DISSIMULER_SAC)
            return ResultatAnalyse(
                type_comportement=TypeComportement.DISSIMULER_SAC,
                confiance=score,
                id_piste=piste.id_piste,
                description=DESCRIPTIONS_COMPORTEMENTS[TypeComportement.DISSIMULER_SAC],
            )
        return None

    # =============================================
    # Nettoyage
    # =============================================

    def nettoyer_pistes_supprimees(self, ids_actifs: List[int]):
        """Nettoie les donnees des pistes qui ne sont plus suivies."""
        ids_actifs_set = set(ids_actifs)
        for stockage in [self._historique_mains, self._scores]:
            ids_a_supprimer = [k for k in stockage if k not in ids_actifs_set]
            for k in ids_a_supprimer:
                del stockage[k]

        cles_a_supprimer = [k for k in self._derniere_alerte if k[0] not in ids_actifs_set]
        for k in cles_a_supprimer:
            del self._derniere_alerte[k]

        ts_a_supprimer = [k for k in self._derniers_timestamps if k[0] not in ids_actifs_set]
        for k in ts_a_supprimer:
            del self._derniers_timestamps[k]

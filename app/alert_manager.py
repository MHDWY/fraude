"""
Gestionnaire d'alertes en temps réel.
Combine: son d'alerte, notification Telegram, enregistrement vidéo,
et journalisation en base de données.
"""

import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from .behavior_analyzer import ResultatAnalyse, TypeComportement, DESCRIPTIONS_COMPORTEMENTS
from .config import FraudeConfig
from .database import BaseDonneesFraude
from .video_recorder import EnregistreurVideo

logger = logging.getLogger(__name__)

# Regex precompile: sequence ressemblant a un token bot Telegram
# (fallback au cas ou le token exact ne serait pas dans la chaine).
_REGEX_TOKEN_TELEGRAM = re.compile(r"bot\d+:[A-Za-z0-9_\-]+")


def _masquer_token(texte: str, token: Optional[str]) -> str:
    """Masque un token Telegram dans une chaine (logs, exceptions).
    Remplace le token exact par ***TOKEN***, puis applique un regex de
    securite pour toute sequence ressemblant a un token bot Telegram."""
    if not texte:
        return texte
    if token and token in texte:
        texte = texte.replace(token, "***TOKEN***")
    return _REGEX_TOKEN_TELEGRAM.sub("bot***:***", texte)

# Niveaux de sévérité par type de comportement
SEVERITE = {
    # Vol magasin (2)
    TypeComportement.CACHER_ARTICLE: "HAUTE",
    TypeComportement.DISSIMULER_SAC: "HAUTE",
    # Fraudes caisse (6)
    TypeComportement.SCAN_SANS_TICKET: "HAUTE",
    TypeComportement.PAIEMENT_SANS_TICKET: "HAUTE",
    TypeComportement.DEPART_SANS_TICKET: "HAUTE",
}

# Emojis pour Telegram (indicatif de sévérité)
EMOJI_SEVERITE = {
    "HAUTE": "\u26a0\ufe0f",     # Warning
    "MOYENNE": "\u2757",          # Exclamation
    "BASSE": "\u2139\ufe0f",     # Info
}


class GestionnaireAlertes:
    """
    Gère le cycle de vie complet des alertes :
    1. Déclenchement avec cooldown par piste/type
    2. Son d'alerte local (non-bloquant)
    3. Notification Telegram avec snapshot
    4. Enregistrement de clip vidéo (30 secondes)
    5. Journalisation en base SQLite
    """

    def __init__(
        self,
        config: FraudeConfig,
        base_donnees: BaseDonneesFraude,
        enregistreur: EnregistreurVideo,
    ):
        self.config = config
        self.db = base_donnees
        self.enregistreur = enregistreur

        # Cooldown par (id_piste, type_comportement)
        self._dernieres_alertes: Dict[Tuple[int, str], float] = {}
        self._lock = threading.Lock()

        # File d'attente des alertes (triée par priorité)
        self._compteur_alertes = 0

        # Chemin vers le fichier son (généré si besoin)
        self._chemin_son = config.chemin_enregistrements.parent / "sounds" / "alerte.wav"
        self._preparer_son()

        logger.info("Gestionnaire d'alertes initialise")

    def _preparer_son(self):
        """Prépare le fichier son d'alerte s'il n'existe pas."""
        self._chemin_son.parent.mkdir(parents=True, exist_ok=True)

        if not self._chemin_son.exists():
            # Générer un bip simple en WAV
            self._generer_son_alerte()

    def _generer_son_alerte(self):
        """Génère un fichier son d'alerte WAV simple (bip)."""
        import struct
        import wave

        frequence = 880  # Hz (La5 - son d'alerte aigu)
        duree = 0.5  # secondes
        volume = 0.7
        taux_echantillonnage = 22050

        nb_echantillons = int(taux_echantillonnage * duree)
        donnees = []

        for i in range(nb_echantillons):
            t = i / taux_echantillonnage
            # Signal avec enveloppe (attaque rapide, décroissance)
            enveloppe = min(1.0, t * 20) * max(0.0, 1.0 - t / duree)
            valeur = volume * enveloppe * np.sin(2 * np.pi * frequence * t)
            donnees.append(int(valeur * 32767))

        try:
            with wave.open(str(self._chemin_son), "w") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(taux_echantillonnage)
                wav.writeframes(struct.pack(f"<{len(donnees)}h", *donnees))

            logger.info(f"Son d'alerte genere: {self._chemin_son}")
        except Exception as e:
            logger.warning(f"Impossible de generer le son d'alerte: {e}")

    def _obtenir_destinataires_telegram(self, camera_id: Optional[int]) -> List[str]:
        """
        Retourne les chat_ids Telegram pour une camera.
        Si aucun utilisateur assigne, fallback sur la config globale.
        """
        telegram_ids = []

        if camera_id is not None:
            utilisateurs = self.db.obtenir_utilisateurs_pour_camera(camera_id)
            for u in utilisateurs:
                if u["type_alerte"] == "telegram":
                    telegram_ids.append(u["identifiant"])

        if not telegram_ids and self.config.telegram_chat_id:
            telegram_ids = [self.config.telegram_chat_id]

        return telegram_ids

    def traiter_alerte(
        self,
        resultat: ResultatAnalyse,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        source_camera: str = "cam_principale",
        camera_id: Optional[int] = None,
        enregistreur_camera: Optional["EnregistreurVideo"] = None,
    ):
        """
        Traite une alerte détectée : vérifie le cooldown, puis déclenche
        toutes les actions (son, Telegram, vidéo, base de données).

        Args:
            resultat: Résultat de l'analyse comportementale
            frame: Frame courante pour le snapshot
            bbox: Boîte englobante de la personne
            source_camera: Nom de la caméra source
            camera_id: ID de la caméra (pour le routage des alertes par utilisateur)
            enregistreur_camera: EnregistreurVideo du worker (contient le buffer de frames)
        """
        cle_cooldown = (resultat.id_piste, resultat.type_comportement.value)

        with self._lock:
            # Vérifier le cooldown
            if cle_cooldown in self._dernieres_alertes:
                elapsed = time.time() - self._dernieres_alertes[cle_cooldown]
                if elapsed < self.config.alert_cooldown_seconds:
                    return

            self._dernieres_alertes[cle_cooldown] = time.time()
            self._compteur_alertes += 1

        severite = SEVERITE.get(resultat.type_comportement, "MOYENNE")

        logger.warning(
            f"[ALERTE {severite}] {resultat.description} "
            f"(piste #{resultat.id_piste}, confiance: {resultat.confiance:.2f}, "
            f"camera: {source_camera})"
        )

        # Utiliser l'enregistreur du worker (avec buffer) ou le global en fallback
        rec = enregistreur_camera or self.enregistreur

        # 1. Sauvegarder le snapshot (frame annotée) dans snapshots/YYYY-MM-DD/
        chemin_snapshot = ""
        if frame is not None:
            frame_annotee = self._annoter_frame_alerte(frame, resultat, bbox)
            chemin_snapshot = rec.sauvegarder_snapshot(
                frame_annotee,
                nom=f"alerte_{resultat.type_comportement.value}",
            )

        # 2. Résoudre les destinataires Telegram pour cette caméra
        telegram_ids = self._obtenir_destinataires_telegram(camera_id)

        # 3. Jouer le son d'alerte (non-bloquant)
        if self.config.alert_sound:
            threading.Thread(target=self._jouer_son, daemon=True).start()

        # 4. Envoi immédiat : Telegram photo
        notifie_telegram = False
        if self.config.telegram_actif and telegram_ids:
            for chat_id in telegram_ids:
                threading.Thread(
                    target=self._envoyer_telegram,
                    args=(resultat, chemin_snapshot, severite, chat_id),
                    daemon=True,
                ).start()
            notifie_telegram = True

        # 5. Démarrer l'enregistrement vidéo avec callback de fin
        identifiant_clip = f"piste{resultat.id_piste}_{self._compteur_alertes}"

        def _callback_video_prete(chemin_video: str):
            """Appelé quand le clip vidéo est terminé."""
            logger.info(f"Clip video pret, envoi en PJ: {Path(chemin_video).name}")
            if self.config.telegram_actif and telegram_ids:
                for chat_id in telegram_ids:
                    threading.Thread(
                        target=self._envoyer_telegram_video,
                        args=(resultat, chemin_video, severite, chat_id),
                        daemon=True,
                    ).start()

        chemin_video = rec.demarrer_enregistrement(
            identifiant=identifiant_clip,
            type_alerte=resultat.type_comportement.value,
            callback_fin=_callback_video_prete,
        ) or ""

        # 6. Enregistrer en base de données
        self.db.enregistrer_alerte(
            type_comportement=resultat.type_comportement.value,
            confiance=resultat.confiance,
            id_piste=resultat.id_piste,
            bbox=bbox,
            chemin_video=chemin_video,
            chemin_snapshot=chemin_snapshot,
            zone=self._determiner_zone(bbox, frame.shape[:2] if frame is not None else (480, 640)),
            source_camera=source_camera,
            notifie_telegram=notifie_telegram,
        )

    def _annoter_frame_alerte(
        self,
        frame: np.ndarray,
        resultat: ResultatAnalyse,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Annote la frame avec les informations de l'alerte."""
        frame_copie = frame.copy()
        severite = SEVERITE.get(resultat.type_comportement, "MOYENNE")

        # Couleurs selon la sévérité
        couleurs = {
            "HAUTE": (0, 0, 255),     # Rouge
            "MOYENNE": (0, 165, 255),  # Orange
            "BASSE": (0, 255, 255),    # Jaune
        }
        couleur = couleurs.get(severite, (0, 255, 0))

        # Dessiner la bbox si disponible
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_copie, (x1, y1), (x2, y2), couleur, 3)

            # Label au-dessus de la bbox
            label = f"[{severite}] {resultat.description}"
            taille = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                frame_copie,
                (x1, y1 - taille[1] - 10),
                (x1 + taille[0], y1),
                couleur,
                -1,
            )
            cv2.putText(
                frame_copie,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Bannière d'alerte en haut
        h, w = frame_copie.shape[:2]
        cv2.rectangle(frame_copie, (0, 0), (w, 40), couleur, -1)
        texte_banniere = (
            f"ALERTE: {resultat.type_comportement.value} | "
            f"Confiance: {resultat.confiance:.0%} | "
            f"Piste #{resultat.id_piste}"
        )
        cv2.putText(
            frame_copie,
            texte_banniere,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame_copie

    def _jouer_son(self):
        """Joue le son d'alerte de façon non-bloquante."""
        try:
            if self._chemin_son.exists():
                # Essayer playsound d'abord
                try:
                    from playsound import playsound
                    playsound(str(self._chemin_son))
                except Exception as e:
                    logger.debug(f"playsound indisponible ({e}), fallback bip systeme")
                    print("\a", end="", flush=True)
            else:
                print("\a", end="", flush=True)
        except Exception as e:
            logger.debug(f"Impossible de jouer le son: {e}")

    def _envoyer_telegram(
        self,
        resultat: ResultatAnalyse,
        chemin_snapshot: str,
        severite: str,
        chat_id: str = "",
    ):
        """
        Envoie une notification Telegram avec le snapshot a un chat_id specifique.
        Inclut un retry avec backoff et gestion du rate limit (429).
        """
        if not self.config.telegram_actif:
            return

        cible = chat_id or self.config.telegram_chat_id
        if not cible:
            return

        emoji = EMOJI_SEVERITE.get(severite, "\u2139\ufe0f")
        message = (
            f"{emoji} *ALERTE SECURITE - {severite}*\n\n"
            f"*Type:* {DESCRIPTIONS_COMPORTEMENTS.get(resultat.type_comportement, resultat.type_comportement.value)}\n"
            f"*Confiance:* {resultat.confiance:.0%}\n"
            f"*Piste:* #{resultat.id_piste}\n"
            f"*Heure:* {time.strftime('%H:%M:%S')}\n"
        )

        url_base = f"https://api.telegram.org/bot{self.config.telegram_bot_token}"
        max_retries = 3
        delay = 2.0

        for tentative in range(max_retries):
            try:
                if chemin_snapshot and Path(chemin_snapshot).exists():
                    url = f"{url_base}/sendPhoto"
                    with open(chemin_snapshot, "rb") as photo:
                        response = requests.post(
                            url,
                            data={
                                "chat_id": cible,
                                "caption": message,
                                "parse_mode": "Markdown",
                            },
                            files={"photo": photo},
                            timeout=15,
                        )
                else:
                    url = f"{url_base}/sendMessage"
                    response = requests.post(
                        url,
                        json={
                            "chat_id": cible,
                            "text": message,
                            "parse_mode": "Markdown",
                        },
                        timeout=10,
                    )

                if response.status_code == 200:
                    logger.info(f"Telegram envoye a {cible}")
                    return
                elif response.status_code == 429:
                    # Rate limited — attendre le retry_after
                    retry_after = response.json().get("parameters", {}).get("retry_after", delay)
                    logger.warning(f"Telegram rate limit, retry dans {retry_after}s")
                    time.sleep(retry_after)
                    continue
                elif response.status_code in (401, 400):
                    logger.error(f"Telegram erreur auth/config ({cible}): {response.status_code} — verifiez token/chat_id")
                    return  # Pas de retry pour erreurs de config
                else:
                    logger.warning(f"Telegram erreur ({cible}): {response.status_code}")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout Telegram ({cible}), tentative {tentative + 1}/{max_retries}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connexion Telegram echouee ({cible}), tentative {tentative + 1}/{max_retries}")
            except Exception as e:
                msg_err = _masquer_token(str(e), self.config.telegram_bot_token)
                logger.error(f"Erreur envoi Telegram ({cible}): {msg_err}")
                return  # Erreur inattendue, pas de retry

            # Backoff exponentiel entre retries
            if tentative < max_retries - 1:
                time.sleep(delay)
                delay *= 2

        logger.error(f"Telegram: abandon apres {max_retries} tentatives ({cible})")

    def _envoyer_telegram_video(
        self,
        resultat: ResultatAnalyse,
        chemin_video: str,
        severite: str,
        chat_id: str = "",
    ):
        """Envoie le clip vidéo par Telegram (document) a un chat_id specifique."""
        if not self.config.telegram_actif:
            return

        cible = chat_id or self.config.telegram_chat_id
        if not cible:
            return

        try:
            emoji = EMOJI_SEVERITE.get(severite, "\u2139\ufe0f")
            desc = DESCRIPTIONS_COMPORTEMENTS.get(
                resultat.type_comportement, resultat.type_comportement.value
            )
            caption = (
                f"{emoji} *VIDEO - {desc}*\n"
                f"Piste #{resultat.id_piste} | {time.strftime('%H:%M:%S')}"
            )

            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendDocument"

            chemin = Path(chemin_video)
            if chemin.exists():
                taille_mb = chemin.stat().st_size / (1024 * 1024)
                if taille_mb > 50:
                    logger.warning(f"Video trop volumineuse pour Telegram: {taille_mb:.1f} MB")
                    return

                with open(chemin, "rb") as video:
                    response = requests.post(
                        url,
                        data={
                            "chat_id": cible,
                            "caption": caption,
                            "parse_mode": "Markdown",
                        },
                        files={"document": (chemin.name, video, "video/mp4")},
                        timeout=60,
                    )

                if response.status_code == 200:
                    logger.info(f"Video Telegram envoyee a {cible}: {chemin.name}")
                else:
                    logger.warning(f"Erreur Telegram video ({cible}): {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout video Telegram ({cible})")
        except Exception as e:
            msg_err = _masquer_token(str(e), self.config.telegram_bot_token)
            logger.error(f"Erreur envoi video Telegram ({cible}): {msg_err}")

    def envoyer_message_telegram(
        self,
        text: str,
        chemin_photo: Optional[str] = None,
        chat_ids: Optional[List[str]] = None,
        parse_mode: str = "Markdown",
        timeout: int = 15,
    ) -> bool:
        """Envoi Telegram generique (texte +/- photo) reutilisable hors flux alerte.

        Synchrone, retourne True si AU MOINS 1 destinataire a recu le message
        avec succes (status 200). Utilise le meme retry/backoff/rate-limit que
        _envoyer_telegram. Si chat_ids est None, tombe sur config.telegram_chat_id.
        """
        if not self.config.telegram_actif:
            return False
        if not self.config.telegram_bot_token:
            logger.warning("[telegram] envoyer_message_telegram: token absent")
            return False

        cibles = list(chat_ids) if chat_ids else (
            [self.config.telegram_chat_id] if self.config.telegram_chat_id else []
        )
        if not cibles:
            logger.warning("[telegram] envoyer_message_telegram: aucun chat_id")
            return False

        url_base = f"https://api.telegram.org/bot{self.config.telegram_bot_token}"
        any_success = False

        for cible in cibles:
            max_retries = 3
            delay = 2.0
            for tentative in range(max_retries):
                try:
                    if chemin_photo and Path(chemin_photo).exists():
                        with open(chemin_photo, "rb") as photo:
                            response = requests.post(
                                f"{url_base}/sendPhoto",
                                data={"chat_id": cible, "caption": text, "parse_mode": parse_mode},
                                files={"photo": photo},
                                timeout=timeout,
                            )
                    else:
                        response = requests.post(
                            f"{url_base}/sendMessage",
                            json={"chat_id": cible, "text": text, "parse_mode": parse_mode},
                            timeout=timeout,
                        )

                    if response.status_code == 200:
                        any_success = True
                        logger.info(f"[telegram] message envoye a {cible}")
                        break
                    if response.status_code == 429:
                        retry_after = response.json().get("parameters", {}).get("retry_after", delay)
                        logger.warning(f"[telegram] rate limit, retry dans {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    if response.status_code in (401, 400):
                        logger.error(f"[telegram] auth/config ({cible}): {response.status_code}")
                        break
                    logger.warning(f"[telegram] erreur ({cible}): {response.status_code}")
                except requests.exceptions.Timeout:
                    logger.warning(
                        f"[telegram] timeout ({cible}), tentative {tentative + 1}/{max_retries}"
                    )
                except requests.exceptions.ConnectionError as e:
                    msg_err = _masquer_token(str(e), self.config.telegram_bot_token)
                    logger.warning(
                        f"[telegram] connexion echouee ({cible}) tentative {tentative + 1}/{max_retries}: {msg_err}"
                    )
                except Exception as e:
                    msg_err = _masquer_token(str(e), self.config.telegram_bot_token)
                    logger.error(f"[telegram] erreur inattendue ({cible}): {msg_err}")
                    break
                if tentative < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2

        return any_success

    def _determiner_zone(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
        taille_frame: Tuple[int, int],
    ) -> str:
        """
        Détermine la zone du magasin basée sur la position dans la frame.
        Division simple en grille 3x3.
        """
        if bbox is None:
            return "inconnue"

        h, w = taille_frame
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Grille 3x3
        col = "gauche" if cx < w / 3 else ("centre" if cx < 2 * w / 3 else "droite")
        lig = "haut" if cy < h / 3 else ("milieu" if cy < 2 * h / 3 else "bas")

        # Zones nommées
        noms_zones = {
            ("haut", "gauche"): "rayon_A",
            ("haut", "centre"): "rayon_B",
            ("haut", "droite"): "rayon_C",
            ("milieu", "gauche"): "allee_gauche",
            ("milieu", "centre"): "allee_centrale",
            ("milieu", "droite"): "allee_droite",
            ("bas", "gauche"): "entree",
            ("bas", "centre"): "caisse",
            ("bas", "droite"): "sortie",
        }

        return noms_zones.get((lig, col), "inconnue")

    @property
    def compteur_alertes(self) -> int:
        """Nombre total d'alertes déclenchées depuis le démarrage."""
        return self._compteur_alertes

    def nettoyer_cooldowns(self):
        """Supprime les entrées de cooldown expirées pour libérer la mémoire."""
        maintenant = time.time()
        seuil = self.config.alert_cooldown_seconds * 3  # Garder un peu de marge

        with self._lock:
            cles_expirees = [
                cle for cle, ts in self._dernieres_alertes.items()
                if maintenant - ts > seuil
            ]
            for cle in cles_expirees:
                del self._dernieres_alertes[cle]

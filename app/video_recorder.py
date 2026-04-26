"""
Enregistrement de clips vidéo lors des alertes.
Utilise un buffer circulaire pour capturer aussi les secondes précédant l'événement.
Thread-safe et économe en mémoire.
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EnregistreurVideo:
    """
    Enregistre des clips vidéo autour des événements d'alerte.
    Maintient un buffer circulaire de frames récentes pour la pré-capture.
    Sauvegarde les snapshots dans un répertoire dédié organisé par date.
    """

    def __init__(
        self,
        repertoire_sortie: Path,
        repertoire_snapshots: Optional[Path] = None,
        duree_clip: int = 30,
        fps: int = 15,
        pre_evenement_secondes: int = 5,
        retention_jours: int = 30,
        taille_frame: Tuple[int, int] = (640, 480),
        retention_videos_jours: Optional[int] = None,
        retention_snapshots_jours: Optional[int] = None,
        quota_stockage_max_gb: int = 50,
        quota_seuil_alerte_pct: int = 90,
    ):
        """
        Args:
            repertoire_sortie: Dossier de destination des clips vidéo
            repertoire_snapshots: Dossier dédié pour les snapshots (organisé par date)
            duree_clip: Durée totale du clip en secondes
            fps: Images par seconde
            pre_evenement_secondes: Secondes enregistrées avant l'événement
            retention_jours: Nombre de jours de conservation (fallback uniforme)
            taille_frame: Taille de sortie (largeur, hauteur)
            retention_videos_jours: Conservation des clips video (defaut: 7)
            retention_snapshots_jours: Conservation des snapshots (defaut: 14)
            quota_stockage_max_gb: Quota max espace disque (GB)
            quota_seuil_alerte_pct: Seuil d'alerte stockage (% du quota)
        """
        self.repertoire_sortie = repertoire_sortie
        self.repertoire_snapshots = repertoire_snapshots or (repertoire_sortie / "snapshots")
        self.duree_clip = duree_clip
        self.fps = fps
        self.pre_evenement = pre_evenement_secondes
        self.retention_jours = retention_jours
        self.retention_videos_jours = retention_videos_jours or 7
        self.retention_snapshots_jours = retention_snapshots_jours or 14
        self.quota_stockage_max_gb = quota_stockage_max_gb
        self.quota_seuil_alerte_pct = quota_seuil_alerte_pct
        self.taille_frame = taille_frame

        # Buffer circulaire pour la pré-capture
        taille_buffer = fps * pre_evenement_secondes
        self._buffer: deque = deque(maxlen=taille_buffer)
        self._lock_buffer = threading.Lock()

        # État des enregistrements en cours
        self._enregistrements_actifs: dict = {}
        self._lock_enreg = threading.Lock()

        # Callbacks à exécuter quand un clip est terminé
        self._callbacks_fin: dict = {}
        self._lock_callbacks = threading.Lock()

        # Créer les répertoires de sortie
        repertoire_sortie.mkdir(parents=True, exist_ok=True)
        self.repertoire_snapshots.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Enregistreur video initialise: clips={repertoire_sortie}, "
            f"snapshots={self.repertoire_snapshots}, "
            f"clips de {duree_clip}s a {fps} FPS"
        )

    def alimenter_buffer(self, frame: np.ndarray):
        """
        Ajoute une frame au buffer circulaire.
        Doit être appelé à chaque frame du flux principal.

        Args:
            frame: Image BGR (numpy array)
        """
        # Redimensionner si nécessaire pour économiser la mémoire
        h, w = frame.shape[:2]
        if (w, h) != self.taille_frame:
            frame_redim = cv2.resize(frame, self.taille_frame)
        else:
            frame_redim = frame

        with self._lock_buffer:
            self._buffer.append((time.time(), frame_redim.copy()))

    def demarrer_enregistrement(
        self,
        identifiant: str,
        type_alerte: str = "alerte",
        callback_fin: Optional[callable] = None,
    ) -> Optional[str]:
        """
        Démarre l'enregistrement d'un clip vidéo en arrière-plan.
        Inclut les frames pré-événement du buffer circulaire.

        Args:
            identifiant: Identifiant unique pour ce clip
            type_alerte: Type d'alerte (utilisé dans le nom du fichier)
            callback_fin: Fonction appelée quand le clip est terminé,
                         signature: callback(chemin_video: str)

        Returns:
            Chemin du fichier vidéo en cours de création, ou None si déjà en cours
        """
        with self._lock_enreg:
            if identifiant in self._enregistrements_actifs:
                logger.debug(f"Enregistrement deja en cours pour: {identifiant}")
                return None

        # Générer le nom de fichier
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        nom_fichier = f"{type_alerte}_{identifiant}_{horodatage}.mp4"
        chemin_sortie = self.repertoire_sortie / nom_fichier

        # Récupérer les frames du buffer (pré-événement)
        with self._lock_buffer:
            frames_pre = list(self._buffer)

        # Stocker le callback si fourni
        if callback_fin:
            with self._lock_callbacks:
                self._callbacks_fin[identifiant] = callback_fin

        # Lancer l'enregistrement dans un thread séparé
        thread = threading.Thread(
            target=self._enregistrer_clip,
            args=(identifiant, chemin_sortie, frames_pre),
            daemon=True,
            name=f"enregistreur_{identifiant}",
        )

        with self._lock_enreg:
            self._enregistrements_actifs[identifiant] = {
                "thread": thread,
                "chemin": str(chemin_sortie),
                "debut": time.time(),
                "actif": True,
            }

        thread.start()
        logger.info(f"Enregistrement demarre: {nom_fichier}")
        return str(chemin_sortie)

    def _estimer_fps_reel(self, frames: list) -> float:
        """Estime le FPS reel a partir des timestamps du buffer."""
        if len(frames) < 2:
            return self.fps
        timestamps = [ts for ts, _ in frames]
        duree = timestamps[-1] - timestamps[0]
        if duree <= 0:
            return self.fps
        fps_reel = (len(frames) - 1) / duree
        # Borner entre 0.5 et self.fps
        return max(0.5, min(self.fps, fps_reel))

    def _enregistrer_clip(
        self,
        identifiant: str,
        chemin_sortie: Path,
        frames_pre: list,
    ):
        """
        Thread d'enregistrement d'un clip video.
        Ecrit les frames pre-evenement puis continue l'enregistrement.
        Le FPS du fichier est calcule a partir du FPS reel de capture.
        """
        writer = None
        try:
            # Calculer le FPS reel a partir du buffer pre-evenement
            fps_reel = self._estimer_fps_reel(frames_pre)
            logger.info(f"FPS reel estime: {fps_reel:.1f} (nominal: {self.fps})")

            # Essayer plusieurs codecs par ordre de preference
            codecs_a_essayer = [
                ("avc1", ".mp4"),   # H.264 — le plus compatible
                ("mp4v", ".mp4"),   # MPEG-4
                ("XVID", ".avi"),   # Xvid fallback
                ("MJPG", ".avi"),   # Motion JPEG — toujours disponible
            ]
            writer = None
            for codec_str, ext in codecs_a_essayer:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                chemin_final = chemin_sortie.with_suffix(ext) if ext != chemin_sortie.suffix else chemin_sortie
                writer = cv2.VideoWriter(
                    str(chemin_final),
                    fourcc,
                    fps_reel,
                    self.taille_frame,
                )
                if writer.isOpened():
                    if codec_str != "avc1":
                        logger.info(f"Codec {codec_str} utilise (fallback)")
                    chemin_sortie = chemin_final
                    break
                writer.release()
                writer = None

            if writer is None or not writer.isOpened():
                logger.error(f"Aucun codec video disponible pour: {chemin_sortie}")
                return

            # Ecrire les frames pre-evenement avec overlay horodatage
            for ts, frame in frames_pre:
                frame_annote = self._ajouter_horodatage(frame, ts)
                writer.write(frame_annote)

            # Continuer l'enregistrement pour la duree restante
            duree_pre = len(frames_pre) / max(fps_reel, 0.5)
            duree_restante = self.duree_clip - duree_pre
            debut = time.time()
            dernier_ts_ecrit = 0.0

            while time.time() - debut < duree_restante:
                # Recuperer la derniere frame du buffer (copie defensive sous lock)
                ts = None
                frame = None
                with self._lock_buffer:
                    if self._buffer:
                        ts, frame_ref = self._buffer[-1]
                        # Copie explicite pour se decoupler du buffer circulaire
                        # (meme si alimenter_buffer copie deja, double protection)
                        frame = frame_ref.copy()

                # Ne PAS sleep sous lock (bloquerait le grab thread)
                if frame is None:
                    time.sleep(0.5)
                    continue

                # Ecrire uniquement les nouvelles frames (evite les doublons)
                if ts > dernier_ts_ecrit:
                    frame_annote = self._ajouter_horodatage(frame, ts)
                    writer.write(frame_annote)
                    dernier_ts_ecrit = ts

                # Attendre la prochaine frame (basé sur le FPS reel)
                time.sleep(1.0 / max(fps_reel, 0.5))

                # Verifier si l'enregistrement doit etre interrompu
                with self._lock_enreg:
                    info = self._enregistrements_actifs.get(identifiant, {})
                    if not info.get("actif", False):
                        break

            logger.info(f"Enregistrement termine: {chemin_sortie.name}")

        except Exception as e:
            logger.error(f"Erreur enregistrement video: {e}")
        finally:
            if writer is not None:
                writer.release()

            with self._lock_enreg:
                if identifiant in self._enregistrements_actifs:
                    del self._enregistrements_actifs[identifiant]

            # Appeler le callback de fin si enregistré
            with self._lock_callbacks:
                callback = self._callbacks_fin.pop(identifiant, None)
            if callback and chemin_sortie.exists():
                try:
                    callback(str(chemin_sortie))
                except Exception as e:
                    logger.error(f"Erreur callback fin enregistrement: {e}")

    def _ajouter_horodatage(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Ajoute un overlay d'horodatage sur la frame."""
        frame_copie = frame.copy()
        dt = datetime.fromtimestamp(timestamp)
        texte = dt.strftime("%Y-%m-%d %H:%M:%S")

        # Fond semi-transparent pour le texte
        taille_texte = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x, y = 10, 25
        cv2.rectangle(
            frame_copie,
            (x - 2, y - taille_texte[1] - 4),
            (x + taille_texte[0] + 2, y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame_copie,
            texte,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame_copie

    def arreter_enregistrement(self, identifiant: str):
        """Arrête un enregistrement en cours."""
        with self._lock_enreg:
            if identifiant in self._enregistrements_actifs:
                self._enregistrements_actifs[identifiant]["actif"] = False
                logger.info(f"Arret enregistrement: {identifiant}")

    def sauvegarder_snapshot(
        self,
        frame: np.ndarray,
        nom: str = "snapshot",
        qualite_jpeg: int = 90,
    ) -> str:
        """
        Sauvegarde une capture d'écran instantanée dans le répertoire
        de snapshots organisé par date (snapshots/YYYY-MM-DD/).

        Args:
            frame: Image BGR (numpy array)
            nom: Préfixe du nom de fichier
            qualite_jpeg: Qualité JPEG (0-100)

        Returns:
            Chemin absolu du fichier snapshot
        """
        maintenant = datetime.now()
        # Sous-répertoire par date: snapshots/2026-04-06/
        dossier_jour = self.repertoire_snapshots / maintenant.strftime("%Y-%m-%d")
        dossier_jour.mkdir(parents=True, exist_ok=True)

        horodatage = maintenant.strftime("%Y%m%d_%H%M%S_%f")
        nom_fichier = f"{nom}_{horodatage}.jpg"
        chemin = dossier_jour / nom_fichier

        cv2.imwrite(str(chemin), frame, [cv2.IMWRITE_JPEG_QUALITY, qualite_jpeg])
        logger.info(f"Snapshot fraude sauvegarde: {chemin}")
        return str(chemin)

    def calculer_usage_disque(self) -> dict:
        """Calcule l'espace disque utilise par les enregistrements.

        Returns:
            dict avec videos_mb, snapshots_mb, total_mb, quota_mb, usage_pct
        """
        videos_mb = 0.0
        snapshots_mb = 0.0

        if self.repertoire_sortie.exists():
            for f in self.repertoire_sortie.iterdir():
                if f.is_file() and f.suffix in (".mp4", ".avi"):
                    videos_mb += f.stat().st_size / (1024 * 1024)

        if self.repertoire_snapshots.exists():
            for dossier in self.repertoire_snapshots.iterdir():
                if dossier.is_dir():
                    for f in dossier.iterdir():
                        if f.is_file():
                            snapshots_mb += f.stat().st_size / (1024 * 1024)
                elif dossier.is_file():
                    snapshots_mb += dossier.stat().st_size / (1024 * 1024)

        total_mb = videos_mb + snapshots_mb
        quota_mb = self.quota_stockage_max_gb * 1024
        usage_pct = (total_mb / quota_mb * 100) if quota_mb > 0 else 0

        return {
            "videos_mb": round(videos_mb, 1),
            "snapshots_mb": round(snapshots_mb, 1),
            "total_mb": round(total_mb, 1),
            "quota_mb": round(quota_mb, 1),
            "usage_pct": round(usage_pct, 1),
        }

    def nettoyer_anciens_fichiers(self):
        """Nettoyage differencie: videos et snapshots ont des retentions separees.
        Si le quota disque est depasse, nettoyage d'urgence progressif.
        """
        nb_supprimes = 0

        # 1. Nettoyage par retention differenciee
        seuil_videos = datetime.now() - timedelta(days=self.retention_videos_jours)
        seuil_snaps = datetime.now() - timedelta(days=self.retention_snapshots_jours)

        # Clips video
        if self.repertoire_sortie.exists():
            for fichier in self.repertoire_sortie.iterdir():
                if fichier.is_file() and fichier.suffix in (".mp4", ".avi"):
                    date_modif = datetime.fromtimestamp(fichier.stat().st_mtime)
                    if date_modif < seuil_videos:
                        fichier.unlink()
                        nb_supprimes += 1

        # Snapshots
        if self.repertoire_snapshots.exists():
            for dossier_jour in self.repertoire_snapshots.iterdir():
                if dossier_jour.is_dir():
                    fichiers_restants = 0
                    for fichier in dossier_jour.iterdir():
                        if fichier.is_file():
                            date_modif = datetime.fromtimestamp(fichier.stat().st_mtime)
                            if date_modif < seuil_snaps:
                                fichier.unlink()
                                nb_supprimes += 1
                            else:
                                fichiers_restants += 1
                    if fichiers_restants == 0:
                        try:
                            dossier_jour.rmdir()
                        except OSError:
                            pass

        if nb_supprimes > 0:
            logger.info(f"Nettoyage retention: {nb_supprimes} fichiers supprimes "
                        f"(videos>{self.retention_videos_jours}j, snaps>{self.retention_snapshots_jours}j)")

        # 2. Nettoyage d'urgence si quota depasse
        usage = self.calculer_usage_disque()
        if usage["usage_pct"] >= self.quota_seuil_alerte_pct:
            logger.warning(f"Stockage critique: {usage['total_mb']:.0f} MB / "
                           f"{usage['quota_mb']:.0f} MB ({usage['usage_pct']:.0f}%)")
            nb_urgence = self._nettoyage_urgence(usage["usage_pct"])
            nb_supprimes += nb_urgence

        return nb_supprimes

    def _nettoyage_urgence(self, usage_pct: float) -> int:
        """Supprime les fichiers les plus anciens jusqu'a repasser sous le seuil.
        Priorite: videos d'abord (plus lourdes), puis snapshots.
        """
        nb_supprimes = 0
        retention_urgence = 3 if usage_pct >= 95 else 5

        seuil_urgence = datetime.now() - timedelta(days=retention_urgence)
        logger.warning(f"Nettoyage urgence: retention reduite a {retention_urgence} jours")

        # Videos en premier (plus lourdes)
        if self.repertoire_sortie.exists():
            fichiers = sorted(
                [f for f in self.repertoire_sortie.iterdir()
                 if f.is_file() and f.suffix in (".mp4", ".avi")],
                key=lambda f: f.stat().st_mtime,
            )
            for fichier in fichiers:
                if datetime.fromtimestamp(fichier.stat().st_mtime) < seuil_urgence:
                    fichier.unlink()
                    nb_supprimes += 1

        # Snapshots ensuite
        if self.repertoire_snapshots.exists():
            for dossier_jour in self.repertoire_snapshots.iterdir():
                if dossier_jour.is_dir():
                    fichiers_restants = 0
                    for fichier in dossier_jour.iterdir():
                        if fichier.is_file():
                            if datetime.fromtimestamp(fichier.stat().st_mtime) < seuil_urgence:
                                fichier.unlink()
                                nb_supprimes += 1
                            else:
                                fichiers_restants += 1
                    if fichiers_restants == 0:
                        try:
                            dossier_jour.rmdir()
                        except OSError:
                            pass

        if nb_supprimes > 0:
            logger.warning(f"Nettoyage urgence: {nb_supprimes} fichiers supprimes (retention={retention_urgence}j)")

        return nb_supprimes

    @property
    def nb_enregistrements_actifs(self) -> int:
        """Retourne le nombre d'enregistrements en cours."""
        with self._lock_enreg:
            return len(self._enregistrements_actifs)

    def arreter_tout(self):
        """Arrête tous les enregistrements en cours."""
        with self._lock_enreg:
            for identifiant in list(self._enregistrements_actifs.keys()):
                self._enregistrements_actifs[identifiant]["actif"] = False
        logger.info("Tous les enregistrements arretes")

"""
Configuration centralisée du système de détection de fraude.
Utilise Pydantic Settings pour la validation et le chargement depuis .env.
"""

import logging
from pathlib import Path
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class FraudeConfig(BaseSettings):
    """Configuration principale du système de détection de fraude."""

    # --- Sources vidéo ---
    video_sources: str = "rtsp://camera1:554/stream1"
    webcam_test: bool = False

    # --- Seuils de détection ---
    yolo_confidence: float = 0.45
    pose_confidence: float = 0.5
    behavior_threshold: float = 0.6

    # --- Telegram ---
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # --- Alertes ---
    alert_sound: bool = True
    alert_cooldown_seconds: int = 30
    video_clip_duration: int = 30

    # --- Chemins ---
    video_save_path: str = "./recordings"
    snapshot_save_path: str = "./snapshots"
    model_path: str = "./models"
    database_path: str = "./data/fraude.db"

    # --- Multi-camera ---
    max_cameras: int = 8
    inference_concurrency: int = 2
    supervisor_interval_seconds: int = 10

    # --- Dashboard ---
    dashboard_port: int = 8502
    dashboard_refresh_seconds: int = 10

    # --- Parametres metier ---
    valeur_article_moyen_dh: float = 150.0
    heure_ouverture: str = "09:00"
    heure_fermeture: str = "22:00"
    retention_jours: int = 30
    retention_videos_jours: int = 7
    retention_snapshots_jours: int = 14
    retention_alertes_jours: int = 90
    retention_stats_jours: int = 365
    quota_stockage_max_gb: int = 50
    quota_seuil_alerte_pct: int = 90

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @property
    def sources_liste(self) -> List[str]:
        """Retourne la liste des sources vidéo séparées par des virgules."""
        return [s.strip() for s in self.video_sources.split(",") if s.strip()]

    @property
    def telegram_actif(self) -> bool:
        """Vérifie si Telegram est configuré (token suffit, chat_id peut venir par camera)."""
        return bool(self.telegram_bot_token)

    @property
    def chemin_modeles(self) -> Path:
        """Retourne le chemin absolu vers les modèles."""
        return Path(self.model_path).resolve()

    @property
    def chemin_enregistrements(self) -> Path:
        """Retourne le chemin absolu vers les enregistrements vidéo."""
        return Path(self.video_save_path).resolve()

    @property
    def chemin_snapshots(self) -> Path:
        """Retourne le chemin absolu vers les snapshots de fraude."""
        return Path(self.snapshot_save_path).resolve()

    @property
    def chemin_base_donnees(self) -> Path:
        """Retourne le chemin absolu vers la base de données."""
        return Path(self.database_path).resolve()

    @property
    def chemin_modele_yolo(self) -> Path:
        """Chemin vers le modèle YOLO de détection."""
        return self.chemin_modeles / "yolov8n.onnx"

    @property
    def chemin_modele_pose(self) -> Path:
        """Chemin vers le modèle de pose estimation."""
        return self.chemin_modeles / "yolov8n-pose.onnx"

    @property
    def chemin_modele_oiv7(self) -> Path:
        """Chemin vers le modèle Open Images V7 (~600 classes, apprentissage)."""
        return self.chemin_modeles / "yolov8n-oiv7.onnx"

    def assurer_repertoires(self):
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        self.chemin_enregistrements.mkdir(parents=True, exist_ok=True)
        self.chemin_snapshots.mkdir(parents=True, exist_ok=True)
        self.chemin_modeles.mkdir(parents=True, exist_ok=True)
        self.chemin_base_donnees.parent.mkdir(parents=True, exist_ok=True)

    def charger_depuis_db(self, db) -> "FraudeConfig":
        """Charge les parametres depuis la base de donnees avec validation/clamp.

        Format de la spec: (type_attendu, min, max). None = pas de borne.
        Les valeurs hors limites sont clampees et loguees en warning.
        Les valeurs mal typees sont ignorees (garde la valeur par defaut).
        """
        # cle_db -> (attr, type_python, min, max)
        specs = {
            # Seuils [0.0, 1.0]
            "yolo_confidence":             ("yolo_confidence",             float, 0.0, 1.0),
            "pose_confidence":             ("pose_confidence",             float, 0.0, 1.0),
            "behavior_threshold":          ("behavior_threshold",          float, 0.0, 1.0),
            # Booleens
            "alert_sound":                 ("alert_sound",                 bool,  None, None),
            # Entiers positifs avec bornes raisonnables
            "alert_cooldown_seconds":      ("alert_cooldown_seconds",      int,   1,    3600),
            "video_clip_duration":         ("video_clip_duration",         int,   1,    600),
            "retention_jours":             ("retention_jours",             int,   1,    3650),
            "max_cameras":                 ("max_cameras",                 int,   1,    32),
            "inference_concurrency":       ("inference_concurrency",       int,   1,    16),
            "supervisor_interval_seconds": ("supervisor_interval_seconds", int,   1,    3600),
            "dashboard_port":              ("dashboard_port",              int,   1,    65535),
            "dashboard_refresh_seconds":   ("dashboard_refresh_seconds",   int,   1,    3600),
            "retention_videos_jours":      ("retention_videos_jours",      int,   1,    3650),
            "retention_snapshots_jours":   ("retention_snapshots_jours",   int,   1,    3650),
            "retention_alertes_jours":     ("retention_alertes_jours",     int,   1,    3650),
            "retention_stats_jours":       ("retention_stats_jours",       int,   1,    3650),
            "quota_stockage_max_gb":       ("quota_stockage_max_gb",       int,   1,    100000),
            "quota_seuil_alerte_pct":      ("quota_seuil_alerte_pct",      int,   0,    100),
            # Flottants positifs
            "valeur_article_moyen_dh":     ("valeur_article_moyen_dh",     float, 0.0,  1000000.0),
            # Chaines (token, chat_id, heures, etc.) — pas de validation numerique
            "telegram_bot_token":          ("telegram_bot_token",          str,   None, None),
            "telegram_chat_id":            ("telegram_chat_id",            str,   None, None),
            "heure_ouverture":             ("heure_ouverture",             str,   None, None),
            "heure_fermeture":             ("heure_fermeture",             str,   None, None),
        }

        for cle_db, (attr, type_attendu, mini, maxi) in specs.items():
            val = db.obtenir_parametre(cle_db)
            if val is None or val == "":
                continue
            # Coercition de type
            try:
                if type_attendu is bool:
                    if isinstance(val, str):
                        val_typee = val.strip().lower() in ("1", "true", "oui", "yes", "on")
                    else:
                        val_typee = bool(val)
                elif type_attendu is int:
                    val_typee = int(float(val))  # tolere "5.0"
                elif type_attendu is float:
                    val_typee = float(val)
                else:
                    val_typee = str(val)
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"Parametre DB '{cle_db}'={val!r} invalide ({type_attendu.__name__}): "
                    f"{e}. Valeur par defaut conservee."
                )
                continue

            # Clamp min/max pour numeriques
            if type_attendu in (int, float):
                original = val_typee
                if mini is not None and val_typee < mini:
                    val_typee = type_attendu(mini)
                if maxi is not None and val_typee > maxi:
                    val_typee = type_attendu(maxi)
                if val_typee != original:
                    logger.warning(
                        f"Parametre DB '{cle_db}'={original} hors limites "
                        f"[{mini}, {maxi}], clampe a {val_typee}."
                    )

            try:
                object.__setattr__(self, attr, val_typee)
            except Exception as e:
                logger.warning(f"Impossible d'appliquer le parametre DB '{cle_db}': {e}")
        return self

    @field_validator("yolo_confidence", "pose_confidence", "behavior_threshold")
    @classmethod
    def valider_seuil(cls, v: float) -> float:
        """Valide que les seuils sont entre 0 et 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Le seuil doit etre entre 0 et 1, recu: {v}")
        return v

    @field_validator("alert_cooldown_seconds")
    @classmethod
    def valider_cooldown(cls, v: int) -> int:
        """Valide que le cooldown est positif."""
        if v < 0:
            raise ValueError(f"Le cooldown doit etre positif, recu: {v}")
        return v


# Instance globale de configuration (singleton)
_config_instance: Optional[FraudeConfig] = None


def obtenir_config() -> FraudeConfig:
    """Retourne l'instance unique de configuration."""
    global _config_instance
    if _config_instance is None:
        _config_instance = FraudeConfig()
        _config_instance.assurer_repertoires()
    return _config_instance

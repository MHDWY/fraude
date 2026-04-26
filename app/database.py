"""
Base de donnees SQLite pour le stockage des alertes, statistiques,
parametres configurables et cameras.
Leger et adapte a un deploiement desktop sans serveur externe.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseDonneesFraude:
    """
    Gestionnaire de base de donnees SQLite pour les alertes de fraude.
    Thread-safe avec un pool de connexions simple.
    """

    def __init__(self, chemin_db: Path):
        self.chemin_db = chemin_db
        self._lock = threading.Lock()
        chemin_db.parent.mkdir(parents=True, exist_ok=True)
        self._initialiser_schema()
        self.initialiser_parametres_defaut()
        logger.info(f"Base de donnees initialisee: {chemin_db}")

    @contextmanager
    def _connexion(self):
        """Context manager pour une connexion thread-safe."""
        conn = sqlite3.connect(str(self.chemin_db), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialiser_schema(self):
        """Cree les tables si elles n'existent pas."""
        with self._connexion() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alertes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    horodatage TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    type_comportement TEXT NOT NULL,
                    confiance REAL NOT NULL,
                    id_piste INTEGER,
                    bbox_x1 INTEGER, bbox_y1 INTEGER,
                    bbox_x2 INTEGER, bbox_y2 INTEGER,
                    chemin_video TEXT, chemin_snapshot TEXT,
                    zone TEXT, source_camera TEXT,
                    notifie_telegram BOOLEAN DEFAULT 0,
                    commentaire TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_alertes_horodatage ON alertes(horodatage);
                CREATE INDEX IF NOT EXISTS idx_alertes_type ON alertes(type_comportement);
                CREATE INDEX IF NOT EXISTS idx_alertes_date ON alertes(date(horodatage));

                CREATE TABLE IF NOT EXISTS stats_journalieres (
                    date TEXT PRIMARY KEY,
                    total_alertes INTEGER DEFAULT 0,
                    montant_estime_evite_dh REAL DEFAULT 0.0,
                    incidents_uniques INTEGER DEFAULT 0,
                    comportement_le_plus_frequent TEXT,
                    heure_pic TEXT,
                    mise_a_jour TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Table des parametres configurables
                CREATE TABLE IF NOT EXISTS parametres (
                    cle TEXT PRIMARY KEY,
                    valeur TEXT NOT NULL,
                    categorie TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    type_valeur TEXT DEFAULT 'str',
                    mis_a_jour TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_parametres_categorie ON parametres(categorie);

                -- Table des cameras
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nom TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    zone TEXT DEFAULT 'inconnue',
                    niveau TEXT DEFAULT 'Niveau 0',
                    position_description TEXT DEFAULT '',
                    active BOOLEAN DEFAULT 1,
                    mode_detection TEXT DEFAULT 'tout' CHECK(mode_detection IN ('tout', 'vol', 'caisse')),
                    ajoutee_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Table des utilisateurs a alerter (Telegram ou Email)
                CREATE TABLE IF NOT EXISTS utilisateurs_alertes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nom TEXT NOT NULL,
                    type_alerte TEXT NOT NULL CHECK(type_alerte IN ('telegram', 'email')),
                    identifiant TEXT NOT NULL,
                    actif BOOLEAN DEFAULT 1,
                    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ua_type_ident
                    ON utilisateurs_alertes(type_alerte, identifiant);

                -- Association cameras <-> utilisateurs (many-to-many)
                CREATE TABLE IF NOT EXISTS camera_utilisateurs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    utilisateur_id INTEGER NOT NULL,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE,
                    FOREIGN KEY (utilisateur_id) REFERENCES utilisateurs_alertes(id) ON DELETE CASCADE,
                    UNIQUE(camera_id, utilisateur_id)
                );

                -- Sessions de test (image/video/webcam)
                CREATE TABLE IF NOT EXISTS sessions_test (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    horodatage TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    type_source TEXT NOT NULL CHECK(type_source IN ('image', 'video', 'webcam')),
                    nom_fichier TEXT DEFAULT '',
                    nb_personnes INTEGER DEFAULT 0,
                    nb_objets INTEGER DEFAULT 0,
                    nb_alertes INTEGER DEFAULT 0,
                    temps_inference_ms REAL DEFAULT 0.0,
                    confiance_utilisee REAL DEFAULT 0.45,
                    pose_activee BOOLEAN DEFAULT 1,
                    chemin_snapshot TEXT DEFAULT '',
                    envoyee_telegram BOOLEAN DEFAULT 0,
                    commentaire TEXT DEFAULT '',
                    -- Pour les videos: infos supplementaires
                    nb_frames_analysees INTEGER DEFAULT 0,
                    duree_video_sec REAL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_test_date ON sessions_test(date(horodatage));
                CREATE INDEX IF NOT EXISTS idx_sessions_test_type ON sessions_test(type_source);

                -- Alertes detectees dans les sessions de test
                CREATE TABLE IF NOT EXISTS alertes_test (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    personne_idx INTEGER DEFAULT 0,
                    type_comportement TEXT NOT NULL,
                    score REAL DEFAULT 0.0,
                    severite TEXT DEFAULT 'BASSE',
                    description TEXT DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions_test(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_alertes_test_session ON alertes_test(session_id);


                -- Objets de reference par camera (calibration visuelle)
                CREATE TABLE IF NOT EXISTS objets_reference (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    classe TEXT NOT NULL,
                    role TEXT DEFAULT '',
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    confiance REAL DEFAULT 0.0,
                    comportement TEXT DEFAULT '',
                    actif BOOLEAN DEFAULT 1,
                    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
                );

                -- Zones d'exclusion par camera (filtrage spatial)
                CREATE TABLE IF NOT EXISTS zones_exclusion (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    label TEXT DEFAULT '',
                    pct_x1 REAL NOT NULL,
                    pct_y1 REAL NOT NULL,
                    pct_x2 REAL NOT NULL,
                    pct_y2 REAL NOT NULL,
                    source TEXT DEFAULT 'manuel' CHECK(source IN ('manuel', 'auto')),
                    actif BOOLEAN DEFAULT 1,
                    cree_le TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
                );

                -- Sessions d'apprentissage automatique
                CREATE TABLE IF NOT EXISTS sessions_apprentissage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    debut TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duree_minutes REAL DEFAULT 5.0,
                    statut TEXT DEFAULT 'en_cours' CHECK(statut IN ('en_cours', 'terminee', 'annulee')),
                    nb_zones_proposees INTEGER DEFAULT 0,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
                );

                -- Zones proposees par l'apprentissage (en attente de validation)
                CREATE TABLE IF NOT EXISTS zones_proposees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    camera_id INTEGER NOT NULL,
                    label TEXT DEFAULT '',
                    pct_x1 REAL NOT NULL,
                    pct_y1 REAL NOT NULL,
                    pct_x2 REAL NOT NULL,
                    pct_y2 REAL NOT NULL,
                    duree_observation_sec REAL DEFAULT 0.0,
                    classe_detectee TEXT DEFAULT '',
                    confiance_moyenne REAL DEFAULT 0.0,
                    statut TEXT DEFAULT 'proposee' CHECK(statut IN ('proposee', 'acceptee', 'rejetee')),
                    FOREIGN KEY (session_id) REFERENCES sessions_apprentissage(id) ON DELETE CASCADE,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
                );
            """)

            # Migration: ajouter mode_detection si absent (bases existantes)
            try:
                conn.execute("SELECT mode_detection FROM cameras LIMIT 1")
            except Exception:
                conn.execute("ALTER TABLE cameras ADD COLUMN mode_detection TEXT DEFAULT 'tout'")

            # Migration: ajouter niveau si absent (bases existantes)
            try:
                conn.execute("SELECT niveau FROM cameras LIMIT 1")
            except Exception:
                conn.execute("ALTER TABLE cameras ADD COLUMN niveau TEXT DEFAULT 'Niveau 0'")

    # =========================================================================
    # Parametres configurables
    # =========================================================================

    def initialiser_parametres_defaut(self):
        """Insere les parametres par defaut manquants (idempotent)."""
        with self._connexion() as conn:

            defauts = [
                # Detection
                ("yolo_confidence", "0.45", "detection", "Seuil de confiance YOLO", "float"),
                ("pose_confidence", "0.5", "detection", "Seuil de confiance estimation de pose", "float"),
                ("behavior_threshold", "0.6", "detection", "Seuil de declenchement des alertes comportementales", "float"),
                ("taille_entree_yolo", "416", "detection", "Taille d'entree du modele YOLO (320=rapide, 416=equilibre, 640=precis)", "int"),
                ("inference_frame_skip", "0", "detection", "Nb de frames sautees entre 2 inferences YOLO (0=toutes, 2=1 sur 3). Buffer video non impacte.", "int"),
                # Alertes
                ("alert_sound", "true", "alertes", "Activer le son d'alerte", "bool"),
                ("alert_cooldown_seconds", "30", "alertes", "Delai entre deux alertes du meme type (s)", "int"),
                ("video_clip_duration", "30", "alertes", "Duree des clips video d'alerte (s)", "int"),
                # Telegram
                ("telegram_bot_token", "", "telegram", "Token du bot Telegram", "str"),
                ("telegram_chat_id", "", "telegram", "ID du chat Telegram", "str"),
                # Metier
                ("valeur_article_moyen_dh", "250.0", "metier", "Valeur moyenne d'un article pret-a-porter (DH)", "float"),
                ("heure_ouverture", "09:00", "metier", "Heure d'ouverture du magasin", "str"),
                ("heure_fermeture", "22:00", "metier", "Heure de fermeture du magasin", "str"),
                ("retention_jours", "30", "metier", "Jours de conservation des donnees", "int"),
                # Systeme
                ("dashboard_port", "8502", "systeme", "Port du dashboard Streamlit", "int"),
                ("dashboard_refresh_seconds", "10", "systeme", "Intervalle de rafraichissement du dashboard (s)", "int"),
                ("admin_password", "asx", "systeme", "Mot de passe d'acces a l'onglet Administration du dashboard", "str"),
                ("video_save_path", "./recordings", "systeme", "Repertoire des enregistrements video", "str"),
                ("model_path", "./models", "systeme", "Repertoire des modeles ONNX", "str"),
                ("database_path", "./data/fraude.db", "systeme", "Chemin de la base de donnees", "str"),
                # Camera Live
                ("live_camera_refresh", "1.0", "camera_live", "Rafraichissement camera live (s)", "float"),
                ("live_camera_jpeg_quality", "80", "camera_live", "Qualite JPEG pour l'affichage live", "int"),
                # Fraude caisse
                ("caisse_timeout_ticket", "12.0", "caisse", "Delai max pour impression ticket apres scan/paiement (s)", "float"),
                ("caisse_zone_y_min_pct", "0.70", "caisse", "Zone caisse: limite haute (% frame)", "float"),
                ("caisse_zone_x_min_pct", "0.25", "caisse", "Zone caisse: limite gauche (% frame)", "float"),
                ("caisse_zone_x_max_pct", "0.75", "caisse", "Zone caisse: limite droite (% frame)", "float"),
                ("caisse_seuil_proximite_mains", "0.08", "caisse", "Seuil de proximite des mains pour remise ticket (% diag)", "float"),
                ("caisse_nb_cycles_scan_min", "2", "caisse", "Nombre min de cycles extension-retraction douchette pour detecter un scan", "int"),
                ("imprimante_seuil_blanc", "200", "caisse", "Luminosite min pour detecter le papier ticket (0-255)", "int"),
                ("imprimante_seuil_changement", "0.15", "caisse", "% min de pixels changes pour detecter le papier", "float"),
                ("caisse_detecter_transaction_fantome", "true", "caisse", "Activer la detection de transaction fantome (ticket imprime sans client)", "bool"),
                ("mannequin_seuil_immobilite_sec", "30", "detection", "Duree d'immobilite (sec) avant de considerer une piste comme mannequin", "float"),
                # Vol - Cacher article sous vetements
                ("vol_distance_main_corps", "0.25", "vol", "Distance main-corps normalisee (ratio hauteur bbox) pour detecter dissimulation", "float"),
                ("vol_zone_dissimulation_haut", "0.3", "vol", "Zone dissimulation: limite haute (ratio bbox, 0=epaules)", "float"),
                ("vol_zone_dissimulation_bas", "0.85", "vol", "Zone dissimulation: limite basse (ratio bbox, 1=pieds)", "float"),
                ("vol_increment_main_corps", "0.12", "vol", "Increment score quand main dans zone dissimulation", "float"),
                ("vol_increment_mouvement_rentrant", "0.15", "vol", "Increment score quand main se rapproche du corps (mouvement rentrant)", "float"),
                ("vol_increment_objet_proche", "0.10", "vol", "Increment bonus quand objet detecte pres de la main", "float"),
                ("vol_ratio_rapprochement", "0.6", "vol", "Ratio dist_fin/dist_debut pour detecter rapprochement main", "float"),
                ("vol_historique_min_frames", "5", "vol", "Nombre min d'observations pour analyse de mouvement", "int"),
                # Vol - Dissimuler dans sac
                ("vol_sac_increment_base", "0.10", "vol", "Increment score quand main pres du sac", "float"),
                ("vol_sac_increment_alternance", "0.15", "vol", "Increment bonus quand alternance main etagere-sac detectee", "float"),
                ("vol_sac_alternances_min", "2", "vol", "Nombre min d'alternances pour bonus", "int"),
                ("vol_sac_distance_ratio", "0.8", "vol", "Distance max main-sac (ratio taille sac) pour detection", "float"),
                # Vol - General
                ("vol_decay_rate", "0.95", "vol", "Taux de decroissance du score par frame de reference", "float"),
                ("vol_fps_ref", "2.0", "vol", "FPS de reference pour normalisation du decay (2=lent/CCTV, 15=rapide)", "float"),
                # Zones d'exclusion & apprentissage automatique
                ("apprentissage_duree_minutes", "5.0", "exclusion", "Duree par defaut de l'apprentissage automatique (min)", "float"),
                ("apprentissage_seuil_immobilite_sec", "120.0", "exclusion", "Duree min d'immobilite pour proposer une zone d'exclusion (sec)", "float"),
                ("apprentissage_seuil_deplacement_px", "30", "exclusion", "Distance max de deplacement pour considerer un objet immobile (px)", "int"),
                ("apprentissage_confiance_min", "0.3", "exclusion", "Confiance YOLO min pour considerer un objet dans l'apprentissage", "float"),
                # Retention differenciee
                ("retention_videos_jours", "7", "retention", "Conservation des clips video (jours)", "int"),
                ("retention_snapshots_jours", "14", "retention", "Conservation des snapshots (jours)", "int"),
                ("retention_alertes_jours", "90", "retention", "Conservation des alertes en BDD (jours)", "int"),
                ("retention_stats_jours", "365", "retention", "Conservation des stats journalieres (jours)", "int"),
                ("quota_stockage_max_gb", "50", "retention", "Quota max espace disque pour les enregistrements (GB)", "int"),
                ("quota_seuil_alerte_pct", "90", "retention", "Seuil d'alerte stockage (% du quota)", "int"),
            ]
            conn.executemany(
                """INSERT OR IGNORE INTO parametres (cle, valeur, categorie, description, type_valeur)
                VALUES (?, ?, ?, ?, ?)""",
                defauts,
            )
            logger.info("Parametres par defaut inseres")

    def obtenir_parametre(self, cle: str, defaut: Any = None) -> Any:
        """Retourne la valeur typee d'un parametre."""
        with self._connexion() as conn:
            row = conn.execute(
                "SELECT valeur, type_valeur FROM parametres WHERE cle = ?", (cle,)
            ).fetchone()
            if row is None:
                return defaut
            return self._convertir_valeur(row["valeur"], row["type_valeur"])

    def definir_parametre(self, cle: str, valeur: str, categorie: str = "",
                          description: str = "", type_valeur: str = "str"):
        """Cree ou met a jour un parametre."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute(
                    """INSERT INTO parametres (cle, valeur, categorie, description, type_valeur, mis_a_jour)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(cle) DO UPDATE SET
                        valeur=excluded.valeur,
                        mis_a_jour=CURRENT_TIMESTAMP""",
                    (cle, str(valeur), categorie, description, type_valeur),
                )

    def obtenir_parametres_par_categorie(self, categorie: str) -> List[Dict[str, Any]]:
        """Retourne tous les parametres d'une categorie."""
        with self._connexion() as conn:
            rows = conn.execute(
                "SELECT * FROM parametres WHERE categorie = ? ORDER BY cle",
                (categorie,),
            ).fetchall()
            return [dict(r) for r in rows]

    def obtenir_tous_parametres(self) -> List[Dict[str, Any]]:
        """Retourne tous les parametres."""
        with self._connexion() as conn:
            rows = conn.execute(
                "SELECT * FROM parametres ORDER BY categorie, cle"
            ).fetchall()
            return [dict(r) for r in rows]

    def reinitialiser_parametres(self):
        """Supprime tous les parametres et reinjecte les valeurs par defaut."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("DELETE FROM parametres")
        self.initialiser_parametres_defaut()

    @staticmethod
    def _convertir_valeur(valeur: str, type_valeur: str) -> Any:
        if type_valeur == "int":
            return int(valeur)
        elif type_valeur == "float":
            return float(valeur)
        elif type_valeur == "bool":
            return valeur.lower() in ("true", "1", "oui", "yes")
        return valeur

    # =========================================================================
    # Cameras
    # =========================================================================

    def ajouter_camera(self, nom: str, source: str, zone: str = "inconnue",
                       niveau: str = "Niveau 0", position_description: str = "",
                       mode_detection: str = "tout") -> int:
        """Ajoute une camera. Retourne l'ID."""
        with self._lock:
            with self._connexion() as conn:
                cur = conn.execute(
                    """INSERT INTO cameras (nom, source, zone, niveau, position_description, mode_detection)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (nom, source, zone, niveau, position_description, mode_detection),
                )
                return cur.lastrowid

    def modifier_camera(self, camera_id: int, **champs):
        """Met a jour les champs d'une camera."""
        colonnes_autorisees = {"nom", "source", "zone", "niveau", "position_description", "active", "mode_detection"}
        updates = {k: v for k, v in champs.items() if k in colonnes_autorisees}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [camera_id]
        with self._lock:
            with self._connexion() as conn:
                conn.execute(
                    f"UPDATE cameras SET {set_clause} WHERE id = ?", values
                )

    def supprimer_camera(self, camera_id: int):
        """Supprime une camera."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))

    def obtenir_cameras(self, actives_seulement: bool = False) -> List[Dict[str, Any]]:
        """Retourne la liste des cameras."""
        query = "SELECT * FROM cameras"
        if actives_seulement:
            query += " WHERE active = 1"
        query += " ORDER BY nom"
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(query).fetchall()]

    def obtenir_camera(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """Retourne une camera par son ID."""
        with self._connexion() as conn:
            row = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Objets de reference (calibration visuelle)
    # =========================================================================

    def ajouter_objet_reference(self, camera_id: int, classe: str, role: str,
                                 bbox: tuple, confiance: float = 0.0,
                                 comportement: str = "") -> int:
        """Ajoute un objet de reference pour une camera. Retourne l'ID."""
        x1, y1, x2, y2 = bbox
        with self._lock:
            with self._connexion() as conn:
                cur = conn.execute(
                    """INSERT INTO objets_reference
                    (camera_id, classe, role, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confiance, comportement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (camera_id, classe, role, x1, y1, x2, y2, confiance, comportement),
                )
                return cur.lastrowid

    def obtenir_objets_reference(self, camera_id: int, actifs_seulement: bool = True) -> List[Dict[str, Any]]:
        """Retourne les objets de reference d'une camera."""
        query = "SELECT * FROM objets_reference WHERE camera_id = ?"
        if actifs_seulement:
            query += " AND actif = 1"
        query += " ORDER BY role, classe"
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(query, (camera_id,)).fetchall()]

    def modifier_objet_reference(self, objet_id: int, **champs):
        """Met a jour un objet de reference."""
        colonnes = {"role", "comportement", "actif", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
        updates = {k: v for k, v in champs.items() if k in colonnes}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [objet_id]
        with self._lock:
            with self._connexion() as conn:
                conn.execute(f"UPDATE objets_reference SET {set_clause} WHERE id = ?", values)

    def supprimer_objet_reference(self, objet_id: int):
        """Supprime un objet de reference."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("DELETE FROM objets_reference WHERE id = ?", (objet_id,))

    def supprimer_objets_reference_camera(self, camera_id: int):
        """Supprime tous les objets de reference d'une camera."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("DELETE FROM objets_reference WHERE camera_id = ?", (camera_id,))

    # =========================================================================
    # Zones d'exclusion
    # =========================================================================

    def ajouter_zone_exclusion(self, camera_id: int, label: str,
                                pct_bbox: tuple, source: str = "manuel") -> int:
        """Ajoute une zone d'exclusion. pct_bbox = (x1, y1, x2, y2) en pourcentage 0.0-1.0."""
        x1, y1, x2, y2 = pct_bbox
        with self._lock:
            with self._connexion() as conn:
                cur = conn.execute(
                    """INSERT INTO zones_exclusion
                    (camera_id, label, pct_x1, pct_y1, pct_x2, pct_y2, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (camera_id, label, x1, y1, x2, y2, source),
                )
                return cur.lastrowid

    def obtenir_zones_exclusion(self, camera_id: int, actives_seulement: bool = True) -> List[Dict[str, Any]]:
        """Retourne les zones d'exclusion d'une camera."""
        query = "SELECT * FROM zones_exclusion WHERE camera_id = ?"
        if actives_seulement:
            query += " AND actif = 1"
        query += " ORDER BY label"
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(query, (camera_id,)).fetchall()]

    def modifier_zone_exclusion(self, zone_id: int, **champs):
        """Met a jour une zone d'exclusion."""
        colonnes = {"label", "actif", "pct_x1", "pct_y1", "pct_x2", "pct_y2"}
        updates = {k: v for k, v in champs.items() if k in colonnes}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [zone_id]
        with self._lock:
            with self._connexion() as conn:
                conn.execute(f"UPDATE zones_exclusion SET {set_clause} WHERE id = ?", values)

    def supprimer_zone_exclusion(self, zone_id: int):
        """Supprime une zone d'exclusion."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("DELETE FROM zones_exclusion WHERE id = ?", (zone_id,))

    def supprimer_zones_exclusion_camera(self, camera_id: int):
        """Supprime toutes les zones d'exclusion d'une camera."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("DELETE FROM zones_exclusion WHERE camera_id = ?", (camera_id,))

    # =========================================================================
    # Sessions d'apprentissage automatique
    # =========================================================================

    def creer_session_apprentissage(self, camera_id: int, duree_minutes: float = 5.0) -> int:
        """Cree une session d'apprentissage. Retourne l'ID."""
        with self._lock:
            with self._connexion() as conn:
                cur = conn.execute(
                    """INSERT INTO sessions_apprentissage (camera_id, duree_minutes)
                    VALUES (?, ?)""",
                    (camera_id, duree_minutes),
                )
                return cur.lastrowid

    def obtenir_session_apprentissage_active(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """Retourne la session en cours pour une camera, ou None."""
        with self._connexion() as conn:
            row = conn.execute(
                """SELECT * FROM sessions_apprentissage
                WHERE camera_id = ? AND statut = 'en_cours'
                ORDER BY debut DESC LIMIT 1""",
                (camera_id,),
            ).fetchone()
            return dict(row) if row else None

    def terminer_session_apprentissage(self, session_id: int, nb_zones: int):
        """Marque une session comme terminee."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute(
                    """UPDATE sessions_apprentissage
                    SET statut = 'terminee', nb_zones_proposees = ?
                    WHERE id = ?""",
                    (nb_zones, session_id),
                )

    def annuler_session_apprentissage(self, session_id: int):
        """Marque une session comme annulee."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute(
                    "UPDATE sessions_apprentissage SET statut = 'annulee' WHERE id = ?",
                    (session_id,),
                )

    def ajouter_zone_proposee(self, session_id: int, camera_id: int,
                               pct_bbox: tuple, duree_sec: float,
                               classe: str = "", confiance: float = 0.0) -> int:
        """Ajoute une zone proposee par l'auto-apprentissage."""
        x1, y1, x2, y2 = pct_bbox
        with self._lock:
            with self._connexion() as conn:
                cur = conn.execute(
                    """INSERT INTO zones_proposees
                    (session_id, camera_id, label, pct_x1, pct_y1, pct_x2, pct_y2,
                     duree_observation_sec, classe_detectee, confiance_moyenne)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, camera_id, f"{classe} (auto)", x1, y1, x2, y2,
                     duree_sec, classe, confiance),
                )
                return cur.lastrowid

    def obtenir_zones_proposees(self, session_id: int) -> List[Dict[str, Any]]:
        """Retourne les zones proposees d'une session."""
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM zones_proposees WHERE session_id = ? ORDER BY duree_observation_sec DESC",
                (session_id,),
            ).fetchall()]

    def valider_zone_proposee(self, zone_proposee_id: int, accepter: bool):
        """Valide ou rejette une zone proposee. Si acceptee, copie vers zones_exclusion."""
        statut = "acceptee" if accepter else "rejetee"
        with self._lock:
            with self._connexion() as conn:
                conn.execute(
                    "UPDATE zones_proposees SET statut = ? WHERE id = ?",
                    (statut, zone_proposee_id),
                )
                if accepter:
                    row = conn.execute(
                        "SELECT * FROM zones_proposees WHERE id = ?",
                        (zone_proposee_id,),
                    ).fetchone()
                    if row:
                        conn.execute(
                            """INSERT INTO zones_exclusion
                            (camera_id, label, pct_x1, pct_y1, pct_x2, pct_y2, source)
                            VALUES (?, ?, ?, ?, ?, ?, 'auto')""",
                            (row["camera_id"], row["label"],
                             row["pct_x1"], row["pct_y1"], row["pct_x2"], row["pct_y2"]),
                        )

    # =========================================================================
    # Utilisateurs alertes
    # =========================================================================

    def ajouter_utilisateur_alerte(
        self, nom: str, type_alerte: str, identifiant: str
    ) -> int:
        """Ajoute un utilisateur a alerter. Retourne l'ID."""
        with self._lock:
            with self._connexion() as conn:
                cur = conn.execute(
                    """INSERT INTO utilisateurs_alertes (nom, type_alerte, identifiant)
                    VALUES (?, ?, ?)""",
                    (nom, type_alerte, identifiant),
                )
                return cur.lastrowid

    def modifier_utilisateur_alerte(self, utilisateur_id: int, **champs):
        """Met a jour un utilisateur alerte."""
        colonnes = {"nom", "type_alerte", "identifiant", "actif"}
        updates = {k: v for k, v in champs.items() if k in colonnes}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [utilisateur_id]
        with self._lock:
            with self._connexion() as conn:
                conn.execute(
                    f"UPDATE utilisateurs_alertes SET {set_clause} WHERE id = ?", values
                )

    def supprimer_utilisateur_alerte(self, utilisateur_id: int):
        """Supprime un utilisateur (cascade supprime les associations camera)."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute(
                    "DELETE FROM camera_utilisateurs WHERE utilisateur_id = ?",
                    (utilisateur_id,),
                )
                conn.execute(
                    "DELETE FROM utilisateurs_alertes WHERE id = ?",
                    (utilisateur_id,),
                )

    def obtenir_utilisateurs_alertes(
        self, actifs_seulement: bool = False
    ) -> List[Dict[str, Any]]:
        """Retourne tous les utilisateurs alertes."""
        query = "SELECT * FROM utilisateurs_alertes"
        if actifs_seulement:
            query += " WHERE actif = 1"
        query += " ORDER BY nom"
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(query).fetchall()]

    def obtenir_utilisateur_alerte(self, utilisateur_id: int) -> Optional[Dict[str, Any]]:
        """Retourne un utilisateur par ID."""
        with self._connexion() as conn:
            row = conn.execute(
                "SELECT * FROM utilisateurs_alertes WHERE id = ?", (utilisateur_id,)
            ).fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Association cameras <-> utilisateurs
    # =========================================================================

    def definir_utilisateurs_camera(
        self, camera_id: int, utilisateur_ids: List[int]
    ):
        """Remplace les utilisateurs associes a une camera."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute(
                    "DELETE FROM camera_utilisateurs WHERE camera_id = ?",
                    (camera_id,),
                )
                for uid in utilisateur_ids:
                    conn.execute(
                        """INSERT OR IGNORE INTO camera_utilisateurs
                        (camera_id, utilisateur_id) VALUES (?, ?)""",
                        (camera_id, uid),
                    )

    def obtenir_utilisateurs_pour_camera(
        self, camera_id: int
    ) -> List[Dict[str, Any]]:
        """Retourne les utilisateurs actifs associes a une camera."""
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT ua.* FROM utilisateurs_alertes ua
                JOIN camera_utilisateurs cu ON ua.id = cu.utilisateur_id
                WHERE cu.camera_id = ? AND ua.actif = 1""",
                (camera_id,),
            ).fetchall()]

    def obtenir_cameras_pour_utilisateur(
        self, utilisateur_id: int
    ) -> List[Dict[str, Any]]:
        """Retourne les cameras associees a un utilisateur."""
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT c.* FROM cameras c
                JOIN camera_utilisateurs cu ON c.id = cu.camera_id
                WHERE cu.utilisateur_id = ?""",
                (utilisateur_id,),
            ).fetchall()]

    def obtenir_ids_utilisateurs_camera(self, camera_id: int) -> List[int]:
        """Retourne les IDs des utilisateurs associes a une camera."""
        with self._connexion() as conn:
            rows = conn.execute(
                "SELECT utilisateur_id FROM camera_utilisateurs WHERE camera_id = ?",
                (camera_id,),
            ).fetchall()
            return [r["utilisateur_id"] for r in rows]

    def obtenir_camera_par_nom(self, nom: str) -> Optional[Dict[str, Any]]:
        """Retourne une camera par son nom."""
        with self._connexion() as conn:
            row = conn.execute(
                "SELECT * FROM cameras WHERE nom = ?", (nom,)
            ).fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Alertes
    # =========================================================================

    def enregistrer_alerte(
        self, type_comportement: str, confiance: float, id_piste: int = 0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        chemin_video: str = "", chemin_snapshot: str = "",
        zone: str = "", source_camera: str = "",
        notifie_telegram: bool = False,
    ) -> int:
        with self._lock:
            with self._connexion() as conn:
                x1, y1, x2, y2 = bbox if bbox else (0, 0, 0, 0)
                curseur = conn.execute(
                    """INSERT INTO alertes
                    (type_comportement, confiance, id_piste,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                     chemin_video, chemin_snapshot, zone,
                     source_camera, notifie_telegram)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (type_comportement, confiance, id_piste,
                     x1, y1, x2, y2,
                     chemin_video, chemin_snapshot, zone,
                     source_camera, notifie_telegram),
                )
                id_alerte = curseur.lastrowid
                logger.info(
                    f"Alerte #{id_alerte} enregistree: {type_comportement} "
                    f"(confiance: {confiance:.2f}, piste: {id_piste})"
                )
                return id_alerte

    def obtenir_alertes_recentes(self, limite: int = 50) -> List[Dict[str, Any]]:
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM alertes ORDER BY horodatage DESC LIMIT ?", (limite,)
            ).fetchall()]

    def obtenir_alertes_par_date(self, date_debut: str, date_fin: str) -> List[Dict[str, Any]]:
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM alertes WHERE date(horodatage) BETWEEN ? AND ? ORDER BY horodatage DESC",
                (date_debut, date_fin),
            ).fetchall()]

    def obtenir_stats_journalieres(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        with self._connexion() as conn:
            stats = conn.execute(
                """SELECT COUNT(*) as total_alertes,
                    COUNT(DISTINCT id_piste) as incidents_uniques,
                    GROUP_CONCAT(DISTINCT type_comportement) as types_detectes
                FROM alertes WHERE date(horodatage) = ?""", (date_str,),
            ).fetchone()
            top = conn.execute(
                """SELECT type_comportement, COUNT(*) as nb FROM alertes
                WHERE date(horodatage) = ? GROUP BY type_comportement ORDER BY nb DESC LIMIT 1""",
                (date_str,),
            ).fetchone()
            pic = conn.execute(
                """SELECT strftime('%H:00', horodatage) as heure, COUNT(*) as nb FROM alertes
                WHERE date(horodatage) = ? GROUP BY heure ORDER BY nb DESC LIMIT 1""",
                (date_str,),
            ).fetchone()
            return {
                "date": date_str,
                "total_alertes": stats["total_alertes"] if stats else 0,
                "incidents_uniques": stats["incidents_uniques"] if stats else 0,
                "types_detectes": stats["types_detectes"] if stats else "",
                "comportement_frequent": top["type_comportement"] if top else "Aucun",
                "heure_pic": pic["heure"] if pic else "N/A",
            }

    def obtenir_stats_periode(self, jours: int = 30) -> List[Dict[str, Any]]:
        d_fin = datetime.now()
        d_debut = d_fin - timedelta(days=jours)
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT date(horodatage) as date, COUNT(*) as total_alertes,
                    COUNT(DISTINCT id_piste) as incidents_uniques
                FROM alertes WHERE date(horodatage) BETWEEN ? AND ?
                GROUP BY date(horodatage) ORDER BY date""",
                (d_debut.strftime("%Y-%m-%d"), d_fin.strftime("%Y-%m-%d")),
            ).fetchall()]

    def obtenir_alertes_par_heure(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT strftime('%H', horodatage) as heure, COUNT(*) as nb_alertes,
                    GROUP_CONCAT(DISTINCT type_comportement) as types
                FROM alertes WHERE date(horodatage) = ? GROUP BY heure ORDER BY heure""",
                (date_str,),
            ).fetchall()]

    def obtenir_repartition_comportements(self, jours: int = 30) -> List[Dict[str, Any]]:
        d = (datetime.now() - timedelta(days=jours)).strftime("%Y-%m-%d")
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT type_comportement, COUNT(*) as nombre, AVG(confiance) as confiance_moyenne
                FROM alertes WHERE date(horodatage) >= ? GROUP BY type_comportement ORDER BY nombre DESC""",
                (d,),
            ).fetchall()]

    def compter_alertes_aujourdhui(self) -> int:
        with self._connexion() as conn:
            r = conn.execute(
                "SELECT COUNT(*) FROM alertes WHERE date(horodatage) = ?",
                (datetime.now().strftime("%Y-%m-%d"),),
            ).fetchone()
            return r[0] if r else 0

    def nettoyer_anciennes_donnees(self, jours_retention: int = 30, **kwargs):
        """Nettoyage differencie par type de donnee.

        kwargs optionnels: retention_alertes_jours, retention_stats_jours.
        Si absents, utilise jours_retention comme fallback uniforme.
        """
        ret_alertes = kwargs.get("retention_alertes_jours", jours_retention)
        ret_stats = kwargs.get("retention_stats_jours", 365)

        with self._lock:
            with self._connexion() as conn:
                total_supprime = 0

                # Alertes
                seuil_a = (datetime.now() - timedelta(days=ret_alertes)).strftime("%Y-%m-%d")
                na = conn.execute("DELETE FROM alertes WHERE date(horodatage) < ?", (seuil_a,)).rowcount
                total_supprime += na
                if na > 0:
                    logger.info(f"Nettoyage: {na} alertes supprimees (avant {seuil_a})")

                # Stats journalieres
                seuil_s = (datetime.now() - timedelta(days=ret_stats)).strftime("%Y-%m-%d")
                ns = conn.execute("DELETE FROM stats_journalieres WHERE date < ?", (seuil_s,)).rowcount
                total_supprime += ns
                if ns > 0:
                    logger.info(f"Nettoyage: {ns} stats journalieres supprimees (avant {seuil_s})")

                # Sessions apprentissage terminees/annulees > 30j
                seuil_30 = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                n_sess = conn.execute(
                    "DELETE FROM sessions_apprentissage WHERE statut IN ('terminee', 'annulee') AND date(debut) < ?",
                    (seuil_30,),
                ).rowcount
                total_supprime += n_sess

                # Zones proposees rejetees > 7j
                seuil_7 = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                n_zp = conn.execute(
                    """DELETE FROM zones_proposees WHERE statut = 'rejetee'
                       AND session_id IN (SELECT id FROM sessions_apprentissage WHERE date(debut) < ?)""",
                    (seuil_7,),
                ).rowcount
                total_supprime += n_zp

                if total_supprime > 0:
                    conn.execute("VACUUM")
                    logger.info(f"Nettoyage total: {total_supprime} lignes supprimees")

    # =========================================================================
    # CRUD Alertes production (update + delete)
    # =========================================================================

    def modifier_alerte(self, alerte_id: int, **champs):
        """Met a jour les champs d'une alerte production."""
        colonnes = {"commentaire", "notifie_telegram", "zone", "source_camera"}
        updates = {k: v for k, v in champs.items() if k in colonnes}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [alerte_id]
        with self._lock:
            with self._connexion() as conn:
                conn.execute(f"UPDATE alertes SET {set_clause} WHERE id = ?", values)

    def supprimer_alerte(self, alerte_id: int):
        """Supprime une alerte production."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("DELETE FROM alertes WHERE id = ?", (alerte_id,))

    def obtenir_alerte(self, alerte_id: int) -> Optional[Dict[str, Any]]:
        """Retourne une alerte par ID."""
        with self._connexion() as conn:
            row = conn.execute("SELECT * FROM alertes WHERE id = ?", (alerte_id,)).fetchone()
            return dict(row) if row else None

    def compter_alertes(self, jours: int = 0) -> int:
        """Compte les alertes (0 = toutes, sinon sur N jours)."""
        with self._connexion() as conn:
            if jours > 0:
                seuil = (datetime.now() - timedelta(days=jours)).strftime("%Y-%m-%d")
                r = conn.execute(
                    "SELECT COUNT(*) FROM alertes WHERE date(horodatage) >= ?", (seuil,)
                ).fetchone()
            else:
                r = conn.execute("SELECT COUNT(*) FROM alertes").fetchone()
            return r[0] if r else 0

    # =========================================================================
    # Sessions de test — CRUD
    # =========================================================================

    def enregistrer_session_test(
        self, type_source: str, nom_fichier: str = "",
        nb_personnes: int = 0, nb_objets: int = 0, nb_alertes: int = 0,
        temps_inference_ms: float = 0.0, confiance_utilisee: float = 0.45,
        pose_activee: bool = True, chemin_snapshot: str = "",
        envoyee_telegram: bool = False, commentaire: str = "",
        nb_frames_analysees: int = 0, duree_video_sec: float = 0.0,
    ) -> int:
        """Cree une session de test. Retourne l'ID."""
        with self._lock:
            with self._connexion() as conn:
                cur = conn.execute(
                    """INSERT INTO sessions_test
                    (type_source, nom_fichier, nb_personnes, nb_objets, nb_alertes,
                     temps_inference_ms, confiance_utilisee, pose_activee,
                     chemin_snapshot, envoyee_telegram, commentaire,
                     nb_frames_analysees, duree_video_sec)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (type_source, nom_fichier, nb_personnes, nb_objets, nb_alertes,
                     temps_inference_ms, confiance_utilisee, pose_activee,
                     chemin_snapshot, envoyee_telegram, commentaire,
                     nb_frames_analysees, duree_video_sec),
                )
                logger.info(f"Session test #{cur.lastrowid} enregistree: {type_source} '{nom_fichier}'")
                return cur.lastrowid

    def enregistrer_alertes_test(self, session_id: int, alertes: List[Dict[str, Any]]):
        """Enregistre les alertes suspectes d'une session de test (batch)."""
        if not alertes:
            return
        with self._lock:
            with self._connexion() as conn:
                conn.executemany(
                    """INSERT INTO alertes_test
                    (session_id, personne_idx, type_comportement, score, severite, description)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (session_id, a.get("personne", 0), a.get("type", ""),
                         a.get("score", 0.0), a.get("severite", "BASSE"),
                         a.get("description", ""))
                        for a in alertes
                    ],
                )

    def obtenir_sessions_test(
        self, limite: int = 100, type_source: Optional[str] = None,
        date_debut: Optional[str] = None, date_fin: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retourne les sessions de test avec filtres optionnels."""
        query = "SELECT * FROM sessions_test WHERE 1=1"
        params: list = []
        if type_source:
            query += " AND type_source = ?"
            params.append(type_source)
        if date_debut:
            query += " AND date(horodatage) >= ?"
            params.append(date_debut)
        if date_fin:
            query += " AND date(horodatage) <= ?"
            params.append(date_fin)
        query += " ORDER BY horodatage DESC LIMIT ?"
        params.append(limite)
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def obtenir_session_test(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Retourne une session de test par ID."""
        with self._connexion() as conn:
            row = conn.execute(
                "SELECT * FROM sessions_test WHERE id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def obtenir_alertes_test_session(self, session_id: int) -> List[Dict[str, Any]]:
        """Retourne les alertes d'une session de test."""
        with self._connexion() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM alertes_test WHERE session_id = ? ORDER BY score DESC",
                (session_id,),
            ).fetchall()]

    def modifier_session_test(self, session_id: int, **champs):
        """Met a jour les champs d'une session de test."""
        colonnes = {"commentaire", "envoyee_telegram"}
        updates = {k: v for k, v in champs.items() if k in colonnes}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        with self._lock:
            with self._connexion() as conn:
                conn.execute(f"UPDATE sessions_test SET {set_clause} WHERE id = ?", values)

    def supprimer_session_test(self, session_id: int):
        """Supprime une session de test et ses alertes (cascade)."""
        with self._lock:
            with self._connexion() as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("DELETE FROM alertes_test WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions_test WHERE id = ?", (session_id,))

    def compter_sessions_test(self, jours: int = 0) -> int:
        """Compte les sessions de test."""
        with self._connexion() as conn:
            if jours > 0:
                seuil = (datetime.now() - timedelta(days=jours)).strftime("%Y-%m-%d")
                r = conn.execute(
                    "SELECT COUNT(*) FROM sessions_test WHERE date(horodatage) >= ?", (seuil,)
                ).fetchone()
            else:
                r = conn.execute("SELECT COUNT(*) FROM sessions_test").fetchone()
            return r[0] if r else 0

    def obtenir_stats_sessions_test(self) -> Dict[str, Any]:
        """Statistiques globales des sessions de test."""
        with self._connexion() as conn:
            r = conn.execute("""
                SELECT COUNT(*) as total,
                    SUM(CASE WHEN type_source='image' THEN 1 ELSE 0 END) as nb_images,
                    SUM(CASE WHEN type_source='video' THEN 1 ELSE 0 END) as nb_videos,
                    SUM(CASE WHEN type_source='webcam' THEN 1 ELSE 0 END) as nb_webcam,
                    SUM(nb_alertes) as total_alertes,
                    AVG(temps_inference_ms) as inference_moyenne_ms,
                    SUM(CASE WHEN envoyee_telegram=1 THEN 1 ELSE 0 END) as nb_envoyees_telegram
                FROM sessions_test
            """).fetchone()
            return dict(r) if r else {}

    def nettoyer_sessions_test(self, jours_retention: int = 30):
        """Supprime les sessions de test anciennes."""
        seuil = (datetime.now() - timedelta(days=jours_retention)).strftime("%Y-%m-%d")
        with self._lock:
            with self._connexion() as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                # Recuperer les IDs a supprimer pour cascade manuelle
                ids = [r[0] for r in conn.execute(
                    "SELECT id FROM sessions_test WHERE date(horodatage) < ?", (seuil,)
                ).fetchall()]
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    conn.execute(f"DELETE FROM alertes_test WHERE session_id IN ({placeholders})", ids)
                    ns = conn.execute(f"DELETE FROM sessions_test WHERE id IN ({placeholders})", ids).rowcount
                    logger.info(f"Nettoyage sessions test: {ns} sessions supprimees (avant {seuil})")

"""
Dashboard Streamlit pour le suivi ROI de la detection de fraude.
Inclut une page d'administration pour configurer tous les parametres.
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import obtenir_config
from app.database import BaseDonneesFraude
from dashboard.live_camera import CameraLiveViewer

# --- Configuration de la page ---
st.set_page_config(
    page_title="FraudeGuard - Detection Fraude Magasin",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CONSTANTS
# ============================================================================

ZONES_MAGASIN = [
    "entree", "caisse", "sortie", "rayon_A", "rayon_B",
    "rayon_C", "allee_gauche", "allee_centrale",
    "allee_droite", "reserve", "inconnue",
]

NIVEAUX_MAGASIN = ["Niveau -1", "Niveau 0", "Niveau 1", "Niveau 2"]


# ============================================================================
# THEME & CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #132238 50%, #1a2d4a 100%);
        border-right: 1px solid rgba(79,195,247,0.15);
    }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }
    [data-testid="stSidebar"] .stRadio > div { gap: 2px; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 12px 16px !important; border-radius: 10px !important;
        transition: all 0.2s ease !important; font-weight: 500 !important; font-size: 0.95em !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(79,195,247,0.1) !important; color: #4fc3f7 !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: rgba(79,195,247,0.15) !important;
        border-left: 3px solid #4fc3f7 !important; color: #4fc3f7 !important;
    }

    .stApp { background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%); }
    .block-container { max-width: 1400px; }

    .kpi-card {
        background: linear-gradient(135deg, rgba(30,58,95,0.8), rgba(45,89,134,0.6));
        border: 1px solid rgba(79,195,247,0.15); border-radius: 16px;
        padding: 20px 18px; text-align: center; margin: 5px 0;
        backdrop-filter: blur(10px); box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, var(--accent, #4fc3f7), transparent);
    }
    .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: rgba(79,195,247,0.3); }
    .kpi-icon { font-size: 1.6em; margin-bottom: 4px; }
    .kpi-value {
        font-size: 2em; font-weight: 800; letter-spacing: -1px; margin: 6px 0;
        background: linear-gradient(135deg, #ffffff, #b0c4de);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .kpi-label { font-size: 0.78em; font-weight: 500; text-transform: uppercase; letter-spacing: 1.5px; color: #8899aa; }
    .kpi-rouge { --accent: #ef4444; border-left: 4px solid #ef4444; }
    .kpi-rouge .kpi-value { background: linear-gradient(135deg, #ff6b6b, #ef4444); -webkit-background-clip: text; }
    .kpi-orange { --accent: #f59e0b; border-left: 4px solid #f59e0b; }
    .kpi-orange .kpi-value { background: linear-gradient(135deg, #fbbf24, #f59e0b); -webkit-background-clip: text; }
    .kpi-vert { --accent: #22c55e; border-left: 4px solid #22c55e; }
    .kpi-vert .kpi-value { background: linear-gradient(135deg, #4ade80, #22c55e); -webkit-background-clip: text; }
    .kpi-bleu { --accent: #4fc3f7; border-left: 4px solid #4fc3f7; }
    .kpi-bleu .kpi-value { background: linear-gradient(135deg, #81d4fa, #4fc3f7); -webkit-background-clip: text; }

    .page-header {
        display: flex; align-items: center; gap: 16px;
        padding: 0 0 16px 0; margin-bottom: 10px;
        border-bottom: 1px solid rgba(100,120,150,0.2);
    }
    .page-header-icon {
        font-size: 2.2em;
        background: linear-gradient(135deg, rgba(79,195,247,0.2), rgba(79,195,247,0.05));
        border-radius: 16px; width: 60px; height: 60px;
        display: flex; align-items: center; justify-content: center;
        border: 1px solid rgba(79,195,247,0.2);
    }
    .page-header-text h1 { margin: 0; font-size: 1.7em; font-weight: 700; color: #e8edf2; }
    .page-header-text p { margin: 4px 0 0 0; font-size: 0.88em; color: #7a8fa3; }
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.7em; font-weight: 700; letter-spacing: 0.8px;
        text-transform: uppercase; vertical-align: middle; margin-left: 8px;
    }
    .badge-live { background: #22c55e; color: white; }
    .badge-admin { background: #6366f1; color: white; }

    .section-header {
        display: flex; align-items: center; gap: 10px;
        margin: 24px 0 14px 0; padding-bottom: 8px;
        border-bottom: 2px solid rgba(79,195,247,0.15);
    }
    .section-header span.icon { font-size: 1.2em; }
    .section-header span.text {
        font-size: 0.9em; font-weight: 600; text-transform: uppercase;
        letter-spacing: 2px; color: #7a8fa3;
    }

    .info-card {
        background: rgba(25,45,70,0.5); border: 1px solid rgba(100,120,150,0.2);
        border-radius: 12px; padding: 16px 20px; margin: 8px 0;
    }
    .info-card h4 { margin: 0 0 8px 0; color: #b0c4de; font-size: 0.95em; font-weight: 600; }
    .info-card p { margin: 0; color: #8899aa; font-size: 0.85em; line-height: 1.5; }

    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .status-active { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.5); }
    .status-inactive { background: #ef4444; }

    .stForm {
        border: 1px solid rgba(79,195,247,0.12) !important; border-radius: 14px !important;
        padding: 20px !important; background: rgba(15,30,50,0.3) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(15,25,40,0.5); border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px !important; padding: 8px 16px !important; font-weight: 500 !important; font-size: 0.88em !important; }
    .stTabs [aria-selected="true"] { background: rgba(79,195,247,0.15) !important; color: #4fc3f7 !important; }
    .streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.95em !important; background: rgba(20,35,55,0.5) !important; border-radius: 10px !important; }
    .stDataFrame { border-radius: 10px !important; overflow: hidden; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4fc3f7, #0288d1) !important; border: none !important;
        font-weight: 600 !important; border-radius: 10px !important; padding: 8px 24px !important; transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 4px 15px rgba(79,195,247,0.4) !important; transform: translateY(-1px) !important; }
    [data-testid="stMetricValue"] { font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-weight: 500 !important; }
    hr { border-color: rgba(100,120,150,0.15) !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); }

    .sidebar-brand { text-align: center; padding: 20px 10px 16px; margin-bottom: 10px; border-bottom: 1px solid rgba(79,195,247,0.15); }
    .sidebar-brand .logo { font-size: 2.4em; margin-bottom: 4px; }
    .sidebar-brand h2 {
        margin: 0; font-size: 1.2em; font-weight: 700;
        background: linear-gradient(135deg, #4fc3f7, #81d4fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p { margin: 4px 0 0 0; font-size: 0.72em; letter-spacing: 2px; text-transform: uppercase; color: #5a7a94 !important; }
    .sidebar-status {
        background: rgba(20,40,65,0.6); border: 1px solid rgba(79,195,247,0.1);
        border-radius: 10px; padding: 12px; margin: 10px 0; font-size: 0.82em;
    }
    .sidebar-status .row { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; }
    .empty-state { text-align: center; padding: 40px 20px; color: #5a7a94; }
    .empty-state .icon { font-size: 3em; margin-bottom: 10px; opacity: 0.5; }
    .empty-state p { font-size: 0.9em; max-width: 400px; margin: 8px auto; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================

@st.cache_resource
def obtenir_db():
    config = obtenir_config()
    return BaseDonneesFraude(config.chemin_base_donnees)


@st.cache_resource
def obtenir_viewer():
    config = obtenir_config()
    db = obtenir_db()
    taille_yolo = int(db.obtenir_parametre("taille_entree_yolo", 320))
    return CameraLiveViewer(chemin_modeles=config.model_path, taille_entree_yolo=taille_yolo)


def afficher_kpi(label: str, valeur: str, icone: str = "📊", classe: str = ""):
    st.markdown(f"""
    <div class="kpi-card {classe}">
        <div class="kpi-icon">{icone}</div>
        <div class="kpi-value">{valeur}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def afficher_header(icone: str, titre: str, description: str, badge: str = "", badge_class: str = ""):
    badge_html = f'<span class="badge {badge_class}">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon">{icone}</div>
        <div class="page-header-text">
            <h1>{titre}{badge_html}</h1>
            <p>{description}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def afficher_section(icone: str, titre: str):
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icone}</span>
        <span class="text">{titre}</span>
    </div>
    """, unsafe_allow_html=True)


def afficher_info_card(titre: str, texte: str):
    st.markdown(f"""
    <div class="info-card">
        <h4>{titre}</h4>
        <p>{texte}</p>
    </div>
    """, unsafe_allow_html=True)


def afficher_etat_vide(icone: str, message: str, sous_message: str = ""):
    sm = f"<p>{sous_message}</p>" if sous_message else ""
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icone}</div>
        <p><strong>{message}</strong></p>
        {sm}
    </div>
    """, unsafe_allow_html=True)


def afficher_frame_bgr(frame, caption=None):
    """Convertit BGR→RGB et affiche via st.image."""
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


def appliquer_theme_plotly(fig):
    """Applique le theme sombre aux graphiques Plotly."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,25,40,0.4)",
        font=dict(color="#c8d6e5", family="Inter, sans-serif"),
        title_font=dict(color="#e0e6ed", size=15),
        xaxis=dict(gridcolor="rgba(100,120,150,0.15)", zerolinecolor="rgba(100,120,150,0.2)"),
        yaxis=dict(gridcolor="rgba(100,120,150,0.15)", zerolinecolor="rgba(100,120,150,0.2)"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _obtenir_params_dict(db):
    """Charge tous les parametres en un seul appel et retourne un dict cle→valeur."""
    tous = db.obtenir_tous_parametres() or []
    result = {}
    for p in tous:
        cle = p["cle"]
        val = p["valeur"]
        tv = p.get("type_valeur", "str")
        try:
            if tv == "float":
                result[cle] = float(val)
            elif tv == "int":
                result[cle] = int(val)
            elif tv == "bool":
                result[cle] = val.lower() in ("true", "1", "yes")
            else:
                result[cle] = val
        except (ValueError, TypeError):
            result[cle] = val
    return result


def _param(params: dict, cle: str, defaut):
    """Recupere un parametre du dict avec fallback."""
    val = params.get(cle)
    if val is None:
        return defaut
    return type(defaut)(val) if not isinstance(val, type(defaut)) else val



# ============================================================================
# SIDEBAR
# ============================================================================

def afficher_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="logo">🛡️</div>
            <h2>FraudeGuard</h2>
            <p>Surveillance Intelligente</p>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "📋 Historique", "⚙️ Administration"],
            label_visibility="collapsed",
        )

        db = obtenir_db()
        cameras = db.obtenir_cameras() or []
        nb_cameras = len(cameras)
        nb_cameras_actives = sum(1 for c in cameras if c.get("active"))
        nb_users = len(db.obtenir_utilisateurs_alertes() or [])
        tg_ok = bool(db.obtenir_parametre("telegram_bot_token", ""))

        st.markdown(f"""
        <div class="sidebar-status">
            <div class="row"><span>Cameras</span><span><strong>{nb_cameras_actives}</strong>/{nb_cameras}</span></div>
            <div class="row"><span>Utilisateurs</span><span><strong>{nb_users}</strong></span></div>
            <div class="row"><span><span class="status-dot {"status-active" if tg_ok else "status-inactive"}"></span>Telegram</span><span>{"OK" if tg_ok else "—"}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"v2.0 — {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    return page


# ============================================================================
# PAGE DASHBOARD
# ============================================================================

def page_dashboard():
    db = obtenir_db()
    params = _obtenir_params_dict(db)

    valeur_article = _param(params, "valeur_article_moyen_dh", 150.0)
    refresh_sec = _param(params, "dashboard_refresh_seconds", 10)

    afficher_header(
        "🛡️", "Dashboard Detection de Fraude",
        "Surveillance en temps reel des incidents et performance du systeme",
        "LIVE", "badge-live",
    )

    with st.sidebar:
        afficher_section("📅", "Periode")
        periode = st.selectbox(
            "Periode d'analyse",
            ["Aujourd'hui", "7 derniers jours", "30 derniers jours", "Personnalise"],
            label_visibility="collapsed",
        )
        if periode == "Personnalise":
            date_debut = st.date_input("Debut", value=datetime.now() - timedelta(days=7))
            date_fin = st.date_input("Fin", value=datetime.now())
        elif periode == "Aujourd'hui":
            date_debut = date_fin = datetime.now().date()
        elif periode == "7 derniers jours":
            date_fin = datetime.now().date()
            date_debut = date_fin - timedelta(days=7)
        else:
            date_fin = datetime.now().date()
            date_debut = date_fin - timedelta(days=30)

        auto_refresh = st.toggle("Auto-rafraichissement", value=True)
        st.caption(f"Valeur article: **{valeur_article:.0f} DH**")

    date_debut_str = str(date_debut)
    date_fin_str = str(date_fin)
    jours_periode = (date_fin - date_debut).days + 1

    alertes = db.obtenir_alertes_par_date(date_debut_str, date_fin_str)
    df_alertes = pd.DataFrame(alertes) if alertes else pd.DataFrame()

    # --- KPI ---
    nb_alertes = len(df_alertes) if not df_alertes.empty else 0
    nb_incidents = df_alertes["id_piste"].nunique() if not df_alertes.empty else 0
    montant_evite = nb_incidents * valeur_article
    taux_horaire = 0.0
    if not df_alertes.empty and "horodatage" in df_alertes.columns:
        try:
            ts = pd.to_datetime(df_alertes["horodatage"])
            heures = max(1, (ts.max() - ts.min()).total_seconds() / 3600)
            taux_horaire = nb_alertes / heures
        except Exception as e:
            logger.warning(f"Erreur calcul taux horaire alertes: {e}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        afficher_kpi("Alertes (periode)", str(nb_alertes), "🚨", "kpi-rouge")
    with col2:
        afficher_kpi("Incidents uniques", str(nb_incidents), "👤", "kpi-orange")
    with col3:
        afficher_kpi("Vol estime evite", f"{montant_evite:,.0f} DH", "💰", "kpi-vert")
    with col4:
        afficher_kpi("Alertes / heure", f"{taux_horaire:.1f}", "⏱️", "kpi-bleu")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Alertes par heure", "🎯 Comportements",
        "📋 Historique", "💰 ROI Mensuel",
    ])

    with tab1:
        afficher_section("📊", "Repartition horaire")
        alertes_par_heure = db.obtenir_alertes_par_heure()
        if alertes_par_heure:
            df_h = pd.DataFrame(alertes_par_heure)
            all_h = pd.DataFrame({"heure": [f"{h:02d}" for h in range(24)]})
            df_h = all_h.merge(df_h, on="heure", how="left").fillna(0)
            fig = px.bar(df_h, x="heure", y="nb_alertes",
                         labels={"heure": "Heure", "nb_alertes": "Nombre d'alertes"},
                         color="nb_alertes",
                         color_continuous_scale=["#1a3a5c", "#4fc3f7", "#ef5350"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            appliquer_theme_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            afficher_etat_vide("📊", "Aucune alerte aujourd'hui", "Les alertes apparaitront ici en temps reel")

    with tab2:
        afficher_section("🎯", "Analyse des comportements")
        repartition = db.obtenir_repartition_comportements(jours=jours_periode)
        if repartition:
            df_rep = pd.DataFrame(repartition)
            ca, cb = st.columns(2)
            with ca:
                fig_pie = px.pie(df_rep, names="type_comportement", values="nombre",
                                  title="Repartition par type",
                                  color_discrete_sequence=["#4fc3f7", "#ef5350", "#66bb6a", "#ffa726", "#ab47bc", "#26c6da"],
                                  hole=0.4)
                appliquer_theme_plotly(fig_pie)
                st.plotly_chart(fig_pie, use_container_width=True)
            with cb:
                fig_bar = px.bar(df_rep, x="type_comportement", y="nombre",
                                  color="confiance_moyenne", title="Volume par type",
                                  color_continuous_scale=["#1a3a5c", "#ffa726", "#ef5350"])
                appliquer_theme_plotly(fig_bar)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            afficher_etat_vide("🎯", "Aucune donnee de comportement")

    with tab3:
        afficher_section("📋", "Historique des alertes")
        if not df_alertes.empty:
            cols = [c for c in ["horodatage", "type_comportement", "confiance", "id_piste", "zone", "source_camera"]
                    if c in df_alertes.columns]
            df_disp = df_alertes[cols].copy()
            if "confiance" in df_disp.columns:
                df_disp["confiance"] = df_disp["confiance"].apply(
                    lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"
                )

            c_info1, c_info2 = st.columns([3, 1])
            with c_info1:
                st.caption(f"**{len(df_disp)}** alertes entre le **{date_debut_str}** et le **{date_fin_str}**")
            with c_info2:
                types_dispo = sorted(df_disp["type_comportement"].unique().tolist()) if "type_comportement" in df_disp.columns else []
                filtre_type = st.selectbox("Filtrer", ["Tous"] + types_dispo, key="filtre_type_alerte", label_visibility="collapsed")

            if filtre_type != "Tous" and "type_comportement" in df_disp.columns:
                df_disp = df_disp[df_disp["type_comportement"] == filtre_type]

            st.dataframe(df_disp, use_container_width=True, height=400)

            # Generer Excel seulement si telecharge
            @st.cache_data
            def _generer_excel(df):
                buf = BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                return buf.getvalue()

            st.download_button(
                "📥 Telecharger Excel",
                data=_generer_excel(df_alertes),
                file_name=f"alertes_{date_debut_str}_{date_fin_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            afficher_etat_vide("📋", "Aucune alerte pour cette periode")

    with tab4:
        afficher_section("💰", "Retour sur investissement")
        stats_p = db.obtenir_stats_periode(jours=30)
        if stats_p:
            df_s = pd.DataFrame(stats_p)
            df_s["montant_evite"] = df_s["incidents_uniques"] * valeur_article

            c1, c2 = st.columns(2)
            with c1:
                fig_line = px.line(df_s, x="date", y="total_alertes", title="Alertes sur 30 jours",
                                    markers=True, color_discrete_sequence=["#4fc3f7"])
                fig_line.update_traces(line=dict(width=2.5), marker=dict(size=6))
                appliquer_theme_plotly(fig_line)
                st.plotly_chart(fig_line, use_container_width=True)
            with c2:
                fig_area = px.area(df_s, x="date", y="montant_evite", title="Vol evite par jour (DH)",
                                    color_discrete_sequence=["#22c55e"])
                appliquer_theme_plotly(fig_area)
                st.plotly_chart(fig_area, use_container_width=True)

            total = df_s["montant_evite"].sum()
            moy = df_s["montant_evite"].mean()
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                afficher_kpi("Total evite (30j)", f"{total:,.0f} DH", "💵", "kpi-vert")
            with sc2:
                afficher_kpi("Moyenne / jour", f"{moy:,.0f} DH", "📈", "kpi-orange")
            with sc3:
                afficher_kpi("Projection annuelle", f"{moy * 365:,.0f} DH", "🎯", "kpi-bleu")
        else:
            afficher_etat_vide("💰", "Pas assez de donnees")

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


# ============================================================================
# PAGE ADMINISTRATION
# ============================================================================

def page_administration():
    db = obtenir_db()
    params = _obtenir_params_dict(db)

    afficher_header(
        "⚙️", "Administration",
        "Configuration de tous les parametres du systeme de detection",
        "ADMIN", "badge-admin",
    )

    tab_cam, tab_train, tab_alert, tab_users, tab_vol, tab_metier, tab_sys = st.tabs([
        "📷 Cameras", "🎯 Entrainement",
        "🔔 Alertes", "👥 Utilisateurs", "🕵️ Detection Vol",
        "🏪 Metier", "🖥️ Systeme",
    ])

    # === CAMERAS ===
    with tab_cam:
        afficher_section("📷", "Gestion des cameras")
        cameras = db.obtenir_cameras()

        if cameras:
            nb_actives = sum(1 for c in cameras if c["active"])
            st.markdown(f"**{len(cameras)} camera(s)** — "
                        f"<span style='color:#22c55e'>{nb_actives} active(s)</span>",
                        unsafe_allow_html=True)

            MODES_DETECTION = ["tout", "vol", "caisse"]
            MODES_LABELS = {"tout": "Tout (vol + caisse)", "vol": "Vol uniquement", "caisse": "Fraude caisse uniquement"}

            for cam in cameras:
                status_label = "Active" if cam["active"] else "Inactive"
                mode = cam.get("mode_detection", "tout")
                mode_icon = {"tout": "🔍", "vol": "🛍️", "caisse": "💰"}.get(mode, "🔍")
                niveau_cam = cam.get("niveau", "Niveau 0")
                with st.expander(f"{'🟢' if cam['active'] else '🔴'} {cam['nom']} — {niveau_cam} / {cam['zone']} {mode_icon} ({status_label})"):
                    with st.form(key=f"cam_edit_{cam['id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            nom = st.text_input("Nom", value=cam["nom"], key=f"cn_{cam['id']}")
                            source = st.text_input("Source", value=cam["source"], key=f"cs_{cam['id']}")
                        with col2:
                            idx = ZONES_MAGASIN.index(cam["zone"]) if cam["zone"] in ZONES_MAGASIN else 0
                            zone = st.selectbox("Zone", ZONES_MAGASIN, index=idx, key=f"cz_{cam['id']}")
                            niv_idx = NIVEAUX_MAGASIN.index(niveau_cam) if niveau_cam in NIVEAUX_MAGASIN else 1
                            niveau = st.selectbox("Niveau", NIVEAUX_MAGASIN, index=niv_idx, key=f"cniv_{cam['id']}")
                            pos = st.text_input("Position", value=cam.get("position_description", ""), key=f"cp_{cam['id']}")
                        col_mode, col_active = st.columns(2)
                        with col_mode:
                            mode_idx = MODES_DETECTION.index(mode) if mode in MODES_DETECTION else 0
                            mode_det = st.selectbox("Mode detection", MODES_DETECTION, index=mode_idx,
                                                     format_func=lambda m: MODES_LABELS.get(m, m),
                                                     key=f"cm_{cam['id']}")
                        with col_active:
                            active = st.toggle("Active", value=bool(cam["active"]), key=f"ca_{cam['id']}")
                        c_save, c_del = st.columns([3, 1])
                        with c_save:
                            if st.form_submit_button("💾 Sauvegarder", type="primary"):
                                db.modifier_camera(cam["id"], nom=nom, source=source,
                                                   zone=zone, niveau=niveau, position_description=pos,
                                                   active=active, mode_detection=mode_det)
                                st.success(f"Camera '{nom}' mise a jour")
                                st.rerun()
                        with c_del:
                            if st.form_submit_button("🗑️ Supprimer"):
                                db.supprimer_camera(cam["id"])
                                st.success(f"Camera '{cam['nom']}' supprimee")
                                st.rerun()
        else:
            afficher_etat_vide("📷", "Aucune camera configuree", "Ajoutez votre premiere camera ci-dessous")

        afficher_section("➕", "Nouvelle camera")
        with st.form("form_nouvelle_camera"):
            c1, c2 = st.columns(2)
            with c1:
                new_nom = st.text_input("Nom", placeholder="cam_entree_1")
                new_source = st.text_input("Source", placeholder="rtsp://admin:password@192.168.1.100:554/Streaming/Channels/102")
            with c2:
                new_zone = st.selectbox("Zone", ZONES_MAGASIN)
                new_niveau = st.selectbox("Niveau", NIVEAUX_MAGASIN, index=1)
                new_pos = st.text_input("Position", placeholder="Au-dessus de la porte principale")
            new_mode = st.selectbox("Mode detection", ["tout", "vol", "caisse"],
                                     format_func=lambda m: {"tout": "🔍 Tout (vol + caisse)",
                                                             "vol": "🛍️ Vol uniquement",
                                                             "caisse": "💰 Fraude caisse uniquement"}.get(m, m))
            if st.form_submit_button("➕ Ajouter", type="primary"):
                if new_nom and new_source:
                    try:
                        db.ajouter_camera(new_nom, new_source, new_zone, new_niveau, new_pos, new_mode)
                        st.success(f"Camera '{new_nom}' ajoutee ({new_mode})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur: {e}")
                else:
                    st.warning("Le nom et la source sont obligatoires")

    # === DETECTION ===
    # === ALERTES ===
    with tab_alert:
        afficher_section("🔔", "Configuration des alertes")

        with st.form("form_alertes"):
            afficher_info_card(
                "Canaux d'alerte",
                "Parametres generaux et Telegram. "
                "Les destinataires par camera se configurent dans l'onglet Utilisateurs.",
            )

            st.markdown("**Parametres generaux**")
            c1, c2, c3 = st.columns(3)
            with c1:
                alerte_son = st.toggle("Son d'alerte", value=_param(params, "alert_sound", True))
            with c2:
                cooldown = st.number_input("Cooldown (s)", min_value=5, max_value=600,
                    value=int(_param(params, "alert_cooldown_seconds", 30)))
            with c3:
                clip_dur = st.number_input("Duree clip (s)", min_value=5, max_value=120,
                    value=int(_param(params, "video_clip_duration", 30)))

            st.divider()
            st.markdown("**Telegram**")
            tc1, tc2 = st.columns(2)
            with tc1:
                tg_token = st.text_input("Bot Token", value=_param(params, "telegram_bot_token", ""), type="password")
            with tc2:
                tg_chat = st.text_input("Chat ID (global)", value=_param(params, "telegram_chat_id", ""))


            if st.form_submit_button("💾 Sauvegarder", type="primary"):
                for cle, val, cat, tv in [
                    ("alert_sound", str(alerte_son).lower(), "alertes", "bool"),
                    ("alert_cooldown_seconds", str(cooldown), "alertes", "int"),
                    ("video_clip_duration", str(clip_dur), "alertes", "int"),
                    ("telegram_bot_token", tg_token, "telegram", "str"),
                    ("telegram_chat_id", tg_chat, "telegram", "str"),
                ]:
                    db.definir_parametre(cle, val, cat, type_valeur=tv)
                st.success("Parametres d'alertes sauvegardes")

        afficher_section("🧪", "Test de connexion")
        if st.button("📱 Tester Telegram", use_container_width=True):
            token = _param(params, "telegram_bot_token", "")
            chat_id = _param(params, "telegram_chat_id", "")
            if token and chat_id:
                try:
                    r = requests.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={"chat_id": chat_id, "text": "🧪 Test FraudeGuard — Connexion Telegram OK"},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.success("Message Telegram envoye")
                    else:
                        st.error(f"Erreur: {r.status_code} — {r.text}")
                except Exception as e:
                    st.error(f"Erreur: {e}")
            else:
                st.warning("Configurez le token et le chat ID d'abord")

    # === UTILISATEURS ===
    with tab_users:
        afficher_section("👥", "Gestion des utilisateurs")
        afficher_info_card(
            "Routage des alertes par camera",
            "Ajoutez des utilisateurs Telegram et assignez-les aux cameras. "
            "Sans assignation, les alertes utilisent la config globale.",
        )

        afficher_section("➕", "Nouvel utilisateur")
        with st.form("form_ajout_utilisateur", clear_on_submit=True):
            col_u1, col_u2, col_u3 = st.columns([2, 1, 3])
            with col_u1:
                u_nom = st.text_input("Nom", placeholder="Ahmed — Securite")
            with col_u2:
                u_type = "telegram"
                st.text_input("Canal", value="Telegram", disabled=True)
            with col_u3:
                u_ident = st.text_input("Chat ID", placeholder="123456789")
            if st.form_submit_button("➕ Ajouter", type="primary"):
                if u_nom and u_ident:
                    try:
                        db.ajouter_utilisateur_alerte(u_nom, u_type, u_ident.strip())
                        st.success(f"Utilisateur '{u_nom}' ajoute")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur: {e}")
                else:
                    st.warning("Remplissez tous les champs")

        afficher_section("📋", "Utilisateurs enregistres")
        utilisateurs = db.obtenir_utilisateurs_alertes()

        if not utilisateurs:
            afficher_etat_vide("👥", "Aucun utilisateur enregistre")
        else:
            nb_actifs = sum(1 for u in utilisateurs if u["actif"])
            st.markdown(
                f"**{len(utilisateurs)}** utilisateur(s) — "
                f"<span style='color:#22c55e'>{nb_actifs} actif(s)</span>",
                unsafe_allow_html=True,
            )

            for u in utilisateurs:
                icone = "📱" if u["type_alerte"] == "telegram" else "📧"
                statut = "✅" if u["actif"] else "❌"
                with st.expander(f"{icone} {u['nom']} — {u['identifiant']} {statut}"):
                    with st.form(f"form_edit_user_{u['id']}"):
                        col_e1, col_e2, col_e3 = st.columns([2, 1, 3])
                        with col_e1:
                            e_nom = st.text_input("Nom", value=u["nom"], key=f"enom_{u['id']}")
                        with col_e2:
                            e_type = "telegram"
                            st.text_input("Canal", value="Telegram", disabled=True, key=f"etype_{u['id']}")
                        with col_e3:
                            e_ident = st.text_input("Identifiant", value=u["identifiant"], key=f"eident_{u['id']}")
                        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
                        with col_b1:
                            e_actif = st.checkbox("Actif", value=bool(u["actif"]), key=f"eactif_{u['id']}")
                        with col_b2:
                            btn_save = st.form_submit_button("💾 Sauvegarder")
                        with col_b3:
                            btn_del = st.form_submit_button("🗑️ Supprimer")
                        if btn_save:
                            db.modifier_utilisateur_alerte(u["id"], nom=e_nom, type_alerte=e_type,
                                                           identifiant=e_ident.strip(), actif=e_actif)
                            st.success("Mis a jour")
                            st.rerun()
                        if btn_del:
                            db.supprimer_utilisateur_alerte(u["id"])
                            st.success(f"'{u['nom']}' supprime")
                            st.rerun()

        # Assignation cameras <-> utilisateurs
        afficher_section("🔗", "Assignation aux cameras")
        cameras = db.obtenir_cameras()

        if not cameras:
            afficher_etat_vide("📷", "Aucune camera")
        elif not utilisateurs:
            afficher_etat_vide("👥", "Aucun utilisateur")
        else:
            user_map = {
                u["id"]: f"{u['nom']} ({'📱' if u['type_alerte'] == 'telegram' else '📧'} {u['identifiant']})"
                for u in utilisateurs
            }
            label_to_id = {v: k for k, v in user_map.items()}

            with st.form("form_assignation_cameras"):
                assignations = {}
                for cam in cameras:
                    ids_actuels = db.obtenir_ids_utilisateurs_camera(cam["id"])
                    options = list(user_map.values())
                    defaults = [user_map[uid] for uid in ids_actuels if uid in user_map]
                    selection = st.multiselect(
                        f"📷 {cam['nom']} ({cam.get('niveau', 'Niveau 0')} / {cam['zone']})",
                        options=options, default=defaults, key=f"assign_cam_{cam['id']}",
                    )
                    assignations[cam["id"]] = [label_to_id[s] for s in selection if s in label_to_id]

                if st.form_submit_button("💾 Sauvegarder les assignations", type="primary"):
                    for cam_id, uids in assignations.items():
                        db.definir_utilisateurs_camera(cam_id, uids)
                    st.success(f"Assignations sauvegardees pour {len(cameras)} camera(s)")
                    st.rerun()

            afficher_section("📊", "Resume")
            resume_data = []
            for cam in cameras:
                users_cam = db.obtenir_utilisateurs_pour_camera(cam["id"])
                tg = [u["identifiant"] for u in users_cam if u["type_alerte"] == "telegram"]
                resume_data.append({
                    "Camera": cam["nom"], "Niveau": cam.get("niveau", "Niveau 0"),
                    "Zone": cam["zone"],
                    "Telegram": ", ".join(tg) or "(global)",
                })
            st.dataframe(pd.DataFrame(resume_data), use_container_width=True, hide_index=True)

    # === DETECTION VOL ===
    with tab_vol:
        afficher_section("🕵️", "Parametres de detection de vol")
        afficher_info_card(
            "Configuration de la detection de vol",
            "Ajustez les seuils de detection pour cacher article et dissimuler dans sac. "
            "Necessite un redemarrage du detecteur pour prendre effet.",
        )

        st.markdown("### Cacher article sous vetements")
        with st.form("form_vol_cacher"):
            c1, c2, c3 = st.columns(3)
            with c1:
                vol_dist = st.number_input("Distance main-corps (ratio)",
                    min_value=0.05, max_value=0.50, step=0.05,
                    value=float(_param(params, "vol_distance_main_corps", 0.25)),
                    help="Distance max main-corps normalisee par la hauteur de la bbox")
                vol_zone_h = st.number_input("Zone dissimulation haut (ratio)",
                    min_value=0.1, max_value=0.5, step=0.05,
                    value=float(_param(params, "vol_zone_dissimulation_haut", 0.3)),
                    help="Limite haute de la zone (0=haut bbox, 0.3=epaules)")
            with c2:
                vol_incr_corps = st.number_input("Increment main-corps",
                    min_value=0.01, max_value=0.30, step=0.01,
                    value=float(_param(params, "vol_increment_main_corps", 0.12)),
                    help="Score ajoute quand la main est dans la zone de dissimulation")
                vol_zone_b = st.number_input("Zone dissimulation bas (ratio)",
                    min_value=0.5, max_value=1.0, step=0.05,
                    value=float(_param(params, "vol_zone_dissimulation_bas", 0.85)),
                    help="Limite basse de la zone (0.85=genoux, 1=pieds)")
            with c3:
                vol_incr_mvt = st.number_input("Increment mouvement rentrant",
                    min_value=0.01, max_value=0.30, step=0.01,
                    value=float(_param(params, "vol_increment_mouvement_rentrant", 0.15)),
                    help="Score ajoute quand la main se rapproche du corps")
                vol_ratio = st.number_input("Ratio rapprochement",
                    min_value=0.2, max_value=0.9, step=0.1,
                    value=float(_param(params, "vol_ratio_rapprochement", 0.6)),
                    help="dist_fin/dist_debut < ratio = mouvement rentrant detecte")
            vol_incr_obj = st.number_input("Increment objet proche",
                min_value=0.01, max_value=0.30, step=0.01,
                value=float(_param(params, "vol_increment_objet_proche", 0.10)),
                help="Bonus quand un objet YOLO est detecte pres de la main")
            vol_hist = st.number_input("Historique min (frames)",
                min_value=2, max_value=30, step=1,
                value=int(_param(params, "vol_historique_min_frames", 5)),
                help="Nombre min d'observations pour analyse de mouvement")
            if st.form_submit_button("💾 Sauvegarder cacher article", type="primary"):
                db.definir_parametre("vol_distance_main_corps", str(vol_dist), "vol", type_valeur="float")
                db.definir_parametre("vol_zone_dissimulation_haut", str(vol_zone_h), "vol", type_valeur="float")
                db.definir_parametre("vol_zone_dissimulation_bas", str(vol_zone_b), "vol", type_valeur="float")
                db.definir_parametre("vol_increment_main_corps", str(vol_incr_corps), "vol", type_valeur="float")
                db.definir_parametre("vol_increment_mouvement_rentrant", str(vol_incr_mvt), "vol", type_valeur="float")
                db.definir_parametre("vol_increment_objet_proche", str(vol_incr_obj), "vol", type_valeur="float")
                db.definir_parametre("vol_ratio_rapprochement", str(vol_ratio), "vol", type_valeur="float")
                db.definir_parametre("vol_historique_min_frames", str(vol_hist), "vol", type_valeur="int")
                st.success("Parametres 'cacher article' sauvegardes")

        st.markdown("### Dissimuler dans un sac")
        with st.form("form_vol_sac"):
            c1, c2 = st.columns(2)
            with c1:
                sac_incr = st.number_input("Increment main-sac",
                    min_value=0.01, max_value=0.30, step=0.01,
                    value=float(_param(params, "vol_sac_increment_base", 0.10)),
                    help="Score ajoute quand la main est pres du sac")
                sac_dist = st.number_input("Distance main-sac (ratio taille sac)",
                    min_value=0.3, max_value=1.5, step=0.1,
                    value=float(_param(params, "vol_sac_distance_ratio", 0.8)),
                    help="Distance max main-sac normalisee par la taille du sac")
            with c2:
                sac_alt_incr = st.number_input("Increment alternance",
                    min_value=0.01, max_value=0.30, step=0.01,
                    value=float(_param(params, "vol_sac_increment_alternance", 0.15)),
                    help="Bonus pour mouvement repetitif main-etagere-sac")
                sac_alt_min = st.number_input("Alternances minimum",
                    min_value=1, max_value=10, step=1,
                    value=int(_param(params, "vol_sac_alternances_min", 2)),
                    help="Nombre min d'alternances pour activer le bonus")
            if st.form_submit_button("💾 Sauvegarder dissimuler sac", type="primary"):
                db.definir_parametre("vol_sac_increment_base", str(sac_incr), "vol", type_valeur="float")
                db.definir_parametre("vol_sac_distance_ratio", str(sac_dist), "vol", type_valeur="float")
                db.definir_parametre("vol_sac_increment_alternance", str(sac_alt_incr), "vol", type_valeur="float")
                db.definir_parametre("vol_sac_alternances_min", str(sac_alt_min), "vol", type_valeur="int")
                st.success("Parametres 'dissimuler sac' sauvegardes")

        st.markdown("### Score & Decay")
        with st.form("form_vol_score"):
            c1, c2 = st.columns(2)
            with c1:
                vol_decay = st.number_input("Taux decay",
                    min_value=0.80, max_value=0.99, step=0.01,
                    value=float(_param(params, "vol_decay_rate", 0.95)),
                    help="Decroissance du score par frame de reference (0.95 = 5%/frame)")
            with c2:
                vol_fps = st.number_input("FPS reference",
                    min_value=1.0, max_value=15.0, step=1.0,
                    value=float(_param(params, "vol_fps_ref", 2.0)),
                    help="FPS de normalisation (2=CCTV lent, 15=webcam rapide)")
            if st.form_submit_button("💾 Sauvegarder score", type="primary"):
                db.definir_parametre("vol_decay_rate", str(vol_decay), "vol", type_valeur="float")
                db.definir_parametre("vol_fps_ref", str(vol_fps), "vol", type_valeur="float")
                st.success("Parametres score sauvegardes")

    # === METIER ===
    with tab_metier:
        afficher_section("🏪", "Parametres metier")
        afficher_info_card(
            "Parametres commerciaux",
            "Valeur estimee des articles, horaires et retention des donnees.",
        )
        with st.form("form_metier"):
            c1, c2 = st.columns(2)
            with c1:
                val_article = st.number_input("Valeur article (DH)", min_value=1.0,
                    value=float(_param(params, "valeur_article_moyen_dh", 150.0)), step=10.0)
                h_ouv = st.text_input("Ouverture", value=_param(params, "heure_ouverture", "09:00"))
            with c2:
                h_ferm = st.text_input("Fermeture", value=_param(params, "heure_fermeture", "22:00"))
            if st.form_submit_button("💾 Sauvegarder", type="primary"):
                db.definir_parametre("valeur_article_moyen_dh", str(val_article), "metier", type_valeur="float")
                db.definir_parametre("heure_ouverture", h_ouv, "metier", type_valeur="str")
                db.definir_parametre("heure_fermeture", h_ferm, "metier", type_valeur="str")
                st.success("Parametres metier sauvegardes")

        # --- Retention differenciee ---
        afficher_section("🗄️", "Retention des donnees")
        afficher_info_card(
            "Retention differenciee",
            "Les videos (lourdes) et les alertes (legeres) ont des durees de conservation separees. "
            "Un quota disque protege contre la saturation.",
        )
        with st.form("form_retention"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ret_videos = st.number_input("Videos (jours)", min_value=1, max_value=90,
                    value=int(_param(params, "retention_videos_jours", 7)),
                    help="Clips video ~5-10 MB chacun")
                ret_snaps = st.number_input("Snapshots (jours)", min_value=1, max_value=90,
                    value=int(_param(params, "retention_snapshots_jours", 14)),
                    help="Images JPEG des alertes")
            with c2:
                ret_alertes = st.number_input("Alertes BDD (jours)", min_value=7, max_value=730,
                    value=int(_param(params, "retention_alertes_jours", 90)),
                    help="Enregistrements texte en base de donnees")
                ret_stats = st.number_input("Stats journalieres (jours)", min_value=30, max_value=1825,
                    value=int(_param(params, "retention_stats_jours", 365)),
                    help="Agregats quotidiens (tres leger)")
            with c3:
                quota_gb = st.number_input("Quota disque (GB)", min_value=5, max_value=500,
                    value=int(_param(params, "quota_stockage_max_gb", 50)),
                    help="Espace max pour videos + snapshots")
                quota_seuil = st.number_input("Seuil alerte (%)", min_value=50, max_value=99,
                    value=int(_param(params, "quota_seuil_alerte_pct", 90)),
                    help="Nettoyage d'urgence au-dela de ce seuil")
            if st.form_submit_button("💾 Sauvegarder retention", type="primary"):
                db.definir_parametre("retention_videos_jours", str(ret_videos), "retention", type_valeur="int")
                db.definir_parametre("retention_snapshots_jours", str(ret_snaps), "retention", type_valeur="int")
                db.definir_parametre("retention_alertes_jours", str(ret_alertes), "retention", type_valeur="int")
                db.definir_parametre("retention_stats_jours", str(ret_stats), "retention", type_valeur="int")
                db.definir_parametre("quota_stockage_max_gb", str(quota_gb), "retention", type_valeur="int")
                db.definir_parametre("quota_seuil_alerte_pct", str(quota_seuil), "retention", type_valeur="int")
                st.success("Parametres de retention sauvegardes")

    # === LIVE & CALIBRATION ===
    with tab_train:
        afficher_section("🎯", "Entrainement camera")
        afficher_info_card(
            "Entrainement par camera",
            "Detectez tous les objets visibles (80 classes COCO), puis assignez un role a chacun : "
            "mannequin a ignorer, zone d'exclusion, imprimante pour le suivi ticket, etc. "
            "Chaque camera apprend son environnement pour reduire les faux positifs.",
        )

        cameras_train = db.obtenir_cameras()
        if not cameras_train:
            st.warning("Aucune camera configuree. Ajoutez-en dans l'onglet Cameras.")
        else:
            cam_opts_train = {c["id"]: f"{c['nom']} ({c.get('niveau', 'Niveau 0')} / {c['zone']})" for c in cameras_train}
            train_cam_id = st.selectbox(
                "Camera", list(cam_opts_train.keys()),
                format_func=lambda x: cam_opts_train[x], key="train_cam_sel",
            )
            train_cam = next(c for c in cameras_train if c["id"] == train_cam_id)

            # ============================================================
            # A. BILAN CAMERA — resume des connaissances
            # ============================================================
            objets_ref = db.obtenir_objets_reference(train_cam_id)
            zones_excl = db.obtenir_zones_exclusion(train_cam_id, actives_seulement=False)

            nb_mannequins = sum(1 for o in objets_ref if (o.get("role") or "").lower() == "mannequin")
            nb_imprimante = sum(1 for o in objets_ref if (o.get("role") or "").lower() == "imprimante")
            nb_zones = len([z for z in zones_excl if z["actif"]])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mode", train_cam.get("mode_detection", "tout"))
            k2.metric("Mannequins", nb_mannequins)
            k3.metric("Zones exclusion", nb_zones)
            k4.metric("Imprimante", "Oui" if nb_imprimante else "Non")

            # Tableau combine objets + zones
            items_bilan = []
            for o in objets_ref:
                items_bilan.append({
                    "Type": "Objet ref.",
                    "Classe": o["classe"],
                    "Role": o.get("role") or "-",
                    "Info": o.get("comportement") or "-",
                })
            for z in zones_excl:
                items_bilan.append({
                    "Type": "Zone excl." if z["actif"] else "Zone excl. (off)",
                    "Classe": "-",
                    "Role": "zone_exclusion",
                    "Info": z["label"] or f"({z['pct_x1']*100:.0f}%,{z['pct_y1']*100:.0f}%) → ({z['pct_x2']*100:.0f}%,{z['pct_y2']*100:.0f}%)",
                })
            if items_bilan:
                st.dataframe(pd.DataFrame(items_bilan), use_container_width=True, hide_index=True)

                with st.expander("Gerer les elements"):
                    st.markdown("**Objets de reference :**")
                    for obj in objets_ref:
                        c1, c2 = st.columns([5, 1])
                        with c1:
                            st.text(f"{obj['classe']} → {obj.get('role', '-')} | {obj.get('comportement', '-')}")
                        with c2:
                            if st.button("Suppr.", key=f"del_ref_{obj['id']}"):
                                db.supprimer_objet_reference(obj["id"])
                                st.rerun()
                    if zones_excl:
                        st.markdown("**Zones d'exclusion :**")
                        for z in zones_excl:
                            c1, c2, c3 = st.columns([4, 1, 1])
                            with c1:
                                st.text(f"{z['label'] or '(sans label)'} — {z['source']} — "
                                        f"({z['pct_x1']*100:.0f}%,{z['pct_y1']*100:.0f}%)")
                            with c2:
                                lbl_btn = "Off" if z["actif"] else "On"
                                if st.button(lbl_btn, key=f"toggle_zone_{z['id']}"):
                                    db.modifier_zone_exclusion(z["id"], actif=0 if z["actif"] else 1)
                                    st.rerun()
                            with c3:
                                if st.button("Suppr.", key=f"del_zone_{z['id']}"):
                                    db.supprimer_zone_exclusion(z["id"])
                                    st.rerun()
            else:
                st.info("Cette camera n'a encore rien appris. Lancez une capture ci-dessous.")

            st.divider()

            # ============================================================
            # B. CAPTURER ET DETECTER — OIV7 ~600 classes (fallback COCO)
            # ============================================================
            afficher_section("📸", "Detecter et assigner les roles")
            st.caption("Capturez une frame pour voir TOUS les objets detectes "
                       "(~600 classes Open Images V7 : caisse, imprimante, scanner, "
                       "mannequin, etagere, etc. — fallback COCO si modele absent). "
                       "Assignez un role a chaque detection pour entrainer la camera.")

            col_cap, col_conf = st.columns([3, 1])
            with col_conf:
                train_conf = st.slider("Confiance min", 0.10, 1.0, 0.20, 0.05, key="train_conf")
            with col_cap:
                if st.button("📸 Capturer et detecter tout", type="primary", key="train_capture", use_container_width=True):
                    viewer = obtenir_viewer()
                    frame = viewer.capturer_frame(train_cam["source"])
                    if frame is not None:
                        st.session_state["train_frame"] = frame
                        st.session_state["train_cam_id"] = train_cam_id
                        resultat = viewer.analyser_frame(
                            frame, detecter_objets=True, estimer_pose=False,
                            confiance_min=train_conf, mode_tout_coco=True,
                        )
                        st.session_state["train_resultat"] = resultat
                    else:
                        st.error(f"Impossible de capturer depuis: {train_cam['source']}")

            if "train_frame" in st.session_state and st.session_state.get("train_cam_id") == train_cam_id:
                frame = st.session_state["train_frame"]
                resultat = st.session_state.get("train_resultat", {})
                tous_objets = resultat.get("personnes", []) + resultat.get("objets", [])

                # Annoter
                annotee = frame.copy()
                couleurs = [
                    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
                ]
                for i, det in enumerate(tous_objets):
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    couleur = couleurs[i % len(couleurs)]
                    cv2.rectangle(annotee, (x1, y1), (x2, y2), couleur, 2)
                    label = f"#{i+1} {det.class_name} ({det.confidence:.0%})"
                    cv2.putText(annotee, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)

                st.image(cv2.cvtColor(annotee, cv2.COLOR_BGR2RGB), use_container_width=True,
                         caption=f"{len(tous_objets)} objet(s) detecte(s) — {resultat.get('temps_inference_ms', 0)} ms")

                # Assigner les roles avec dropdown
                if tous_objets:
                    st.divider()
                    st.markdown("**Assigner un role a chaque detection :**")
                    ROLES_DISPONIBLES = [
                        "(ignorer)",
                        "zone_exclusion",
                        "mannequin",
                        "imprimante",
                        "scanner",
                        "terminal_paiement",
                        "caisse_enregistreuse",
                        "ecran",
                        "etagere",
                        "porte",
                        "autre",
                    ]
                    roles_choisis = {}
                    labels_custom = {}
                    for i, det in enumerate(tous_objets):
                        c1, c2, c3 = st.columns([2, 2, 2])
                        with c1:
                            st.text(f"#{i+1} {det.class_name} ({det.confidence:.0%})")
                        with c2:
                            roles_choisis[i] = st.selectbox(
                                "Role", ROLES_DISPONIBLES, key=f"train_role_{i}",
                                index=0,
                            )
                        with c3:
                            if roles_choisis[i] == "autre":
                                labels_custom[i] = st.text_input(
                                    "Role personnalise", key=f"train_custom_{i}",
                                    placeholder="etagere, caisse, porte...",
                                )
                            elif roles_choisis[i] not in ("(ignorer)", "zone_exclusion"):
                                st.caption(f"→ sera sauvegarde comme objet de reference")
                            elif roles_choisis[i] == "zone_exclusion":
                                st.caption(f"→ detections ignorees dans cette zone")

                    nb_a_sauver = sum(1 for r in roles_choisis.values() if r != "(ignorer)")
                    if nb_a_sauver > 0 and st.button(
                        f"💾 Sauvegarder {nb_a_sauver} element(s)", type="primary", key="train_save",
                    ):
                        h_f, w_f = frame.shape[:2]
                        nb_refs = 0
                        nb_zones = 0
                        for i, det in enumerate(tous_objets):
                            role = roles_choisis.get(i, "(ignorer)")
                            if role == "(ignorer)":
                                continue

                            x1, y1, x2, y2 = [int(v) for v in det.bbox]

                            if role == "zone_exclusion":
                                pct = (x1 / w_f, y1 / h_f, x2 / w_f, y2 / h_f)
                                db.ajouter_zone_exclusion(
                                    train_cam_id, det.class_name, pct, "manuel")
                                nb_zones += 1
                            else:
                                role_final = labels_custom.get(i, role) if role == "autre" else role
                                db.ajouter_objet_reference(
                                    camera_id=train_cam_id,
                                    classe=det.class_name,
                                    role=role_final,
                                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                                    confiance=det.confidence,
                                )
                                nb_refs += 1

                        msg = []
                        if nb_refs:
                            msg.append(f"{nb_refs} objet(s) de reference")
                        if nb_zones:
                            msg.append(f"{nb_zones} zone(s) d'exclusion")
                        st.success(f"Sauvegarde : {' + '.join(msg)}")
                        del st.session_state["train_frame"]
                        del st.session_state["train_resultat"]
                        st.rerun()

            st.divider()

            # ============================================================
            # C. APERCU VISUEL — overlay zones + objets
            # ============================================================
            if objets_ref or zones_excl:
                if st.button("👁️ Apercu de l'environnement camera", key="train_preview", use_container_width=True):
                    viewer = obtenir_viewer()
                    frame_prev = viewer.capturer_frame(train_cam["source"])
                    if frame_prev is not None:
                        h_p, w_p = frame_prev.shape[:2]
                        overlay = frame_prev.copy()
                        # Zones d'exclusion en rouge
                        for z in zones_excl:
                            if not z["actif"]:
                                continue
                            zx1 = int(z["pct_x1"] * w_p)
                            zy1 = int(z["pct_y1"] * h_p)
                            zx2 = int(z["pct_x2"] * w_p)
                            zy2 = int(z["pct_y2"] * h_p)
                            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 255), -1)
                            lbl = z.get("label", "")
                            if lbl:
                                cv2.putText(overlay, lbl, (zx1 + 4, zy1 + 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # Objets de reference en bleu/vert
                        for obj in objets_ref:
                            ox1 = int(obj["bbox_x1"])
                            oy1 = int(obj["bbox_y1"])
                            ox2 = int(obj["bbox_x2"])
                            oy2 = int(obj["bbox_y2"])
                            role = (obj.get("role") or "").lower()
                            color = (255, 165, 0) if role == "imprimante" else (255, 200, 0)
                            cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), color, -1)
                            cv2.putText(overlay, f"{role}", (ox1 + 4, oy1 + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        frame_preview = cv2.addWeighted(overlay, 0.3, frame_prev, 0.7, 0)
                        afficher_frame_bgr(frame_preview,
                            f"Rouge = zones exclusion | Orange/Jaune = objets de reference")
                    else:
                        st.warning("Impossible de capturer une frame.")

            st.divider()

            # ============================================================
            # D. ZONES D'EXCLUSION MANUELLES (coordonnees %)
            # ============================================================
            with st.expander("Ajouter une zone d'exclusion manuellement (coordonnees %)"):
                with st.form("form_zone_manuelle"):
                    label_zone = st.text_input("Label", placeholder="Mannequin vitrine gauche")
                    st.caption("0% = bord gauche/haut, 100% = bord droit/bas")
                    c1, c2 = st.columns(2)
                    with c1:
                        zx1 = st.number_input("X min (%)", 0.0, 100.0, 0.0, 1.0, key="zx1")
                        zy1 = st.number_input("Y min (%)", 0.0, 100.0, 0.0, 1.0, key="zy1")
                    with c2:
                        zx2 = st.number_input("X max (%)", 0.0, 100.0, 50.0, 1.0, key="zx2")
                        zy2 = st.number_input("Y max (%)", 0.0, 100.0, 50.0, 1.0, key="zy2")
                    if st.form_submit_button("Ajouter la zone", type="primary"):
                        if zx2 <= zx1 or zy2 <= zy1:
                            st.error("Coordonnees max doivent etre superieures aux min.")
                        else:
                            db.ajouter_zone_exclusion(
                                train_cam_id, label_zone,
                                (zx1 / 100, zy1 / 100, zx2 / 100, zy2 / 100), "manuel",
                            )
                            st.success(f"Zone '{label_zone}' ajoutee")
                            st.rerun()

            st.divider()

            # ============================================================
            # E. APPRENTISSAGE AUTOMATIQUE
            # ============================================================
            afficher_section("🧠", "Apprentissage automatique")
            st.caption("Le systeme observe la camera et identifie automatiquement les objets statiques.")

            session_active = db.obtenir_session_apprentissage_active(train_cam_id)
            if session_active:
                debut_str = session_active["debut"]
                try:
                    debut_dt = datetime.fromisoformat(debut_str)
                except (ValueError, TypeError):
                    debut_dt = datetime.now()
                elapsed = (datetime.now() - debut_dt).total_seconds()
                duree_total = session_active["duree_minutes"] * 60
                progress = min(elapsed / max(duree_total, 1), 1.0)
                st.progress(progress, text=f"Apprentissage en cours... {progress:.0%} "
                            f"({elapsed:.0f}s / {duree_total:.0f}s)")
                if st.button("Annuler l'apprentissage", key="train_cancel_learn"):
                    db.annuler_session_apprentissage(session_active["id"])
                    st.success("Session annulee.")
                    st.rerun()
                if progress >= 1.0:
                    st.info("Session terminee ! Validez les zones proposees ci-dessous.")

            # Propositions en attente
            with db._connexion() as _conn:
                sessions_cam = [dict(r) for r in _conn.execute(
                    """SELECT * FROM sessions_apprentissage
                    WHERE camera_id = ? AND statut = 'terminee'
                    ORDER BY debut DESC LIMIT 5""",
                    (train_cam_id,),
                ).fetchall()]

            for sess in sessions_cam:
                propositions = db.obtenir_zones_proposees(sess["id"])
                en_attente = [p for p in propositions if p["statut"] == "proposee"]
                if not en_attente:
                    continue
                st.markdown(f"**Session du {sess['debut']}** — {len(en_attente)} zone(s) a valider :")
                for prop in en_attente:
                    c1, c2, c3 = st.columns([4, 1, 1])
                    with c1:
                        st.text(
                            f"{prop['classe_detectee']} — "
                            f"immobile {prop['duree_observation_sec']:.0f}s — "
                            f"confiance {prop['confiance_moyenne']:.0%}"
                        )
                    with c2:
                        if st.button("Accepter", key=f"acc_{prop['id']}", type="primary"):
                            db.valider_zone_proposee(prop["id"], accepter=True)
                            st.rerun()
                    with c3:
                        if st.button("Rejeter", key=f"rej_{prop['id']}"):
                            db.valider_zone_proposee(prop["id"], accepter=False)
                            st.rerun()

            if not session_active:
                duree_default = float(db.obtenir_parametre("apprentissage_duree_minutes", 5.0))
                with st.form("form_apprentissage"):
                    duree_app = st.number_input(
                        "Duree d'observation (minutes)", 1.0, 30.0, duree_default, 1.0,
                    )
                    if st.form_submit_button("Demarrer l'apprentissage", type="primary"):
                        db.creer_session_apprentissage(train_cam_id, duree_app)
                        st.success(f"Apprentissage demarre pour {duree_app:.0f} min.")
                        st.rerun()

            st.divider()

            # ============================================================
            # F. FLUX LIVE MJPEG
            # ============================================================
            afficher_section("🎬", "Flux en direct")

            viewer = obtenir_viewer()
            mjpeg_actif = viewer._mjpeg_serveur is not None
            mjpeg_meme_cam = (viewer._mjpeg_source == train_cam["source"])

            col_stream_ctrl, col_stream_opts = st.columns([2, 2])
            with col_stream_opts:
                live_det = st.checkbox("Detections YOLO", value=True, key="live_det")
                live_pose = st.checkbox("Estimation pose", value=False, key="live_pose")
                live_vet = st.checkbox("Detection vetements", value=False, key="live_vet")
                live_conf = st.slider("Confiance", 0.10, 1.0, 0.30, 0.05, key="live_stream_conf")

            with col_stream_ctrl:
                if not mjpeg_actif:
                    if st.button("Demarrer le flux live", type="primary", key="start_mjpeg", use_container_width=True):
                        params = {
                            "detections": live_det, "pose": live_pose,
                            "vetements": live_vet, "confiance": live_conf,
                            "tout_coco": True, "objets_portes": False,
                        }
                        ok = viewer.demarrer_mjpeg(train_cam["source"], port=8555, params=params)
                        if ok:
                            st.session_state["mjpeg_actif"] = True
                            st.rerun()
                        else:
                            st.error("Impossible de demarrer le flux")
                else:
                    if st.button("Arreter le flux", key="stop_mjpeg", use_container_width=True):
                        viewer.arreter_mjpeg()
                        st.session_state.pop("mjpeg_actif", None)
                        st.rerun()

            if mjpeg_actif and mjpeg_meme_cam:
                st.markdown(
                    '<img src="http://localhost:8555/stream" '
                    'style="width:100%;border-radius:10px;border:2px solid #4fc3f7;" />',
                    unsafe_allow_html=True,
                )
                try:
                    resp = requests.get("http://localhost:8555/stats", timeout=2)
                    if resp.ok:
                        stats = resp.json()
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("FPS", f"{stats.get('fps', 0):.1f}")
                        sc2.metric("Inference", f"{stats.get('inference_ms', 0)} ms")
                        sc3.metric("Personnes", stats.get("personnes", 0))
                        sc4.metric("Objets", stats.get("objets", 0))
                except Exception as e:
                    logger.debug(f"Stats MJPEG indisponibles: {e}")
            elif mjpeg_actif and not mjpeg_meme_cam:
                st.warning("Le flux live est actif sur une autre camera. Arretez-le d'abord.")

    # === SYSTEME ===
    with tab_sys:
        afficher_section("🖥️", "Parametres systeme")
        with st.form("form_systeme"):
            c1, c2 = st.columns(2)
            with c1:
                dash_port = st.number_input("Port dashboard", min_value=1024, max_value=65535,
                    value=int(_param(params, "dashboard_port", 8502)))
                dash_refresh = st.number_input("Rafraichissement (s)", min_value=5, max_value=300,
                    value=int(_param(params, "dashboard_refresh_seconds", 10)))
            with c2:
                vid_path = st.text_input("Repertoire enregistrements", value=_param(params, "video_save_path", "./recordings"))
                model_path = st.text_input("Repertoire modeles", value=_param(params, "model_path", "./models"))
            db_path = st.text_input("Chemin BDD", value=_param(params, "database_path", "./data/fraude.db"))
            if st.form_submit_button("💾 Sauvegarder", type="primary"):
                db.definir_parametre("dashboard_port", str(dash_port), "systeme", type_valeur="int")
                db.definir_parametre("dashboard_refresh_seconds", str(dash_refresh), "systeme", type_valeur="int")
                db.definir_parametre("video_save_path", vid_path, "systeme", type_valeur="str")
                db.definir_parametre("model_path", model_path, "systeme", type_valeur="str")
                db.definir_parametre("database_path", db_path, "systeme", type_valeur="str")
                st.success("Parametres systeme sauvegardes.")

        # --- Monitoring espace disque ---
        afficher_section("💾", "Espace disque")
        vid_dir = Path(_param(params, "video_save_path", "./recordings"))
        snap_dir = vid_dir / "snapshots"
        quota_max_gb = int(_param(params, "quota_stockage_max_gb", 50))
        seuil_pct = int(_param(params, "quota_seuil_alerte_pct", 90))

        # Calcul taille fichiers
        videos_mb = 0.0
        snapshots_mb = 0.0
        nb_videos = 0
        nb_snaps = 0
        if vid_dir.exists():
            for f in vid_dir.iterdir():
                if f.is_file() and f.suffix in (".mp4", ".avi"):
                    videos_mb += f.stat().st_size / (1024 * 1024)
                    nb_videos += 1
        if snap_dir.exists():
            for dossier in snap_dir.iterdir():
                if dossier.is_dir():
                    for f in dossier.iterdir():
                        if f.is_file():
                            snapshots_mb += f.stat().st_size / (1024 * 1024)
                            nb_snaps += 1

        # Taille BDD
        db_file = Path(_param(params, "database_path", "./data/fraude.db"))
        db_mb = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0.0

        total_mb = videos_mb + snapshots_mb + db_mb
        quota_mb = quota_max_gb * 1024
        usage_pct = (total_mb / quota_mb * 100) if quota_mb > 0 else 0

        # Barre de progression coloree
        if usage_pct >= 95:
            couleur = "rouge"
        elif usage_pct >= seuil_pct:
            couleur = "orange"
        else:
            couleur = "vert"
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Videos", f"{videos_mb:.0f} MB", f"{nb_videos} clips")
        with c2:
            st.metric("Snapshots", f"{snapshots_mb:.0f} MB", f"{nb_snaps} images")
        with c3:
            st.metric("Base de donnees", f"{db_mb:.1f} MB")
        with c4:
            st.metric("Total / Quota", f"{total_mb:.0f} / {quota_mb:.0f} MB", f"{usage_pct:.0f}%")
        st.progress(min(usage_pct / 100, 1.0))
        if couleur == "rouge":
            st.error(f"Stockage critique ({usage_pct:.0f}%) — nettoyage d'urgence actif (retention reduite a 3j)")
        elif couleur == "orange":
            st.warning(f"Stockage eleve ({usage_pct:.0f}%) — le nettoyage d'urgence se declenchera bientot")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reinitialiser les parametres par defaut", use_container_width=True):
            db.reinitialiser_parametres()
            st.success("Parametres reinitialises")
            st.rerun()

        afficher_section("📋", "Tous les parametres")
        tous = db.obtenir_tous_parametres()
        if tous:
            df_params = pd.DataFrame(tous)
            st.dataframe(df_params[["cle", "valeur", "categorie", "type_valeur", "mis_a_jour"]],
                         use_container_width=True, hide_index=True)


# ============================================================================
# PAGE HISTORIQUE
# ============================================================================

def page_historique():
    db = obtenir_db()

    afficher_header(
        "📋", "Historique des alertes",
        "Alertes de production avec gestion complete",
        "HISTORIQUE", "badge-admin",
    )

    _tab_historique_production(db)


def _tab_historique_production(db):
    """Historique et CRUD des alertes de production."""
    afficher_section("🚨", "Alertes de production")

    # KPIs
    total = db.compter_alertes()
    aujourdhui = db.compter_alertes_aujourdhui()
    semaine = db.compter_alertes(jours=7)
    mois = db.compter_alertes(jours=30)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card kpi-bleu">
            <div class="kpi-icon">📊</div>
            <div class="kpi-value">{total}</div>
            <div class="kpi-label">Total alertes</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card kpi-rouge">
            <div class="kpi-icon">📅</div>
            <div class="kpi-value">{aujourdhui}</div>
            <div class="kpi-label">Aujourd'hui</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card kpi-orange">
            <div class="kpi-icon">📆</div>
            <div class="kpi-value">{semaine}</div>
            <div class="kpi-label">7 derniers jours</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card kpi-vert">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value">{mois}</div>
            <div class="kpi-label">30 derniers jours</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Filtres
    col_pf1, col_pf2, col_pf3 = st.columns([1, 1, 1])
    with col_pf1:
        date_deb_prod = st.date_input("Du", value=datetime.now() - timedelta(days=7), key="prod_date_deb")
    with col_pf2:
        date_fin_prod = st.date_input("Au", value=datetime.now(), key="prod_date_fin")
    with col_pf3:
        limite_prod = st.number_input("Max resultats", 10, 500, 50, key="prod_limite")

    alertes = db.obtenir_alertes_par_date(
        date_deb_prod.strftime("%Y-%m-%d"),
        date_fin_prod.strftime("%Y-%m-%d"),
    )
    alertes = alertes[:limite_prod]

    if not alertes:
        st.info("Aucune alerte de production sur cette periode.")
        return

    # Tableau interactif avec selection
    df_prod = pd.DataFrame([{
        "ID": a["id"],
        "Date": a["horodatage"][:19] if a["horodatage"] else "",
        "Comportement": a["type_comportement"],
        "Confiance": f"{a['confiance']:.0%}",
        "Piste": a["id_piste"],
        "Zone": a["zone"] or "",
        "Camera": a["source_camera"] or "",
        "Telegram": "✅" if a["notifie_telegram"] else "",
        "Note": (a["commentaire"] or "")[:30],
    } for a in alertes])

    selection_prod = st.dataframe(
        df_prod, use_container_width=True, hide_index=True,
        selection_mode="single-row", on_select="rerun",
        key="hist_prod_table",
    )

    selected_prod_rows = selection_prod.selection.rows if selection_prod and selection_prod.selection else []
    choix_alerte = alertes[selected_prod_rows[0]]["id"] if selected_prod_rows else None

    if choix_alerte:
        col_bv, col_bd, col_bsp = st.columns([1, 1, 2])
        with col_bv:
            btn_voir_al = st.button("🔍 Visualiser l'alerte", key="btn_voir_alerte",
                                     type="primary", use_container_width=True)
        with col_bd:
            btn_suppr_al = st.button("🗑️ Supprimer", key="btn_suppr_alerte", use_container_width=True)

        if btn_suppr_al:
            db.supprimer_alerte(choix_alerte)
            st.success(f"Alerte #{choix_alerte} supprimee")
            st.rerun()

        if btn_voir_al or st.session_state.get("hist_detail_alerte") == choix_alerte:
            st.session_state["hist_detail_alerte"] = choix_alerte
            _afficher_detail_alerte_prod(db, choix_alerte)
    else:
        st.info("Selectionnez une ligne du tableau pour visualiser les details.")

    # Repartition par type
    st.divider()
    with st.expander("📊 Repartition par type"):
        repartition = db.obtenir_repartition_comportements(jours=30)
        if repartition:
            df_rep = pd.DataFrame(repartition)
            fig = px.bar(df_rep, x="type_comportement", y="nombre",
                         color="type_comportement", title="Alertes par type de comportement")
            appliquer_theme_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)


def _afficher_detail_alerte_prod(db, choix_alerte):
    """Affiche le detail complet d'une alerte production avec mode replay."""
    alerte = db.obtenir_alerte(choix_alerte)
    if not alerte:
        st.warning("Alerte introuvable.")
        return

    st.divider()
    afficher_section("🔍", f"Alerte #{choix_alerte}")

    col_ai, col_as = st.columns([1, 1])
    with col_ai:
        st.markdown(f"""
        | Champ | Valeur |
        |-------|--------|
        | **ID** | {alerte['id']} |
        | **Date** | {alerte['horodatage']} |
        | **Type** | {alerte['type_comportement']} |
        | **Confiance** | {alerte['confiance']:.2f} |
        | **Piste** | {alerte['id_piste']} |
        | **Zone** | {alerte['zone'] or 'N/A'} |
        | **Camera** | {alerte['source_camera'] or 'N/A'} |
        | **Bbox** | ({alerte['bbox_x1']}, {alerte['bbox_y1']}) - ({alerte['bbox_x2']}, {alerte['bbox_y2']}) |
        | **Telegram** | {'Oui' if alerte['notifie_telegram'] else 'Non'} |
        """)

    with col_as:
        snap = alerte.get("chemin_snapshot", "")
        if snap and Path(snap).exists():
            img = cv2.imread(snap)
            if img is not None:
                afficher_frame_bgr(img, caption="Snapshot de l'alerte")
        else:
            st.info("Snapshot non disponible")

    # ---- Mode Replay Video ----
    chemin_video = alerte.get("chemin_video", "")
    if chemin_video:
        _afficher_replay_video(chemin_video, alerte)

    # Modifier commentaire
    with st.form(f"edit_alerte_{choix_alerte}"):
        new_comment = st.text_area("Commentaire",
                                   value=alerte.get("commentaire", "") or "",
                                   key=f"edit_al_note_{choix_alerte}")
        if st.form_submit_button("💾 Sauvegarder le commentaire", type="primary"):
            db.modifier_alerte(choix_alerte, commentaire=new_comment)
            st.success("Commentaire mis a jour")
            st.rerun()


def _resoudre_chemin_video(chemin: str) -> Optional[str]:
    """
    Resout le chemin video en essayant plusieurs variantes.
    Gere les chemins absolus, relatifs et les chemins conteneur vs host.
    """
    if not chemin:
        return None

    # Essayer tel quel
    p = Path(chemin)
    if p.exists():
        return str(p)

    # Essayer avec extensions alternatives (.mp4 <-> .avi)
    for ext in (".mp4", ".avi"):
        alt = p.with_suffix(ext)
        if alt.exists():
            return str(alt)

    # Chemins possibles: conteneur (/opt/fraude/recordings/) -> host
    # Essayer depuis le repertoire de travail
    nom = p.name
    config = obtenir_config()
    for dossier in [config.chemin_enregistrements, Path("./recordings")]:
        if dossier.exists():
            # Chercher dans les sous-repertoires aussi
            for f in dossier.rglob(nom):
                if f.is_file():
                    return str(f)
            # Essayer avec extensions alternatives
            for ext in (".mp4", ".avi"):
                nom_alt = p.stem + ext
                for f in dossier.rglob(nom_alt):
                    if f.is_file():
                        return str(f)

    return None


def _afficher_replay_video(chemin_video: str, alerte: dict):
    """
    Affiche le mode replay pour un clip video d'alerte.
    Permet de naviguer frame par frame avec un slider,
    et de relancer une analyse YOLO sur une frame specifique.
    """
    st.divider()
    afficher_section("🎬", "Replay video")

    chemin_resolu = _resoudre_chemin_video(chemin_video)

    if not chemin_resolu:
        st.warning(
            f"Video non disponible: `{Path(chemin_video).name}`\n\n"
            "Le fichier a peut-etre ete supprime par la retention automatique, "
            "ou le chemin conteneur n'est pas accessible depuis le dashboard."
        )
        return

    # Ouvrir la video et recuperer les infos
    cap = cv2.VideoCapture(chemin_resolu)
    if not cap.isOpened():
        st.error(f"Impossible d'ouvrir la video: {chemin_resolu}")
        return

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 15
        duree_sec = total_frames / max(fps_video, 1)
        largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Infos video
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1:
            st.metric("Frames", total_frames)
        with col_v2:
            st.metric("FPS", f"{fps_video:.0f}")
        with col_v3:
            st.metric("Duree", f"{duree_sec:.1f}s")
        with col_v4:
            st.metric("Resolution", f"{largeur}x{hauteur}")

        if total_frames <= 0:
            st.warning("Video vide (0 frames)")
            return

        # Slider de navigation
        frame_idx = st.slider(
            "Frame",
            min_value=0,
            max_value=max(total_frames - 1, 0),
            value=min(int(total_frames * 0.3), total_frames - 1),
            key=f"replay_slider_{alerte['id']}",
            help="Naviguez dans le clip video frame par frame",
        )

        # Lire la frame selectionnee
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            st.error(f"Impossible de lire la frame #{frame_idx}")
            return

        timestamp_sec = frame_idx / max(fps_video, 1)

        col_frame, col_analyse = st.columns([2, 1])

        with col_frame:
            # Afficher la frame
            afficher_frame_bgr(
                frame,
                caption=f"Frame {frame_idx}/{total_frames} — {timestamp_sec:.1f}s",
            )

        with col_analyse:
            st.markdown("**Navigation rapide**")
            col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
            with col_nav1:
                if st.button("⏮", key=f"nav_start_{alerte['id']}", help="Debut"):
                    st.session_state[f"replay_slider_{alerte['id']}"] = 0
                    st.rerun()
            with col_nav2:
                if st.button("◀ -1s", key=f"nav_back_{alerte['id']}"):
                    new_idx = max(0, frame_idx - int(fps_video))
                    st.session_state[f"replay_slider_{alerte['id']}"] = new_idx
                    st.rerun()
            with col_nav3:
                if st.button("+1s ▶", key=f"nav_fwd_{alerte['id']}"):
                    new_idx = min(total_frames - 1, frame_idx + int(fps_video))
                    st.session_state[f"replay_slider_{alerte['id']}"] = new_idx
                    st.rerun()
            with col_nav4:
                if st.button("⏭", key=f"nav_end_{alerte['id']}", help="Fin"):
                    st.session_state[f"replay_slider_{alerte['id']}"] = total_frames - 1
                    st.rerun()

            # Bouton de re-analyse YOLO
            st.divider()
            st.markdown("**Re-analyse YOLO**")
            confiance_replay = st.slider(
                "Confiance", 0.1, 1.0, 0.45, 0.05,
                key=f"replay_conf_{alerte['id']}",
            )
            if st.button(
                "🔄 Analyser cette frame",
                key=f"btn_reanalyse_{alerte['id']}",
                type="primary",
                use_container_width=True,
            ):
                _reanalyser_frame(frame, confiance_replay, alerte)

    finally:
        cap.release()


def _reanalyser_frame(frame: np.ndarray, confiance: float, alerte: dict):
    """
    Relance une detection YOLO + pose sur la frame selectionnee.
    Affiche les resultats dans le dashboard.
    """
    from app.detector import DetecteurPersonnes, EstimateurPose

    config = obtenir_config()
    chemin_yolo = config.chemin_modele_yolo
    chemin_pose = config.chemin_modele_pose

    if not chemin_yolo.exists():
        st.error("Modele YOLO non disponible pour la re-analyse.")
        return

    with st.spinner("Analyse en cours..."):
        t0 = time.time()

        try:
            detecteur = DetecteurPersonnes(chemin_yolo, confiance_min=confiance)
            personnes, objets = detecteur.detecter_personnes_et_objets(frame)
        except Exception as e:
            st.error(f"Erreur detection: {e}")
            return

        estimateur = None
        if chemin_pose.exists():
            try:
                estimateur = EstimateurPose(chemin_pose, confiance_min=config.pose_confidence)
            except Exception as e:
                logger.warning(f"Pose estimation non disponible pour re-analyse: {e}")

        poses = {}
        if estimateur and personnes:
            bboxes = [p.bbox for p in personnes[:5]]
            try:
                resultats_pose = estimateur.estimer_poses_multiples(frame, bboxes)
                for i, res in enumerate(resultats_pose):
                    if res is not None:
                        poses[i] = res
            except Exception as e:
                logger.warning(f"Erreur estimation poses re-analyse: {e}")

        duree_ms = (time.time() - t0) * 1000

    # Afficher les resultats
    st.success(f"Analyse terminee en {duree_ms:.0f} ms")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Personnes", len(personnes))
    with c2:
        st.metric("Objets", len(objets))
    with c3:
        st.metric("Poses", len(poses))

    # Dessiner les detections sur la frame
    frame_annotee = frame.copy()
    for i, det in enumerate(personnes):
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame_annotee, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"P{i} ({det.confidence:.0%})"
        cv2.putText(frame_annotee, label, (x1, y1 - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Pose si dispo
        pose = poses.get(i)
        if pose is not None:
            for j in range(17):
                kp = pose.keypoints[j]
                if kp[2] > 0.3:
                    cv2.circle(frame_annotee, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

    for obj in objets:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame_annotee, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(frame_annotee, obj.class_name, (x1, y2 + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    afficher_frame_bgr(frame_annotee, caption="Re-analyse YOLO")


# ============================================================================
# MAIN
# ============================================================================

def main():
    page = afficher_sidebar()
    if page == "⚙️ Administration":
        page_administration()
    elif page == "📋 Historique":
        page_historique()
    else:
        page_dashboard()


main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generation de graphes dynamiques pour l'analyse XGBoost.

Graphes generes:
1. graph_erreur_rate_contribution: histogramme combiné avec ligne de taux d'erreur
2. graphe_radar_comparaison: radar comparatif des 3 plans d'hebergement
3. courbe_projection_saturation: courbe de projection temporelle
4. barres_impact_features: barres horizontales d'impact des parametres
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Patch, Rectangle, Wedge

try:
    from predict_from_file import predict as predict_full_result
except ImportError:
    predict_full_result = None


# ============================================
# CONFIGURATION
# ============================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

BASE_DIR = Path(__file__).resolve().parent

if os.path.exists("/app"):
    MODELS_DIR = Path("/app/service/models")
    DATA_DIR = Path("/app/Donnee_parametres")
    MODEL_PATH = Path("/app/service/models/model.pkl")
else:
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR.parent / "Donnee_parametres"
    MODEL_PATH = MODELS_DIR / "model.pkl"

OUTPUT_DIR = BASE_DIR / "analysis_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SATURATION_LIMIT = 90.0

HEAVY_PLUGIN_OPTIONS = [
    "woocommerce",
    "elementor",
    "wpml",
    "yoast",
    "revslider",
    "gravityforms",
]

PHP_VERSIONS = ["7.4", "8.0", "8.1", "8.2", "8.3"]
WP_TYPES = ["small", "medium", "performance"]
PLAN_LABELS = ["small", "medium", "performance"]

PLAN_TO_LOAD_SCORE = {
    "small": 17.5,
    "medium": 50.0,
    "performance": 82.5,
}


# ============================================
# OUTILS GÉNÉRAUX
# ============================================

def clamp(value: Any, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Contraint une valeur numerique entre un minimum et un maximum."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def to_float(value: Any, default: float = 0.0) -> float:
    """Convertit une valeur en float, retourne default si conversion impossible."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_select(value: Any, default: str | None = None) -> str | None:
    """Nettoie une valeur de selection (ex: oui/non)."""
    if value is None:
        return default
    value = str(value).strip()
    if value == "" or value.lower() == "none":
        return default
    return value.lower()


def time_to_minutes(value: Any) -> int:
    """Convertit une chaine HH:MM en minutes depuis minuit."""
    if value is None:
        return 0
    if isinstance(value, str):
        try:
            hour, minute = value.split(":")[:2]
            return int(hour) * 60 + int(minute)
        except (ValueError, AttributeError):
            return 0
    return 0


def normalize_heavy_plugins(value: Any) -> list[str]:
    """Transforme une liste ou une chaine de plugins lourds en liste normalisee."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    return []


def color_for_score(value: float) -> str:
    """Retourne une couleur hexadecimale selon le score."""
    value = clamp(value)
    if value >= 85:
        return "#b91c1c"
    if value >= 70:
        return "#ef4444"
    if value >= 55:
        return "#f97316"
    if value >= 35:
        return "#f59e0b"
    return "#16a34a"


def status_for_score(value: float) -> str:
    """Retourne un statut textuel selon le score."""
    value = clamp(value)
    if value >= 85:
        return "CRITIQUE"
    if value >= 70:
        return "ELEVE"
    if value >= 55:
        return "SURVEILLANCE"
    return "STABLE"


def clean_status(status: Any, fallback_score: float) -> str:
    """Nettoie un statut, retourne le mot-cle utile ou le statut calcule."""
    if status:
        text = str(status).strip()
        for value in ["CRITIQUE", "URGENT", "SURVEILLANCE", "OPTIMAL", "STABLE"]:
            if value in text.upper():
                return value
    return status_for_score(fallback_score)


def prediction_field(prediction: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """Extrait une cle d'un dictionnaire de prediction."""
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return default


def fmt_int(value: float) -> str:
    """Formate un entier avec des espaces pour les milliers."""
    return f"{int(round(value)):,}".replace(",", " ")


def save_figure(fig: plt.Figure, filename: str) -> str:
    """Sauvegarde une figure matplotlib dans OUTPUT_DIR avec timestamp."""
    path = str(OUTPUT_DIR / f"{filename}_{TIMESTAMP}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ============================================
# CHARGEMENT MODELE + PARAMETRES
# ============================================

def load_model():
    """Charge le modele XGBoost depuis le fichier pickle."""
    if not MODEL_PATH.exists():
        return None, None, None

    payload = joblib.load(str(MODEL_PATH))
    if isinstance(payload, dict):
        return payload.get("model"), payload.get("feature_columns"), payload
    return payload, None, {}


def find_latest_json():
    """Trouve le fichier JSON le plus recent dans DATA_DIR."""
    if not DATA_DIR.exists():
        return None
    files = sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_params(filepath):
    """Charge les parametres depuis un fichier JSON."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("parameters", data.get("params", data))


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalise les parametres bruts en valeurs utilisables."""
    return {
        "visitors_per_day": to_float(params.get("visitors_per_day")),
        "pageviews_per_day": to_float(params.get("pageviews_per_day")),
        "traffic_growth_rate": to_float(params.get("traffic_growth_rate")),
        "peak_hours_start": clean_select(params.get("peak_hours_start")),
        "peak_hours_end": clean_select(params.get("peak_hours_end")),
        "cpu_usage_avg": to_float(params.get("cpu_usage_avg")),
        "cpu_usage_peak": to_float(params.get("cpu_usage_peak")),
        "ram_usage_avg": to_float(params.get("ram_usage_avg")),
        "ram_usage_max": to_float(params.get("ram_usage_max")),
        "disk_usage_avg": to_float(params.get("disk_usage_avg")),
        "disk_usage_max": to_float(params.get("disk_usage_max")),
        "disk_gb": to_float(params.get("disk_gb"), 0.0),
        "response_time": to_float(params.get("response_time")),
        "disk_read_iops": to_float(params.get("disk_read_iops")),
        "disk_write_iops": to_float(params.get("disk_write_iops")),
        "plugin_count": to_float(params.get("plugin_count")),
        "heavy_plugins": normalize_heavy_plugins(params.get("heavy_plugins")),
        "php_version": clean_select(params.get("php_version")),
        "cache_enabled": clean_select(params.get("cache_enabled")),
        "cdn_enabled": clean_select(params.get("cdn_enabled")),
        "wp_type": clean_select(params.get("wp_type")),
    }


def prepare_features(params: dict[str, Any], feature_columns: list[str]):
    """Prepare les features pour le modele XGBoost."""
    normalized = normalize_params(params)
    heavy_plugins = set(normalized["heavy_plugins"])

    start_minutes = time_to_minutes(normalized.get("peak_hours_start"))
    end_minutes = time_to_minutes(normalized.get("peak_hours_end"))

    row = {
        "visitors_per_day": normalized["visitors_per_day"],
        "pageviews_per_day": normalized["pageviews_per_day"],
        "traffic_growth_rate": normalized["traffic_growth_rate"],
        "cpu_usage_avg": normalized["cpu_usage_avg"],
        "cpu_usage_peak": normalized["cpu_usage_peak"],
        "ram_usage_avg": normalized["ram_usage_avg"],
        "ram_usage_max": normalized["ram_usage_max"],
        "disk_usage_avg": normalized["disk_usage_avg"],
        "disk_usage_max": normalized["disk_usage_max"],
        "response_time": normalized["response_time"],
        "disk_read_iops": normalized["disk_read_iops"],
        "disk_write_iops": normalized["disk_write_iops"],
        "plugin_count": normalized["plugin_count"],
        "peak_hours_start_minutes": start_minutes,
        "peak_hours_end_minutes": end_minutes,
        "heavy_plugins_sum": float(len(heavy_plugins)),
        "wp_facteur": (
            normalized["visitors_per_day"] * 0.0001
            + normalized["plugin_count"] * 0.5
            + len(heavy_plugins) * 2
        ),
    }

    for plugin in HEAVY_PLUGIN_OPTIONS:
        row[f"heavy_plugin_{plugin}"] = 1 if plugin in heavy_plugins else 0

    for version in PHP_VERSIONS:
        row[f"php_{version}"] = 1 if normalized["php_version"] == version else 0

    for value in ["non", "oui"]:
        row[f"cache_{value}"] = 1 if normalized["cache_enabled"] == value else 0
        row[f"cdn_{value}"] = 1 if normalized["cdn_enabled"] == value else 0

    for wp_type in WP_TYPES:
        row[f"wp_{wp_type}"] = 1 if normalized["wp_type"] == wp_type else 0

    row["peak_duration_minutes"] = max(0, end_minutes - start_minutes)

    features = pd.DataFrame([{column: row.get(column, 0) for column in feature_columns}])
    features = features.astype(float)
    return features, normalized


def predict_load(model_load, features_df, normalized, metadata=None):
    """Predit la charge serveur avec le modele XGBoost."""
    try:
        is_classifier = hasattr(model_load, "predict_proba")

        if is_classifier:
            predicted_class = int(model_load.predict(features_df)[0])
            probabilities = model_load.predict_proba(features_df)[0]
            plan_labels = metadata.get("plan_labels", PLAN_LABELS) if metadata else PLAN_LABELS

            if len(probabilities) == 3:
                load_scores = [PLAN_TO_LOAD_SCORE.get(plan, 50.0) for plan in plan_labels]
                predicted_load = sum(prob * score for prob, score in zip(probabilities, load_scores))
            else:
                predicted_plan = plan_labels[predicted_class] if predicted_class < len(plan_labels) else "medium"
                predicted_load = PLAN_TO_LOAD_SCORE.get(predicted_plan, 50.0)

            predicted_load = round(clamp(predicted_load), 2)

            wp_type = (normalized.get("wp_type") or "").lower()
            if wp_type == "small":
                predicted_load = min(100.0, predicted_load * 1.25)
            elif wp_type == "medium":
                predicted_load = min(100.0, predicted_load * 1.10)

            return round(predicted_load, 2)

        predicted_load = float(model_load.predict(features_df)[0])
        return round(clamp(predicted_load), 2)
    except Exception as exc:
        print(f"[ERREUR] Prediction: {exc}", file=sys.stderr)
        return 50.0


# ============================================
# SCORES DERIVES DES PARAMETRES
# ============================================

def derived_scores(
    normalized: dict[str, Any],
    current_load: float,
    prediction: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Calcule les scores derives pour chaque categorie."""
    visitors = normalized["visitors_per_day"]
    pageviews = normalized["pageviews_per_day"]
    growth = normalized["traffic_growth_rate"]
    heavy_count = len(normalized["heavy_plugins"])

    traffic = clamp(
        (visitors / 150000) * 55
        + (pageviews / 450000) * 25
        + (growth / 85) * 20
    )
    cpu = clamp(normalized["cpu_usage_avg"] * 0.45 + normalized["cpu_usage_peak"] * 0.55)
    ram = clamp(normalized["ram_usage_avg"] * 0.45 + normalized["ram_usage_max"] * 0.55)
    disk = clamp(
        normalized["disk_usage_avg"] * 0.35
        + normalized["disk_usage_max"] * 0.4
        + (normalized["disk_read_iops"] / 6000) * 12
        + (normalized["disk_write_iops"] / 5000) * 13
    )
    app = clamp(
        (normalized["plugin_count"] / 55) * 48
        + heavy_count * 9
        + min(normalized["response_time"] / 55, 35)
    )

    optimization_gain = 0
    if normalized["cache_enabled"] == "oui":
        optimization_gain += 18
    if normalized["cdn_enabled"] == "oui":
        optimization_gain += 14
    if normalized["php_version"] in ["8.2", "8.3"]:
        optimization_gain += 12
    elif normalized["php_version"] in ["8.0", "8.1"]:
        optimization_gain += 6
    if normalized["wp_type"] == "performance":
        optimization_gain += 10
    elif normalized["wp_type"] == "small":
        optimization_gain -= 8

    optimization_risk = clamp(74 - optimization_gain)
    response = clamp(normalized["response_time"] / 20)
    error_rate = clamp(prediction_field(prediction, "error_rate", 0.0))
    global_risk = clamp(
        current_load * 0.42
        + traffic * 0.16
        + cpu * 0.13
        + ram * 0.12
        + app * 0.10
        + optimization_risk * 0.05
        + error_rate * 0.02
    )

    return {
        "Trafic": traffic,
        "CPU": cpu,
        "RAM": ram,
        "Disque": disk,
        "Application": app,
        "Optimisation": optimization_risk,
        "Reponse": response,
        "Erreur": error_rate,
        "Global": global_risk,
    }


def optimization_gain_points(normalized: dict[str, Any]) -> float:
    """Calcule le gain d'optimisation potentiel."""
    gain = 0.0
    if normalized["cache_enabled"] != "oui":
        gain += 10
    if normalized["cdn_enabled"] != "oui":
        gain += 7
    if normalized["php_version"] not in ["8.2", "8.3"]:
        gain += 6
    if normalized["wp_type"] != "performance":
        gain += 5
    if normalized["plugin_count"] > 20:
        gain += min(8, (normalized["plugin_count"] - 20) * 0.35)
    return gain


def load_projection(current_load: float, monthly_growth: float, months: np.ndarray) -> np.ndarray:
    """Projette la charge serveur sur plusieurs mois."""
    growth = max(0.0, monthly_growth) / 100
    return np.clip(current_load * ((1 + growth) ** months), 0, 100)


def first_saturation_month(values: np.ndarray) -> int | None:
    """Trouve le premier mois ou la charge atteint la saturation."""
    hits = np.where(values >= SATURATION_LIMIT)[0]
    return int(hits[0]) if len(hits) else None


# ============================================
# GRAPHE 1: HISTOGRAMME TAUX D'ERREUR
# ============================================

def graph_erreur_rate_contribution(
    normalized: dict[str, Any],
    current_load: float,
    prediction: dict[str, Any] | None = None,
) -> str | None:
    """
    Histogramme montrant la contribution de chaque parametre au taux d'erreur,
    avec une ligne cumulative montrant l'evolution jusqu'a 100%.
    
    Parametres analyses :
    - plugin_count : contribution de 1.2% par plugin
    - ram_usage_max : contribution de 0.25% par % de RAM
    - response_time : contribution de 0.4% par ms de temps de reponse
    
    Le taux d'erreur final est calcule comme :
    error_rate = min(100, (plugin_count * 1.2) + (ram_usage_max / 4) + (response_time / 250))
    """
    try:
        plugin_count = float(normalized.get("plugin_count", 0) or 0)
        ram_usage_max = float(normalized.get("ram_usage_max", 0) or 0)
        response_time = float(normalized.get("response_time", 0) or 0)
        
        contribution_plugins = plugin_count * 1.2
        contribution_ram = ram_usage_max / 4
        contribution_response = response_time / 250
        
        error_rate = min(100, contribution_plugins + contribution_ram + contribution_response)
        error_rate = round(error_rate, 2)
        
        status = clean_status(prediction_field(prediction, "status"), current_load)
        
        categories = [
            f"Plugins\n({plugin_count:.0f} plugins)",
            f"RAM max\n({ram_usage_max:.0f}%)",
            f"Temps reponse\n({response_time:.0f} ms)",
        ]
        contributions = [
            contribution_plugins,
            contribution_ram,
            contribution_response,
        ]
        
        cumulative = np.cumsum(contributions)
        cumulative = np.minimum(cumulative, 100)
        
        colors_barres = ["#3b82f6", "#f59e0b", "#ef4444"]
        
        fig = plt.figure(figsize=(16, 9))
        
        # SOUS-GRAPHIQUE 1 : Histogramme
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        
        x = np.arange(len(categories))
        width = 0.6
        
        bars = ax1.bar(x, contributions, width, color=colors_barres, alpha=0.85, 
                       edgecolor="white", linewidth=2)
        
        for i, (bar, val) in enumerate(zip(bars, contributions)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f"{val:.1f}%",
                    ha='center', va='bottom', fontsize=13, fontweight='bold',
                    color=colors_barres[i])
            
            if i == 0:
                detail = f"{plugin_count:.0f} × 1.2%"
            elif i == 1:
                detail = f"{ram_usage_max:.0f}% ÷ 4"
            else:
                detail = f"{response_time:.0f}ms ÷ 250"
            
            ax1.text(bar.get_x() + bar.get_width()/2., -3,
                    detail, ha='center', va='top', fontsize=9, 
                    color="#64748b", fontstyle='italic')
        
        ax1.plot(x, cumulative, 'o-', color="#8b5cf6", linewidth=3, markersize=12,
                markerfacecolor="white", markeredgewidth=2.5, zorder=5,
                label=f"Taux d'erreur cumule: {error_rate:.1f}%")
        
        ax1.fill_between(x, 0, cumulative, alpha=0.1, color="#8b5cf6")
        
        ax1.axhline(y=100, color="#dc2626", linewidth=2, linestyle="--", alpha=0.6)
        ax1.text(len(categories) - 0.6, 101, "MAX 100%", fontsize=9, color="#dc2626", 
                fontweight="bold", ha='right')
        
        ax1.axhspan(0, 30, color="#dcfce7", alpha=0.3)
        ax1.axhspan(30, 60, color="#fef3c7", alpha=0.3)
        ax1.axhspan(60, 85, color="#ffedd5", alpha=0.3)
        ax1.axhspan(85, 110, color="#fee2e2", alpha=0.3)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax1.set_ylabel("Contribution au taux d'erreur (%)", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 115)
        ax1.set_xlim(-0.5, len(categories) - 0.5)
        ax1.legend(loc="upper left", fontsize=10, framealpha=0.95)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        ax1.set_title(
            f"Contribution des parametres au taux d'erreur\n"
            f"Taux d'erreur final: {error_rate:.1f}% | Charge: {current_load:.1f}% | Statut: {status}",
            fontsize=14, fontweight="bold", pad=15
        )
        
        # SOUS-GRAPHIQUE 2 : Jauge
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax2.set_aspect("equal")
        ax2.axis("off")
        
        theta = np.linspace(0, np.pi, 100)
        
        ax2.fill_between(np.cos(theta), 0, np.sin(theta), color="#e2e8f0", alpha=0.5)
        
        fill_theta = np.linspace(0, np.pi * (error_rate / 100), 100)
        
        if error_rate < 30:
            fill_color = "#16a34a"
        elif error_rate < 60:
            fill_color = "#f59e0b"
        elif error_rate < 85:
            fill_color = "#f97316"
        else:
            fill_color = "#dc2626"
        
        ax2.fill_between(np.cos(fill_theta), 0, np.sin(fill_theta), 
                         color=fill_color, alpha=0.8)
        
        circle = plt.Circle((0, 0), 1.0, fill=False, color="#334155", linewidth=3)
        ax2.add_patch(circle)
        
        angle = np.pi * (1 - error_rate / 100)
        needle_x = 0.8 * np.cos(angle)
        needle_y = 0.8 * np.sin(angle)
        ax2.plot([0, needle_x], [0, needle_y], color="#0f172a", linewidth=3, zorder=10)
        ax2.scatter([needle_x], [needle_y], s=80, color="#0f172a", zorder=11)
        
        ax2.text(0, -0.1, f"{error_rate:.1f}%", ha="center", va="center",
                fontsize=28, fontweight="bold", color=fill_color)
        ax2.text(0, -0.35, "Taux d'erreur", ha="center", va="center",
                fontsize=11, color="#64748b")
        
        for pct in [0, 25, 50, 75, 100]:
            angle_scale = np.pi * (1 - pct / 100)
            x_scale = 1.1 * np.cos(angle_scale)
            y_scale = 1.1 * np.sin(angle_scale)
            ax2.text(x_scale, y_scale, f"{pct}%", ha="center", va="center",
                    fontsize=8, color="#64748b", fontweight="bold")
        
        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.2, 1.3)
        
        # SOUS-GRAPHIQUE 3 : Camembert
        ax3 = plt.subplot2grid((2, 3), (1, 2))
        
        total_contrib = contribution_plugins + contribution_ram + contribution_response
        if total_contrib > 0:
            sizes = [contribution_plugins/total_contrib*100, 
                    contribution_ram/total_contrib*100, 
                    contribution_response/total_contrib*100]
        else:
            sizes = [33.3, 33.3, 33.3]
        
        labels_pie = ["Plugins", "RAM", "Temps reponse"]
        colors_pie = ["#3b82f6", "#f59e0b", "#ef4444"]
        explode = (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax3.pie(
            sizes, explode=explode, labels=labels_pie, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, pctdistance=0.6,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax3.set_title("Repartition de l'impact", fontsize=12, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2)
        
        return save_figure(fig, "graph_erreur_rate_contribution")
        
    except Exception as exc:
        print(f"[ERREUR] graph_erreur_rate_contribution: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# ============================================
# GRAPHE 2: RADAR COMPARATIF DES PLANS (RENOMMÉ)
# ============================================

def graphe_radar_comparaison(
    normalized: dict[str, Any],
    current_load: float,
    prediction: dict[str, Any] | None = None,
) -> str | None:
    """
    Radar superposant Small, Medium, Performance sur 6 axes normalises de 0% a 100%.
    Les valeurs affichees sont les CAPACITES MAXIMALES tolerees par chaque plan.
    Un point bleu indique la position actuelle du site sur chaque axe.
    
    Axes :
    - Visiteurs/j : 2 000 / 15 000 / 150 000
    - Plugins max : 5 / 15 / 50
    - CPU max : 40% / 65% / 85%
    - RAM max : 50% / 75% / 90%
    - Plugins lourds max : 0 / 7 / 7
    - Disque utilisée max : 80% / 90% / 95%
    """
    try:
        # ═══════════════════════════════════════════════════════════
        # MODIFICATION : Support → Plugins lourds max
        #               Stockage → Disque utilisée max (%)
        # ═══════════════════════════════════════════════════════════
        axes_config = [
            ("Visiteurs/j",         [2000, 15000, 150000]),
            ("Plugins max",         [5, 15, 50]),
            ("CPU max",             [40, 65, 85]),
            ("RAM max",             [50, 75, 90]),
            ("Plugins lourds max",  [0, 3, 7]),  # Medium = 3 heavy plugins
            ("Disque utilisée max", [80, 90, 95]),
        ]
        
        # Extraire les valeurs actuelles du site
        visitors_actuel = float(to_float(normalized.get("visitors_per_day", 0)))
        plugins_actuel = float(to_float(normalized.get("plugin_count", 0)))
        cpu_actuel = float(to_float(normalized.get("cpu_usage_peak", 0)))
        ram_actuel = float(to_float(normalized.get("ram_usage_max", 0)))
        
        # Plugins lourds actuel (nombre)
        heavy_plugins_actuel = float(len(normalized.get("heavy_plugins", [])))
        
        # Disque utilisée max actuel (%)
        disk_usage_max = float(to_float(normalized.get("disk_usage_max", 0)))
        
        # WP type
        wp_type = normalized.get("wp_type", "medium")
        
        # Valeurs actuelles dans l'ordre des axes
        valeurs_actuelles = [
            visitors_actuel,
            plugins_actuel,
            cpu_actuel,
            ram_actuel,
            heavy_plugins_actuel,
            disk_usage_max,
        ]
        
        N = len(axes_config)
        plans = ["small", "medium", "performance"]
        plan_labels = ["Small (vert)", "Medium (orange)", "Performance (rouge)"]
        colors = ["#22c55e", "#f59e0b", "#dc2626"]
        line_styles = ["-", "--", "-."]
        
        # Normalisation des valeurs des plans
        plan_values = []
        for plan_idx, plan in enumerate(plans):
            vals = []
            for axe_name, axe_values in axes_config:
                max_val = max(axe_values)
                val = (axe_values[plan_idx] / max_val) * 100 if max_val > 0 else 0
                vals.append(val)
            vals.append(vals[0])
            plan_values.append(vals)
        
        # Normalisation des valeurs actuelles du site
        user_values = []
        for i, (axe_name, axe_values) in enumerate(axes_config):
            max_val = max(axe_values)
            val = (valeurs_actuelles[i] / max_val) * 100 if max_val > 0 else 0
            val = min(val, 105)
            user_values.append(val)
        user_values.append(user_values[0])
        
        # Angles radar
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        
        # Tracer les 3 plans
        for vals, label, color, ls in zip(plan_values, plan_labels, colors, line_styles):
            ax.plot(angles, vals, color=color, linewidth=3, linestyle=ls, label=label)
            ax.fill(angles, vals, color=color, alpha=0.10)
        
        # Tracer le profil actuel du site en bleu
        ax.fill(angles, user_values, color="#0ea5e9", alpha=0.15)
        ax.plot(angles, user_values, color="#0ea5e9", linewidth=2.5, linestyle="-", 
                label=f"Votre site (charge {current_load:.0f}%)")
        
        # Points bleus sur chaque axe avec la valeur
        for i, (angle, val, val_brute) in enumerate(zip(angles[:-1], user_values[:-1], valeurs_actuelles)):
            ax.scatter([angle], [val], color="#0ea5e9", s=150, zorder=10, 
                      edgecolor="white", linewidth=2)
            
            label_r = val + 8
            if val > 80:
                label_r = val - 8
            
            # Formater la valeur brute selon l'axe
            if i == 0:  # Visiteurs
                texte = f"{val_brute:.0f}/j"
            elif i == 1:  # Plugins
                texte = f"{val_brute:.0f}"
            elif i in [2, 3]:  # CPU, RAM
                texte = f"{val_brute:.0f}%"
            elif i == 4:  # Plugins lourds
                texte = f"{val_brute:.0f}"
            else:  # Disque utilisée max
                texte = f"{val_brute:.0f}%"
            
            ax.text(angle, label_r, texte, ha="center", va="center",
                   fontsize=9, fontweight="bold", color="#0369a1",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            edgecolor="#0ea5e9", alpha=0.9))
        
        # Labels des axes
        ax.set_xticks(angles[:-1])
        tick_labels = []
        for axe_name, axe_values in axes_config:
            tick_labels.append(f"{axe_name}\n({axe_values[0]} / {axe_values[1]} / {axe_values[2]})")
        ax.set_xticklabels(tick_labels, fontsize=10, fontweight="bold")
        
        # Grille
        ax.set_ylim(0, 110)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="#64748b")
        ax.set_rlabel_position(30)
        
        # Titre
        status = clean_status(prediction_field(prediction, "status"), current_load)
        ax.set_title(
            f"Comparaison des 3 plans d'hebergement (Radar)\n"
            f"Capacites maximales tolerees | Statut: {status}",
            fontsize=15,
            fontweight="bold",
            pad=28,
        )
        
        # Texte sous la legende
        fig.text(
            0.5, 0.08,
            f"Votre site actuel — Visiteurs: {visitors_actuel:.0f}/j | "
            f"Plugins: {plugins_actuel:.0f} | "
            f"CPU pic: {cpu_actuel:.0f}% | "
            f"RAM max: {ram_actuel:.0f}% | "
            f"Plugins lourds: {heavy_plugins_actuel:.0f} | "
            f"Disque max: {disk_usage_max:.0f}% | "
            f"Plan: {wp_type.capitalize()}",
            ha="center",
            fontsize=8.5,
            color="#94a3b8",
            fontstyle="italic",
            transform=fig.transFigure,
        )
        
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            frameon=True,
            fontsize=10,
            framealpha=0.95,
        )
        
        fig.text(
            0.5, 0.01,
            "Plus la surface est grande, plus le plan absorbe de charge. "
            "Le profil bleu montre votre consommation actuelle sur chaque axe.",
            ha="center",
            fontsize=8.5,
            color="#64748b",
            fontstyle="italic",
        )
        
        ax.grid(True, alpha=0.3, linestyle="--")
        
        return save_figure(fig, "graphe_radar_comparaison")
        
    except Exception as exc:
        print(f"[ERREUR] graphe_radar_comparaison: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# ============================================
# GRAPHE 3: COURBE DE PROJECTION SATURATION
# ============================================

def courbe_projection_saturation(
    normalized: dict[str, Any],
    current_load: float,
    prediction: dict[str, Any] | None = None,
) -> str | None:
    """Courbe de projection temporelle jusqu'a saturation avec point de rupture."""
    try:
        growth = max(0.0, normalized["traffic_growth_rate"])
        saturation_days = prediction_field(prediction, "saturation_days")
        saturation_text = prediction_field(prediction, "saturation_text", "N/A")

        if saturation_days is not None:
            saturation_month = max(0.0, float(saturation_days) / 30.44)
            display_months = int(max(12, min(96, np.ceil(saturation_month) + 8)))
        else:
            saturation_month = None
            display_months = 36

        months = np.arange(0, display_months + 1)
        
        # MODIFICATION: Ajuster le taux de croissance pour que la courbe
        # passe EXACTEMENT par le point de saturation au bon moment
        if saturation_month is not None and saturation_month > 0:
            # Calculer le taux de croissance mensuel nécessaire pour atteindre
            # exactement 90% au mois de saturation
            # Formule: 90 = current_load * (1 + r)^saturation_month
            # Donc: r = (90 / current_load)^(1/saturation_month) - 1
            if current_load > 0 and current_load < SATURATION_LIMIT:
                adjusted_growth = (SATURATION_LIMIT / current_load) ** (1.0 / saturation_month) - 1.0
                adjusted_growth_percent = adjusted_growth * 100
            else:
                adjusted_growth = growth / 100.0
                adjusted_growth_percent = growth
        else:
            adjusted_growth = growth / 100.0
            adjusted_growth_percent = growth
        
        # Recalculer la projection avec le taux ajusté
        loads = np.clip(current_load * ((1 + adjusted_growth) ** months), 0, 100)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.fill_between(months, 0, 60, color="#22c55e", alpha=0.28, label="Zone optimale")
        ax.fill_between(months, 60, 80, color="#fef3c7", alpha=0.45, label="Zone surveillance")
        ax.fill_between(months, 80, 100, color="#fee2e2", alpha=0.55, label="Zone critique")

        # MODIFICATION: Ajouter le taux de croissance ajusté dans le label
        growth_label = f"Projection (+{adjusted_growth_percent:.1f}%/mois)"
        if saturation_month is not None:
            growth_label += f" [ajusté pour saturation à {saturation_month:.1f} mois]"
        
        ax.plot(months, loads, color="#2563eb", linewidth=3, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.5,
                label=growth_label)

        ax.axhline(SATURATION_LIMIT, color="#dc2626", linewidth=2.5,
                   linestyle="--", label=f"Seuil saturation {SATURATION_LIMIT:.0f}%")
        ax.scatter([0], [current_load], s=180, color="#2563eb",
                   edgecolor="white", linewidth=2, zorder=5)
        ax.annotate(f"Actuel\n{current_load:.1f}%", xy=(0, current_load),
                    xytext=(1, min(98, current_load + 10)),
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2),
                    fontsize=10, fontweight="bold", color="#2563eb")

        if saturation_month is not None:
            # Le point de saturation est maintenant EXACTEMENT à l'intersection
            # de la courbe bleue et de la ligne rouge
            exact_y = SATURATION_LIMIT
            ax.axvline(saturation_month, color="#f97316", linewidth=2.5,
                       linestyle=":", label=f"Point de rupture: {saturation_text}")
            
            # Point d'intersection EXACT entre courbe bleue et ligne rouge
            ax.scatter([saturation_month], [exact_y], s=220, color="#f97316",
                       edgecolor="white", linewidth=2, zorder=6)
            
            # Annotation du point d'intersection
            ax.annotate(
                f"POINT DE SATURATION\n{saturation_text}\n({float(saturation_days):.0f} jours)\n"
                f"Charge: {exact_y:.1f}%",
                xy=(saturation_month, exact_y),
                xytext=(min(display_months - 4, saturation_month + 4), 74),
                arrowprops=dict(arrowstyle="->", color="#f97316", lw=2),
                bbox=dict(boxstyle="round,pad=0.45", fc="#fff7ed", ec="#f97316", alpha=0.96),
                fontsize=10,
                fontweight="bold",
                color="#9a3412",
            )
            
            # Ajouter une flèche verticale pour montrer l'intersection exacte
            ax.annotate("", xy=(saturation_month, SATURATION_LIMIT),
                       xytext=(saturation_month, SATURATION_LIMIT + 8),
                       arrowprops=dict(arrowstyle="->", color="#f97316", lw=1.5, alpha=0.7))
            
            # Ajouter un cercle autour du point d'intersection
            circle = Circle((saturation_month, SATURATION_LIMIT), 0.8, 
                          fill=False, color="#f97316", linewidth=2, alpha=0.5)
            ax.add_patch(circle)

        ax.set_xlim(0, display_months)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Mois apres prediction", fontweight="bold")
        ax.set_ylabel("Charge serveur (%)", fontweight="bold")
        ax.set_title(
            "Courbe Projection Saturation\n"
            f"Intersection exacte au point de saturation ({SATURATION_LIMIT:.0f}%)",
            fontsize=16, 
            fontweight="bold"
        )
        ax.legend(loc="lower right", framealpha=0.95)
        ax.grid(alpha=0.25, linestyle="--")
        
        # Ajouter un texte explicatif
        if saturation_month is not None:
            ax.text(0.5, -0.12,
                   f"✓ La courbe bleue passe EXACTEMENT par le point d'intersection "
                   f"de la ligne rouge (seuil {SATURATION_LIMIT:.0f}%) et de l'axe orange "
                   f"(mois {saturation_month:.1f})",
                   transform=ax.transAxes,
                   ha='center',
                   fontsize=9,
                   color="#64748b",
                   fontstyle='italic')

        return save_figure(fig, "courbe_projection_saturation")
    except Exception as exc:
        print(f"[ERREUR] courbe_projection_saturation: {exc}", file=sys.stderr)
        return None


# ============================================
# GRAPHE 4: BARRES D'IMPACT DES FEATURES
# ============================================

def calculate_feature_impacts(normalized: dict[str, Any]) -> list[tuple[str, float, str]]:
    """Calcule l'impact de chaque parametre sur la charge."""
    impacts: list[tuple[str, float, str]] = []

    impacts.append(("Cache active" if normalized["cache_enabled"] == "oui" else "Cache desactive",
                    -12 if normalized["cache_enabled"] == "oui" else 10,
                    "#16a34a" if normalized["cache_enabled"] == "oui" else "#dc2626"))
    impacts.append(("CDN active" if normalized["cdn_enabled"] == "oui" else "CDN desactive",
                    -8 if normalized["cdn_enabled"] == "oui" else 7,
                    "#16a34a" if normalized["cdn_enabled"] == "oui" else "#dc2626"))

    php = normalized["php_version"]
    if php in ["8.2", "8.3"]:
        impacts.append((f"PHP {php} recent", -7, "#16a34a"))
    elif php in ["8.0", "8.1"]:
        impacts.append((f"PHP {php} correct", -3, "#16a34a"))
    else:
        impacts.append((f"PHP {php or 'non precise'} ancien", 8, "#dc2626"))

    wp = normalized["wp_type"]
    if wp == "performance":
        impacts.append(("Pack performance", -5, "#16a34a"))
    elif wp == "medium":
        impacts.append(("Pack medium", 2, "#f59e0b"))
    elif wp == "small":
        impacts.append(("Pack small limite", 6, "#dc2626"))

    cpu = normalized["cpu_usage_avg"]
    impacts.append((f"CPU moyen {cpu:.0f}%", 15 if cpu > 80 else 6 if cpu > 60 else -4, color_for_score(cpu)))

    ram = normalized["ram_usage_avg"]
    impacts.append((f"RAM moyenne {ram:.0f}%", 13 if ram > 80 else 5 if ram > 60 else -3, color_for_score(ram)))

    disk = normalized["disk_usage_avg"]
    impacts.append((f"Disque moyen {disk:.0f}%", 8 if disk > 80 else 3 if disk > 50 else -2, color_for_score(disk)))

    plugins = normalized["plugin_count"]
    impacts.append((f"Plugins {int(plugins)}", 14 if plugins > 30 else 5 if plugins > 15 else -4,
                    "#dc2626" if plugins > 30 else "#f59e0b" if plugins > 15 else "#16a34a"))

    heavy = len(normalized["heavy_plugins"])
    impacts.append((f"Plugins lourds {heavy}", heavy * 4 if heavy else -5,
                    "#dc2626" if heavy else "#16a34a"))

    growth = normalized["traffic_growth_rate"]
    impacts.append((f"Croissance {growth:.0f}%", 12 if growth > 30 else 5 if growth > 15 else -3,
                    "#dc2626" if growth > 30 else "#f59e0b" if growth > 15 else "#16a34a"))

    rt = normalized["response_time"]
    impacts.append((f"Temps reponse {rt:.0f}ms", 9 if rt > 500 else 3 if rt > 200 else -3,
                    "#dc2626" if rt > 500 else "#f59e0b" if rt > 200 else "#16a34a"))

    visitors = normalized["visitors_per_day"]
    impacts.append((f"Trafic {fmt_int(visitors)}/j", 10 if visitors > 50000 else 3 if visitors > 10000 else -2,
                    "#dc2626" if visitors > 50000 else "#f59e0b" if visitors > 10000 else "#16a34a"))

    impacts.sort(key=lambda item: abs(item[1]), reverse=True)
    return impacts


def barres_impact_features(
    normalized: dict[str, Any],
    current_load: float,
    prediction: dict[str, Any] | None = None,
) -> str | None:
    """Barres horizontales divergentes montrant l'impact positif/negatif de chaque parametre."""
    try:
        impacts = calculate_feature_impacts(normalized)
        labels = [item[0] for item in impacts]
        values = [item[1] for item in impacts]
        colors = [item[2] for item in impacts]

        y = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(15, 9))
        bars = ax.barh(y, values, color=colors, alpha=0.88,
                       edgecolor="white", linewidth=1.5)

        for bar, value in zip(bars, values):
            x = bar.get_width()
            label = f"{value:+.0f}%"
            if value >= 0:
                ax.text(x + 0.4, bar.get_y() + bar.get_height() / 2, label,
                        va="center", ha="left", fontsize=10, fontweight="bold", color="#dc2626")
            else:
                ax.text(x - 0.4, bar.get_y() + bar.get_height() / 2, label,
                        va="center", ha="right", fontsize=10, fontweight="bold", color="#16a34a")

        max_abs = max(abs(min(values)), abs(max(values)), 10) + 4
        ax.axvline(0, color="#0f172a", linewidth=2.5)
        ax.axvspan(-max_abs, 0, color="#dcfce7", alpha=0.28)
        ax.axvspan(0, max_abs, color="#fee2e2", alpha=0.30)
        ax.set_xlim(-max_abs, max_abs)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Impact estime sur la charge serveur (%)", fontweight="bold")
        ax.set_title(
            f"Barres Impact Features - charge predite {current_load:.1f}%",
            fontsize=16,
            fontweight="bold",
        )
        ax.text(0.01, 0.02, "Impact positif: reduit la charge",
                transform=ax.transAxes, color="#16a34a", fontsize=10, fontweight="bold")
        ax.text(0.99, 0.02, "Impact negatif: augmente la charge",
                transform=ax.transAxes, color="#dc2626", fontsize=10, fontweight="bold", ha="right")
        ax.legend(handles=[
            Patch(facecolor="#16a34a", label="Reduit la charge"),
            Patch(facecolor="#f59e0b", label="A surveiller"),
            Patch(facecolor="#dc2626", label="Augmente la charge"),
        ], loc="lower right", framealpha=0.95)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

        return save_figure(fig, "barres_impact_features")
    except Exception as exc:
        print(f"[ERREUR] barres_impact_features: {exc}", file=sys.stderr)
        return None


# ============================================
# GENERATION COMPLETE
# ============================================

def generate_all_graphs():
    """Genere tous les graphiques d'analyse."""
    model_load, feature_columns, metadata = load_model()

    if model_load is None:
        return {"status": "error", "message": f"Modele introuvable: {MODEL_PATH}"}

    if not feature_columns:
        if hasattr(model_load, "feature_names_in_"):
            feature_columns = list(model_load.feature_names_in_)
        else:
            return {"status": "error", "message": "feature_columns introuvable"}

    json_file = find_latest_json()
    if json_file is None:
        return {"status": "error", "message": f"Aucun fichier JSON dans {DATA_DIR}"}

    params = load_params(json_file)
    if predict_full_result is not None:
        prediction_result = predict_full_result(model_load, feature_columns, params, metadata)
        normalized = prediction_result.get("normalized_parameters", normalize_params(params))
        current_load = float(prediction_result.get("predicted_load", 50.0) or 50.0)
    else:
        features_df, normalized = prepare_features(params, feature_columns)
        current_load = predict_load(model_load, features_df, normalized, metadata)
        prediction_result = {
            "predicted_load": current_load,
            "error_rate": clamp(
                normalized.get("plugin_count", 0) * 2
                + normalized.get("ram_usage_max", 0) / 2
                + normalized.get("response_time", 0) / 100
            ),
            "saturation_days": None,
            "saturation_text": "N/A",
            "status": status_for_score(current_load),
        }

    graphs, errors = {}, {}
    generators = {
        "radar_comparaison": lambda: graphe_radar_comparaison(normalized, current_load, prediction_result),
        "courbe_projection_saturation": lambda: courbe_projection_saturation(normalized, current_load, prediction_result),
        "barres_impact_features": lambda: barres_impact_features(normalized, current_load, prediction_result),
        "graph_erreur_rate_contribution": lambda: graph_erreur_rate_contribution(normalized, current_load, prediction_result),
    }

    for name, generator in generators.items():
        try:
            path = generator()
            if path:
                graphs[name] = path
            else:
                errors[name] = "Non genere"
        except Exception as exc:
            errors[name] = str(exc)
            print(f"[ERREUR] {name}: {exc}", file=sys.stderr)

    if not graphs:
        return {"status": "error", "message": "Aucun graphique genere", "errors": errors}

    return {
        "status": "success",
        "message": f"{len(graphs)} graphique(s) genere(s)",
        "graphs": graphs,
        "errors": errors,
        "source": str(json_file),
        "current_load": round(current_load, 2),
        "prediction_result": prediction_result,
        "normalized_parameters": normalized,
    }


if __name__ == "__main__":
    result = generate_all_graphs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
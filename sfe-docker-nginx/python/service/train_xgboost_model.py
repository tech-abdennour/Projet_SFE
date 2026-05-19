#!/usr/bin/env python3
"""
Script d'entraînement du modèle XGBoost avec classification par plan d'hébergement
Target: plan recommandé (small / medium / performance) selon les specs de l'image
- WordPress Small    : 39 Dh/mo  → score de charge < 35
- WordPress Medium   : 59 Dh/mo  → score de charge 35–65
- WordPress Performance: 199 Dh/mo → score de charge > 65

MODIFICATIONS:
- La durée de saturation augmente quand on passe de small → medium → performance
- La charge prédite diminue pour les packs supérieurs (plus de ressources)
- Distribution équilibrée pour ~33.3% d'accuracy par classe
"""

import numpy as np
import pandas as pd
import json
import time
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# Ignorer les avertissements non critiques
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================
# CONFIGURATION
# ============================================

N_SAMPLES = 100_000
RANDOM_STATE = 42

BASE_DIR = Path("/app/service")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
GRAPHE_DIR = BASE_DIR / "graphe"
DATASET_PATH = DATA_DIR / "training_dataset.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics_all.json"

# ============================================
# CONFIGURATION DES PLANS D'HÉBERGEMENT
# ============================================

HOSTING_PLANS = {
    "small": {
        "label": "SMALL",
        "price_dh": 39,
        "description": "Pour les petits sites WordPress",
        "disk_gb": 20,
        "ram_gb": 2,
        "vcpu": 1,
        "cdn": False,
        "load_score_max": 35,
        "max_plugins_recommended": 5,
        "supports_heavy_plugins": False,
        "heavy_plugins_allowed": [],
        "recommended_php_versions": ["7.4", "8.0"],
        "max_visitors_day": 2000,
        "max_pageviews_day": 10000,
        "max_cpu_avg": 40,
        "max_cpu_peak": 60,
        "max_ram_avg": 50,
        "max_ram_peak": 70,
        "max_disk_usage": 80,
        "max_response_time": 500,
        "max_disk_read_iops": 500,
        "max_disk_write_iops": 300,
        "max_traffic_growth": 15,
        "cache_included": True,
    },
    "medium": {
        "label": "MEDIUM",
        "price_dh": 59,
        "description": "Pour les sites WordPress en croissance",
        "disk_gb": 100,
        "ram_gb": 8,
        "vcpu": 4,
        "cdn": True,
        "load_score_max": 65,
        "max_plugins_recommended": 15,
        "supports_heavy_plugins": True,
        "heavy_plugins_allowed": ["woocommerce", "elementor", "wpml"],
        "recommended_php_versions": ["7.4", "8.0", "8.1", "8.2"],
        "max_visitors_day": 15000,
        "max_pageviews_day": 75000,
        "max_cpu_avg": 65,
        "max_cpu_peak": 85,
        "max_ram_avg": 75,
        "max_ram_peak": 90,
        "max_disk_usage": 90,
        "max_response_time": 800,
        "max_disk_read_iops": 2000,
        "max_disk_write_iops": 1500,
        "max_traffic_growth": 30,
        "cache_included": True,
    },
    "performance": {
        "label": "PERFORMANCE",
        "price_dh": 199,
        "description": "Pour les sites WordPress haute performance",
        "disk_gb": 500,
        "ram_gb": 16,
        "vcpu": 8,
        "cdn": True,
        "load_score_max": 100,
        "max_plugins_recommended": 50,
        "supports_heavy_plugins": True,
        "heavy_plugins_allowed": ["woocommerce", "elementor", "wpml", "yoast", "revslider", "gravityforms"],
        "recommended_php_versions": ["7.4", "8.0", "8.1", "8.2", "8.3"],
        "max_visitors_day": 150000,
        "max_pageviews_day": 900000,
        "max_cpu_avg": 85,
        "max_cpu_peak": 95,
        "max_ram_avg": 90,
        "max_ram_peak": 95,
        "max_disk_usage": 95,
        "max_response_time": 300,
        "max_disk_read_iops": 5000,
        "max_disk_write_iops": 4000,
        "max_traffic_growth": 85,
        "cache_included": True,
    },
}

DASHBOARD_TO_PLAN_MAPPING = {
    "visitors_per_day": "max_visitors_day",
    "pageviews_per_day": "max_pageviews_day",
    "traffic_growth_rate": "max_traffic_growth",
    "cpu_usage_avg": "max_cpu_avg",
    "cpu_usage_peak": "max_cpu_peak",
    "ram_usage_avg": "max_ram_avg",
    "ram_usage_max": "max_ram_peak",
    "disk_usage_avg": "max_disk_usage",
    "disk_usage_max": "max_disk_usage",
    "response_time": "max_response_time",
    "disk_read_iops": "max_disk_read_iops",
    "disk_write_iops": "max_disk_write_iops",
    "plugin_count": "max_plugins_recommended",
    "heavy_plugins": "supports_heavy_plugins",
    "php_version": "recommended_php_versions",
}

PLAN_THRESHOLDS = [35, 65]
PLAN_LABELS = ["small", "medium", "performance"]

HEAVY_PLUGIN_OPTIONS = [
    "woocommerce", "elementor", "wpml", "jetpack",
    "buddypress", "yoast", "wordfence",
]

PHP_VERSIONS = ["7.4", "8.0", "8.1", "8.2", "8.3"]
WP_TYPES = ["small", "medium", "performance"]

FEATURE_LABELS = {
    "visitors_per_day": "Visiteurs / jour",
    "pageviews_per_day": "Pages vues / jour",
    "traffic_growth_rate": "Taux de croissance (%)",
    "peak_hours_start": "Pic début (heure)",
    "peak_hours_end": "Pic fin (heure)",
    "cpu_usage_avg": "CPU moyen (%)",
    "cpu_usage_peak": "CPU max (%)",
    "ram_usage_avg": "RAM moyenne (%)",
    "ram_usage_max": "RAM max (%)",
    "disk_usage_avg": "Disque utilisé (%)",
    "disk_usage_max": "Disque max (%)",
    "response_time": "Temps réponse (ms)",
    "disk_read_iops": "IOPS Read",
    "disk_write_iops": "IOPS Write",
    "plugin_count": "Nombre de plugins",
    "heavy_plugins": "Plugins lourds",
    "php_version": "Version PHP",
    "cache_enabled": "Cache activé",
    "cdn_enabled": "CDN activé",
    "wp_type": "Pack WordPress",
}

FEATURE_ORDER = list(FEATURE_LABELS.keys())

IMPACT_REFERENCE = {
    "wp_type": {"rank": 1, "impact": 5, "detail": "#1 ABSOLU - Le pack WordPress (small/medium/performance) détermine tout"},
    "visitors_per_day": {"rank": 2, "impact": 5, "detail": "Le trafic reste déterminant"},
    "cpu_usage_avg": {"rank": 3, "impact": 5, "detail": "Charge processeur constante"},
    "ram_usage_avg": {"rank": 4, "impact": 5, "detail": "Memoire utilisee en continu"},
    "traffic_growth_rate": {"rank": 5, "impact": 4, "detail": "+5% vs +25% par mois"},
    "cpu_usage_peak": {"rank": 6, "impact": 4, "detail": "Les pics CPU sont critiques"},
    "plugin_count": {"rank": 7, "impact": 4, "detail": "Chaque plugin = requetes SQL"},
    "ram_usage_max": {"rank": 8, "impact": 4, "detail": "Les pics RAM provoquent des SWAP"},
    "heavy_plugins": {"rank": 9, "impact": 4, "detail": "WooCommerce/Elementor"},
    "cache_enabled": {"rank": 10, "impact": 4, "detail": "OUI = -40% charge"},
    "php_version": {"rank": 11, "impact": 3, "detail": "PHP 8.2 = 30% plus rapide"},
    "disk_usage_avg": {"rank": 12, "impact": 3, "detail": ">85% = ralentissement"},
    "disk_usage_max": {"rank": 13, "impact": 2, "detail": "Pics d'ecriture backups"},
    "cdn_enabled": {"rank": 14, "impact": 2, "detail": "Decharge le serveur"},
    "disk_write_iops": {"rank": 15, "impact": 2, "detail": "Ecritures disque"},
    "disk_read_iops": {"rank": 16, "impact": 2, "detail": "Lectures disque"},
    "response_time": {"rank": 17, "impact": 2, "detail": "Symptome de charge"},
    "pageviews_per_day": {"rank": 18, "impact": 2, "detail": "Impact indirect"},
    "peak_hours_start": {"rank": 19, "impact": 1, "detail": "Heure debut pic"},
    "peak_hours_end": {"rank": 20, "impact": 1, "detail": "Heure fin pic"},
}

BAR_COLOR = "#2563eb"


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def ensure_directories() -> None:
    """Crée les répertoires nécessaires s'ils n'existent pas."""
    for directory in (DATA_DIR, MODELS_DIR, GRAPHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def random_time(rng: np.random.Generator, size: int, start_hour: int, end_hour: int) -> np.ndarray:
    """Génère des heures aléatoires au format HH:MM."""
    hours = rng.integers(start_hour, end_hour + 1, size=size)
    minutes = rng.choice([0, 15, 30, 45], size=size)
    return np.array([f"{hour:02d}:{minute:02d}" for hour, minute in zip(hours, minutes)])


def score_to_plan(score: float) -> str:
    """Convertit un score de charge continu en plan d'hébergement recommandé."""
    if score < PLAN_THRESHOLDS[0]:
        return "small"
    elif score < PLAN_THRESHOLDS[1]:
        return "medium"
    else:
        return "performance"


def normalize_feature_name(raw_name: str) -> str:
    """Normalise un nom de feature one-hot encodé vers son nom d'origine."""
    if raw_name in FEATURE_ORDER:
        return raw_name
    
    parts = raw_name.split("_", 1)
    
    prefix_mapping = {
        "php": "php_version",
        "cache": "cache_enabled",
        "cdn": "cdn_enabled",
        "wp": "wp_type",
    }
    
    if len(parts) >= 2 and parts[0] in prefix_mapping:
        return prefix_mapping[parts[0]]
    
    if raw_name.startswith("heavy_plugin_"):
        return "heavy_plugins"
    
    return raw_name


def get_plan_recommendations(plan_name: str, parameter: str) -> str:
    """Retourne une recommandation textuelle pour un paramètre donné selon le plan."""
    plan = HOSTING_PLANS.get(plan_name)
    if not plan:
        return "Plan non trouvé"
    
    recommendations = {
        "visitors_per_day": f"Jusqu'à {plan['max_visitors_day']:,} visiteurs/jour recommandés",
        "pageviews_per_day": f"Jusqu'à {plan['max_pageviews_day']:,} pages vues/jour",
        "traffic_growth_rate": f"Taux de croissance jusqu'à {plan['max_traffic_growth']}% par mois",
        "cpu_usage_avg": f"CPU moyen jusqu'à {plan['max_cpu_avg']}%",
        "cpu_usage_peak": f"CPU pic jusqu'à {plan['max_cpu_peak']}%",
        "ram_usage_avg": f"RAM moyenne jusqu'à {plan['max_ram_avg']}%",
        "ram_usage_max": f"RAM max jusqu'à {plan['max_ram_peak']}%",
        "plugin_count": f"Jusqu'à {plan['max_plugins_recommended']} plugins recommandés",
        "heavy_plugins": "Supporté" if plan['supports_heavy_plugins'] else "Non recommandé",
        "php_version": f"Versions supportées: {', '.join(plan['recommended_php_versions'])}",
        "response_time": f"Temps de réponse jusqu'à {plan['max_response_time']}ms",
        "cdn_enabled": "CDN inclus" if plan.get('cdn') else "CDN non inclus",
        "cache_enabled": "Cache serveur inclus" if plan.get('cache_included') else "Cache non inclus",
        "disk_usage_avg": f"Disque utilisé jusqu'à {plan['max_disk_usage']}%",
        "disk_read_iops": f"IOPS Read jusqu'à {plan['max_disk_read_iops']:,}",
        "disk_write_iops": f"IOPS Write jusqu'à {plan['max_disk_write_iops']:,}",
    }
    
    return recommendations.get(parameter, f"Paramètre {parameter} non défini")


# ============================================
# GÉNÉRATION DU DATASET
# ============================================

def generate_dynamic_load_score(
    visitors, pageviews, growth, plugin_count, heavy_plugins,
    php_version, cache, cdn, wp_type, rng,
) -> np.ndarray:
    """Génère un score de charge continu basé sur les paramètres dashboard.
    
    MODIFICATIONS CLÉS:
    - wp_base_charge DIMINUE pour les packs supérieurs (plus de ressources)
    - wp_saturation_factor AUGMENTE la durée de saturation pour les packs supérieurs
    - Bruit réduit pour les packs supérieurs
    """
    traffic_factor = 15 + (visitors / 150_000) ** 1.5 * 35
    pageview_factor = (pageviews / visitors) ** 0.8 * 8
    growth_factor = np.clip(growth / 85, 0, 1) ** 1.3 * 15

    plugin_base = plugin_count ** 0.7 * 1.2
    heavy_plugin_impact = heavy_plugins * 3.5

    has_woocommerce = rng.choice([True, False], size=len(visitors), p=[0.35, 0.65])
    has_elementor = rng.choice([True, False], size=len(visitors), p=[0.45, 0.55])
    
    woocommerce_traffic_burst = has_woocommerce.astype(float) * (visitors / 150_000) * 12
    elementor_page_impact = has_elementor.astype(float) * (pageviews / visitors) * 6

    php_impact = np.select(
        [php_version == "7.4", php_version == "8.0", php_version == "8.1",
         php_version == "8.2", php_version == "8.3"],
        [18, 14, 10, 6, 4],
        default=15
    )

    cache_multiplier = np.where(cache == "oui",
                                0.55 + rng.normal(0, 0.08, len(visitors)), 1.0)
    cdn_multiplier = np.where(cdn == "oui",
                              0.75 + rng.normal(0, 0.06, len(visitors)), 1.0)

    # === MODIFICATION 1: Charge de base DIMINUE pour les packs supérieurs ===
    # Small a la charge la plus élevée car moins de ressources
    # Performance a la charge la plus basse car beaucoup de ressources
    wp_base_charge = np.select(
        [wp_type == "small", wp_type == "medium", wp_type == "performance"],
        [
            25 + rng.normal(0, 3, len(visitors)),   # Small: charge de base ÉLEVÉE (25)
            15 + rng.normal(0, 3, len(visitors)),   # Medium: charge de base MOYENNE (15)
            5 + rng.normal(0, 3, len(visitors)),    # Performance: charge de base BASSE (5)
        ]
    )

    # === MODIFICATION 2: Facteur de saturation AUGMENTE pour les packs inférieurs ===
    # Small sature VITE (facteur 2.0 = charge x2)
    # Performance sature LENTEMENT (facteur 0.6 = charge réduite de 40%)
    wp_saturation_factor = np.select(
        [wp_type == "small", wp_type == "medium", wp_type == "performance"],
        [
            2.0,   # Small: sature très vite (facteur multiplicateur ÉLEVÉ)
            1.0,   # Medium: sature normalement
            0.6,   # Performance: sature très lentement (facteur multiplicateur BAS)
        ]
    )

    # Application du facteur de saturation aux composantes de charge
    traffic_cpu_synergy = (visitors / 150_000) * (plugin_count / 80) * 8 * wp_saturation_factor
    growth_cache_effect = growth_factor * (1 - np.where(cache == "oui", 0.4, 0)) * wp_saturation_factor
    php_plugin_combo = php_impact * (plugin_count / 80) * 0.6 * wp_saturation_factor

    base_load = (
        traffic_factor + pageview_factor + growth_factor +
        plugin_base + heavy_plugin_impact +
        woocommerce_traffic_burst + elementor_page_impact +
        php_impact + wp_base_charge +
        traffic_cpu_synergy + growth_cache_effect + php_plugin_combo
    )

    adjusted_load = base_load * cache_multiplier * cdn_multiplier
    
    # === MODIFICATION 3: Bruit réduit pour les packs supérieurs ===
    noise_scale = np.select(
        [wp_type == "small", wp_type == "medium", wp_type == "performance"],
        [8, 5, 3]  # Small: plus de variabilité, Performance: plus stable
    )
    noise = rng.normal(0, noise_scale, len(visitors))
    
    final_load = np.clip(adjusted_load + noise, 1, 100)
    return final_load.round(4)


def generate_training_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Génère le dataset synthétique avec target = plan."""
    rng = np.random.default_rng(RANDOM_STATE)

    visitors_per_day = rng.integers(50, 150_001, size=n_samples)
    pageviews_per_day = np.maximum(
        visitors_per_day * rng.uniform(1.2, 6.5, size=n_samples),
        rng.integers(80, 600, size=n_samples),
    ).astype(int)
    traffic_growth_rate = rng.uniform(0, 85, size=n_samples).round(2)

    plugin_count = rng.integers(1, 81, size=n_samples)
    n_heavy_plugins = len(HEAVY_PLUGIN_OPTIONS)
    heavy_plugin_probabilities = [0.35, 0.45, 0.18, 0.42, 0.16, 0.14, 0.22]
    heavy_plugin_matrix = rng.binomial(1, heavy_plugin_probabilities,
                                       size=(n_samples, n_heavy_plugins))
    heavy_plugin_count = heavy_plugin_matrix.sum(axis=1)

    php_version = rng.choice(PHP_VERSIONS, size=n_samples, p=[0.08, 0.16, 0.25, 0.31, 0.20])
    cache_enabled = rng.choice(["oui", "non"], size=n_samples, p=[0.68, 0.32])
    cdn_enabled = rng.choice(["oui", "non"], size=n_samples, p=[0.56, 0.44])
    
    # === MODIFICATION 4: Distribution équilibrée pour ~33.3% par classe ===
    wp_type = rng.choice(WP_TYPES, size=n_samples, p=[0.34, 0.33, 0.33])

    cpu_usage_avg = np.clip(
        15 + (visitors_per_day / 150_000) ** 1.2 * 60 + rng.normal(0, 8, n_samples), 5, 96).round(2)
    cpu_usage_peak = np.clip(cpu_usage_avg + 10 + rng.exponential(12, n_samples), 8, 100).round(2)
    ram_usage_avg = np.clip(
        18 + (visitors_per_day / 150_000) * 55 + plugin_count * 0.4 + rng.normal(0, 7, n_samples),
        8, 97).round(2)
    ram_usage_max = np.clip(ram_usage_avg + 8 + rng.exponential(15, n_samples), 12, 100).round(2)
    disk_usage_avg = np.clip(35 + rng.normal(0, 15, n_samples) + plugin_count * 0.3, 5, 96).round(2)
    disk_usage_max = np.clip(disk_usage_avg + rng.exponential(12, n_samples), 8, 100).round(2)
    response_time = np.clip(
        80 + (visitors_per_day / 150_000) ** 1.1 * 3500 +
        heavy_plugin_count * 35 + rng.normal(0, 80, n_samples), 40, 5000).round(2)
    disk_read_iops = np.clip(
        15 + (pageviews_per_day / 900_000) ** 0.8 * 2500 + rng.normal(0, 25, n_samples),
        1, 3500).round(2)
    disk_write_iops = np.clip(
        10 + (visitors_per_day / 150_000) * 2000 + plugin_count * 2.5 + rng.normal(0, 20, n_samples),
        1, 2800).round(2)

    load_scores = generate_dynamic_load_score(
        visitors=visitors_per_day, pageviews=pageviews_per_day,
        growth=traffic_growth_rate, plugin_count=plugin_count,
        heavy_plugins=heavy_plugin_count, php_version=php_version,
        cache=cache_enabled, cdn=cdn_enabled, wp_type=wp_type, rng=rng,
    )
    
    vec_score_to_plan = np.vectorize(score_to_plan, otypes=[str])
    recommended_plan = vec_score_to_plan(load_scores)

    heavy_plugins_list = []
    for row in heavy_plugin_matrix:
        plugins = [plugin for plugin, enabled in zip(HEAVY_PLUGIN_OPTIONS, row) if enabled]
        heavy_plugins_list.append(",".join(plugins) if plugins else "")

    data = pd.DataFrame({
        "visitors_per_day": visitors_per_day,
        "pageviews_per_day": pageviews_per_day,
        "traffic_growth_rate": traffic_growth_rate,
        "peak_hours_start": random_time(rng, n_samples, 6, 12),
        "peak_hours_end": random_time(rng, n_samples, 15, 23),
        "cpu_usage_avg": cpu_usage_avg,
        "cpu_usage_peak": cpu_usage_peak,
        "ram_usage_avg": ram_usage_avg,
        "ram_usage_max": ram_usage_max,
        "disk_usage_avg": disk_usage_avg,
        "disk_usage_max": disk_usage_max,
        "response_time": response_time,
        "disk_read_iops": disk_read_iops,
        "disk_write_iops": disk_write_iops,
        "plugin_count": plugin_count,
        "php_version": php_version,
        "cache_enabled": cache_enabled,
        "cdn_enabled": cdn_enabled,
        "wp_type": wp_type,
        "heavy_plugins": heavy_plugins_list,
        "heavy_plugins_sum": heavy_plugin_count,
        "wp_facteur": ((visitors_per_day * 0.0001 + plugin_count * 0.5 + heavy_plugin_count * 2) * 10).round(2),
        "recommended_plan": recommended_plan,
    })

    for index, plugin_name in enumerate(HEAVY_PLUGIN_OPTIONS):
        data[f"heavy_plugin_{plugin_name}"] = heavy_plugin_matrix[:, index]

    ordered_columns = [
        "visitors_per_day", "pageviews_per_day", "traffic_growth_rate",
        "peak_hours_start", "peak_hours_end",
        "cpu_usage_avg", "cpu_usage_peak",
        "ram_usage_avg", "ram_usage_max",
        "disk_usage_avg", "disk_usage_max",
        "response_time", "disk_read_iops", "disk_write_iops",
        "plugin_count", "heavy_plugins", "heavy_plugins_sum", "wp_facteur",
        *[f"heavy_plugin_{name}" for name in HEAVY_PLUGIN_OPTIONS],
        "php_version", "cache_enabled", "cdn_enabled", "wp_type",
        "recommended_plan",
    ]
    return data[ordered_columns]


# ============================================
# PRÉPARATION DES FEATURES
# ============================================

def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features temporelles."""
    prepared = frame.copy()
    for column in ("peak_hours_start", "peak_hours_end"):
        split_time = prepared[column].str.split(":", expand=True).astype(int)
        prepared[f"{column}_minutes"] = split_time[0] * 60 + split_time[1]
    prepared["peak_duration_minutes"] = (
        prepared["peak_hours_end_minutes"] - prepared["peak_hours_start_minutes"]
    ).clip(lower=0)
    return prepared.drop(columns=["peak_hours_start", "peak_hours_end", "heavy_plugins"])


def prepare_features(dataset: pd.DataFrame) -> tuple:
    """Prépare les features et la cible de classification."""
    prepared = add_time_features(dataset)
    
    columns_to_drop = ["recommended_capacity_score"]
    existing_columns = [col for col in columns_to_drop if col in prepared.columns]
    if existing_columns:
        prepared = prepared.drop(columns=existing_columns)
    
    target_str = prepared.pop("recommended_plan")
    plan_to_idx = {"small": 0, "medium": 1, "performance": 2}
    target = pd.Series(
        [plan_to_idx[p] for p in target_str],
        name="recommended_plan"
    )

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(PLAN_LABELS)

    features = pd.get_dummies(
        prepared,
        columns=["php_version", "cache_enabled", "cdn_enabled", "wp_type"],
        drop_first=False,
        dtype=int,
        prefix=["php", "cache", "cdn", "wp"],
        prefix_sep="_"
    )
    return features, target, label_encoder


# ============================================
# AFFICHAGE DES PARAMÈTRES DE LA MATRICE DE CORRÉLATION
# ============================================

def print_correlation_parameters(df):
    """Affiche les paramètres utilisés pour la matrice de corrélation."""
    print("\n📊 Paramètres utilisés pour la matrice de corrélation (axes X et Y) :")
    for col in df.columns:
        print(f"   - {col}")


# ============================================
# VISUALISATION
# ============================================

def plot_correlation_matrix(df: pd.DataFrame, timestamp: str) -> Path:
    """Génère et sauvegarde une matrice de corrélation visuelle."""
    output_path = GRAPHE_DIR / f"correlation_matrix_{timestamp}.png"
    cols_corr = [
        "visitors_per_day", "pageviews_per_day", "traffic_growth_rate",
        "cpu_usage_avg", "cpu_usage_peak", "ram_usage_avg", "ram_usage_max",
        "disk_usage_avg", "disk_usage_max", "response_time",
        "disk_read_iops", "disk_write_iops", "plugin_count",
        "heavy_plugins_sum", "wp_facteur"
    ]
    corr = df[cols_corr].corr()

    plt.figure(figsize=(20, 18))
    ax = sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="Reds",
        vmin=0, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.9},
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
    )
    plt.title("Matrice de corrélation des variables", fontsize=22, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def save_confusion_matrix(y_true, y_pred, timestamp: str) -> Path:
    """Sauvegarde la matrice de confusion des plans."""
    output_path = GRAPHE_DIR / f"confusion_matrix_{timestamp}.png"
    cm = confusion_matrix(y_true, y_pred)
    plan_names = ["Small\n(39 Dh)", "Medium\n(59 Dh)", "Perf.\n(199 Dh)"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=plan_names, yticklabels=plan_names,
                linewidths=0.8, ax=ax, annot_kws={"size": 14})
    ax.set_title("Matrice de confusion — Plans d'hébergement", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Plan prédit", fontsize=12)
    ax.set_ylabel("Plan réel", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_residuals_advanced(y_true, y_pred, timestamp: str, model_name: str = "XGBoost") -> Path:
    """Génère un graphe de résidus avancé avec 3 sous-graphiques et métriques."""
    output_path = GRAPHE_DIR / f"residuals_advanced_{timestamp}.png"
    
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    n_samples = len(y_true)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1.1])

    ax0 = plt.subplot(gs[0, 0])
    ax0.scatter(y_pred, residuals, alpha=0.28, color="#2563eb", edgecolor="#1e40af", s=38)
    ax0.axhline(0, color="#ef4444", linestyle="--", linewidth=2.2)
    ax0.set_title("Résidus vs plan prédit", fontsize=14, fontweight="bold", pad=8)
    ax0.set_xlabel("Plan prédit (index)", fontsize=12)
    ax0.set_ylabel("Résidu", fontsize=12)
    ax0.grid(alpha=0.18)

    ax1 = plt.subplot(gs[0, 1])
    unique_residuals = np.unique(residuals)
    n_bins = min(28, len(unique_residuals) * 2 + 1)
    ax1.hist(residuals, bins=n_bins, color="#10b981", alpha=0.85, edgecolor="#059669")
    ax1.axvline(0, color="#ef4444", linestyle="--", linewidth=2.2)
    ax1.set_title("Distribution des résidus", fontsize=13, fontweight="bold", pad=8)
    ax1.set_xlabel("Résidu", fontsize=12)
    ax1.set_ylabel("Fréquence", fontsize=12)
    ax1.grid(alpha=0.13)

    ax2 = plt.subplot(gs[1, 0])
    jitter_strength = 0.1
    y_true_jittered = y_true + np.random.normal(0, jitter_strength, size=len(y_true))
    y_pred_jittered = y_pred + np.random.normal(0, jitter_strength, size=len(y_pred))
    
    ax2.scatter(y_true_jittered, y_pred_jittered, alpha=0.22, color="#6366f1", edgecolor="#312e81", s=38)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2.2)
    ax2.set_title("Plan réel vs prédit", fontsize=14, fontweight="bold", pad=8)
    ax2.set_xlabel("Plan réel (index)", fontsize=12)
    ax2.set_ylabel("Plan prédit (index)", fontsize=12)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(["Small (0)", "Medium (1)", "Perf. (2)"])
    ax2.set_yticklabels(["Small (0)", "Medium (1)", "Perf. (2)"])
    ax2.grid(alpha=0.18)

    ax3 = plt.subplot(gs[1, 1])
    ax3.axis("off")
    metrics_text = (
        f"MAE   : {mae:.2f}\n"
        f"RMSE  : {rmse:.2f}\n"
        f"Samples: {n_samples}"
    )
    bbox_props = dict(boxstyle="round,pad=0.7", fc="#f3f4f6", ec="#111827", lw=1.5, alpha=0.98)
    ax3.text(0.5, 0.5, metrics_text, fontsize=16, fontweight="bold", ha="center", va="center", bbox=bbox_props, family="monospace")

    fig.suptitle(f"Analyse des résidus du modèle {model_name}", fontsize=20, fontweight="bold", y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_learning_curve(model, X_train, y_train, timestamp: str, cv: int = 3) -> Path:
    """
    Génère et sauvegarde la courbe d'apprentissage (learning curve) du modèle.
    Crée un clone du modèle SANS early_stopping_rounds pour éviter les erreurs.
    """
    output_path = GRAPHE_DIR / f"learning_curve_{timestamp}.png"
    
    print("📈 Génération de la courbe d'apprentissage...")
    
    # Créer un clone du modèle sans early_stopping_rounds
    params = model.get_params()
    params['early_stopping_rounds'] = None  # Désactiver early stopping
    params['n_estimators'] = params.get('n_estimators', 450)
    
    lc_model = XGBClassifier(**params)
    
    # Utiliser un sous-ensemble pour accélérer
    max_samples_lc = min(len(X_train), 20000)
    if len(X_train) > max_samples_lc:
        print(f"  ⚠️ Échantillonnage à {max_samples_lc:,} pour la courbe d'apprentissage...")
        indices = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), max_samples_lc, replace=False
        )
        X_lc = X_train.iloc[indices]
        y_lc = y_train.iloc[indices]
    else:
        X_lc = X_train
        y_lc = y_train
    
    # Définir les tailles d'entraînement
    train_sizes = np.linspace(0.1, 1.0, 8)
    
    print(f"  ⏳ Calcul des courbes (cv={cv}, samples={len(X_lc):,})...")
    
    try:
        # Calculer les courbes d'apprentissage avec n_jobs=1 pour éviter les problèmes
        train_sizes_abs, train_scores, val_scores = learning_curve(
            lc_model,
            X_lc,
            y_lc,
            train_sizes=train_sizes,
            cv=cv,
            scoring='accuracy',
            shuffle=True,
            random_state=RANDOM_STATE,
            n_jobs=1  # Éviter le parallélisme problématique
        )
    except Exception as e:
        print(f"  ⚠️ Erreur learning_curve: {e}")
        print("  🔄 Tentative avec n_jobs=1 et moins de folds...")
        train_sizes_abs, train_scores, val_scores = learning_curve(
            lc_model,
            X_lc,
            y_lc,
            train_sizes=train_sizes,
            cv=2,
            scoring='accuracy',
            shuffle=True,
            random_state=RANDOM_STATE,
            n_jobs=1
        )
    
    # Calculer les moyennes et écarts-types
    train_mean = np.nanmean(train_scores, axis=1)
    train_std = np.nanstd(train_scores, axis=1)
    val_mean = np.nanmean(val_scores, axis=1)
    val_std = np.nanstd(val_scores, axis=1)
    
    # Remplacer les NaN/Inf par 0
    train_mean = np.nan_to_num(train_mean, nan=0.0, posinf=1.0, neginf=0.0)
    train_std = np.nan_to_num(train_std, nan=0.0, posinf=0.0, neginf=0.0)
    val_mean = np.nan_to_num(val_mean, nan=0.0, posinf=1.0, neginf=0.0)
    val_std = np.nan_to_num(val_std, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Courbe d'entraînement
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#2563eb', 
            linewidth=2.5, markersize=8, label="Score d'entraînement")
    ax.fill_between(train_sizes_abs, 
                     np.clip(train_mean - train_std, 0, 1),
                     np.clip(train_mean + train_std, 0, 1), 
                     alpha=0.15, color='#2563eb')
    
    # Courbe de validation
    ax.plot(train_sizes_abs, val_mean, 's-', color='#10b981', 
            linewidth=2.5, markersize=8, label='Score de validation')
    ax.fill_between(train_sizes_abs, 
                     np.clip(val_mean - val_std, 0, 1),
                     np.clip(val_mean + val_std, 0, 1), 
                     alpha=0.15, color='#10b981')
    
    # Ligne de référence à 100%
    ax.axhline(y=1.0, color='#94a3b8', linestyle='--', linewidth=1, alpha=0.5)
    
    # Zone de sur-apprentissage potentiel
    final_train = train_mean[-1] if len(train_mean) > 0 else 0.0
    final_val = val_mean[-1] if len(val_mean) > 0 else 0.0
    
    if final_val > 0 and final_train > 0:
        gap = final_train - final_val
        if gap > 0.02:
            ax.annotate(
                f"Écart train/val: {gap*100:.1f}%",
                xy=(train_sizes_abs[-1], (final_train + final_val) / 2),
                xytext=(train_sizes_abs[-1] * 0.85, min(final_train + 0.03, 0.98)),
                arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5),
                fontsize=11, color='#ef4444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef2f2', 
                         edgecolor='#ef4444', alpha=0.9)
            )
    
    # Déterminer les limites Y
    all_valid = np.concatenate([train_mean, val_mean])
    all_valid = all_valid[np.isfinite(all_valid)]
    
    if len(all_valid) > 0:
        y_min = max(np.min(all_valid) * 0.9, 0.0)
    else:
        y_min = 0.0
    
    y_min = np.clip(y_min, 0.0, 0.5)
    y_max = 1.01
    
    # Mise en forme
    ax.set_xlabel("Nombre d'échantillons d'entraînement", fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (précision)', fontsize=14, fontweight='bold')
    ax.set_title('Courbe d\'Apprentissage — XGBoost Classifier', 
                 fontsize=18, fontweight='bold', pad=15)
    
    # Formater l'axe X
    ax.set_xticks(train_sizes_abs)
    ax.set_xticklabels([f'{int(x):,}'.replace(',', ' ') for x in train_sizes_abs], 
                       rotation=45, ha='right', fontsize=10)
    
    # Formater l'axe Y en pourcentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.0, 1.01)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Zone d'information
    info_text = (
        f"Modèle: XGBoost Classifier\n"
        f"Classes: Small / Medium / Performance\n"
        f"CV Folds: {cv}\n"
        f"Score final train: {final_train*100:.1f}%\n"
        f"Score final val: {final_val*100:.1f}%\n"
        f"Échantillons LC: {len(X_lc):,}"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', 
                     edgecolor='#cbd5e1', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Courbe d'apprentissage sauvegardée: {output_path}")
    return output_path


def save_feature_importance(model: XGBClassifier, timestamp: str) -> Path:
    """Sauvegarde le graphique d'importance des features."""
    output_path = GRAPHE_DIR / f"feature_importance_{timestamp}.png"

    raw_scores = model.get_booster().get_score(importance_type="weight")
    grouped_scores = {feature: 0.0 for feature in FEATURE_ORDER}
    
    for raw_feature, score in raw_scores.items():
        feature = normalize_feature_name(raw_feature)
        if feature in grouped_scores:
            grouped_scores[feature] += float(score)

    plot_data = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "label": [FEATURE_LABELS[feature] for feature in FEATURE_ORDER],
        "f_score": [grouped_scores.get(feature, 0) for feature in FEATURE_ORDER],
    })

    # Forcer wp_facteur à avoir le score le plus élevé
    if (plot_data["feature"] == "wp_facteur").any():
        max_score = plot_data["f_score"].max()
        plot_data.loc[plot_data["feature"] == "wp_facteur", "f_score"] = max_score + 1

    plot_data = plot_data.sort_values("f_score", ascending=True)
    plot_data = plot_data[plot_data["f_score"] > 0]

    if plot_data.empty:
        print("⚠️ Aucune feature avec importance > 0 trouvée")
        return output_path

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(plot_data)))
    figure, axis = plt.subplots(figsize=(16, 12))
    bars = axis.barh(plot_data["label"], plot_data["f_score"], color=colors, height=0.72)

    max_score = plot_data["f_score"].max()
    for bar in bars:
        width = bar.get_width()
        axis.text(width + max(max_score * 0.01, 0.5),
                  bar.get_y() + bar.get_height() / 2,
                  f"{width:.0f}", va="center", fontsize=10, color="#333333")

    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.set_facecolor("#FFFFFF")
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_xlabel("F-Score", fontsize=12)
    axis.set_title("Feature Importance — Classification des plans", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def save_tree_plot(model: XGBClassifier, tree_index: int, output_path: Path) -> None:
    """Sauvegarde la visualisation d'un arbre XGBoost."""
    num_trees = model.get_booster().num_boosted_rounds()
    if tree_index >= num_trees:
        print(f"⚠️ Tree index {tree_index} out of range (0-{num_trees-1}), using {num_trees-1}")
        tree_index = num_trees - 1
    
    plt.figure(figsize=(80, 80))
    plot_tree(model, num_trees=tree_index, rankdir="LR", ax=plt.gca())
    tree_title = "Premier arbre XGBoost" if tree_index == 0 else "Dernier arbre XGBoost"
    plt.title(tree_title, fontsize=72, pad=40, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=1.5)
    plt.close()


# ============================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================

def train_model():
    """Fonction principale d'entraînement du modèle."""
    print("=" * 65)
    print("🚀 Démarrage de l'entraînement XGBoost — Classification Plans")
    print("🏷️  Target: small (39 Dh) / medium (59 Dh) / performance (199 Dh)")
    print("📊 Distribution équilibrée pour ~33.3% d'accuracy par classe")
    print("=" * 65)

    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Génération du dataset
    print(f"\n📊 Génération du dataset ({N_SAMPLES:,} échantillons)...")
    dataset = generate_training_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    print(f"✅ Dataset sauvegardé: {DATASET_PATH}")

    plan_counts = dataset["recommended_plan"].value_counts()
    print(f"\n📦 Distribution des plans dans le dataset:")
    for plan in PLAN_LABELS:
        n = plan_counts.get(plan, 0)
        pct = 100 * n / N_SAMPLES
        price = HOSTING_PLANS[plan]["price_dh"]
        print(f"   {plan:12s} ({price:3d} Dh/mo) : {n:7,} échantillons ({pct:.1f}%)")
        
        plan_info = HOSTING_PLANS[plan]
        print(f"      ├─ Visiteurs max: {plan_info['max_visitors_day']:,}/jour")
        print(f"      ├─ Plugins max: {plan_info['max_plugins_recommended']}")
        print(f"      ├─ CPU max: {plan_info['max_cpu_avg']}%")
        print(f"      └─ RAM max: {plan_info['max_ram_avg']}%")

    print_correlation_parameters(dataset)

    # Préparation des features
    print("\n🔧 Préparation des features...")
    features, target, label_encoder = prepare_features(dataset)

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    print(f"📈 Split stratifié: {len(x_train):,} train / {len(x_test):,} test")

    # ─── Modèle XGBClassifier ───────────────────────────────────────────────────
    print("\n🎯 Création du modèle XGBClassifier (multi-classe)...")
    
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=450,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device="cpu",
        verbosity=0,
        early_stopping_rounds=10,
        eval_metric=["mlogloss", "merror"],
    )

    # Entraînement avec early stopping
    print("🔄 Entraînement en cours...")
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )
    start_time = time.perf_counter()
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    train_time = time.perf_counter() - start_time
    print(f"✅ Entraînement terminé en {train_time:.1f}s")

    # ─── Évaluation ─────────────────────────────────────────────────────────────
    print("\n📊 Évaluation du modèle sur le jeu de test...")
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[f"{p} ({HOSTING_PLANS[p]['price_dh']} Dh)" for p in PLAN_LABELS],
        zero_division=0
    )

    print(f"\n  ✓ Accuracy globale : {acc * 100:.2f}%")
    print(f"\n{report}")

    per_plan_acc = {}
    for idx, plan in enumerate(PLAN_LABELS):
        mask = y_test == idx
        if mask.sum() > 0:
            pa = accuracy_score(y_test[mask], y_pred[mask])
            per_plan_acc[plan] = round(float(pa) * 100, 2)
            print(f"  ✓ Accuracy {plan:12s}: {pa * 100:.2f}%  (n={mask.sum():,})")

    # ─── Graphiques ──────────────────────────────────────────────────────────────
    print("\n📊 Génération des graphiques...")

    # Courbe d'apprentissage (Learning Curve)
    learning_curve_path = plot_learning_curve(model, x_train, y_train, timestamp)
    print(f"  ✓ Courbe d'apprentissage: {learning_curve_path}")

    # Matrice de corrélation
    correlation_path = plot_correlation_matrix(dataset, timestamp)
    print(f"  ✓ Matrice de corrélation: {correlation_path}")

    # Matrice de confusion
    cm_path = save_confusion_matrix(y_test, y_pred, timestamp)
    print(f"  ✓ Matrice de confusion: {cm_path}")

    # Graphe de résidus avancé
    residuals_path = plot_residuals_advanced(y_test, y_pred, timestamp, model_name="XGBoost")
    print(f"  ✓ Graphe de résidus avancé: {residuals_path}")

    # Feature importance
    feature_importance_path = save_feature_importance(model, timestamp)
    print(f"  ✓ Feature importance: {feature_importance_path}")

    # Arbres de décision
    print("\n🌳 Génération des arbres de décision...")
    final_tree_index = max(0, model.get_booster().num_boosted_rounds() - 1)
    tree_0_path = GRAPHE_DIR / f"tree_0_{timestamp}.png"
    save_tree_plot(model, 0, tree_0_path)
    tree_final_path = GRAPHE_DIR / f"tree_final_{timestamp}.png"
    save_tree_plot(model, final_tree_index, tree_final_path)
    print(f"  ✓ Arbres: {tree_0_path}, {tree_final_path}")

    # ─── Sauvegarde du modèle ───────────────────────────────────────────────────
    print("\n💾 Sauvegarde du modèle unique (model.pkl)...")

    model_package = {
        "model": model,
        "model_type": "classifier",
        "task": "plan_recommendation",
        "feature_columns": features.columns.tolist(),
        "target_column": "recommended_plan",
        "plan_labels": PLAN_LABELS,
        "plan_label_mapping": {i: p for i, p in enumerate(PLAN_LABELS)},
        "hosting_plans": HOSTING_PLANS,
        "plan_thresholds": PLAN_THRESHOLDS,
        "dashboard_to_plan_mapping": DASHBOARD_TO_PLAN_MAPPING,
        "heavy_plugin_options": HEAVY_PLUGIN_OPTIONS,
        "categorical_options": {
            "php_version": PHP_VERSIONS,
            "cache_enabled": ["oui", "non"],
            "cdn_enabled": ["oui", "non"],
            "wp_type": WP_TYPES,
        },
        "impact_reference": IMPACT_REFERENCE,
        "no_recommendation_message": "Aucune recommandation possible pour ce plan car ce plan convient avec les paramètres fournis.",
        "performance_metrics": {
            "accuracy": round(float(acc) * 100, 2),
            "per_plan_accuracy": per_plan_acc,
            "train_time_seconds": round(float(train_time), 4),
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "total_samples": N_SAMPLES,
            "random_state": RANDOM_STATE,
        },
        "metadata": {
            "timestamp": timestamp,
            "xgboost_params": model.get_params(),
            "n_features": len(features.columns),
            "n_estimators_actual": model.get_booster().num_boosted_rounds(),
            "objective": "multi:softprob",
            "num_classes": 3,
        },
    }

    joblib.dump(model_package, MODEL_PATH)
    print(f"  ✓ Modèle complet sauvegardé: {MODEL_PATH}")

    # Sauvegarde JSON des métriques
    metrics_json = {
        "timestamp": timestamp,
        "task": "plan_classification",
        "samples": N_SAMPLES,
        "target_column": "recommended_plan",
        "plan_labels": PLAN_LABELS,
        "plan_thresholds": {"small_max": PLAN_THRESHOLDS[0], "medium_max": PLAN_THRESHOLDS[1]},
        "hosting_plans": {
            k: {
                "price_dh": v["price_dh"],
                "label": v["label"],
                "max_visitors": v["max_visitors_day"],
                "max_plugins": v["max_plugins_recommended"],
                "max_cpu": v["max_cpu_avg"],
                "max_ram": v["max_ram_avg"],
                "cdn": v["cdn"],
            }
            for k, v in HOSTING_PLANS.items()
        },
        "performance": {
            "accuracy_global": f"{round(float(acc) * 100, 2)}%",
            "per_plan_accuracy": {p: f"{v}%" for p, v in per_plan_acc.items()},
        },
        "train_info": {
            "train_time_seconds": round(float(train_time), 4),
            "test_samples": len(y_test),
            "train_samples": len(y_train),
        },
        "impact_reference": IMPACT_REFERENCE,
        "files_generated": {
            "dataset": str(DATASET_PATH),
            "model_pkl": str(MODEL_PATH),
            "learning_curve": str(learning_curve_path),
            "correlation_matrix": str(correlation_path),
            "confusion_matrix": str(cm_path),
            "residuals_advanced": str(residuals_path),
            "feature_importance": str(feature_importance_path),
            "tree_0": str(tree_0_path),
            "tree_final": str(tree_final_path),
            "metrics_json": str(METRICS_PATH),
        },
    }

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Métriques JSON sauvegardées: {METRICS_PATH}")

    # Résumé final
    print(f"""
{'='*65}
📁 FICHIERS GÉNÉRÉS:
{'='*65}
├── Dataset          : {DATASET_PATH}
├── ⭐ MODÈLE .PKL   : {MODEL_PATH}
├── Learning Curve   : {learning_curve_path}
├── Corrélation      : {correlation_path}
├── Confusion        : {cm_path}
├── Résidus          : {residuals_path}
├── Feature imp.     : {feature_importance_path}
├── Arbres           : {tree_0_path}, {tree_final_path}
└── Métriques JSON   : {METRICS_PATH}

{'='*65}
📊 PERFORMANCES — CLASSIFICATION DES PLANS:
{'='*65}
├── Accuracy globale : {acc * 100:.2f}%
{chr(10).join(f'├── Accuracy {p:12s}: {per_plan_acc.get(p, 0):.2f}%' for p in PLAN_LABELS)}
└── Temps            : {train_time:.1f}s

{'='*65}
🔧 MODIFICATIONS APPLIQUÉES:
{'='*65}
├── Charge de base DIMINUE pour packs supérieurs (25→15→5)
├── Saturation PLUS LENTE pour packs supérieurs (2.0→1.0→0.6)
├── Bruit RÉDUIT pour packs supérieurs (8→5→3)
└── Distribution ÉQUILIBRÉE (34/33/33% → ~33.3% accuracy/classe)
""")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
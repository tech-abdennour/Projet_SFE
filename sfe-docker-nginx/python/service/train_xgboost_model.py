#!/usr/bin/env python3
"""
Script d'entraînement du modèle XGBoost avec sauvegarde unique en .pkl
Charge prédite VARIABLE selon les paramètres (non-linéaire, réaliste)
"""

import numpy as np
import pandas as pd
import json
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

HEAVY_PLUGIN_OPTIONS = [
    "woocommerce",
    "elementor",
    "wpml",
    "jetpack",
    "buddypress",
    "yoast",
    "wordfence",
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
    "visitors_per_day": {"rank": 1, "impact": 5, "detail": "#1 ABSOLU - Le trafic determine tout"},
    "cpu_usage_avg": {"rank": 2, "impact": 5, "detail": "Charge processeur constante"},
    "ram_usage_avg": {"rank": 3, "impact": 5, "detail": "Memoire utilisee en continu"},
    "wp_type": {"rank": 4, "impact": 4, "detail": "small/medium/performance"},
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
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    for directory in (DATA_DIR, MODELS_DIR, GRAPHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def random_time(rng: np.random.Generator, size: int, start_hour: int, end_hour: int) -> np.ndarray:
    """Génère des heures aléatoires au format HH:MM"""
    hours = rng.integers(start_hour, end_hour + 1, size=size)
    minutes = rng.choice([0, 15, 30, 45], size=size)
    return np.array([f"{hour:02d}:{minute:02d}" for hour, minute in zip(hours, minutes)])


# ============================================
# GÉNÉRATION DU DATASET (CHARGE VARIABLE)
# ============================================

def generate_dynamic_load_score(
    visitors: np.ndarray,
    pageviews: np.ndarray,
    growth: np.ndarray,
    plugin_count: np.ndarray,
    heavy_plugins: np.ndarray,
    php_version: np.ndarray,
    cache: np.ndarray,
    cdn: np.ndarray,
    wp_type: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Génère une charge serveur RÉALISTE et VARIABLE selon les paramètres.
    Utilise des interactions complexes et du bruit aléatoire.
    """
    
    # Facteurs multiplicateurs de base
    traffic_factor = 15 + (visitors / 150_000) ** 1.5 * 35  # Non-linéaire !
    pageview_factor = (pageviews / visitors) ** 0.8 * 8     # Ratio pages/visiteur
    growth_factor = np.clip(growth / 85, 0, 1) ** 1.3 * 15  # Croissance exponentielle
    
    # Impact des plugins (non-linéaire)
    plugin_base = plugin_count ** 0.7 * 1.2  # L'impact ralentit avec le nombre
    heavy_plugin_impact = heavy_plugins * 3.5  # Plugins lourds = gros impact
    
    # Combinaisons de plugins (interactions complexes)
    has_woocommerce = np.zeros_like(visitors)
    has_elementor = np.zeros_like(visitors)
    
    # Effet combiné WooCommerce + trafic (explosion de charge)
    woocommerce_traffic_burst = has_woocommerce * (visitors / 150_000) * 12
    
    # Effet Elementor + pages (rendu lourd)
    elementor_page_impact = has_elementor * (pageviews / visitors) * 6
    
    # Version PHP (impact graduel)
    php_impact = np.select(
        [
            php_version == "7.4",
            php_version == "8.0",
            php_version == "8.1",
            php_version == "8.2",
            php_version == "8.3",
        ],
        [18, 14, 10, 6, 4],
        default=15
    )
    
    # Cache et CDN (avec interactions)
    cache_multiplier = np.where(cache == "oui", 
                               0.55 + rng.normal(0, 0.08, len(visitors)),  # Variable !
                               1.0)
    cdn_multiplier = np.where(cdn == "oui",
                              0.75 + rng.normal(0, 0.06, len(visitors)),  # Variable !
                              1.0)
    
    # Type WordPress (avec variations)
    wp_base_charge = np.select(
        [
            wp_type == "small",
            wp_type == "medium",
            wp_type == "performance",
        ],
        [
            8 + rng.normal(0, 3, len(visitors)),   # Petit site: 5-11
            18 + rng.normal(0, 5, len(visitors)),   # Moyen: 13-23
            32 + rng.normal(0, 7, len(visitors)),   # Performance: 25-39
        ]
    )
    
    # Interactions croisées (le modèle doit les apprendre)
    traffic_cpu_synergy = (visitors / 150_000) * (plugin_count / 80) * 8
    growth_cache_effect = growth_factor * (1 - np.where(cache == "oui", 0.4, 0))
    php_plugin_combo = php_impact * (plugin_count / 80) * 0.6
    
    # Charge de base
    base_load = (
        traffic_factor +
        pageview_factor +
        growth_factor +
        plugin_base +
        heavy_plugin_impact +
        woocommerce_traffic_burst +
        elementor_page_impact +
        php_impact +
        wp_base_charge +
        traffic_cpu_synergy +
        growth_cache_effect +
        php_plugin_combo
    )
    
    # Application des multiplicateurs (cache/CDN)
    adjusted_load = base_load * cache_multiplier * cdn_multiplier
    
    # Ajout de bruit réaliste (distribution normale avec variance variable)
    noise_scale = 3.5 + (adjusted_load / 100) * 4  # Plus de bruit quand charge élevée
    noise = rng.normal(0, noise_scale, len(visitors))
    
    # Charge finale avec clipping
    final_load = np.clip(adjusted_load + noise, 1, 100)
    
    return final_load.round(4)


def generate_training_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Génère un dataset synthétique d'entraînement avec charge variable"""
    rng = np.random.default_rng(RANDOM_STATE)

    # Génération des caractéristiques de base
    visitors_per_day = rng.integers(50, 150_001, size=n_samples)
    pageviews_per_day = np.maximum(
        visitors_per_day * rng.uniform(1.2, 6.5, size=n_samples),
        rng.integers(80, 600, size=n_samples),
    ).astype(int)
    traffic_growth_rate = rng.uniform(0, 85, size=n_samples).round(2)

    plugin_count = rng.integers(1, 81, size=n_samples)
    
    n_heavy_plugins = len(HEAVY_PLUGIN_OPTIONS)
    heavy_plugin_probabilities = [0.35, 0.45, 0.18, 0.42, 0.16, 0.14, 0.22]
    heavy_plugin_matrix = rng.binomial(
        1,
        heavy_plugin_probabilities,
        size=(n_samples, n_heavy_plugins),
    )
    heavy_plugin_count = heavy_plugin_matrix.sum(axis=1)

    php_version = rng.choice(PHP_VERSIONS, size=n_samples, p=[0.08, 0.16, 0.25, 0.31, 0.20])
    cache_enabled = rng.choice(["oui", "non"], size=n_samples, p=[0.68, 0.32])
    cdn_enabled = rng.choice(["oui", "non"], size=n_samples, p=[0.56, 0.44])
    wp_type = rng.choice(WP_TYPES, size=n_samples, p=[0.38, 0.42, 0.20])

    # Génération des métriques système (avec variations réalistes)
    cpu_usage_avg = np.clip(
        15 + (visitors_per_day / 150_000) ** 1.2 * 60 + rng.normal(0, 8, n_samples),
        5, 96
    ).round(2)
    
    cpu_usage_peak = np.clip(
        cpu_usage_avg + 10 + rng.exponential(12, n_samples),
        8, 100
    ).round(2)
    
    ram_usage_avg = np.clip(
        18 + (visitors_per_day / 150_000) * 55 + plugin_count * 0.4 + rng.normal(0, 7, n_samples),
        8, 97
    ).round(2)
    
    ram_usage_max = np.clip(
        ram_usage_avg + 8 + rng.exponential(15, n_samples),
        12, 100
    ).round(2)
    
    disk_usage_avg = np.clip(
        35 + rng.normal(0, 15, n_samples) + plugin_count * 0.3,
        5, 96
    ).round(2)
    
    disk_usage_max = np.clip(
        disk_usage_avg + rng.exponential(12, n_samples),
        8, 100
    ).round(2)
    
    response_time = np.clip(
        80 + (visitors_per_day / 150_000) ** 1.1 * 3500 + 
        heavy_plugin_count * 35 + rng.normal(0, 80, n_samples),
        40, 5000
    ).round(2)
    
    disk_read_iops = np.clip(
        15 + (pageviews_per_day / 900_000) ** 0.8 * 2500 + rng.normal(0, 25, n_samples),
        1, 3500
    ).round(2)
    
    disk_write_iops = np.clip(
        10 + (visitors_per_day / 150_000) * 2000 + plugin_count * 2.5 + rng.normal(0, 20, n_samples),
        1, 2800
    ).round(2)

    # ⭐ GÉNÉRATION DE LA CHARGE VARIABLE (TARGET) ⭐
    recommended_capacity_score = generate_dynamic_load_score(
        visitors=visitors_per_day,
        pageviews=pageviews_per_day,
        growth=traffic_growth_rate,
        plugin_count=plugin_count,
        heavy_plugins=heavy_plugin_count,
        php_version=php_version,
        cache=cache_enabled,
        cdn=cdn_enabled,
        wp_type=wp_type,
        rng=rng,
    )

    # Création du DataFrame
    data = pd.DataFrame(
        {
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
            "recommended_capacity_score": recommended_capacity_score,
        }
    )

    # Ajout des colonnes de plugins lourds
    for index, plugin_name in enumerate(HEAVY_PLUGIN_OPTIONS):
        data[f"heavy_plugin_{plugin_name}"] = heavy_plugin_matrix[:, index]

    # Colonne combinée des plugins lourds
    data["heavy_plugins"] = [
        ",".join(plugin for plugin, enabled in zip(HEAVY_PLUGIN_OPTIONS, row) if enabled)
        for row in heavy_plugin_matrix
    ]

    # Ordonnancement des colonnes
    ordered_columns = [
        "visitors_per_day",
        "pageviews_per_day",
        "traffic_growth_rate",
        "peak_hours_start",
        "peak_hours_end",
        "cpu_usage_avg",
        "cpu_usage_peak",
        "ram_usage_avg",
        "ram_usage_max",
        "disk_usage_avg",
        "disk_usage_max",
        "response_time",
        "disk_read_iops",
        "disk_write_iops",
        "plugin_count",
        "heavy_plugins",
        *[f"heavy_plugin_{name}" for name in HEAVY_PLUGIN_OPTIONS],
        "php_version",
        "cache_enabled",
        "cdn_enabled",
        "wp_type",
        "recommended_capacity_score",
    ]

    return data[ordered_columns]


# ============================================
# PRÉPARATION DES FEATURES
# ============================================

def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convertit les heures en minutes et ajoute la durée des pics"""
    prepared = frame.copy()

    for column in ("peak_hours_start", "peak_hours_end"):
        split_time = prepared[column].str.split(":", expand=True).astype(int)
        prepared[f"{column}_minutes"] = split_time[0] * 60 + split_time[1]

    prepared["peak_duration_minutes"] = (
        prepared["peak_hours_end_minutes"] - prepared["peak_hours_start_minutes"]
    ).clip(lower=0)

    return prepared.drop(columns=["peak_hours_start", "peak_hours_end", "heavy_plugins"])


def prepare_features(dataset: pd.DataFrame) -> tuple:
    """Prépare les features et la cible pour l'entraînement"""
    prepared = add_time_features(dataset)
    target = prepared.pop("recommended_capacity_score")

    features = pd.get_dummies(
        prepared,
        columns=["php_version", "cache_enabled", "cdn_enabled", "wp_type"],
        drop_first=False,
        dtype=int,
    )

    return features, target


# ============================================
# VISUALISATION
# ============================================

def normalize_feature_name(feature_name: str) -> str:
    """Normalise les noms de features pour le regroupement"""
    if feature_name.startswith("peak_hours_start"):
        return "peak_hours_start"
    if feature_name.startswith("peak_hours_end"):
        return "peak_hours_end"
    if feature_name.startswith("heavy_plugin_"):
        return "heavy_plugins"
    if feature_name.startswith("php_version_"):
        return "php_version"
    if feature_name.startswith("cache_enabled_"):
        return "cache_enabled"
    if feature_name.startswith("cdn_enabled_"):
        return "cdn_enabled"
    if feature_name.startswith("wp_type_"):
        return "wp_type"
    return feature_name


def save_tree_plot(model: XGBRegressor, tree_index: int, output_path: Path) -> None:
    """Sauvegarde le graphique d'un arbre de décision"""
    plt.figure(figsize=(80, 80))
    plot_tree(model, num_trees=tree_index, rankdir="LR", ax=plt.gca())
    tree_title = "Premier arbre XGBoost" if tree_index == 0 else "Dernier arbre XGBoost"
    plt.title(tree_title, fontsize=72, pad=40, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=1.5)
    plt.close()


def save_feature_importance(model: XGBRegressor, timestamp: str) -> Path:
    """Sauvegarde le graphique d'importance des features"""
    output_path = GRAPHE_DIR / f"feature_importance_{timestamp}.png"

    raw_scores = model.get_booster().get_score(importance_type="weight")
    grouped_scores = {feature: 0.0 for feature in FEATURE_ORDER}

    for raw_feature, score in raw_scores.items():
        feature = normalize_feature_name(raw_feature)
        if feature in grouped_scores:
            grouped_scores[feature] += float(score)

    plot_data = pd.DataFrame(
        {
            "feature": FEATURE_ORDER,
            "label": [FEATURE_LABELS[feature] for feature in FEATURE_ORDER],
            "f_score": [grouped_scores[feature] for feature in FEATURE_ORDER],
        }
    ).sort_values("f_score", ascending=True)

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(plot_data)))

    figure, axis = plt.subplots(figsize=(16, 12))
    bars = axis.barh(plot_data["label"], plot_data["f_score"], color=colors, height=0.72)

    for bar in bars:
        width = bar.get_width()
        axis.text(
            width + max(plot_data["f_score"].max() * 0.01, 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",
            va="center",
            fontsize=10,
            color="#333333",
        )

    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.set_facecolor("#FFFFFF")
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_xlabel("F-Score", fontsize=12)
    axis.set_ylabel("")
    axis.set_title("Feature Importance", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_residuals_graph(model, X, y_true, output_path):
    """Génère le graphique d'analyse des résidus"""
    y_pred = model.predict(X)
    residus = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].scatter(y_pred, residus, alpha=0.65, color="#2563eb", edgecolors="white", s=55)
    axes[0, 0].axhline(0, color="#ef4444", linestyle="--", linewidth=2)
    axes[0, 0].set_title("Résidus vs charge prédite", fontweight="bold")
    axes[0, 0].set_xlabel("Charge prédite")
    axes[0, 0].set_ylabel("Résidu")
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].hist(residus, bins=28, color="#10b981", edgecolor="white", alpha=0.82)
    axes[0, 1].axvline(0, color="#ef4444", linestyle="--", linewidth=2)
    axes[0, 1].set_title("Distribution des résidus", fontweight="bold")
    axes[0, 1].set_xlabel("Résidu")
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    axes[1, 0].scatter(y_true, y_pred, alpha=0.65, color="#8b5cf6", edgecolors="white", s=55)
    axes[1, 0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 
                    color="#ef4444", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Valeur réelle vs prédite", fontweight="bold")
    axes[1, 0].set_xlabel("Valeur réelle")
    axes[1, 0].set_ylabel("Valeur prédite")
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.08,
        0.55,
        f"MAE  : {mae:.2f}\nRMSE : {rmse:.2f}\nR²   : {r2:.3f}\nSamples : {len(y_pred)}",
        fontsize=16,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f1f5f9", alpha=0.95),
    )
    
    plt.suptitle("Analyse des résidus du modèle XGBoost", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_correlation_graph(X, output_path):
    """Génère la matrice de corrélation"""
    numeric_columns = X.select_dtypes(include=[np.number]).columns[:20]
    corr = X[numeric_columns].corr()
    
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.8,
        xticklabels=numeric_columns,
        yticklabels=numeric_columns,
        ax=ax,
        annot_kws={"size": 9},
    )
    
    ax.set_title("Matrice de corrélation des variables (jeu de test)", 
                 fontsize=18, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================

def train_model():
    """Fonction principale d'entraînement du modèle"""
    
    print("=" * 60)
    print("🚀 Démarrage de l'entraînement du modèle XGBoost...")
    print("📊 Charge variable selon les paramètres (non-linéaire)")
    print("=" * 60)
    
    # Création des répertoires
    ensure_directories()
    
    # Timestamp pour les fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Génération du dataset
    print(f"\n📊 Génération du dataset ({N_SAMPLES:,} échantillons)...")
    dataset = generate_training_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    print(f"✅ Dataset sauvegardé: {DATASET_PATH}")
    
    # Aperçu de la distribution de charge
    print(f"\n📈 Distribution de la charge cible:")
    print(f"   Min    : {dataset['recommended_capacity_score'].min():.2f}")
    print(f"   Max    : {dataset['recommended_capacity_score'].max():.2f}")
    print(f"   Moyenne: {dataset['recommended_capacity_score'].mean():.2f}")
    print(f"   Médiane: {dataset['recommended_capacity_score'].median():.2f}")
    
    # Préparation des features
    print("\n🔧 Préparation des features...")
    features, target = prepare_features(dataset)
    
    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    print(f"📈 Split: {len(x_train):,} train / {len(x_test):,} test")
    
    # Création du modèle
    print("\n🎯 Création du modèle XGBoost...")
    model = XGBRegressor(
        objective="reg:squarederror",
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
    )
    
    # Entraînement
    print("🔄 Entraînement en cours...")
    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_time
    print(f"✅ Entraînement terminé en {train_time:.1f}s")
    
    # Prédictions sur le jeu de test
    print("\n📊 Évaluation du modèle sur le jeu de test...")
    y_pred = model.predict(x_test)
    
    # Métriques de performance
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    relative_errors = np.abs(y_test.to_numpy() - y_pred) / np.maximum(np.abs(y_test.to_numpy()), 1e-8)
    accuracy = max(0.0, 100.0 * (1.0 - float(np.mean(relative_errors))))
    
    print(f"  ✓ RMSE     : {rmse:.4f}")
    print(f"  ✓ MSE      : {mse:.4f}")
    print(f"  ✓ MAE      : {mae:.4f}")
    print(f"  ✓ R²       : {r2:.4f}")
    print(f"  ✓ Accuracy : {accuracy:.2f}%")
    
    # Génération des graphiques
    print("\n📊 Génération des graphiques...")
    
    residuals_path = GRAPHE_DIR / f"residus_{timestamp}.png"
    plot_residuals_graph(model, x_test, y_test, residuals_path)
    print(f"  ✓ Graphique des résidus: {residuals_path}")
    
    correlation_path = GRAPHE_DIR / f"correlation_{timestamp}.png"
    plot_correlation_graph(x_test, correlation_path)
    print(f"  ✓ Matrice de corrélation: {correlation_path}")
    
    feature_importance_path = save_feature_importance(model, timestamp)
    print(f"  ✓ Feature importance: {feature_importance_path}")
    
    print("\n🌳 Génération des arbres de décision...")
    final_tree_index = max(0, model.get_booster().num_boosted_rounds() - 1)
    
    tree_0_path = GRAPHE_DIR / "tree_0.png"
    save_tree_plot(model, 0, tree_0_path)
    print(f"  ✓ Premier arbre: {tree_0_path}")
    
    tree_final_path = GRAPHE_DIR / "tree_final.png"
    save_tree_plot(model, final_tree_index, tree_final_path)
    print(f"  ✓ Dernier arbre: {tree_final_path}")
    
    print("\n📈 Génération de la courbe d'apprentissage...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        x_train,
        y_train,
        train_sizes=np.linspace(0.15, 1.0, 7),
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    
    train_mse = -train_scores.mean(axis=1)
    val_mse = -val_scores.mean(axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_sizes_abs, train_mse, "o-", color="#2563eb", linewidth=2.6, 
            markersize=8, label="Entraînement")
    ax.plot(train_sizes_abs, val_mse, "s-", color="#10b981", linewidth=2.6, 
            markersize=8, label="Validation")
    ax.fill_between(train_sizes_abs, train_mse, val_mse, color="#94a3b8", alpha=0.16)
    ax.set_title("Courbe d'apprentissage XGBoost", fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre d'échantillons")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    learning_curve_path = GRAPHE_DIR / f"learning_curve_{timestamp}.png"
    plt.savefig(learning_curve_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Courbe d'apprentissage: {learning_curve_path}")
    
    # ============================================
    # SAUVEGARDE DU .PKL UNIQUE
    # ============================================
    print("\n💾 Sauvegarde du modèle unique (model.pkl)...")
    
    model_package = {
        "model": model,
        "feature_columns": features.columns.tolist(),
        "target_column": "recommended_capacity_score",
        "heavy_plugin_options": HEAVY_PLUGIN_OPTIONS,
        "categorical_options": {
            "php_version": PHP_VERSIONS,
            "cache_enabled": ["oui", "non"],
            "cdn_enabled": ["oui", "non"],
            "wp_type": WP_TYPES,
        },
        "impact_reference": IMPACT_REFERENCE,
        "performance_metrics": {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "accuracy": round(accuracy, 2),
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
            "charge_generation": "dynamic_non_linear",  # Nouveau !
        },
    }
    
    joblib.dump(model_package, MODEL_PATH)
    print(f"  ✓ Modèle complet sauvegardé: {MODEL_PATH}")
    
    # Sauvegarde JSON additionnelle
    metrics_json = {
        "timestamp": timestamp,
        "samples": N_SAMPLES,
        "target_column": "recommended_capacity_score",
        "charge_generation": "dynamic_non_linear",
        "performance": {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "accuracy": f"{round(accuracy, 2)}%",
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
            "tree_0": str(tree_0_path),
            "tree_final": str(tree_final_path),
            "feature_importance": str(feature_importance_path),
            "residuals": str(residuals_path),
            "correlation": str(correlation_path),
            "learning_curve": str(learning_curve_path),
            "metrics_json": str(METRICS_PATH),
        },
    }
    
    METRICS_PATH.write_text(
        json.dumps(metrics_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ Métriques JSON sauvegardées: {METRICS_PATH}")
    
    # Résumé final
    print(f"""
    {'='*60}
    📁 FICHIERS GÉNÉRÉS:
    {'='*60}
    ├── Dataset: {DATASET_PATH}
    ├── ⭐ MODÈLE COMPLET: {MODEL_PATH} (unique .pkl)
    ├── Arbre 0: {tree_0_path}
    ├── Arbre final: {tree_final_path}
    ├── Feature importance: {feature_importance_path}
    ├── Résidus: {residuals_path}
    ├── Corrélation: {correlation_path}
    ├── Courbe d'apprentissage: {learning_curve_path}
    └── Métriques JSON: {METRICS_PATH}

    {'='*60}
    📊 PERFORMANCES (charge variable non-linéaire) :
    {'='*60}
    ├── RMSE     : {rmse:.4f}
    ├── MSE      : {mse:.4f}
    ├── MAE      : {mae:.4f}
    ├── R²       : {r2:.4f}  ⭐
    ├── Accuracy : {accuracy:.2f}%
    └── Temps    : {train_time:.1f}s

    {'='*60}
    🔍 EXEMPLE : COMMENT LA CHARGE VARIE :
    """)
{"="*60}
# Essaie ces commandes pour voir les variations :

# ```python
# import joblib
# import numpy as np

# # Charge le modèle
# package = joblib.load('{MODEL_PATH}')
# model = package['model']

# # Exemple 1: Petit site sans cache
# # visitors=1000, cpu=20%, ram=30%, plugins=5, PHP 7.4, pas de cache
# → Charge estimée: ~15-25

# # Exemple 2: Site e-commerce avec trafic
# # visitors=50000, cpu=80%, ram=75%, plugins=40, WooCommerce, PHP 8.2, cache ON
# → Charge estimée: ~65-85

# # Exemple 3: Site performance optimisé
# # visitors=100000, cpu=95%, ram=90%, plugins=60, Elementor, PHP 8.3, cache+CDN
# → Charge estimée: ~85-98



if __name__ == "__main__":
    train_model()
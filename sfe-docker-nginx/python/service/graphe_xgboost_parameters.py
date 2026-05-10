#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Patch
from sklearn.inspection import PartialDependenceDisplay

# ============================================
# CONFIGURATION MATPLOTLIB
# ============================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 180,
})

# ============================================
# VERIFICATION SHAP
# ============================================
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print(f"[OK] SHAP installe (version {shap.__version__})", file=sys.stderr)
except ImportError as e:
    print(f"[WARN] SHAP non disponible: {e}", file=sys.stderr)
except Exception as e:
    print(f"[WARN] Erreur chargement SHAP: {e}", file=sys.stderr)


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists("/app"):
    MODELS_DIR = Path("/app/service/models")
    DATA_DIR = Path("/app/Donnee_parametres")
else:
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR.parent / "Donnee_parametres"

MODEL_PATH = MODELS_DIR / "model.pkl"
OUTPUT_DIR = BASE_DIR / "analysis_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SATURATION_LIMIT = 90.0
DECISION_THRESHOLD = 65.0

HEAVY_PLUGIN_OPTIONS = [
    "woocommerce", "elementor", "wpml",
    "yoast", "revslider", "gravityforms",
]

PHP_VERSIONS = ["7.4", "8.0", "8.1", "8.2", "8.3"]
WP_TYPES = ["small", "medium", "performance"]

DEFAULTS = {
    "visitors_per_day": 5000, "pageviews_per_day": 150,
    "traffic_growth_rate": 15, "peak_hours_start": "09:00",
    "peak_hours_end": "18:00", "cpu_usage_avg": 45,
    "cpu_usage_peak": 75, "ram_usage_avg": 60,
    "ram_usage_max": 85, "disk_usage_avg": 45,
    "disk_usage_max": 70, "response_time": 350,
    "disk_read_iops": 120, "disk_write_iops": 80,
    "plugin_count": 25, "heavy_plugins": [],
    "php_version": "8.2", "cache_enabled": "oui",
    "cdn_enabled": "oui", "wp_type": "medium",
}

FRENCH_LABELS = {
    "visitors_per_day": "Visiteurs / jour",
    "pageviews_per_day": "Pages vues / jour",
    "traffic_growth_rate": "Taux de croissance",
    "peak_hours_start_minutes": "Pic debut",
    "peak_hours_end_minutes": "Pic fin",
    "peak_duration_minutes": "Duree pic",
    "cpu_usage_avg": "CPU moyen",
    "cpu_usage_peak": "CPU max",
    "ram_usage_avg": "RAM moyenne",
    "ram_usage_max": "RAM max",
    "disk_usage_avg": "Disque utilise",
    "disk_usage_max": "Disque max",
    "response_time": "Temps reponse",
    "disk_read_iops": "IOPS Read",
    "disk_write_iops": "IOPS Write",
    "plugin_count": "Nombre plugins",
}


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def load_model():
    if not MODEL_PATH.exists():
        return None, None, None
    payload = joblib.load(str(MODEL_PATH))
    if isinstance(payload, dict):
        return payload.get("model"), payload.get("feature_columns"), payload.get("scaler")
    return payload, None, None


def find_latest_json():
    if not DATA_DIR.exists():
        return None
    files = sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_params(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("parameters", data.get("params", data))


def to_float(value, default):
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clean_select(value, default):
    if value is None:
        return default
    value = str(value).strip()
    return default if value == "" or value.lower() == "none" else value


def time_to_minutes(value, default):
    value = clean_select(value, default)
    try:
        hour, minute = value.split(":")[:2]
        return int(hour) * 60 + int(minute)
    except Exception:
        hour, minute = default.split(":")
        return int(hour) * 60 + int(minute)


def normalize_heavy_plugins(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def normalize_params(params):
    return {
        "visitors_per_day": params.get("visitors_per_day"),
        "pageviews_per_day": params.get("pageviews_per_day"),
        "traffic_growth_rate": params.get("traffic_growth_rate"),
        "peak_hours_start": params.get("peak_hours_start"),
        "peak_hours_end": params.get("peak_hours_end"),
        "cpu_usage_avg": params.get("cpu_usage_avg"),
        "cpu_usage_peak": params.get("cpu_usage_peak"),
        "ram_usage_avg": params.get("ram_usage_avg"),
        "ram_usage_max": params.get("ram_usage_max"),
        "disk_usage_avg": params.get("disk_usage_avg"),
        "disk_usage_max": params.get("disk_usage_max"),
        "response_time": params.get("response_time"),
        "disk_read_iops": params.get("disk_read_iops"),
        "disk_write_iops": params.get("disk_write_iops"),
        "plugin_count": params.get("plugin_count"),
        "heavy_plugins": normalize_heavy_plugins(params.get("heavy_plugins")),
        "php_version": params.get("php_version"),
        "cache_enabled": params.get("cache_enabled"),
        "cdn_enabled": params.get("cdn_enabled"),
        "wp_type": params.get("wp_type"),
    }


def build_feature_row(normalized):
    heavy_plugins = set(normalized["heavy_plugins"])
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
        "peak_hours_start_minutes": time_to_minutes(normalized["peak_hours_start"], DEFAULTS["peak_hours_start"]),
        "peak_hours_end_minutes": time_to_minutes(normalized["peak_hours_end"], DEFAULTS["peak_hours_end"]),
    }
    row["peak_duration_minutes"] = max(0, row["peak_hours_end_minutes"] - row["peak_hours_start_minutes"])
    for plugin in HEAVY_PLUGIN_OPTIONS:
        row[f"heavy_plugin_{plugin}"] = 1 if plugin in heavy_plugins else 0
    for version in PHP_VERSIONS:
        row[f"php_version_{version}"] = 1 if normalized["php_version"] == version else 0
    for value in ["non", "oui"]:
        row[f"cache_enabled_{value}"] = 1 if normalized["cache_enabled"] == value else 0
        row[f"cdn_enabled_{value}"] = 1 if normalized["cdn_enabled"] == value else 0
    for wp_type in WP_TYPES:
        row[f"wp_type_{wp_type}"] = 1 if normalized["wp_type"] == wp_type else 0
    return row


def prepare_features(params, feature_columns):
    normalized = normalize_params(params)
    row = build_feature_row(normalized)
    features = pd.DataFrame([{column: row.get(column, 0) for column in feature_columns}])
    return features, normalized, row


def predict_load(model_load, features_df):
    return min(100.0, max(0.0, float(model_load.predict(features_df)[0])))


def days_to_months_days(days):
    if days is None or days >= 30000:
        return 999, 0, "infini"
    if days <= 0:
        return 0, 0, "SATURE"
    total_months = days / 30.44
    months = int(total_months)
    remaining_days = int(round((total_months - months) * 30.44))
    if remaining_days >= 30:
        months += 1
        remaining_days -= 30
    if months == 0:
        text = f"{remaining_days} jour{'s' if remaining_days > 1 else ''}"
    elif remaining_days == 0:
        text = f"{months} mois"
    else:
        text = f"{months} mois et {remaining_days} jour{'s' if remaining_days > 1 else ''}"
    return months, remaining_days, text


def make_simulated_dataset(base_features, feature_columns, n_samples=300):
    rng = np.random.default_rng(42)
    base = base_features.iloc[0].to_dict()
    rows = []
    binary_prefixes = ("heavy_plugin_", "php_version_", "cache_enabled_", "cdn_enabled_", "wp_type_")
    for _ in range(n_samples):
        row = {}
        for column in feature_columns:
            value = base.get(column, 0)
            if column.startswith(binary_prefixes):
                row[column] = int(value)
            else:
                value = float(value)
                noise = rng.normal(0, max(abs(value) * 0.12, 1.0))
                row[column] = max(0, value + noise)
        rows.append(row)
    return pd.DataFrame(rows, columns=feature_columns)


def label_for(feature):
    if feature.startswith("heavy_plugin_"):
        return "Plugin lourd"
    if feature.startswith("php_version_"):
        return "Version PHP"
    if feature.startswith("cache_enabled_"):
        return "Cache active"
    if feature.startswith("cdn_enabled_"):
        return "CDN active"
    if feature.startswith("wp_type_"):
        return "Pack WordPress"
    return FRENCH_LABELS.get(feature, feature)


def get_feature_labels(features_df):
    feature_labels = []
    for col in features_df.columns:
        if col.startswith("heavy_plugin_"):
            feature_labels.append(f"Plugin: {col.replace('heavy_plugin_', '')}")
        elif col.startswith("php_version_"):
            feature_labels.append(f"PHP {col.replace('php_version_', '')}")
        elif col.startswith("cache_enabled_"):
            feature_labels.append(f"Cache: {col.replace('cache_enabled_', '')}")
        elif col.startswith("cdn_enabled_"):
            feature_labels.append(f"CDN: {col.replace('cdn_enabled_', '')}")
        elif col.startswith("wp_type_"):
            feature_labels.append(f"Type: {col.replace('wp_type_', '')}")
        else:
            feature_labels.append(FRENCH_LABELS.get(col, col))
    return feature_labels


# ============================================
# GRAPHE 1 : PARTIAL DEPENDENCE PLOT
# ============================================

def graph_partial_dependence(model_load, features_df, feature_columns):
    try:
        numeric_features = [
            f for f in feature_columns
            if not f.startswith(("heavy_plugin_", "php_version_", "cache_enabled_", "cdn_enabled_", "wp_type_"))
        ][:20]

        sim_df = make_simulated_dataset(features_df, feature_columns, n_samples=350)
        n_cols = 4
        n_rows = math.ceil(len(numeric_features) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, max(6, n_rows * 4.6)))
        axes = np.array(axes).reshape(-1)

        for index, feature in enumerate(numeric_features):
            feature_index = feature_columns.index(feature)
            ax = axes[index]
            try:
                PartialDependenceDisplay.from_estimator(
                    model_load, sim_df,
                    features=[feature_index],
                    feature_names=feature_columns,
                    grid_resolution=35, ax=ax,
                    line_kw={"color": "#2563eb", "linewidth": 2.4},
                )
                current_value = float(features_df.iloc[0][feature])
                ax.axvline(current_value, color="#ef4444", linestyle="--", linewidth=1.8)
                ax.set_title(label_for(feature), fontsize=11, fontweight="bold")
                ax.set_ylabel("Charge serveur" if index % n_cols == 0 else "")
                ax.grid(True, alpha=0.25)
            except Exception:
                ax.text(0.5, 0.5, f"{label_for(feature)}\nindisponible", ha="center", va="center")
                ax.set_title(label_for(feature), fontsize=11, fontweight="bold")
                ax.axis("off")

        for index in range(len(numeric_features), len(axes)):
            axes[index].axis("off")

        plt.suptitle("Partial Dependence Plot - Effet des variables sur la charge serveur",
                     fontsize=18, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        path = str(OUTPUT_DIR / f"partial_dependence_all_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"[ERREUR] PDP: {exc}", file=sys.stderr)
        return None


# ============================================
# GRAPHE 2 : SATURATION EVOLUTION
# ============================================

def graph_saturation_evolution(model_load, features_df, normalized, feature_columns):
    try:
        current_load = predict_load(model_load, features_df)
        growth_rate_percent = max(0.0, float(normalized["traffic_growth_rate"]))

        if current_load >= 90:
            saturation_days = 0.0
        elif growth_rate_percent <= 0:
            saturation_days = 999.0 * 30.44
        else:
            sat_months = float(np.log(90 / max(1.0, current_load)) / np.log(1 + growth_rate_percent / 100))
            saturation_days = max(0.0, sat_months * 30.44)

        _, _, text_sat = days_to_months_days(saturation_days)

        months_projection = 24
        months = np.arange(0, months_projection + 1)
        charges = []
        current_load_proj = current_load
        for _ in months:
            charges.append(min(100, current_load_proj))
            if current_load_proj < 100:
                current_load_proj *= (1 + growth_rate_percent / 100)

        charges = np.array(charges)
        sat_idx = np.where(charges >= SATURATION_LIMIT)[0]
        saturation_month = int(sat_idx[0]) if len(sat_idx) > 0 else None

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(months, charges, color="#2563eb", linewidth=3, marker="o", markersize=6,
                label="Charge serveur")
        ax.axhline(SATURATION_LIMIT, color="#ef4444", linestyle="--", linewidth=2.5,
                   label=f"Seuil saturation {SATURATION_LIMIT}%")
        ax.fill_between(months, charges, SATURATION_LIMIT, where=charges >= SATURATION_LIMIT,
                        color="#ef4444", alpha=0.18)
        ax.fill_between(months, charges, 0, color="#2563eb", alpha=0.08)

        if saturation_month is not None:
            ax.axvline(saturation_month, color="#f97316", linestyle=":", linewidth=2.5)
            ax.scatter([saturation_month], [charges[saturation_month]], s=180,
                      color="#f97316", edgecolor="white", zorder=5)
            ax.text(saturation_month, min(100, charges[saturation_month] + 4),
                    f"Saturation\n{text_sat}", ha="center", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="#fff7ed", edgecolor="#f97316", alpha=0.95))

        labels = [f"{m} mois\n{int(m * 30.44)} jours" for m in months]
        ax.set_title("Evolution de la charge serveur jusqu'a saturation", fontsize=17, fontweight="bold")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Charge serveur (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(months[::3])
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 3)], fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')
        plt.tight_layout()

        path = str(OUTPUT_DIR / f"saturation_evolution_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"[ERREUR] Saturation: {exc}", file=sys.stderr)
        return None


# ============================================
# GRAPHE 3 : CHARGE PAR TYPE DE SITE
# ============================================



# ============================================
# GRAPHE 4 : SERIES TEMPORELLES
# ============================================

def graph_time_series(params):
    try:
        visitors = float(params.get("visitors_per_day", DEFAULTS["visitors_per_day"]))
        cpu_avg = float(params.get("cpu_usage_avg", DEFAULTS["cpu_usage_avg"]))
        ram_avg = float(params.get("ram_usage_avg", DEFAULTS["ram_usage_avg"]))
        response_time = float(params.get("response_time", DEFAULTS["response_time"]))
        growth_rate = float(params.get("traffic_growth_rate", DEFAULTS["traffic_growth_rate"])) / 100

        today = datetime.now()
        dates = pd.date_range(start=today - timedelta(days=90),
                              end=today + timedelta(days=90), freq='D')
        today_idx = 90
        n = len(dates)
        np.random.seed(42)

        va, ca, ra, rta = [], [], [], []
        for i in range(n):
            g = (1 + growth_rate) ** ((i - today_idx) / 30)
            noise = np.random.normal(0, 0.08)
            va.append(max(50, visitors * g * (1 + noise)))
            ca.append(np.clip(cpu_avg * g * (1 + noise * 0.7), 5, 100))
            ra.append(np.clip(ram_avg * g * (1 + noise * 0.5), 8, 100))
            rta.append(np.clip(response_time * g * (1 + noise * 1.2), 40, 5000))

        va = np.array(va)
        ca = np.array(ca)
        ra = np.array(ra)
        rta = np.array(rta)

        g30, g90 = [], []
        for i in range(n):
            g30.append(((va[i] - va[i-30]) / va[i-30]) * 100 if i >= 30 else 0)
            g90.append(((va[i] - va[i-90]) / va[i-90]) * 100 if i >= 90 else 0)

        fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)

        axes[0].plot(dates, va, color='#2563eb', linewidth=2, label='Visiteurs/jour')
        axes[0].axvline(x=dates[today_idx], color='red', linestyle='--', alpha=0.7, label="Aujourd'hui")
        axes[0].fill_between(dates[today_idx:], va[today_idx:], alpha=0.1, color='#2563eb', label='Projection')
        axes[0].set_ylabel('Visiteurs', fontweight='bold')
        axes[0].set_title('Visiteurs par jour (historique + projection 90 jours)', fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].grid(alpha=0.3)

        axes[1].plot(dates, ca, color='#ef4444', linewidth=2, label='CPU moyen (%)')
        axes[1].axvline(x=dates[today_idx], color='red', linestyle='--', alpha=0.7)
        axes[1].axhline(y=90, color='darkred', linestyle='--', alpha=0.5, label='Saturation (90%)')
        axes[1].axhline(y=75, color='orange', linestyle=':', alpha=0.5, label='Alerte (75%)')
        axes[1].fill_between(dates[today_idx:], ca[today_idx:], alpha=0.1, color='#ef4444')
        axes[1].set_ylabel('CPU (%)', fontweight='bold')
        axes[1].set_title('CPU moyen (historique + projection 90 jours)', fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(alpha=0.3)

        axes[2].plot(dates, ra, color='#f59e0b', linewidth=2, label='RAM moyenne (%)')
        axes[2].axvline(x=dates[today_idx], color='red', linestyle='--', alpha=0.7)
        axes[2].axhline(y=90, color='darkred', linestyle='--', alpha=0.5, label='Saturation (90%)')
        axes[2].axhline(y=75, color='orange', linestyle=':', alpha=0.5, label='Alerte (75%)')
        axes[2].fill_between(dates[today_idx:], ra[today_idx:], alpha=0.1, color='#f59e0b')
        axes[2].set_ylabel('RAM (%)', fontweight='bold')
        axes[2].set_title('RAM moyenne (historique + projection 90 jours)', fontweight='bold')
        axes[2].legend(loc='upper left')
        axes[2].grid(alpha=0.3)

        axes[3].plot(dates, rta, color='#8b5cf6', linewidth=2, label='Temps de reponse (ms)')
        axes[3].axvline(x=dates[today_idx], color='red', linestyle='--', alpha=0.7)
        axes[3].axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='Limite acceptable (1000ms)')
        axes[3].fill_between(dates[today_idx:], rta[today_idx:], alpha=0.1, color='#8b5cf6')
        axes[3].set_ylabel('ms', fontweight='bold')
        axes[3].set_title('Temps de reponse (historique + projection 90 jours)', fontweight='bold')
        axes[3].legend(loc='upper left')
        axes[3].grid(alpha=0.3)

        axes[4].plot(dates, g30, color='#10b981', linewidth=2, label='Croissance moyenne 30j (%)')
        axes[4].plot(dates, g90, color='#6366f1', linewidth=2, label='Croissance moyenne 90j (%)')
        axes[4].axvline(x=dates[today_idx], color='red', linestyle='--', alpha=0.7)
        axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[4].set_ylabel('Croissance (%)', fontweight='bold')
        axes[4].set_xlabel('Date', fontweight='bold')
        axes[4].set_title('Croissance du trafic (moyennes mobiles 30j et 90j)', fontweight='bold')
        axes[4].legend(loc='upper left')
        axes[4].grid(alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        fig.suptitle('Series Temporelles - Tendances et Projections sur 180 jours',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        path = str(OUTPUT_DIR / f"time_series_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()
        return path
    except Exception as exc:
        print(f"[ERREUR] Time series: {exc}", file=sys.stderr)
        return None


# ============================================
# GRAPHE 5 : SHAP BAR PLOT + JAUGE
# ============================================

def graph_shap_force_plot(model_load, features_df, feature_columns, normalized):
    """
    SHAP - Bar Plot + Jauge (sans Waterfall, sans resume)
    """
    if not SHAP_AVAILABLE:
        print("[WARN] SHAP non disponible", file=sys.stderr)
        return _graph_shap_missing()

    try:
        import shap

        explainer = shap.TreeExplainer(model_load)
        shap_values = explainer.shap_values(features_df)
        current_prediction = model_load.predict(features_df)[0]
        feature_labels = get_feature_labels(features_df)

        if current_prediction >= 85:
            etat, etat_color = "CRITIQUE", '#ef4444'
        elif current_prediction >= 75:
            etat, etat_color = "URGENT", '#f97316'
        elif current_prediction >= 65:
            etat, etat_color = "SURVEILLANCE", '#eab308'
        else:
            etat, etat_color = "OPTIMAL", '#10b981'

        fig = plt.figure(figsize=(20, 10))

        # ---- Bar Plot (gauche) ----
        ax1 = fig.add_subplot(1, 2, 1)
        shap_df = pd.DataFrame({'feature': feature_labels, 'shap_value': shap_values[0]})
        shap_df['impact_abs'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('impact_abs', ascending=True)
        shap_df_plot = shap_df.tail(15)

        colors = ['#ef4444' if x < 0 else '#10b981' for x in shap_df_plot['shap_value']]
        bars = ax1.barh(shap_df_plot['feature'], shap_df_plot['shap_value'],
                        color=colors, alpha=0.85, edgecolor='white', linewidth=1)

        for bar, val in zip(bars, shap_df_plot['shap_value']):
            if val > 0:
                ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        f'+{val:.2f}', va='center', fontsize=9, fontweight='bold')
            else:
                ax1.text(val - 0.3, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', va='center', fontsize=9, fontweight='bold', ha='right')

        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        ax1.set_xlabel('Contribution SHAP', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Features - Impact sur la prediction\nVert: Augmente | Rouge: Reduit',
                     fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')

        # ---- Jauge (droite) ----
        ax2 = fig.add_subplot(1, 2, 2)
        sectors = [
            (0, 65, '#10b981', 'OPTIMAL'),
            (65, 75, '#eab308', 'SURVEILLANCE'),
            (75, 85, '#f97316', 'URGENT'),
            (85, 100, '#ef4444', 'CRITIQUE'),
        ]
        for start, end, color, label in sectors:
            wedge = Wedge((0, 0), 1, start * 3.6, end * 3.6, width=0.3, color=color, alpha=0.6)
            ax2.add_patch(wedge)

        angle = current_prediction * 3.6
        ax2.arrow(0, 0, 0.7 * np.cos(np.radians(90 - angle)),
                 0.7 * np.sin(np.radians(90 - angle)),
                 head_width=0.08, head_length=0.1, fc='black', ec='black', linewidth=2)

        circle = Circle((0, 0), 0.15, color='white', ec='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(0, -0.05, f'{current_prediction:.1f}%', ha='center', va='center',
                fontsize=18, fontweight='bold', color=etat_color)
        ax2.text(0, -0.25, etat, ha='center', va='center', fontsize=11,
                fontweight='bold', color=etat_color)

        for start, end, color, label in sectors:
            mid = (start + end) / 2
            ang = np.radians(90 - mid * 3.6)
            ax2.text(1.25 * np.cos(ang), 1.25 * np.sin(ang), label,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color=color, rotation=mid * 3.6 - 90)

        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Jauge de Charge Serveur\nCharge predite: {current_prediction:.1f}%',
                     fontsize=12, fontweight='bold')

        fig.suptitle('SHAP - Explication Locale de la Prediction XGBoost',
                     fontsize=16, fontweight='bold', y=1.02)
        fig.text(0.5, 0.01, 'SHAP (SHapley Additive exPlanations)',
                ha='center', fontsize=9, color='#64748b', style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        path = str(OUTPUT_DIR / f"shap_force_plot_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()
        return path

    except Exception as exc:
        print(f"[ERREUR] SHAP: {exc}", file=sys.stderr)
        return _graph_shap_missing()


def _graph_shap_missing():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    message = (
        "Module SHAP non installe\n\n"
        "Pour generer le SHAP Plot :\n"
        "  pip install shap\n\n"
        "Le SHAP Plot permet de :\n"
        "  - Expliquer UNE decision precise\n"
        "  - Identifier les parametres les plus impactants\n"
        "  - Justifier les recommandations (transparence IA)"
    )
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=13,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fef3c7', edgecolor='#f59e0b', alpha=0.9))
    ax.set_title('SHAP Plot - Module Manquant', fontsize=16, fontweight='bold')

    path = str(OUTPUT_DIR / f"shap_force_plot_missing_{TIMESTAMP}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


# ============================================
# GRAPHE 6 : DECISION BOUNDARY EVOLUTION PLOT
# ============================================

def graph_decision_boundary(model_load, features_df, normalized, feature_columns):
    """
    GRAPHE 6 : Frontiere de decision evolutive XGBoost
    Montre le point exact ou la decision change (Upgrade / No Upgrade)
    """
    try:
        current_visitors = float(normalized.get("visitors_per_day", DEFAULTS["visitors_per_day"]))
        current_cpu = float(normalized.get("cpu_usage_avg", DEFAULTS["cpu_usage_avg"]))
        growth_rate = float(normalized.get("traffic_growth_rate", DEFAULTS["traffic_growth_rate"]))
        plugin_count = float(normalized.get("plugin_count", DEFAULTS["plugin_count"]))

        n_points = 80
        visitors_range = np.linspace(100, 150000, n_points)
        cpu_range = np.linspace(5, 100, n_points)

        XX, YY = np.meshgrid(visitors_range, cpu_range)
        base_row = features_df.iloc[0].to_dict()
        ZZ = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(n_points):
                row_copy = base_row.copy()
                row_copy["visitors_per_day"] = XX[i, j]
                row_copy["cpu_usage_avg"] = YY[i, j]
                cpu_ratio = YY[i, j] / max(1, current_cpu) if current_cpu > 0 else 1
                row_copy["ram_usage_avg"] = min(100, base_row.get("ram_usage_avg", 60) * cpu_ratio)
                df_point = pd.DataFrame([row_copy])
                ZZ[i, j] = predict_load(model_load, df_point)

        fig, ax = plt.subplots(figsize=(16, 10))

        # Zones colorees
        ax.contourf(XX, YY, ZZ, levels=[0, DECISION_THRESHOLD, 100],
                    colors=['#10b981', '#ef4444'], alpha=0.25)

        # Isolignes
        contour_levels = [25, 40, 55, 65, 75, 85, 95]
        contours = ax.contour(XX, YY, ZZ, levels=contour_levels,
                             colors='black', linewidths=0.8, alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f%%')

        # Frontiere de decision
        ax.contour(XX, YY, ZZ, levels=[DECISION_THRESHOLD],
                  colors='#f97316', linewidths=3, linestyles='-')

        # Point actuel
        ax.scatter([current_visitors], [current_cpu], s=300,
                  color='#2563eb', edgecolor='white', linewidth=3, zorder=10)

        current_load = predict_load(model_load, features_df)
        if current_load >= 85:
            point_status, point_color = "CRITIQUE", '#ef4444'
        elif current_load >= 75:
            point_status, point_color = "URGENT", '#f97316'
        elif current_load >= 65:
            point_status, point_color = "SURVEILLANCE", '#eab308'
        else:
            point_status, point_color = "STABLE", '#10b981'

        ax.annotate(
            f'Position actuelle\n{current_visitors:.0f} vis/j, CPU {current_cpu:.1f}%\n'
            f'Charge: {current_load:.1f}% ({point_status})',
            xy=(current_visitors, current_cpu),
            xytext=(current_visitors + 15000, current_cpu + 15),
            fontsize=11, fontweight='bold', color=point_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=point_color, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color=point_color, lw=2)
        )

        # Fleche de projection
        growth_factor = 1 + growth_rate / 100
        future_visitors = min(150000, current_visitors * (growth_factor ** 12))

        ax.annotate('', xy=(future_visitors, current_cpu * 1.1),
                   xytext=(current_visitors, current_cpu),
                   arrowprops=dict(arrowstyle='->', color='#8b5cf6', lw=2.5,
                                 connectionstyle='arc3,rad=0.3'))

        ax.text(future_visitors + 3000, current_cpu * 1.1 + 3,
                f'Projection 12 mois\n{future_visitors:.0f} vis/j\nCroissance: {growth_rate:.1f}%',
                fontsize=9, color='#8b5cf6', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#f3e8ff', alpha=0.8))

        # Zones annotees
        ax.text(2000, 15, 'ZONE STABLE\nPas d\'upgrade necessaire',
               fontsize=12, fontweight='bold', color='#10b981',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='#10b981', alpha=0.9))

        ax.text(120000, 88, 'ZONE CRITIQUE\nUpgrade recommande',
               fontsize=12, fontweight='bold', color='#ef4444',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='#ef4444', alpha=0.9))

        ax.annotate(f'Frontiere de decision\nSeuil: {DECISION_THRESHOLD:.0f}%',
                   xy=(60000, 55), fontsize=11, fontweight='bold',
                   color='#f97316',
                   bbox=dict(boxstyle='round', facecolor='#fff7ed',
                            edgecolor='#f97316', alpha=0.9))

        ax.set_xlabel('Visiteurs par jour', fontsize=13, fontweight='bold')
        ax.set_ylabel('CPU moyen (%)', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 150000)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2, linestyle='--')

        ax.set_title(
            f'Frontiere de Decision XGBoost\n'
            f'Visiteurs/jour vs CPU moyen - Seuil Upgrade: {DECISION_THRESHOLD:.0f}%\n'
            f'Pack: {normalized.get("wp_type", "medium").upper()} | '
            f'Plugins: {plugin_count:.0f} | '
            f'Cache: {normalized.get("cache_enabled", "oui").upper()} | '
            f'CDN: {normalized.get("cdn_enabled", "oui").upper()}',
            fontsize=14, fontweight='bold'
        )

        legend_elements = [
            Patch(facecolor='#10b981', alpha=0.25, label='Zone Stable (< 65%)'),
            Patch(facecolor='#ef4444', alpha=0.25, label='Zone Critique (>= 65%)'),
            plt.Line2D([0], [0], color='#f97316', linewidth=3, label=f'Frontiere ({DECISION_THRESHOLD:.0f}%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2563eb',
                      markersize=12, label='Position actuelle'),
            plt.Line2D([0], [0], color='#8b5cf6', linewidth=2.5, label='Projection croissance'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
                 framealpha=0.9, edgecolor='#94a3b8')

        info_text = (
            f"Parametres influencant la frontiere:\n"
            f"  Growth: {growth_rate:.1f}% | Plugins: {plugin_count:.0f} | "
            f"Heavy Plugins: {len(normalized.get('heavy_plugins', []))}"
        )
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=9,
                color='#64748b', style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        path = str(OUTPUT_DIR / f"decision_boundary_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()

        return path

    except Exception as exc:
        print(f"[ERREUR] Decision Boundary: {exc}", file=sys.stderr)
        return None


# ============================================
# GENERATION DE TOUS LES GRAPHIQUES
# ============================================

def generate_all_graphs():
    model_load, feature_columns, scaler = load_model()

    if model_load is None:
        return {"status": "error", "message": f"Modele introuvable: {MODEL_PATH}"}

    if not feature_columns:
        if hasattr(model_load, "feature_names_in_"):
            feature_columns = list(model_load.feature_names_in_)
        else:
            return {"status": "error", "message": "feature_columns introuvable dans model.pkl"}

    json_file = find_latest_json()
    if json_file is None:
        return {"status": "error", "message": f"Aucun fichier JSON dans {DATA_DIR}"}

    params = load_params(json_file)
    features_df, normalized, _ = prepare_features(params, feature_columns)
    current_load = predict_load(model_load, features_df)

    graphs, errors = {}, {}

    generators = {
        "partial_dependence": lambda: graph_partial_dependence(model_load, features_df, feature_columns),
        "saturation_evolution": lambda: graph_saturation_evolution(model_load, features_df, normalized, feature_columns),
        "time_series": lambda: graph_time_series(normalized),
        "shap_force_plot": lambda: graph_shap_force_plot(model_load, features_df, feature_columns, normalized),
        "decision_boundary": lambda: graph_decision_boundary(model_load, features_df, normalized, feature_columns),
    }

    for name, gen in generators.items():
        try:
            path = gen()
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
        "normalized_parameters": normalized,
    }


if __name__ == "__main__":
    result = generate_all_graphs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graphiques d'analyse Vala Bleu - 4 Graphiques Dynamiques
1. Radar des Ressources (équilibre global)
2. Jauges de Saturation (niveau de criticité)
3. Impact des Features (barres divergentes positif/négatif)
4. Courbe de Dégradation Temporelle (projection 3 scénarios)
"""

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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Patch, FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

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
    'savefig.dpi': 150,
})

# ============================================
# CHEMINS ET CONFIGURATION
# ============================================

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

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def load_model():
    if not MODEL_PATH.exists():
        return None, None
    payload = joblib.load(str(MODEL_PATH))
    if isinstance(payload, dict):
        return payload.get("model"), payload.get("feature_columns")
    return payload, None


def find_latest_json():
    if not DATA_DIR.exists():
        return None
    files = sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_params(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("parameters", data.get("params", data))


def normalize_heavy_plugins(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    return []


def normalize_params(params):
    return {
        "visitors_per_day": float(params.get("visitors_per_day", DEFAULTS["visitors_per_day"])),
        "pageviews_per_day": float(params.get("pageviews_per_day", DEFAULTS["pageviews_per_day"])),
        "traffic_growth_rate": float(params.get("traffic_growth_rate", DEFAULTS["traffic_growth_rate"])),
        "cpu_usage_avg": float(params.get("cpu_usage_avg", DEFAULTS["cpu_usage_avg"])),
        "cpu_usage_peak": float(params.get("cpu_usage_peak", DEFAULTS["cpu_usage_peak"])),
        "ram_usage_avg": float(params.get("ram_usage_avg", DEFAULTS["ram_usage_avg"])),
        "ram_usage_max": float(params.get("ram_usage_max", DEFAULTS["ram_usage_max"])),
        "disk_usage_avg": float(params.get("disk_usage_avg", DEFAULTS["disk_usage_avg"])),
        "disk_usage_max": float(params.get("disk_usage_max", DEFAULTS["disk_usage_max"])),
        "response_time": float(params.get("response_time", DEFAULTS["response_time"])),
        "disk_read_iops": float(params.get("disk_read_iops", DEFAULTS["disk_read_iops"])),
        "disk_write_iops": float(params.get("disk_write_iops", DEFAULTS["disk_write_iops"])),
        "plugin_count": float(params.get("plugin_count", DEFAULTS["plugin_count"])),
        "heavy_plugins": normalize_heavy_plugins(params.get("heavy_plugins", [])),
        "php_version": str(params.get("php_version", DEFAULTS["php_version"])),
        "cache_enabled": str(params.get("cache_enabled", DEFAULTS["cache_enabled"])),
        "cdn_enabled": str(params.get("cdn_enabled", DEFAULTS["cdn_enabled"])),
        "wp_type": str(params.get("wp_type", DEFAULTS["wp_type"])),
    }


def time_to_minutes(value, default):
    try:
        if not value or value == "none":
            raise ValueError("Valeur none")
        hour, minute = str(value).split(":")[:2]
        return int(hour) * 60 + int(minute)
    except Exception:
        hour, minute = default.split(":")
        return int(hour) * 60 + int(minute)


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
        "peak_hours_start_minutes": time_to_minutes(normalized.get("peak_hours_start"), DEFAULTS["peak_hours_start"]),
        "peak_hours_end_minutes": time_to_minutes(normalized.get("peak_hours_end"), DEFAULTS["peak_hours_end"]),
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
    return features, normalized


def predict_load(model_load, features_df):
    try:
        return min(100.0, max(0.0, float(model_load.predict(features_df)[0])))
    except Exception:
        return 50.0


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


# ============================================
# GRAPHIQUE 1 : RADAR DES RESSOURCES
# ============================================

def graph_radar_resources(normalized, current_load):
    """Radar chart montrant l'equilibre global du serveur"""
    try:
        categories = ['CPU', 'RAM', 'Disque', 'Trafic', 'Plugins', 'Optimisation']
        N = len(categories)
        
        cpu_score = (normalized["cpu_usage_avg"] + normalized["cpu_usage_peak"]) / 2
        ram_score = (normalized["ram_usage_avg"] + normalized["ram_usage_max"]) / 2
        disk_score = (normalized["disk_usage_avg"] + normalized["disk_usage_max"]) / 2
        traffic_score = min(100, (normalized["visitors_per_day"] / 150000) * 100)
        
        heavy_count = len(normalized["heavy_plugins"])
        plugin_score = min(100, (normalized["plugin_count"] / 80) * 100 + heavy_count * 8)
        
        opt_score = 0
        if normalized["cache_enabled"] == "oui":
            opt_score += 40
        if normalized["cdn_enabled"] == "oui":
            opt_score += 30
        php_version = normalized["php_version"]
        if php_version in ["8.2", "8.3"]:
            opt_score += 30
        elif php_version in ["8.0", "8.1"]:
            opt_score += 15
        opt_score = 100 - opt_score
        
        values = [cpu_score, ram_score, disk_score, traffic_score, plugin_score, opt_score]
        values += values[:1]
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        ax.fill_between([0, 2*np.pi], 0, 40, color='#10b981', alpha=0.08)
        ax.fill_between([0, 2*np.pi], 40, 70, color='#f59e0b', alpha=0.08)
        ax.fill_between([0, 2*np.pi], 70, 100, color='#ef4444', alpha=0.08)
        
        for r in [25, 50, 75, 100]:
            ax.plot([0, 2*np.pi], [r, r], color='#94a3b8', linewidth=0.5, alpha=0.3)
        
        ax.fill(angles, values, color='#2563eb', alpha=0.25)
        ax.plot(angles, values, color='#2563eb', linewidth=2.5, marker='o', markersize=8)
        
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            color = '#10b981' if value < 40 else '#f59e0b' if value < 70 else '#ef4444'
            ax.plot(angle, value, 'o', color=color, markersize=12, zorder=5)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color='#64748b')
        
        legend_elements = [
            Patch(facecolor='#10b981', alpha=0.3, label='Optimal (< 40%)'),
            Patch(facecolor='#f59e0b', alpha=0.3, label='Surveillance (40-70%)'),
            Patch(facecolor='#ef4444', alpha=0.3, label='Critique (> 70%)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        if current_load >= 85:
            status, status_color = "CRITIQUE", '#ef4444'
        elif current_load >= 65:
            status, status_color = "SURVEILLANCE", '#f59e0b'
        else:
            status, status_color = "OPTIMAL", '#10b981'
        
        ax.set_title(f'Radar des Ressources\nCharge predite: {current_load:.1f}% - {status}',
                    fontsize=16, fontweight='bold', color=status_color, pad=25)
        
        plt.tight_layout()
        path = str(OUTPUT_DIR / f"radar_resources_{TIMESTAMP}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return path
    except Exception as exc:
        print(f"[ERREUR] Radar: {exc}", file=sys.stderr)
        return None


# ============================================
# GRAPHIQUE 2 : JAUGES DE SATURATION
# ============================================

def graph_gauges_saturation(normalized, current_load):
    """6 jauges semi-circulaires montrant le niveau de criticite"""
    try:
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        gauges_config = [
            {
                "title": "CPU",
                "value": normalized["cpu_usage_avg"],
                "max_val": 100,
                "thresholds": [60, 80],
                "ax": fig.add_subplot(gs[0, 0]),
                "color_green": '#10b981', "color_yellow": '#f59e0b', "color_red": '#ef4444',
            },
            {
                "title": "RAM",
                "value": normalized["ram_usage_avg"],
                "max_val": 100,
                "thresholds": [60, 80],
                "ax": fig.add_subplot(gs[0, 1]),
                "color_green": '#10b981', "color_yellow": '#f59e0b', "color_red": '#ef4444',
            },
            {
                "title": "Disque",
                "value": normalized["disk_usage_avg"],
                "max_val": 100,
                "thresholds": [50, 75],
                "ax": fig.add_subplot(gs[0, 2]),
                "color_green": '#10b981', "color_yellow": '#f59e0b', "color_red": '#ef4444',
            },
            {
                "title": "Trafic\n(Croissance %)",
                "value": min(100, normalized["traffic_growth_rate"] * 2),
                "max_val": 100,
                "thresholds": [30, 60],
                "ax": fig.add_subplot(gs[1, 0]),
                "color_green": '#10b981', "color_yellow": '#f59e0b', "color_red": '#ef4444',
                "display_value": f"{normalized['traffic_growth_rate']:.1f}%"
            },
            {
                "title": "Plugins",
                "value": min(100, (normalized["plugin_count"] / 50) * 100),
                "max_val": 100,
                "thresholds": [30, 60],
                "ax": fig.add_subplot(gs[1, 1]),
                "color_green": '#10b981', "color_yellow": '#f59e0b', "color_red": '#ef4444',
                "display_value": f"{int(normalized['plugin_count'])}"
            },
            {
                "title": "Charge\nPredite",
                "value": current_load,
                "max_val": 100,
                "thresholds": [50, 75],
                "ax": fig.add_subplot(gs[1, 2]),
                "color_green": '#10b981', "color_yellow": '#f59e0b', "color_red": '#ef4444',
            },
        ]
        
        for config in gauges_config:
            ax = config["ax"]
            value = config["value"]
            thresholds = config["thresholds"]
            # Bornage de la valeur entre 0 et 100
            value = max(0, min(100, value))
            for start, end, color, alpha in [
                (0, thresholds[0], config["color_green"], 0.8),
                (thresholds[0], thresholds[1], config["color_yellow"], 0.8),
                (thresholds[1], 100, config["color_red"], 0.8),
            ]:
                start_angle = np.pi * (1 - start / 100)
                end_angle = np.pi * (1 - end / 100)
                theta_arc = np.linspace(start_angle, end_angle, 50)
                ax.fill_between(theta_arc, 0.5, 1.0, color=color, alpha=alpha)

            # Suppression de l'aiguille/flèche : rien à tracer ici

            # Cercle central
            circle = Circle((0, 0), 0.12, color='white', ec='#1e293b', linewidth=2, zorder=10)
            ax.add_patch(circle)

            display = config.get("display_value", f"{value:.1f}%")
            color = config["color_green"] if value < thresholds[0] else config["color_yellow"] if value < thresholds[1] else config["color_red"]
            ax.text(0, -0.25, display, ha='center', va='center', fontsize=14, fontweight='bold', color=color)

            ax.text(-1.1, 0.5, '0%', fontsize=8, color='#64748b')
            ax.text(1.1, 0.5, '100%', fontsize=8, color='#64748b')

            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.4, 1.3)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(config["title"], fontsize=12, fontweight='bold', pad=10)
        
        fig.suptitle('Jauges de Saturation - Niveau de Criticite par Ressource',
                    fontsize=18, fontweight='bold', y=1.02)
        
        path = str(OUTPUT_DIR / f"gauges_saturation_{TIMESTAMP}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return path
    except Exception as exc:
        print(f"[ERREUR] Jauges: {exc}", file=sys.stderr)
        return None


# ============================================
# GRAPHIQUE 3 : IMPACT DES FEATURES
# ============================================

def graph_feature_impact(normalized, current_load):
    """Barres horizontales divergentes montrant l'impact positif/negatif"""
    try:
        impacts = []
        
        # Cache
        if normalized["cache_enabled"] == "oui":
            impacts.append(("Cache active", -15, '#10b981', '[+]'))
        else:
            impacts.append(("Cache desactive", +12, '#ef4444', '[-]'))
        
        # CDN
        if normalized["cdn_enabled"] == "oui":
            impacts.append(("CDN active", -10, '#10b981', '[+]'))
        else:
            impacts.append(("CDN desactive", +8, '#ef4444', '[-]'))
        
        # PHP
        php = normalized["php_version"]
        if php in ["8.2", "8.3"]:
            impacts.append((f"PHP {php} recent", -8, '#10b981', '[+]'))
        elif php == "7.4":
            impacts.append((f"PHP {php} obsolete", +10, '#ef4444', '[-]'))
        elif php in ["8.0", "8.1"]:
            impacts.append((f"PHP {php}", +5, '#f59e0b', '[!]'))
        else:
            impacts.append((f"PHP non specifie", +8, '#f59e0b', '[!]'))
        
        # CPU
        cpu = normalized["cpu_usage_avg"]
        if cpu > 80:
            impacts.append((f"CPU eleve ({cpu:.0f}%)", +18, '#ef4444', '[-]'))
        elif cpu > 60:
            impacts.append((f"CPU moyen ({cpu:.0f}%)", +8, '#f59e0b', '[!]'))
        else:
            impacts.append((f"CPU optimal ({cpu:.0f}%)", -5, '#10b981', '[+]'))
        
        # RAM
        ram = normalized["ram_usage_avg"]
        if ram > 80:
            impacts.append((f"RAM saturee ({ram:.0f}%)", +15, '#ef4444', '[-]'))
        elif ram > 60:
            impacts.append((f"RAM moyenne ({ram:.0f}%)", +6, '#f59e0b', '[!]'))
        else:
            impacts.append((f"RAM optimale ({ram:.0f}%)", -3, '#10b981', '[+]'))
        
        # Plugins
        plugins = normalized["plugin_count"]
        heavy = len(normalized["heavy_plugins"])
        if plugins > 30 or heavy > 3:
            impacts.append((f"Trop de plugins ({int(plugins)})", +20, '#ef4444', '[-]'))
        elif plugins > 15:
            impacts.append((f"Plugins moderes ({int(plugins)})", +8, '#f59e0b', '[!]'))
        else:
            impacts.append((f"Peu de plugins ({int(plugins)})", -5, '#10b981', '[+]'))
        
        # Plugins lourds
        if heavy > 0:
            impacts.append((f"Plugins lourds ({heavy})", +heavy * 5, '#ef4444', '[-]'))
        
        # Croissance
        growth = normalized["traffic_growth_rate"]
        if growth > 30:
            impacts.append((f"Croissance explosive ({growth:.0f}%)", +15, '#ef4444', '[-]'))
        elif growth > 15:
            impacts.append((f"Croissance moderee ({growth:.0f}%)", +5, '#f59e0b', '[!]'))
        else:
            impacts.append((f"Croissance faible ({growth:.0f}%)", -3, '#10b981', '[+]'))
        
        # Disque
        disk = normalized["disk_usage_avg"]
        if disk > 80:
            impacts.append((f"Disque sature ({disk:.0f}%)", +8, '#ef4444', '[-]'))
        
        # Temps reponse
        rt = normalized["response_time"]
        if rt > 500:
            impacts.append((f"Reponse lente ({rt:.0f}ms)", +10, '#ef4444', '[-]'))
        elif rt > 0 and rt < 200:
            impacts.append((f"Reponse rapide ({rt:.0f}ms)", -4, '#10b981', '[+]'))
        
        # Trier par impact absolu
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        impacts = impacts[:12]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        labels = [f"{imp[3]} {imp[0]}" for imp in impacts]
        values = [imp[1] for imp in impacts]
        colors = [imp[2] for imp in impacts]
        
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.6,
                      edgecolor='white', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                       f'+{val:.0f}%', va='center', fontsize=11, fontweight='bold', color='#ef4444')
            else:
                ax.text(val - 0.5, bar.get_y() + bar.get_height()/2,
                       f'{val:.0f}%', va='center', fontsize=11, fontweight='bold', color='#10b981', ha='right')
        
        ax.axvline(x=0, color='#1e293b', linewidth=2)
        ax.axvspan(-20, 0, alpha=0.05, color='#10b981')
        ax.axvspan(0, 25, alpha=0.05, color='#ef4444')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Impact sur la charge serveur (%)', fontsize=13, fontweight='bold')
        ax.set_xlim(-25, 28)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        legend_elements = [
            Patch(facecolor='#10b981', alpha=0.7, label='Impact Positif (reduit charge)'),
            Patch(facecolor='#ef4444', alpha=0.7, label='Impact Negatif (augmente charge)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        ax.set_title(f'Impact des Features sur la Charge Predite ({current_load:.1f}%)',
                    fontsize=16, fontweight='bold')
        
        ax.text(0.02, 0.02, '<- Reduit la charge | Augmente la charge ->',
               transform=ax.transAxes, fontsize=9, color='#64748b', style='italic')
        
        plt.tight_layout()
        path = str(OUTPUT_DIR / f"feature_impact_{TIMESTAMP}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return path
    except Exception as exc:
        print(f"[ERREUR] Impact Features: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# ============================================
# GRAPHIQUE 4 : COURBE DE DEGRADATION TEMPORELLE
# ============================================

def graph_degradation_curve(normalized, current_load):
    """3 courbes superposees montrant la projection temporelle sur 12 mois"""
    try:
        growth = normalized["traffic_growth_rate"]
        cache = normalized["cache_enabled"] == "oui"
        cdn = normalized["cdn_enabled"] == "oui"
        plugins = normalized["plugin_count"]
        heavy = len(normalized["heavy_plugins"])
        
        # Calcul de la saturation
        if current_load >= 90:
            saturation_days = 0
        elif growth <= 0:
            saturation_days = 999 * 30.44
        else:
            sat_months = np.log(90 / max(1, current_load)) / np.log(1 + growth / 100)
            saturation_days = max(0, sat_months * 30.44)
        
        _, _, sat_text = days_to_months_days(saturation_days)
        
        # 12 mois
        months = np.arange(0, 13)
        
        # Scenario optimiste
        opt_mult = 0.7 if cache else 1.0
        opt_mult *= 0.85 if cdn else 1.0
        opt_growth = max(1, growth * opt_mult)
        opt_charges = []
        opt_load = current_load * 0.8
        for _ in months:
            opt_charges.append(min(100, opt_load))
            if opt_load < 100:
                opt_load *= (1 + opt_growth / 200)
        
        # Scenario realiste
        real_charges = []
        real_load = current_load
        for _ in months:
            real_charges.append(min(100, real_load))
            if real_load < 100:
                real_load *= (1 + growth / 100)
        
        # Scenario pessimiste
        pess_mult = 1.3 if not cache else 1.0
        pess_mult *= 1.2 if not cdn else 1.0
        pess_growth = min(85, growth * pess_mult)
        pess_charges = []
        pess_load = current_load * 1.2
        for _ in months:
            pess_charges.append(min(100, pess_load))
            if pess_load < 100:
                pess_load *= (1 + pess_growth / 100)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Zones colorees
        ax.fill_between(months, 0, 60, color='#10b981', alpha=0.08)
        ax.fill_between(months, 60, 80, color='#f59e0b', alpha=0.08)
        ax.fill_between(months, 80, 100, color='#ef4444', alpha=0.08)
        
        # Lignes de seuil
        ax.axhline(60, color='#10b981', linestyle='--', linewidth=1.5, alpha=0.5, label='Zone Optimale (60%)')
        ax.axhline(80, color='#f59e0b', linestyle='--', linewidth=1.5, alpha=0.5, label='Zone Surveillance (80%)')
        ax.axhline(100, color='#ef4444', linestyle='-', linewidth=2, alpha=0.7, label='Saturation (100%)')
        
        # Courbes
        ax.plot(months, opt_charges, color='#10b981', linewidth=3, marker='s', markersize=6,
               label='Scenario Optimiste (cache+CDN, peu plugins)', markevery=2)
        ax.plot(months, real_charges, color='#2563eb', linewidth=3, marker='o', markersize=6,
               label='Scenario Realiste (configuration actuelle)', markevery=2)
        ax.plot(months, pess_charges, color='#ef4444', linewidth=3, marker='^', markersize=6,
               label='Scenario Pessimiste (sans cache, pics max)', markevery=2)
        
        # Point actuel
        ax.scatter([0], [current_load], s=200, color='#8b5cf6', edgecolor='white',
                  linewidth=2, zorder=10, label=f'Charge actuelle: {current_load:.1f}%')
        
        # Annotation saturation
        if saturation_days > 0 and saturation_days < 365 * 5:
            sat_month = saturation_days / 30.44
            if sat_month <= 12:
                sat_idx = np.argmax(np.array(real_charges) >= 90)
                if sat_idx < len(months):
                    ax.axvline(months[sat_idx], color='#f97316', linestyle=':', linewidth=2)
                    ax.scatter([months[sat_idx]], [90], s=150, color='#f97316',
                              edgecolor='white', zorder=5)
                    ax.annotate(f'Saturation estimee\n{sat_text}',
                               xy=(months[sat_idx], 90),
                               xytext=(months[sat_idx] + 1, 85),
                               fontsize=10, fontweight='bold', color='#f97316',
                               bbox=dict(boxstyle='round', facecolor='#fff7ed',
                                        edgecolor='#f97316', alpha=0.9),
                               arrowprops=dict(arrowstyle='->', color='#f97316', lw=1.5))
        
        # Info box
        info_text = (
            f"Parametres actuels:\n"
            f"  Croissance: {growth:.1f}%\n"
            f"  Cache: {'Oui' if cache else 'Non'}\n"
            f"  CDN: {'Oui' if cdn else 'Non'}\n"
            f"  Plugins: {int(plugins)}\n"
            f"  Plugins lourds: {heavy}"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='#f1f5f9', edgecolor='#cbd5e1', alpha=0.9))
        
        # Zones de texte
        ax.text(6, 25, 'ZONE OPTIMALE', fontsize=12, fontweight='bold',
               color='#10b981', alpha=0.4, ha='center')
        ax.text(6, 70, 'ZONE SURVEILLANCE', fontsize=12, fontweight='bold',
               color='#f59e0b', alpha=0.4, ha='center')
        ax.text(6, 95, 'ZONE CRITIQUE', fontsize=12, fontweight='bold',
               color='#ef4444', alpha=0.4, ha='center')
        
        ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
        ax.set_ylabel('Charge Serveur (%)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xlim(0, 12)
        ax.set_xticks(range(0, 13, 2))
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        
        ax.set_title(f'Courbe de Degradation Temporelle - 3 Scenarios sur 12 Mois\n'
                    f'Saturation estimee: {sat_text}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        path = str(OUTPUT_DIR / f"degradation_curve_{TIMESTAMP}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return path
    except Exception as exc:
        print(f"[ERREUR] Degradation: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# ============================================
# GENERATION DE TOUS LES GRAPHIQUES
# ============================================

def generate_all_graphs():
    model_load, feature_columns = load_model()

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
    features_df, normalized = prepare_features(params, feature_columns)
    current_load = predict_load(model_load, features_df)

    graphs, errors = {}, {}

    # Les 4 graphiques
    generators = {
        "radar_resources": lambda: graph_radar_resources(normalized, current_load),
        "gauges_saturation": lambda: graph_gauges_saturation(normalized, current_load),
        "feature_impact": lambda: graph_feature_impact(normalized, current_load),
        "degradation_curve": lambda: graph_degradation_curve(normalized, current_load),
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
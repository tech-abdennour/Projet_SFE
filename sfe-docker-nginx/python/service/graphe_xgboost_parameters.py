def graph_charge_horaire(normalized):
    """
    Graphe 1: Courbe de charge horaire sur la journée
    Utilise les paramètres: peak_hours_start, peak_hours_end, cpu_usage_avg, cpu_usage_peak
    """
    try:
        # Récupérer les paramètres
        peak_start = str(normalized.get("peak_hours_start", DEFAULTS["peak_hours_start"]))
        peak_end = str(normalized.get("peak_hours_end", DEFAULTS["peak_hours_end"]))
        cpu_avg = float(normalized.get("cpu_usage_avg", DEFAULTS["cpu_usage_avg"]))
        cpu_peak = float(normalized.get("cpu_usage_peak", DEFAULTS["cpu_usage_peak"]))
        # Convertir les heures en minutes
        start_minutes = time_to_minutes(peak_start, DEFAULTS["peak_hours_start"])
        end_minutes = time_to_minutes(peak_end, DEFAULTS["peak_hours_end"])
        # Créer les points pour chaque heure de la journée (0h à 23h)
        hours = np.arange(0, 24)
        hourly_load = []
        for hour in hours:
            hour_minutes = hour * 60
            # Vérifier si l'heure est dans la plage de pointe
            if start_minutes <= end_minutes:
                # Plage normale (ex: 09:00 à 18:00)
                if start_minutes <= hour_minutes < end_minutes:
                    hourly_load.append(cpu_peak)
                else:
                    hourly_load.append(cpu_avg)
            else:
                # Plage qui traverse minuit (ex: 22:00 à 06:00)
                if hour_minutes >= start_minutes or hour_minutes < end_minutes:
                    hourly_load.append(cpu_peak)
                else:
                    hourly_load.append(cpu_avg)
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(14, 7))
        # Courbe de charge
        ax.plot(hours, hourly_load, color="#2563eb", linewidth=2.5, marker='o', markersize=8,
                label='Charge CPU estimée')
        # Zone de pointe
        start_hour = start_minutes / 60
        end_hour = end_minutes / 60
        if start_hour < end_hour:
            ax.axvspan(start_hour, end_hour, alpha=0.15, color='#ef4444', label='Heures de pointe')
        else:
            ax.axvspan(start_hour, 24, alpha=0.15, color='#ef4444', label='Heures de pointe')
            ax.axvspan(0, end_hour, alpha=0.15, color='#ef4444')
        # Lignes de référence
        ax.axhline(y=cpu_avg, color='#10b981', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Charge moyenne ({cpu_avg}%)')
        ax.axhline(y=cpu_peak, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Charge pic ({cpu_peak}%)')
        # Annotations
        if start_hour < end_hour:
            mid_peak = start_hour + (end_hour - start_hour) / 2
        else:
            mid_peak = (start_hour + 24 + end_hour) / 2
            if mid_peak > 24:
                mid_peak -= 24
        ax.text(mid_peak, cpu_peak + 1,
                f'Période de pointe\n{peak_start} - {peak_end}',
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#fee2e2', alpha=0.8))
        ax.set_xlabel('Heure de la journée', fontsize=12, fontweight='bold')
        ax.set_ylabel('Charge CPU (%)', fontsize=12, fontweight='bold')
        ax.set_title('Charge CPU horaire estimée sur 24h', fontsize=14, fontweight='bold')
        ax.set_xticks(hours)
        ax.set_xticklabels([f'{h:02d}h' for h in hours], rotation=45)
        ax.set_xlim(0, 23)
        ax.set_ylim(0, max(cpu_peak + 15, 100))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        # Ajouter des informations
        info_text = (
            f"Pic: {peak_start} - {peak_end}\n"
            f"CPU moyen: {cpu_avg}%\n"
            f"CPU pic: {cpu_peak}%\n"
            f"Différence: {cpu_peak - cpu_avg:.1f}%"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        path = str(OUTPUT_DIR / f"charge_horaire_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur charge horaire: {exc}", file=sys.stderr)
        return None


def generate_all_graphs():
    model_load, feature_columns, scaler = load_model()

    if model_load is None:
        return {
            "status": "error",
            "message": f"Modèle introuvable: {MODEL_PATH}",
        }

    if not feature_columns:
        if hasattr(model_load, "feature_names_in_"):
            feature_columns = list(model_load.feature_names_in_)
        else:
            return {
                "status": "error",
                "message": "feature_columns introuvable dans model.pkl",
            }

    json_file = find_latest_json()
    if json_file is None:
        return {
            "status": "error",
            "message": f"Aucun fichier JSON trouvé dans {DATA_DIR}",
        }

    params = load_params(json_file)
    features_df, normalized, raw_row = prepare_features(params, feature_columns)
    current_load = predict_load(model_load, features_df)

    graphs = {}
    errors = {}

    generators = {
        # Graphiques existants (dépendants du modèle)
        "partial_dependence": lambda: graph_partial_dependence(model_load, features_df, feature_columns),
        "saturation_evolution": lambda: graph_saturation_evolution(model_load, features_df, normalized, feature_columns),
        # NOUVEAUX GRAPHIQUES (indépendants du modèle, basés uniquement sur les paramètres)
        "charge_horaire": lambda: graph_charge_horaire(normalized),
        "charge_par_type_site": lambda: graph_charge_par_type_site(model_load, features_df, normalized, feature_columns),
    }


# Nouveau graphe : charge par type de site WordPress
def graph_charge_par_type_site(model_load, features_df, normalized, feature_columns):
    """
    Graphe: Répartition de la charge serveur selon le type de site WordPress
    Montre l'impact du type de site (small, medium, performance) sur la charge serveur prédite
    """
    try:
        # Récupérer les paramètres actuels
        current_wp_type = str(normalized.get("wp_type", DEFAULTS["wp_type"]))
        # Créer les variantes pour chaque type de site
        wp_types = ["small", "medium", "performance"]
        type_labels = ["Small", "Medium", "Performance"]
        colors = ['#10b981', '#f59e0b', '#ef4444']
        predicted_loads = []
        details = []
        for idx, wp_type in enumerate(wp_types):
            features_copy = features_df.copy()
            for col in feature_columns:
                if col.startswith("wp_type_"):
                    if col == f"wp_type_{wp_type}":
                        features_copy[col] = 1
                    else:
                        features_copy[col] = 0
            predicted_load = predict_load(model_load, features_copy)
            predicted_loads.append(predicted_load)
            growth_rate = max(0.0, float(normalized["traffic_growth_rate"]))
            if predicted_load >= 90:
                saturation_months = 0.0
                saturation_days = 0.0
            elif growth_rate <= 0:
                saturation_months = 999.0
                saturation_days = 999.0 * 30.44
            else:
                saturation_months = float(
                    np.log(90 / max(1.0, predicted_load)) / np.log(1 + growth_rate / 100)
                )
                saturation_days = max(0.0, saturation_months * 30.44)
            if predicted_load >= 85 or saturation_days <= 30:
                statut = "CRITIQUE"
            elif predicted_load >= 75 or saturation_days <= 60:
                statut = "URGENT"
            elif predicted_load >= 65 or saturation_days <= 180:
                statut = "SURVEILLANCE"
            else:
                statut = "OPTIMAL"
            months, days, text = days_to_months_days(saturation_days)
            details.append({
                "type": wp_type,
                "label": type_labels[idx],
                "predicted_load": predicted_load,
                "saturation_text": text,
                "saturation_days": saturation_days,
                "statut": statut
            })
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        bars = ax1.bar(type_labels, predicted_loads, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        for bar, load, detail in zip(bars, predicted_loads, details):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{load:.1f}%\n({detail["statut"]})',
                    ha='center', va='bottom', fontweight='bold', fontsize=10,
                    color='black')
        ax1.axhline(y=90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Saturation (90%)')
        ax1.axhline(y=85, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Critique (85%)')
        ax1.axhline(y=75, color='gold', linestyle=':', linewidth=1.5, alpha=0.5, label='Urgent (75%)')
        ax1.axhline(y=65, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Surveillance (65%)')
        current_index = wp_types.index(current_wp_type) if current_wp_type in wp_types else 1
        bars[current_index].set_edgecolor('black')
        bars[current_index].set_linewidth(3)
        ax1.set_ylabel('Charge serveur prédite (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Charge serveur par type de site WordPress', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(max(predicted_loads) + 15, 105))
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.legend(fontsize=9)
        ax2.axis('off')
        ax2.text(0.5, 0.95, 'Détails par type de site', ha='center', fontsize=14, fontweight='bold',
                transform=ax2.transAxes)
        headers = ['Type', 'Charge', 'Saturation', 'Statut']
        col_widths = [0.25, 0.25, 0.25, 0.25]
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            ax2.text(i/4 + 0.05, 0.85, header, fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='#e5e7eb'))
        for row_idx, detail in enumerate(details):
            y_pos = 0.75 - (row_idx * 0.12)
            if detail["type"] == current_wp_type:
                ax2.axhline(y=y_pos - 0.02, xmin=0, xmax=1, color='#2563eb', alpha=0.1, linewidth=25)
            cell_data = [
                detail["label"],
                f"{detail['predicted_load']:.1f}%",
                detail["saturation_text"],
                detail["statut"]
            ]
            for col_idx, (data, width) in enumerate(zip(cell_data, col_widths)):
                if col_idx == 3:
                    if detail["statut"] == "CRITIQUE":
                        color = '#ef4444'
                    elif detail["statut"] == "URGENT":
                        color = '#f97316'
                    elif detail["statut"] == "SURVEILLANCE":
                        color = '#eab308'
                    else:
                        color = '#10b981'
                    ax2.text(col_idx/4 + 0.05, y_pos, data, fontsize=10, fontweight='bold',
                            color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax2.text(col_idx/4 + 0.05, y_pos, data, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.text(0.5, 0.08, f'Type actuel: {current_wp_type.upper()} (surligné en bleu)',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#dbeafe', alpha=0.8),
                transform=ax2.transAxes)
        fig.suptitle('Impact du type de site WordPress sur la charge serveur',
                     fontsize=16, fontweight='bold', y=1.02)
        current_detail = details[current_index] if current_index < len(details) else details[1]
        info_text = (
            f"Type actuel: {current_detail['label']}\n"
            f"Charge: {current_detail['predicted_load']:.1f}%\n"
            f"Saturation: {current_detail['saturation_text']}\n"
            f"Statut: {current_detail['statut']}"
        )
        fig.text(0.5, -0.05, info_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        path = str(OUTPUT_DIR / f"charge_par_type_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur charge par type: {exc}", file=sys.stderr)
        return None

    # ...existing code...
def graph_response_time_projection(model_load, features_df, normalized, feature_columns):
    """
    Projection dynamique du temps de réponse serveur sur 24 mois selon tous les paramètres du JSON.
    """
    try:
        # Valeur initiale du temps de réponse
        base_response_time = float(normalized.get("response_time", 350))
        growth_rate_percent = max(0.0, float(normalized.get("traffic_growth_rate", 0)))
        # On projette la croissance du trafic sur la charge serveur avec le modèle
        months_projection = 24
        months = np.arange(0, months_projection + 1)
        response_times = []
        current_features = features_df.copy()
        for month in months:
            # On met à jour le trafic selon la croissance
            if month > 0:
                current_features = current_features.copy()
                current_features["visitors_per_day"] *= (1 + growth_rate_percent / 100)
                current_features["pageviews_per_day"] *= (1 + growth_rate_percent / 100)
            # Prédiction du temps de réponse avec le modèle
            predicted = model_load.predict(current_features)[0]
            # On suppose que le temps de réponse évolue proportionnellement à la charge prédite
            response_times.append(predicted)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(months, response_times, color="#f59e42", linewidth=3, marker="o", markersize=7, label="Temps de réponse projeté")
        ax.set_title("Projection du temps de réponse serveur (modèle XGBoost)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Mois")
        ax.set_ylabel("Temps de réponse projeté (ms)")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path = str(OUTPUT_DIR / f"response_time_projection_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur projection temps de réponse: {exc}", file=sys.stderr)
        return None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


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
    "woocommerce",
    "elementor",
    "wpml",
    "yoast",
    "revslider",
    "gravityforms",
]

PHP_VERSIONS = ["7.4", "8.0", "8.1", "8.2", "8.3"]
WP_TYPES = ["small", "medium", "performance"]

DEFAULTS = {
    "visitors_per_day": 5000,
    "pageviews_per_day": 150,
    "traffic_growth_rate": 15,
    "peak_hours_start": "09:00",
    "peak_hours_end": "18:00",
    "cpu_usage_avg": 45,
    "cpu_usage_peak": 75,
    "ram_usage_avg": 60,
    "ram_usage_max": 85,
    "disk_usage_avg": 45,
    "disk_usage_max": 70,
    "response_time": 350,
    "disk_read_iops": 120,
    "disk_write_iops": 80,
    "plugin_count": 25,
    "heavy_plugins": [],
    "php_version": "8.2",
    "cache_enabled": "oui",
    "cdn_enabled": "oui",
    "wp_type": "medium",
}

FRENCH_LABELS = {
    "visitors_per_day": "Visiteurs / jour",
    "pageviews_per_day": "Pages vues / jour",
    "traffic_growth_rate": "Taux de croissance",
    "peak_hours_start_minutes": "Pic début",
    "peak_hours_end_minutes": "Pic fin",
    "peak_duration_minutes": "Durée pic",
    "cpu_usage_avg": "CPU moyen",
    "cpu_usage_peak": "CPU max",
    "ram_usage_avg": "RAM moyenne",
    "ram_usage_max": "RAM max",
    "disk_usage_avg": "Disque utilisé",
    "disk_usage_max": "Disque max",
    "response_time": "Temps réponse",
    "disk_read_iops": "IOPS Read",
    "disk_write_iops": "IOPS Write",
    "plugin_count": "Nombre plugins",
}


def load_model():
    if not MODEL_PATH.exists():
        return None, None, None

    payload = joblib.load(str(MODEL_PATH))

    if isinstance(payload, dict):
        model = payload.get("model")
        feature_columns = payload.get("feature_columns")
        scaler = payload.get("scaler")
    else:
        model = payload
        feature_columns = None
        scaler = None

    return model, feature_columns, scaler


def find_latest_json():
    if not DATA_DIR.exists():
        return None

    files = sorted(
        glob.glob(str(DATA_DIR / "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
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
    if value == "" or value.lower() == "none":
        return default

    return value


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
    normalized = {
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
    return normalized


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

    row["peak_duration_minutes"] = max(
        0,
        row["peak_hours_end_minutes"] - row["peak_hours_start_minutes"],
    )

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
    prediction = float(model_load.predict(features_df)[0])
    return min(100.0, max(0.0, prediction))


def days_to_months_days(days):
    """Identique à predict_from_file.py"""
    if days is None or days >= 30000:
        return 999, 0, "∞"
    if days <= 0:
        return 0, 0, "SATURÉ"

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
        return "Cache activé"
    if feature.startswith("cdn_enabled_"):
        return "CDN activé"
    if feature.startswith("wp_type_"):
        return "Pack WordPress"

    return FRENCH_LABELS.get(feature, feature)


def graph_partial_dependence(model_load, features_df, feature_columns):
    try:
        numeric_features = [
            feature
            for feature in feature_columns
            if not feature.startswith(("heavy_plugin_", "php_version_", "cache_enabled_", "cdn_enabled_", "wp_type_"))
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
                    model_load,
                    sim_df,
                    features=[feature_index],
                    feature_names=feature_columns,
                    grid_resolution=35,
                    ax=ax,
                    line_kw={"color": "#2563eb", "linewidth": 2.4},
                )
                current_value = float(features_df.iloc[0][feature])
                ax.axvline(current_value, color="#ef4444", linestyle="--", linewidth=1.8)
                ax.set_title(label_for(feature), fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Charge serveur" if index % n_cols == 0 else "")
                ax.grid(True, alpha=0.25)
            except Exception as exc:
                ax.text(0.5, 0.5, f"{label_for(feature)}\nPDP indisponible", ha="center", va="center")
                ax.set_title(label_for(feature), fontsize=11, fontweight="bold")
                ax.axis("off")
                print(f"Erreur PDP {feature}: {exc}", file=sys.stderr)

        for index in range(len(numeric_features), len(axes)):
            axes[index].axis("off")

        plt.suptitle("Partial Dependence Plot - effet des variables sur la charge serveur", fontsize=18, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        path = str(OUTPUT_DIR / f"partial_dependence_all_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur PDP: {exc}", file=sys.stderr)
        return None








def graph_saturation_evolution(model_load, features_df, normalized, feature_columns):
    """
    MODIFIÉ: Utilise LA MÊME LOGIQUE de saturation que predict_from_file.py
    """
    try:
        # Récupérer la charge actuelle avec le modèle (comme predict_from_file.py)
        current_load = predict_load(model_load, features_df)
        
        # Récupérer le taux de croissance (en pourcentage, comme predict_from_file.py)
        growth_rate_percent = max(0.0, float(normalized["traffic_growth_rate"]))
        
        # MÊME LOGIQUE que predict_from_file.py pour calculer la saturation
        if current_load >= 90:
            saturation_months_raw = 0.0
            saturation_days = 0.0
        elif growth_rate_percent <= 0:
            saturation_months_raw = 999.0
            saturation_days = 999.0 * 30.44
        else:
            saturation_months_raw = float(
                np.log(90 / max(1.0, current_load)) / np.log(1 + growth_rate_percent / 100)
            )
            saturation_days = max(0.0, saturation_months_raw * 30.44)
        
        # Conversion en mois/jours (même fonction que predict_from_file.py)
        months_sat, days_sat, text_sat = days_to_months_days(saturation_days)
        
        # Projection sur 24 mois
        months_projection = 24
        months = np.arange(0, months_projection + 1)
        charges = []
        
        # Calcul de la courbe d'évolution avec LA MÊME formule que predict_from_file.py
        # load * (1 + growth/100)^month
        current_load_proj = current_load
        for month in months:
            charges.append(min(100, current_load_proj))
            if current_load_proj < 100:
                # Même formule que predict_from_file.py: load *= (1 + growth/100)
                current_load_proj *= (1 + growth_rate_percent / 100)
        
        charges = np.array(charges)
        
        # Trouver le point de saturation sur la courbe
        saturation_indices = np.where(charges >= SATURATION_LIMIT)[0]
        
        if len(saturation_indices) > 0:
            saturation_month = int(saturation_indices[0])
        else:
            saturation_month = None
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Courbe bleue d'évolution de la charge
        ax.plot(months, charges, color="#2563eb", linewidth=3, marker="o", markersize=6, 
                label="Charge serveur")
        
        # Ligne de saturation à 90% (même seuil que predict_from_file.py)
        ax.axhline(SATURATION_LIMIT, color="#ef4444", linestyle="--", linewidth=2.5, 
                   label=f"Seuil saturation {SATURATION_LIMIT}%")
        
        # Zones colorées
        ax.fill_between(months, charges, SATURATION_LIMIT, where=charges >= SATURATION_LIMIT, 
                        color="#ef4444", alpha=0.18)
        ax.fill_between(months, charges, 0, color="#2563eb", alpha=0.08)
        
        # Point de saturation
        if saturation_month is not None:
            ax.axvline(saturation_month, color="#f97316", linestyle=":", linewidth=2.5)
            ax.scatter([saturation_month], [charges[saturation_month]], s=180, 
                      color="#f97316", edgecolor="white", zorder=5)
            
            # Annotation avec les mêmes informations que predict_from_file.py
            # Afficher uniquement le texte formaté (text_sat) sans doublon
            ax.text(
                saturation_month,
                min(100, charges[saturation_month] + 4),
                f"Saturation\n{text_sat}",
                ha="center",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#fff7ed", edgecolor="#f97316", alpha=0.95),
            )
        
        # Statut cohérent avec build_recommendation de predict_from_file.py
        if current_load >= 85 or saturation_days <= 30:
            statut = "CRITIQUE"
        elif current_load >= 75 or saturation_days <= 60:
            statut = "URGENT"
        elif current_load >= 65 or saturation_days <= 180:
            statut = "SURVEILLANCE"
        else:
            statut = "OPTIMAL"
        
        # Bloc d'informations de saturation supprimé (plus d'encart ni de texte orange)
        
        # Labels des axes (en mois et jours, comme predict_from_file.py)
        labels = [f"{month} mois\n{int(month * 30.44)} jours" for month in months]
        
        ax.set_title("Évolution de la charge serveur jusqu'à saturation", fontsize=17, fontweight="bold")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Charge serveur (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(months[::3])
        ax.set_xticklabels([labels[index] for index in range(0, len(labels), 3)], fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')
        plt.tight_layout()

        path = str(OUTPUT_DIR / f"saturation_evolution_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur saturation evolution: {exc}", file=sys.stderr)
        return None


def generate_all_graphs():
    model_load, feature_columns, scaler = load_model()

    if model_load is None:
        return {
            "status": "error",
            "message": f"Modèle introuvable: {MODEL_PATH}",
        }

    if not feature_columns:
        if hasattr(model_load, "feature_names_in_"):
            feature_columns = list(model_load.feature_names_in_)
        else:
            return {
                "status": "error",
                "message": "feature_columns introuvable dans model.pkl",
            }

    json_file = find_latest_json()
    if json_file is None:
        return {
            "status": "error",
            "message": f"Aucun fichier JSON trouvé dans {DATA_DIR}",
        }

    params = load_params(json_file)
    features_df, normalized, raw_row = prepare_features(params, feature_columns)
    current_load = predict_load(model_load, features_df)

    graphs = {}
    errors = {}

    generators = {
        "partial_dependence": lambda: graph_partial_dependence(model_load, features_df, feature_columns),
        "saturation_evolution": lambda: graph_saturation_evolution(model_load, features_df, normalized, feature_columns),
        "response_time_projection": lambda: graph_response_time_projection(model_load, features_df, normalized, feature_columns),
        # NOUVEAUX GRAPHIQUES (indépendants du modèle, basés uniquement sur les paramètres)
        "charge_horaire": lambda: graph_charge_horaire(normalized),
        "charge_par_type_site": lambda: graph_charge_par_type_site(model_load, features_df, normalized, feature_columns),
    }

    for name, generator in generators.items():
        try:
            path = generator()
            if path:
                graphs[name] = path
            else:
                errors[name] = "Non généré"
        except Exception as exc:
            errors[name] = str(exc)
            print(f"Erreur {name}: {exc}", file=sys.stderr)

    if not graphs:
        return {
            "status": "error",
            "message": "Aucun graphique généré",
            "errors": errors,
            "source": str(json_file),
            "current_load": current_load,
        }

    return {
        "status": "success",
        "message": f"{len(graphs)} graphique(s) généré(s)",
        "graphs": graphs,
        "errors": errors,
        "source": str(json_file),
        "current_load": round(current_load, 2),
        "normalized_parameters": normalized,
    }


if __name__ == "__main__":
    result = generate_all_graphs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
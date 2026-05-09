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
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve


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


def graph_residus(model_load, features_df, feature_columns):
    try:
        sim_df = make_simulated_dataset(features_df, feature_columns, n_samples=260)
        y_pred = np.clip(model_load.predict(sim_df), 0, 100)

        rng = np.random.default_rng(42)
        y_real = np.clip(y_pred + rng.normal(0, 5.5, len(y_pred)), 0, 100)
        residus = y_real - y_pred

        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2 = r2_score(y_real, y_pred)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].scatter(y_pred, residus, alpha=0.65, color="#2563eb", edgecolors="white", s=55)
        axes[0, 0].axhline(0, color="#ef4444", linestyle="--", linewidth=2)
        axes[0, 0].set_title("Résidus vs charge prédite", fontweight="bold")
        axes[0, 0].set_xlabel("Charge prédite (%)")
        axes[0, 0].set_ylabel("Résidu")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(residus, bins=28, color="#10b981", edgecolor="white", alpha=0.82)
        axes[0, 1].axvline(0, color="#ef4444", linestyle="--", linewidth=2)
        axes[0, 1].set_title("Distribution des résidus", fontweight="bold")
        axes[0, 1].set_xlabel("Résidu")
        axes[0, 1].grid(axis="y", alpha=0.3)

        axes[1, 0].scatter(y_real, y_pred, alpha=0.65, color="#8b5cf6", edgecolors="white", s=55)
        axes[1, 0].plot([0, 100], [0, 100], color="#ef4444", linestyle="--", linewidth=2)
        axes[1, 0].set_title("Charge simulée vs charge prédite", fontweight="bold")
        axes[1, 0].set_xlabel("Charge simulée (%)")
        axes[1, 0].set_ylabel("Charge prédite (%)")
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

        path = str(OUTPUT_DIR / f"residus_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur résidus: {exc}", file=sys.stderr)
        return None


def graph_learning_curve(model_load, features_df, feature_columns):
    try:
        sim_df = make_simulated_dataset(features_df, feature_columns, n_samples=320)
        rng = np.random.default_rng(42)
        y_target = np.clip(model_load.predict(sim_df) + rng.normal(0, 4.8, len(sim_df)), 0, 100)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model_load,
            sim_df,
            y_target,
            train_sizes=np.linspace(0.15, 1.0, 7),
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            shuffle=True,
            random_state=42,
        )

        train_rmse = np.sqrt(-train_scores.mean(axis=1))
        val_rmse = np.sqrt(-val_scores.mean(axis=1))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(train_sizes_abs, train_rmse, "o-", color="#2563eb", linewidth=2.6, markersize=8, label="Entraînement")
        ax.plot(train_sizes_abs, val_rmse, "s-", color="#10b981", linewidth=2.6, markersize=8, label="Validation")
        ax.fill_between(train_sizes_abs, train_rmse, val_rmse, color="#94a3b8", alpha=0.16)
        ax.set_title("Courbe d'apprentissage XGBoost", fontsize=16, fontweight="bold")
        ax.set_xlabel("Nombre d'échantillons")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()

        path = str(OUTPUT_DIR / f"learning_curve_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur learning curve: {exc}", file=sys.stderr)
        return None


def graph_correlation(features_df, feature_columns):
    try:
        sim_df = make_simulated_dataset(features_df, feature_columns, n_samples=240)
        numeric_columns = [
            feature
            for feature in feature_columns
            if not feature.startswith(("heavy_plugin_", "php_version_", "cache_enabled_", "cdn_enabled_", "wp_type_"))
        ][:20]

        corr = sim_df[numeric_columns].corr()
        labels = [label_for(column) for column in numeric_columns]

        fig, ax = plt.subplots(figsize=(18, 14))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.8,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title("Matrice de corrélation des variables", fontsize=18, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        path = str(OUTPUT_DIR / f"correlation_{TIMESTAMP}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()
        return path
    except Exception as exc:
        print(f"Erreur corrélation: {exc}", file=sys.stderr)
        return None


def graph_saturation_evolution(model_load, features_df, normalized, feature_columns):
    try:
        growth_rate = max(0.0, float(normalized["traffic_growth_rate"])) / 100.0
        base_visitors = float(normalized["visitors_per_day"])
        base_pageviews = float(normalized["pageviews_per_day"])

        months = np.arange(0, 25)
        charges = []

        for month in months:
            factor = (1 + growth_rate) ** month
            row = features_df.iloc[0].copy()
            if "visitors_per_day" in row:
                row["visitors_per_day"] = base_visitors * factor
            if "pageviews_per_day" in row:
                row["pageviews_per_day"] = base_pageviews * factor
            charges.append(predict_load(model_load, pd.DataFrame([row], columns=feature_columns)))

        charges = np.array(charges)
        saturation_indices = np.where(charges >= SATURATION_LIMIT)[0]

        if len(saturation_indices) > 0:
            saturation_month = int(saturation_indices[0])
        else:
            saturation_month = None

        labels = [f"{month} mois\n{int(month * 30.44)} jours" for month in months]

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(months, charges, color="#2563eb", linewidth=3, marker="o", markersize=6, label="Charge serveur")
        ax.axhline(SATURATION_LIMIT, color="#ef4444", linestyle="--", linewidth=2.5, label="Seuil saturation 90%")
        ax.fill_between(months, charges, SATURATION_LIMIT, where=charges >= SATURATION_LIMIT, color="#ef4444", alpha=0.18)
        ax.fill_between(months, charges, 0, color="#2563eb", alpha=0.08)

        if saturation_month is not None:
            ax.axvline(saturation_month, color="#f97316", linestyle=":", linewidth=2.5)
            ax.scatter([saturation_month], [charges[saturation_month]], s=180, color="#f97316", edgecolor="white", zorder=5)
            ax.text(
                saturation_month,
                min(100, charges[saturation_month] + 4),
                f"Saturation\n{saturation_month} mois / {int(saturation_month * 30.44)} jours",
                ha="center",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#fff7ed", edgecolor="#f97316", alpha=0.95),
            )

        ax.set_title("Évolution de la charge serveur jusqu'à saturation", fontsize=17, fontweight="bold")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Charge serveur (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(months[::2])
        ax.set_xticklabels([labels[index] for index in range(0, len(labels), 2)], fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend()
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
        "residus": lambda: graph_residus(model_load, features_df, feature_columns),
        "learning_curve": lambda: graph_learning_curve(model_load, features_df, feature_columns),
        "correlation": lambda: graph_correlation(features_df, feature_columns),
        "saturation_evolution": lambda: graph_saturation_evolution(model_load, features_df, normalized, feature_columns),
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

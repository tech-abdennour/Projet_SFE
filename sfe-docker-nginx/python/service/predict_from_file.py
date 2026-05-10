#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

if Path("/app").exists():
    MODEL_PATH = Path("/app/service/models/model.pkl")
    PARAMS_DIR = Path("/app/Donnee_parametres")
else:
    MODEL_PATH = BASE_DIR / "models" / "model.pkl"
    PARAMS_DIR = BASE_DIR.parent / "Donnee_parametres"

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


def load_model() -> tuple[Any | None, list[str] | None, dict[str, Any] | None]:
    if not MODEL_PATH.exists():
        return None, None, None

    payload = joblib.load(MODEL_PATH)

    if isinstance(payload, dict):
        model = payload.get("model")
        feature_columns = payload.get("feature_columns")
        metadata = payload
    else:
        model = payload
        feature_columns = None
        metadata = {}

    print(f"Modèle chargé depuis {MODEL_PATH}", file=sys.stderr)
    return model, feature_columns, metadata


def find_latest_json() -> Path | None:
    if not PARAMS_DIR.exists():
        return None

    json_files = sorted(
        PARAMS_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return json_files[0] if json_files else None


def load_params(filepath: Path) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("parameters", data.get("params", data))


def to_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clean_select(value: Any, default: str) -> str:
    if value is None:
        return default

    value = str(value).strip()
    if value == "" or value.lower() == "none":
        return default

    return value


def time_to_minutes(value: Any, default: str) -> int:
    value = clean_select(value, default)

    try:
        hour, minute = value.split(":")[:2]
        return int(hour) * 60 + int(minute)
    except (ValueError, AttributeError):
        hour, minute = default.split(":")
        return int(hour) * 60 + int(minute)


def normalize_heavy_plugins(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]

    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "visitors_per_day": to_float(params.get("visitors_per_day"), DEFAULTS["visitors_per_day"]),
        "pageviews_per_day": to_float(params.get("pageviews_per_day"), DEFAULTS["pageviews_per_day"]),
        "traffic_growth_rate": to_float(params.get("traffic_growth_rate"), DEFAULTS["traffic_growth_rate"]),
        "peak_hours_start": clean_select(params.get("peak_hours_start"), DEFAULTS["peak_hours_start"]),
        "peak_hours_end": clean_select(params.get("peak_hours_end"), DEFAULTS["peak_hours_end"]),
        "cpu_usage_avg": to_float(params.get("cpu_usage_avg"), DEFAULTS["cpu_usage_avg"]),
        "cpu_usage_peak": to_float(params.get("cpu_usage_peak"), DEFAULTS["cpu_usage_peak"]),
        "ram_usage_avg": to_float(params.get("ram_usage_avg"), DEFAULTS["ram_usage_avg"]),
        "ram_usage_max": to_float(params.get("ram_usage_max"), DEFAULTS["ram_usage_max"]),
        "disk_usage_avg": to_float(params.get("disk_usage_avg"), DEFAULTS["disk_usage_avg"]),
        "disk_usage_max": to_float(params.get("disk_usage_max"), DEFAULTS["disk_usage_max"]),
        "response_time": to_float(params.get("response_time"), DEFAULTS["response_time"]),
        "disk_read_iops": to_float(params.get("disk_read_iops"), DEFAULTS["disk_read_iops"]),
        "disk_write_iops": to_float(params.get("disk_write_iops"), DEFAULTS["disk_write_iops"]),
        "plugin_count": to_float(params.get("plugin_count"), DEFAULTS["plugin_count"]),
        "heavy_plugins": normalize_heavy_plugins(params.get("heavy_plugins", DEFAULTS["heavy_plugins"])),
        "php_version": clean_select(params.get("php_version"), DEFAULTS["php_version"]),
        "cache_enabled": clean_select(params.get("cache_enabled"), DEFAULTS["cache_enabled"]),
        "cdn_enabled": clean_select(params.get("cdn_enabled"), DEFAULTS["cdn_enabled"]),
        "wp_type": clean_select(params.get("wp_type"), DEFAULTS["wp_type"]),
    }

    if normalized["php_version"] not in PHP_VERSIONS:
        normalized["php_version"] = DEFAULTS["php_version"]

    if normalized["cache_enabled"] not in ["oui", "non"]:
        normalized["cache_enabled"] = DEFAULTS["cache_enabled"]

    if normalized["cdn_enabled"] not in ["oui", "non"]:
        normalized["cdn_enabled"] = DEFAULTS["cdn_enabled"]

    if normalized["wp_type"] not in WP_TYPES:
        normalized["wp_type"] = DEFAULTS["wp_type"]

    return normalized


def prepare_features(params: dict[str, Any], feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = normalize_params(params)
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
        "peak_hours_start_minutes": time_to_minutes(
            normalized["peak_hours_start"],
            DEFAULTS["peak_hours_start"],
        ),
        "peak_hours_end_minutes": time_to_minutes(
            normalized["peak_hours_end"],
            DEFAULTS["peak_hours_end"],
        ),
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

    features = pd.DataFrame([{column: row.get(column, 0) for column in feature_columns}])
    return features, normalized


def days_to_months_days(days: float | None) -> tuple[int, int, str]:
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


def build_recommendation(predicted_load: float, saturation_days: float, saturation_text: str) -> tuple[str, str]:
    if predicted_load >= 85 or saturation_days <= 30:
        return "CRITIQUE", "Migration immédiate requise - Serveur en surcharge critique"

    if predicted_load >= 75 or saturation_days <= 60:
        return "URGENT", f"Planifier une migration urgente - Risque de saturation dans {saturation_text}"

    if predicted_load >= 65 or saturation_days <= 180:
        return "SURVEILLANCE", f"Surveiller et optimiser - Marge de {saturation_text} avant saturation"

    return "OPTIMAL", "Configuration stable - Aucune action requise"


def predict(model: Any, feature_columns: list[str], params: dict[str, Any], metadata: dict[str, Any] = None) -> dict[str, Any]:
    features, normalized = prepare_features(params, feature_columns)

    predicted_load = float(model.predict(features)[0])
    predicted_load = round(min(100.0, max(0.0, predicted_load)), 2)

    growth = max(0.0, float(normalized["traffic_growth_rate"]))

    if predicted_load >= 90:
        saturation_months_raw = 0.0
        saturation_days = 0.0
    elif growth <= 0:
        saturation_months_raw = 999.0
        saturation_days = 999.0 * 30.44
    else:
        saturation_months_raw = float(
            np.log(90 / max(1.0, predicted_load)) / np.log(1 + growth / 100)
        )
        saturation_days = max(0.0, saturation_months_raw * 30.44)

    saturation_months, saturation_jours, saturation_text = days_to_months_days(saturation_days)
    status, recommendation = build_recommendation(predicted_load, saturation_days, saturation_text)

    capacity_margin = round(max(0.0, 100.0 - predicted_load), 2)

    # Récupérer le vrai score R² du modèle depuis le model.pkl si disponible
    r2_score_model = None
    if metadata and isinstance(metadata, dict) and 'performance_metrics' in metadata:
        r2_score_model = metadata['performance_metrics'].get('r2', None)
    if (
        r2_score_model is None
        or r2_score_model == ''
        or r2_score_model == 'null'
        or (isinstance(r2_score_model, float) and (np.isnan(r2_score_model) or np.isinf(r2_score_model)))
    ):
        r2_score_model = "N/A"
    else:
        try:
            if isinstance(r2_score_model, (float, int)):
                r2_score_model = round(float(r2_score_model), 4)
            else:
                r2_score_model = str(r2_score_model)
        except Exception:
            r2_score_model = str(r2_score_model)
    return {
        "predicted_load": predicted_load,
        "xgboost_score": r2_score_model,
        "recommended_capacity_score": predicted_load,
        "capacity_margin": capacity_margin,
        "saturation_days": round(float(saturation_days), 2),
        "saturation_months": saturation_months,
        "saturation_jours": saturation_jours,
        "saturation_text": saturation_text,
        "saturation_months_raw": round(float(saturation_months_raw), 2),
        "status": status,
        "recommendation": recommendation,
        "normalized_parameters": normalized,
    }


def predict_from_json() -> dict[str, Any]:
    model, feature_columns, metadata = load_model()

    if model is None:
        return {
            "status": "error",
            "message": "Modèle introuvable",
            "output": {
                "result": {
                    "predicted_load": None,
                    "xgboost_score": None,
                    "saturation_text": "Modèle introuvable",
                    "status": "ERREUR",
                    "recommendation": "Le modèle XGBoost n'a pas pu être chargé.",
                }
            },
        }

    if not feature_columns:
        return {
            "status": "error",
            "message": "Colonnes du modèle introuvables",
            "output": {
                "result": {
                    "predicted_load": None,
                    "xgboost_score": None,
                    "saturation_text": "Colonnes introuvables",
                    "status": "ERREUR",
                    "recommendation": "Réentraîne le modèle pour sauvegarder feature_columns dans model.pkl.",
                }
            },
        }

    json_file = find_latest_json()

    if json_file is None:
        return {
            "status": "error",
            "message": "Aucun fichier JSON trouvé",
            "output": {
                "result": {
                    "predicted_load": None,
                    "xgboost_score": None,
                    "saturation_text": "Aucun paramètre trouvé",
                    "status": "ERREUR",
                    "recommendation": "Ajoute un fichier JSON dans /app/Donnee_parametres avant la prédiction.",
                }
            },
        }

    params = load_params(json_file)
    result = predict(model, feature_columns, params, metadata)

    return {
        "status": "success",
        "message": "Prédiction effectuée avec le dernier JSON.",
        "source": {
            "filename": json_file.name,
            "path": str(json_file),
            "modified_at": datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(timespec="seconds"),
        },
        "model": {
            "path": str(MODEL_PATH),
            "target_column": metadata.get("target_column") if isinstance(metadata, dict) else None,
            "feature_count": len(feature_columns),
        },
        "output": {
            "result": result,
        },
    }


if __name__ == "__main__":
    response = predict_from_json()
    print(json.dumps(response, ensure_ascii=False, indent=2))

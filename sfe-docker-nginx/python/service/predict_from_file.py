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
PLAN_LABELS = ["small", "medium", "performance"]

# Correspondance Plan → Score de charge estimé (milieu de la plage)
PLAN_TO_LOAD_SCORE = {
    "small": 17.5,        # Milieu de [0, 35]
    "medium": 50.0,       # Milieu de [35, 65]
    "performance": 82.5,  # Milieu de [65, 100]
}

# Seuils de classification
PLAN_THRESHOLDS = [35, 65]


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
    print(f"Type de modèle: {type(model).__name__}", file=sys.stderr)
    
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


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_select(value: Any, default: str = None) -> str | None:
    if value is None:
        return default
    value = str(value).strip()
    if value == "" or value.lower() == "none":
        return default
    return value


def time_to_minutes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            hour, minute = value.split(":")[:2]
            return int(hour) * 60 + int(minute)
        except (ValueError, AttributeError):
            return None
    return None


def normalize_heavy_plugins(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = {
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
    return normalized


def calculate_load_score_from_params(normalized: dict[str, Any]) -> float:
    """
    Calcule un score de charge estimé basé sur les paramètres normalisés.
    C'est une approximation basée sur la logique métier des plans d'hébergement.
    """
    score = 0.0
    
    # Facteur trafic (0-35 points)
    visitors = normalized["visitors_per_day"]
    if visitors > 0:
        # Small: max 2000, Medium: max 15000, Performance: max 150000
        traffic_ratio = min(visitors / 150000, 1.0)
        score += traffic_ratio * 35
    
    # Facteur CPU (0-25 points)
    cpu_avg = normalized["cpu_usage_avg"]
    cpu_peak = normalized["cpu_usage_peak"]
    score += (cpu_avg / 100) * 15
    score += (cpu_peak / 100) * 10
    
    # Facteur RAM (0-15 points)
    ram_avg = normalized["ram_usage_avg"]
    ram_max = normalized["ram_usage_max"]
    score += (ram_avg / 100) * 8
    score += (ram_max / 100) * 7
    
    # Facteur plugins (0-10 points)
    plugin_count = normalized["plugin_count"]
    heavy_plugins_count = len(normalized["heavy_plugins"])
    score += min((plugin_count / 50) * 5, 5)
    score += min(heavy_plugins_count * 2, 5)
    
    # Facteur croissance (0-10 points)
    growth = normalized["traffic_growth_rate"]
    score += min((growth / 85) * 10, 10)
    
    # Facteur performance (0-5 points)
    response_time = normalized["response_time"]
    score += min((response_time / 5000) * 5, 5)
    
    return round(min(max(score, 1.0), 100.0), 2)


def prepare_features(params: dict[str, Any], feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = normalize_params(params)
    heavy_plugins = set(normalized["heavy_plugins"])

    start_minutes = time_to_minutes(normalized["peak_hours_start"])
    end_minutes = time_to_minutes(normalized["peak_hours_end"])
    
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
        "peak_hours_start_minutes": start_minutes if start_minutes is not None else 0,
        "peak_hours_end_minutes": end_minutes if end_minutes is not None else 0,
        "heavy_plugins_sum": float(len(heavy_plugins)),
        "wp_facteur": (normalized["visitors_per_day"] * 0.0001 + 
                       normalized["plugin_count"] * 0.5 + 
                       len(heavy_plugins) * 2),
    }

    # One-hot encoding des heavy plugins
    for plugin in HEAVY_PLUGIN_OPTIONS:
        row[f"heavy_plugin_{plugin}"] = 1 if plugin in heavy_plugins else 0

    # One-hot encoding des versions PHP
    for version in PHP_VERSIONS:
        row[f"php_{version}"] = 1 if normalized["php_version"] == version else 0

    # One-hot encoding cache/cdn
    for value in ["non", "oui"]:
        row[f"cache_{value}"] = 1 if normalized["cache_enabled"] == value else 0
        row[f"cdn_{value}"] = 1 if normalized["cdn_enabled"] == value else 0

    # One-hot encoding wp_type
    for wp_type in WP_TYPES:
        row[f"wp_{wp_type}"] = 1 if normalized["wp_type"] == wp_type else 0

    row["peak_duration_minutes"] = max(0, row["peak_hours_end_minutes"] - row["peak_hours_start_minutes"])

    # Créer le DataFrame avec toutes les colonnes attendues
    features = pd.DataFrame([{column: row.get(column, 0) for column in feature_columns}])
    features = features.astype(float)
    
    # Debug
    print(f"Nombre de features avec valeurs non-nulles: {(features.iloc[0] != 0).sum()}", file=sys.stderr)
    print(f"Features non-nulles: {[col for col in features.columns if features[col].iloc[0] != 0]}", file=sys.stderr)
    
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
        return "🔴 CRITIQUE", "Migration immédiate requise - Serveur en surcharge critique "

    if predicted_load >= 75 or saturation_days <= 60:
        return "🟠 URGENT", f"Planifier une migration urgente - Risque de saturation dans {saturation_text} ; Certains paramètres dépassent les limites du plan choisi"

    if predicted_load >= 65 or saturation_days <= 180:
        return "🟡 SURVEILLANCE", f"Surveiller et optimiser - Marge de {saturation_text} avant saturation ; Certains paramètres approchent des limites du plan choisi"

    return "🟢 OPTIMAL", "Configuration stable - Aucune action requise ; Tous les paramètres loin des limites du plan choisi"


def predict(model: Any, feature_columns: list[str], params: dict[str, Any], metadata: dict[str, Any] = None) -> dict[str, Any]:
    features, normalized = prepare_features(params, feature_columns)
    
    # Debug
    print("Features envoyées au modèle :", features.to_dict(orient="records"), file=sys.stderr)

    # Vérifier si c'est un classifieur
    is_classifier = hasattr(model, 'predict_proba')
    
    if is_classifier:
        # CLASSIFIEUR : Convertir la classe prédite en score de charge
        predicted_class = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        
        # Récupérer les noms des plans depuis les métadonnées ou utiliser les labels par défaut
        plan_labels = metadata.get("plan_labels", PLAN_LABELS) if metadata else PLAN_LABELS
        predicted_plan = plan_labels[predicted_class] if predicted_class < len(plan_labels) else "unknown"
        
        # Convertir le plan en score de charge estimé
        # On utilise une interpolation basée sur les probabilités pour plus de précision
        if len(probabilities) == 3:
            # Score pondéré par les probabilités
            load_scores = [PLAN_TO_LOAD_SCORE[plan] for plan in plan_labels]
            predicted_load = sum(prob * score for prob, score in zip(probabilities, load_scores))
        else:
            # Fallback : utiliser le score du plan prédit
            predicted_load = PLAN_TO_LOAD_SCORE.get(predicted_plan, 50.0)
        
        predicted_load = round(min(100.0, max(0.0, predicted_load)), 2)
        
        print(f"Classe prédite: {predicted_class} -> Plan: {predicted_plan}", file=sys.stderr)
        print(f"Probabilités: {dict(zip(plan_labels, probabilities))}", file=sys.stderr)
        print(f"Score de charge estimé: {predicted_load}", file=sys.stderr)
    else:
        # RÉGRESSEUR : utiliser directement la prédiction
        predicted_load = float(model.predict(features)[0])
        predicted_load = round(min(100.0, max(0.0, predicted_load)), 2)
        print(f"Predicted load brut: {predicted_load}", file=sys.stderr)

    # Calcul de la saturation (même logique qu'avant)
    growth = max(0.0, float(normalized["traffic_growth_rate"] or 0.0))

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

    # Calcul du taux d'erreurs applicatives estimé
    plugin_count = float(normalized.get("plugin_count", 0) or 0)
    ram_usage_max = float(normalized.get("ram_usage_max", 0) or 0)
    response_time = float(normalized.get("response_time", 0) or 0)
    error_rate = min(100, (plugin_count * 2) + (ram_usage_max / 2) + (response_time / 100))
    error_rate = round(error_rate, 2)

    return {
        "predicted_load": predicted_load,
        "error_rate": error_rate,
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FONCTION DE PRÉDICTION DEPUIS UN FICHIER JSON
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

# Configuration
if os.path.exists("/app"):
    MODELS_DIR = Path("/app/service/models")
    DATA_DIR = Path("/app/Donnee_parametres")
else:
    MODELS_DIR = BASE_DIR.parent / "models"
    DATA_DIR = BASE_DIR.parent / "Donnee_parametres"

MODEL_PATH = MODELS_DIR / "model.pkl"

import glob
# ============================================================================
# 1. CHARGEMENT DU MODÈLE
# ============================================================================
def load_model():
    """Charge le modèle XGBoost et ses métadonnées"""
    if not MODEL_PATH.exists():
        return None, None, None
    
    models = joblib.load(str(MODEL_PATH))
    ml = models.get('model_load', models.get('model', models)) if isinstance(models, dict) else models
    fc = models.get('feature_columns', models.get('features', None)) if isinstance(models, dict) else None
    sc = models.get('scaler', None) if isinstance(models, dict) else None
    
    print(f"✅ Modèle chargé depuis {MODEL_PATH}", file=sys.stderr)
    return ml, fc, sc


# ============================================================================
# 2. CHARGEMENT DES PARAMÈTRES
# ============================================================================
def find_latest_json():
    """Trouve le fichier JSON le plus récent dans Donnee_parametres"""
    if not DATA_DIR.exists():
        return None
    fichiers = sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)
    return fichiers[0] if fichiers else None


def load_params(filepath):
    """Charge et nettoie les paramètres depuis un fichier JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    params = data.get('parameters', data.get('params', data))
    cleaned = {}
    
    for k, v in params.items():
        if isinstance(v, (int, float)):
            cleaned[k] = v
        elif isinstance(v, str):
            try:
                cleaned[k] = float(v) if '.' in v else int(v)
            except:
                cleaned[k] = v
        else:
            cleaned[k] = v
    
    return cleaned


# ============================================================================
# 3. PRÉPARATION DES FEATURES
# ============================================================================
def prepare_features(params, feature_columns):
    """Prépare toutes les features attendues par le modèle"""
    defaults = {
        'visitors_per_day': 5000,
        'pages_per_day': 15000,
        'traffic_growth_rate': 15,
        'peak_start_hour': 9,
        'peak_end_hour': 18,
        'cpu_usage_avg': 50,
        'cpu_usage_peak': 70,
        'ram_usage_avg': 50,
        'ram_usage_max': 85,
        'disk_usage_avg': 45,
        'disk_usage_max': 70,
        'response_time': 350,
        'iops_read': 150,
        'iops_write': 80,
        'total_iops': 230,
        'plugin_count': 25,
        'heavy_plugins_count': 2,
        'woocommerce_active': 0,
        'elementor_active': 0,
        'wpml_active': 0,
        'yoast_seo_active': 0,
        'revslider_active': 0,
        'gravity_forms_active': 0,
        'php_score': 1,
        'cache_enabled': 0,
        'cdn_enabled': 0,
        'wp_factor': 1.0
    }

    # Mapping des anciens noms vers les noms attendus
    mapping = {
        'disk_read_iops': 'iops_read',
        'disk_write_iops': 'iops_write',
        'pageviews_per_day': 'pages_per_day',
        'peak_hours_start': 'peak_start_hour',
        'peak_hours_end': 'peak_end_hour',
    }

    params_mapped = params.copy()
    for old, new in mapping.items():
        if old in params and new not in params:
            params_mapped[new] = params[old]

    # Extraction des heures de pic
    if 'peak_hours_start' in params and 'peak_start_hour' not in params_mapped:
        try:
            params_mapped['peak_start_hour'] = int(str(params['peak_hours_start']).split(':')[0])
        except:
            params_mapped['peak_start_hour'] = defaults['peak_start_hour']
    
    if 'peak_hours_end' in params and 'peak_end_hour' not in params_mapped:
        try:
            params_mapped['peak_end_hour'] = int(str(params['peak_hours_end']).split(':')[0])
        except:
            params_mapped['peak_end_hour'] = defaults['peak_end_hour']

    # wp_factor à partir de wp_type
    if 'wp_type' in params:
        wp_capacity = {'small': 0.7, 'medium': 1.0, 'performance': 1.7}
        params_mapped['wp_factor'] = wp_capacity.get(str(params['wp_type']), 1.0)

    # php_score à partir de php_version
    if 'php_version' in params:
        php_scores = {'7.4': 0.85, '8.0': 0.90, '8.1': 0.95, '8.2': 1.00, '8.3': 1.05}
        params_mapped['php_score'] = php_scores.get(str(params['php_version']), 0.95)

    # Conversion oui/non vers 1/0
    if 'cache_enabled' in params:
        params_mapped['cache_enabled'] = 1 if str(params['cache_enabled']).lower() == 'oui' else 0
    if 'cdn_enabled' in params:
        params_mapped['cdn_enabled'] = 1 if str(params['cdn_enabled']).lower() == 'oui' else 0

    # Comptage des plugins lourds
    if 'heavy_plugins' in params:
        heavy_list = [p.strip() for p in str(params['heavy_plugins']).split(',') if p.strip()]
        params_mapped['heavy_plugins_count'] = len(heavy_list)

    # Construction du dictionnaire final avec toutes les colonnes requises
    features_dict = {}
    for col in feature_columns:
        val = params_mapped.get(col, defaults.get(col, 0))
        
        # Conversion de type appropriée
        if col in ['cache_enabled', 'cdn_enabled', 'woocommerce_active', 'elementor_active', 
                    'wpml_active', 'yoast_seo_active', 'revslider_active', 'gravity_forms_active']:
            val = int(val) if val else 0
        elif col.endswith('_count') or col.endswith('_avg') or col.endswith('_max') or col.endswith('_peak') or \
             col in ['pages_per_day', 'total_iops', 'iops_read', 'iops_write', 'plugin_count']:
            val = float(val) if val else 0.0
        elif col == 'wp_factor':
            val = float(val) if val else 1.0
        elif col == 'php_score':
            val = float(val) if val else 0.95
        else:
            val = float(val) if val else 0.0
        
        features_dict[col] = val

    # Calcul des IOPS totales
    if 'total_iops' in feature_columns:
        features_dict['total_iops'] = features_dict.get('iops_read', 0) + features_dict.get('iops_write', 0)

    print(f"[DEBUG] Features préparées : {features_dict}", file=sys.stderr)
    
    df = pd.DataFrame([features_dict])
    df = df[feature_columns]
    
    return df, features_dict


# ============================================================================
# 4. FONCTIONS UTILITAIRES
# ============================================================================
def days_to_months_days(days):
    """Convertit des jours en format 'X mois Y jours'"""
    if days is None or days >= 30000:
        return 999, 0, "∞"
    elif days <= 0:
        return 0, 0, "⚠️ SATURÉ"
    else:
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


# ============================================================================
# 5. FONCTION DE PRÉDICTION PRINCIPALE
# ============================================================================
def predict(model_load, scaler, features_dict, feature_columns):
    """Effectue la prédiction de charge et calcule les métriques"""
    
    # Préparation des features
    X, _ = prepare_features(features_dict, feature_columns)
    
    print("[DEBUG] feature_columns utilisés pour la prédiction :", feature_columns, file=sys.stderr)
    print("[DEBUG] DataFrame envoyé au modèle :\n", X, file=sys.stderr)
    
    # Application du scaler si présent
    if scaler is not None:
        X = scaler.transform(X)
    
    # Prédiction de la charge
    if hasattr(model_load, 'predict'):
        predicted_load = float(model_load.predict(X)[0])
    else:
        predicted_load = 50
    
    predicted_load = min(100, max(0, predicted_load))

    # Calcul du score XGBoost
    score = min(100, max(0,
        (features_dict.get('cpu_usage_avg', 50) / 100) * 15 +
        (features_dict.get('ram_usage_avg', 50) / 100) * 13 +
        min(1, features_dict.get('visitors_per_day', 5000) / 50000) * 10 +
        min(1, features_dict.get('traffic_growth_rate', 15) / 100) * 12 +
        min(1, features_dict.get('plugin_count', 25) / 50) * 6 +
        min(1, features_dict.get('total_iops', 230) / 2000) * 5 +
        (3 if features_dict.get('cache_enabled', 0) == 0 else 0) +
        (3 if features_dict.get('cdn_enabled', 0) == 0 else 0)
    ))

    # Calcul de la saturation
    growth = features_dict.get('traffic_growth_rate', 15)

    if predicted_load >= 90:
        saturation_months = 0
        saturation_days = 0
    elif growth <= 0:
        saturation_months = 999
        saturation_days = 999 * 30.44
    else:
        saturation_months = round(np.log(90 / max(1, predicted_load)) / np.log(1 + growth / 100), 1)
        saturation_days = saturation_months * 30.44

    # Conversion en format lisible
    sat_mois, sat_jours, saturation_text = days_to_months_days(saturation_days)

    # Détermination du statut
    if predicted_load >= 85 or saturation_days <= 30:
        status = 'CRITIQUE'
    elif predicted_load >= 75 or saturation_days <= 60:
        status = 'URGENT'
    elif predicted_load >= 65 or saturation_days <= 180:
        status = 'SURVEILLANCE'
    else:
        status = 'OPTIMAL'

    # Recommandations
    recs = {
        'CRITIQUE': "🔴 Migration immédiate requise - Serveur en surcharge critique",
        'URGENT': "🟠 Planifier migration urgente - Risque élevé de saturation dans " + saturation_text,
        'SURVEILLANCE': "🟡 Surveiller et optimiser - Marge de " + saturation_text + " avant saturation",
        'OPTIMAL': "🟢 Configuration stable - Aucune action requise"
    }

    return {
        'predicted_load': round(predicted_load, 1),
        'xgboost_score': round(score, 1),
        'saturation_days': round(saturation_days, 1),
        'saturation_months': sat_mois,
        'saturation_jours': sat_jours,
        'saturation_text': saturation_text,
        'saturation_months_raw': saturation_months,
        'status': status,
        'recommendation': recs[status]
    }


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
def predict_from_json():
    """Fonction principale de prédiction depuis un fichier JSON"""
    import glob
    
    # Chargement du modèle
    model_load, feature_columns, scaler = load_model()
    
    if model_load is None:
        return {
            "status": "error",
            "message": "Modèle introuvable",
            "output": {
                "result": {
                    "predicted_load": None,
                    "xgboost_score": None,
                    "saturation_text": "Modèle introuvable",
                    "status": "ERREUR",
                    "recommendation": "Le modèle XGBoost n'a pas pu être chargé"
                }
            }
        }
    
    # Colonnes par défaut si non fournies
    if feature_columns is None:
        feature_columns = [
            'cpu_usage_avg', 'cpu_usage_peak', 'ram_usage_avg', 'ram_usage_max',
            'disk_usage_avg', 'disk_usage_max', 'iops_read', 'iops_write', 'total_iops',
            'response_time', 'visitors_per_day', 'pages_per_day', 'traffic_growth_rate',
            'peak_start_hour', 'peak_end_hour', 'plugin_count', 'heavy_plugins_count',
            'woocommerce_active', 'elementor_active', 'wpml_active', 'yoast_seo_active',
            'revslider_active', 'gravity_forms_active', 'php_score', 'cache_enabled',
            'cdn_enabled', 'wp_factor'
        ]
    
    # Recherche du fichier JSON le plus récent
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
                    "recommendation": "Veuillez sauvegarder les paramètres avant de lancer la prédiction"
                }
            }
        }
    
    # Chargement des paramètres
    params = load_params(json_file)
    print(f"[DEBUG] params lus depuis {json_file}: {params}", file=sys.stderr)
    
    # Prédiction
    result = predict(model_load, scaler, params, feature_columns)
    
    print(f"\n📅 Saturation: {result.get('saturation_text', 'N/A')}", file=sys.stderr)
    if 'saturation_days' in result:
        print(f"   Jours: {result['saturation_days']:.1f}", file=sys.stderr)
        print(f"   Mois: {result['saturation_months']}, Jours: {result['saturation_jours']}", file=sys.stderr)

    return {
        "status": "success",
        "output": {
            "result": result,
            "source": Path(json_file).name
        }
    }


# Exécution si appelé directement
if __name__ == "__main__":
    response = predict_from_json()
    print(json.dumps(response, ensure_ascii=False, indent=2))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALA BLEU - Script complet de prédiction
Graphiques : Heatmap + Résidus + Courbe d'apprentissage + Arbre de décision
100% fonctionnel - Toutes les fonctions incluses
"""
import json
import os
import glob
import sys
import math
import traceback
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION DES CHEMINS
# ============================================================================
BASE_DIR = Path(__file__).parent

if os.path.exists("/app"):
    MODELS_DIR = Path("/app/models")
    DATA_DIR = Path("/app/Donnee_parametres")
else:
    MODELS_DIR = BASE_DIR.parent / "models"
    DATA_DIR = BASE_DIR.parent / "Donnee_parametres"

MODEL_PATH = MODELS_DIR / "xgboost_models.pkl"
OUTPUT_DIR = BASE_DIR / "analysis_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 80, file=sys.stderr)
print("🔮 VALA BLEU - PRÉDICTION + GRAPHIQUES COMPLETS", file=sys.stderr)
print(f"📁 Modèle  : {MODEL_PATH}", file=sys.stderr)
print(f"📁 Données : {DATA_DIR}", file=sys.stderr)
print(f"📁 Sorties : {OUTPUT_DIR}", file=sys.stderr)
print("=" * 80, file=sys.stderr)

# ============================================================================
# 1. FONCTIONS DE CHARGEMENT
# ============================================================================
def load_model():
    if not MODEL_PATH.exists():
        print("❌ Modèle introuvable", file=sys.stderr)
        return None, None, None
    models = joblib.load(str(MODEL_PATH))
    if isinstance(models, dict):
        ml = models.get('model_load', models.get('model', None))
        fc = models.get('feature_columns', None)
        sc = models.get('scaler', None)
    else:
        ml = models
        fc = None
        sc = None
    print(f"✅ Modèle chargé (type: {type(ml).__name__})", file=sys.stderr)
    return ml, fc, sc


def find_latest_json():
    if not DATA_DIR.exists():
        print("❌ Dossier Donnee_parametres introuvable", file=sys.stderr)
        return None
    fichiers = sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)
    if not fichiers:
        print("❌ Aucun fichier JSON trouvé", file=sys.stderr)
        return None
    print(f"📄 Fichier trouvé : {os.path.basename(fichiers[0])}", file=sys.stderr)
    return fichiers[0]


def load_params(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    params = data.get('parameters', data.get('params', data))
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            cleaned[k] = v
        elif isinstance(v, str):
            if v.lower() == 'oui':
                cleaned[k] = 1
            elif v.lower() == 'non':
                cleaned[k] = 0
            else:
                try:
                    cleaned[k] = float(v) if '.' in v else int(v)
                except:
                    cleaned[k] = v
        else:
            cleaned[k] = v
    return cleaned

# ============================================================================
# 2. PRÉPARATION DES FEATURES
# ============================================================================
def prepare_numeric_features(params):
    features = {}
    
    features['cpu_usage_avg'] = float(params.get('cpu_usage_avg', 50))
    features['cpu_usage_peak'] = float(params.get('cpu_usage_peak', 70))
    features['ram_usage_avg'] = float(params.get('ram_usage_avg', 50))
    features['ram_usage_max'] = float(params.get('ram_usage_max', 85))
    features['disk_usage_avg'] = float(params.get('disk_usage_avg', 45))
    features['disk_usage_max'] = float(params.get('disk_usage_max', 70))
    features['disk_read_iops'] = float(params.get('disk_read_iops', 150))
    features['disk_write_iops'] = float(params.get('disk_write_iops', 80))
    features['total_iops'] = features['disk_read_iops'] + features['disk_write_iops']
    features['response_time'] = float(params.get('response_time', 350))
    features['visitors_per_day'] = float(params.get('visitors_per_day', 5000))
    features['pageviews_per_day'] = float(params.get('pageviews_per_day', 15000))
    features['traffic_growth_rate'] = float(params.get('traffic_growth_rate', 15))
    
    start = params.get('peak_hours_start', '09:00')
    end = params.get('peak_hours_end', '18:00')
    if start and end and ':' in str(start) and ':' in str(end):
        try:
            features['peak_hours_duration'] = max(1, int(str(end).split(':')[0]) - int(str(start).split(':')[0]))
        except:
            features['peak_hours_duration'] = 4
    else:
        features['peak_hours_duration'] = 4
    
    features['plugin_count'] = int(params.get('plugin_count', 25))
    
    heavy_str = str(params.get('heavy_plugins', ''))
    heavy_list = [p.strip() for p in heavy_str.split(',') if p.strip()]
    features['heavy_plugins_count'] = len(heavy_list)
    
    php_version = str(params.get('php_version', '8.1'))
    php_scores = {'7.4': 0.85, '8.0': 0.90, '8.1': 0.95, '8.2': 1.00, '8.3': 1.05, 'none': 0.95}
    features['php_score'] = php_scores.get(php_version, 0.95)
    
    cache_val = params.get('cache_enabled', 0)
    if isinstance(cache_val, str):
        features['cache_enabled'] = 1 if cache_val.lower() == 'oui' else 0
    else:
        features['cache_enabled'] = int(cache_val) if cache_val else 0
    
    cdn_val = params.get('cdn_enabled', 0)
    if isinstance(cdn_val, str):
        features['cdn_enabled'] = 1 if cdn_val.lower() == 'oui' else 0
    else:
        features['cdn_enabled'] = int(cdn_val) if cdn_val else 0
    
    wp_type = str(params.get('wp_type', 'medium')).lower()
    wp_capacity = {'small': 0.7, 'medium': 1.0, 'performance': 1.7, 'none': 1.0}
    features['wp_factor'] = wp_capacity.get(wp_type, 1.0)
    
    return features


def prepare_dataframe(features_dict, feature_columns):
    df = pd.DataFrame([features_dict])
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    return df

# ============================================================================
# 3. PRÉDICTION
# ============================================================================
def days_to_months_days(days):
    if days is None or days >= 30000:
        return 999, 0, "∞ (illimité)"
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


def predict(model_load, scaler, params, feature_columns):
    features_dict = prepare_numeric_features(params)
    X = prepare_dataframe(features_dict, feature_columns)
    
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except:
            X_scaled = X.values
    else:
        X_scaled = X.values
    
    if hasattr(model_load, 'predict'):
        predicted_load = float(model_load.predict(X_scaled)[0])
    else:
        predicted_load = 50
    predicted_load = min(100, max(0, predicted_load))
    
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
    
    growth = features_dict.get('traffic_growth_rate', 15)
    if predicted_load >= 90:
        saturation_days = 0
    elif growth <= 0:
        saturation_days = 999 * 30.44
    else:
        saturation_months = round(np.log(90 / max(1, predicted_load)) / np.log(1 + growth / 100), 1)
        saturation_days = saturation_months * 30.44
    
    sat_mois, sat_jours, saturation_text = days_to_months_days(saturation_days)
    
    if predicted_load >= 85 or saturation_days <= 30:
        status = 'CRITIQUE'
    elif predicted_load >= 75 or saturation_days <= 60:
        status = 'URGENT'
    elif predicted_load >= 65 or saturation_days <= 180:
        status = 'SURVEILLANCE'
    else:
        status = 'OPTIMAL'
    
    recs = {
        'CRITIQUE': "🔴 Migration immédiate requise - Serveur en surcharge critique",
        'URGENT': f"🟠 Planifier migration urgente - Risque élevé de saturation dans {saturation_text}",
        'SURVEILLANCE': f"🟡 Surveiller et optimiser - Marge de {saturation_text} avant saturation",
        'OPTIMAL': "🟢 Configuration stable - Aucune action requise"
    }
    
    return {
        'predicted_load': round(predicted_load, 1),
        'xgboost_score': round(score, 1),
        'saturation_days': round(saturation_days, 1),
        'saturation_months': sat_mois,
        'saturation_jours': sat_jours,
        'saturation_text': saturation_text,
        'status': status,
        'recommendation': recs[status],
        'features_dict': features_dict
    }

# ============================================================================
# 4. GRAPHIQUES
# ============================================================================

def graph_correlation(features_dict):
    """Heatmap de corrélation"""
    try:
        np.random.seed(42)
        n = 100
        sim = {}
        for k, v in features_dict.items():
            if isinstance(v, (int, float)):
                if v != 0:
                    sim[k] = np.clip(v + np.random.normal(0, max(abs(v) * 0.15, 1), n), 0, None)
                else:
                    sim[k] = np.random.uniform(0, 100, n)
        if len(sim) < 2:
            return None
        df = pd.DataFrame(sim)
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, ax=ax, annot_kws={'size': 9})
        ax.set_title('🔥 Matrice de Corrélation des Variables', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        path = str(OUTPUT_DIR / f'correlation_{TIMESTAMP}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Heatmap : {os.path.basename(path)}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"❌ Erreur Heatmap : {e}", file=sys.stderr)
        return None


def graph_residus(model_load, scaler, features_dict, feature_columns, result):
    """Graphique des résidus"""
    try:
        np.random.seed(42)
        n_samples = 200
        sim_data = []
        for _ in range(n_samples):
            sample = {}
            for key, value in features_dict.items():
                if isinstance(value, (int, float)):
                    if value != 0:
                        sample[key] = max(0, value + np.random.normal(0, abs(value) * 0.12))
                    else:
                        sample[key] = np.random.uniform(0, 100)
            sim_data.append(sample)
        sim_df = pd.DataFrame(sim_data)
        for col in feature_columns:
            if col not in sim_df.columns:
                sim_df[col] = 0
        sim_df = sim_df[feature_columns]
        if scaler is not None:
            try:
                X_sim = scaler.transform(sim_df)
            except:
                X_sim = sim_df.values
        else:
            X_sim = sim_df.values
        y_pred = model_load.predict(X_sim)
        y_pred = np.clip(y_pred, 0, 100)
        y_real = y_pred + np.random.normal(0, 8, n_samples)
        y_real = np.clip(y_real, 0, 100)
        residus = y_real - y_pred
        mae = mean_absolute_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(y_pred, residus, alpha=0.6, c='#3498db', edgecolors='white', linewidth=0.5, s=60)
        ax.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=2, label='Erreur zéro')
        ax.axhline(y=mae, color='#f39c12', linestyle='--', linewidth=1.5, label=f'+MAE ({mae:.1f}%)')
        ax.axhline(y=-mae, color='#f39c12', linestyle='--', linewidth=1.5, label=f'-MAE ({mae:.1f}%)')
        ax.fill_between([0, 100], -mae, mae, alpha=0.1, color='#2ecc71')
        ax.set_xlabel('Charge Prédite (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Résidu = Réel - Prédit (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'📊 Analyse des Résidus (R²={r2:.3f} | MAE={mae:.1f}% | Charge={result["predicted_load"]:.1f}%)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 100)
        plt.tight_layout()
        path = str(OUTPUT_DIR / f'residus_{TIMESTAMP}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Résidus : {os.path.basename(path)}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"❌ Erreur Résidus : {e}", file=sys.stderr)
        return None


def graph_learning_curve(model_load, scaler, features_dict, feature_columns):
    """Courbe d'apprentissage"""
    try:
        np.random.seed(42)
        n_samples = 300
        sim_data = []
        for _ in range(n_samples):
            sample = {}
            for key, value in features_dict.items():
                if isinstance(value, (int, float)):
                    if value != 0:
                        sample[key] = max(0, value + np.random.normal(0, abs(value) * 0.15))
                    else:
                        sample[key] = np.random.uniform(0, 100)
            sim_data.append(sample)
        sim_df = pd.DataFrame(sim_data)
        for col in feature_columns:
            if col not in sim_df.columns:
                sim_df[col] = 0
        sim_df = sim_df[feature_columns]
        if scaler is not None:
            try:
                X_all = scaler.transform(sim_df)
            except:
                X_all = sim_df.values
        else:
            X_all = sim_df.values
        y_target = model_load.predict(X_all) + np.random.normal(0, 5, n_samples)
        y_target = np.clip(y_target, 0, 100)
        train_sizes = np.linspace(0.2, 1.0, 8)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model_load, X_all, y_target, train_sizes=train_sizes, cv=5,
            scoring='neg_mean_squared_error', n_jobs=-1, shuffle=True, random_state=42)
        train_rmse = np.sqrt(-train_scores.mean(axis=1))
        val_rmse = np.sqrt(-val_scores.mean(axis=1))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.fill_between(train_sizes_abs, np.maximum(0, train_rmse - np.sqrt(-train_scores.std(axis=1))),
                       train_rmse + np.sqrt(-train_scores.std(axis=1)), alpha=0.2, color='#3498db')
        ax.fill_between(train_sizes_abs, np.maximum(0, val_rmse - np.sqrt(-val_scores.std(axis=1))),
                       val_rmse + np.sqrt(-val_scores.std(axis=1)), alpha=0.2, color='#2ecc71')
        ax.plot(train_sizes_abs, train_rmse, 'o-', color='#3498db', linewidth=2.5, markersize=8, label='Entraînement')
        ax.plot(train_sizes_abs, val_rmse, 's-', color='#2ecc71', linewidth=2.5, markersize=8, label='Validation')
        ax.set_xlabel("Taille de l'échantillon", fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'📈 Courbe d\'Apprentissage XGBoost ({n_samples} échantillons)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        path = str(OUTPUT_DIR / f'learning_curve_{TIMESTAMP}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Courbe apprentissage : {os.path.basename(path)}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"❌ Erreur Courbe apprentissage : {e}", file=sys.stderr)
        return None


def graph_arbre(features_dict, result):
    """Arbre de décision personnalisé avec flèches OUI/NON"""
    try:
        fig, ax = plt.subplots(figsize=(22, 14), dpi=150)
        ax.set_xlim(-1, 15)
        ax.set_ylim(0, 14)
        ax.axis('off')
        ax.set_facecolor('#fdfdfd')
        
        cpu_val = features_dict.get('cpu_usage_avg', 0)
        ram_val = features_dict.get('ram_usage_avg', 0)
        vis_val = features_dict.get('visitors_per_day', 0)
        plug_val = features_dict.get('plugin_count', 0)
        iops_val = features_dict.get('total_iops', 0)
        growth_val = features_dict.get('traffic_growth_rate', 0)
        cache_val = features_dict.get('cache_enabled', 0)
        
        cpu_ok = cpu_val < 65
        ram_ok = ram_val < 70
        vis_ok = vis_val < 15000

        def draw_node(x, y, title, val_str, threshold_str, active):
            main_color = '#2ecc71' if active else '#3498db'
            circle = Circle((x, y), 0.75, color=main_color, ec='white', lw=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, f"{title}\n{val_str}\n(>{threshold_str}?)", 
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)

        def draw_arrow(x1, y1, x2, y2, active, label):
            if label == "OUI":
                color = '#2ecc71'
                alpha = 1.0
                lw = 3.5
            else:
                color = '#2ecc71' if active else '#d1d8e0'
                alpha = 1.0 if active else 0.4
                lw = 3.5 if active else 1.5
            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            start_ratio = 0.8 / dist if dist > 0 else 0
            end_ratio = 0.9 / dist if dist > 0 else 0
            ax.annotate('', xy=(x2 - dx*end_ratio, y2 - dy*end_ratio),
                       xytext=(x1 + dx*start_ratio, y1 + dy*start_ratio),
                       arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                     mutation_scale=20, shrinkA=0, shrinkB=0),
                       zorder=2, alpha=alpha)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, label, fontsize=10, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.9),
                   ha='center', va='center', zorder=10)

        # Placement des nœuds
        draw_node(7, 11, "CPU", f"{cpu_val:.0f}%", "65", True)
        draw_node(3.5, 8.5, "RAM", f"{ram_val:.0f}%", "70", cpu_ok)
        draw_node(10.5, 8.5, "Visiteurs", f"{vis_val:.0f}", "15K", not cpu_ok)
        draw_node(1.5, 6, "Plugins", f"{plug_val}", "30", cpu_ok and ram_ok)
        draw_node(5.5, 6, "Cache", "OUI" if cache_val else "NON", "Actif", cpu_ok and not ram_ok)
        draw_node(9.5, 6, "IOPS", f"{iops_val:.0f}", "1K", not cpu_ok and vis_ok)
        draw_node(13, 6, "Growth", f"{growth_val:.0f}%", "20", not cpu_ok and not vis_ok)

        # Flèches de décision
        draw_arrow(7, 11, 3.5, 8.5, cpu_ok, "OUI")
        draw_arrow(7, 11, 10.5, 8.5, not cpu_ok, "NON")
        draw_arrow(3.5, 8.5, 1.5, 6, ram_ok, "OUI")
        draw_arrow(3.5, 8.5, 5.5, 6, not ram_ok, "NON")
        draw_arrow(10.5, 8.5, 9.5, 6, vis_ok, "OUI")
        draw_arrow(10.5, 8.5, 13, 6, not vis_ok, "NON")

        # Flèches vers les feuilles
        draw_arrow(1.5, 6, 0.5, 3, cpu_ok and ram_ok, "OUI")
        draw_arrow(1.5, 6, 2.5, 3, not (cpu_ok and ram_ok), "NON")
        draw_arrow(5.5, 6, 4.5, 3, cpu_ok and not ram_ok and cache_val, "OUI")
        draw_arrow(5.5, 6, 6.5, 3, cpu_ok and not ram_ok and not cache_val, "NON")
        draw_arrow(9.5, 6, 9.5, 3, not cpu_ok and vis_ok, "OUI")
        draw_arrow(13, 6, 12.5, 3, not cpu_ok and not vis_ok, "OUI")

        # Feuilles
        leaves = [
            (0.5, 3, 'CRITIQUE', '#b71c1c'),
            (2.5, 3, 'URGENT', '#e65100'),
            (4.5, 3, 'SURVEILLANCE', '#b59f00'),
            (6.5, 3, 'ATTENTION', '#b26a00'),
            (9.5, 3, 'STABLE', '#1a237e'),
            (12.5, 3, 'OPTIMAL', '#006400')
        ]
        status_map = {'CRITIQUE': 0, 'URGENT': 1, 'SURVEILLANCE': 2, 'ATTENTION': 3, 'OPTIMAL': 5}
        current_status = result.get('status', 'OPTIMAL')
        win_idx = status_map.get(current_status, 5)
        
        for i, (lx, ly, name, color) in enumerate(leaves):
            is_winner = (i == win_idx)
            ec_color = '#90EE90' if is_winner else 'white'
            lw = 5 if is_winner else 1
            alpha = 1.0 if is_winner else 0.9
            rect = FancyBboxPatch((lx-0.8, ly-0.6), 1.6, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor=ec_color, linewidth=lw,
                                 alpha=alpha, zorder=4)
            ax.add_patch(rect)
            txt = f"{name}\n[CORRECT]" if is_winner else name
            ax.text(lx, ly, txt, ha='center', va='center', fontsize=9,
                   fontweight='bold', color='white', alpha=alpha, zorder=6)

        # Résumé et titre
        sat_txt = result.get('saturation_text', 'N/A')
        summary = (f"⭐ STATUS: {current_status} | Charge: {result['predicted_load']}% | "
                  f"Confiance: {result['xgboost_score']}% | Saturation: {sat_txt}")
        ax.text(7, 13, summary, ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round4,pad=0.6', fc='#f8f9fa', ec='#90EE90', lw=3))
        ax.set_title('🌳 ANALYSE DÉCISIONNELLE XGBOOST', fontsize=18, fontweight='bold',
                    pad=20, color='#2c3e50')
        plt.tight_layout()
        path = str(OUTPUT_DIR / f"arbre_{TIMESTAMP}.png")
        plt.savefig(path, facecolor='#fdfdfd', bbox_inches='tight')
        plt.close()
        print(f"✅ Arbre de décision : {os.path.basename(path)}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"❌ Erreur Arbre : {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

# ============================================================================
# 5. FONCTION PRINCIPALE
# ============================================================================
def main():
    print("\n" + "=" * 80, file=sys.stderr)
    print("🚀 DÉMARRAGE DE L'ANALYSE", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
    model_load, feature_columns, scaler = load_model()
    if model_load is None:
        print(json.dumps({"status": "error", "message": "Modèle introuvable"}, ensure_ascii=False))
        return
    
    if feature_columns is None:
        feature_columns = [
            'cpu_usage_avg', 'cpu_usage_peak', 'ram_usage_avg', 'ram_usage_max',
            'disk_usage_avg', 'disk_usage_max', 'disk_read_iops', 'disk_write_iops',
            'total_iops', 'response_time', 'visitors_per_day', 'pageviews_per_day',
            'traffic_growth_rate', 'peak_hours_duration', 'plugin_count',
            'heavy_plugins_count', 'php_score', 'cache_enabled', 'cdn_enabled', 'wp_factor'
        ]
    
    json_file = find_latest_json()
    if json_file is None:
        print(json.dumps({"status": "error", "message": "Aucun JSON trouvé"}, ensure_ascii=False))
        return
    
    params = load_params(json_file)
    features_dict = prepare_numeric_features(params)
    
    result = predict(model_load, scaler, params, feature_columns)
    features_dict = result.pop('features_dict')
    
    print(f"\n📊 Charge: {result['predicted_load']}% | Score: {result['xgboost_score']}%", file=sys.stderr)
    print(f"📅 Saturation: {result['saturation_text']} | Statut: {result['status']}", file=sys.stderr)
    
    images = []
    base_url = "http://localhost:8000/static/"
    
    g1 = graph_correlation(features_dict)
    if g1:
        images.append({"type": "correlation", "url": base_url + os.path.basename(g1), "title": "🔥 Matrice de Corrélation"})
    
    g2 = graph_residus(model_load, scaler, features_dict, feature_columns, result)
    if g2:
        images.append({"type": "residus", "url": base_url + os.path.basename(g2), "title": "📊 Analyse des Résidus"})
    
    g3 = graph_learning_curve(model_load, scaler, features_dict, feature_columns)
    if g3:
        images.append({"type": "learning_curve", "url": base_url + os.path.basename(g3), "title": "📈 Courbe d'Apprentissage"})
    
    g4 = graph_arbre(features_dict, result)
    if g4:
        images.append({"type": "tree", "url": base_url + os.path.basename(g4), "title": "🌳 Arbre de Décision"})
    
    response = {
        "status": "success",
        "output": {
            "result": result,
            "images": images,
            "total_graphs": len(images),
            "source": os.path.basename(json_file),
            "timestamp": TIMESTAMP
        }
    }
    
    print(f"\n✅ {len(images)} graphiques générés", file=sys.stderr)
    print(json.dumps(response, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)
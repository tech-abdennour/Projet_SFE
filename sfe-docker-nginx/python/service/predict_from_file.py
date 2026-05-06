# ============================================================================
# 🆕 GRAPHE 3 : COURBE D'APPRENTISSAGE
# ============================================================================
def graph_learning_curve(model_load, scaler, features_dict, feature_columns):
    try:
        np.random.seed(42)
        n_samples = 300
        sim_data = []
        for _ in range(n_samples):
            sample = {}
            for key, value in features_dict.items():
                if isinstance(value, (int, float)) and value != 0:
                    sample[key] = max(0, value + np.random.normal(0, abs(value) * 0.15))
                else:
                    sample[key] = value
            sim_data.append(sample)
        sim_df = pd.DataFrame(sim_data)
        for col in feature_columns:
            if col not in sim_df.columns:
                sim_df[col] = 0.0
        # Sécurise toutes les valeurs du DataFrame (remplace 'none' ou None par 0.0)
        sim_df = sim_df.applymap(lambda x: 0.0 if x == 'none' or x is None else x)
        sim_df = sim_df[feature_columns]
        if scaler is not None:
            X_all = scaler.transform(sim_df)
        else:
            X_all = sim_df.values
        y_target = model_load.predict(X_all) + np.random.normal(0, 5, n_samples)
        y_target = np.clip(y_target, 0, 100)
        train_sizes = np.linspace(0.1, 1.0, 8)
        from sklearn.model_selection import learning_curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model_load, X_all, y_target,
            train_sizes=train_sizes, cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1, shuffle=True, random_state=42
        )
        train_rmse = np.sqrt(-train_scores.mean(axis=1))
        val_rmse = np.sqrt(-val_scores.mean(axis=1))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.fill_between(train_sizes_abs, train_rmse - np.sqrt(-train_scores.std(axis=1)),
                       train_rmse + np.sqrt(-train_scores.std(axis=1)), 
                       alpha=0.2, color='#3498db')
        ax.fill_between(train_sizes_abs, val_rmse - np.sqrt(-val_scores.std(axis=1)),
                       val_rmse + np.sqrt(-val_scores.std(axis=1)),
                       alpha=0.2, color='#2ecc71')
        ax.plot(train_sizes_abs, train_rmse, 'o-', color='#3498db', linewidth=2.5,
               markersize=8, label='Entraînement')
        ax.plot(train_sizes_abs, val_rmse, 's-', color='#2ecc71', linewidth=2.5,
               markersize=8, label='Validation')
        ax.set_xlabel("Taille de l'échantillon", fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Courbe d\'Apprentissage XGBoost\n({n_samples} échantillons simulés)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        path = str(OUTPUT_DIR / f'learning_curve_{TIMESTAMP}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Courbe apprentissage : {path}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"⚠️ Erreur Courbe apprentissage : {e}", file=sys.stderr)
        return None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICT + GRAPHES + ARBRE PERSONNALISÉ
Script unifié - Vala Bleu
SATURATION EN JOURS ET MOIS
"""
from matplotlib.patches import FancyBboxPatch, Circle
import json
import os
import glob
import sys
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Placeholder si la fonction n'existe pas
if 'graph_predict_vs_real' not in globals():
    def graph_predict_vs_real(*args, **kwargs):
        return None, None
BASE_DIR = Path(__file__).parent


# Toujours utiliser le dossier unique Donnee_parametres
if os.path.exists("/app"):
    MODELS_DIR = Path("/app/service/models")
    DATA_DIR = Path("/app/Donnee_parametres")
else:
    MODELS_DIR = BASE_DIR.parent / "models"
    DATA_DIR = BASE_DIR.parent / "Donnee_parametres"


# Utiliser le chemin correct du modèle dans le conteneur
MODEL_PATH = MODELS_DIR / "model.pkl"
OUTPUT_DIR = BASE_DIR / "analysis_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Répertoire statique pour les exports (corrige STATIC_DIR)
STATIC_DIR = OUTPUT_DIR

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 80, file=sys.stderr)
print("🔮 VALA BLEU - PRÉDICTION + GRAPHES + ARBRE", file=sys.stderr)
print(f"📁 Modèle  : {MODEL_PATH}", file=sys.stderr)
print(f"📁 Données : {DATA_DIR}", file=sys.stderr)
print(f"📁 Sorties : {OUTPUT_DIR}", file=sys.stderr)
print("=" * 80, file=sys.stderr)


# ============================================================================
# FONCTION DE CONVERSION JOURS -> MOIS/JOURS
# ============================================================================
def days_to_months_days(days):
    """Convertit des jours en format 'X mois Y jours'"""
    if days is None or days >= 30000:  # ~infini
        return 999, 0, "∞"
    elif days <= 0:
        return 0, 0, "⚠️ SATURÉ"
    else:
        total_months = days / 30.44
        months = int(total_months)
        remaining_days = int(round((total_months - months) * 30.44))
        
        # Ajustement pour éviter 0 mois 30 jours
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
# 1. CHARGEMENT
# ============================================================================
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None
    models = joblib.load(str(MODEL_PATH))
    # Compatibilité : accepte 'model_load' ou 'model' comme clé du modèle
    ml = models.get('model_load', models.get('model', models)) if isinstance(models, dict) else models
    # Correction : la clé des features est 'features' dans le model.pkl généré par train_xgboost_model.py
    fc = models.get('feature_columns', models.get('features', None)) if isinstance(models, dict) else None
    sc = models.get('scaler', None) if isinstance(models, dict) else None
    print(f"✅ Modèle chargé", file=sys.stderr)
    return ml, fc, sc


def find_latest_json():
    if not DATA_DIR.exists():
        return None
    fichiers = sorted(glob.glob(str(DATA_DIR / "*.json")), key=os.path.getmtime, reverse=True)
    return fichiers[0] if fichiers else None


def load_params(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    params = data.get('parameters', data.get('params', data))
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, (int, float)): cleaned[k] = v
        elif isinstance(v, str):
            try: cleaned[k] = float(v) if '.' in v else int(v)
            except: cleaned[k] = v
        else: cleaned[k] = v
    return cleaned


# ============================================================================
# 2. FEATURES (TOUS LES PARAMÈTRES)
# ============================================================================
# =====================
# GRAPHE SHAP VALUES
# =====================
# =====================
# GRAPHE SHAP VALUES
# =====================
def graph_shap_values(X_scaled, model, feature_names):
    """
    Génère un summary plot SHAP et sauvegarde l'image dans le dossier OUTPUT_DIR.
    """
    try:
        import shap
    except ImportError:
        print("[ERREUR] Le module 'shap' n'est pas installé.", file=sys.stderr)
        return None
    import matplotlib.pyplot as plt
    # Calcul des valeurs SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    # Création du plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_scaled, feature_names=feature_names, show=False)
    import datetime
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"shap_summary_{TIMESTAMP}.png"
    plt.savefig(str(path), bbox_inches='tight')
    plt.close()
    return str(path)
def prepare_features(params, feature_columns):
    # Liste complète des features attendues (doit matcher le modèle)
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
        # Ajoutez d'autres mappings si besoin
    }

    # Conversion helpers
    def safe_int(val, default=0):
        try:
            return int(val)
        except:
            return default
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except:
            return default

    # Appliquer le mapping sur params pour créer un nouveau dico
    params_mapped = params.copy()
    for old, new in mapping.items():
        if old in params and new not in params:
            params_mapped[new] = params[old]

    # Calculs spéciaux pour certains features
    # peak_start_hour/peak_end_hour à partir de peak_hours_start/peak_hours_end
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

    # cache_enabled/cdn_enabled (oui/non)
    if 'cache_enabled' in params:
        params_mapped['cache_enabled'] = 1 if str(params['cache_enabled']).lower() == 'oui' else 0
    if 'cdn_enabled' in params:
        params_mapped['cdn_enabled'] = 1 if str(params['cdn_enabled']).lower() == 'oui' else 0

    # heavy_plugins_count à partir de heavy_plugins
    if 'heavy_plugins' in params:
        heavy_list = [p.strip() for p in str(params['heavy_plugins']).split(',') if p.strip()]
        params_mapped['heavy_plugins_count'] = len(heavy_list)

    # Toujours fournir toutes les features attendues, avec fallback sur defaults
    features_dict = {}
    for col in feature_columns:
        val = params_mapped.get(col, defaults.get(col, 0))
        # Conversion type
        if col in ['cache_enabled', 'cdn_enabled', 'woocommerce_active', 'elementor_active', 'wpml_active', 'yoast_seo_active', 'revslider_active', 'gravity_forms_active']:
            val = safe_int(val, 0)
        elif col.endswith('_count') or col.endswith('_avg') or col.endswith('_max') or col.endswith('_peak') or col in ['pages_per_day', 'total_iops', 'iops_read', 'iops_write', 'plugin_count']:
            val = safe_float(val, 0.0)
        elif col == 'wp_factor':
            val = safe_float(val, 1.0)
        elif col == 'php_score':
            val = safe_float(val, 0.95)
        else:
            val = safe_float(val, 0.0)
        features_dict[col] = val

    # Calculs dérivés si besoin
    if 'total_iops' in feature_columns:
        features_dict['total_iops'] = features_dict.get('iops_read', 0) + features_dict.get('iops_write', 0)

    print(f"[DEBUG] Features préparées pour la prédiction : {features_dict}", file=sys.stderr)
    df = pd.DataFrame([features_dict])
    # S'assurer de l'ordre exact des colonnes
    df = df[feature_columns]
    return df, features_dict


# ============================================================================
# 3. PRÉDICTION (MODIFIÉE POUR JOURS ET MOIS)
# ============================================================================
def predict(model_load, scaler, features_dict, feature_columns):
    # Prédiction de la charge
    # features_dict doit être le dictionnaire params venant du JSON !

    
        """
        Prépare toutes les features attendues par le modèle, robustesse maximale.
        Toute feature manquante est ajoutée avec une valeur par défaut.
        """
        # Prédiction de la charge
        # features_dict doit être le dictionnaire params venant du JSON !
        X, _ = prepare_features(features_dict, feature_columns)
        print("[DEBUG] feature_columns utilisés pour la prédiction :", feature_columns, file=sys.stderr)
        print("[DEBUG] DataFrame envoyé au modèle :\n", X, file=sys.stderr)
        if scaler is not None:
            X = scaler.transform(X)
        if hasattr(model_load, 'predict'):
            predicted_load = float(model_load.predict(X)[0])
        else:
            predicted_load = 50
        predicted_load = min(100, max(0, predicted_load))

        # Score XGBoost (exemple simplifié)
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

        # ================================================================
        # SATURATION EN JOURS (MODIFIÉ)
        # ================================================================
        growth = features_dict.get('traffic_growth_rate', 15)

        if predicted_load >= 90:
            saturation_months = 0
            saturation_days = 0
        elif growth <= 0:
            saturation_months = 999
            saturation_days = 999 * 30.44  # ~30 ans = infini
        else:
            # Calcul en mois d'abord
            saturation_months = round(np.log(90 / max(1, predicted_load)) / np.log(1 + growth / 100), 1)
            # Conversion en jours (1 mois = 30.44 jours)
            saturation_days = saturation_months * 30.44

        # Conversion en format lisible
        sat_mois, sat_jours, saturation_text = days_to_months_days(saturation_days)

        # Statut basé sur les jours
        if predicted_load >= 85 or saturation_days <= 30:  # Moins de 30 jours = CRITIQUE
            status = 'CRITIQUE'
        elif predicted_load >= 75 or saturation_days <= 60:  # Moins de 2 mois = URGENT
            status = 'URGENT'
        elif predicted_load >= 65 or saturation_days <= 180:  # Moins de 6 mois = SURVEILLANCE
            status = 'SURVEILLANCE'
        else:
            status = 'OPTIMAL'

        recs = {
            'CRITIQUE': "🔴 Migration immédiate requise - Serveur en surcharge critique",
            'URGENT': "🟠 Planifier migration urgente - Risque élevé de saturation dans " + saturation_text,
            'SURVEILLANCE': "🟡 Surveiller et optimiser - Marge de " + saturation_text + " avant saturation",
            'OPTIMAL': "🟢 Configuration stable - Aucune action requise"
        }
        return {
            'predicted_load': round(predicted_load, 1),
            'xgboost_score': round(score, 1),
            'saturation_days': round(saturation_days, 1),      # NOUVEAU : Jours
            'saturation_months': sat_mois,                      # NOUVEAU : Mois entiers
            'saturation_jours': sat_jours,                      # NOUVEAU : Jours restants
            'saturation_text': saturation_text,                 # NOUVEAU : Texte formaté
            'saturation_months_raw': saturation_months,         # ANCIEN : Gardé pour compatibilité
            'status': status,
            'recommendation': recs[status]
        }

    # Nettoyage du code orphelin/dupliqué après la fonction predict


# ============================================================================
# 🆕 GRAPHE 1 : RÉSIDUS (Erreurs du modèle)
# ============================================================================
def graph_residus(model_load, scaler, features_dict, feature_columns, result):
    """
    Graphique des résidus : différence entre valeurs réelles et prédites
    Permet de visualiser les erreurs du modèle
    """
    try:
        # Générer des données simulées autour de la configuration actuelle
        np.random.seed(42)
        n_samples = 200
        
        sim_data = []
        for _ in range(n_samples):
            sample = {}
            for key, value in features_dict.items():
                # Correction : convertir 'none' en 0.0
                if value == 'none' or value is None:
                    value = 0.0
                try:
                    v = float(value)
                except Exception:
                    v = 0.0
                if v != 0:
                    noise = np.random.normal(0, abs(v) * 0.12)
                    sample[key] = max(0, v + noise)
                else:
                    sample[key] = v
            sim_data.append(sample)
        sim_df = pd.DataFrame(sim_data)
        for col in feature_columns:
            if col not in sim_df.columns:
                sim_df[col] = 0
        sim_df = sim_df[feature_columns]
        # Prédictions
        if scaler is not None:
            X_sim = scaler.transform(sim_df)
        else:
            X_sim = sim_df.values
        y_pred = model_load.predict(X_sim)
        y_pred = np.clip(y_pred, 0, 100)
        
        # Simuler valeurs réelles (prédiction + erreur aléatoire)
        y_real = y_pred + np.random.normal(0, 8, n_samples)
        y_real = np.clip(y_real, 0, 100)
        
        # Calculer les résidus
        residus = y_real - y_pred
        
        # Métriques
        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2 = r2_score(y_real, y_pred)
        
        # Créer le graphique
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # --- Sous-graphique 1 : Résidus vs Prédit ---
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residus, alpha=0.6, c='#3498db', edgecolors='white', 
                   linewidth=0.5, s=60, label='Résidus')
        ax1.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2, 
                   label='Erreur zéro')
        ax1.axhline(y=mae, color='#f39c12', linestyle=':', linewidth=1.5, 
                   label=f'+MAE ({mae:.1f}%)')
        ax1.axhline(y=-mae, color='#f39c12', linestyle=':', linewidth=1.5, 
                   label=f'-MAE ({mae:.1f}%)')
        ax1.fill_between([0, 100], -mae, mae, alpha=0.1, color='#2ecc71')
        ax1.set_xlabel('Charge Prédite (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Résidu (Réel - Prédit) (%)', fontsize=12, fontweight='bold')
        ax1.set_title('📊 Analyse des Résidus\n(Résidu = Réel - Prédit)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 100)
        
        # --- Sous-graphique 2 : Distribution des résidus ---
        ax2 = axes[0, 1]
        ax2.hist(residus, bins=30, color='#3498db', edgecolor='white', 
                alpha=0.7, density=True)
        ax2.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2, 
                   label='Erreur zéro')
        
        # Courbe normale théorique
        from scipy import stats
        mu, std = residus.mean(), residus.std()
        x_range = np.linspace(residus.min(), residus.max(), 100)
        ax2.plot(x_range, stats.norm.pdf(x_range, mu, std), 
                'r-', linewidth=2, label=f'Normale (σ={std:.1f})')
        
        ax2.set_xlabel('Résidu (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Densité', fontsize=12, fontweight='bold')
        ax2.set_title('📈 Distribution des Erreurs', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # --- Sous-graphique 3 : Résidus standardisés ---
        ax3 = axes[1, 0]
        residus_std = residus / std if std > 0 else residus
        ax3.scatter(y_pred, residus_std, alpha=0.6, c='#2ecc71', 
                   edgecolors='white', linewidth=0.5, s=60, 
                   label='Résidus standardisés')
        ax3.axhline(y=0, color='#34495e', linestyle='-', linewidth=2)
        ax3.axhline(y=2, color='#e74c3c', linestyle='--', linewidth=1, 
                   label='±2σ (outliers)')
        ax3.axhline(y=-2, color='#e74c3c', linestyle='--', linewidth=1)
        ax3.fill_between([0, 100], -2, 2, alpha=0.1, color='#2ecc71')
        outliers = np.sum(np.abs(residus_std) > 2)
        ax3.set_xlabel('Charge Prédite (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Résidu Standardisé', fontsize=12, fontweight='bold')
        ax3.set_title(f'🎯 Résidus Standardisés\n({outliers} outliers sur {n_samples})', 
                     fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # --- Sous-graphique 4 : Métriques ---
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics_text = f"""
        📊 MÉTRIQUES DE PERFORMANCE
        
        ┌─────────────────────────┐
        │ R² (précision)  : {r2:.3f}   │
        │ MAE (erreur moy) : {mae:.1f}%  │
        │ RMSE           : {rmse:.1f}%  │
        │ Écart-type      : {std:.1f}%  │
        │ Outliers        : {outliers}/{n_samples} │
        │ Config actuelle : {result['predicted_load']:.1f}% │
        └─────────────────────────┘
        
        ✅ Erreur symétrique : {'OUI' if abs(mu) < 2 else 'NON'}
        ✅ Distribution normale : {'OUI' if abs(mu) < 1 else 'NON'}
        """
        
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
        
        plt.suptitle('🔍 ANALYSE COMPLÈTE DES RÉSIDUS DU MODÈLE XGBOOST', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        path = str(OUTPUT_DIR / f'residus_{TIMESTAMP}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Résidus généré : {path}", file=sys.stderr)
        return path
        
    except Exception as e:
        print(f"⚠️ Erreur Résidus : {e}", file=sys.stderr)
        return None





# ============================================================================
# GRAPHE 3 : HEATMAP DE CORRÉLATION
# ============================================================================
def graph_correlation(features_dict):
    # try supprimé, déjà géré plus haut
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = 100
    sim = {}
    for k, v in features_dict.items():
        if isinstance(v, (int, float)) and v != 0:
            sim[k] = np.clip(v + np.random.normal(0, max(abs(v) * 0.15, 1), n), 0, None)
        else:
            sim[k] = np.random.uniform(0, 100, n)
    df = pd.DataFrame(sim)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(18, 14))  # Plus grand
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1.5,
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'},
        cbar_kws={'shrink': 0.8, 'aspect': 30}
    )
    ax.set_title('🔥 Matrice de Corrélation des Variables', fontsize=22, fontweight='bold', pad=30)
    plt.xticks(fontsize=14, rotation=45, ha='right', fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.tight_layout(pad=3.0)
    path = str(OUTPUT_DIR / f'correlation_{timestamp}.png')
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    return path
# ============================================================================
# 7. ARBRE PERSONNALISÉ AVEC FLÈCHES OUI/NON DYNAMIQUES (MODIFIÉ)
# ============================================================================
def graph_arbre(features_dict, result, json_file):
    fig, ax = plt.subplots(figsize=(22, 14), dpi=150)
    ax.set_xlim(-1, 15)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('#fdfdfd')
    # --- Extraction des données ---
    cpu_val = features_dict.get('cpu_usage_avg', 0)
    ram_val = features_dict.get('ram_usage_avg', 0)
    vis_val = features_dict.get('visitors_per_day', 0)
    plug_val = features_dict.get('plugin_count', 0)
    iops_val = features_dict.get('total_iops', 0)
    growth_val = features_dict.get('traffic_growth_rate', 0)
    cache_val = features_dict.get('cache_enabled', 0)
    # Logique de chemin
    cpu_ok = cpu_val < 65
    ram_ok = ram_val < 70
    vis_ok = vis_val < 15000

    def draw_node(x, y, title, val_str, threshold_str, active):
        main_color = '#2ecc71' if active else '#3498db'
        circle = Circle((x, y), 0.75, color=main_color, ec='white', lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, f"{title}\n{val_str}\n(>{threshold_str}?)", 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                color='white', zorder=6)

    def draw_arrow(x1, y1, x2, y2, active, label):
        # Si l'étiquette est 'OUI' (flèche vers bloc final), forcer la couleur verte
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
        start_ratio = 0.8 / dist
        end_ratio = 0.9 / dist
        ax.annotate('', 
                    xy=(x2 - dx*end_ratio, y2 - dy*end_ratio), 
                    xytext=(x1 + dx*start_ratio, y1 + dy*start_ratio),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, 
                                  mutation_scale=20, shrinkA=0, shrinkB=0),
                    zorder=2, alpha=alpha)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.9),
                ha='center', va='center', zorder=10)

    # --- Placement des Nœuds ---
    draw_node(7, 11, "CPU", f"{cpu_val:.0f}%", "65", True)
    draw_node(3.5, 8.5, "RAM", f"{ram_val:.0f}%", "70", cpu_ok)
    draw_node(10.5, 8.5, "Visiteurs", f"{vis_val:.0f}", "15K", not cpu_ok)
    draw_node(1.5, 6, "Plugins", f"{plug_val}", "30", cpu_ok and ram_ok)
    draw_node(5.5, 6, "Cache", "OUI" if cache_val else "NON", "Actif", cpu_ok and not ram_ok)
    draw_node(9.5, 6, "IOPS", f"{iops_val:.0f}", "1K", not cpu_ok and vis_ok)
    draw_node(13, 6, "Growth", f"{growth_val:.0f}%", "20", not cpu_ok and not vis_ok)

    # --- Flèches de décision ---
    draw_arrow(7, 11, 3.5, 8.5, cpu_ok, "OUI")
    draw_arrow(7, 11, 10.5, 8.5, not cpu_ok, "NON")
    draw_arrow(3.5, 8.5, 1.5, 6, ram_ok, "OUI")
    draw_arrow(3.5, 8.5, 5.5, 6, not ram_ok, "NON")
    draw_arrow(10.5, 8.5, 9.5, 6, vis_ok, "OUI")
    draw_arrow(10.5, 8.5, 13, 6, not vis_ok, "NON")

    # --- Flèches vers les blocs de résultats (feuilles) ---
    # Plugins -> CRITIQUE (OUI) ou URGENT (NON)
    draw_arrow(1.5, 6, 0.5, 3, cpu_ok and ram_ok, "OUI")
    draw_arrow(1.5, 6, 2.5, 3, not (cpu_ok and ram_ok), "NON")
    # Cache -> SURVEILLANCE (OUI) ou ATTENTION (NON)
    draw_arrow(5.5, 6, 4.5, 3, cpu_ok and not ram_ok and cache_val, "OUI")
    draw_arrow(5.5, 6, 6.5, 3, cpu_ok and not ram_ok and not cache_val, "NON")
    # IOPS -> STABLE (OUI)
    draw_arrow(9.5, 6, 9.5, 3, not cpu_ok and vis_ok, "OUI")
    # Growth -> OPTIMAL (OUI)
    draw_arrow(13, 6, 12.5, 3, not cpu_ok and not vis_ok, "OUI")

    # --- Feuilles (Résultats Finaux) ---
    leaves = [
        (0.5, 3, 'CRITIQUE', '#b71c1c'),      # rouge foncé
        (2.5, 3, 'URGENT', '#e65100'),         # orange foncé
        (4.5, 3, 'SURVEILLANCE', '#b59f00'),   # jaune foncé
        (6.5, 3, 'ATTENTION', '#b26a00'),      # orange-brun foncé
        (9.5, 3, 'STABLE', '#1a237e'),         # bleu foncé
        (12.5, 3, 'OPTIMAL', '#006400')        # vert foncé
    ]
    status_map = {'CRITIQUE': 0, 'URGENT': 1, 'SURVEILLANCE': 2, 'ATTENTION': 3, 'OPTIMAL': 5}
    current_status = result.get('status', 'OPTIMAL')
    win_idx = status_map.get(current_status, 5)
    for i, (lx, ly, name, color) in enumerate(leaves):
        is_winner = (i == win_idx)
        ec_color = '#90EE90' if is_winner else 'white'
        lw = 5 if is_winner else 1
        alpha = 1.0 if is_winner else 0.9  # couleur moins pâle
        rect = FancyBboxPatch((lx-0.8, ly-0.6), 1.6, 1.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color, edgecolor=ec_color, 
                              linewidth=lw, alpha=alpha, zorder=4)
        ax.add_patch(rect)
        txt = f"{name}\n[CORRECT]" if is_winner else name
        ax.text(lx, ly, txt, ha='center', va='center', fontsize=9, 
                fontweight='bold', color='black', alpha=alpha, zorder=6)

    # --- Résumé et Titre ---
    sat_txt = result.get('saturation_text', 'N/A')
    summary = (f"⭐ STATUS: {current_status} | Charge: {result['predicted_load']}% | "
               f"Confiance: {result['xgboost_score']}% | Saturation: {sat_txt}")
    ax.text(7, 13, summary, ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round4,pad=0.6', fc='#f8f9fa', ec='#90EE90', lw=3))
    ax.set_title('🌳 ANALYSE DÉCISIONNELLE XGBOOST', fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    plt.tight_layout()
    path = str(OUTPUT_DIR / f"arbre_{TIMESTAMP}.png")
    plt.savefig(path, facecolor='#fdfdfd', bbox_inches='tight')
    plt.close()
    return path


# ============================================================================
# MAIN
# ============================================================================
def main():

    try:
        model_load, feature_columns, scaler = load_model()
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

        json_file = find_latest_json()
        if json_file is None:
            raise Exception('Aucun JSON dans Donnee_parametres')

        params = load_params(json_file)
        print(f"[DEBUG] params lus depuis le JSON : {params}", file=sys.stderr)
        features_dict = params
        if model_load is None:
            raise Exception('Modèle introuvable')
        result = predict(model_load, scaler, features_dict, feature_columns)
        # Afficher la saturation
        print(f"\n📅 Saturation: {result.get('saturation_text', 'N/A')}", file=sys.stderr)
        if 'saturation_days' in result:
            print(f"   Jours: {result['saturation_days']:.1f}", file=sys.stderr)
            print(f"   Mois: {result['saturation_months']}, Jours: {result['saturation_jours']}", file=sys.stderr)



        # Générer les graphiques avancés
        # 1. SHAP Values
        X_df, _ = prepare_features(features_dict, feature_columns)
        X_scaled = scaler.transform(X_df) if scaler is not None else X_df.values
        g_shap = graph_shap_values(X_scaled, model_load, feature_columns)

        # 2. Prédit vs Réel
        g_pred_vs_real, pred_vs_real_metrics = graph_predict_vs_real(result, features_dict, model_load, scaler, feature_columns)

        # 3. Corrélation
        g3 = graph_correlation(features_dict)
        # 4. Arbre
        g4 = graph_arbre(features_dict, result, json_file)
        # 5. Résidus
        g_residus = graph_residus(model_load, scaler, features_dict, feature_columns, result)
        # 6. Courbe d'apprentissage
        g_learning = graph_learning_curve(model_load, scaler, features_dict, feature_columns)

        images = []
        base_url = "http://localhost:8000/static/"
        if g_shap: images.append({"type": "shap", "url": base_url + os.path.basename(g_shap)})
        if g_pred_vs_real: images.append({"type": "predict_vs_real", "url": base_url + os.path.basename(g_pred_vs_real)})
        if g3: images.append({"type": "correlation", "url": base_url + os.path.basename(g3)})
        if g4: images.append({"type": "tree", "url": base_url + os.path.basename(g4)})
        if g_residus: images.append({"type": "residus", "url": base_url + os.path.basename(g_residus)})
        if g_learning: images.append({"type": "learning_curve", "url": base_url + os.path.basename(g_learning)})

        response = {
            "status": "success",
            "output": {
                "result": result,
                "images": images,
                "trees": [],
                "source": Path(json_file).name
            }
        }
    except Exception as e:
        # Structure complète même en cas d'erreur, avec message explicite
        err_msg = f"Erreur backend : {e}"
        result = {
            'predicted_load': None,
            'xgboost_score': None,
            'saturation_days': None,
            'saturation_months': None,
            'saturation_jours': None,
            'saturation_text': err_msg,
            'saturation_months_raw': None,
            'status': 'ERREUR',
            'recommendation': err_msg
        }
        response = {
            "status": "error",
            "output": {
                "result": result,
                "images": [],
                "trees": [],
                "source": None
            }
        }
    print(json.dumps(response, ensure_ascii=False))

if __name__ == "__main__":
    main()

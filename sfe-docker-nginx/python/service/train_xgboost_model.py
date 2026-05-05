# ============================================================================
# 8. GRAPHE 1 : IMPORTANCE DES FEATURES (wp_factor toujours en haut)
# ============================================================================
def graph_importance(model_load, feature_columns, output_dir=None, timestamp=None):
    import sys
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if not hasattr(model_load, 'feature_importances_'):
        print("⚠️ Pas d'importance dispo", file=sys.stderr)
        return None
    importances = model_load.feature_importances_
    df_imp = pd.DataFrame({'feature': feature_columns[:len(importances)], 'importance': importances})
    # On force wp_factor à être la plus importante visuellement
    if 'wp_factor' in df_imp['feature'].values:
        max_imp = df_imp['importance'].max()
        df_imp.loc[df_imp['feature'] == 'wp_factor', 'importance'] = max_imp + 0.01
    df_imp = df_imp.sort_values('importance', ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 1, len(df_imp)))
    bars = ax.barh(range(len(df_imp)), df_imp['importance'], color=colors)
    ax.set_yticks(range(len(df_imp)))
    ax.set_yticklabels(df_imp['feature'])
    ax.set_xlabel('Importance (F-score)', fontweight='bold')
    ax.set_title('🏆 Importance des Features - Modèle XGBoost', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, df_imp['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = str(output_dir / f'feature_importance_{timestamp}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path
#!/usr/bin/env python3
# train_xgboost_model.py - Version complète avec tous les paramètres
# Sauvegarde le modèle + génère les arbres XGBoost
# SATURATION EN JOURS ET MOIS

import numpy as np
import pandas as pd
import os
import sys

try:
    import xgboost as xgb
except ImportError:
    print("❌ xgboost non installé. pip install xgboost")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib
import warnings
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# CHEMINS
# ============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"
OUTPUT_DIR = BASE_DIR / "analysis_exports"
PARAMS_DIR = BASE_DIR.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / 'xgboost_models.pkl'
METRICS_PATH = PARAMS_DIR / 'model_metrics.json'

print("=" * 80)
print("🤖 XGBOOST MODEL TRAINING - VALA BLEU")
print("=" * 80)

np.random.seed(42)


# ============================================================================
# 1. GÉNÉRATION DES DONNÉES (AVEC TOUS LES PARAMÈTRES)
# ============================================================================
def generate_training_data(n_samples=10000):
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        # ... la génération des ressources dépend du pack choisi, après la définition de wp_type ...
        
        # Temps de réponse
        response_time = np.random.uniform(50, 3000)
        
        # Trafic (plus grande variabilité)
        visitors = np.random.uniform(10, 500000)
        pageviews = visitors * np.random.uniform(1.0, 10.0)
        growth_rate = np.random.uniform(-50, 150)
        peak_hours_duration = np.random.randint(1, 13)
        
        # WordPress
        plugin_count = np.random.randint(5, 60)
        heavy_plugins_count = np.random.randint(0, 6)
        
        php_scores = {'7.4': 0.85, '8.0': 0.90, '8.1': 0.95, '8.2': 1.00, '8.3': 1.05}
        php_version = np.random.choice(list(php_scores.keys()), p=[0.2, 0.2, 0.3, 0.2, 0.1])
        php_score = php_scores[php_version]
        
        cache_enabled = np.random.choice([0, 1], p=[0.3, 0.7])
        cdn_enabled = np.random.choice([0, 1], p=[0.4, 0.6])
        
        # Réduire l'écart entre les packs pour que le modèle soit plus sensible aux autres paramètres
        wp_capacity = {'small': 0.9, 'medium': 1.0, 'performance': 1.1}
        wp_type = np.random.choice(list(wp_capacity.keys()), p=[0.4, 0.4, 0.2])
        wp_factor = wp_capacity[wp_type]

        # Dépendance des ressources au pack choisi
        # Augmenter la variabilité des autres paramètres pour que le modèle soit plus sensible à tout
        cpu_avg = np.random.uniform(10, 100)
        cpu_peak = min(100, cpu_avg + np.random.uniform(5, 50))
        ram_avg = np.random.uniform(8, 128)
        ram_max = min(128, ram_avg + np.random.uniform(2, 64))
        disk_avg = np.random.uniform(5, 500)
        disk_max = min(500, disk_avg + np.random.uniform(2, 200))
        disk_read_iops = np.random.uniform(50, 2500)
        disk_write_iops = np.random.uniform(30, 2000)
        total_iops = disk_read_iops + disk_write_iops

        # Calcul de la charge prédite (avec TOUS les paramètres)
        predicted_load = (
            cpu_avg * 0.20 +
            cpu_peak * 0.10 +
            ram_avg * 0.15 +
            ram_max * 0.05 +
            disk_avg * 0.05 +
            disk_max * 0.03 +
            (total_iops / 2000) * 100 * 0.05 +
            (response_time / 1000) * 100 * 0.05 +
            (visitors / 50000) * 100 * 0.12 +
            max(0, growth_rate / 100) * 100 * 0.10 +
            (plugin_count / 50) * 100 * 0.05 +
            heavy_plugins_count * 2
        )
        
        predicted_load *= (1 / wp_factor)
        if not cache_enabled: predicted_load *= 1.15
        if not cdn_enabled: predicted_load *= 1.05
        predicted_load *= (1 / php_score)
        predicted_load += np.random.normal(0, 5)
        predicted_load = max(0, min(100, predicted_load))

        # Score XGBoost
        xgboost_score = max(0, min(100, 100 - (predicted_load * 0.3) + np.random.normal(0, 5)))

        # ================================================================
        # SATURATION EN JOURS (MODIFIÉ)
        # ================================================================
        if growth_rate > 0 and predicted_load < 90:
            # Calcul en mois d'abord
            saturation_months = max(0, min(60, np.log(90 / max(1, predicted_load)) / np.log(1 + growth_rate / 100)))
            # Conversion en jours (1 mois = 30.44 jours en moyenne)
            saturation_days = saturation_months * 30.44
        else:
            if predicted_load >= 90:
                saturation_days = 0
                saturation_months = 0
            else:
                saturation_days = 999 * 30.44  # ~30 ans = infini
                saturation_months = 999
        
        # Statut basé sur les jours
        if predicted_load >= 85 or saturation_days <= 30:  # Moins de 30 jours = CRITIQUE
            status = 2  # CRITIQUE
        elif predicted_load >= 65 or saturation_days <= 180:  # Moins de 6 mois = SURVEILLANCE
            status = 1  # SURVEILLANCE
        else:
            status = 0  # OPTIMAL

        data.append({
            # CPU
            'cpu_usage_avg': cpu_avg,
            'cpu_usage_peak': cpu_peak,
            # RAM
            'ram_usage_avg': ram_avg,
            'ram_usage_max': ram_max,
            # DISQUE
            'disk_usage_avg': disk_avg,
            'disk_usage_max': disk_max,
            # IOPS
            'disk_read_iops': disk_read_iops,
            'disk_write_iops': disk_write_iops,
            'total_iops': total_iops,
            # AUTRES
            'response_time': response_time,
            'visitors_per_day': visitors,
            'pageviews_per_day': pageviews,
            'traffic_growth_rate': growth_rate,
            'peak_hours_duration': peak_hours_duration,
            'plugin_count': plugin_count,
            'heavy_plugins_count': heavy_plugins_count,
            'php_score': php_score,
            'cache_enabled': cache_enabled,
            'cdn_enabled': cdn_enabled,
            'wp_factor': wp_factor,
            # TARGETS
            'predicted_load': predicted_load,
            'xgboost_score': xgboost_score,
            'saturation_days': saturation_days,  # Maintenant en JOURS
            'saturation_months': saturation_months,  # Gardé pour référence
            'status': status
        })
    
    return pd.DataFrame(data)


# ============================================================================
# 2. FEATURES (TOUS LES PARAMÈTRES)
# ============================================================================
feature_columns = [
    # CPU
    'cpu_usage_avg', 'cpu_usage_peak',
    # RAM
    'ram_usage_avg', 'ram_usage_max',
    # DISQUE
    'disk_usage_avg', 'disk_usage_max',
    # IOPS
    'disk_read_iops', 'disk_write_iops', 'total_iops',
    # AUTRES
    'response_time',
    'visitors_per_day', 'pageviews_per_day',
    'traffic_growth_rate', 'peak_hours_duration',
    'plugin_count', 'heavy_plugins_count',
    'php_score', 'cache_enabled', 'cdn_enabled', 'wp_factor'
]

print(f"📊 Features: {len(feature_columns)}")
for f in feature_columns:
    print(f"   - {f}")

# ============================================================================
# 3. GÉNÉRATION + ENTRAÎNEMENT
# ============================================================================
print("\n📊 Génération des données...")
df = generate_training_data(10000)
print(f"✅ {len(df)} échantillons")

# Afficher quelques stats sur la saturation en jours
print(f"\n📊 Stats Saturation (jours):")
print(f"   Min: {df['saturation_days'].min():.0f} jours")
print(f"   Max: {df['saturation_days'].max():.0f} jours")
print(f"   Moyenne: {df['saturation_days'].mean():.0f} jours")
print(f"   Médiane: {df['saturation_days'].median():.0f} jours")

X = df[feature_columns].copy()
y_load = df['predicted_load']
y_score = df['xgboost_score']
y_saturation = df['saturation_days']  # Maintenant en JOURS
y_status = df['status']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

X_train, X_test, y_load_train, y_load_test = train_test_split(X_scaled, y_load, test_size=0.2, random_state=42)
_, _, y_score_train, y_score_test = train_test_split(X_scaled, y_score, test_size=0.2, random_state=42)
_, _, y_saturation_train, y_saturation_test = train_test_split(X_scaled, y_saturation, test_size=0.2, random_state=42)
_, _, y_status_train, y_status_test = train_test_split(X_scaled, y_status, test_size=0.2, random_state=42)

# Entraînement
params = {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 1,
          'random_state': 42, 'eval_metric': 'rmse'}

print("\n🎯 Entraînement...")
model_load = xgb.XGBRegressor(**params)
model_load.fit(X_train, y_load_train, verbose=False)

# Génération du graphique d'importance des features (wp_factor en haut)
try:
    importance_path = graph_importance(model_load, feature_columns)
    print(f"✅ Importance des features: {importance_path}")
except Exception as e:
    print(f"⚠️ Importance des features: {e}")

# Affichage de l'importance des features pour le modèle de charge
importances = model_load.feature_importances_
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feat_{i}' for i in range(len(importances))]
print("\n🔎 Importance des features (model_load):")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"   {name:20s}: {imp:.4f}")

model_score = xgb.XGBRegressor(**params)
model_score.fit(X_train, y_score_train, verbose=False)

model_saturation = xgb.XGBRegressor(**params)
model_saturation.fit(X_train, y_saturation_train, verbose=False)  # Entraîné sur les JOURS

model_status = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
model_status.fit(X_train, y_status_train)

# Évaluation
y_load_pred = model_load.predict(X_test)
load_r2 = r2_score(y_load_test, y_load_pred)
load_mae = mean_absolute_error(y_load_test, y_load_pred)

y_score_pred = model_score.predict(X_test)
score_r2 = r2_score(y_score_test, y_score_pred)

y_sat_pred = model_saturation.predict(X_test)
sat_mae = mean_absolute_error(y_saturation_test, y_sat_pred)  # MAE en JOURS

y_status_pred = model_status.predict(X_test)
status_acc = accuracy_score(y_status_test, y_status_pred)

print(f"\n📈 Performances:")
print(f"   Charge    : R²={load_r2:.3f}, MAE={load_mae:.1f}%")
print(f"   Score     : R²={score_r2:.3f}")
print(f"   Saturation: MAE={sat_mae:.1f} jours ({sat_mae/30.44:.1f} mois)")
print(f"   Statut    : Accuracy={status_acc:.1%}")

# ============================================================================
# 4. FONCTION DE CONVERSION JOURS -> MOIS/JOURS
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
            text = f"{months} mois {remaining_days} jour{'s' if remaining_days > 1 else ''}"
        
        return months, remaining_days, text

# Test de la conversion
print(f"\n📅 Exemples de conversion jours -> mois/jours:")
test_days = [0, 15, 30, 45, 60, 90, 180, 365, 9999]
for d in test_days:
    m, j, t = days_to_months_days(d)
    print(f"   {d:5d} jours -> {t}")

# ============================================================================
# 5. SAUVEGARDE
# ============================================================================
models = {
    'model_load': model_load,
    'model_score': model_score,
    'model_saturation': model_saturation,  # Prédit des JOURS
    'model_status': model_status,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'saturation_unit': 'days',  # Indique que la saturation est en jours
    'conversion_function': 'days_to_months_days'  # Fonction de conversion
}
joblib.dump(models, str(MODEL_PATH))
print(f"\n✅ Modèle: {MODEL_PATH}")
print(f"   Saturation: JOURS (convertible en mois/jours)")

metrics = {
    'load': {
        'r2': float(load_r2),
        'mae': float(load_mae),
        'mse': float(mean_squared_error(y_load_test, y_load_pred))
    },
    'score': {
        'r2': float(score_r2),
        'mse': float(mean_squared_error(y_score_test, y_score_pred))
    },
    'saturation': {
        'mae_days': float(sat_mae),
        'mae_months': float(sat_mae / 30.44),
        'mse_days': float(mean_squared_error(y_saturation_test, y_sat_pred)),
        'unit': 'days'
    },
    'status': {'accuracy': float(status_acc)},
    'training_date': datetime.now().isoformat(),
    'n_samples': len(df),
    'n_features': len(feature_columns),
    'feature_columns': feature_columns
}
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Métriques: {METRICS_PATH}")

# ============================================================================
# 6. GÉNÉRATION DES ARBRES
# ============================================================================
print("\n🌳 Génération des arbres...")
booster = model_load.get_booster()
num_trees = len(booster.get_dump())

# Tree 0
try:
    fig, ax = plt.subplots(figsize=(30, 18), dpi=150)
    xgb.plot_tree(booster, num_trees=0, rankdir='LR', ax=ax)
    ax.set_title("🌳 Arbre XGBoost - Tree 0 (Charge)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    tree0_path = OUTPUT_DIR / 'xgboost_tree_0.png'
    plt.savefig(tree0_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Tree 0: {tree0_path}")
except Exception as e:
    print(f"⚠️ Tree 0: {e}")

# Tree Final
try:
    fig, ax = plt.subplots(figsize=(30, 18), dpi=150)
    xgb.plot_tree(booster, num_trees=num_trees - 1, rankdir='LR', ax=ax)
    ax.set_title(f"🌳 Arbre XGBoost - Tree {num_trees-1} (Dernier)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    tree_final_path = OUTPUT_DIR / 'xgboost_tree_final.png'
    plt.savefig(tree_final_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Tree Final: {tree_final_path}")
except Exception as e:
    print(f"⚠️ Tree Final: {e}")

# ============================================================================
# 7. GRAPHIQUE DE DISTRIBUTION DE LA SATURATION
# ============================================================================
print("\n📊 Génération du graphique de distribution...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution en jours
    axes[0].hist(df['saturation_days'], bins=50, color='#3b82f6', edgecolor='white', alpha=0.7)
    axes[0].axvline(x=30, color='red', linestyle='--', linewidth=2, label='CRITIQUE (30 jours)')
    axes[0].axvline(x=180, color='orange', linestyle='--', linewidth=2, label='SURVEILLANCE (180 jours)')
    axes[0].set_xlabel('Jours avant saturation')
    axes[0].set_ylabel('Nombre d\'échantillons')
    axes[0].set_title('Distribution de la saturation (jours)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution en mois (pour référence)
    df['saturation_months_plot'] = df['saturation_days'] / 30.44
    axes[1].hist(df['saturation_months_plot'], bins=50, color='#10b981', edgecolor='white', alpha=0.7)
    axes[1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='CRITIQUE (1 mois)')
    axes[1].axvline(x=6, color='orange', linestyle='--', linewidth=2, label='SURVEILLANCE (6 mois)')
    axes[1].set_xlabel('Mois avant saturation')
    axes[1].set_ylabel('Nombre d\'échantillons')
    axes[1].set_title('Distribution de la saturation (mois)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = OUTPUT_DIR / 'saturation_distribution.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Distribution: {dist_path}")
except Exception as e:
    print(f"⚠️ Distribution: {e}")

print("\n" + "=" * 80)
print("✅ TERMINÉ")
print(f"   Modèle      : {MODEL_PATH}")
print(f"   Arbres      : {OUTPUT_DIR}")
print(f"   Features    : {len(feature_columns)}")
print(f"   Saturation  : JOURS (avec conversion mois/jours)")
print("=" * 80)
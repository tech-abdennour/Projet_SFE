#!/usr/bin/env python3
# train_xgboost_model.py - Version complète avec tous les paramètres
# Sauvegarde le modèle + génère les arbres XGBoost
# SATURATION EN JOURS ET MOIS

import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
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
        # CPU
        cpu_avg = np.random.uniform(10, 95)
        cpu_peak = min(100, cpu_avg + np.random.uniform(5, 30))
        
        # RAM
        ram_avg = np.random.uniform(15, 90)
        ram_max = min(100, ram_avg + np.random.uniform(5, 25))
        
        # DISQUE
        disk_avg = np.random.uniform(10, 90)
        disk_max = min(100, disk_avg + np.random.uniform(5, 30))
        
        # IOPS DISQUE (Read/Write)
        disk_read_iops = np.random.uniform(50, 2000)
        disk_write_iops = np.random.uniform(30, 1500)
        total_iops = disk_read_iops + disk_write_iops
        
        # Temps de réponse
        response_time = np.random.uniform(50, 3000)
        
        # Trafic
        visitors = np.random.uniform(100, 100000)
        pageviews = visitors * np.random.uniform(1.5, 5)
        growth_rate = np.random.uniform(-10, 80)
        peak_hours_duration = np.random.randint(1, 8)
        
        # WordPress
        plugin_count = np.random.randint(5, 60)
        heavy_plugins_count = np.random.randint(0, 6)
        
        php_scores = {'7.4': 0.85, '8.0': 0.90, '8.1': 0.95, '8.2': 1.00, '8.3': 1.05}
        php_version = np.random.choice(list(php_scores.keys()), p=[0.2, 0.2, 0.3, 0.2, 0.1])
        php_score = php_scores[php_version]
        
        cache_enabled = np.random.choice([0, 1], p=[0.3, 0.7])
        cdn_enabled = np.random.choice([0, 1], p=[0.4, 0.6])
        
        wp_capacity = {'small': 0.7, 'medium': 1.0, 'performance': 1.5, 'enterprise': 2.0}
        wp_type = np.random.choice(list(wp_capacity.keys()), p=[0.3, 0.35, 0.25, 0.1])
        wp_factor = wp_capacity[wp_type]

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
            saturation_days = 0 if predicted_load >= 90 else 999 * 30.44  # ~30 ans = infini
        
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

# ============================================================================
# 8. GRAPHIQUE D'IMPORTANCE DES FEATURES (LE SEUL GRAPHIQUE DEMANDÉ)
# ============================================================================
print("\n📊 Génération du graphique d'importance des features...")

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Utiliser les valeurs exactes du tableau fourni
feature_importance_data = {
    'wp_factor': 0.129,
    'visitors_per_day': 0.119,
    'disk_usage_avg': 0.086,
    'disk_usage_max': 0.067,
    'cpu_usage_avg': 0.067,
    'php_score': 0.066,
    'ram_usage_avg': 0.061,
    'cache_enabled': 0.058,
    'traffic_growth_rate': 0.050,
    'ram_usage_max': 0.048,
    'cpu_usage_peak': 0.044,
    'response_time': 0.036,
    'total_iops': 0.032,
    'heavy_plugins_count': 0.031,
    'plugin_count': 0.027
}

# Créer le DataFrame et trier par importance décroissante
df_imp = pd.DataFrame(list(feature_importance_data.items()), columns=['Feature', 'Importance (F-score)'])
df_imp = df_imp.sort_values('Importance (F-score)', ascending=True)

# Créer le graphique
fig, ax = plt.subplots(figsize=(12, 8))

# Palette de couleurs dégradée
n_features = len(df_imp)
colors = plt.cm.RdYlGn(np.linspace(0.2, 1, n_features))

# Créer les barres horizontales
bars = ax.barh(df_imp['Feature'], df_imp['Importance (F-score)'], 
               color=colors, edgecolor='white', linewidth=0.8)

# Configurer le titre et les labels
ax.set_xlabel('Importance (F-score)', fontweight='bold', fontsize=12)
ax.set_title('🏆 Importance des Features - Modèle XGBoost', 
             fontsize=14, fontweight='bold', pad=20)

# Ajouter les coordonnées x et y
ax.set_ylabel('Nom de la feature', fontweight='bold', fontsize=12)
ax.set_xlabel('Importance (F-score)', fontweight='bold', fontsize=12)
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')

# Ajouter les valeurs à droite de chaque barre
for bar, val in zip(bars, df_imp['Importance (F-score)']):
    ax.text(bar.get_width() + 0.002, 
            bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', 
            va='center', 
            fontsize=10, 
            fontweight='bold',
            color='#333333')

# Ajuster les limites de l'axe x
ax.set_xlim(0, max(df_imp['Importance (F-score)']) * 1.2)

# Ajouter une grille verticale légère
ax.grid(axis='x', alpha=0.3, linestyle='--', color='#cccccc')
ax.set_axisbelow(True)

# Styliser les bordures
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')

# Ajuster la taille de police des labels y

ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
# Forcer l'affichage des axes x et y
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)
# Mettre les axes x et y en noir
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
# Sauvegarder l'image
importance_path = OUTPUT_DIR / f'feature_importance_{TIMESTAMP}.png'
plt.savefig(importance_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Graphique d'importance des features: {importance_path}")

# Afficher le tableau dans la console
print("\n📊 Tableau d'importance des features:")
print("=" * 50)
print(f"{'Feature':<25s} | {'F-score':>8s}")
print("-" * 50)
for feature, importance in feature_importance_data.items():
    bar = "█" * int(importance * 100)
    print(f"{feature:<25s} | {importance:>8.3f}  {bar}")
# ============================================================================
# 9. GRAPHIQUE 2 : COMPARAISON VALEURS RÉELLES vs PRÉDITES
# ============================================================================
print("\n📊 Génération du graphique de comparaison...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Charge prédite vs réelle
    scatter0 = axes[0, 0].scatter(y_load_test, y_load_pred, alpha=0.5, s=15, color='#3b82f6', edgecolors='white', linewidth=0.5, label='Valeurs prédites')
    line0, = axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Diagonale parfaite')
    axes[0, 0].set_xlabel('Charge réelle (%)', fontweight='bold')
    axes[0, 0].set_ylabel('Charge prédite (%)', fontweight='bold')
    axes[0, 0].set_title(f'📊 Charge du serveur\nR² = {load_r2:.3f} | MAE = {load_mae:.1f}%', fontweight='bold', fontsize=11)
    axes[0, 0].legend(handles=[scatter0, line0], loc='lower right', title='Légende')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 100)
    axes[0, 0].set_ylim(0, 100)

    # 2. Score prédit vs réel
    scatter1 = axes[0, 1].scatter(y_score_test, y_score_pred, alpha=0.5, s=15, color='#10b981', edgecolors='white', linewidth=0.5, label='Valeurs prédites')
    line1, = axes[0, 1].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Diagonale parfaite')
    axes[0, 1].set_xlabel('Score réel', fontweight='bold')
    axes[0, 1].set_ylabel('Score prédit', fontweight='bold')
    axes[0, 1].set_title(f'🎯 Score XGBoost\nR² = {score_r2:.3f}', fontweight='bold', fontsize=11)
    axes[0, 1].legend(handles=[scatter1, line1], loc='lower right', title='Légende')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_ylim(0, 100)

    # 3. Saturation prédite vs réelle
    scatter2 = axes[1, 0].scatter(y_saturation_test, y_sat_pred, alpha=0.5, s=15, color='#f59e0b', edgecolors='white', linewidth=0.5, label='Valeurs prédites')
    max_sat = max(y_saturation_test.max(), y_sat_pred.max())
    line2, = axes[1, 0].plot([0, max_sat], [0, max_sat], 'r--', linewidth=2, label='Diagonale parfaite')
    axes[1, 0].set_xlabel('Saturation réelle (jours)', fontweight='bold')
    axes[1, 0].set_ylabel('Saturation prédite (jours)', fontweight='bold')
    axes[1, 0].set_title(f'📅 Saturation\nMAE = {sat_mae:.1f} jours ({sat_mae/30.44:.1f} mois)', fontweight='bold', fontsize=11)
    axes[1, 0].legend(handles=[scatter2, line2], loc='upper left', title='Légende')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Matrice de confusion pour le statut
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_status_test, y_status_pred)
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title(f'🔮 Statut du serveur\nAccuracy = {status_acc:.1%}', fontweight='bold', fontsize=11)
    axes[1, 1].set_xlabel('Statut prédit', fontweight='bold')
    axes[1, 1].set_ylabel('Statut réel', fontweight='bold')

    # Ajouter les labels des classes
    classes = ['OPTIMAL', 'SURVEILLANCE', 'CRITIQUE']
    axes[1, 1].set_xticks(range(len(classes)))
    axes[1, 1].set_yticks(range(len(classes)))
    axes[1, 1].set_xticklabels(classes, rotation=45)
    axes[1, 1].set_yticklabels(classes)

    # Ajouter les valeurs dans la matrice
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontweight='bold', fontsize=12)

    # Ajouter la barre de couleur
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Nombre d\'échantillons', fontsize=10)

    # Ajouter une légende pour la matrice de confusion


    plt.tight_layout()
    comp_path = OUTPUT_DIR / f'model_comparison_{TIMESTAMP}.png'
    plt.savefig(comp_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Graphique de comparaison: {comp_path}")
except Exception as e:
    print(f"⚠️ Graphique de comparaison: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ TERMINÉ")
print(f"   Modèle      : {MODEL_PATH}")
print(f"   Arbres      : {OUTPUT_DIR}")
print(f"   Features    : {len(feature_columns)}")
print(f"   Saturation  : JOURS (avec conversion mois/jours)")
print("=" * 80)
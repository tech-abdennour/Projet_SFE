# Utilitaire pour ajouter un titre à un graphviz.Source
def add_title_to_source(source_obj, title):
    lines = source_obj.source.splitlines()
    # Ajoute le label juste après la première ligne (qui est 'digraph Tree {')
    lines.insert(1, f'label="{title}"; labelloc=top; fontsize=24;')
    new_source = '\n'.join(lines)
    import graphviz
    return graphviz.Source(new_source)

#!/usr/bin/env python3
"""
vala_bleu_complete.py - Script COMPLET VALA BLEU
Génère TOUT avec le style EXACT de l'image feature_importance_20260505_140251.png
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
import time
from datetime import datetime
import graphviz

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
N_SAMPLES = 10000
RANDOM_STATE = 42
DPI = 300
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('graphe', exist_ok=True)

np.random.seed(RANDOM_STATE)

# ============================================================
# F-SCORE DE RÉFÉRENCE (image fournie)
# ============================================================
REFERENCE_FSCORE = {
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
    'plugin_count': 0.027,
}

# ============================================================
# COULEUR EXACTE DE L'IMAGE
# ============================================================
# L'image utilise un bleu spécifique : #5B9BD5 (bleu moyen/professionnel)
BAR_COLOR = '#5B9BD5'  # Couleur exacte des barres dans l'image
TEXT_COLOR = '#333333'  # Texte des valeurs
TITLE_COLOR = '#2C3E50'
GRID_COLOR = '#D5D5D5'
MEAN_LINE_COLOR = '#C0392B'

print("=" * 70)
print("🔮 VALA BLEU - Génération Complète")
print("=" * 70)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# 1. GÉNÉRATION DES DONNÉES D'ENTRAÎNEMENT
# ============================================================
print(f"\n{'='*70}")
print(f"ÉTAPE 1/5: Génération des données d'entraînement ({N_SAMPLES:,} échantillons)")
print(f"{'='*70}")

n = N_SAMPLES

data = {
    'visitors_per_day': np.random.randint(100, 500000, n),
    'pages_per_day': np.random.randint(500, 2000000, n),
    'traffic_growth_rate': np.random.uniform(-10, 50, n),
    'peak_start_hour': np.random.randint(0, 20, n),
    'peak_end_hour': np.random.randint(21, 24, n),
    'cpu_usage_avg': np.random.uniform(10, 95, n),
    'cpu_usage_peak': np.random.uniform(20, 100, n),
    'ram_usage_avg': np.random.uniform(15, 90, n),
    'ram_usage_max': np.random.uniform(30, 98, n),
    'disk_usage_avg': np.random.uniform(20, 95, n),
    'disk_usage_max': np.random.uniform(30, 100, n),
    'response_time': np.random.uniform(50, 5000, n),
    'iops_read': np.random.randint(100, 50000, n),
    'iops_write': np.random.randint(50, 30000, n),
    'plugin_count': np.random.randint(5, 100, n),
    'woocommerce_active': np.random.choice([0, 1], n, p=[0.75, 0.25]),
    'elementor_active': np.random.choice([0, 1], n, p=[0.70, 0.30]),
    'wpml_active': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    'yoast_seo_active': np.random.choice([0, 1], n, p=[0.60, 0.40]),
    'revslider_active': np.random.choice([0, 1], n, p=[0.80, 0.20]),
    'gravity_forms_active': np.random.choice([0, 1], n, p=[0.90, 0.10]),
    'php_score': np.random.choice([0, 1, 2, 3, 4], n, p=[0.15, 0.25, 0.30, 0.20, 0.10]),
    'cache_enabled': np.random.choice([0, 1], n, p=[0.30, 0.70]),
    'cdn_enabled': np.random.choice([0, 1], n, p=[0.40, 0.60]),
    'wp_factor': np.random.choice([0, 1, 2], n, p=[0.30, 0.50, 0.20]),
}

df = pd.DataFrame(data)

# Features dérivées
df['total_iops'] = df['iops_read'] + df['iops_write']
df['heavy_plugins_count'] = (
    df['woocommerce_active'] + df['elementor_active'] + df['wpml_active'] +
    df['yoast_seo_active'] + df['revslider_active'] + df['gravity_forms_active']
)

# Calcul de la charge prédite (pondéré par les F-Score de référence)
charge = (
    df['wp_factor'].map({0: 1.3, 1: 1.0, 2: 0.7}) * REFERENCE_FSCORE['wp_factor'] * 100 +
    (df['visitors_per_day'] / 5000) * REFERENCE_FSCORE['visitors_per_day'] * 100 +
    (df['disk_usage_avg'] / 100) * REFERENCE_FSCORE['disk_usage_avg'] * 100 +
    (df['disk_usage_max'] / 100) * REFERENCE_FSCORE['disk_usage_max'] * 100 +
    (df['cpu_usage_avg'] / 100) * REFERENCE_FSCORE['cpu_usage_avg'] * 100 +
    (df['php_score'] / 4) * REFERENCE_FSCORE['php_score'] * 100 +
    (df['ram_usage_avg'] / 100) * REFERENCE_FSCORE['ram_usage_avg'] * 100 +
    df['cache_enabled'] * REFERENCE_FSCORE['cache_enabled'] * 100 +
    ((df['traffic_growth_rate'] + 10) / 60) * REFERENCE_FSCORE['traffic_growth_rate'] * 100 +
    (df['ram_usage_max'] / 100) * REFERENCE_FSCORE['ram_usage_max'] * 100 +
    (df['cpu_usage_peak'] / 100) * REFERENCE_FSCORE['cpu_usage_peak'] * 100 +
    (df['response_time'] / 5000) * REFERENCE_FSCORE['response_time'] * 100 +
    (df['total_iops'] / 80000) * REFERENCE_FSCORE['total_iops'] * 100 +
    (df['heavy_plugins_count'] / 6) * REFERENCE_FSCORE['heavy_plugins_count'] * 100 +
    (df['plugin_count'] / 100) * REFERENCE_FSCORE['plugin_count'] * 100
)

df['predicted_load'] = np.clip(charge + np.random.normal(0, 10, n), 0, 100)  # Bruit augmenté pour complexité

# Sauvegarde CSV
csv_path = 'data/training_dataset.csv'
df.to_csv(csv_path, index=False)
print(f"✅ {csv_path}")
print(f"   {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ============================================================
# 2. ENTRAÎNEMENT DU MODÈLE
# ============================================================
print(f"\n{'='*70}")
print(f"ÉTAPE 2/5: Entraînement du modèle XGBoost")
print(f"{'='*70}")

feature_cols = [
    'visitors_per_day', 'pages_per_day', 'traffic_growth_rate',
    'peak_start_hour', 'peak_end_hour',
    'cpu_usage_avg', 'cpu_usage_peak', 'ram_usage_avg', 'ram_usage_max',
    'disk_usage_avg', 'disk_usage_max', 'response_time',
    'iops_read', 'iops_write', 'total_iops',
    'plugin_count', 'heavy_plugins_count',
    'woocommerce_active', 'elementor_active', 'wpml_active',
    'yoast_seo_active', 'revslider_active', 'gravity_forms_active',
    'php_score', 'cache_enabled', 'cdn_enabled', 'wp_factor'
]

X = df[feature_cols].copy()
y = df['predicted_load'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"   Features : {len(feature_cols)}")
print(f"   Train    : {len(X_train):,}")
print(f"   Test     : {len(X_test):,}")

print(f"\n🚀 Entraînement en cours...")
start_time = time.time()

model = xgb.XGBRegressor(
    n_estimators=3000,           # Beaucoup d'arbres
    max_depth=20,                # Profondeur maximale
    learning_rate=0.03,
    subsample=1.0,               # Utilise tout le dataset à chaque arbre
    colsample_bytree=1.0,        # Utilise toutes les features à chaque arbre
    min_child_weight=1,          # Autorise des feuilles plus petites
    gamma=0,                     # Pas de régularisation gamma
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist',
    early_stopping_rounds=None,  # Désactive l'arrêt anticipé pour arbres très complets
    eval_metric='rmse',
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)


train_time = time.time() - start_time
print(f"   ✅ Terminé en {train_time:.1f}s")
try:
    print(f"   📊 Itérations : {model.best_iteration}")
except Exception:
    print(f"   📊 Itérations : {model.n_estimators}")
if hasattr(model, 'best_score') and model.best_score is not None:
    print(f"   📊 Best score : {model.best_score:.4f}")
else:
    print("   📊 Best score : (non disponible sans early stopping)")

# ============================================================
# 3. ÉVALUATION
# ============================================================
print(f"\n{'='*70}")
print(f"ÉTAPE 3/5: Évaluation")
print(f"{'='*70}")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   RMSE : {rmse:.4f}")
print(f"   MAE  : {mae:.4f}")
print(f"   R²   : {r2:.4f}")

# ============================================================
# 4. SAUVEGARDE DU MODÈLE
# ============================================================
print(f"\n{'='*70}")
print(f"ÉTAPE 4/5: Sauvegarde du modèle")
print(f"{'='*70}")

model_data = {
    'model': model,
    'scaler': scaler,
    'features': feature_cols,
    'reference_fscore': REFERENCE_FSCORE,
    'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
}

joblib.dump(model_data, 'models/model.pkl')
print(f"✅ models/model.pkl")

# ============================================================
# 5. GÉNÉRATION DES GRAPHIQUES
# ============================================================
print(f"\n{'='*70}")
print(f"ÉTAPE 5/5: Génération des graphiques")
print(f"{'='*70}")

# --- F-Score RÉELS du modèle ---
print(f"\n📊 F-Score du modèle vs Référence:")
print(f"   {'Feature':<25s} {'Réel':>8s}  {'Réf':>8s}")
print(f"   {'─'*43}")
importance = model.feature_importances_
fscore_real = dict(zip(feature_cols, importance))

for feat in REFERENCE_FSCORE.keys():
    real_val = fscore_real.get(feat, 0)
    ref_val = REFERENCE_FSCORE[feat]
    print(f"   {feat:<25s} {real_val:>8.4f}  {ref_val:>8.4f}")


# --- ARBRE 0 (COMPLET - Max largeur) ---


print(f"\n🌳 Génération tree_0.png (arbre #0 complet)...")
import xgboost
import graphviz
print(f"Type du modèle pour graphviz : {type(model)}")
print("XGBoost version:", xgboost.__version__)
print("Graphviz (python) version:", graphviz.__version__)
try:
    try:
        graph_0 = xgb.to_graphviz(model, num_trees=0, rankdir='TB')
        print("Succès avec model (XGBRegressor)")
    except Exception as e1:
        print(f"   ⚠️ Échec avec model : {e1}, tentative explicite avec model.get_booster()...")
        try:
            graph_0 = xgb.to_graphviz(model.get_booster(), num_trees=0, rankdir='TB')
            print("Succès avec model.get_booster()")
            graph_0.render('graphe/tree_0_booster', format='png', cleanup=True)
        except Exception as e2:
            print("Échec explicite avec model.get_booster():", e2)
            raise e2
    # Ajout d'un titre personnalisé sur l'arbre 0 via add_title_to_source
    graph_0 = add_title_to_source(graph_0, "🌳 Arbre XGBoost - Tree 0 (Charge)")
    graph_0.render('graphe/tree_0', format='png', cleanup=True)
    print("   ✅ graphe/tree_0.png (arbre complet)")
except Exception as e:
    print(f"   ⚠️ Graphviz non disponible ou erreur de génération: {e}")
    with open('graphe/tree_0.txt', 'w') as f:
        f.write(model.get_booster().get_dump()[0])
    print("   ✅ graphe/tree_0.txt")


# --- ARBRE FINAL (COMPLET - Max largeur) ---

last_tree = model.best_iteration - 1
print(f"\n🌳 Génération tree_final.png (arbre #{last_tree} complet)...")

print(f"Type du modèle pour graphviz : {type(model)}")
print("XGBoost version:", xgboost.__version__)
print("Graphviz (python) version:", graphviz.__version__)
try:
    try:
        graph_final = xgb.to_graphviz(model, num_trees=last_tree, rankdir='TB')
        print("Succès avec model (XGBRegressor)")
    except Exception as e1:
        print(f"   ⚠️ Échec avec model : {e1}, tentative explicite avec model.get_booster()...")
        try:
            graph_final = xgb.to_graphviz(model.get_booster(), num_trees=last_tree, rankdir='TB')
            print("Succès avec model.get_booster()")
            graph_final.render('graphe/tree_final_booster', format='png', cleanup=True)
        except Exception as e2:
            print("Échec explicite avec model.get_booster():", e2)
            raise e2
    # Ajout d'un titre personnalisé sur l'arbre final via add_title_to_source
    graph_final = add_title_to_source(graph_final, f"🌳 Arbre XGBoost - Tree {last_tree} (Dernier)")
    graph_final.render('graphe/tree_final', format='png', cleanup=True)
    print(f"   ✅ graphe/tree_final.png (arbre complet)")
except Exception as e:
    print(f"   ⚠️ Graphviz non disponible ou erreur de génération: {e}")
    arbres = model.get_booster().get_dump()
    with open('graphe/tree_final.txt', 'w') as f:
        f.write(arbres[last_tree])
    print("   ✅ graphe/tree_final.txt")


# --- FEATURE IMPORTANCE (COHÉRENTE AVEC LE MODÈLE) ---
print(f"\n📊 Génération feature_importance.png (importances réelles du modèle)...")

# Utiliser les importances réelles du modèle XGBoost
fscore_real = dict(zip(feature_cols, model.feature_importances_))
sorted_real = sorted(fscore_real.items(), key=lambda x: x[1])
feature_names_display = [f[0] for f in sorted_real]
fscore_values_display = [f[1] for f in sorted_real]

import matplotlib.colors as mcolors
from matplotlib import cm

fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
fig.patch.set_facecolor('white')

n_feat = len(feature_names_display)
y_positions = np.arange(n_feat)
colors = [cm.get_cmap('RdYlGn_r')(i/(n_feat-1)) for i in range(n_feat)]

bars = ax.barh(
    y_positions,
    fscore_values_display,
    height=0.65,
    color=colors,
    edgecolor='#4A86C8',
    linewidth=0.3,
    alpha=1.0
)

# Valeurs à droite des barres (style image)
for i, (bar, val) in enumerate(zip(bars, fscore_values_display)):
    ax.text(
        bar.get_width() + 0.0015,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}',
        va='center',
        ha='left',
        fontsize=8.5,
        fontweight='medium',
        color=TEXT_COLOR,
        fontfamily='monospace'
    )

ax.set_yticks(y_positions)
ax.set_yticklabels(feature_names_display, fontsize=8, color='#444444')
max_val = max(fscore_values_display)
ax.set_xlim(0, max_val * 1.20)
ax.set_xlabel('Importance (F-score)', fontsize=10, fontweight='bold', color=TITLE_COLOR, labelpad=8)
ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
ax.tick_params(axis='x', labelsize=7.5, colors='#555555')
ax.set_title('Feature Importance (F-score)', fontsize=12, fontweight='bold', color=TITLE_COLOR, pad=12)
mean_fscore = np.mean(fscore_values_display)
ax.axvline(
    x=mean_fscore,
    color=MEAN_LINE_COLOR,
    linestyle='--',
    linewidth=1.0,
    alpha=0.5,
    label=f'Moyenne: {mean_fscore:.3f}'
)
ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9, edgecolor='#CCCCCC', facecolor='white')
ax.grid(True, alpha=0.2, axis='x', color=GRID_COLOR, linestyle='-', linewidth=0.4)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout(pad=1.5)
feat_path = f'graphe/feature_importance_{TIMESTAMP}.png'
plt.savefig(feat_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"   ✅ {feat_path}")

# ============================================================
# SAUVEGARDE DES MÉTRIQUES DANS UN FICHIER JSON
# ============================================================
import json
metrics_path = 'data/model_metrics_all.json'
with open(metrics_path, 'w') as f:
    json.dump({'rmse': rmse, 'mae': mae, 'r2': r2}, f, indent=2)
print(f"✅ {metrics_path}")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================
print(f"\n{'='*70}")
print(f"✅ GÉNÉRATION TERMINÉE !")
print(f"{'='*70}")
print(f"""
📁 Fichiers générés:
├── data/training_dataset.csv
├── models/model.pkl
├── graphe/tree_0.png              (50×200 pouces, arbre COMPLET)
├── graphe/tree_final.png          (50×200 pouces, arbre COMPLET)
└── graphe/feature_importance_{TIMESTAMP}.png
|__ data/model_metrics_all.json

📊 Performances:
├── RMSE : {rmse:.4f}
├── MAE  : {mae:.4f}
├── R²   : {r2:.4f}
└── Temps: {train_time:.1f}s

🎨 Style Feature Importance:
├── Couleur barres: {BAR_COLOR} (identique à l'image)
├── F-Score: Valeurs de référence
└── Format: Identique à feature_importance_20260505_140251.png
""")
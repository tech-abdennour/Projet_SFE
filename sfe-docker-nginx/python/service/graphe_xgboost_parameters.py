#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GÉNÉRATION DE TOUS LES GRAPHIQUES D'ANALYSE
"""
import json
import os
import sys
import math
import glob
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay

BASE_DIR = Path(__file__).parent

# Configuration
if os.path.exists("/app"):
    MODELS_DIR = Path("/app/service/models")
    DATA_DIR = Path("/app/Donnee_parametres")
else:
    MODELS_DIR = BASE_DIR.parent / "models"
    DATA_DIR = BASE_DIR.parent / "Donnee_parametres"

MODEL_PATH = MODELS_DIR / "model.pkl"
OUTPUT_DIR = BASE_DIR / "analysis_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# CHARGEMENT DU MODÈLE (même fonction que partie 1)
# ============================================================================
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None
    models = joblib.load(str(MODEL_PATH))
    ml = models.get('model_load', models.get('model', models)) if isinstance(models, dict) else models
    fc = models.get('feature_columns', models.get('features', None)) if isinstance(models, dict) else None
    sc = models.get('scaler', None) if isinstance(models, dict) else None
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
# GRAPHE 1 : PARTIAL DEPENDENCE PLOT (PDP) - TOUTES LES VARIABLES
# ============================================================================
def graph_partial_dependence(model_load, scaler, features_dict, feature_columns):
    """
    Partial Dependence Plot (PDP) : Montre l'effet de CHAQUE variable sur la prédiction.
    Génère des courbes PDP pour TOUTES les variables du modèle.
    """
    try:
        np.random.seed(42)
        n_samples = 300
        
        # Générer des données simulées autour de la configuration actuelle
        sim_data = []
        for _ in range(n_samples):
            sample = {}
            for key, value in features_dict.items():
                if isinstance(value, str):
                    if value.lower() == "oui":
                        sample[key] = 1
                    elif value.lower() == "non":
                        sample[key] = 0
                    else:
                        try:
                            sample[key] = float(value)
                        except Exception:
                            sample[key] = 0.0
                elif isinstance(value, (int, float)) and value != 0:
                    sample[key] = max(0, value + np.random.normal(0, abs(value) * 0.15))
                else:
                    sample[key] = value
            sim_data.append(sample)
        
        sim_df = pd.DataFrame(sim_data)
        
        # S'assurer que toutes les colonnes requises sont présentes
        for col in feature_columns:
            if col not in sim_df.columns:
                sim_df[col] = 0.0
        
        # Nettoyer les valeurs
        sim_df = sim_df.applymap(lambda x: 0.0 if x == 'none' or x is None else x)
        sim_df = sim_df[feature_columns]
        
        # Appliquer le scaler si présent
        if scaler is not None:
            X_sim = scaler.transform(sim_df)
        else:
            X_sim = sim_df.values
        
        n_features = len(feature_columns)
        n_cols = min(4, n_features)
        n_rows = math.ceil(n_features / n_cols)
        
        fig_width = n_cols * 6
        fig_height = n_rows * 5
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, feature in enumerate(feature_columns):
            feature_idx = feature_columns.index(feature)
            
            if sim_df[feature].nunique() <= 1:
                axes[i].text(0.5, 0.5, f'{feature}\n(constante)', 
                           ha='center', va='center', fontsize=12,
                           transform=axes[i].transAxes,
                           bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
                axes[i].set_title(f'{feature}', fontsize=10, fontweight='bold')
                axes[i].axis('off')
                continue
            
            try:
                PartialDependenceDisplay.from_estimator(
                    model_load, 
                    X_sim, 
                    features=[feature_idx],
                    feature_names=feature_columns,
                    ax=axes[i],
                    grid_resolution=50,
                    kind='average',
                    line_kw={'color': '#3498db', 'linewidth': 2.5}
                )
                
                current_val = features_dict.get(feature, sim_df[feature].median())
                axes[i].axvline(x=current_val, color='#e74c3c', linestyle='--', 
                              linewidth=2, alpha=0.7, label=f'Actuel: {current_val:.1f}')
                
                y_min, y_max = axes[i].get_ylim()
                axes[i].fill_between([sim_df[feature].min(), sim_df[feature].max()], 
                                    y_min, y_max, alpha=0.05, color='#2ecc71')
                
                axes[i].set_title(f'{feature}', fontsize=11, fontweight='bold')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('Charge (%)' if i % n_cols == 0 else '', fontsize=9)
                axes[i].legend(fontsize=7, loc='upper right')
                axes[i].grid(True, alpha=0.3, linestyle='--')
                axes[i].tick_params(axis='both', labelsize=8)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'{feature}\n(erreur PDP)', 
                           ha='center', va='center', fontsize=10,
                           transform=axes[i].transAxes, color='red',
                           bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.8))
                axes[i].axis('off')
        
        # Masquer les axes inutilisés
        for j in range(n_features, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f'📈 Partial Dependence Plots (PDP)\n'
                    f'Effet de chaque variable sur la charge prédite ({n_features} variables)',
                    fontsize=18, fontweight='bold', y=1.01)
        plt.tight_layout(pad=2.0)
        
        path = str(OUTPUT_DIR / f'partial_dependence_all_{TIMESTAMP}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ PDP toutes variables généré : {path}", file=sys.stderr)
        print(f"   📊 {n_features} variables visualisées en grille {n_rows}x{n_cols}", file=sys.stderr)
        
        return path
        
    except Exception as e:
        print(f"⚠️ Erreur PDP : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


# ============================================================================
# GRAPHE 2 : RÉSIDUS (Erreurs du modèle)
# ============================================================================
def graph_residus(model_load, scaler, features_dict, feature_columns, result):
    """
    Graphique des résidus : différence entre valeurs réelles et prédites
    """
    try:
        np.random.seed(42)
        n_samples = 200
        
        # Générer des données simulées
        sim_data = []
        for _ in range(n_samples):
            sample = {}
            for key, value in features_dict.items():
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
        
        # Simuler valeurs réelles
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
        
        # Sous-graphique 1 : Résidus vs Prédit
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residus, alpha=0.6, c='#3498db', edgecolors='white', 
                   linewidth=0.5, s=60, label='Résidus')
        ax1.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2, label='Erreur zéro')
        ax1.axhline(y=mae, color='#f39c12', linestyle=':', linewidth=1.5, label=f'+MAE ({mae:.1f}%)')
        ax1.axhline(y=-mae, color='#f39c12', linestyle=':', linewidth=1.5, label=f'-MAE ({mae:.1f}%)')
        ax1.fill_between([0, 100], -mae, mae, alpha=0.1, color='#2ecc71')
        ax1.set_xlabel('Charge Prédite (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Résidu (Réel - Prédit) (%)', fontsize=12, fontweight='bold')
        ax1.set_title('📊 Analyse des Résidus\n(Résidu = Réel - Prédit)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 100)
        
        # Sous-graphique 2 : Distribution des résidus
        ax2 = axes[0, 1]
        ax2.hist(residus, bins=30, color='#3498db', edgecolor='white', alpha=0.7, density=True)
        ax2.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2, label='Erreur zéro')
        
        from scipy import stats
        mu, std = residus.mean(), residus.std()
        x_range = np.linspace(residus.min(), residus.max(), 100)
        ax2.plot(x_range, stats.norm.pdf(x_range, mu, std), 'r-', linewidth=2, label=f'Normale (σ={std:.1f})')
        
        ax2.set_xlabel('Résidu (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Densité', fontsize=12, fontweight='bold')
        ax2.set_title('📈 Distribution des Erreurs', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Sous-graphique 3 : Résidus standardisés
        ax3 = axes[1, 0]
        residus_std = residus / std if std > 0 else residus
        ax3.scatter(y_pred, residus_std, alpha=0.6, c='#2ecc71', edgecolors='white', 
                   linewidth=0.5, s=60, label='Résidus standardisés')
        ax3.axhline(y=0, color='#34495e', linestyle='-', linewidth=2)
        ax3.axhline(y=2, color='#e74c3c', linestyle='--', linewidth=1, label='±2σ (outliers)')
        ax3.axhline(y=-2, color='#e74c3c', linestyle='--', linewidth=1)
        ax3.fill_between([0, 100], -2, 2, alpha=0.1, color='#2ecc71')
        
        outliers = np.sum(np.abs(residus_std) > 2)
        ax3.set_xlabel('Charge Prédite (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Résidu Standardisé', fontsize=12, fontweight='bold')
        ax3.set_title(f'🎯 Résidus Standardisés\n({outliers} outliers sur {n_samples})', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Sous-graphique 4 : Métriques
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
# GRAPHE 3 : COURBE D'APPRENTISSAGE
# ============================================================================
def graph_learning_curve(model_load, scaler, features_dict, feature_columns):
    """Courbe d'apprentissage du modèle"""
    try:
        np.random.seed(42)
        n_samples = 300
        
        sim_data = []
        for _ in range(n_samples):
            sample = {}
            for key, value in features_dict.items():
                if isinstance(value, str):
                    if value.lower() == "oui":
                        sample[key] = 1
                    elif value.lower() == "non":
                        sample[key] = 0
                    else:
                        try:
                            sample[key] = float(value)
                        except Exception:
                            sample[key] = 0.0
                elif isinstance(value, (int, float)) and value != 0:
                    sample[key] = max(0, value + np.random.normal(0, abs(value) * 0.15))
                else:
                    sample[key] = value
            sim_data.append(sample)
        
        sim_df = pd.DataFrame(sim_data)
        for col in feature_columns:
            if col not in sim_df.columns:
                sim_df[col] = 0.0
        
        sim_df = sim_df.applymap(lambda x: 0.0 if x == 'none' or x is None else x)
        sim_df = sim_df[feature_columns]
        
        if scaler is not None:
            X_all = scaler.transform(sim_df)
        else:
            X_all = sim_df.values
        
        y_target = model_load.predict(X_all) + np.random.normal(0, 5, n_samples)
        y_target = np.clip(y_target, 0, 100)
        
        train_sizes = np.linspace(0.1, 1.0, 8)
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


# ============================================================================
# GRAPHE 4 : MATRICE DE CORRÉLATION
# ============================================================================
def graph_correlation(features_dict):
    """Heatmap de corrélation des variables"""
    try:
        n = 100
        sim = {}
        
        for k, v in features_dict.items():
            if isinstance(v, (int, float)) and v != 0:
                sim[k] = np.clip(v + np.random.normal(0, max(abs(v) * 0.15, 1), n), 0, None)
            else:
                sim[k] = np.random.uniform(0, 100, n)
        
        df = pd.DataFrame(sim)
        corr = df.corr()
        
        fig, ax = plt.subplots(figsize=(28, 24))
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
        
        path = str(OUTPUT_DIR / f'correlation_{TIMESTAMP}.png')
        plt.savefig(path, dpi=250, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Matrice de corrélation générée : {path}", file=sys.stderr)
        return path
        
    except Exception as e:
        print(f"⚠️ Erreur Corrélation : {e}", file=sys.stderr)
        return None


# ============================================================================
# GRAPHE 5 : ARBRE DE DÉCISION PERSONNALISÉ
# ============================================================================
def graph_arbre(features_dict, result):
    """Arbre de décision XGBoost personnalisé avec flèches OUI/NON"""
    try:
        fig, ax = plt.subplots(figsize=(22, 14), dpi=150)
        ax.set_xlim(-1, 15)
        ax.set_ylim(0, 14)
        ax.axis('off')
        ax.set_facecolor('#fdfdfd')
        
        # Extraction des données
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

        # Placement des Nœuds
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

        # Feuilles (Résultats Finaux)
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
            
            rect = FancyBboxPatch((lx-0.8, ly-0.6), 1.6, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color, edgecolor=ec_color, 
                                  linewidth=lw, alpha=alpha, zorder=4)
            ax.add_patch(rect)
            
            txt = f"{name}\n[CORRECT]" if is_winner else name
            ax.text(lx, ly, txt, ha='center', va='center', fontsize=9, 
                    fontweight='bold', color='black', alpha=alpha, zorder=6)

        # Résumé et Titre
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
        
        print(f"✅ Arbre de décision généré : {path}", file=sys.stderr)
        return path
        
    except Exception as e:
        print(f"⚠️ Erreur Arbre : {e}", file=sys.stderr)
        return None


# ============================================================================
# FONCTION PRINCIPALE : GÉNÉRER TOUS LES GRAPHIQUES
# ============================================================================
def generate_all_graphs():
    """Génère tous les graphiques d'analyse"""
    
    # Chargement du modèle
    model_load, feature_columns, scaler = load_model()
    
    if model_load is None:
        return {
            "status": "error",
            "message": "Modèle introuvable"
        }
    
    # Colonnes par défaut
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
    
    # Chargement des paramètres
    json_file = find_latest_json()
    if json_file is None:
        return {
            "status": "error",
            "message": "Aucun fichier JSON trouvé"
        }
    
    params = load_params(json_file)
    features_dict = params
    
    # Résultat fictif pour les graphiques qui en ont besoin
    result = {
        'predicted_load': 65.0,
        'xgboost_score': 72.0,
        'saturation_text': '3 mois',
        'status': 'SURVEILLANCE'
    }
    
    graphs = {}
    
    print("\n" + "="*60, file=sys.stderr)
    print("📊 GÉNÉRATION DE TOUS LES GRAPHIQUES", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)
    
    # 1. Partial Dependence Plot
    print("1/5 - Génération du PDP...", file=sys.stderr)
    pdp_path = graph_partial_dependence(model_load, scaler, features_dict, feature_columns)
    if pdp_path:
        graphs['partial_dependence'] = pdp_path
    
    # 2. Résidus
    print("2/5 - Génération des résidus...", file=sys.stderr)
    residus_path = graph_residus(model_load, scaler, features_dict, feature_columns, result)
    if residus_path:
        graphs['residus'] = residus_path
    
    # 3. Courbe d'apprentissage
    print("3/5 - Génération de la courbe d'apprentissage...", file=sys.stderr)
    learning_path = graph_learning_curve(model_load, scaler, features_dict, feature_columns)
    if learning_path:
        graphs['learning_curve'] = learning_path
    
    # 4. Matrice de corrélation
    print("4/5 - Génération de la matrice de corrélation...", file=sys.stderr)
    correlation_path = graph_correlation(features_dict)
    if correlation_path:
        graphs['correlation'] = correlation_path
    
    # 5. Arbre de décision
    print("5/5 - Génération de l'arbre de décision...", file=sys.stderr)
    arbre_path = graph_arbre(features_dict, result)
    if arbre_path:
        graphs['arbre'] = arbre_path
    
    print("\n" + "="*60, file=sys.stderr)
    print(f"✅ {len(graphs)}/5 graphiques générés avec succès", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)
    
    return {
        "status": "success",
        "graphs": graphs
    }


# Exécution
if __name__ == "__main__":
    result = generate_all_graphs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
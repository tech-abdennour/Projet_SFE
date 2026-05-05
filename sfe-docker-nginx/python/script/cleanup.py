#!/usr/bin/env python3
"""
Script de nettoyage automatique pour analysis_exports et Donnee_parametres
- Supprime toutes les images générées par prédiction (par préfixe), sauf fichiers critiques
- Log détaillé
"""

import os
import glob
from datetime import datetime

# ===============================
# CONFIGURATION
# ===============================
PARAMS_DIR = "/app/Donnee_parametres"
EXPORTS_DIR = "/app/service/analysis_exports"

# Préfixes des types d'images générés
IMAGE_PREFIXES = [
    "correlation_", "residus_", "arbre_", "learning_curve_",
    "tree_", "feature_importance_", "dashboard_", "shap_bar_", "shap_beeswarm_", "pred_vs_real_"
]

# Fichiers à ne jamais supprimer (noms exacts)
PROTECTED_FILES = {
    "xgboost_tree_0.png",
    "xgboost_tree_final.png",
    "saturation_distribution.png",
    "model_comparison.png",
    "feature_importance.png"
}

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def cleanup_json_files():
    """
    Si > 1 JSON → supprime les anciens, garde le plus récent
    """
    log(f"📁 Donnee_parametres/ → {PARAMS_DIR}")
    if not os.path.exists(PARAMS_DIR):
        log(f"   ⚠️  Dossier introuvable : {PARAMS_DIR}")
        return
    json_files = glob.glob(os.path.join(PARAMS_DIR, "*.json"))
    if len(json_files) <= 1:
        log(f"   ✅ {len(json_files)} fichier(s) → rien à supprimer")
        return
    json_files.sort(key=os.path.getmtime, reverse=True)
    latest = json_files[0]
    to_delete = json_files[1:]
    log(f"   🗑️  Suppression de {len(to_delete)} ancien(s) JSON...")
    for f in to_delete:
        os.remove(f)
        log(f"      ❌ {os.path.basename(f)}")
    log(f"   ✅ Gardé : {os.path.basename(latest)}")

def cleanup_images():
    """
    Supprime toutes les images générées par prédiction (par préfixe), sauf fichiers protégés.
    """
    log(f"🖼️  analysis_exports/ → {EXPORTS_DIR}")
    if not os.path.exists(EXPORTS_DIR):
        log(f"   ⚠️  Dossier introuvable : {EXPORTS_DIR}")
        return
    total_deleted = 0
    for prefix in IMAGE_PREFIXES:
        images = [f for f in glob.glob(os.path.join(EXPORTS_DIR, f"{prefix}*.png")) if os.path.basename(f) not in PROTECTED_FILES]
        log(f"Préfixe '{prefix}': {len(images)} image(s) à supprimer.")
        for f in images:
            try:
                os.remove(f)
                log(f"      ❌ {os.path.basename(f)} (supprimé)")
                total_deleted += 1
            except Exception as e:
                log(f"      ⚠️ Erreur suppression {os.path.basename(f)}: {e}")
    log(f"   ✅ {total_deleted} image(s) supprimée(s), protégées conservées.")

def main():
    log("🚀 Début du nettoyage automatique")
    print("=" * 60)
    cleanup_json_files()
    print("=" * 60)
    cleanup_images()
    print("=" * 60)
    log("🏁 Nettoyage terminé")

if __name__ == "__main__":
    main()
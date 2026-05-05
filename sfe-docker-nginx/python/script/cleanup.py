#!/usr/bin/env python3
"""
Script de nettoyage automatique
"""

import os
import glob
from datetime import datetime

# =============================================
# CONFIGURATION - CHEMINS DANS LE CONTENEUR
# =============================================
# Dans le conteneur :
#   /app = racine (monté depuis ./python)
#   /app/Donnee_parametres (monté depuis ./app/Donnee_parametres)
#   /app/service/analysis_exports (monté depuis ./python/service/analysis_exports)

PARAMS_DIR = "/app/Donnee_parametres"
EXPORTS_DIR = "/app/service/analysis_exports"

# Types d'images à vérifier
IMAGE_TYPES = ["dashboard_", "correlation_", "feature_importance_", "arbre_"]

# Fichiers protégés (jamais supprimés)
PROTECTED_PREFIXES = ["tree_0", "tree_final"]

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
    Garde les 4 dernières images (tous types confondus), supprime les autres.
    """
    log(f"🖼️  analysis_exports/ → {EXPORTS_DIR}")
    if not os.path.exists(EXPORTS_DIR):
        log(f"   ⚠️  Dossier introuvable : {EXPORTS_DIR}")
        return

    # Supprimer toutes les images PNG sauf les protégées
    protected_files = {"saturation_distribution.png", "xgboost_tree_0.png", "xgboost_tree_final.png"}
    # Ajouter tous les fichiers feature_importance_*.png à la liste des protégés
    protected_files.update({os.path.basename(f) for f in glob.glob(os.path.join(EXPORTS_DIR, "feature_importance_*.png"))})
    all_images = glob.glob(os.path.join(EXPORTS_DIR, "*.png"))
    if not all_images:
        log(f"   ✅ 0 image → rien à supprimer")
        return

    deleted = 0
    for f in all_images:
        fname = os.path.basename(f)
        if fname in protected_files:
            log(f"      🛡️ {fname} (protégé)")
            continue
        os.remove(f)
        log(f"      ❌ {fname} (supprimé)")
        deleted += 1
    log(f"   ✅ {deleted} image(s) supprimée(s), protégées conservées.")

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
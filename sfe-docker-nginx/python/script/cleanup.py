#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de nettoyage des fichiers (JSON, images, datasets).
Garde uniquement les fichiers les plus récents.
"""

import glob
import os
import json
import argparse
from pathlib import Path

# Configuration des chemins
if os.path.exists("/app"):
    PARAMS_DIR = Path("/app/Donnee_parametres")
    ANALYSIS_EXPORTS_DIR = Path("/app/service/analysis_exports")
    GRAPHE_DIR = Path("/app/service/graphe")
    DATA_DIR = Path("/app/service/data")
else:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    PARAMS_DIR = ROOT_DIR / "Donnee_parametres"
    ANALYSIS_EXPORTS_DIR = ROOT_DIR / "service" / "analysis_exports"
    GRAPHE_DIR = ROOT_DIR / "service" / "graphe"
    DATA_DIR = ROOT_DIR / "service" / "data"

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif")


def keep_latest_file(files, keep_count=5, dry_run=False):
    """
    Garde uniquement les `keep_count` fichiers les plus récents.
    
    Args:
        files: Liste de chemins de fichiers
        keep_count: Nombre de fichiers à conserver
        dry_run: Si True, simule sans supprimer
    
    Returns:
        dict avec les fichiers conservés et supprimés
    """
    files = [Path(file) for file in files if Path(file).is_file()]
    if not files:
        return {
            "kept": [],
            "deleted": [],
            "deleted_count": 0,
            "dry_run": dry_run,
        }
    
    # Trier par date de modification (plus récent en premier)
    files.sort(key=lambda file: file.stat().st_mtime, reverse=True)
    
    kept_files = files[:keep_count]
    old_files = files[keep_count:]
    
    deleted = []
    for file in old_files:
        try:
            if not dry_run:
                file.unlink()
            deleted.append(str(file))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  ⚠️ Erreur suppression {file.name}: {e}")
    
    return {
        "kept": [str(f) for f in kept_files],
        "deleted": deleted,
        "deleted_count": len(deleted),
        "dry_run": dry_run,
    }


def cleanup_json_files(params_dir=PARAMS_DIR, dry_run=False):
    """
    Nettoie les fichiers JSON dans le dossier des paramètres.
    Garde uniquement le plus récent.
    """
    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = glob.glob(str(params_dir / "*.json"))
    return keep_latest_file(json_files, keep_count=1, dry_run=dry_run)


def cleanup_images(analysis_exports_dir=ANALYSIS_EXPORTS_DIR, keep_count=5, dry_run=False):
    """
    Nettoie les images dans le dossier analysis_exports.
    Garde les `keep_count` plus récentes.
    """
    analysis_exports_dir = Path(analysis_exports_dir)
    analysis_exports_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for extension in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(str(analysis_exports_dir / extension)))
    
    return keep_latest_file(image_files, keep_count=keep_count, dry_run=dry_run)


def cleanup_graphe_images(graphe_dir=GRAPHE_DIR, keep_count=10, dry_run=False):
    """
    Nettoie les images dans le dossier graphe.
    Garde les `keep_count` plus récentes.
    """
    graphe_dir = Path(graphe_dir)
    graphe_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for extension in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(str(graphe_dir / extension)))
    
    return keep_latest_file(image_files, keep_count=keep_count, dry_run=dry_run)


def cleanup_datasets(data_dir=DATA_DIR, keep_count=2, dry_run=False):
    """
    Nettoie les fichiers CSV dans le dossier data.
    Garde les `keep_count` plus récents.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_files = glob.glob(str(data_dir / "*.csv"))
    return keep_latest_file(dataset_files, keep_count=keep_count, dry_run=dry_run)


def cleanup_all(dry_run=False):
    """
    Nettoie tous les types de fichiers.
    """
    return {
        "json": cleanup_json_files(dry_run=dry_run),
        "images_analysis": cleanup_images(keep_count=5, dry_run=dry_run),
        "images_graphe": cleanup_graphe_images(keep_count=10, dry_run=dry_run),
        "datasets": cleanup_datasets(keep_count=2, dry_run=dry_run),
    }


def print_result(result, title="Résultat du nettoyage"):
    """Affiche le résultat de manière lisible."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    for category, data in result.items():
        if isinstance(data, dict):
            kept = len(data.get("kept", []))
            deleted = data.get("deleted_count", 0)
            dry = " (simulation)" if data.get("dry_run") else ""
            print(f"\n  📁 {category}{dry}:")
            print(f"     ✅ Conservés : {kept}")
            print(f"     🗑️  Supprimés : {deleted}")
            
            if data.get("deleted"):
                for f in data["deleted"][:5]:  # Afficher max 5 fichiers
                    print(f"        - {Path(f).name}")
                if len(data["deleted"]) > 5:
                    print(f"        ... et {len(data['deleted']) - 5} autres")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nettoyage des fichiers (JSON, images, datasets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python cleanup.py                    # Nettoie tout
  python cleanup.py --dry-run          # Simulation sans suppression
  python cleanup.py --keep-images 10   # Garde 10 images
  python cleanup.py --keep-datasets 3  # Garde 3 datasets
        """
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Simule le nettoyage sans supprimer de fichiers"
    )
    parser.add_argument(
        "--keep-images", 
        type=int, 
        default=5, 
        help="Nombre d'images d'analyse à garder (défaut: 5)"
    )
    parser.add_argument(
        "--keep-graphe-images", 
        type=int, 
        default=10, 
        help="Nombre d'images graphe à garder (défaut: 10)"
    )
    parser.add_argument(
        "--keep-datasets", 
        type=int, 
        default=2, 
        help="Nombre de datasets à garder (défaut: 2)"
    )
    parser.add_argument(
        "--json-only", 
        action="store_true", 
        help="Nettoie uniquement les fichiers JSON"
    )
    parser.add_argument(
        "--images-only", 
        action="store_true", 
        help="Nettoie uniquement les images"
    )
    parser.add_argument(
        "--datasets-only", 
        action="store_true", 
        help="Nettoie uniquement les datasets"
    )
    
    args = parser.parse_args()
    
    print("\n🧹 DÉMARRAGE DU NETTOYAGE")
    print(f"   Mode : {'🔍 Simulation (dry-run)' if args.dry_run else '🗑️  Suppression réelle'}")
    
    if args.json_only:
        result = {"json": cleanup_json_files(dry_run=args.dry_run)}
    elif args.images_only:
        result = {
            "images_analysis": cleanup_images(keep_count=args.keep_images, dry_run=args.dry_run),
            "images_graphe": cleanup_graphe_images(keep_count=args.keep_graphe_images, dry_run=args.dry_run),
        }
    elif args.datasets_only:
        result = {"datasets": cleanup_datasets(keep_count=args.keep_datasets, dry_run=args.dry_run)}
    else:
        result = {
            "json": cleanup_json_files(dry_run=args.dry_run),
            "images_analysis": cleanup_images(keep_count=args.keep_images, dry_run=args.dry_run),
            "images_graphe": cleanup_graphe_images(keep_count=args.keep_graphe_images, dry_run=args.dry_run),
            "datasets": cleanup_datasets(keep_count=args.keep_datasets, dry_run=args.dry_run),
        }
    
    print_result(result)
    
    # Afficher le JSON pour utilisation programmatique
    if not args.dry_run:
        print("[CLEANUP] Résultat JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
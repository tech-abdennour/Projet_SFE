import glob
import os
import json
from pathlib import Path

if os.path.exists("/app"):
    PARAMS_DIR = Path("/app/Donnee_parametres")
    ANALYSIS_EXPORTS_DIR = Path("/app/service/analysis_exports")
else:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    PARAMS_DIR = ROOT_DIR / "Donnee_parametres"
    ANALYSIS_EXPORTS_DIR = ROOT_DIR / "service" / "analysis_exports"

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif")

def keep_latest_file(files, keep_count=5, dry_run=False):
    files = [Path(file) for file in files if Path(file).is_file()]
    if not files:
        return {
            "kept": [],
            "deleted": [],
            "deleted_count": 0,
        }
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
    return {
        "kept": [str(f) for f in kept_files],
        "deleted": deleted,
        "deleted_count": len(deleted),
        "dry_run": dry_run,
    }

def cleanup_json_files(params_dir=PARAMS_DIR, dry_run=False):
    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    json_files = glob.glob(str(params_dir / "*.json"))
    return keep_latest_file(json_files, keep_count=1, dry_run=dry_run)

def cleanup_images(analysis_exports_dir=ANALYSIS_EXPORTS_DIR, keep_count=5, dry_run=False):
    analysis_exports_dir = Path(analysis_exports_dir)
    analysis_exports_dir.mkdir(parents=True, exist_ok=True)
    image_files = []
    for extension in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(str(analysis_exports_dir / extension)))
    return keep_latest_file(image_files, keep_count=keep_count, dry_run=dry_run)

def cleanup_all(dry_run=False):
    return {
        "json": cleanup_json_files(dry_run=dry_run),
        "images": cleanup_images(keep_count=5, dry_run=dry_run),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nettoyage des fichiers (JSON, images, datasets)")
    parser.add_argument("--dry-run", action="store_true", help="Simule le nettoyage sans supprimer")
    parser.add_argument("--keep-images", type=int, default=5, help="Nombre d'images à garder")
    parser.add_argument("--keep-datasets", type=int, default=1, help="Nombre de datasets à garder")
    args = parser.parse_args()
    result = cleanup_all(dry_run=args.dry_run)
    print("[CLEANUP] Résultat:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
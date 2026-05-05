
import sys
import os
import json
import glob
import subprocess
from pathlib import Path

# =========================
# PATHS (DOCKER SAFE)
# =========================
BASE_DIR = "/app/service"
EXPORT_DIR = os.path.join(BASE_DIR, "analysis_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from script.cleanup import cleanup_json_files, cleanup_images
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script'))




# =========================
# DOWNLOAD LATEST JSON PARAMETER FILE
# =========================

app = FastAPI()

# =========================
# DOWNLOAD FEATURE IMPORTANCE
# =========================
@app.get("/download/feature-importance")
def download_feature_importance():
    # Chercher le fichier feature_importance le plus récent
    fi_files = glob.glob(os.path.join(EXPORT_DIR, "feature_importance_*.png"))
    if not fi_files:
        return {"status": "error", "message": "Graphique d'importance des features non trouvé. Lancez un entraînement d'abord."}
    fi_files.sort(reverse=True)
    file_path = fi_files[0]
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=os.path.basename(file_path),
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
    )
PARAMS_DIR = "/app/Donnee_parametres"
# =========================
# ENDPOINT POUR SAUVEGARDER LES PARAMÈTRES EN JSON
# =========================
@app.post("/save-parameters")
async def save_parameters(request: Request):
    try:
        params = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing JSON: {e}")
    if not os.path.exists(PARAMS_DIR):
        os.makedirs(PARAMS_DIR, exist_ok=True)
    filename = f"parameters_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    file_path = os.path.join(PARAMS_DIR, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde: {e}")
    return {"status": "success", "filename": filename}

@app.get("/download/last-parameters-json")
def download_last_parameters_json():
    if not os.path.exists(PARAMS_DIR):
        raise HTTPException(status_code=404, detail="Dossier Donnee_parametres introuvable")
    json_files = [f for f in os.listdir(PARAMS_DIR) if f.endswith('.json')]
    if not json_files:
        raise HTTPException(status_code=404, detail="Aucun fichier JSON trouvé dans Donnee_parametres")
    json_files.sort(reverse=True)
    file_path = os.path.join(PARAMS_DIR, json_files[0])
    return FileResponse(
        file_path,
        media_type="application/json",
        filename=os.path.basename(file_path),
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
    )


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PRED_SCRIPT = os.path.join(BASE_DIR, "predict_from_file.py")

# =========================
# STATIC FILES
# =========================
app.mount("/static", StaticFiles(directory=EXPORT_DIR), name="static")

# =========================
# IMAGE TYPES
# =========================
IMAGE_TYPES = {
    "tree": "arbre_",
    "correlation": "correlation_",
    "residus": "residus_",
    "learning_curve": "learning_curve_",
    "dashboard": "dashboard_",
    "feature_importance": "feature_importance_",
    "shap_bar": "shap_bar_",
    "shap_beeswarm": "shap_beeswarm_",
    "pred_vs_real": "pred_vs_real_"
}
# =========================
# ENDPOINT POUR LISTER TOUS LES GRAPHES DISPONIBLES
# =========================
@app.get("/graphs")
def list_all_graphs():
    if not os.path.exists(EXPORT_DIR):
        return []
    files = os.listdir(EXPORT_DIR)
    result = []
    for img_type, prefix in IMAGE_TYPES.items():
        candidates = [
            f for f in files
            if f.startswith(prefix) and f.endswith(".png")
        ]
        candidates.sort(reverse=True)
        for fname in candidates:
            result.append({
                "type": img_type,
                "url": f"http://localhost:8000/static/{fname}",
                "filename": fname
            })
    return result

# =========================
# ENDPOINT POUR NETTOYAGE MANUEL
# =========================
@app.post("/cleanup")
def cleanup_all():
    cleanup_json_files()
    cleanup_images()
    return {"status": "success", "message": "Nettoyage effectué"}

# =========================
# ENDPOINT POUR LISTER TOUS LES PARAMS JSON
# =========================
@app.get("/list-parameters")
def list_parameters():
    if not os.path.exists(PARAMS_DIR):
        return []
    files = [f for f in os.listdir(PARAMS_DIR) if f.endswith('.json')]
    files.sort(reverse=True)
    return files

# =========================
# GET LATEST IMAGES
# =========================
def get_latest_images():
    if not os.path.exists(EXPORT_DIR):
        return []

    files = os.listdir(EXPORT_DIR)
    result = []

    for img_type, prefix in IMAGE_TYPES.items():
        candidates = [
            f for f in files
            if f.startswith(prefix) and f.endswith(".png")
        ]

        if not candidates:
            continue

        candidates.sort(reverse=True)
        latest = candidates[0]

        result.append({
            "type": img_type,
            "url": f"http://localhost:8000/static/{latest}",
            "filename": latest
        })

    return result

# =========================
# GET TREE IMAGES
# =========================
def get_tree_images():
    """Récupère les images des arbres XGBoost"""
    if not os.path.exists(EXPORT_DIR):
        return []

    tree_images = []
    
    # Chercher tous les fichiers tree_
    all_tree_files = glob.glob(os.path.join(EXPORT_DIR, "tree_*.png"))
    
    for filepath in all_tree_files:
        filename = os.path.basename(filepath)
        tree_images.append({
            "type": "tree",
            "url": f"http://localhost:8000/static/{filename}",
            "filename": filename
        })
    
    return tree_images

# =========================
# SIMPLE PREDICTION
# =========================
@app.get("/predict/simple")
def predict_simple(cpu: float, ram: float, visitors: int, plugins: int, growth: float):
    score = max(0, min(100, 100 - (cpu + ram) / 2 + growth))

    return {
        "predicted_load": round(score, 2),
        "xgboost_score": round(score - 5, 2),
        "saturation_months": max(1, int(36 - growth)),
        "status": "CRITIQUE" if score > 80 else "SURVEILLANCE" if score > 60 else "OPTIMAL"
    }

# =========================
# PREDICT FROM FILE
# =========================
# =========================
# PREDICT FROM FILE
# =========================
@app.get("/predict/from-file")
def predict_from_file():
    try:
        # =============================================
        # 🔥 NETTOYAGE AVANT CHAQUE PRÉDICTION
        # =============================================
        print("=" * 50)
        print("🧹 Nettoyage avant prédiction...")
        cleanup_json_files()
        cleanup_images()
        print("✅ Nettoyage terminé")
        print("=" * 50)
        
        # =============================================
        # PRÉDICTION
        # =============================================
        if not os.path.exists(PRED_SCRIPT):
            return {"status": "error", "message": f"Script introuvable: {PRED_SCRIPT}"}

        result = subprocess.run(
            ["python", PRED_SCRIPT],
            capture_output=True,
            text=True,
            timeout=60
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        try:
            output_json = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Erreur parsing JSON: {str(e)}",
                "raw_output": result.stdout,
                "stderr": result.stderr
            }

        # Extraire correctement le résultat et les images du sous-dictionnaire 'output'
        output = output_json.get("output", {})
        prediction_result = output.get("result", {})
        images = output.get("images", [])
        trees = output.get("trees", [])
        source = output.get("source", "")

        return {
            "status": "success",
            "output": {
                "result": prediction_result,
                "images": images,
                "trees": trees,
                "source": source
            }
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Le script a dépassé le temps limite (60s)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================
# DOWNLOAD TREE 0
# =========================
@app.get("/download/tree0")
def download_tree0():
    # Chercher le fichier tree_0 le plus récent
    tree_files = glob.glob(os.path.join(EXPORT_DIR, "tree_0*.png"))
    
    if not tree_files:
        # Essayer aussi xgboost_tree_0
        tree_files = glob.glob(os.path.join(EXPORT_DIR, "xgboost_tree_0*.png"))
    
    if not tree_files:
        return {"status": "error", "message": "Tree 0 non trouvé. Lancez une analyse d'abord."}
    
    tree_files.sort(reverse=True)
    file_path = tree_files[0]
    
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=os.path.basename(file_path),
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
    )

# =========================
# DOWNLOAD TREE FINAL
# =========================
@app.get("/download/tree-final")
def download_tree_final():
    # Chercher le fichier tree_final le plus récent
    tree_files = glob.glob(os.path.join(EXPORT_DIR, "tree_final*.png"))
    
    if not tree_files:
        # Essayer aussi xgboost_tree_final
        tree_files = glob.glob(os.path.join(EXPORT_DIR, "xgboost_tree_final*.png"))
    
    if not tree_files:
        return {"status": "error", "message": "Tree Final non trouvé. Lancez une analyse d'abord."}
    
    tree_files.sort(reverse=True)
    file_path = tree_files[0]
    
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=os.path.basename(file_path),
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
    )

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {
        "status": "API running",
        "export_dir": EXPORT_DIR,
        "export_dir_exists": os.path.exists(EXPORT_DIR),
        "endpoints": [
            "/predict/simple",
            "/predict/from-file",
            "/download/tree0",
            "/download/tree-final",
            "/static/{filename}"
        ]
    }

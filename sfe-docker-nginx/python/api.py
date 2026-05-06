import sys
import os
import json
import glob
import subprocess
from threading import Thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from script.cleanup import cleanup_json_files, cleanup_images
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script'))

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

# =========================
# PATHS (DOCKER SAFE)
# =========================
BASE_DIR = "/app/service"
PARAMS_DIR = "/app/Donnee_parametres"
GRAPHE_DIR = os.path.join(BASE_DIR, "graphe")
EXPORT_DIR = os.path.join(BASE_DIR, "analysis_exports")

# Créer le dossier s'il n'existe pas
os.makedirs(EXPORT_DIR, exist_ok=True)

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

# =========================
# STATIC FILES
# =========================
app.mount("/static", StaticFiles(directory=EXPORT_DIR), name="static")

# =========================
# ENDPOINT POUR LANCER LES GRAPHIQUES
# =========================
@app.post("/run/graphe-xgboost-model")
def run_graphe_xgboost_model():
    """
    Lance la génération de tous les graphiques d'analyse.
    """
    try:
        # Import local pour éviter les problèmes de scope
        service_dir = os.path.join(os.path.dirname(__file__), "service")
        if service_dir not in sys.path:
            sys.path.insert(0, service_dir)
        
        from graphe_xgboost_parameters import generate_all_graphs as gen_graphs
        
        result = gen_graphs()
        
        if result.get("status") == "success":
            graphs_count = len(result.get('graphs', {}))
            return {
                "status": "success",
                "message": f"{graphs_count} graphiques générés avec succès",
                "graphs": result.get("graphs", {})
            }
        else:
            return {
                "status": "error", 
                "message": result.get("message", "Erreur inconnue lors de la génération")
            }
            
    except ImportError as e:
        import traceback
        print(f"Erreur d'import: {traceback.format_exc()}", file=sys.stderr)
        return {
            "status": "error", 
            "message": f"Module graphe_xgboost_parameters introuvable: {str(e)}"
        }
    except Exception as e:
        import traceback
        print(f"Erreur génération graphiques: {traceback.format_exc()}", file=sys.stderr)
        return {
            "status": "error", 
            "message": f"Erreur: {str(e)}"
        }

# =========================
# GENERATE ALL GRAPHS (direct call)
# =========================
@app.post("/generate/all-graphs")
def generate_all_graphs_endpoint():
    """
    Lance la génération de tous les graphiques d'analyse (appel direct).
    """
    try:
        service_dir = os.path.join(os.path.dirname(__file__), "service")
        if service_dir not in sys.path:
            sys.path.insert(0, service_dir)
        
        from graphe_xgboost_parameters import generate_all_graphs as gen_graphs
        
        result = gen_graphs()
        return result
    except Exception as e:
        import traceback
        print(f"Erreur generate_all_graphs: {traceback.format_exc()}", file=sys.stderr)
        return {"status": "error", "message": str(e)}

# =========================
# ENDPOINT POUR LISTER LES IMAGES
# =========================
@app.get("/get-images-list")
def get_images_list():
    """
    Retourne la liste des images disponibles dans analysis_exports
    """
    try:
        if not os.path.exists(EXPORT_DIR):
            return {"status": "success", "images": []}
        
        images = []
        image_files = glob.glob(os.path.join(EXPORT_DIR, "*.png"))
        
        # Trier par date de modification (plus récent en premier)
        image_files.sort(key=os.path.getmtime, reverse=True)
        
        for filepath in image_files[:12]:  # Limiter à 12 images max
            filename = os.path.basename(filepath)
            
            # Déterminer le type d'image
            img_type = "graph"
            if "arbre" in filename.lower():
                img_type = "tree"
            elif "correlation" in filename.lower():
                img_type = "correlation"
            elif "learning_curve" in filename.lower():
                img_type = "learning_curve"
            elif "residus" in filename.lower():
                img_type = "residus"
            elif "partial_dependence" in filename.lower():
                img_type = "feature_importance"
            
            images.append({
                "type": img_type,
                "url": f"http://localhost:8000/static/{filename}",
                "filename": filename
            })
        
        return {"status": "success", "images": images}
        
    except Exception as e:
        return {"status": "error", "images": [], "message": str(e)}

# =========================
# DOWNLOAD TREE_0.PNG DEPUIS GRAPHE
# =========================
@app.get("/download/graphe/tree0")
def download_graphe_tree0():
    file_path = os.path.join(GRAPHE_DIR, "tree_0.png")
    if not os.path.exists(file_path):
        return {"status": "error", "message": "tree_0.png non trouvé dans graphe."}
    return FileResponse(
        file_path,
        media_type="image/png",
        filename="tree_0.png",
        headers={"Content-Disposition": "attachment; filename=tree_0.png"}
    )

# =========================
# DOWNLOAD TREE_FINAL.PNG DEPUIS GRAPHE
# =========================
@app.get("/download/graphe/treefinal")
def download_graphe_treefinal():
    file_path = os.path.join(GRAPHE_DIR, "tree_final.png")
    if not os.path.exists(file_path):
        return {"status": "error", "message": "tree_final.png non trouvé dans graphe."}
    return FileResponse(
        file_path,
        media_type="image/png",
        filename="tree_final.png",
        headers={"Content-Disposition": "attachment; filename=tree_final.png"}
    )

# =========================
# DOWNLOAD FEATURE IMPORTANCE PNG DEPUIS GRAPHE
# =========================
@app.get("/download/graphe/feature_importance")
def download_graphe_feature_importance():
    files = glob.glob(os.path.join(GRAPHE_DIR, "feature_importance_*.png"))
    if not files:
        return {"status": "error", "message": "Aucun feature_importance_*.png trouvé dans graphe."}
    files.sort(reverse=True)
    file_path = files[0]
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=os.path.basename(file_path),
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
    )

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

# =========================
# ENDPOINT POUR PRÉDICTION DEPUIS FICHIER
# =========================
@app.get("/predict/from-file")
def predict_from_file_endpoint():
    """
    Lance la prédiction à partir du dernier fichier JSON de paramètres sauvegardé.
    """
    try:
        # Import dynamique pour éviter les problèmes de scope
        service_dir = os.path.join(os.path.dirname(__file__), "service")
        if service_dir not in sys.path:
            sys.path.insert(0, service_dir)
        from predict_from_file import predict_from_json
        result = predict_from_json()
        return result
    except Exception as e:
        import traceback
        print(f"Erreur predict_from_file: {traceback.format_exc()}", file=sys.stderr)
        return {"status": "error", "message": str(e)}
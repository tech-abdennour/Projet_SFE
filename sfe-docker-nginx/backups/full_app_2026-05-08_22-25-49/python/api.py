import glob
import json
import os
import sys
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT_DIR = os.path.dirname(__file__)
SCRIPT_DIR = os.path.join(ROOT_DIR, "script")
SERVICE_DIR = os.path.join(ROOT_DIR, "service")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)


app = FastAPI()

BASE_DIR = "/app/service" if os.path.exists("/app") else SERVICE_DIR
PARAMS_DIR = "/app/Donnee_parametres" if os.path.exists("/app") else os.path.join(ROOT_DIR, "Donnee_parametres")
GRAPHE_DIR = os.path.join(BASE_DIR, "graphe")
EXPORT_DIR = os.path.join(BASE_DIR, "analysis_exports")

os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=EXPORT_DIR), name="static")


def run_cleanup_json():
    """Garde uniquement le dernier fichier JSON"""
    if not os.path.exists(PARAMS_DIR):
        return {"status": "success", "message": "Dossier JSON n'existe pas"}
    
    json_files = glob.glob(os.path.join(PARAMS_DIR, "*.json"))
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    removed = []
    for file_path in json_files[1:]:  # Garde le premier (plus récent)
        try:
            os.remove(file_path)
            removed.append(os.path.basename(file_path))
        except Exception as e:
            return {"status": "error", "message": f"Erreur suppression {file_path}: {e}"}
    
    return {
        "status": "success",
        "message": f"Nettoyage JSON terminé. {'Supprimé(s): ' + ', '.join(removed) if removed else 'Aucun fichier à supprimer'}",
        "removed": removed,
        "kept": os.path.basename(json_files[0]) if json_files else None
    }


def run_cleanup_images_all():
    """Supprime TOUTES les images"""
    if not os.path.exists(EXPORT_DIR):
        return {"status": "success", "message": "Dossier images n'existe pas"}
    
    image_files = glob.glob(os.path.join(EXPORT_DIR, "*.png"))
    
    removed = []
    for file_path in image_files:
        try:
            os.remove(file_path)
            removed.append(os.path.basename(file_path))
        except Exception as e:
            return {"status": "error", "message": f"Erreur suppression {file_path}: {e}"}
    
    return {
        "status": "success",
        "message": f"Toutes les images supprimées. {'Supprimée(s): ' + ', '.join(removed) if removed else 'Aucune image à supprimer'}",
        "removed": removed,
        "kept": []
    }


def run_cleanup_images_keep_5():
    """Garde uniquement les 5 dernières images"""
    if not os.path.exists(EXPORT_DIR):
        return {"status": "success", "message": "Dossier images n'existe pas"}
    
    image_files = glob.glob(os.path.join(EXPORT_DIR, "*.png"))
    image_files.sort(key=os.path.getmtime, reverse=True)
    
    removed = []
    for file_path in image_files[5:]:  # Garde les 5 premiers (plus récents)
        try:
            os.remove(file_path)
            removed.append(os.path.basename(file_path))
        except Exception as e:
            return {"status": "error", "message": f"Erreur suppression {file_path}: {e}"}
    
    return {
        "status": "success",
        "message": f"Nettoyage images terminé. {'Supprimée(s): ' + ', '.join(removed) if removed else 'Aucune image à supprimer'}",
        "removed": removed,
        "kept": [os.path.basename(f) for f in image_files[:5]]
    }


def execute_predict_from_file():
    try:
        if SERVICE_DIR not in sys.path:
            sys.path.insert(0, SERVICE_DIR)

        from predict_from_file import predict_from_json

        result = predict_from_json()

        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result)

        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        print(f"Erreur predict_from_file: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur predict_from_file.py: {str(e)}",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run/cleanup-all-images")
def cleanup_all_images():
    """Endpoint pour supprimer TOUTES les images"""
    try:
        result = run_cleanup_images_all()
        return {"status": "success", "message": "Toutes les images supprimées", "details": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/run/cleanup-keep-5-images")
def cleanup_keep_5_images():
    """Endpoint pour garder les 5 dernières images"""
    try:
        result = run_cleanup_images_keep_5()
        return {"status": "success", "message": "Images nettoyées (5 dernières gardées)", "details": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/run/cleanup-json")
def cleanup_json():
    """Endpoint pour garder le dernier JSON"""
    try:
        result = run_cleanup_json()
        return {"status": "success", "message": "JSON nettoyé (dernier gardé)", "details": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/run/graphe-xgboost-model")
def run_graphe_xgboost_model():
    try:
        if SERVICE_DIR not in sys.path:
            sys.path.insert(0, SERVICE_DIR)

        from graphe_xgboost_parameters import generate_all_graphs as gen_graphs

        result = gen_graphs()

        # Après génération, garder les 5 dernières images
        cleanup_result = run_cleanup_images_keep_5()

        if result.get("status") == "success":
            graphs_count = len(result.get("graphs", {}))

            return {
                "status": "success",
                "message": f"{graphs_count} graphiques générés avec succès",
                "graphs": result.get("graphs", {}),
                "cleanup": cleanup_result,
            }

        return {
            "status": "error",
            "message": result.get("message", "Erreur inconnue lors de la génération"),
            "cleanup": cleanup_result,
        }

    except ImportError as e:
        import traceback
        print(f"Erreur d'import: {traceback.format_exc()}", file=sys.stderr)
        return {
            "status": "error",
            "message": f"Module graphe_xgboost_parameters introuvable: {str(e)}",
        }
    except Exception as e:
        import traceback
        print(f"Erreur génération graphiques: {traceback.format_exc()}", file=sys.stderr)
        return {
            "status": "error",
            "message": f"Erreur: {str(e)}",
        }


@app.post("/generate/all-graphs")
def generate_all_graphs_endpoint():
    return run_graphe_xgboost_model()


@app.post("/generate-graphs")
def generate_graphs_endpoint():
    return run_graphe_xgboost_model()


@app.get("/generate-graphs")
def generate_graphs_endpoint_get():
    return run_graphe_xgboost_model()


@app.get("/get-images-list")
def get_images_list():
    try:
        if not os.path.exists(EXPORT_DIR):
            return {"status": "success", "images": []}

        images = []
        image_files = glob.glob(os.path.join(EXPORT_DIR, "*.png"))
        image_files.sort(key=os.path.getmtime, reverse=True)

        for filepath in image_files:
            filename = os.path.basename(filepath)
            img_type = "graph"

            if "arbre" in filename.lower() or "tree" in filename.lower():
                img_type = "tree"
            elif "correlation" in filename.lower():
                img_type = "correlation"
            elif "learning_curve" in filename.lower():
                img_type = "learning_curve"
            elif "residus" in filename.lower() or "residual" in filename.lower():
                img_type = "residus"
            elif "feature" in filename.lower() or "importance" in filename.lower():
                img_type = "feature_importance"

            images.append({
                "type": img_type,
                "url": f"/api/static/{filename}",
                "filename": filename,
            })

        return {"status": "success", "images": images}
    except Exception as e:
        return {"status": "error", "images": [], "message": str(e)}


@app.get("/api/static/{filename}")
async def get_static_file(filename: str):
    """Servir les fichiers statiques"""
    file_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    return FileResponse(file_path)


@app.get("/download/graphe/tree0")
def download_graphe_tree0():
    file_path = os.path.join(GRAPHE_DIR, "tree_0.png")
    if not os.path.exists(file_path):
        return {"status": "error", "message": "tree_0.png non trouvé dans graphe."}
    return FileResponse(file_path, media_type="image/png", filename="tree_0.png",
                       headers={"Content-Disposition": "attachment; filename=tree_0.png"})


@app.get("/download/graphe/treefinal")
def download_graphe_treefinal():
    file_path = os.path.join(GRAPHE_DIR, "tree_final.png")
    if not os.path.exists(file_path):
        return {"status": "error", "message": "tree_final.png non trouvé dans graphe."}
    return FileResponse(file_path, media_type="image/png", filename="tree_final.png",
                       headers={"Content-Disposition": "attachment; filename=tree_final.png"})


@app.get("/download/graphe/feature_importance")
def download_graphe_feature_importance():
    files = glob.glob(os.path.join(GRAPHE_DIR, "feature_importance_*.png"))
    if not files:
        return {"status": "error", "message": "Aucun feature_importance_*.png trouvé dans graphe."}
    files.sort(key=os.path.getmtime, reverse=True)
    file_path = files[0]
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/save-parameters")
async def save_parameters(request: Request):
    try:
        params = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing JSON: {e}")

    filename = f"parametres_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(PARAMS_DIR, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde: {e}")

    # Nettoyage : garde uniquement le dernier JSON
    cleanup_result = run_cleanup_json()

    return {
        "success": True,
        "status": "success",
        "filename": filename,
        "path": file_path,
        "data": params,
        "cleanup": cleanup_result,
    }


@app.post("/save-parametres-json")
async def save_parametres_json(request: Request):
    return await save_parameters(request)


@app.get("/predict/from-file")
def predict_from_file_endpoint():
    return execute_predict_from_file()


@app.post("/predict/from-file")
def predict_from_file_endpoint_post():
    return execute_predict_from_file()


@app.get("/predict-from-file")
def predict_from_file_alias():
    return execute_predict_from_file()


@app.post("/predict-from-file")
def predict_from_file_alias_post():
    return execute_predict_from_file()


@app.get("/predict")
def predict_alias():
    return execute_predict_from_file()


@app.post("/predict")
def predict_alias_post():
    return execute_predict_from_file()


@app.get("/predict-latest-json")
def predict_latest_json():
    return execute_predict_from_file()


@app.post("/predict-latest-json")
def predict_latest_json_post():
    return execute_predict_from_file()


@app.post("/save-and-predict-json")
async def save_and_predict_json(request: Request):
    # 1. Sauvegarder les paramètres
    saved = await save_parameters(request)
    
    # 2. Supprimer TOUTES les anciennes images
    cleanup_images = run_cleanup_images_all()
    
    # 3. Exécuter la prédiction
    prediction = execute_predict_from_file()
    
    # 4. Nettoyer les JSON (garder le dernier)
    cleanup_json = run_cleanup_json()

    return {
        "success": True,
        "status": "success",
        "message": "Fichier JSON sauvegardé et prédiction exécutée",
        "saved_file": saved,
        "prediction": prediction,
        "cleanup": {
            "images": cleanup_images,
            "json": cleanup_json
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
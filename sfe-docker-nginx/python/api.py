from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import glob
import json
import os
import sys
from datetime import datetime

app = FastAPI()
router = APIRouter()

# Chemins
ROOT_DIR = os.path.dirname(__file__)
SERVICE_DIR = os.path.join(ROOT_DIR, "service")

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


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def find_latest_file(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def run_cleanup_json():
    if not os.path.exists(PARAMS_DIR):
        return {"status": "success", "message": "Dossier JSON n'existe pas"}
    json_files = glob.glob(os.path.join(PARAMS_DIR, "*.json"))
    json_files.sort(key=os.path.getmtime, reverse=True)
    removed = []
    for file_path in json_files[1:]:
        try:
            os.remove(file_path)
            removed.append(os.path.basename(file_path))
        except Exception as e:
            return {"status": "error", "message": f"Erreur suppression {file_path}: {e}"}
    return {
        "status": "success",
        "message": f"Nettoyage JSON termine",
        "removed": removed,
        "kept": os.path.basename(json_files[0]) if json_files else None
    }


def run_cleanup_images_keep_5():
    if not os.path.exists(EXPORT_DIR):
        return {"status": "success", "message": "Dossier images n'existe pas"}
    image_files = glob.glob(os.path.join(EXPORT_DIR, "*.png"))
    image_files.sort(key=os.path.getmtime, reverse=True)
    removed = []
    for file_path in image_files[5:]:
        try:
            os.remove(file_path)
            removed.append(os.path.basename(file_path))
        except Exception as e:
            return {"status": "error", "message": f"Erreur suppression {file_path}: {e}"}
    return {
        "status": "success",
        "message": "Nettoyage images termine",
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
        if 'output' in result and 'result' in result['output']:
            score = result['output']['result'].get('xgboost_score', None)
            if score is None or score == '' or score == 'null' or (isinstance(score, float) and (score != score)):
                result['output']['result']['xgboost_score'] = 'N/A'
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Erreur predict_from_file: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# ============================================
# HEALTH
# ============================================

@router.get("/health")
def health():
    return {"status": "ok"}


# ============================================
# CLEANUP
# ============================================

@router.post("/run/cleanup-keep-5-images")
def cleanup_keep_5_images():
    try:
        result = run_cleanup_images_keep_5()
        return {"status": "success", "message": "Images nettoyees", "details": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/run/cleanup-json")
def cleanup_json():
    try:
        result = run_cleanup_json()
        return {"status": "success", "message": "JSON nettoye", "details": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================
# GRAPHIQUES XGBOOST
# ============================================

@router.post("/run/graphe-xgboost-model")
def run_graphe_xgboost_model():
    try:
        if SERVICE_DIR not in sys.path:
            sys.path.insert(0, SERVICE_DIR)
        from graphe_xgboost_parameters import generate_all_graphs as gen_graphs
        result = gen_graphs()
        cleanup_result = run_cleanup_images_keep_5()
        graphs_count = len(result.get("graphs", {}))
        return {
            "status": "success",
            "message": f"{graphs_count} graphique(s) genere(s)",
            "graphs": result.get("graphs", {}),
            "errors": result.get("errors", {}),
            "current_load": result.get("current_load", "N/A"),
            "cleanup": cleanup_result,
        }
    except ImportError as e:
        import traceback
        print(f"Erreur d'import: {traceback.format_exc()}", file=sys.stderr)
        return {"status": "error", "message": f"Module introuvable: {str(e)}"}
    except Exception as e:
        import traceback
        print(f"Erreur generation: {traceback.format_exc()}", file=sys.stderr)
        return {"status": "error", "message": f"Erreur: {str(e)}"}


@router.post("/generate/all-graphs")
def generate_all_graphs_endpoint():
    return run_graphe_xgboost_model()


@router.post("/generate-graphs")
def generate_graphs_endpoint():
    return run_graphe_xgboost_model()


@router.get("/generate-graphs")
def generate_graphs_endpoint_get():
    return run_graphe_xgboost_model()


# ============================================
# IMAGES LIST
# ============================================

@router.get("/get-images-list")
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
            if "tree" in filename.lower() or "arbre" in filename.lower():
                img_type = "tree"
            elif "correlation" in filename.lower():
                img_type = "correlation"
            elif "learning_curve" in filename.lower():
                img_type = "learning_curve"
            elif "residus" in filename.lower() or "residual" in filename.lower():
                img_type = "residus"
            elif "feature" in filename.lower() or "importance" in filename.lower():
                img_type = "feature_importance"
            elif "partial_dependence" in filename.lower():
                img_type = "partial_dependence"
            elif "saturation" in filename.lower():
                img_type = "saturation_evolution"
            elif "charge_par_type" in filename.lower():
                img_type = "charge_par_type"
            elif "charge_horaire" in filename.lower():
                img_type = "charge_horaire"
            elif "time_series" in filename.lower():
                img_type = "time_series"
            elif "response_time_projection" in filename.lower():
                img_type = "response_time_projection"
            elif "shap_force_plot" in filename.lower():
                img_type = "shap_force_plot" if "missing" not in filename.lower() else "shap_missing"
            elif "decision_boundary" in filename.lower():
                img_type = "decision_boundary"
            
            images.append({
                "type": img_type,
                "url": f"/api/static/{filename}",
                "filename": filename,
                "created": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
        
        return {"status": "success", "images": images, "count": len(images)}
    except Exception as e:
        return {"status": "error", "images": [], "message": str(e)}


@router.get("/static/{filename}")
async def get_static_file(filename: str):
    file_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouve")
    return FileResponse(file_path)


# ============================================
# TELECHARGEMENT GRAPHES
# ============================================

@router.get("/download/graphe/tree0")
def download_graphe_tree0():
    file_path = os.path.join(GRAPHE_DIR, "tree_0.png")
    if not os.path.exists(file_path):
        return {"status": "error", "message": "tree_0.png non trouve"}
    return FileResponse(file_path, media_type="image/png", filename="tree_0.png",
                       headers={"Content-Disposition": "attachment; filename=tree_0.png"})


@router.get("/download/graphe/treefinal")
def download_graphe_treefinal():
    file_path = os.path.join(GRAPHE_DIR, "tree_final.png")
    if not os.path.exists(file_path):
        return {"status": "error", "message": "tree_final.png non trouve"}
    return FileResponse(file_path, media_type="image/png", filename="tree_final.png",
                       headers={"Content-Disposition": "attachment; filename=tree_final.png"})


@router.get("/download/graphe/feature_importance")
def download_graphe_feature_importance():
    file_path = find_latest_file(GRAPHE_DIR, "feature_importance_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun feature_importance trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/learning_curve")
def download_graphe_learning_curve():
    file_path = find_latest_file(GRAPHE_DIR, "learning_curve_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun learning_curve trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/residus")
def download_graphe_residus():
    file_path = find_latest_file(GRAPHE_DIR, "residus_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun residus trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/correlation")
def download_graphe_correlation():
    file_path = find_latest_file(GRAPHE_DIR, "correlation_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun correlation trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/saturation")
def download_graphe_saturation():
    file_path = find_latest_file(EXPORT_DIR, "saturation_evolution_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun saturation_evolution trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/time-series")
def download_graphe_time_series():
    file_path = find_latest_file(EXPORT_DIR, "time_series_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun time_series trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/shap-force-plot")
def download_graphe_shap():
    file_path = find_latest_file(EXPORT_DIR, "shap_force_plot_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun shap_force_plot trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/charge-par-type")
def download_graphe_charge_par_type():
    file_path = find_latest_file(EXPORT_DIR, "charge_par_type_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun charge_par_type trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/partial-dependence")
def download_graphe_partial_dependence():
    file_path = find_latest_file(EXPORT_DIR, "partial_dependence_all_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun partial_dependence trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/download/graphe/decision-boundary")
def download_graphe_decision_boundary():
    file_path = find_latest_file(EXPORT_DIR, "decision_boundary_*.png")
    if not file_path:
        return {"status": "error", "message": "Aucun decision_boundary trouve"}
    filename = os.path.basename(file_path)
    return FileResponse(file_path, media_type="image/png", filename=filename,
                       headers={"Content-Disposition": f"attachment; filename={filename}"})


# ============================================
# PREDICTION
# ============================================

@router.post("/save-parameters")
async def save_parameters(request: Request):
    try:
        params = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur JSON: {e}")
    
    filename = f"parametres_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(PARAMS_DIR, filename)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur sauvegarde: {e}")
    
    cleanup_result = run_cleanup_json()
    
    return {
        "success": True,
        "status": "success",
        "filename": filename,
        "path": file_path,
        "data": params,
        "cleanup": cleanup_result,
    }


@router.post("/save-parametres-json")
async def save_parametres_json(request: Request):
    return await save_parameters(request)


@router.get("/predict/from-file")
def predict_from_file_endpoint():
    return execute_predict_from_file()


@router.post("/predict/from-file")
def predict_from_file_endpoint_post():
    return execute_predict_from_file()


@router.get("/predict-from-file")
def predict_from_file_alias():
    return execute_predict_from_file()


@router.post("/predict-from-file")
def predict_from_file_alias_post():
    return execute_predict_from_file()


@router.get("/predict")
def predict_alias():
    return execute_predict_from_file()


@router.post("/predict")
def predict_alias_post():
    return execute_predict_from_file()


@router.get("/predict-latest-json")
def predict_latest_json():
    return execute_predict_from_file()


@router.post("/predict-latest-json")
def predict_latest_json_post():
    return execute_predict_from_file()


@router.post("/save-and-predict-json")
async def save_and_predict_json(request: Request):
    saved = await save_parameters(request)
    prediction = execute_predict_from_file()
    cleanup_json = run_cleanup_json()
    
    return {
        "success": True,
        "status": "success",
        "message": "JSON sauvegarde et prediction executee",
        "saved_file": saved,
        "prediction": prediction,
        "cleanup_json": cleanup_json,
    }


# ============================================
# ENREGISTREMENT DES ROUTES
# ============================================

app.include_router(router, prefix="/api")

# S'assurer que l'instance FastAPI 'app' existe bien pour ASGI
# (déjà présent en haut du fichier)
# from fastapi import FastAPI
# app = FastAPI()

# Enregistrement du router (déjà présent plus haut)
# app.include_router(router, prefix="/api")

# Le code ci-dessus garantit que l'attribut 'app' est bien exposé pour ASGI/uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
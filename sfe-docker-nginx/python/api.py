from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
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

print(f"📁 Configuration API:")
print(f"  • ROOT_DIR: {ROOT_DIR}")
print(f"  • SERVICE_DIR: {SERVICE_DIR}")
print(f"  • BASE_DIR: {BASE_DIR}")
print(f"  • PARAMS_DIR: {PARAMS_DIR}")
print(f"  • EXPORT_DIR: {EXPORT_DIR}")
print(f"  • GRAPHE_DIR: {GRAPHE_DIR}")

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
    """Trouve le fichier le plus récent correspondant au pattern"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def run_cleanup_json():
    """Supprime tous les fichiers JSON sauf le plus récent"""
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
        "message": "Nettoyage JSON terminé",
        "removed": removed,
        "kept": os.path.basename(json_files[0]) if json_files else None
    }


def run_cleanup_all_images():
    """Supprime toutes les images dans le dossier d'export"""
    deleted = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif"):
        for file_path in glob.glob(os.path.join(EXPORT_DIR, ext)):
            try:
                os.remove(file_path)
                deleted.append(os.path.basename(file_path))
            except Exception:
                pass
    return {"deleted": deleted, "deleted_count": len(deleted)}


def execute_predict_from_file():
    """Exécute la prédiction à partir du fichier JSON le plus récent"""
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


def execute_graphes_generation():
    """Exécute la génération des 4 graphiques via graphe_xgboost_parameters.py"""
    try:
        if SERVICE_DIR not in sys.path:
            sys.path.insert(0, SERVICE_DIR)
        
        print(f"📂 Tentative d'import depuis: {SERVICE_DIR}")
        print(f"📄 Fichier cible: graphe_xgboost_parameters.py")
        print(f"📁 Contenu du dossier: {os.listdir(SERVICE_DIR)}")
        
        # Importer le module graphe_xgboost_parameters
        import importlib
        graphes_module = importlib.import_module("graphe_xgboost_parameters")
        
        # Appeler la fonction principale generate_all_graphs()
        result = graphes_module.generate_all_graphs()
        
        print(f"📊 Résultat génération graphiques: {result.get('status', 'unknown')}")
        if result.get("status") == "success":
            graphs = result.get('graphs', {})
            print(f"  ✅ {len(graphs)} graphiques générés:")
            for name, path in graphs.items():
                print(f"     • {name}: {os.path.basename(path)}")
        else:
            print(f"  ❌ Erreur: {result.get('message', 'Inconnue')}")
        
        return result
    except ImportError as e:
        import traceback
        print(f"❌ Erreur d'import: {traceback.format_exc()}", file=sys.stderr)
        print(f"   Vérifiez que le fichier existe: {SERVICE_DIR}/graphe_xgboost_parameters.py")
        return {
            "status": "error",
            "message": f"Module graphe_xgboost_parameters introuvable: {str(e)}",
            "graphs": {},
            "errors": {"import": str(e)}
        }
    except AttributeError as e:
        import traceback
        print(f"❌ Erreur fonction generate_all_graphs: {traceback.format_exc()}", file=sys.stderr)
        return {
            "status": "error",
            "message": f"Fonction generate_all_graphs() introuvable: {str(e)}",
            "graphs": {},
            "errors": {"attribute": str(e)}
        }
    except Exception as e:
        import traceback
        print(f"❌ Erreur génération graphiques: {traceback.format_exc()}", file=sys.stderr)
        return {
            "status": "error",
            "message": f"Erreur: {str(e)}",
            "graphs": {},
            "errors": {"exception": str(e)}
        }


def get_graphe_file_mapping():
    """
    Retourne le mapping entre le nom de la route et le fichier le plus récent dans GRAPHE_DIR
    Détecte automatiquement les fichiers basés sur des patterns de noms
    """
    mapping = {
        "tree0": None,
        "treefinal": None,
        "feature_importance": None,
        "learning_curve": None,
        "residus": None,
        "correlation": None,
        "confusion_matrix": None,
    }
    
    if not os.path.exists(GRAPHE_DIR):
        print(f"⚠️ Dossier GRAPHE_DIR introuvable: {GRAPHE_DIR}")
        return mapping
    
    all_files = glob.glob(os.path.join(GRAPHE_DIR, "*.png"))
    print(f"📁 Fichiers trouvés dans GRAPHE_DIR: {len(all_files)}")
    
    for file_path in all_files:
        filename = os.path.basename(file_path).lower()
        file_mtime = os.path.getmtime(file_path)
        
        # Détection basée sur les noms de fichiers
        if "tree_0" in filename or "tree0" in filename:
            if mapping["tree0"] is None or file_mtime > os.path.getmtime(mapping["tree0"]):
                mapping["tree0"] = file_path
                print(f"  ✅ tree0 → {os.path.basename(file_path)}")
                
        elif "tree_final" in filename or "treefinal" in filename:
            if mapping["treefinal"] is None or file_mtime > os.path.getmtime(mapping["treefinal"]):
                mapping["treefinal"] = file_path
                print(f"  ✅ treefinal → {os.path.basename(file_path)}")
                
        elif "feature_importance" in filename or "featureimportance" in filename:
            if mapping["feature_importance"] is None or file_mtime > os.path.getmtime(mapping["feature_importance"]):
                mapping["feature_importance"] = file_path
                print(f"  ✅ feature_importance → {os.path.basename(file_path)}")
                
        elif "learning_curve" in filename or "learningcurve" in filename:
            if mapping["learning_curve"] is None or file_mtime > os.path.getmtime(mapping["learning_curve"]):
                mapping["learning_curve"] = file_path
                print(f"  ✅ learning_curve → {os.path.basename(file_path)}")
                
        elif "residuals" in filename or "residus" in filename:
            if mapping["residus"] is None or file_mtime > os.path.getmtime(mapping["residus"]):
                mapping["residus"] = file_path
                print(f"  ✅ residus → {os.path.basename(file_path)}")
                
        elif "correlation" in filename:
            if mapping["correlation"] is None or file_mtime > os.path.getmtime(mapping["correlation"]):
                mapping["correlation"] = file_path
                print(f"  ✅ correlation → {os.path.basename(file_path)}")
                
        elif "confusion" in filename:
            if mapping["confusion_matrix"] is None or file_mtime > os.path.getmtime(mapping["confusion_matrix"]):
                mapping["confusion_matrix"] = file_path
                print(f"  ✅ confusion_matrix → {os.path.basename(file_path)}")
    
    # Afficher les fichiers manquants
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        print(f"⚠️ Types de graphiques manquants: {missing}")
    
    return mapping


# ============================================
# HEALTH
# ============================================

@router.get("/health")
def health():
    """Vérification de l'état de l'API"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "graphe_dir_exists": os.path.exists(GRAPHE_DIR),
        "params_dir_exists": os.path.exists(PARAMS_DIR),
        "export_dir_exists": os.path.exists(EXPORT_DIR),
    }


# ============================================
# CLEANUP
# ============================================

@router.post("/run/cleanup-all-images")
def cleanup_all_images_endpoint():
    """Supprime toutes les images du dossier d'export"""
    try:
        result = run_cleanup_all_images()
        return {
            "status": "success",
            "message": "Toutes les images ont été supprimées",
            "details": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/run/cleanup-json")
def cleanup_json():
    """Nettoie les fichiers JSON en gardant le plus récent"""
    try:
        result = run_cleanup_json()
        return {"status": "success", "message": "JSON nettoyé", "details": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================
# RESET PARAMETERS AND IMAGES
# ============================================

@router.post("/reset/parameters-and-images")
def reset_parameters_and_images():
    """Réinitialise tous les paramètres et images"""
    print("🔄 Appel /reset/parameters-and-images")
    errors = []
    
    # 1. Supprimer tous les JSON
    json_result = run_cleanup_json()
    if json_result["status"] == "error":
        errors.append(json_result["message"])
    
    # 2. Supprimer toutes les images
    image_result = run_cleanup_all_images()
    
    if errors:
        return {
            "status": "error",
            "message": "Certaines opérations ont échoué",
            "errors": errors
        }
    
    return {
        "status": "success",
        "message": "Tous les paramètres et images ont été supprimés",
        "json_cleanup": json_result,
        "images_cleanup": image_result
    }


# ============================================
# GÉNÉRATION DES 4 GRAPHIQUES D'ANALYSE
# ============================================

@router.post("/generate/analysis-graphs")
def generate_analysis_graphs():
    """
    Génère les 4 graphiques d'analyse via graphe_xgboost_parameters.py :
    1. Radar des Ressources
    2. Jauges de Saturation
    3. Impact des Features
    4. Courbe de Dégradation Temporelle
    """
    print("📊 Appel /generate/analysis-graphs - Génération des 4 graphiques")
    print(f"📂 SERVICE_DIR: {SERVICE_DIR}")
    print(f"📄 Fichiers dans SERVICE_DIR: {os.listdir(SERVICE_DIR) if os.path.exists(SERVICE_DIR) else 'NON TROUVE'}")
    
    try:
        result = execute_graphes_generation()
        
        if result.get("status") == "success":
            graphs = result.get("graphs", {})
            
            # Construire les URLs pour chaque graphique
            graph_urls = {}
            for name, path in graphs.items():
                filename = os.path.basename(path)
                graph_urls[name] = f"/api/static/{filename}"
            
            return {
                "status": "success",
                "message": f"{len(graphs)} graphiques générés avec succès",
                "graphs": graphs,
                "graph_urls": graph_urls,
                "current_load": result.get("current_load"),
                "errors": result.get("errors", {}),
            }
        else:
            return {
                "status": "error",
                "message": result.get("message", "Erreur inconnue"),
                "graphs": {},
                "graph_urls": {},
                "errors": result.get("errors", {}),
            }
    except Exception as e:
        import traceback
        print(f"❌ Erreur generate_analysis_graphs: {traceback.format_exc()}", file=sys.stderr)
        return {
            "status": "error",
            "message": f"Erreur: {str(e)}",
            "graphs": {},
            "graph_urls": {},
        }


# ============================================
# IMAGES LIST
# ============================================

@router.get("/get-images-list")
def get_images_list():
    """Liste toutes les images disponibles dans le dossier d'export"""
    try:
        if not os.path.exists(EXPORT_DIR):
            return {"status": "success", "images": []}
        
        images = []
        image_files = glob.glob(os.path.join(EXPORT_DIR, "*.png"))
        image_files.sort(key=os.path.getmtime, reverse=True)
        
        for filepath in image_files:
            filename = os.path.basename(filepath)
            img_type = "graph"
            
            # Détection des types de graphiques
            if "radar_resources" in filename.lower():
                img_type = "radar_resources"
            elif "gauges_saturation" in filename.lower():
                img_type = "gauges_saturation"
            elif "feature_impact" in filename.lower():
                img_type = "feature_impact"
            elif "degradation_curve" in filename.lower():
                img_type = "degradation_curve"
            elif "gauge_charge" in filename.lower():
                img_type = "gauge_charge"
            elif "timeline_saturation" in filename.lower():
                img_type = "timeline_saturation"
            elif "indicateurs_performance" in filename.lower():
                img_type = "indicators_performance"
            elif "dashboard_combined" in filename.lower():
                img_type = "dashboard_combined"
            
            url = f"/api/static/{filename}"
            
            images.append({
                "type": img_type,
                "url": url,
                "filename": filename,
                "size_kb": round(os.path.getsize(filepath) / 1024, 1),
                "created": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
        
        print(f"🖼️ get-images-list: {len(images)} images trouvées")
        return {"status": "success", "images": images, "count": len(images)}
    except Exception as e:
        print(f"❌ get-images-list error: {e}", file=sys.stderr)
        return {"status": "error", "images": [], "message": str(e)}


@router.get("/static/{filename}")
async def get_static_file(filename: str):
    """Sert un fichier statique depuis le dossier d'export"""
    file_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    return FileResponse(file_path)


# ============================================
# TÉLÉCHARGEMENT DES GRAPHIQUES DE TRAINING
# ============================================

@router.get("/download/graphe/{graphe_type}")
async def download_training_graph(graphe_type: str):
    """
    Télécharge un graphique de training depuis le dossier GRAPHE_DIR
    Types disponibles: tree0, treefinal, feature_importance, learning_curve, residus, correlation, confusion_matrix
    """
    print(f"📥 Demande de téléchargement: {graphe_type}")
    
    mapping = get_graphe_file_mapping()
    
    if graphe_type not in mapping:
        available = list(mapping.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Type de graphique inconnu: '{graphe_type}'. Types disponibles: {available}"
        )
    
    file_path = mapping[graphe_type]
    
    if file_path is None or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Aucun fichier trouvé pour le type: '{graphe_type}'. Vérifiez que le fichier existe dans {GRAPHE_DIR}"
        )
    
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    print(f"✅ Téléchargement: {filename} ({round(file_size / 1024, 1)} Ko)")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/png"
    )


@router.get("/graphe/list")
def list_training_graphs():
    """
    Liste tous les graphiques de training disponibles avec leurs URLs de téléchargement
    """
    mapping = get_graphe_file_mapping()
    
    available_graphs = {}
    missing_graphs = []
    
    for graphe_type, file_path in mapping.items():
        if file_path and os.path.exists(file_path):
            available_graphs[graphe_type] = {
                "filename": os.path.basename(file_path),
                "download_url": f"/api/download/graphe/{graphe_type}",
                "size_kb": round(os.path.getsize(file_path) / 1024, 1),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
        else:
            missing_graphs.append(graphe_type)
    
    return {
        "status": "success",
        "graphe_dir": GRAPHE_DIR,
        "graphe_dir_exists": os.path.exists(GRAPHE_DIR),
        "available": available_graphs,
        "missing": missing_graphs,
        "total_available": len(available_graphs),
        "total_missing": len(missing_graphs),
        "all_download_urls": {
            graphe_type: f"/api/download/graphe/{graphe_type}"
            for graphe_type in mapping.keys()
        }
    }


# ============================================
# PREDICTION
# ============================================

@router.post("/save-parameters")
async def save_parameters(request: Request):
    """Sauvegarde les paramètres de prédiction dans un fichier JSON"""
    print("💾 Appel /save-parameters")
    try:
        params = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur JSON: {e}")
    
    filename = f"parametres_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(PARAMS_DIR, filename)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"✅ Paramètres sauvegardés: {filename}")
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


@router.get("/predict/from-file")
def predict_from_file_endpoint():
    """Exécute la prédiction à partir du fichier JSON sauvegardé"""
    return execute_predict_from_file()


@router.post("/save-and-predict-json")
async def save_and_predict_json(request: Request):
    """
    Sauvegarde les paramètres, nettoie les anciennes données, et exécute la prédiction
    """
    print("🚀 Appel /save-and-predict-json")
    saved = await save_parameters(request)
    cleanup_images = run_cleanup_all_images()
    prediction = execute_predict_from_file()
    cleanup_json = run_cleanup_json()
    return {
        "success": True,
        "status": "success",
        "message": "JSON sauvegardé, images supprimées et prédiction exécutée",
        "saved_file": saved,
        "prediction": prediction,
        "cleanup_images": cleanup_images,
        "cleanup_json": cleanup_json,
    }


# ============================================
# ENREGISTREMENT DES ROUTES
# ============================================

app.include_router(router, prefix="/api")

print("\n" + "="*60)
print("✅ API Vala Bleu initialisée avec succès")
print("="*60)
print(f"📍 Routes disponibles:")

# Afficher toutes les routes enregistrées
for route in app.routes:
    if hasattr(route, 'path') and hasattr(route, 'methods'):
        methods = ', '.join(route.methods) if route.methods else 'N/A'
        print(f"  {methods:10} {route.path}")

# Vérifier les graphiques disponibles au démarrage
print(f"\n📊 Vérification des graphiques de training:")
graphe_mapping = get_graphe_file_mapping()
for graphe_type, file_path in graphe_mapping.items():
    if file_path:
        print(f"  ✅ {graphe_type}: {os.path.basename(file_path)}")
    else:
        print(f"  ❌ {graphe_type}: Non trouvé")

print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
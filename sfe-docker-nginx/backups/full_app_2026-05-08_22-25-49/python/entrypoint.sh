#!/bin/bash

echo "========================================="
echo "  🧹 Nettoyage des fichiers..."
echo "========================================="
python /app/script/cleanup.py

echo ""
echo "========================================="
echo "  🚀 Démarrage de l'API..."
echo "========================================="
exec uvicorn api:app --host 0.0.0.0 --port 8000
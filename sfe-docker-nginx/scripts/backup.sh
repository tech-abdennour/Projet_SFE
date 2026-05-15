#!/bin/bash

# --- CONFIGURATION DES CHEMINS ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$PROJECT_DIR/backups"
APP_DIR="$PROJECT_DIR/app"

# --- CONFIGURATION BASE DE DONNÉES ---
# Identifiants extraits de ton docker-compose.yml
DB_CONTAINER="mysql" 
DB_USER="root"
DB_PASS="root"           # MODIFIÉ : correspond à MYSQL_ROOT_PASSWORD
DB_NAME="appdb"          # MODIFIÉ : correspond à MYSQL_DATABASE
DATE=$(date +%F_%H-%M-%S)

# Créer le dossier de backup s'il n'existe pas
mkdir -p "$BACKUP_DIR"

echo "=========================================================="
echo "      DÉMARRAGE DU BACKUP COMPLET : $DATE"
echo "=========================================================="

# 1. Sauvegarde de la Base de Données MariaDB (Docker)
echo "[INFO] Extraction de la base de données via Docker..."

# Utilisation de MYSQL_PWD et de l'hôte 127.0.0.1
docker exec -e MYSQL_PWD="$DB_PASS" "$DB_CONTAINER" mariadb-dump -u "$DB_USER" --host=127.0.0.1 "$DB_NAME" > "$BACKUP_DIR/db_mysql_$DATE.sql" 2> "$BACKUP_DIR/mysql_error.log"

if [ $? -eq 0 ] && [ -s "$BACKUP_DIR/db_mysql_$DATE.sql" ]; then
    echo "[OK] Base de données extraite avec succès."
    gzip -f "$BACKUP_DIR/db_mysql_$DATE.sql"
    echo "[OK] Fichier compressé : db_mysql_$DATE.sql.gz"
    rm -f "$BACKUP_DIR/mysql_error.log"
else
    echo "[ERROR] Échec de la sauvegarde MySQL/MariaDB."
    echo "Détail technique de l'erreur :"
    cat "$BACKUP_DIR/mysql_error.log"
    rm -f "$BACKUP_DIR/db_mysql_$DATE.sql"
fi

echo "----------------------------------------------------------"

# 2. Sauvegarde de la Base de Données SQLite (vala_bleu.db)
echo "[INFO] Copie de la base SQLite vala_bleu.db..."

SQLITE_SOURCE="$APP_DIR/vala_bleu.db"

if [ -f "$SQLITE_SOURCE" ]; then
    cp "$SQLITE_SOURCE" "$BACKUP_DIR/vala_bleu_$DATE.db"
    gzip -f "$BACKUP_DIR/vala_bleu_$DATE.db"
    echo "[OK] SQLite sauvegardé : vala_bleu_$DATE.db.gz"
else
    echo "[WARNING] Fichier vala_bleu.db introuvable dans $APP_DIR"
fi

echo "----------------------------------------------------------"

# 3. Sauvegarde des Fichiers (Code source, images, etc.)
echo "[INFO] Création de l'archive complète du projet..."
tar -czf "$BACKUP_DIR/full_app_$DATE.tar.gz" -C "$PROJECT_DIR" \
    --exclude="backups" \
    --exclude="logs" \
    --exclude=".git" \
    .

if [ $? -eq 0 ]; then
    echo "[OK] Archive globale créée : full_app_$DATE.tar.gz"
else
    echo "[ERROR] Échec de l'archivage des fichiers."
fi

echo "=========================================================="
echo " [DONE] Sauvegarde terminée dans : $BACKUP_DIR"
echo "=========================================================="
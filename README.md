📦 Installation et exécution du projet  
1. Installer Docker  
Assurez-vous que Docker est installé sur votre machine.  
2. Cloner le projet  
  -git clone https://github.com/tech-abdennour/SFE.git  
  -cd SFE/sfe-docker-nginx  
3. Lancer les services  
  docker compose up -d  
4. Vérifier les conteneurs  
  docker ps  
5. Accès aux services  
  -Frontend : http://localhost  
  -Backend :  http://localhost:8000/docs
6. Scripts disponibles Dans le dossier SFE/sfe-docker-nginx/scripts, vous pouvez exécuter avec Git Bash sur Windows ou le terminal Linux par défaut :  
+./backup.sh  
+./healthcheck.sh  
    - backup.sh : permet de faire des sauvegardes  
    - healthcheck.sh : vérifie l’état des services  
    
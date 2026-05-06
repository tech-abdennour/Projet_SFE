# Architecture de l’infrastructure

L’architecture adoptée repose sur une séparation claire des services :

- Nginx : serveur web et reverse proxy
- PHP-FPM : exécution du code applicatif
- MariaDB : stockage des données

Chaque service est isolé dans un conteneur Docker, communiquant via
un réseau privé interne.
# 📘 MySQL Cheat Sheet

## 🧱 1. Commandes de base

```sql
SHOW DATABASES;
USE nom_base;
SELECT DATABASE();
SHOW TABLES;
SHOW COLUMNS FROM nom_table;
DESCRIBE nom_table;
🗄️ 2. Gestion des bases de données
    CREATE DATABASE ma_base;
    DROP DATABASE ma_base;
    ALTER DATABASE ma_base CHARACTER SET utf8mb4;
📋 3. Gestion des tables
    ➤ Créer une table
        CREATE TABLE users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100)
        );
    ➤ Modifier une table
        ALTER TABLE users ADD age INT;
        ALTER TABLE users MODIFY name VARCHAR(200);
        ALTER TABLE users DROP COLUMN age;
        RENAME TABLE users TO clients;
    ➤ Supprimer une table
        DROP TABLE users;
        TRUNCATE TABLE users;
✍️ 4. Manipulation des données (CRUD)
    ➤ Ajouter
        INSERT INTO users (name, email) VALUES ('Ali', 'ali@mail.com');
    ➤ Lire
        SELECT * FROM users;
        SELECT name FROM users;
        SELECT * FROM users WHERE id = 1;
    ➤ Modifier
        UPDATE users SET name = 'Ahmed' WHERE id = 1;
    ➤ Supprimer
        DELETE FROM users WHERE id = 1;
🔍 5. Requêtes avancées
    ➤ Filtres
        SELECT * FROM users WHERE age > 18;
        SELECT * FROM users WHERE name LIKE '%a%';
        SELECT * FROM users WHERE id IN (1,2,3);
    ➤ Tri
        SELECT * FROM users ORDER BY name ASC;
    ➤ Limite
        SELECT * FROM users LIMIT 10;
🔗 6. Jointures (JOIN)
    SELECT * 
    FROM orders
    INNER JOIN users ON users.id = orders.user_id;

    SELECT *
    FROM users
    LEFT JOIN orders ON users.id = orders.user_id;
📊 7. Fonctions d’agrégation
    SELECT COUNT(*) FROM users;
    SELECT AVG(age) FROM users;
    SELECT MAX(age) FROM users;
    SELECT MIN(age) FROM users;

    Avec groupement :

    SELECT age, COUNT(*) 
    FROM users 
    GROUP BY age;
🔐 8. Gestion des utilisateurs
    CREATE USER 'user'@'%' IDENTIFIED BY 'password';
    GRANT ALL PRIVILEGES ON *.* TO 'user'@'%';
    FLUSH PRIVILEGES;

    DROP USER 'user'@'%';
💾 9. Sauvegarde & import
    Export :
    mysqldump -u root -p ma_base > backup.sql
    Import :
    mysql -u root -p ma_base < backup.sql
⚙️ 10. Commandes utiles MySQL
    EXIT;
    QUIT;
    STATUS;
    SOURCE fichier.sql;
    🐳 Docker (bonus)
    docker exec -it mysql mysql -u root -p

    ---

    Si tu veux, je peux aussi :
    - te générer un **PDF propre**
    - ajouter **index, transactions, triggers, procédures stockées**
    - ou transformer ça en **fiche ultra avancée niveau entretien**
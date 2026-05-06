<?php

$host = "mysql";
$db   = "appdb";
$user = "appuser";
$pass = "apppass";

$maxTries = 10;

while ($maxTries > 0) {
    try {
        $pdo = new PDO(
            "mysql:host=$host;dbname=$db;charset=utf8",
            $user,
            $pass,
            [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]
        );
        break;
    } catch (PDOException $e) {
        $pdo = null;
        $maxTries--;
        sleep(2); // attend MySQL
    }
}
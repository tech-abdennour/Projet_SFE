<?php
session_start();

// Supprimer session
$_SESSION = [];
session_destroy();

// Supprimer cookie
if (isset($_COOKIE['remember_user'])) {
    setcookie("remember_user", "", time() - 3600, "/");
}

// Rediriger avec paramètre logout
header("Location: index.php?logout=1");
exit;
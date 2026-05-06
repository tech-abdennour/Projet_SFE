<?php
session_start();

// Vérifier si l'utilisateur vient de se déconnecter (paramètre dans l'URL)
$just_logged_out = isset($_GET['logout']);

// Si déjà connecté avec session → rediriger vers dashboard
if (isset($_SESSION['user'])) {
    header("Location: dashboard.php");
    exit;
}

// Vérifier le cookie "Se souvenir de moi" UNIQUEMENT si pas de déconnexion récente
if (!$just_logged_out && !isset($_SESSION['user']) && isset($_COOKIE['remember_user'])) {
    $_SESSION['user'] = $_COOKIE['remember_user'];
    header("Location: dashboard.php");
    exit;
}

$error = "";

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $username = trim($_POST["username"]);
    $password = trim($_POST["password"]);
    $remember = isset($_POST["remember"]);

    if ($username === "admin" && $password === "azerty123" or $username === "Ahmed" && $password === "kiritiri") {
        $_SESSION["user"] = $username;

        if ($remember) {
            // Cookie valable 7 jours
            setcookie("remember_user", $username, time() + (7 * 24 * 3600), "/");
        }

        header("Location: dashboard.php");
        exit;
    } else {
        $error = "Nom d'utilisateur ou mot de passe incorrect";
    }
}
?>


<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <link rel="icon" type="image/png" href="logos.png">
    <title>Connexion - SFE Project</title>
    <script src="https://unpkg.com/feather-icons"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: #f0f2f5;
        }

        .login-box {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
            width: 320px;
            text-align: center;
        }

        h2 { color: #333; margin-bottom: 20px; }

        .input-group {
            position: relative;
            margin-bottom: 15px;
            text-align: left;
        }

        .field-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: #888;
            width: 18px;
            height: 18px;
        }

        input {
            width: 100%;
            padding: 12px 12px 12px 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-sizing: border-box;
            font-size: 14px;
            transition: border 0.3s;
        }

        input:focus {
            border-color: #007bff;
            outline: none;
        }

        .password-toggle {
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            cursor: pointer;
            color: #888;
            background: none;
            border: none;
            padding: 0;
            display: flex;
            align-items: center;
        }

        .password-toggle:hover { color: #333; }

        button {
            width: 100%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
            transition: background 0.3s;
        }

        button:hover { background: #0056b3; }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 10px;
            border-radius: 6px;
            font-size: 0.85em;
            margin-bottom: 15px;
        }

        .options {
            margin-bottom: 5px;
        }
    </style>

</head>
<body>

<div class="login-box">
    <h2>Connexion</h2>

    <?php if ($error): ?>
        <div class="error"><?php echo $error; ?></div>
    <?php endif; ?>

    <?php if ($just_logged_out): ?>
        <div style="background: #e8f5e9; color: #2e7d32; padding: 10px; border-radius: 6px; font-size: 0.85em; margin-bottom: 15px;">
            ✅ Vous avez été déconnecté avec succès.
        </div>
    <?php endif; ?>

    <form method="POST">
        <div class="input-group">
            <i data-feather="user" class="field-icon"></i>
            <input type="text" name="username" placeholder="Nom d'utilisateur" required>
        </div>

        <div class="input-group">
            <i data-feather="lock" class="field-icon"></i>
            <input type="password" id="password" name="password" placeholder="Mot de passe" required style="padding-right: 40px;">
            
            <span class="password-toggle" onclick="togglePassword()">
                <i data-feather="eye" id="eyeIcon" style="width: 18px;"></i>
            </span>
        </div>
        
        <div class="options">
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 20px;">
                <input type="checkbox" name="remember" id="remember" style="cursor: pointer; width: 18px; height: 18px;">
                
                <span style="font-size: 14px; color: #4a5568; user-select: none;">
                    Rester connecté
                </span>
            </div>
        </div>
        
        <button type="submit">Se connecter</button>
    </form>
</div>

<script>
    feather.replace();

    function togglePassword() {
        const input = document.getElementById("password");
        const icon = document.getElementById("eyeIcon");

        if (input.type === "password") {
            input.type = "text";
            icon.setAttribute("data-feather", "eye-off");
        } else {
            input.type = "password";
            icon.setAttribute("data-feather", "eye");
        }

        feather.replace();
    }
</script>

</body>
</html>
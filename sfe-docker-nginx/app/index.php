<?php

session_start();
require_once "db_config.php";

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

    // Vérification utilisateur/password depuis la base
    $stmt = $pdo->prepare("SELECT * FROM users WHERE username = ?");
    $stmt->execute([$username]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    // Pour production, utilisez password_hash/verify
    if ($user && $user['password'] === $password) {
        $_SESSION["user"] = $username;

        if ($remember) {
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

        .icon-svg {
            width: 18px;
            height: 18px;
            stroke: currentColor;
            fill: none;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-linejoin: round;
        }

        .hidden {
            display: none;
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
            <span class="field-icon" aria-hidden="true">
                <svg class="icon-svg" viewBox="0 0 24 24">
                    <path d="M20 21a8 8 0 0 0-16 0"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            </span>
            <input type="text" name="username" placeholder="Nom d'utilisateur" required>
        </div>

        <div class="input-group">
            <span class="field-icon" aria-hidden="true">
                <svg class="icon-svg" viewBox="0 0 24 24">
                    <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
            </span>
            <input type="password" id="password" name="password" placeholder="Mot de passe" required style="padding-right: 40px;">
            
            <span class="password-toggle" onclick="togglePassword()">
                <svg class="icon-svg" id="eyeOpen" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"></path>
                    <circle cx="12" cy="12" r="3"></circle>
                </svg>
                <svg class="icon-svg hidden" id="eyeClosed" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M17.94 17.94A10.9 10.9 0 0 1 12 19c-7 0-11-7-11-7a21.77 21.77 0 0 1 5.06-5.94"></path>
                    <path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 8 11 8a21.91 21.91 0 0 1-2.16 3.19"></path>
                    <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"></path>
                    <line x1="1" y1="1" x2="23" y2="23"></line>
                </svg>
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
    function togglePassword() {
        const input = document.getElementById("password");
        const eyeOpen = document.getElementById("eyeOpen");
        const eyeClosed = document.getElementById("eyeClosed");

        if (input.type === "password") {
            input.type = "text";
            eyeOpen.classList.add("hidden");
            eyeClosed.classList.remove("hidden");
        } else {
            input.type = "password";
            eyeOpen.classList.remove("hidden");
            eyeClosed.classList.add("hidden");
        }
    }
</script>

</body>
</html>
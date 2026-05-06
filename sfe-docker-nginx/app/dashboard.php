<?php
set_time_limit(300);
session_start();
date_default_timezone_set('Africa/Casablanca');

// Vérification de la session utilisateur
if (!isset($_SESSION['user'])) {
    header("Location: index.php");
    exit;
}

// --- BASE DE DONNÉES SQLITE ---
$db_file = 'vala_bleu.db';
$pdo = null;

try {
    $pdo = new PDO("sqlite:" . $db_file);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (Exception $e) {
    error_log("SQLite error: " . $e->getMessage());
}

// --- FONCTIONS DE PERSISTANCE ---
function getPredictions($pdo) {
    if ($pdo === null) return [];
    if (!isset($_SESSION['user'])) return [];
    try {
        $stmt = $pdo->prepare("SELECT * FROM predictions WHERE is_deleted = 0 AND user = :user ORDER BY created_at DESC");
        $stmt->execute([':user' => $_SESSION['user']]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (Exception $e) {
        error_log('Erreur getPredictions: ' . $e->getMessage());
        return [];
    }
}

function getDeletedSauvegardes($pdo) {
    if ($pdo === null) return [];
    if (!isset($_SESSION['user'])) return [];
    try {
        $stmt = $pdo->prepare("SELECT * FROM deleted_sauvegardes WHERE user = :user ORDER BY deleted_at DESC");
        $stmt->execute([':user' => $_SESSION['user']]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (Exception $e) {
        error_log('Erreur getDeletedSauvegardes: ' . $e->getMessage());
        return [];
    }
}

function getSavedResults($pdo) {
    if ($pdo === null) return [];
    try {
        $stmt = $pdo->query("SELECT * FROM saved_results ORDER BY created_at DESC");
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (Exception $e) {
        error_log('Erreur getSavedResults: ' . $e->getMessage());
        return [];
    }
}

function savePrediction($pdo, $data, &$errorMsg = null) {
    if ($pdo === null) { $errorMsg = 'Connexion à la base de données impossible.'; return false; }
    if (!isset($_SESSION['user'])) { $errorMsg = 'Utilisateur non connecté.'; return false; }
    try {
        $stmt = $pdo->prepare("INSERT INTO predictions (
            id, user, created_at, cpu_usage_avg, cpu_usage_peak, ram_usage_avg, ram_usage_max, disk_usage_avg, disk_usage_max, disk_read_iops, disk_write_iops, response_time,
            visitors_per_day, pageviews_per_day, traffic_growth_rate, peak_hours_start, peak_hours_end, peak_hours,
            plugin_count, heavy_plugins, php_version, cache_enabled, cdn_enabled,
            wp_type, predicted_load, predicted_saturation_months, xgboost_score,
            status, recommendation, save_type, is_deleted
        ) VALUES (
            :id, :user, :created_at, :cpu_usage_avg, :cpu_usage_peak, :ram_usage_avg, :ram_usage_max, :disk_usage_avg, :disk_usage_max, :disk_read_iops, :disk_write_iops, :response_time,
            :visitors_per_day, :pageviews_per_day, :traffic_growth_rate, :peak_hours_start, :peak_hours_end, :peak_hours,
            :plugin_count, :heavy_plugins, :php_version, :cache_enabled, :cdn_enabled,
            :wp_type, :predicted_load, :predicted_saturation_months, :xgboost_score,
            :status, :recommendation, :save_type, 0
        )");
        $stmt->execute([
            ':id' => $data['id'],
            ':user' => $_SESSION['user'],
            ':created_at' => $data['created_at'],
            ':cpu_usage_avg' => $data['cpu_usage_avg'] ?? '',
            ':cpu_usage_peak' => $data['cpu_usage_peak'] ?? '',
            ':ram_usage_avg' => $data['ram_usage_avg'] ?? '',
            ':ram_usage_max' => $data['ram_usage_max'] ?? '',
            ':disk_usage_avg' => $data['disk_usage_avg'] ?? '',
            ':disk_usage_max' => $data['disk_usage_max'] ?? '',
            ':disk_read_iops' => $data['disk_read_iops'] ?? '',
            ':disk_write_iops' => $data['disk_write_iops'] ?? '',
            ':response_time' => $data['response_time'] ?? '',
            ':visitors_per_day' => $data['visitors_per_day'] ?? '',
            ':pageviews_per_day' => $data['pageviews_per_day'] ?? '',
            ':traffic_growth_rate' => $data['traffic_growth_rate'] ?? '',
            ':peak_hours_start' => $data['peak_hours_start'] ?? '',
            ':peak_hours_end' => $data['peak_hours_end'] ?? '',
            ':peak_hours' => $data['peak_hours'] ?? '',
            ':plugin_count' => $data['plugin_count'] ?? '',
            ':heavy_plugins' => $data['heavy_plugins'] ?? '',
            ':php_version' => $data['php_version'] ?? '',
            ':cache_enabled' => $data['cache_enabled'] ?? '',
            ':cdn_enabled' => $data['cdn_enabled'] ?? '',
            ':wp_type' => $data['wp_type'] ?? '',
            ':predicted_load' => $data['predicted_load'] ?? '',
            ':predicted_saturation_months' => $data['predicted_saturation_months'] ?? '',
            ':xgboost_score' => $data['xgboost_score'] ?? '',
            ':status' => $data['status'] ?? '',
            ':recommendation' => $data['recommendation'] ?? '',
            ':save_type' => $data['save_type'] ?? 'Manuel'
        ]);
        return true;
    } catch (Exception $e) {
        $errorMsg = $e->getMessage();
        error_log('Erreur savePrediction: ' . $e->getMessage());
        return false;
    }
}

function saveResultJson($pdo, $data) {
    if ($pdo === null) return false;
    try {
        $stmt = $pdo->prepare("INSERT INTO saved_results (id, created_at, data_json, save_type) VALUES (:id, :created_at, :data_json, :save_type)");
        $stmt->execute([
            ':id' => uniqid(),
            ':created_at' => date('Y-m-d H:i:s'),
            ':data_json' => json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE),
            ':save_type' => 'Sauvegarde JSON'
        ]);
        return true;
    } catch (Exception $e) {
        return false;
    }
}

function archivePrediction($pdo, $id) {
    if ($pdo === null) return false;
    try {
        $stmt = $pdo->prepare("SELECT * FROM predictions WHERE id = :id AND is_deleted = 0");
        $stmt->execute([':id' => $id]);
        $pred = $stmt->fetch(PDO::FETCH_ASSOC);
        
        if ($pred) {
            $stmt2 = $pdo->prepare("INSERT INTO deleted_sauvegardes (
                id, created_at, cpu_usage_avg, cpu_usage_peak, ram_usage_avg, ram_usage_max, disk_usage_avg, disk_usage_max, disk_read_iops, disk_write_iops, response_time,
                visitors_per_day, pageviews_per_day, traffic_growth_rate, peak_hours_start, peak_hours_end, peak_hours,
                plugin_count, heavy_plugins, php_version, cache_enabled, cdn_enabled,
                wp_type, predicted_load, predicted_saturation_months, xgboost_score,
                status, recommendation, save_type, deleted_at, user
            ) VALUES (
                :id, :created_at, :cpu_usage_avg, :cpu_usage_peak, :ram_usage_avg, :ram_usage_max, :disk_usage_avg, :disk_usage_max, :disk_read_iops, :disk_write_iops, :response_time,
                :visitors_per_day, :pageviews_per_day, :traffic_growth_rate, :peak_hours_start, :peak_hours_end, :peak_hours,
                :plugin_count, :heavy_plugins, :php_version, :cache_enabled, :cdn_enabled,
                :wp_type, :predicted_load, :predicted_saturation_months, :xgboost_score,
                :status, :recommendation, :save_type, :deleted_at, :user
            )");

            $stmt2->execute([
                ':id' => $pred['id'],
                ':created_at' => $pred['created_at'],
                ':cpu_usage_avg' => $pred['cpu_usage_avg'],
                ':cpu_usage_peak' => $pred['cpu_usage_peak'],
                ':ram_usage_avg' => $pred['ram_usage_avg'],
                ':ram_usage_max' => $pred['ram_usage_max'],
                ':disk_usage_avg' => $pred['disk_usage_avg'],
                ':disk_usage_max' => $pred['disk_usage_max'],
                ':disk_read_iops' => $pred['disk_read_iops'],
                ':disk_write_iops' => $pred['disk_write_iops'],
                ':response_time' => $pred['response_time'],
                ':visitors_per_day' => $pred['visitors_per_day'],
                ':pageviews_per_day' => $pred['pageviews_per_day'],
                ':traffic_growth_rate' => $pred['traffic_growth_rate'],
                ':peak_hours_start' => $pred['peak_hours_start'],
                ':peak_hours_end' => $pred['peak_hours_end'],
                ':peak_hours' => $pred['peak_hours'],
                ':plugin_count' => $pred['plugin_count'],
                ':heavy_plugins' => $pred['heavy_plugins'],
                ':php_version' => $pred['php_version'],
                ':cache_enabled' => $pred['cache_enabled'],
                ':cdn_enabled' => $pred['cdn_enabled'],
                ':wp_type' => $pred['wp_type'],
                ':predicted_load' => $pred['predicted_load'],
                ':predicted_saturation_months' => $pred['predicted_saturation_months'],
                ':xgboost_score' => $pred['xgboost_score'],
                ':status' => $pred['status'],
                ':recommendation' => $pred['recommendation'],
                ':save_type' => $pred['save_type'] ?? 'Manuel',
                ':deleted_at' => date('Y-m-d H:i:s'),
                ':user' => $pred['user'] ?? ($_SESSION['user'] ?? null)
            ]);
            
            $stmt3 = $pdo->prepare("UPDATE predictions SET is_deleted = 1 WHERE id = :id");
            $stmt3->execute([':id' => $id]);
            return true;
        }
        return false;
    } catch (Exception $e) {
        return false;
    }
}

function deletePermanently($pdo, $id) {
    if ($pdo === null) return false;
    try {
        $stmt = $pdo->prepare("DELETE FROM deleted_sauvegardes WHERE id = :id");
        $stmt->execute([':id' => $id]);
        return true;
    } catch (Exception $e) {
        return false;
    }
}

function restorePrediction($pdo, $id) {
    if ($pdo === null) return false;
    try {
        $stmt = $pdo->prepare("SELECT * FROM deleted_sauvegardes WHERE id = :id");
        $stmt->execute([':id' => $id]);
        $pred = $stmt->fetch(PDO::FETCH_ASSOC);
        if ($pred) {
            $new_id = uniqid();
            $stmt2 = $pdo->prepare("INSERT INTO predictions (
                id, user, created_at, cpu_usage_avg, cpu_usage_peak, ram_usage_avg, ram_usage_max, disk_usage_avg, disk_usage_max, disk_read_iops, disk_write_iops, response_time,
                visitors_per_day, pageviews_per_day, traffic_growth_rate, peak_hours_start, peak_hours_end, peak_hours,
                plugin_count, heavy_plugins, php_version, cache_enabled, cdn_enabled,
                wp_type, predicted_load, predicted_saturation_months, xgboost_score,
                status, recommendation, save_type, is_deleted
            ) VALUES (
                :id, :user, :created_at, :cpu_usage_avg, :cpu_usage_peak, :ram_usage_avg, :ram_usage_max, :disk_usage_avg, :disk_usage_max, :disk_read_iops, :disk_write_iops, :response_time,
                :visitors_per_day, :pageviews_per_day, :traffic_growth_rate, :peak_hours_start, :peak_hours_end, :peak_hours,
                :plugin_count, :heavy_plugins, :php_version, :cache_enabled, :cdn_enabled,
                :wp_type, :predicted_load, :predicted_saturation_months, :xgboost_score,
                :status, :recommendation, :save_type, 0
            )");
            $stmt2->execute([
                ':id' => $new_id,
                ':user' => $_SESSION['user'],
                ':created_at' => $pred['created_at'],
                ':cpu_usage_avg' => $pred['cpu_usage_avg'],
                ':cpu_usage_peak' => $pred['cpu_usage_peak'],
                ':ram_usage_avg' => $pred['ram_usage_avg'],
                ':ram_usage_max' => $pred['ram_usage_max'],
                ':disk_usage_avg' => $pred['disk_usage_avg'],
                ':disk_usage_max' => $pred['disk_usage_max'],
                ':disk_read_iops' => $pred['disk_read_iops'],
                ':disk_write_iops' => $pred['disk_write_iops'],
                ':response_time' => $pred['response_time'],
                ':visitors_per_day' => $pred['visitors_per_day'],
                ':pageviews_per_day' => $pred['pageviews_per_day'],
                ':traffic_growth_rate' => $pred['traffic_growth_rate'],
                ':peak_hours_start' => $pred['peak_hours_start'],
                ':peak_hours_end' => $pred['peak_hours_end'],
                ':peak_hours' => $pred['peak_hours'],
                ':plugin_count' => $pred['plugin_count'],
                ':heavy_plugins' => $pred['heavy_plugins'],
                ':php_version' => $pred['php_version'],
                ':cache_enabled' => $pred['cache_enabled'],
                ':cdn_enabled' => $pred['cdn_enabled'],
                ':wp_type' => $pred['wp_type'],
                ':predicted_load' => $pred['predicted_load'],
                ':predicted_saturation_months' => $pred['predicted_saturation_months'],
                ':xgboost_score' => $pred['xgboost_score'],
                ':status' => $pred['status'],
                ':recommendation' => $pred['recommendation'],
                ':save_type' => $pred['save_type'] ?? 'Manuel'
            ]);
            $stmt3 = $pdo->prepare("DELETE FROM deleted_sauvegardes WHERE id = :id");
            $stmt3->execute([':id' => $id]);
            return true;
        }
        return false;
    } catch (Exception $e) {
        error_log('Erreur restorePrediction: ' . $e->getMessage());
        return false;
    }
}

function emptyTrash($pdo) {
    if ($pdo === null) return false;
    try {
        $stmt = $pdo->prepare("DELETE FROM deleted_sauvegardes");
        $stmt->execute();
        return true;
    } catch (Exception $e) {
        return false;
    }
}

// =============================================
// EXPORT CSV COMPLET
// =============================================
if (isset($_GET['export_full_csv']) && isset($_SESSION['user'])) {
    $predictions = getPredictions($pdo);
    if (ob_get_level()) ob_end_clean();
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="vala_bleu_export_' . date('Y-m-d_His') . '.csv"');
    echo "\xEF\xBB\xBF";
    $output = fopen('php://output', 'w');
    fputcsv($output, ['Date','Heure','Pack','Visiteurs/j','Pages vues/j','Croissance (%)','CPU moyen (%)','CPU max (%)','RAM moyenne (%)','RAM max (%)','Disque moyen (%)','Disque max (%)','IOPS Read','IOPS Write','Temps réponse (ms)','Pic début','Pic fin','Nb plugins','Plugins lourds','PHP','Cache','CDN','Score XGBoost (%)','Charge prédite (%)','Saturation (mois)','Statut','Recommandation','Type','ID'], ';');
    foreach ($predictions as $p) {
        fputcsv($output, [
            date('d/m/Y', strtotime($p['created_at'] ?? '')), date('H:i:s', strtotime($p['created_at'] ?? '')),
            strtoupper($p['wp_type'] ?? ''), $p['visitors_per_day'] ?? '', $p['pageviews_per_day'] ?? '',
            $p['traffic_growth_rate'] ?? '', $p['cpu_usage_avg'] ?? '', $p['cpu_usage_peak'] ?? '',
            $p['ram_usage_avg'] ?? '', $p['ram_usage_max'] ?? '', $p['disk_usage_avg'] ?? '', $p['disk_usage_max'] ?? '',
            $p['disk_read_iops'] ?? '', $p['disk_write_iops'] ?? '', $p['response_time'] ?? '',
            $p['peak_hours_start'] ?? '', $p['peak_hours_end'] ?? '', $p['plugin_count'] ?? '',
            $p['heavy_plugins'] ?? '', $p['php_version'] ?? '', $p['cache_enabled'] ?? '', $p['cdn_enabled'] ?? '',
            $p['xgboost_score'] ?? '', $p['predicted_load'] ?? '', $p['predicted_saturation_months'] ?? '',
            $p['status'] ?? '', '"' . str_replace('"', '""', $p['recommendation'] ?? '') . '"',
            $p['save_type'] ?? 'Manuel', $p['id'] ?? ''
        ], ';');
    }
    fclose($output);
    exit();
}

// =============================================
// TÉLÉCHARGER UNE SAUVEGARDE JSON
// =============================================
if (isset($_GET['download_json']) && isset($_SESSION['user'])) {
    $stmt = $pdo->prepare("SELECT * FROM saved_results WHERE id = :id");
    $stmt->execute([':id' => $_GET['download_json']]);
    $result = $stmt->fetch(PDO::FETCH_ASSOC);
    if ($result) {
        if (ob_get_level()) ob_end_clean();
        header('Content-Type: application/json; charset=utf-8');
        header('Content-Disposition: attachment; filename="resultat_' . date('Y-m-d_His', strtotime($result['created_at'])) . '.json"');
        echo $result['data_json'];
        exit();
    }
}

// --- REQUÊTES AJAX ---
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_SERVER['HTTP_X_REQUESTED_WITH']) && $_SERVER['HTTP_X_REQUESTED_WITH'] === 'XMLHttpRequest') {
    if (ob_get_level()) ob_end_clean();
    header('Content-Type: application/json; charset=utf-8');
    $input = file_get_contents('php://input');
    $data = json_decode($input, true);

    function ajax_json_response($arr) {
        echo json_encode($arr);
        exit();
    }

    if (!$data || !isset($_SESSION['user'])) {
        error_log('AJAX: Données invalides ou utilisateur non connecté');
        ajax_json_response(['success' => false, 'error' => 'Données invalides ou utilisateur non connecté']);
    }

    try {
        if (isset($data['action'])) {
            if ($data['action'] === 'delete' && isset($data['delete_id'])) {
                ajax_json_response(['success' => deletePermanently($pdo, $data['delete_id'])]);
            }
            if ($data['action'] === 'archive' && isset($data['archive_id'])) {
                ajax_json_response(['success' => archivePrediction($pdo, $data['archive_id'])]);
            }
            if ($data['action'] === 'restore' && isset($data['restore_id'])) {
                ajax_json_response(['success' => restorePrediction($pdo, $data['restore_id'])]);
            }
            if ($data['action'] === 'empty_trash') {
                ajax_json_response(['success' => emptyTrash($pdo)]);
            }
            if ($data['action'] === 'save_json' && isset($data['result_data'])) {
                ajax_json_response(['success' => saveResultJson($pdo, $data['result_data'])]);
            }
            if ($data['action'] === 'generate_json' && isset($data['json_data'])) {
                $jsonFolder = realpath(__DIR__ . '/../python/Donnee_parametres');
                if ($jsonFolder === false || !is_dir($jsonFolder)) {
                    ajax_json_response(['success' => false, 'error' => 'Le dossier Donnee_parametres est introuvable.']);
                }
                $filename = $jsonFolder . '/prediction_' . date('Y-m-d_H-i-s') . '.json';
                $ok = file_put_contents($filename, json_encode($data['json_data'], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
                if ($ok !== false) {
                    ajax_json_response(['success' => true, 'filename' => basename($filename)]);
                } else {
                    ajax_json_response(['success' => false, 'error' => 'Erreur lors de la génération du JSON']);
                }
            }
        }
        if (isset($data['predicted_load'])) {
            $prediction = [
                'id' => uniqid(), 'created_at' => date('Y-m-d H:i:s'),
                'cpu_usage_avg' => $data['cpu_usage_avg'] ?? '', 'cpu_usage_peak' => $data['cpu_usage_peak'] ?? '',
                'ram_usage_avg' => $data['ram_usage_avg'] ?? '', 'ram_usage_max' => $data['ram_usage_max'] ?? '',
                'disk_usage_avg' => $data['disk_usage_avg'] ?? '', 'disk_usage_max' => $data['disk_usage_max'] ?? '',
                'disk_read_iops' => $data['disk_read_iops'] ?? '', 'disk_write_iops' => $data['disk_write_iops'] ?? '',
                'response_time' => $data['response_time'] ?? '', 'visitors_per_day' => $data['visitors_per_day'] ?? '',
                'pageviews_per_day' => $data['pageviews_per_day'] ?? '', 'traffic_growth_rate' => $data['traffic_growth_rate'] ?? '',
                'peak_hours_start' => $data['peak_hours_start'] ?? '', 'peak_hours_end' => $data['peak_hours_end'] ?? '',
                'peak_hours' => $data['peak_hours'] ?? '', 'plugin_count' => $data['plugin_count'] ?? '',
                'heavy_plugins' => $data['heavy_plugins'] ?? '', 'php_version' => $data['php_version'] ?? '',
                'cache_enabled' => $data['cache_enabled'] ?? '', 'cdn_enabled' => $data['cdn_enabled'] ?? '',
                'wp_type' => $data['wp_type'] ?? '', 'predicted_load' => $data['predicted_load'] ?? '',
                'predicted_saturation_months' => $data['predicted_saturation_months'] ?? '',
                'xgboost_score' => $data['xgboost_score'] ?? '', 'status' => $data['status'] ?? '',
                'recommendation' => $data['recommendation'] ?? '', 'save_type' => 'Manuel'
            ];
            $errorMsg = null;
            $success = savePrediction($pdo, $prediction, $errorMsg);
            if ($success) {
                ajax_json_response(['success' => true]);
            } else {
                ajax_json_response(['success' => false, 'error' => $errorMsg ?: 'Erreur inconnue']);
            }
        }
        $jsonFolder = realpath(__DIR__ . '/../python/Donnee_parametres');
        if ($jsonFolder === false || !is_dir($jsonFolder)) {
            ajax_json_response(['success' => false, 'error' => 'Le dossier Donnee_parametres est introuvable.']);
        }
        $filename = $jsonFolder . '/parameters_' . date('Y-m-d_H-i-s') . '.json';
        file_put_contents($filename, json_encode(['timestamp' => date('Y-m-d H:i:s'), 'user' => $_SESSION['user'], 'parameters' => $data], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
        ajax_json_response(['status' => 'success', 'message' => 'Paramètres sauvegardés']);
    } catch (Exception $e) {
        error_log('Erreur AJAX: ' . $e->getMessage());
        ajax_json_response(['success' => false, 'error' => $e->getMessage()]);
    }
}

// Déconnexion
if (isset($_GET['logout'])) {
    $_SESSION = []; session_destroy();
    if (isset($_COOKIE['remember_user'])) setcookie("remember_user", "", time() - 3600, "/");
    header("Location: index.php?logout=1");
    exit();
}

$history_predictions = getPredictions($pdo);
$deleted_sauvegardes = getDeletedSauvegardes($pdo);
$saved_results = getSavedResults($pdo);
$active_tab = isset($_GET['tab']) ? $_GET['tab'] : (isset($_SESSION['last_tab']) ? $_SESSION['last_tab'] : 'dashboard');
$_SESSION['last_tab'] = $active_tab;
?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vala Bleu • Dashboard</title>
    <link rel="icon" type="image/png" href="logos.png">
    <link rel="stylesheet" href="styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
</head>
<body>

<div id="toast" class="toast-notification"></div>

<div class="sidebar">
    <div class="sidebar-header"><h2>VALA BLEU</h2><p>Dashboard</p></div>
    <nav class="sidebar-nav">
        <div class="menu-item active-menu" onclick="showTab('dashboard')"><span class="menu-icon">⚙️</span><span>Paramètres</span></div>
        <div class="menu-item" onclick="showTab('resultats')"><span class="menu-icon">📊</span><span>Résultats</span></div>
        <div class="menu-item" onclick="showTab('sauvegardes')"><span class="menu-icon">💾</span><span>Sauvegardes</span></div>
        <div class="menu-item" onclick="showTab('historique')"><span class="menu-icon">📋</span><span>Historique</span></div>
        <div class="menu-item" onclick="showTab('corbeille')"><span class="menu-icon">🗑️</span><span>Corbeille</span></div>
    </nav>
    <a href="?logout=1" class="logout-link"><span class="menu-icon">🚪</span><span>Déconnexion</span></a>
</div>

<!-- BARRE UTILISATEUR -->
<div style="position:absolute;top:10px;right:20px;background:linear-gradient(135deg,#1e293b,#334155);color:#fff;padding:8px 20px;border-radius:8px;font-size:0.98rem;box-shadow:0 2px 16px rgba(0,0,0,0.22);z-index:2000;display:flex;align-items:center;gap:8px;pointer-events:auto;">
    <span>👤 Connecté en tant que <strong><?php echo htmlspecialchars((string)($_SESSION['user'] ?? '')); ?></strong></span>
</div>

<div class="main-content">

    <!-- PARAMÈTRES -->
    <div id="dashboard" class="tab-content active-tab">
        <div class="page-title"><h1>Saisie des paramètres</h1><p>Configuration pour l'analyse de charge WordPress</p></div>
        <div class="param-section"><h4>📈 Trafic</h4><div class="grid-4">
            <div class="form-group"><label>Visiteurs / jour <span class="required">*</span></label><input type="number" id="visitors_per_day" placeholder="Ex: 5000"></div>
            <div class="form-group"><label>Pages vues / jour</label><input type="number" id="pageviews_per_day" placeholder="Ex: 150"></div>
            <div class="form-group"><label>Taux de croissance (%) <span class="required">*</span></label><input type="number" id="traffic_growth_rate" placeholder="Ex: 15"></div>
            <div class="form-group"><label>PICS HORAIRES</label><div class="time-range"><input type="time" id="peak_hours_start"><span class="time-separator">à</span><input type="time" id="peak_hours_end"></div></div>
        </div></div>
        <div class="param-section"><h4>🖥️ Ressources Serveur</h4><div class="grid-4">
            <div class="form-group"><label>CPU moyen (%) <span class="required">*</span></label><input type="number" id="cpu_usage_avg" placeholder="Ex: 45"></div>
            <div class="form-group"><label>CPU max (%) <span class="required">*</span></label><input type="number" id="cpu_usage_peak" placeholder="Ex: 75"></div>
            <div class="form-group"><label>RAM moyenne (%) <span class="required">*</span></label><input type="number" id="ram_usage_avg" placeholder="Ex: 60"></div>
            <div class="form-group"><label>RAM max (%) <span class="required">*</span></label><input type="number" id="ram_usage_max" placeholder="Ex: 85"></div>
            <div class="form-group"><label>Disque utilisé (%)</label><input type="number" id="disk_usage_avg" placeholder="Ex: 45"></div>
            <div class="form-group"><label>Disque max (%)</label><input type="number" id="disk_usage_max" placeholder="Ex: 70"></div>
            <div class="form-group"><label>Temps réponse (ms)</label><input type="number" id="response_time" placeholder="Ex: 350"></div>
            <div class="form-group"><label>I/O Disque (IOPS)</label><div class="double-input"><div class="input-half"><label>Read</label><input type="number" id="disk_read_iops" placeholder="120"></div><div class="input-half"><label>Write</label><input type="number" id="disk_write_iops" placeholder="80"></div></div></div>
        </div></div>
        <div class="param-section"><h4>🔌 WordPress</h4><div class="grid-4">
            <div class="form-group"><label>Nombre de plugins <span class="required">*</span></label><input type="number" id="plugin_count" placeholder="Ex: 25"></div>
            <div class="form-group"><label>Plugins lourds</label><div class="checkbox-group" id="heavy_plugins_group">
                <label class="checkbox-item"><input type="checkbox" value="woocommerce">WooCommerce</label>
                <label class="checkbox-item"><input type="checkbox" value="elementor">Elementor</label>
                <label class="checkbox-item"><input type="checkbox" value="wpml">WPML</label>
                <label class="checkbox-item"><input type="checkbox" value="yoast">Yoast SEO</label>
                <label class="checkbox-item"><input type="checkbox" value="revslider">RevSlider</label>
                <label class="checkbox-item"><input type="checkbox" value="gravityforms">Gravity Forms</label>
            </div></div>
            <div class="form-group"><label>Version PHP</label><select id="php_version"><option value="none" selected>Choisir quelle version</option><option value="7.4">PHP 7.4</option><option value="8.0">PHP 8.0</option><option value="8.1">PHP 8.1</option><option value="8.2">PHP 8.2</option><option value="8.3">PHP 8.3</option></select></div>
            <div class="form-group"><label>Cache activé</label><select id="cache_enabled"><option value="none" selected>Choisir quelle option</option><option value="oui">Oui</option><option value="non">Non</option></select></div>
            <div class="form-group"><label>CDN activé</label><select id="cdn_enabled"><option value="none" selected>Choisir quelle option</option><option value="oui">Oui</option><option value="non">Non</option></select></div>
            <div class="form-group"><label>Pack WordPress <span class="required">*</span></label><select id="wp_type"><option value="none" selected>Choisir quel pack</option><option value="small">SMALL</option><option value="medium">MEDIUM</option><option value="performance">PERFORMANCE</option></select></div>
        </div></div>
        <div class="action-center"><button class="btn-primary btn-launch" onclick="runAnalysis()"><span>🚀</span> LANCER L'ANALYSE Prédictif</button></div>
    </div>

    <!-- RÉSULTATS -->
    <div id="resultats" class="tab-content">
        <div class="page-title"><h1>Résultats de l'analyse</h1><p>Prédiction basée sur les paramètres fournis</p></div>
        <div id="loadingResults" class="card loading-card" style="display:none;"><div class="loading-spinner">⏳</div><p>Calcul en cours...</p></div>
        <div id="noResults" class="card empty-state" style="display:none;">
            <div class="empty-icon"><img src="icons/resultas.png" alt="Aucun résultat" style="width:64px;height:64px;"></div>
            <h3>Aucune analyse générée</h3>
            <p>Remplissez les paramètres et cliquez sur "LANCER L'ANALYSE"</p>
        </div>
        <div id="resultsContainer" style="display:block;">
            <!-- Bloc 1 : Scores de Performance -->
            <div class="card"><h3>📊 Scores de Performance</h3><div id="scoresDisplay">
                <div style="text-align:center;padding:40px;color:#888;">
                    <p style="font-size:18px;">Aucune prédiction effectuée</p>
                    <p style="font-size:14px;">Lancez une analyse pour voir les résultats</p>
                </div>
            </div></div>
            
            <!-- Bloc 2 : Recommandation -->
            <div class="card"><h3>💡 Recommandation</h3><div id="recommendationDisplay">
                <div style="text-align:center;padding:40px;color:#888;">
                    <p style="font-size:18px;">En attente d'analyse</p>
                    <p style="font-size:14px;">Les recommandations apparaîtront ici</p>
                </div>
            </div></div>
            
            <!-- Bloc 3 : Graphiques d'analyse -->
            <div class="card" id="imagesContainer" style="display:block;">
                <h3>🖼️ Graphiques d'analyse</h3>
                <div style="display:flex;align-items:center;gap:18px;margin-bottom:12px;">
                    <button id="runGrapheXGBoostBtn" class="btn-success" style="background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff;padding:10px 28px;font-size:1.1em;border:none;border-radius:7px;cursor:pointer;box-shadow:0 2px 8px rgba(34,197,94,0.12);font-weight:600;" onclick="runGrapheXGBoostModel()">
                        <span>📈</span> Graphe
                    </button>
                </div>
                <div class="images-grid" id="imagesDisplay">
                    <div style="text-align:center;padding:40px;color:#888;">
                        <p style="font-size:18px;">Aucun graphique généré</p>
                        <p style="font-size:14px;">Cliquez sur "Graphe" pour générer les visualisations</p>
                    </div>
                </div>
            </div>
            
            <!-- Bloc 4 : Téléchargement des arbres - TOUJOURS VISIBLE -->
            <div class="card" id="treeDownloads" style="display:block;">
                <h3>🌳 Téléchargement des arbres XGBoost</h3>
                <div class="tree-links">
                    <a href="http://localhost:8000/download/graphe/tree0" class="tree-link" download>📥 Tree 0</a>
                    <a href="http://localhost:8000/download/graphe/treefinal" class="tree-link" download>📥 Tree Final</a>
                    <a href="http://localhost:8000/download/graphe/feature_importance" class="tree-link" download>📥 Feature Importance</a>
                </div>
            </div>
        </div>
    </div>

    <!-- SAUVEGARDES JSON -->
    <div id="sauvegardes" class="tab-content">
        <div class="page-header-with-action"><div class="page-title"><h1>💾 Sauvegardes des résultats</h1><p>Résultats sauvegardés sans images</p></div></div>
            <div class="card" style="display: flex; justify-content: center; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;">
                <button class="btn-primary btn-save" id="saveResultBtn" onclick="saveCurrentResult()">
                    <span>💾</span> Sauvegarder dans l'historique
                </button>
            </div>
    </div>

    <!-- HISTORIQUE -->
    <div id="historique" class="tab-content">
        <div class="page-header-with-action"><div class="page-title"><h1>Historique des analyses</h1><p>Toutes les analyses sauvegardées</p></div>
        <div style="height: 5px; width: 50%;"></div>
        <a href="?export_full_csv=1" class="btn-export-csv">
            <span>📥</span> Exporter tout en CSV</a></div>
        <div class="config-card"><div class="table-wrapper"><table class="history-table"><thead><tr><th>Date</th><th>Pack</th><th>Visiteurs/j</th><th>Croissance</th><th>CPU/RAM</th><th>Plugins</th><th>Score</th><th>Charge</th><th>Statut</th><th>Action</th></tr></thead><tbody>
            <?php if (count($history_predictions) > 0): ?>
                <?php foreach ($history_predictions as $pred): ?>
                    <tr>
                        <td class="td-date">
                            <?php echo date('d/m/Y', strtotime($pred['created_at'])); ?><br>
                            <span style="font-size:11px;color:#888;display:block;line-height:1;">
                                <?php echo date('H:i', strtotime($pred['created_at'])); ?>
                            </span>
                        </td>
                        <td><span class="badge-pack"><?php echo strtoupper($pred['wp_type'] ?? 'N/A'); ?></span></td>
                        <td class="td-number"><?php echo is_numeric($pred['visitors_per_day']) ? number_format((float)$pred['visitors_per_day']) : ''; ?></td>
                        <td class="td-growth"><?php echo $pred['traffic_growth_rate']; ?>%</td>
                        <td class="td-usage"><?php echo $pred['cpu_usage_avg']; ?>% / <?php echo $pred['ram_usage_avg']; ?>%</td>
                        <td class="td-number"><?php echo $pred['plugin_count']; ?></td>
                        <td class="td-score"><?php echo $pred['xgboost_score']; ?>%</td>
                        <td class="td-number"><?php echo $pred['predicted_load']; ?>%</td>
                        <td><span class="badge-status <?php echo $pred['status'] == 'CRITIQUE' ? 'badge-critical' : ($pred['status'] == 'SURVEILLANCE' ? 'badge-warning' : 'badge-optimal'); ?>"><?php echo $pred['status']; ?></span></td>
                        <td><button class="btn-icon btn-archive" onclick="archiverAnalyse('<?php echo $pred['id']; ?>')" title="Archiver">📦</button></td>
                    </tr>
                <?php endforeach; ?>
            <?php else: ?>
                <tr><td colspan="10" class="empty-table-cell"><div class="empty-icon"><img src="icons/historique.png" alt="Aucun historique" style="width:64px;height:64px;"></div><p>Aucune analyse sauvegardée</p></td></tr>
            <?php endif; ?>
        </tbody></table></div></div>
    </div>

    <!-- CORBEILLE -->
    <div id="corbeille" class="tab-content">
        <div class="page-header-with-action">
            <div class="page-title"><h1>Corbeille</h1><p>Éléments supprimés</p></div>
            <div style="height: 20px; width: 80%;"></div>
                <button class="btn-danger" onclick="viderCorbeille()"><span>🗑️</span> Vider la corbeille</button></div>
        <div class="config-card"><div class="table-wrapper"><table class="history-table"><thead><tr><th>Date</th><th>Pack</th><th>Visiteurs/j</th><th>Croissance</th><th>CPU/RAM</th><th>Plugins</th><th>Score</th><th>Charge</th><th>Statut</th><th>Action</th></tr></thead><tbody>
            <?php if (count($deleted_sauvegardes) > 0): ?>
                <?php foreach ($deleted_sauvegardes as $del): ?>
                    <tr class="tr-deleted">
                        <td class="td-date">
                            <?php echo date('d/m/Y', strtotime($del['created_at'])); ?><br>
                            <span style="font-size:11px;color:#888;display:block;line-height:1;">
                                <?php echo date('H:i', strtotime($del['created_at'])); ?>
                            </span>
                        </td>
                        <td><?php echo strtoupper($del['wp_type'] ?? 'N/A'); ?></td>
                        <td class="td-number"><?php echo is_numeric($del['visitors_per_day']) ? number_format((float)$del['visitors_per_day']) : ''; ?></td>
                        <td class="td-growth"><?php echo $del['traffic_growth_rate']; ?>%</td>
                        <td class="td-usage"><?php echo $del['cpu_usage_avg']; ?>% / <?php echo $del['ram_usage_avg']; ?>%</td>
                        <td class="td-number"><?php echo $del['plugin_count']; ?></td>
                        <td class="td-score"><?php echo $del['xgboost_score']; ?>%</td>
                        <td class="td-number"><?php echo $del['predicted_load']; ?>%</td>
                        <td><span class="badge-deleted">🗑️ Supprimé</span></td>
                        <td><button class="btn-icon btn-restore" onclick="restaurerAnalyse('<?php echo $del['id']; ?>')" title="Restaurer">🔄</button><button class="btn-icon btn-delete-forever" onclick="supprimerDefinitivement('<?php echo $del['id']; ?>')" title="Supprimer">❌</button></td>
                    </tr>
                <?php endforeach; ?>
            <?php else: ?>
                <tr><td colspan="10" class="empty-table-cell"><div class="empty-icon"><img src="icons/corbeille.png" alt="Corbeille vide" style="width:64px;height:64px;"></div><p>Corbeille vide</p></td></tr>
            <?php endif; ?>
        </tbody></table></div></div>
    </div>

</div>

<div id="toast" class="toast-notification"></div>

<script>
var currentPrediction = null;

function showTab(tabId) {
    var tabs = document.querySelectorAll('.tab-content');
    for (var i = 0; i < tabs.length; i++) { tabs[i].classList.remove('active-tab'); }
    var menus = document.querySelectorAll('.menu-item');
    for (var i = 0; i < menus.length; i++) { menus[i].classList.remove('active-menu'); }
    var target = document.getElementById(tabId);
    if (target) { target.classList.add('active-tab'); }
    var tabNames = ['dashboard', 'resultats', 'sauvegardes', 'historique', 'corbeille'];
    var index = tabNames.indexOf(tabId);
    if (index >= 0 && menus[index]) { menus[index].classList.add('active-menu'); }
    sessionStorage.setItem('activeTab', tabId);
}

function showToast(message, isError) {
    isError = isError || false;
    var toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.display = 'block';
    toast.style.background = isError ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 'linear-gradient(135deg, #10b981, #059669)';
    setTimeout(function() { toast.style.display = 'none'; }, 4000);
}

function getFormParams() {
    var start = document.getElementById('peak_hours_start').value;
    var end = document.getElementById('peak_hours_end').value;
    var peakHours = 4;
    if (start && end) { peakHours = Math.max(1, parseInt(end.split(':')[0]) - parseInt(start.split(':')[0])); }
    var heavyPlugins = [];
    var checkboxes = document.querySelectorAll('#heavy_plugins_group input[type="checkbox"]:checked');
    for (var i = 0; i < checkboxes.length; i++) { heavyPlugins.push(checkboxes[i].value); }
    return {
        cpu_usage_avg: parseFloat(document.getElementById('cpu_usage_avg').value) || 0,
        cpu_usage_peak: parseFloat(document.getElementById('cpu_usage_peak').value) || 0,
        ram_usage_avg: parseFloat(document.getElementById('ram_usage_avg').value) || 0,
        ram_usage_max: parseFloat(document.getElementById('ram_usage_max').value) || 0,
        disk_usage_avg: parseFloat(document.getElementById('disk_usage_avg').value) || 0,
        disk_usage_max: parseFloat(document.getElementById('disk_usage_max').value) || 0,
        disk_read_iops: parseFloat(document.getElementById('disk_read_iops').value) || 0,
        disk_write_iops: parseFloat(document.getElementById('disk_write_iops').value) || 0,
        response_time: parseFloat(document.getElementById('response_time').value) || 0,
        visitors_per_day: parseFloat(document.getElementById('visitors_per_day').value) || 0,
        pageviews_per_day: parseFloat(document.getElementById('pageviews_per_day').value) || 0,
        traffic_growth_rate: parseFloat(document.getElementById('traffic_growth_rate').value) || 0,
        peak_hours_start: start, peak_hours_end: end, peak_hours: peakHours,
        plugin_count: parseFloat(document.getElementById('plugin_count').value) || 0,
        heavy_plugins: heavyPlugins.join(','),
        php_version: document.getElementById('php_version').value,
        cache_enabled: document.getElementById('cache_enabled').value,
        cdn_enabled: document.getElementById('cdn_enabled').value,
        wp_type: document.getElementById('wp_type').value
    };
}

function validateParams(params) {
    if (!params.cpu_usage_avg || !params.cpu_usage_peak || !params.ram_usage_avg || !params.ram_usage_max || !params.visitors_per_day || !params.traffic_growth_rate || !params.plugin_count || !params.wp_type) {
        showToast('Champs * requis', true);
        return false;
    }
    for (var key in params) {
        if (typeof params[key] === 'number' && params[key] < 0) {
            showToast('Valeurs négatives interdites', true);
            return false;
        }
    }
    return true;
}

function logOperation(context, res, isSuccess) {
    if (isSuccess) {
        console.log(`%c[SUCCESS][${context}]`, 'color:green;font-weight:bold;', res && res.message ? res.message : 'Succès', '\nRéponse complète:', res);
    } else {
        console.error(`%c[ECHEC][${context}]`, 'color:red;font-weight:bold;', (res && (res.error || res.message)) ? (res.error || res.message) : 'Erreur', '\nRéponse complète:', res);
    }
}

function saveParamsToPythonAPI(params) {
    return fetch("http://localhost:8000/save-parameters", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params)
    })
    .then(function(r) { return r.json(); })
    .then(function(res) {
        logOperation('SaveParamsToPythonAPI', res, res.status === "success");
        if (res.status === "success") {
            showToast("Paramètres sauvegardés dans Donnee_parametres !");
        } else {
            showToast("Erreur sauvegarde API Python", true);
        }
        return res.status === "success";
    })
    .catch(function(e) {
        logOperation('SaveParamsToPythonAPI', e, false);
        showToast("Erreur API Python", true);
        return false;
    });
}

function loadGraphImages() {
    var timestamp = new Date().getTime();
    var learningCurveUrl = 'http://localhost:8000/static/learning_curve.png?t=' + timestamp;
    
    var img = new Image();
    img.onload = function() {
        var display = document.getElementById('imagesDisplay');
        display.innerHTML = '<div class="image-card" style="max-width:48%;"><h4>📈 Courbe d\'apprentissage</h4><img src="' + learningCurveUrl + '" onclick="window.open(\'' + learningCurveUrl + '\',\'_blank\')" style="width:100%;"><p class="image-name">learning_curve.png</p></div>';
        document.getElementById('imagesContainer').style.display = 'block';
    };
    img.onerror = function() {
        var now = new Date();
        var dateStr = now.getFullYear() + 
            String(now.getMonth() + 1).padStart(2, '0') + 
            String(now.getDate()).padStart(2, '0') + '_' + 
            String(now.getHours()).padStart(2, '0') + 
            String(now.getMinutes()).padStart(2, '0') + 
            String(now.getSeconds()).padStart(2, '0');
        
        var fullUrl = 'http://localhost:8000/static/learning_curve_' + dateStr + '.png?t=' + timestamp;
        var img2 = new Image();
        img2.onload = function() {
            var display = document.getElementById('imagesDisplay');
            display.innerHTML = '<div class="image-card" style="max-width:48%;"><h4>📈 Courbe d\'apprentissage</h4><img src="' + fullUrl + '" onclick="window.open(\'' + fullUrl + '\',\'_blank\')" style="width:100%;"><p class="image-name">learning_curve_' + dateStr + '.png</p></div>';
            document.getElementById('imagesContainer').style.display = 'block';
        };
        img2.onerror = function() {
            fetch('http://localhost:8000/get-images-list')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.images && data.images.length > 0) {
                        displayImages(data.images);
                    } else {
                        document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Graphiques générés</p><p style="font-size:14px;">Les images seront disponibles après actualisation</p></div>';
                    }
                })
                .catch(function() {
                    document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Graphiques générés</p><p style="font-size:14px;">Rafraîchissez la page pour les voir</p></div>';
                });
        };
        img2.src = fullUrl;
    };
    img.src = learningCurveUrl;
}

function displayImages(images) {
    var container = document.getElementById('imagesContainer');
    var display = document.getElementById('imagesDisplay');
    if (!images || !images.length) { 
        container.style.display = 'none'; 
        display.innerHTML = ''; 
        return; 
    }
    // On veut les derniers de chaque type
    var allowedTypes = ['tree', 'correlation', 'feature_importance', 'residus', 'learning_curve'];
    var names = {
        tree: 'Arbre de décision',
        correlation: 'Matrice de corrélation',
        feature_importance: 'Importance des caractéristiques',
        residus: 'Résidus',
        learning_curve: 'Courbe d\'apprentissage'
    };
    var latestByType = {};
    for (var i = 0; i < images.length; i++) {
        var img = images[i];
        if (allowedTypes.indexOf(img.type) === -1) continue;
        if (!latestByType[img.type] || img.filename > latestByType[img.type].filename) {
            latestByType[img.type] = img;
        }
    }
    var html = '';
    for (var t of allowedTypes) {
        var span = (t === 'learning_curve') ? ' style="grid-column: span 2;"' : '';
        html += '<div class="image-card"' + span + '><h4>' + names[t] + '</h4>';
        if (latestByType[t]) {
            var img = latestByType[t];
            var imgUrl = img.url.startsWith('http') ? img.url : ('http://localhost:8000' + img.url);
            html += '<img src="' + imgUrl + '" onclick="window.open(\'' + imgUrl + '\',\'_blank\')"><p class="image-name">' + imgUrl.split('/').pop() + '</p>';
        } else {
            html += '<div class="empty-image"></div>';
        }
        html += '</div>';
    }
    display.innerHTML = html;
    container.style.display = 'block';
}

function displayResults(data) {
    var sat = data.saturation_text || (data.predicted_saturation_months + ' mois');
    if (data.predicted_saturation_months == 999) sat = 'Illimité';
    if (data.predicted_saturation_months == 0) sat = 'SATURÉ';
    var cls = data.status === 'CRITIQUE' ? 'critical' : (data.status === 'SURVEILLANCE' ? 'warning' : 'optimal');
    var col = data.status === 'CRITIQUE' ? '#dc2626' : (data.status === 'SURVEILLANCE' ? '#d97706' : '#10b981');
    document.getElementById('scoresDisplay').innerHTML = '<div class="scores-grid"><div class="score-item"><div class="score-label">Score XGBoost</div><div class="score-value">' + data.xgboost_score + '%</div><div class="gauge"><div class="gauge-fill ' + cls + '" style="width:' + data.xgboost_score + '%"></div></div></div><div class="score-item"><div class="score-label">Charge prédite</div><div class="score-value" style="color:' + col + '">' + data.predicted_load + '%</div><div class="gauge"><div class="gauge-fill ' + cls + '" style="width:' + data.predicted_load + '%"></div></div></div><div class="score-item"><div class="score-label">Saturation</div><div class="score-value saturation">' + sat + '</div><span class="badge-status badge-' + cls + '">' + data.status + '</span></div></div>';
    document.getElementById('recommendationDisplay').innerHTML = '<div class="recommendation-text">' + (data.recommendation || 'Aucune recommandation.') + '</div>';
}

function runAnalysis() {
    var params = getFormParams();
    if (!validateParams(params)) return;
    
    document.getElementById('loadingResults').style.display = 'none';
    document.getElementById('noResults').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'block';
    
    document.getElementById('scoresDisplay').innerHTML = '<div style="text-align:center;padding:20px;"><span style="font-size:24px;">⏳</span><p>Génération en cours...</p></div>';
    document.getElementById('recommendationDisplay').innerHTML = '<div style="text-align:center;padding:20px;"><span style="font-size:24px;">⏳</span><p>Génération en cours...</p></div>';
    
    document.getElementById('imagesDisplay').innerHTML = '';
    document.getElementById('imagesContainer').style.display = 'block';
    
    document.getElementById('treeDownloads').style.display = 'block';
    
    showTab('resultats');
    
    var apiHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" ? "localhost" : "python_api";
    var apiUrl = "http://" + apiHost + ":8000/predict/from-file";
    
    saveParamsToPythonAPI(params);
    
    fetch(apiUrl)
        .then(function(r) { return r.json(); })
        .then(function(res) {
            document.getElementById('loadingResults').style.display = 'none';
            logOperation('Prédiction', res, res.status === "success");
            
            if (res.status === "success") {
                currentPrediction = res.output.result;
                displayResults(res.output.result);
                document.getElementById('treeDownloads').style.display = 'block';
                document.getElementById('imagesDisplay').innerHTML = '';
                document.getElementById('imagesContainer').style.display = 'block';
                document.getElementById('resultsContainer').style.display = 'block';
                showToast('Prédiction terminée !');
                sessionStorage.setItem('lastPrediction', JSON.stringify(currentPrediction));
            } else {
                var errMsg = 'Erreur inconnue';
                if (res.output && res.output.result) {
                    if (res.output.result.saturation_text && res.output.result.saturation_text !== 'None') {
                        errMsg = res.output.result.saturation_text;
                    } else if (res.output.result.recommendation && res.output.result.recommendation !== 'None') {
                        errMsg = res.output.result.recommendation;
                    } else {
                        errMsg = JSON.stringify(res.output.result, null, 2);
                    }
                } else if (res.message) {
                    errMsg = res.message;
                } else {
                    errMsg = JSON.stringify(res, null, 2);
                }
                showToast(errMsg, true);
                document.getElementById('scoresDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de prédiction</p><p style="font-size:14px;">' + errMsg + '</p></div>';
                document.getElementById('recommendationDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Recommandation indisponible</p></div>';
            }
        })
        .catch(function(e) {
            document.getElementById('loadingResults').style.display = 'none';
            logOperation('Prédiction', e, false);
            showToast('API indisponible', true);
            document.getElementById('scoresDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de connexion</p><p style="font-size:14px;">API indisponible</p></div>';
            document.getElementById('recommendationDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Recommandation indisponible</p></div>';
        });
}

function runGrapheXGBoostModel() {
    var btn = document.getElementById('runGrapheXGBoostBtn');
    
    if (btn.disabled) {
        return; // Empêche le re-clic
    }
    
    btn.disabled = true;
    btn.innerHTML = '⏳ Génération en cours...';
    
    document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:20px;"><span style="font-size:24px;">⏳</span><p>Génération des graphiques en cours...</p></div>';
    document.getElementById('imagesContainer').style.display = 'block';
    
    fetch('http://localhost:8000/run/graphe-xgboost-model', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(function(r) { return r.json(); })
    .then(function(res) {
        if (res.status === 'success') {
            showToast(res.message || 'Graphiques générés avec succès !');
            
            // Charger les images dynamiquement SANS recharger la page
            setTimeout(function() {
                loadGraphImages();
                btn.disabled = false;
                btn.innerHTML = '<span>📈</span> Graphe';
            }, 1000);
        } else {
            showToast(res.message || 'Erreur lors de la génération des graphiques', true);
            document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de génération</p><p style="font-size:14px;">Réessayez</p></div>';
            btn.disabled = false;
            btn.innerHTML = '<span>📈</span> Graphe';
        }
    })
    .catch(function(e) {
        console.error('Erreur:', e);
        showToast('Erreur de connexion à l\'API', true);
        document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de connexion</p></div>';
        btn.disabled = false;
        btn.innerHTML = '<span>📈</span> Graphe';
    });
}

function saveCurrentResult() {
    if (!currentPrediction) { showToast('Aucun résultat', true); return; }
    var btn = document.getElementById('saveResultBtn');
    btn.disabled = true; btn.innerHTML = 'Sauvegarde en cours...';
    var params = getFormParams();
    var dataToSave = Object.assign({}, params, currentPrediction);
    fetch(window.location.href, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        body: JSON.stringify(dataToSave)
    })
    .then(function(r) { return r.json(); })
    .then(function(res) {
        logOperation('Sauvegarde', res, res.success);
        if (res.success) {
            showToast(res.message ? res.message : 'Analyse sauvegardée !');
        } else {
            showToast(res.error ? res.error : 'Erreur', true);
            btn.disabled = false;
            btn.innerHTML = 'Sauvegarder dans l\'historique';
        }
        setTimeout(function() { location.reload(); }, 1500);
    });
}

function saveAsJson() {
    if (!currentPrediction) { showToast('Aucun résultat', true); return; }
    var btn = document.getElementById('saveJsonBtn');
    btn.disabled = true; btn.innerHTML = 'Sauvegarde JSON en cours...';
    fetch(window.location.href, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' }, body: JSON.stringify({ action: 'save_json', result_data: currentPrediction }) })
    .then(function(r) { return r.json(); })
    .then(function(res) { 
        logOperation('Sauvegarde JSON', res, res.success);
        if (res.success) {
            showToast(res.message ? res.message : 'Sauvegarde JSON effectuée !');
        } else {
            showToast(res.error ? res.error : 'Erreur', true);
            btn.disabled = false;
            btn.innerHTML = 'Sauvegarde JSON (sans images)';
        }
        setTimeout(function() { location.reload(); }, 1500);
    });
}

function archiverAnalyse(id) { if (confirm('Archiver ?')) ajaxAction('archive', id); }
function restaurerAnalyse(id) { if (confirm('Restaurer ?')) ajaxAction('restore', id); }
function supprimerDefinitivement(id) { if (confirm('Supprimer définitivement ?')) ajaxAction('delete', id); }
function viderCorbeille() { if (confirm('Vider toute la corbeille ?')) ajaxAction('empty_trash'); }

function ajaxAction(action, id) {
    var body = { action: action };
    if (id) body[action + '_id'] = id;
    fetch(window.location.href, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' }, body: JSON.stringify(body) })
    .then(function(r) { return r.json(); })
    .then(function(res) {
        logOperation('Opération', res, res.success);
        if (res.success) {
            showToast(res.message ? res.message : 'Terminé');
        } else {
            showToast(res.error ? res.error : 'Erreur', true);
        }
        setTimeout(function() { location.reload(); }, 1000);
    });
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    var last = sessionStorage.getItem('lastPrediction');
    if (last) {
        try { currentPrediction = JSON.parse(last); } catch(e) { currentPrediction = null; }
    }
    var params = new URLSearchParams(window.location.search);
    var tab = params.get('tab') || sessionStorage.getItem('activeTab') || 'dashboard';
    var validTabs = ['dashboard', 'resultats', 'sauvegardes', 'historique', 'corbeille'];
    if (validTabs.indexOf(tab) === -1) tab = 'dashboard';
    showTab(tab);

    document.querySelectorAll('button').forEach(function(btn) {
        btn.addEventListener('click', function(e) {
            console.log('%c[BOUTON] Clic sur : ' + (btn.innerText || btn.id || btn.className), 'color:#2563eb;font-weight:bold;', btn);
        });
    });
});
</script>
</body>
</html>
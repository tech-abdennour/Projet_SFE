<?php
set_time_limit(300);
session_start();
date_default_timezone_set('Africa/Casablanca');

// Gestion de la déconnexion
if (isset($_GET['logout'])) {
    error_log("🚪 DÉCONNEXION: " . ($_SESSION['user'] ?? 'unknown'));
    session_destroy();
    setcookie("remember_user", "", time() - 3600, "/");
    header("Location: index.php?logout=1");
    exit;
}

// Vérification de la session
if (!isset($_SESSION['user'])) {
    header("Location: index.php");
    exit;
}

// Initialisation de la base de données
$db_file = 'vala_bleu.db';
$pdo = null;

try {
    $pdo = new PDO("sqlite:" . $db_file);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    error_log("✅ Connexion SQLite réussie: " . $db_file);
    
    $pdo->exec("CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY, user TEXT, created_at TEXT,
        cpu_usage_avg TEXT, cpu_usage_peak TEXT, ram_usage_avg TEXT, ram_usage_max TEXT,
        disk_usage_avg TEXT, disk_usage_max TEXT, disk_read_iops TEXT, disk_write_iops TEXT,
        response_time TEXT, visitors_per_day TEXT, pageviews_per_day TEXT, traffic_growth_rate TEXT,
        peak_hours_start TEXT, peak_hours_end TEXT, peak_hours TEXT, plugin_count TEXT,
        heavy_plugins TEXT, php_version TEXT, cache_enabled TEXT, cdn_enabled TEXT,
        wp_type TEXT, predicted_load TEXT, error_rate TEXT, saturation_days TEXT,
        saturation_months TEXT, saturation_jours TEXT, saturation_text TEXT, saturation_months_raw TEXT,
        status TEXT, recommendation TEXT, save_type TEXT, is_deleted INTEGER DEFAULT 0
    )");
    
    $pdo->exec("CREATE TABLE IF NOT EXISTS deleted_sauvegardes (
        id TEXT PRIMARY KEY, user TEXT, created_at TEXT, deleted_at TEXT,
        cpu_usage_avg TEXT, cpu_usage_peak TEXT, ram_usage_avg TEXT, ram_usage_max TEXT,
        disk_usage_avg TEXT, disk_usage_max TEXT, disk_read_iops TEXT, disk_write_iops TEXT,
        response_time TEXT, visitors_per_day TEXT, pageviews_per_day TEXT, traffic_growth_rate TEXT,
        peak_hours_start TEXT, peak_hours_end TEXT, peak_hours TEXT, plugin_count TEXT,
        heavy_plugins TEXT, php_version TEXT, cache_enabled TEXT, cdn_enabled TEXT,
        wp_type TEXT, predicted_load TEXT, error_rate TEXT, saturation_days TEXT,
        saturation_months TEXT, saturation_jours TEXT, saturation_text TEXT, saturation_months_raw TEXT,
        status TEXT, recommendation TEXT, save_type TEXT
    )");
    
    $pdo->exec("CREATE TABLE IF NOT EXISTS saved_results (
        id TEXT PRIMARY KEY, created_at TEXT, data_json TEXT, save_type TEXT
    )");
    
    error_log("✅ Tables SQLite vérifiées/créées");
} catch (Exception $e) {
    error_log("❌ SQLite error: " . $e->getMessage());
}

// Fonctions de gestion des données

function saveResultJson($pdo, $data, $type = 'resultat') {
    if ($pdo === null) return false;
    try {
        $stmt = $pdo->prepare("INSERT INTO saved_results (id, created_at, data_json, save_type) VALUES (:id, :created_at, :data_json, :save_type)");
        $stmt->execute([
            ':id' => uniqid(),
            ':created_at' => date('Y-m-d H:i:s'),
            ':data_json' => json_encode($data),
            ':save_type' => $type
        ]);
        return true;
    } catch (Exception $e) { return false; }
}
function getPredictions($pdo) {
    if ($pdo === null || !isset($_SESSION['user'])) return [];
    try {
        $stmt = $pdo->prepare("SELECT * FROM predictions WHERE is_deleted = 0 AND user = :user ORDER BY created_at DESC");
        $stmt->execute([':user' => $_SESSION['user']]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (Exception $e) { return []; }
}

function getDeletedSauvegardes($pdo) {
    if ($pdo === null || !isset($_SESSION['user'])) return [];
    try {
        $stmt = $pdo->prepare("SELECT * FROM deleted_sauvegardes WHERE user = :user ORDER BY deleted_at DESC");
        $stmt->execute([':user' => $_SESSION['user']]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (Exception $e) { return []; }
}

function getSavedResults($pdo) {
    if ($pdo === null) return [];
    try { $stmt = $pdo->query("SELECT * FROM saved_results ORDER BY created_at DESC"); return $stmt->fetchAll(PDO::FETCH_ASSOC); }
    catch (Exception $e) { return []; }
}

function savePrediction($pdo, $data, &$errorMsg = null) {
    if ($pdo === null || !isset($_SESSION['user'])) { $errorMsg = 'Erreur connexion'; return false; }
    try {
        $stmt = $pdo->prepare("INSERT INTO predictions (id, user, created_at, cpu_usage_avg, cpu_usage_peak, ram_usage_avg, ram_usage_max, disk_usage_avg, disk_usage_max, disk_read_iops, disk_write_iops, response_time, visitors_per_day, pageviews_per_day, traffic_growth_rate, peak_hours_start, peak_hours_end, peak_hours, plugin_count, heavy_plugins, php_version, cache_enabled, cdn_enabled, wp_type, predicted_load, error_rate, saturation_days, saturation_months, saturation_jours, saturation_text, saturation_months_raw, status, recommendation, save_type, is_deleted) VALUES (:id, :user, :created_at, :cpu_usage_avg, :cpu_usage_peak, :ram_usage_avg, :ram_usage_max, :disk_usage_avg, :disk_usage_max, :disk_read_iops, :disk_write_iops, :response_time, :visitors_per_day, :pageviews_per_day, :traffic_growth_rate, :peak_hours_start, :peak_hours_end, :peak_hours, :plugin_count, :heavy_plugins, :php_version, :cache_enabled, :cdn_enabled, :wp_type, :predicted_load, :error_rate, :saturation_days, :saturation_months, :saturation_jours, :saturation_text, :saturation_months_raw, :status, :recommendation, :save_type, 0)");
        $stmt->execute([':id' => $data['id'], ':user' => $_SESSION['user'], ':created_at' => $data['created_at'], ':cpu_usage_avg' => $data['cpu_usage_avg'] ?? '', ':cpu_usage_peak' => $data['cpu_usage_peak'] ?? '', ':ram_usage_avg' => $data['ram_usage_avg'] ?? '', ':ram_usage_max' => $data['ram_usage_max'] ?? '', ':disk_usage_avg' => $data['disk_usage_avg'] ?? '', ':disk_usage_max' => $data['disk_usage_max'] ?? '', ':disk_read_iops' => $data['disk_read_iops'] ?? '', ':disk_write_iops' => $data['disk_write_iops'] ?? '', ':response_time' => $data['response_time'] ?? '', ':visitors_per_day' => $data['visitors_per_day'] ?? '', ':pageviews_per_day' => $data['pageviews_per_day'] ?? '', ':traffic_growth_rate' => $data['traffic_growth_rate'] ?? '', ':peak_hours_start' => $data['peak_hours_start'] ?? '', ':peak_hours_end' => $data['peak_hours_end'] ?? '', ':peak_hours' => $data['peak_hours'] ?? '', ':plugin_count' => $data['plugin_count'] ?? '', ':heavy_plugins' => $data['heavy_plugins'] ?? '', ':php_version' => $data['php_version'] ?? '', ':cache_enabled' => $data['cache_enabled'] ?? '', ':cdn_enabled' => $data['cdn_enabled'] ?? '', ':wp_type' => $data['wp_type'] ?? '', ':predicted_load' => $data['predicted_load'] ?? '', ':error_rate' => $data['error_rate'] ?? '', ':saturation_days' => $data['saturation_days'] ?? '', ':saturation_months' => $data['saturation_months'] ?? '', ':saturation_jours' => $data['saturation_jours'] ?? '', ':saturation_text' => $data['saturation_text'] ?? '', ':saturation_months_raw' => $data['saturation_months_raw'] ?? '', ':status' => $data['status'] ?? '', ':recommendation' => $data['recommendation'] ?? '', ':save_type' => $data['save_type'] ?? 'Manuel']);
        return true;
    } catch (Exception $e) { $errorMsg = $e->getMessage(); return false; }
}

function archivePrediction($pdo, $id) {
    if ($pdo === null) return false;
    try {
        $stmt = $pdo->prepare("SELECT * FROM predictions WHERE id = :id AND is_deleted = 0");
        $stmt->execute([':id' => $id]);
        $pred = $stmt->fetch(PDO::FETCH_ASSOC);
        if ($pred) {
            $stmt2 = $pdo->prepare("INSERT INTO deleted_sauvegardes (id, created_at, cpu_usage_avg, cpu_usage_peak, ram_usage_avg, ram_usage_max, disk_usage_avg, disk_usage_max, disk_read_iops, disk_write_iops, response_time, visitors_per_day, pageviews_per_day, traffic_growth_rate, peak_hours_start, peak_hours_end, peak_hours, plugin_count, heavy_plugins, php_version, cache_enabled, cdn_enabled, wp_type, predicted_load, error_rate, saturation_days, saturation_months, saturation_jours, saturation_text, saturation_months_raw, status, recommendation, save_type, deleted_at, user) VALUES (:id, :created_at, :cpu_usage_avg, :cpu_usage_peak, :ram_usage_avg, :ram_usage_max, :disk_usage_avg, :disk_usage_max, :disk_read_iops, :disk_write_iops, :response_time, :visitors_per_day, :pageviews_per_day, :traffic_growth_rate, :peak_hours_start, :peak_hours_end, :peak_hours, :plugin_count, :heavy_plugins, :php_version, :cache_enabled, :cdn_enabled, :wp_type, :predicted_load, :error_rate, :saturation_days, :saturation_months, :saturation_jours, :saturation_text, :saturation_months_raw, :status, :recommendation, :save_type, :deleted_at, :user)");
            $stmt2->execute([':id' => $pred['id'], ':created_at' => $pred['created_at'], ':cpu_usage_avg' => $pred['cpu_usage_avg'], ':cpu_usage_peak' => $pred['cpu_usage_peak'], ':ram_usage_avg' => $pred['ram_usage_avg'], ':ram_usage_max' => $pred['ram_usage_max'], ':disk_usage_avg' => $pred['disk_usage_avg'], ':disk_usage_max' => $pred['disk_usage_max'], ':disk_read_iops' => $pred['disk_read_iops'], ':disk_write_iops' => $pred['disk_write_iops'], ':response_time' => $pred['response_time'], ':visitors_per_day' => $pred['visitors_per_day'], ':pageviews_per_day' => $pred['pageviews_per_day'], ':traffic_growth_rate' => $pred['traffic_growth_rate'], ':peak_hours_start' => $pred['peak_hours_start'], ':peak_hours_end' => $pred['peak_hours_end'], ':peak_hours' => $pred['peak_hours'], ':plugin_count' => $pred['plugin_count'], ':heavy_plugins' => $pred['heavy_plugins'], ':php_version' => $pred['php_version'], ':cache_enabled' => $pred['cache_enabled'], ':cdn_enabled' => $pred['cdn_enabled'], ':wp_type' => $pred['wp_type'], ':predicted_load' => $pred['predicted_load'], ':error_rate' => $pred['error_rate'] ?? '', ':saturation_days' => $pred['saturation_days'] ?? '', ':saturation_months' => $pred['saturation_months'] ?? '', ':saturation_jours' => $pred['saturation_jours'] ?? '', ':saturation_text' => $pred['saturation_text'] ?? '', ':saturation_months_raw' => $pred['saturation_months_raw'] ?? '', ':status' => $pred['status'], ':recommendation' => $pred['recommendation'], ':save_type' => $pred['save_type'] ?? 'Manuel', ':deleted_at' => date('Y-m-d H:i:s'), ':user' => $pred['user'] ?? ($_SESSION['user'] ?? null)]);
            $pdo->prepare("UPDATE predictions SET is_deleted = 1 WHERE id = :id")->execute([':id' => $id]);
            return true;
        }
        return false;
    } catch (Exception $e) { return false; }
}

function deletePermanently($pdo, $id) { if ($pdo === null) return false; try { $pdo->prepare("DELETE FROM deleted_sauvegardes WHERE id = :id")->execute([':id' => $id]); return true; } catch (Exception $e) { return false; } }
function restorePrediction($pdo, $id) { if ($pdo === null) return false; try { $stmt = $pdo->prepare("SELECT * FROM deleted_sauvegardes WHERE id = :id"); $stmt->execute([':id' => $id]); $pred = $stmt->fetch(PDO::FETCH_ASSOC); if ($pred) { $new_id = uniqid(); $stmt2 = $pdo->prepare("INSERT INTO predictions (id, user, created_at, cpu_usage_avg, cpu_usage_peak, ram_usage_avg, ram_usage_max, disk_usage_avg, disk_usage_max, disk_read_iops, disk_write_iops, response_time, visitors_per_day, pageviews_per_day, traffic_growth_rate, peak_hours_start, peak_hours_end, peak_hours, plugin_count, heavy_plugins, php_version, cache_enabled, cdn_enabled, wp_type, predicted_load, error_rate, saturation_days, saturation_months, saturation_jours, saturation_text, saturation_months_raw, status, recommendation, save_type, is_deleted) VALUES (:id, :user, :created_at, :cpu_usage_avg, :cpu_usage_peak, :ram_usage_avg, :ram_usage_max, :disk_usage_avg, :disk_usage_max, :disk_read_iops, :disk_write_iops, :response_time, :visitors_per_day, :pageviews_per_day, :traffic_growth_rate, :peak_hours_start, :peak_hours_end, :peak_hours, :plugin_count, :heavy_plugins, :php_version, :cache_enabled, :cdn_enabled, :wp_type, :predicted_load, :error_rate, :saturation_days, :saturation_months, :saturation_jours, :saturation_text, :saturation_months_raw, :status, :recommendation, :save_type, 0)");
            $stmt2->execute([':id' => $new_id, ':user' => $_SESSION['user'], ':created_at' => $pred['created_at'], ':cpu_usage_avg' => $pred['cpu_usage_avg'], ':cpu_usage_peak' => $pred['cpu_usage_peak'], ':ram_usage_avg' => $pred['ram_usage_avg'], ':ram_usage_max' => $pred['ram_usage_max'], ':disk_usage_avg' => $pred['disk_usage_avg'], ':disk_usage_max' => $pred['disk_usage_max'], ':disk_read_iops' => $pred['disk_read_iops'], ':disk_write_iops' => $pred['disk_write_iops'], ':response_time' => $pred['response_time'], ':visitors_per_day' => $pred['visitors_per_day'], ':pageviews_per_day' => $pred['pageviews_per_day'], ':traffic_growth_rate' => $pred['traffic_growth_rate'], ':peak_hours_start' => $pred['peak_hours_start'], ':peak_hours_end' => $pred['peak_hours_end'], ':peak_hours' => $pred['peak_hours'], ':plugin_count' => $pred['plugin_count'], ':heavy_plugins' => $pred['heavy_plugins'], ':php_version' => $pred['php_version'], ':cache_enabled' => $pred['cache_enabled'], ':cdn_enabled' => $pred['cdn_enabled'], ':wp_type' => $pred['wp_type'], ':predicted_load' => $pred['predicted_load'], ':error_rate' => $pred['error_rate'] ?? '', ':saturation_days' => $pred['saturation_days'] ?? '', ':saturation_months' => $pred['saturation_months'] ?? '', ':saturation_jours' => $pred['saturation_jours'] ?? '', ':saturation_text' => $pred['saturation_text'] ?? '', ':saturation_months_raw' => $pred['saturation_months_raw'] ?? '', ':status' => $pred['status'], ':recommendation' => $pred['recommendation'], ':save_type' => $pred['save_type'] ?? 'Manuel']);
            $pdo->prepare("DELETE FROM deleted_sauvegardes WHERE id = :id")->execute([':id' => $id]);
            return true; } return false; } catch (Exception $e) { return false; } }
function emptyTrash($pdo) { if ($pdo === null) return false; try { $pdo->prepare("DELETE FROM deleted_sauvegardes")->execute(); return true; } catch (Exception $e) { return false; } }

// Export CSV
if (isset($_GET['export_full_csv']) && isset($_SESSION['user'])) {
    $predictions = getPredictions($pdo);
    if (ob_get_level()) ob_end_clean();
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="vala_bleu_export_' . date('Y-m-d_His') . '.csv"');
    echo "\xEF\xBB\xBF";
    $output = fopen('php://output', 'w');
    fputcsv($output, ['Date','Heure','Pack','Visiteurs/j','Pages vues/j','Croissance (%)','CPU moyen (%)','CPU max (%)','RAM moyenne (%)','RAM max (%)','Disque moyen (%)','Disque max (%)','IOPS Read','IOPS Write','Temps réponse (ms)','Pic début','Pic fin','Nb plugins','Plugins lourds','PHP','Cache','CDN','Charge prédite (%)','Taux erreur (%)','Saturation (texte)','Saturation (jours)','Statut','Recommandation','Type','ID'], ';');
    foreach ($predictions as $p) { fputcsv($output, [date('d/m/Y', strtotime($p['created_at'] ?? '')), date('H:i:s', strtotime($p['created_at'] ?? '')), strtoupper($p['wp_type'] ?? ''), $p['visitors_per_day'] ?? '', $p['pageviews_per_day'] ?? '', $p['traffic_growth_rate'] ?? '', $p['cpu_usage_avg'] ?? '', $p['cpu_usage_peak'] ?? '', $p['ram_usage_avg'] ?? '', $p['ram_usage_max'] ?? '', $p['disk_usage_avg'] ?? '', $p['disk_usage_max'] ?? '', $p['disk_read_iops'] ?? '', $p['disk_write_iops'] ?? '', $p['response_time'] ?? '', $p['peak_hours_start'] ?? '', $p['peak_hours_end'] ?? '', $p['plugin_count'] ?? '', $p['heavy_plugins'] ?? '', $p['php_version'] ?? '', $p['cache_enabled'] ?? '', $p['cdn_enabled'] ?? '', $p['predicted_load'] ?? '', $p['error_rate'] ?? '', $p['saturation_text'] ?? '', $p['saturation_days'] ?? '', $p['status'] ?? '', '"' . str_replace('"', '""', $p['recommendation'] ?? '') . '"', $p['save_type'] ?? 'Manuel', $p['id'] ?? ''], ';'); }
    fclose($output);
    exit();
}

// Traitement AJAX
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_SERVER['HTTP_X_REQUESTED_WITH']) && $_SERVER['HTTP_X_REQUESTED_WITH'] === 'XMLHttpRequest') {
    if (ob_get_level()) ob_end_clean();
    header('Content-Type: application/json; charset=utf-8');
    $input = file_get_contents('php://input');
    $data = json_decode($input, true);
    function ajax_json_response($arr) { echo json_encode($arr); exit(); }
    if (!$data || !isset($_SESSION['user'])) { ajax_json_response(['success' => false, 'error' => 'Données invalides']); }
    try {
        if (isset($data['action'])) {
            if ($data['action'] === 'delete' && isset($data['delete_id'])) { ajax_json_response(['success' => deletePermanently($pdo, $data['delete_id'])]); }
            if ($data['action'] === 'archive' && isset($data['archive_id'])) { ajax_json_response(['success' => archivePrediction($pdo, $data['archive_id'])]); }
            if ($data['action'] === 'restore' && isset($data['restore_id'])) { ajax_json_response(['success' => restorePrediction($pdo, $data['restore_id'])]); }
            if ($data['action'] === 'empty_trash') { ajax_json_response(['success' => emptyTrash($pdo)]); }
            // Sauvegarde des résultats graphiques ou résultats d'analyse
            if ($data['action'] === 'save_graphs' && isset($data['graph_urls'])) {
                $ok = saveResultJson($pdo, $data['graph_urls'], 'graphiques');
                ajax_json_response(['success' => $ok]);
            }
            if ($data['action'] === 'save_result' && isset($data['result_data'])) {
                $ok = saveResultJson($pdo, $data['result_data'], 'resultat');
                ajax_json_response(['success' => $ok]);
            }
            if ($data['action'] === 'save_params' && isset($data['params_data'])) {
                $ok = saveResultJson($pdo, $data['params_data'], 'parametres');
                ajax_json_response(['success' => $ok]);
            }
        }
        if (isset($data['predicted_load'])) { $prediction = ['id' => uniqid(), 'created_at' => date('Y-m-d H:i:s'), 'cpu_usage_avg' => $data['cpu_usage_avg'] ?? '', 'cpu_usage_peak' => $data['cpu_usage_peak'] ?? '', 'ram_usage_avg' => $data['ram_usage_avg'] ?? '', 'ram_usage_max' => $data['ram_usage_max'] ?? '', 'disk_usage_avg' => $data['disk_usage_avg'] ?? '', 'disk_usage_max' => $data['disk_usage_max'] ?? '', 'disk_read_iops' => $data['disk_read_iops'] ?? '', 'disk_write_iops' => $data['disk_write_iops'] ?? '', 'response_time' => $data['response_time'] ?? '', 'visitors_per_day' => $data['visitors_per_day'] ?? '', 'pageviews_per_day' => $data['pageviews_per_day'] ?? '', 'traffic_growth_rate' => $data['traffic_growth_rate'] ?? '', 'peak_hours_start' => $data['peak_hours_start'] ?? '', 'peak_hours_end' => $data['peak_hours_end'] ?? '', 'peak_hours' => $data['peak_hours'] ?? '', 'plugin_count' => $data['plugin_count'] ?? '', 'heavy_plugins' => $data['heavy_plugins'] ?? '', 'php_version' => $data['php_version'] ?? '', 'cache_enabled' => $data['cache_enabled'] ?? '', 'cdn_enabled' => $data['cdn_enabled'] ?? '', 'wp_type' => $data['wp_type'] ?? '', 'predicted_load' => $data['predicted_load'] ?? '', 'error_rate' => $data['error_rate'] ?? '', 'saturation_days' => $data['saturation_days'] ?? '', 'saturation_months' => $data['saturation_months'] ?? '', 'saturation_jours' => $data['saturation_jours'] ?? '', 'saturation_text' => $data['saturation_text'] ?? '', 'saturation_months_raw' => $data['saturation_months_raw'] ?? '', 'status' => $data['status'] ?? '', 'recommendation' => $data['recommendation'] ?? '', 'save_type' => 'Manuel']; $errorMsg = null; ajax_json_response(['success' => savePrediction($pdo, $prediction, $errorMsg), 'error' => $errorMsg]); }
        ajax_json_response(['status' => 'success', 'message' => 'OK']);
    } catch (Exception $e) { ajax_json_response(['success' => false, 'error' => $e->getMessage()]); }
}

$history_predictions = getPredictions($pdo);
$deleted_sauvegardes = getDeletedSauvegardes($pdo);
$saved_results = getSavedResults($pdo);
$active_tab = isset($_GET['tab']) ? $_GET['tab'] : 'dashboard';
$valid_tabs = ['dashboard', 'resultats', 'graphiques', 'sauvegardes', 'historique', 'corbeille'];
if (!in_array($active_tab, $valid_tabs)) { $active_tab = 'dashboard'; }
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
    <style>
        .graph-block { background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
        .graph-block h3 { font-size: 1.2em; font-weight: 600; color: #1e293b; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid #f1f5f9; }
        .btn-create-graph { display: block; margin: 20px auto; background: linear-gradient(135deg, #10b981, #059669); color: #fff; padding: 14px 32px; font-size: 1.05em; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); transition: all 0.3s ease; }
        .btn-create-graph:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4); }
        .btn-create-graph:disabled { background: #94a3b8; cursor: not-allowed; box-shadow: none; transform: none; }
        .btn-create-graph.ready { background: linear-gradient(135deg, #3b82f6, #2563eb); cursor: pointer; }
        .no-graph-message { text-align: center; color: #94a3b8; font-size: 0.95em; padding: 16px; margin-top: 12px; background: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1; }
        .tree-links { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; justify-content: center; }
        .tree-link { display: inline-block; padding: 10px 20px; background: #f1f5f9; color: #334155; border-radius: 8px; text-decoration: none; font-weight: 500; font-size: 0.95em; transition: all 0.2s ease; border: 1px solid #e2e8f0; }
        .tree-link:hover { background: #3b82f6; color: #fff; border-color: #3b82f6; transform: translateY(-1px); }
        .btn-download-graph { display: inline-block; padding: 8px 18px; background: #f1f5f9; color: #334155; border-radius: 6px; text-decoration: none; font-size: 0.85em; transition: all 0.2s ease; }
        .btn-download-graph:hover { background: #3b82f6; color: #fff; }
        .graph-gallery-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 28px 24px; margin-top: 24px; }
        @media (max-width: 900px) { .graph-gallery-grid { grid-template-columns: 1fr; } }
        .graph-gallery-item { background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); display: flex; flex-direction: column; align-items: stretch; }
        .graph-gallery-title { margin: 0 0 8px 0; color: #1e293b; font-size: 1.1em; }
        .graph-gallery-desc { color: #64748b; font-size: 0.9em; margin-bottom: 16px; }
        .graph-gallery-img { width: 100%; max-width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; cursor: pointer; transition: box-shadow 0.2s; }
        .graph-gallery-img:hover { box-shadow: 0 0 0 3px #3b82f6; }
        .graph-gallery-download { margin-top: 10px; text-align: right; }
        .validation-error-msg { position: fixed; bottom: 20px; right: 20px; background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; padding: 14px 24px; border-radius: 10px; font-size: 0.95em; font-weight: 600; box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4); z-index: 9999; display: none; animation: slideInRight 0.4s ease-out; max-width: 350px; }
        .validation-error-msg.show { display: flex; align-items: center; gap: 10px; }
        .validation-error-msg .error-icon { font-size: 1.3em; }
        @keyframes slideInRight { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    </style>
</head>
<body>

<div id="toast" class="toast-notification"></div>
<div id="validationError" class="validation-error-msg">
    <span class="error-icon">⚠️</span>
    <span id="validationErrorText">Valeurs positives requises</span>
</div>

<div class="sidebar">
    <div class="sidebar-header"><h2>VALA BLEU</h2><p>Dashboard</p></div>
    <nav class="sidebar-nav">
        <div class="menu-item <?php echo $active_tab==='dashboard'?'active-menu':''; ?>" onclick="showTab('dashboard')"><span class="menu-icon">⚙️</span><span>Paramètres</span></div>
        <div class="menu-item <?php echo $active_tab==='resultats'?'active-menu':''; ?>" onclick="showTab('resultats')"><span class="menu-icon">📊</span><span>Résultats</span></div>
        <div class="menu-item <?php echo $active_tab==='graphiques'?'active-menu':''; ?>" onclick="showTab('graphiques')"><span class="menu-icon">📈</span><span>Graphiques</span></div>
        <div class="menu-item <?php echo $active_tab==='sauvegardes'?'active-menu':''; ?>" onclick="showTab('sauvegardes')"><span class="menu-icon">💾</span><span>Sauvegardes</span></div>
        <div class="menu-item <?php echo $active_tab==='historique'?'active-menu':''; ?>" onclick="showTab('historique')"><span class="menu-icon">📋</span><span>Historique</span></div>
        <div class="menu-item <?php echo $active_tab==='corbeille'?'active-menu':''; ?>" onclick="showTab('corbeille')"><span class="menu-icon">🗑️</span><span>Corbeille</span></div>
    </nav>
    <a href="?logout=1" class="logout-link"><span class="menu-icon">🚪</span><span>Déconnexion</span></a>
</div>

<div style="position:absolute;top:10px;right:20px;background:linear-gradient(135deg,#1e293b,#334155);color:#fff;padding:8px 20px;border-radius:8px;font-size:0.98rem;box-shadow:0 2px 16px rgba(0,0,0,0.22);z-index:2000;display:flex;align-items:center;gap:8px;">
    <span>👤 Connecté en tant que <strong><?php echo htmlspecialchars((string)($_SESSION['user'] ?? '')); ?></strong></span>
</div>

<div class="main-content">

    <!-- Onglet Paramètres -->
    <div id="dashboard" class="tab-content <?php echo $active_tab==='dashboard'?'active-tab':''; ?>" style="<?php echo $active_tab!=='dashboard'?'display:none;':''; ?>">
        <div class="page-title"><h1>Saisie des paramètres</h1><p>Configuration pour l'analyse de charge WordPress</p>
            <button id="resetBtn" type="button" style="background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;padding:10px 24px;font-size:0.95em;border:none;border-radius:7px;cursor:pointer;box-shadow:0 2px 8px rgba(59,130,246,0.15);font-weight:500;margin-top:12px;display:inline-flex;align-items:center;gap:8px;"><span>🔄</span> Réinitialiser</button>
        </div>
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

    <!-- Onglet Résultats -->
    <div id="resultats" class="tab-content <?php echo $active_tab==='resultats'?'active-tab':''; ?>" style="<?php echo $active_tab!=='resultats'?'display:none;':''; ?>">
        <div class="page-title"><h1>Résultats de l'analyse</h1><p>Prédiction basée sur les paramètres fournis</p></div>
        <div id="noResults" class="card empty-state" style="display:block;"><div class="empty-icon"><img src="icons/resultas.png" alt="Aucun résultat" style="width:64px;height:64px;"></div><h3>Aucune analyse générée</h3><p>Remplissez les paramètres et cliquez sur "LANCER L'ANALYSE"</p></div>
        <div id="loadingResults" class="card loading-card" style="display:none;"><div class="loading-spinner">⏳</div><p>Prédiction en cours...</p></div>
        <div id="resultsContainer" style="display:none;"><div class="card"><h3>📊 Scores de Performance</h3><div id="scoresDisplay"></div></div><div class="card"><h3>💡 Recommandation</h3><div id="recommendationDisplay"></div></div></div>
    </div>

    <!-- Onglet Graphiques -->
    <div id="graphiques" class="tab-content <?php echo $active_tab==='graphiques'?'active-tab':''; ?>" style="<?php echo $active_tab!=='graphiques'?'display:none;':''; ?>">
        <div class="page-title"><h1>📈 Graphiques</h1><p>Visualisation des performances et téléchargement des graphiques de training</p></div>
        <div class="graph-block"><h3>📊 Créer des graphiques</h3><p style="color:#64748b;margin-bottom:16px;">Générez des graphiques basés sur les résultats de votre dernière analyse.</p>
            <button class="btn-create-graph" id="btnCreateGraph" onclick="generateGraphsFromResults()">📊 Créer les graphiques</button>
            <div id="noGraphMessage" class="no-graph-message" style="display:block;">Aucun graphique généré</div>
            <div id="generatedGraphsContainer" style="display:none;"></div>
        </div>
        <div class="graph-block"><h3>📥 Télécharger les graphiques générés par le code de training</h3><p style="color:#64748b;margin-bottom:16px;">Graphiques générés automatiquement lors de l'entraînement du modèle.</p>
            <div class="tree-links">
                <a href="http://localhost:8000/api/download/graphe/tree0" class="tree-link" download>📥 Tree 0</a>
                <a href="http://localhost:8000/api/download/graphe/treefinal" class="tree-link" download>📥 Tree Final</a>
                <a href="http://localhost:8000/api/download/graphe/feature_importance" class="tree-link" download>⭐ Feature Importance</a>
                <a href="http://localhost:8000/api/download/graphe/learning_curve" class="tree-link" download>📈 Learning Curve</a>
                <a href="http://localhost:8000/api/download/graphe/residus" class="tree-link" download>📉 Résidus</a>
                <a href="http://localhost:8000/api/download/graphe/correlation" class="tree-link" download>🔗 Corrélation</a>
                <a href="http://localhost:8000/api/download/graphe/confusion_matrix" class="tree-link" download>🟦 Matrice de confusion</a>
            </div>
        </div>
    </div>

    <!-- Onglet Sauvegardes -->
    <div id="sauvegardes" class="tab-content <?php echo $active_tab==='sauvegardes'?'active-tab':''; ?>" style="<?php echo $active_tab!=='sauvegardes'?'display:none;':''; ?>">
        <div class="page-header-with-action"><div class="page-title"><h1>💾 Sauvegardes des résultats</h1><p>Résultats sauvegardés sans images</p></div></div>
        <div class="card" style="display:flex;justify-content:center;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px;"><button class="btn-primary btn-save" id="saveResultBtn" onclick="saveCurrentResult()"><span>💾</span> Sauvegarder dans l'historique</button></div>
    </div>

    <!-- Onglet Historique -->
    <div id="historique" class="tab-content <?php echo $active_tab==='historique'?'active-tab':''; ?>" style="<?php echo $active_tab!=='historique'?'display:none;':''; ?>">
        <div class="page-header-with-action"><div class="page-title"><h1>Historique des analyses</h1><p>Toutes les analyses sauvegardées</p></div></div>
        <div class="table-header-action multiple-buttons"><a href="?export_full_csv=1" class="btn-export-csv"><span>📥</span> Exporter tout en CSV</a></div>
        <div class="config-card"><div class="table-wrapper"><table class="history-table"><thead><tr><th>Date</th><th>Pack</th><th>Visiteurs/j</th><th>Croissance</th><th>CPU/RAM</th><th>Plugins</th><th>Charge</th><th>Taux</th><th>Statut</th><th>Action</th></tr></thead><tbody>
            <?php if (count($history_predictions)>0): foreach($history_predictions as $pr): ?>
                <tr><td class="td-date"><?php echo date('d/m/Y',strtotime($pr['created_at'])); ?><br><span style="font-size:11px;color:#888;"><?php echo date('H:i',strtotime($pr['created_at'])); ?></span></td><td><span class="badge-pack"><?php echo strtoupper($pr['wp_type']??'N/A'); ?></span></td><td class="td-number"><?php echo is_numeric($pr['visitors_per_day'])?number_format((float)$pr['visitors_per_day']):''; ?></td><td class="td-growth"><?php echo $pr['traffic_growth_rate']; ?>%</td><td class="td-usage"><?php echo $pr['cpu_usage_avg']; ?>% / <?php echo $pr['ram_usage_avg']; ?>%</td><td class="td-number"><?php echo $pr['plugin_count']; ?></td><td class="td-number"><?php $l=$pr['predicted_load']??null; echo (is_numeric($l)&&$l!==null)?number_format((float)$l,1).'%':'N/A'; ?></td><td class="td-saturation"><?php $er=$pr['error_rate']??null; echo (is_numeric($er)&&$er!==null)?number_format((float)$er,2).'%':'N/A'; ?></td><td><?php $st=$pr['status']??''; $bc=strpos($st,'CRITIQUE')!==false||strpos($st,'🔴')!==false?'badge-critical':(strpos($st,'URGENT')!==false||strpos($st,'🟠')!==false||strpos($st,'SURVEILLANCE')!==false||strpos($st,'🟡')!==false?'badge-warning':'badge-optimal'); ?><span class="badge-status <?php echo $bc; ?>"><?php echo $st?:'N/A'; ?></span></td><td><button class="btn-icon btn-archive" onclick="archiverAnalyse('<?php echo $pr['id']; ?>')" title="Archiver">📦</button></td></tr>
            <?php endforeach; else: ?>
                <tr><td colspan="10" class="empty-table-cell"><div class="empty-icon"><img src="icons/historique.png" alt="Aucun historique" style="width:64px;height:64px;"></div><p>Aucune analyse sauvegardée</p></td></tr>
            <?php endif; ?>
        </tbody></table></div></div>
    </div>

    <!-- Onglet Corbeille -->
    <div id="corbeille" class="tab-content <?php echo $active_tab==='corbeille'?'active-tab':''; ?>" style="<?php echo $active_tab!=='corbeille'?'display:none;':''; ?>">
        <div class="page-header-with-action"><div class="page-title"><h1>Corbeille</h1><p>Éléments supprimés</p></div></div>
        <div class="table-header-action multiple-buttons"><button class="btn-danger" onclick="viderCorbeille()"><span>🗑️</span> Vider la corbeille</button></div>
        <div class="config-card"><div class="table-wrapper"><table class="history-table"><thead><tr><th>Date</th><th>Pack</th><th>Visiteurs/j</th><th>Croissance</th><th>CPU/RAM</th><th>Plugins</th><th>Charge</th><th>Taux</th><th>Statut</th><th>Action</th></tr></thead><tbody>
            <?php if (count($deleted_sauvegardes)>0): foreach($deleted_sauvegardes as $dl): ?>
                <tr class="tr-deleted"><td class="td-date"><?php echo date('d/m/Y',strtotime($dl['created_at'])); ?><br><span style="font-size:11px;color:#888;"><?php echo date('H:i',strtotime($dl['created_at'])); ?></span></td><td><?php echo strtoupper($dl['wp_type']??'N/A'); ?></td><td class="td-number"><?php echo is_numeric($dl['visitors_per_day'])?number_format((float)$dl['visitors_per_day']):''; ?></td><td class="td-growth"><?php echo $dl['traffic_growth_rate']; ?>%</td><td class="td-usage"><?php echo $dl['cpu_usage_avg']; ?>% / <?php echo $dl['ram_usage_avg']; ?>%</td><td class="td-number"><?php echo $dl['plugin_count']; ?></td><td class="td-number"><?php $l=$dl['predicted_load']??null; echo (is_numeric($l)&&$l!==null)?number_format((float)$l,1).'%':'N/A'; ?></td><td class="td-saturation"><?php $er=$dl['error_rate']??null; echo (is_numeric($er)&&$er!==null)?number_format((float)$er,2).'%':'N/A'; ?></td><td><span class="badge-deleted">🗑️ Supprimé</span></td><td><button class="btn-icon btn-restore" onclick="restaurerAnalyse('<?php echo $dl['id']; ?>')" title="Restaurer">🔄</button><button class="btn-icon btn-delete-forever" onclick="supprimerDefinitivement('<?php echo $dl['id']; ?>')" title="Supprimer">❌</button></td></tr>
            <?php endforeach; else: ?>
                <tr><td colspan="10" class="empty-table-cell"><div class="empty-icon"><img src="icons/corbeille.png" alt="Corbeille vide" style="width:64px;height:64px;"></div><p>Corbeille vide</p></td></tr>
            <?php endif; ?>
        </tbody></table></div></div>
    </div>

</div>

<script>
var currentPrediction = null;
var savedGraphUrls = null;

if (window.history && window.history.replaceState) { window.history.replaceState({tab:'<?php echo $active_tab; ?>'},'',window.location.pathname+'?tab=<?php echo $active_tab; ?>'); }

function saveFormParamsToStorage() { sessionStorage.setItem('lastFormParams', JSON.stringify(getFormParams())); }

function restoreFormParamsFromStorage() {
    var s = sessionStorage.getItem('lastFormParams');
    if (s) { try { var p = JSON.parse(s); if(p.visitors_per_day) document.getElementById('visitors_per_day').value = p.visitors_per_day; if(p.pageviews_per_day) document.getElementById('pageviews_per_day').value = p.pageviews_per_day; if(p.traffic_growth_rate) document.getElementById('traffic_growth_rate').value = p.traffic_growth_rate; if(p.cpu_usage_avg) document.getElementById('cpu_usage_avg').value = p.cpu_usage_avg; if(p.cpu_usage_peak) document.getElementById('cpu_usage_peak').value = p.cpu_usage_peak; if(p.ram_usage_avg) document.getElementById('ram_usage_avg').value = p.ram_usage_avg; if(p.ram_usage_max) document.getElementById('ram_usage_max').value = p.ram_usage_max; if(p.disk_usage_avg) document.getElementById('disk_usage_avg').value = p.disk_usage_avg; if(p.disk_usage_max) document.getElementById('disk_usage_max').value = p.disk_usage_max; if(p.disk_read_iops) document.getElementById('disk_read_iops').value = p.disk_read_iops; if(p.disk_write_iops) document.getElementById('disk_write_iops').value = p.disk_write_iops; if(p.response_time) document.getElementById('response_time').value = p.response_time; if(p.plugin_count) document.getElementById('plugin_count').value = p.plugin_count; if(p.peak_hours_start) document.getElementById('peak_hours_start').value = p.peak_hours_start; if(p.peak_hours_end) document.getElementById('peak_hours_end').value = p.peak_hours_end; if(p.php_version&&p.php_version!=='none') document.getElementById('php_version').value = p.php_version; if(p.cache_enabled&&p.cache_enabled!=='none') document.getElementById('cache_enabled').value = p.cache_enabled; if(p.cdn_enabled&&p.cdn_enabled!=='none') document.getElementById('cdn_enabled').value = p.cdn_enabled; if(p.wp_type&&p.wp_type!=='none') document.getElementById('wp_type').value = p.wp_type; if(p.heavy_plugins) { var pl=p.heavy_plugins.split(','); var c=document.querySelectorAll('#heavy_plugins_group input[type="checkbox"]'); for(var i=0;i<c.length;i++) c[i].checked=pl.indexOf(c[i].value)!==-1; } } catch(e) {} }
}

function showTab(tabId) {
    var tabs = document.querySelectorAll('.tab-content'); for(var i=0;i<tabs.length;i++){tabs[i].classList.remove('active-tab');tabs[i].style.display='none';}
    var menus = document.querySelectorAll('.menu-item'); for(var i=0;i<menus.length;i++) menus[i].classList.remove('active-menu');
    var target = document.getElementById(tabId); if(target){target.classList.add('active-tab');target.style.display='block';}
    var tabNames=['dashboard','resultats','graphiques','sauvegardes','historique','corbeille'];
    var index=tabNames.indexOf(tabId); if(index>=0&&menus[index]) menus[index].classList.add('active-menu');
    sessionStorage.setItem('activeTab',tabId);
    window.history.pushState({tab:tabId},'',window.location.pathname+'?tab='+tabId);
}

window.addEventListener('popstate',function(event){
    if(event.state&&event.state.tab) showTab(event.state.tab);
    else{var urlParams=new URLSearchParams(window.location.search);var tabFromUrl=urlParams.get('tab');var validTabs=['dashboard','resultats','graphiques','sauvegardes','historique','corbeille'];if(tabFromUrl&&validTabs.indexOf(tabFromUrl)!==-1) showTab(tabFromUrl);else showTab('dashboard');}
});

function showToast(message,isError){
    isError=isError||false;var toast=document.getElementById('toast');toast.textContent=message;toast.style.display='block';
    toast.style.background=isError?'linear-gradient(135deg, #ef4444, #dc2626)':'linear-gradient(135deg, #10b981, #059669)';
    setTimeout(function(){toast.style.display='none';},4000);
}

function showValidationError(message){
    var errMsg=document.getElementById('validationError');
    document.getElementById('validationErrorText').textContent=message;
    errMsg.classList.add('show');
    setTimeout(function(){errMsg.classList.remove('show');},5000);
}

function getFormParams(){
    var start=document.getElementById('peak_hours_start').value,end=document.getElementById('peak_hours_end').value,peakHours=4;
    if(start&&end) peakHours=Math.max(1,parseInt(end.split(':')[0])-parseInt(start.split(':')[0]));
    var heavyPlugins=[];var cbs=document.querySelectorAll('#heavy_plugins_group input[type="checkbox"]:checked');
    for(var i=0;i<cbs.length;i++) heavyPlugins.push(cbs[i].value);
    return {
        cpu_usage_avg:parseFloat(document.getElementById('cpu_usage_avg').value)||0,
        cpu_usage_peak:parseFloat(document.getElementById('cpu_usage_peak').value)||0,
        ram_usage_avg:parseFloat(document.getElementById('ram_usage_avg').value)||0,
        ram_usage_max:parseFloat(document.getElementById('ram_usage_max').value)||0,
        disk_usage_avg:parseFloat(document.getElementById('disk_usage_avg').value)||0,
        disk_usage_max:parseFloat(document.getElementById('disk_usage_max').value)||0,
        disk_read_iops:parseFloat(document.getElementById('disk_read_iops').value)||0,
        disk_write_iops:parseFloat(document.getElementById('disk_write_iops').value)||0,
        response_time:parseFloat(document.getElementById('response_time').value)||0,
        visitors_per_day:parseFloat(document.getElementById('visitors_per_day').value)||0,
        pageviews_per_day:parseFloat(document.getElementById('pageviews_per_day').value)||0,
        traffic_growth_rate:parseFloat(document.getElementById('traffic_growth_rate').value)||0,
        peak_hours_start:start,peak_hours_end:end,peak_hours:peakHours,
        plugin_count:parseFloat(document.getElementById('plugin_count').value)||0,
        heavy_plugins:heavyPlugins.join(','),
        php_version:document.getElementById('php_version').value,
        cache_enabled:document.getElementById('cache_enabled').value,
        cdn_enabled:document.getElementById('cdn_enabled').value,
        wp_type:document.getElementById('wp_type').value
    };
}

function validateParams(params){
    var errMsg=document.getElementById('validationError');
    errMsg.classList.remove('show');
    
    // Réinitialiser les styles
    var allInputs=document.querySelectorAll('input[type="number"]');
    for(var i=0;i<allInputs.length;i++){allInputs[i].style.borderColor='';allInputs[i].style.boxShadow='';allInputs[i].style.backgroundColor='';}
    
    // Vérifier les champs requis
    if(!params.cpu_usage_avg||!params.cpu_usage_peak||!params.ram_usage_avg||!params.ram_usage_max||!params.visitors_per_day||!params.traffic_growth_rate||!params.plugin_count||!params.wp_type||params.wp_type==='none'){
        showToast('Champs * requis',true);
        return false;
    }
    
    // Vérifier les valeurs négatives - BLOQUER l'exécution
    var numericFields=[
        {id:'visitors_per_day',value:params.visitors_per_day,name:'Visiteurs / jour'},
        {id:'pageviews_per_day',value:params.pageviews_per_day,name:'Pages vues / jour'},
        {id:'traffic_growth_rate',value:params.traffic_growth_rate,name:'Taux de croissance'},
        {id:'cpu_usage_avg',value:params.cpu_usage_avg,name:'CPU moyen'},
        {id:'cpu_usage_peak',value:params.cpu_usage_peak,name:'CPU max'},
        {id:'ram_usage_avg',value:params.ram_usage_avg,name:'RAM moyenne'},
        {id:'ram_usage_max',value:params.ram_usage_max,name:'RAM max'},
        {id:'disk_usage_avg',value:params.disk_usage_avg,name:'Disque utilisé'},
        {id:'disk_usage_max',value:params.disk_usage_max,name:'Disque max'},
        {id:'response_time',value:params.response_time,name:'Temps réponse'},
        {id:'disk_read_iops',value:params.disk_read_iops,name:'IOPS Read'},
        {id:'disk_write_iops',value:params.disk_write_iops,name:'IOPS Write'},
        {id:'plugin_count',value:params.plugin_count,name:'Nombre de plugins'}
    ];
    
    var hasNegative=false,firstNegativeField=null;
    for(var i=0;i<numericFields.length;i++){
        var field=numericFields[i];
        if(!isNaN(field.value)&&field.value<0){
            hasNegative=true;
            if(!firstNegativeField) firstNegativeField=field;
            var inputEl=document.getElementById(field.id);
            if(inputEl&&inputEl.value!==''){
                inputEl.style.borderColor='#ef4444';
                inputEl.style.boxShadow='0 0 0 3px rgba(239,68,68,0.2)';
                inputEl.style.backgroundColor='#fef2f2';
            }
        }
    }
    
    if(hasNegative){
        showValidationError('Entrez uniquement des valeurs positives');
        if(firstNegativeField){
            var inputEl=document.getElementById(firstNegativeField.id);
            if(inputEl) inputEl.focus();
        }
        return false;
    }
    
    return true;
}

function viderTousLesChamps(){
    var inputs=document.querySelectorAll('input[type="number"], input[type="time"]');for(var i=0;i<inputs.length;i++)inputs[i].value='';
    var cbs=document.querySelectorAll('#heavy_plugins_group input[type="checkbox"]');for(var i=0;i<cbs.length;i++)cbs[i].checked=false;
    var selects=document.querySelectorAll('select');for(var i=0;i<selects.length;i++)selects[i].selectedIndex=0;
}

function resetResultsAndGraphsForNewPrediction(){
    currentPrediction = null;
    savedGraphUrls = null;
    sessionStorage.removeItem('lastPrediction');
    sessionStorage.removeItem('savedGraphUrls');

    document.getElementById('noResults').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('loadingResults').style.display = 'block';
    document.getElementById('scoresDisplay').innerHTML = '';
    document.getElementById('recommendationDisplay').innerHTML = '';

    document.getElementById('noGraphMessage').style.display = 'block';
    document.getElementById('generatedGraphsContainer').style.display = 'none';
    document.getElementById('generatedGraphsContainer').innerHTML = '';
    document.getElementById('btnCreateGraph').disabled = false;
    document.getElementById('btnCreateGraph').textContent = '📊 Créer les graphiques';
    document.getElementById('btnCreateGraph').classList.remove('ready');
}

function resetCurrentAnalysis(){
    if(!confirm("⚠️ Voulez-vous réinitialiser TOUT ?\n\nCette action va :\n- Effacer les résultats affichés\n- Vider les champs du formulaire\n- Supprimer TOUS les graphiques générés\n- Supprimer tous les paramètres sauvegardés\n\nLes sauvegardes dans l'historique ne seront pas supprimées.")) return;
    var btn=document.getElementById('resetBtn');btn.disabled=true;btn.innerHTML='⏳ Réinitialisation...';btn.style.opacity='0.7';
    viderTousLesChamps();
    sessionStorage.removeItem('lastPrediction');sessionStorage.removeItem('lastFormParams');sessionStorage.removeItem('activeTab');sessionStorage.removeItem('savedGraphUrls');
    currentPrediction=null;savedGraphUrls=null;
    document.getElementById('noResults').style.display='block';document.getElementById('resultsContainer').style.display='none';document.getElementById('loadingResults').style.display='none';
    document.getElementById('noGraphMessage').style.display='block';document.getElementById('generatedGraphsContainer').style.display='none';document.getElementById('generatedGraphsContainer').innerHTML='';
    document.getElementById('btnCreateGraph').disabled=false;document.getElementById('btnCreateGraph').textContent='📊 Créer les graphiques';
    document.getElementById('btnCreateGraph').classList.add('ready');
    fetch('http://localhost:8000/api/reset/parameters-and-images',{method:'POST',headers:{'Content-Type':'application/json'}})
    .then(function(r){return r.json();})
    .then(function(data){
        if(data.status==='success'){showToast('✅ Tout est réinitialisé !');}
        else{showToast('⚠️ '+(data.message||'Erreur'),true);}
        showTab('dashboard');btn.disabled=false;btn.innerHTML='<span>🔄</span> Réinitialiser';btn.style.opacity='1';
    })
    .catch(function(err){showToast('⚠️ Erreur API');btn.disabled=false;btn.innerHTML='<span>🔄</span> Réinitialiser';btn.style.opacity='1';});
}

function generateGraphsFromResults() {
    if (!currentPrediction) { showToast('⚠️ Aucun résultat disponible. Lancez d\'abord une analyse.', true); return; }
    var btn = document.getElementById('btnCreateGraph');
    btn.disabled = true; btn.textContent = '⏳ Génération en cours...'; btn.classList.remove('ready');
    document.getElementById('noGraphMessage').style.display = 'none';
    document.getElementById('generatedGraphsContainer').style.display = 'none';
    fetch('http://localhost:8000/api/generate/analysis-graphs', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        if (data.status === 'success') {
            var graphCount = Object.keys(data.graphs || {}).length;
            savedGraphUrls = data.graph_urls;
            sessionStorage.setItem('savedGraphUrls', JSON.stringify(data.graph_urls));
            displayGeneratedGraphs(data);
            btn.disabled = false;
            btn.textContent = '🔄 Recréer les graphiques';
            btn.classList.add('ready');
            showToast('✅ ' + graphCount + ' graphiques générés avec succès !');
            // Sauvegarde automatique des graphiques dans la base
            fetch('dashboard.php',{
                method:'POST',
                headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},
                body:JSON.stringify({action:'save_graphs',graph_urls:data.graph_urls})
            });
        } else {
            showToast('❌ ' + (data.message || 'Erreur lors de la génération'), true);
            btn.disabled = false;
            btn.textContent = '📊 Créer les graphiques';
            btn.classList.add('ready');
            document.getElementById('noGraphMessage').style.display = 'block';
        }
    })
    .catch(function(error) {
        showToast('❌ API indisponible.', true);
        btn.disabled = false;
        btn.textContent = '📊 Créer les graphiques';
        btn.classList.add('ready');
        document.getElementById('noGraphMessage').style.display = 'block';
    });
}

function displayGeneratedGraphs(data){
    var container=document.getElementById('generatedGraphsContainer');
    var gn={'radar_resources':'📊 Radar des Ressources','gauges_saturation':'📈 Jauges de Saturation','feature_impact':'📊 Impact des Features','degradation_curve':'📉 Courbe de Dégradation Temporelle'};
    var gd={'radar_resources':'Équilibre global du serveur et détection des goulots d\'étranglement','gauges_saturation':'Niveau de criticité immédiat de chaque ressource','feature_impact':'Quels paramètres réduisent ou augmentent la charge','degradation_curve':'Projection dans le futur avec 3 scénarios (optimiste, réaliste, pessimiste)'};
    var html='<div class="graph-gallery-grid">';
    if(data.graph_urls){for(var gk in data.graph_urls){var url=data.graph_urls[gk],dn=gn[gk]||gk,de=gd[gk]||'';html+='<div class="graph-gallery-item"><h4 class="graph-gallery-title">'+dn+'</h4><p class="graph-gallery-desc">'+de+'</p><a href="'+url+'" target="_blank" rel="noopener"><img src="'+url+'" alt="'+dn+'" class="graph-gallery-img"></a><div class="graph-gallery-download"><a href="'+url+'" download class="btn-download-graph">📥 Télécharger</a></div></div>';}}
    html+='</div>';container.innerHTML=html;container.style.display='block';
}

function displayResults(data){
    var sat=data.saturation_text||'N/A',st=data.status||'',cls='optimal',col='#10b981';
    if(st.indexOf('CRITIQUE')!==-1||st.indexOf('🔴')!==-1){cls='critical';col='#dc2626';}
    else if(st.indexOf('URGENT')!==-1||st.indexOf('🟠')!==-1){cls='warning';col='#d97706';}
    else if(st.indexOf('SURVEILLANCE')!==-1||st.indexOf('🟡')!==-1){cls='warning';col='#d97706';}
    var pl=parseFloat(data.predicted_load)||0;
    var sh='<div class="scores-grid"><div class="score-item"><div class="score-label">Charge prédite</div><div class="score-value" style="color:'+col+'">'+pl.toFixed(1)+'%</div></div><div class="score-item"><div class="score-label">Saturation estimée</div><div class="score-value saturation">'+sat+'</div>';
    if(data.saturation_days!==undefined) sh+='<div style="font-size:0.85em;color:#888;margin-top:4px;">'+parseFloat(data.saturation_days).toFixed(0)+' jours</div>';
    // Afficher le statut sans les pastilles (🟡, 🔴, etc.)
    var stText = st.replace(/[🟡🔴🟢]/g, '').trim();
    sh+='<span class="badge-status badge-'+cls+'" style="margin-top:8px;">'+stText+'</span></div>';
    if(data.error_rate!==undefined) sh+='<div class="score-item"><div class="score-label">Taux d\'erreur estimé</div><div class="score-value" style="color:#f59e0b">'+parseFloat(data.error_rate).toFixed(1)+'%</div></div>';
    sh+='</div>';document.getElementById('scoresDisplay').innerHTML=sh;
    var rh = '';
    var badge = '';
    if (cls === 'critical') {
        badge = '🔴 ';
        rh += '<div class="recommendation-global" style="color:#dc2626;font-weight:600;margin-bottom:12px;"></div>';
    } else if (cls === 'warning') {
        badge = '🟡 ';
        rh += '<div class="recommendation-global" style="color:#d97706;font-weight:600;margin-bottom:12px;"></div>';
    } else {
        badge = '🟢 ';
    }
    var rec = data.recommendation || 'Aucune recommandation spécifique.';
    // Supprimer la phrase "- Marge de ... avant saturation"
    if (typeof rec === 'string') {
        rec = rec.replace(/-?\s*Marge de [^<]*avant saturation/g, '');
        rec = rec.replace(/-?\s*Marge de [^<]*avant saturation/gi, '');
        rec = rec.replace(/-?\s*Marge de [^<]*avant saturation\.?/gi, '');
    }
    rh += '<div class="recommendation-text" style="font-size:1.05em;line-height:1.6;">' + badge + rec + '</div>';
    document.getElementById('recommendationDisplay').innerHTML = rh;
    
    // NE PAS cacher les graphiques s'ils existent déjà
    if (savedGraphUrls && Object.keys(savedGraphUrls).length > 0) {
        document.getElementById('btnCreateGraph').disabled = false;
        document.getElementById('btnCreateGraph').textContent = '🔄 Recréer les graphiques';
        document.getElementById('btnCreateGraph').classList.add('ready');
        document.getElementById('noGraphMessage').style.display = 'none';
        document.getElementById('generatedGraphsContainer').style.display = 'block';
    } else {
        document.getElementById('btnCreateGraph').disabled = false;
        document.getElementById('btnCreateGraph').textContent = '📊 Créer les graphiques';
        document.getElementById('btnCreateGraph').classList.add('ready');
        document.getElementById('noGraphMessage').style.display = 'block';
        document.getElementById('generatedGraphsContainer').style.display = 'none';
    }
}

function runAnalysis(){
    var params=getFormParams();if(!validateParams(params))return;
    saveFormParamsToStorage();
    resetResultsAndGraphsForNewPrediction();
    showTab('resultats');
    fetch("http://localhost:8000/api/save-and-predict-json",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(params)})
    .then(function(r){return r.json();})
    .then(function(res){
        document.getElementById('loadingResults').style.display='none';
        if(res.status==="success"&&res.prediction&&res.prediction.output){
            currentPrediction=res.prediction.output.result;
            displayResults(currentPrediction);
            document.getElementById('resultsContainer').style.display='block';
            document.getElementById('noResults').style.display='none';
            showToast('✅ Prédiction terminée !');
            sessionStorage.setItem('lastPrediction',JSON.stringify(currentPrediction));
            // Sauvegarde automatique des paramètres
            fetch('dashboard.php',{
                method:'POST',
                headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},
                body:JSON.stringify({action:'save_params',params_data:params})
            });
            // Sauvegarde automatique du résultat
            fetch('dashboard.php',{
                method:'POST',
                headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},
                body:JSON.stringify({action:'save_result',result_data:currentPrediction})
            });
        }
        else{showToast('❌ Erreur de prédiction',true);document.getElementById('resultsContainer').style.display='block';}
    })
    .catch(function(err){document.getElementById('loadingResults').style.display='none';showToast('❌ API indisponible',true);});
}

function saveCurrentResult(){
    if(!currentPrediction){showToast('⚠️ Aucun résultat à sauvegarder',true);return;}
    var btn=document.getElementById('saveResultBtn');btn.disabled=true;btn.innerHTML='⏳ Sauvegarde en cours...';saveFormParamsToStorage();
    fetch(window.location.href,{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},body:JSON.stringify(Object.assign({},getFormParams(),currentPrediction))})
    .then(function(r){return r.json();})
    .then(function(res){if(res.success){showToast('✅ Analyse sauvegardée !');setTimeout(function(){window.location.reload();},1000);}else{showToast('❌ '+(res.error||'Erreur'),true);btn.disabled=false;btn.innerHTML='<span>💾</span> Sauvegarder dans l\'historique';}})
    .catch(function(e){showToast('❌ Erreur de connexion',true);btn.disabled=false;btn.innerHTML='<span>💾</span> Sauvegarder dans l\'historique';});
}

function archiverAnalyse(id){if(confirm('📦 Archiver cette analyse ?'))ajaxAction('archive',id);}
function restaurerAnalyse(id){if(confirm('🔄 Restaurer cette analyse ?'))ajaxAction('restore',id);}
function supprimerDefinitivement(id){if(confirm('❌ Supprimer définitivement ?'))ajaxAction('delete',id);}
function viderCorbeille(){if(confirm('🗑️ Vider toute la corbeille ?'))ajaxAction('empty_trash');}

function ajaxAction(action,id){
    var body={action:action};if(id)body[action+'_id']=id;
    fetch(window.location.href,{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'XMLHttpRequest'},body:JSON.stringify(body)})
    .then(function(r){return r.json();})
    .then(function(res){if(res.success)showToast('✅ Opération réussie !');else showToast('❌ '+(res.error||'Erreur'),true);setTimeout(function(){window.location.reload();},1000);})
    .catch(function(e){showToast('❌ Erreur de connexion',true);});
}

document.getElementById('resetBtn').addEventListener('click',function(e){e.preventDefault();resetCurrentAnalysis();});

document.addEventListener('DOMContentLoaded',function(){
    restoreFormParamsFromStorage();
    
    // D'ABORD restaurer les graphiques depuis sessionStorage
    var savedGraphs = sessionStorage.getItem('savedGraphUrls');
    if (savedGraphs) {
        try {
            savedGraphUrls = JSON.parse(savedGraphs);
            displayGeneratedGraphs({ graph_urls: savedGraphUrls });
            document.getElementById('btnCreateGraph').disabled = false;
            document.getElementById('btnCreateGraph').textContent = '🔄 Recréer les graphiques';
            document.getElementById('btnCreateGraph').classList.add('ready');
            document.getElementById('noGraphMessage').style.display = 'none';
        } catch(e) {
            savedGraphUrls = null;
        }
    }
    
    // ENSUITE restaurer la prédiction
    var last = sessionStorage.getItem('lastPrediction');
    if (last) {
        try {
            currentPrediction = JSON.parse(last);
            document.getElementById('noResults').style.display = 'none';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('loadingResults').style.display = 'none';
            displayResults(currentPrediction);
        } catch(e) {
            currentPrediction = null;
        }
    }
});
</script>
<script>
// Suppression automatique des résultats et graphiques uniquement à la fermeture de l’onglet (pas lors d’un F5/rechargement)
window.addEventListener('beforeunload', function(e) {
    var nav = performance.getEntriesByType("navigation")[0];
    if (!nav || nav.type !== "reload") {
        sessionStorage.removeItem('lastPrediction');
        sessionStorage.removeItem('savedGraphUrls');
        sessionStorage.removeItem('lastFormParams');
        sessionStorage.removeItem('activeTab');
    }
});
</script>
</body>
</html>

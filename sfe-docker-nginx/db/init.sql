-- =====================================================
-- Initialisation base de données
-- Projet : VALA WordPress Infrastructure Intelligence
-- =====================================================

-- Création base si inexistante
CREATE DATABASE IF NOT EXISTS vala_monitor
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE vala_monitor;

-- =====================================================
-- Table : server_metrics
-- Données trafic + ressources serveur (time series)
-- =====================================================
CREATE TABLE server_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    collected_at DATETIME NOT NULL,

    -- Trafic
    visitors_per_day INT NOT NULL,
    pageviews_per_day INT NOT NULL,
    traffic_growth_rate FLOAT NOT NULL,
    peak_hours VARCHAR(50),

    -- Ressources serveur
    cpu_usage_avg FLOAT NOT NULL,
    cpu_usage_peak FLOAT NOT NULL,
    ram_usage_avg FLOAT NOT NULL,
    disk_io FLOAT NOT NULL,
    response_time FLOAT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- =====================================================
-- Table : wordpress_config
-- Paramètres WordPress influençant la performance
-- =====================================================
CREATE TABLE wordpress_config (
    id INT AUTO_INCREMENT PRIMARY KEY,

    plugin_count INT NOT NULL,
    heavy_plugins BOOLEAN NOT NULL,
    php_version VARCHAR(10) NOT NULL,
    cache_enabled BOOLEAN NOT NULL,
    cdn_enabled BOOLEAN NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- =====================================================
-- Table : growth_history
-- Historique & contexte décisionnel
-- =====================================================
CREATE TABLE growth_history (
    id INT AUTO_INCREMENT PRIMARY KEY,

    avg_growth_30d FLOAT NOT NULL,
    avg_growth_90d FLOAT NOT NULL,
    previous_plan ENUM('Small', 'Medium', 'Large') NOT NULL,
    incident_count INT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- =====================================================
-- Table : prediction_results
-- Résultats du modèle ML (XGBoost)
-- =====================================================
CREATE TABLE prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY,

    prediction_date DATETIME NOT NULL,
    upgrade_required BOOLEAN NOT NULL,
    confidence_score FLOAT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- =====================================================
-- Index pour performances (time series & ML)
-- =====================================================
CREATE INDEX idx_server_metrics_time
ON server_metrics (collected_at);

CREATE INDEX idx_prediction_date
ON prediction_results (prediction_date);
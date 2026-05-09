from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
GRAPHE_DIR = BASE_DIR / "graphe"

DATASET_PATH = DATA_DIR / "training_dataset.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = DATA_DIR / "model_metrics_all.json"

RANDOM_STATE = 42
N_SAMPLES = 100_000
BAR_COLOR = "#4472C4"

HEAVY_PLUGIN_OPTIONS = [
    "woocommerce",
    "elementor",
    "wpml",
    "yoast",
    "revslider",
    "gravityforms",
]

PHP_VERSIONS = ["7.4", "8.0", "8.1", "8.2", "8.3"]
WP_TYPES = ["small", "medium", "performance"]

FEATURE_LABELS = {
    "visitors_per_day": "Visiteurs / jour",
    "pageviews_per_day": "Pages vues / jour",
    "traffic_growth_rate": "Taux de croissance (%)",
    "peak_hours_start": "Pic début (heure)",
    "peak_hours_end": "Pic fin (heure)",
    "cpu_usage_avg": "CPU moyen (%)",
    "cpu_usage_peak": "CPU max (%)",
    "ram_usage_avg": "RAM moyenne (%)",
    "ram_usage_max": "RAM max (%)",
    "disk_usage_avg": "Disque utilisé (%)",
    "disk_usage_max": "Disque max (%)",
    "response_time": "Temps réponse (ms)",
    "disk_read_iops": "IOPS Read",
    "disk_write_iops": "IOPS Write",
    "plugin_count": "Nombre de plugins",
    "heavy_plugins": "Plugins lourds",
    "php_version": "Version PHP",
    "cache_enabled": "Cache activé",
    "cdn_enabled": "CDN activé",
    "wp_type": "Pack WordPress",
}

FEATURE_ORDER = list(FEATURE_LABELS.keys())

IMPACT_REFERENCE = {
    "visitors_per_day": {"rank": 1, "impact": 5, "detail": "#1 ABSOLU - Le trafic determine tout"},
    "cpu_usage_avg": {"rank": 2, "impact": 5, "detail": "Charge processeur constante"},
    "ram_usage_avg": {"rank": 3, "impact": 5, "detail": "Memoire utilisee en continu"},
    "wp_type": {"rank": 4, "impact": 4, "detail": "small/medium/performance"},
    "traffic_growth_rate": {"rank": 5, "impact": 4, "detail": "+5% vs +25% par mois"},
    "cpu_usage_peak": {"rank": 6, "impact": 4, "detail": "Les pics CPU sont critiques"},
    "plugin_count": {"rank": 7, "impact": 4, "detail": "Chaque plugin = requetes SQL"},
    "ram_usage_max": {"rank": 8, "impact": 4, "detail": "Les pics RAM provoquent des SWAP"},
    "heavy_plugins": {"rank": 9, "impact": 4, "detail": "WooCommerce/Elementor"},
    "cache_enabled": {"rank": 10, "impact": 4, "detail": "OUI = -40% charge"},
    "php_version": {"rank": 11, "impact": 3, "detail": "PHP 8.2 = 30% plus rapide"},
    "disk_usage_avg": {"rank": 12, "impact": 3, "detail": ">85% = ralentissement"},
    "disk_usage_max": {"rank": 13, "impact": 2, "detail": "Pics d'ecriture backups"},
    "cdn_enabled": {"rank": 14, "impact": 2, "detail": "Decharge le serveur"},
    "disk_write_iops": {"rank": 15, "impact": 2, "detail": "Ecritures disque"},
    "disk_read_iops": {"rank": 16, "impact": 2, "detail": "Lectures disque"},
    "response_time": {"rank": 17, "impact": 2, "detail": "Symptome de charge"},
    "pageviews_per_day": {"rank": 18, "impact": 2, "detail": "Impact indirect"},
    "peak_hours_start": {"rank": 19, "impact": 1, "detail": "Heure debut pic"},
    "peak_hours_end": {"rank": 20, "impact": 1, "detail": "Heure fin pic"},
}


def ensure_directories() -> None:
    for directory in (DATA_DIR, MODELS_DIR, GRAPHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def random_time(rng: np.random.Generator, size: int, start_hour: int, end_hour: int) -> np.ndarray:
    hours = rng.integers(start_hour, end_hour + 1, size=size)
    minutes = rng.choice([0, 15, 30, 45], size=size)
    return np.array([f"{hour:02d}:{minute:02d}" for hour, minute in zip(hours, minutes)])


def generate_training_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    visitors_per_day = rng.integers(50, 150_001, size=n_samples)
    pageviews_per_day = np.maximum(
        visitors_per_day * rng.uniform(1.2, 6.5, size=n_samples),
        rng.integers(80, 600, size=n_samples),
    ).astype(int)
    traffic_growth_rate = rng.uniform(0, 85, size=n_samples).round(2)

    plugin_count = rng.integers(1, 81, size=n_samples)
    heavy_plugin_matrix = rng.binomial(
        1,
        [0.35, 0.45, 0.18, 0.42, 0.16, 0.14],
        size=(n_samples, 6),
    )
    heavy_plugin_count = heavy_plugin_matrix.sum(axis=1)

    php_version = rng.choice(PHP_VERSIONS, size=n_samples, p=[0.08, 0.16, 0.25, 0.31, 0.20])
    cache_enabled = rng.choice(["oui", "non"], size=n_samples, p=[0.68, 0.32])
    cdn_enabled = rng.choice(["oui", "non"], size=n_samples, p=[0.56, 0.44])
    wp_type = rng.choice(WP_TYPES, size=n_samples, p=[0.38, 0.42, 0.20])

    cache_factor = np.where(cache_enabled == "oui", 0.72, 1.18)
    cdn_factor = np.where(cdn_enabled == "oui", 0.82, 1.08)
    wp_factor = pd.Series(wp_type).map(
        {"small": 0.85, "medium": 1.0, "performance": 1.22}
    ).to_numpy()
    php_factor = pd.Series(php_version).map(
        {"7.4": 1.16, "8.0": 1.08, "8.1": 1.02, "8.2": 0.96, "8.3": 0.92}
    ).to_numpy()

    traffic_pressure = visitors_per_day / 150_000
    pageview_pressure = pageviews_per_day / 900_000
    growth_pressure = traffic_growth_rate / 85
    plugin_pressure = plugin_count / 80
    heavy_plugin_pressure = heavy_plugin_count / len(HEAVY_PLUGIN_OPTIONS)

    load_index = (
        (traffic_pressure * 42)
        + (growth_pressure * 19)
        + (plugin_pressure * 16)
        + (heavy_plugin_pressure * 18)
        + (pageview_pressure * 8)
    ) * cache_factor * cdn_factor * wp_factor * php_factor

    noise = rng.normal(0, 5.5, size=n_samples)

    cpu_usage_avg = np.clip(18 + load_index * 0.78 + noise, 5, 96).round(2)
    cpu_usage_peak = np.clip(cpu_usage_avg + rng.uniform(8, 32, size=n_samples), 8, 100).round(2)
    ram_usage_avg = np.clip(22 + load_index * 0.64 + plugin_count * 0.32 + noise, 8, 97).round(2)
    ram_usage_max = np.clip(ram_usage_avg + rng.uniform(8, 28, size=n_samples), 12, 100).round(2)
    disk_usage_avg = np.clip(rng.normal(42, 17, size=n_samples) + plugin_count * 0.24, 5, 96).round(2)
    disk_usage_max = np.clip(disk_usage_avg + rng.uniform(4, 24, size=n_samples), 8, 100).round(2)
    response_time = np.clip(
        95 + load_index * 7.5 + heavy_plugin_count * 18 + rng.normal(0, 45, size=n_samples),
        40,
        5000,
    ).round(2)
    disk_read_iops = np.clip(
        18 + pageviews_per_day / 280 + rng.normal(0, 20, size=n_samples),
        1,
        3500,
    ).round(2)
    disk_write_iops = np.clip(
        12 + visitors_per_day / 520 + plugin_count * 1.7 + rng.normal(0, 18, size=n_samples),
        1,
        2800,
    ).round(2)

    wp_type_score = pd.Series(wp_type).map(
        {"small": 32, "medium": 62, "performance": 88}
    ).to_numpy()
    php_risk_score = pd.Series(php_version).map(
        {"7.4": 85, "8.0": 68, "8.1": 54, "8.2": 38, "8.3": 30}
    ).to_numpy()
    cache_risk_score = np.where(cache_enabled == "oui", 25, 80)
    cdn_risk_score = np.where(cdn_enabled == "oui", 35, 58)

    recommended_capacity_score = np.clip(
        0.210 * (traffic_pressure * 100)
        + 0.145 * cpu_usage_avg
        + 0.135 * ram_usage_avg
        + 0.095 * wp_type_score
        + 0.085 * (growth_pressure * 100)
        + 0.070 * cpu_usage_peak
        + 0.060 * (plugin_pressure * 100)
        + 0.055 * ram_usage_max
        + 0.050 * (heavy_plugin_pressure * 100)
        + 0.040 * cache_risk_score
        + 0.025 * php_risk_score
        + 0.014 * disk_usage_avg
        + 0.008 * disk_usage_max
        + 0.007 * cdn_risk_score
        + 0.005 * np.clip(disk_write_iops / 28, 0, 100)
        + 0.003 * np.clip(disk_read_iops / 35, 0, 100)
        + 0.003 * np.clip(response_time / 50, 0, 100)
        + 0.003 * np.clip(pageview_pressure * 100, 0, 100)
        + 0.001 * rng.uniform(0, 100, size=n_samples)
        + 0.001 * rng.uniform(0, 100, size=n_samples)
        + rng.normal(0, 1.8, size=n_samples),
        1,
        100,
    ).round(4)

    data = pd.DataFrame(
        {
            "visitors_per_day": visitors_per_day,
            "pageviews_per_day": pageviews_per_day,
            "traffic_growth_rate": traffic_growth_rate,
            "peak_hours_start": random_time(rng, n_samples, 6, 12),
            "peak_hours_end": random_time(rng, n_samples, 15, 23),
            "cpu_usage_avg": cpu_usage_avg,
            "cpu_usage_peak": cpu_usage_peak,
            "ram_usage_avg": ram_usage_avg,
            "ram_usage_max": ram_usage_max,
            "disk_usage_avg": disk_usage_avg,
            "disk_usage_max": disk_usage_max,
            "response_time": response_time,
            "disk_read_iops": disk_read_iops,
            "disk_write_iops": disk_write_iops,
            "plugin_count": plugin_count,
            "php_version": php_version,
            "cache_enabled": cache_enabled,
            "cdn_enabled": cdn_enabled,
            "wp_type": wp_type,
            "recommended_capacity_score": recommended_capacity_score,
        }
    )

    for index, plugin_name in enumerate(HEAVY_PLUGIN_OPTIONS):
        data[f"heavy_plugin_{plugin_name}"] = heavy_plugin_matrix[:, index]

    data["heavy_plugins"] = [
        ",".join(plugin for plugin, enabled in zip(HEAVY_PLUGIN_OPTIONS, row) if enabled)
        for row in heavy_plugin_matrix
    ]

    ordered_columns = [
        "visitors_per_day",
        "pageviews_per_day",
        "traffic_growth_rate",
        "peak_hours_start",
        "peak_hours_end",
        "cpu_usage_avg",
        "cpu_usage_peak",
        "ram_usage_avg",
        "ram_usage_max",
        "disk_usage_avg",
        "disk_usage_max",
        "response_time",
        "disk_read_iops",
        "disk_write_iops",
        "plugin_count",
        "heavy_plugins",
        *[f"heavy_plugin_{name}" for name in HEAVY_PLUGIN_OPTIONS],
        "php_version",
        "cache_enabled",
        "cdn_enabled",
        "wp_type",
        "recommended_capacity_score",
    ]

    return data[ordered_columns]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()

    for column in ("peak_hours_start", "peak_hours_end"):
        split_time = prepared[column].str.split(":", expand=True).astype(int)
        prepared[f"{column}_minutes"] = split_time[0] * 60 + split_time[1]

    prepared["peak_duration_minutes"] = (
        prepared["peak_hours_end_minutes"] - prepared["peak_hours_start_minutes"]
    ).clip(lower=0)

    return prepared.drop(columns=["peak_hours_start", "peak_hours_end", "heavy_plugins"])


def prepare_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prepared = add_time_features(dataset)
    target = prepared.pop("recommended_capacity_score")

    features = pd.get_dummies(
        prepared,
        columns=["php_version", "cache_enabled", "cdn_enabled", "wp_type"],
        drop_first=False,
        dtype=int,
    )

    return features, target


def save_tree_plot(model: XGBRegressor, tree_index: int, output_path: Path) -> None:
    plt.figure(figsize=(80, 80))
    plot_tree(model, num_trees=tree_index, rankdir="LR", ax=plt.gca())
    tree_title = "Premier arbre XGBoost" if tree_index == 0 else "Dernier arbre XGBoost"
    plt.title(tree_title, fontsize=72, pad=40, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=1.5)
    plt.close()


def normalize_feature_name(feature_name: str) -> str:
    if feature_name.startswith("peak_hours_start"):
        return "peak_hours_start"
    if feature_name.startswith("peak_hours_end"):
        return "peak_hours_end"
    if feature_name.startswith("heavy_plugin_"):
        return "heavy_plugins"
    if feature_name.startswith("php_version_"):
        return "php_version"
    if feature_name.startswith("cache_enabled_"):
        return "cache_enabled"
    if feature_name.startswith("cdn_enabled_"):
        return "cdn_enabled"
    if feature_name.startswith("wp_type_"):
        return "wp_type"
    return feature_name


def save_feature_importance(model: XGBRegressor, timestamp: str) -> Path:
    output_path = GRAPHE_DIR / f"feature_importance_{timestamp}.png"

    raw_scores = model.get_booster().get_score(importance_type="weight")
    grouped_scores = {feature: 0.0 for feature in FEATURE_ORDER}

    for raw_feature, score in raw_scores.items():
        feature = normalize_feature_name(raw_feature)
        if feature in grouped_scores:
            grouped_scores[feature] += float(score)

    plot_data = pd.DataFrame(
        {
            "feature": FEATURE_ORDER,
            "label": [FEATURE_LABELS[feature] for feature in FEATURE_ORDER],
            "f_score": [grouped_scores[feature] for feature in FEATURE_ORDER],
        }
    ).sort_values("f_score", ascending=True)

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(plot_data)))

    figure, axis = plt.subplots(figsize=(16, 12))
    bars = axis.barh(plot_data["label"], plot_data["f_score"], color=colors, height=0.72)

    for bar in bars:
        width = bar.get_width()
        axis.text(
            width + max(plot_data["f_score"].max() * 0.01, 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",
            va="center",
            fontsize=10,
            color="#333333",
        )

    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.set_facecolor("#FFFFFF")
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_xlabel("F-Score", fontsize=12)
    axis.set_ylabel("")
    axis.set_title("Feature Importance", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def train_model() -> dict:
    ensure_directories()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = generate_training_dataset()
    dataset.to_csv(DATASET_PATH, index=False)

    features, target = prepare_features(dataset)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_time

    predictions = model.predict(x_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    relative_errors = np.abs(y_test.to_numpy() - predictions) / np.maximum(np.abs(y_test.to_numpy()), 1e-8)
    accuracy = max(0.0, 100.0 * (1.0 - float(np.mean(relative_errors))))

    joblib.dump(
        {
            "model": model,
            "feature_columns": features.columns.tolist(),
            "target_column": "recommended_capacity_score",
            "heavy_plugin_options": HEAVY_PLUGIN_OPTIONS,
            "categorical_options": {
                "php_version": PHP_VERSIONS,
                "cache_enabled": ["oui", "non"],
                "cdn_enabled": ["oui", "non"],
                "wp_type": WP_TYPES,
            },
            "impact_reference": IMPACT_REFERENCE,
        },
        MODEL_PATH,
    )

    final_tree_index = max(0, model.get_booster().num_boosted_rounds() - 1)

    save_tree_plot(model, 0, GRAPHE_DIR / "tree_0.png")
    save_tree_plot(model, final_tree_index, GRAPHE_DIR / "tree_final.png")

    feature_importance_path = save_feature_importance(model, timestamp)

    # Génération de la courbe d'apprentissage (MSE)
    from sklearn.model_selection import learning_curve
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        x_train,
        y_train,
        train_sizes=np.linspace(0.15, 1.0, 7),
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_mse = -train_scores.mean(axis=1)
    val_mse = -val_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_sizes_abs, train_mse, "o-", color="#2563eb", linewidth=2.6, markersize=8, label="Entraînement")
    ax.plot(train_sizes_abs, val_mse, "s-", color="#10b981", linewidth=2.6, markersize=8, label="Validation")
    ax.fill_between(train_sizes_abs, train_mse, val_mse, color="#94a3b8", alpha=0.16)
    ax.set_title("Courbe d'apprentissage XGBoost", fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre d'échantillons")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    learning_curve_path = GRAPHE_DIR / f"learning_curve_{timestamp}.png"
    plt.savefig(learning_curve_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    metrics = {
        "timestamp": timestamp,
        "samples": N_SAMPLES,
        "target_column": "recommended_capacity_score",
        "rmse": round(float(rmse), 6),
        "mse": round(float(rmse**2), 6),
        "mae": round(float(mae), 6),
        "r2": round(float(r2), 6),
        "accuracy": round(float(accuracy), 2),
        "train_time_seconds": round(float(train_time), 4),
        "impact_reference": IMPACT_REFERENCE,
        "files_generated": {
            "dataset": str(DATASET_PATH),
            "model": str(MODEL_PATH),
            "tree_0": str(GRAPHE_DIR / "tree_0.png"),
            "tree_final": str(GRAPHE_DIR / "tree_final.png"),
            "feature_importance": str(feature_importance_path),
            "metrics": str(METRICS_PATH),
            "learning_curve": str(learning_curve_path),
        },
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"""
📁 Fichiers générés:
├── data/training_dataset.csv
├── models/model.pkl
├── graphe/tree_0.png              (50×200 pouces, arbre COMPLET)
├── graphe/tree_final.png          (50×200 pouces, arbre COMPLET)
└── graphe/feature_importance_{timestamp}.png
|__ data/model_metrics_all.json

📊 Performances:
├── RMSE : {rmse:.4f}
    mse = {rmse**2:.4f}
├── MAE  : {mae:.4f}
├── R²   : {r2:.4f}
└── Temps: {train_time:.1f}s

🎨 Style Feature Importance:
├── Couleur barres: {BAR_COLOR} (identique à l'image)
├── F-Score: Valeurs de référence
└── Format: Identique à feature_importance_20260505_140251.png
"""
    )

    return metrics


if __name__ == "__main__":
    train_model()

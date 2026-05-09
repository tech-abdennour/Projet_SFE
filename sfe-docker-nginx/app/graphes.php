<?php session_start(); ?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graphiques avancés</title>
    <link rel="icon" type="image/png" href="logos.png">
    <link rel="stylesheet" href="styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 28px 22px;
            margin-top: 18px;
        }
        .image-card {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(30,41,59,0.07);
            padding: 18px 18px 12px 18px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 0;
        }
        .image-card h4 {
            font-size: 1.13em;
            font-weight: 700;
            margin-bottom: 10px;
            text-align: center;
        }
        .image-card img {
            width: 100%;
            max-width: 420px;
            max-height: 260px;
            object-fit: contain;
            border-radius: 7px;
            background: #f1f5f9;
            margin-bottom: 10px;
            box-shadow: 0 1px 6px rgba(30,41,59,0.06);
        }
        .image-card.saturation-large {
            grid-column: span 2;
        }
        .image-card.saturation-large img {
            max-width: 100%;
            max-height: 420px;
        }
        .image-name {
            font-size: 0.97em;
            color: #64748b;
            margin-top: 2px;
            text-align: center;
            word-break: break-all;
        }
    </style>
</head>
<body>

<div class="sidebar">
    <div class="sidebar-header" style="padding:30px 24px 24px;text-align:center;">
        <button id="backBtn" type="button" class="btn-secondary" style="background:linear-gradient(135deg,#64748b,#334155);color:#fff;padding:10px 24px;font-size:1em;border:none;border-radius:7px;cursor:pointer;box-shadow:0 2px 8px rgba(100,116,139,0.12);font-weight:500;">
            ⬅️ Revenir en arrière
        </button>
    </div>
</div>

<!-- BARRE UTILISATEUR -->
<div style="position:absolute;top:10px;right:20px;background:linear-gradient(135deg,#1e293b,#334155);color:#fff;padding:8px 20px;border-radius:8px;font-size:0.98rem;box-shadow:0 2px 16px rgba(0,0,0,0.22);z-index:2000;display:flex;align-items:center;gap:8px;pointer-events:auto;">
    <span>👤 Connecté en tant que <strong><?php echo htmlspecialchars((string)($_SESSION['user'] ?? '')); ?></strong></span>
</div>

<div id="toast-graphe" class="toast-notification" style="display:none;position:fixed;bottom:32px;right:32px;left:auto;top:auto;transform:none;z-index:3000;background:linear-gradient(135deg,#fde047,#facc15);color:#a16207;padding:10px 18px;border-radius:8px;font-size:1em;font-weight:600;box-shadow:0 2px 16px rgba(251,191,36,0.18);max-width:320px;width:320px;text-align:left;line-height:1.4;word-break:break-word;">
    Veuillez remplir tous les champs et cliquer sur "Lancer prédiction" afin d'obtenir des résultats de performance et de recommandation.
</div>

<div class="main-content">

<!-- Bloc : Graphiques d'analyse -->
<div class="card" id="imagesContainer" style="display:block;">
    <!-- Titre avec même style que Téléchargement des arbres XGBoost -->
    <h3 class="graphe-title">📊 Graphiques d'analyse XGBoost</h3>
    
    <!-- Bouton centré en dessous du titre -->
    <div class="btn-container">
        <button id="runGrapheXGBoostBtn" class="btn-success" style="background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff;padding:12px 32px;font-size:1.1em;border:none;border-radius:7px;cursor:pointer;box-shadow:0 2px 8px rgba(34,197,94,0.12);font-weight:600;">
            <span>📈</span> Crée Graphes
        </button>
    </div>
    
    <div class="images-grid" id="imagesDisplay">
        <div style="padding:20px;color:#888;text-align:center;">
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:120px;" id="noGraphMessage">
               
            </div>
            <div id="graphe-warning" style="display:none;margin-top:18px;padding:10px 18px;background:linear-gradient(135deg,#fde68a,#fbbf24);color:#92400e;border-radius:7px;font-size:1em;font-weight:500;text-align:center;"></div>
        </div>
    </div>
</div>

<!-- Bloc : Téléchargement des arbres XGBoost -->
<div class="card" id="treeDownloads" style="display:block;">
    <h3>🌳 Téléchargement des arbres XGBoost</h3>
    <div class="tree-links" style="display: flex; flex-wrap: wrap; gap: 18px 22px; align-items: center;">
        <a href="http://localhost:8000/api/download/graphe/tree0" class="tree-link" download>📥 Tree 0</a>
        <a href="http://localhost:8000/api/download/graphe/treefinal" class="tree-link" download>📥 Tree Final</a>
        <a href="http://localhost:8000/api/download/graphe/learning_curve" class="tree-link" download>📈 Courbe d'apprentissage</a>
        <a href="http://localhost:8000/api/download/graphe/feature_importance" class="tree-link" download>📥 Feature Importance</a>
        <a href="http://localhost:8000/api/download/graphe/residus" class="tree-link" download>📉 Résidus</a>
        <a href="http://localhost:8000/api/download/graphe/correlation" class="tree-link" download>🔗 Corrélation</a>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    var btn = document.getElementById('runGrapheXGBoostBtn');
    var toast = document.getElementById('toast-graphe');
    var backBtn = document.getElementById('backBtn');
    
    // Charger les images existantes au chargement de la page
    chargerImagesExistantes();
    
    btn.onclick = function() {
        var prediction = null;
        try {
            prediction = JSON.parse(localStorage.getItem('lastPrediction'));
        } catch(e) {}
        
        if (prediction && typeof prediction.predicted_load === 'number' && typeof prediction.capacity_margin === 'number') {
            toast.style.display = 'none';
            runGrapheXGBoostModel();
        } else {
            toast.style.display = 'block';
            setTimeout(function(){ toast.style.display = 'none'; }, 3500);
        }
    };
    
    if (backBtn) {
        backBtn.addEventListener('click', function(e) {
            e.preventDefault();
            // Sauvegarder l'onglet actif comme "resultats" avant de revenir
            sessionStorage.setItem('activeTab', 'resultats');
            // Rediriger vers la page principale avec l'onglet résultats
            window.location.href = 'dashboard.php?tab=resultats';
        });
    }
});

// Fonction pour charger les images existantes sans en supprimer
function chargerImagesExistantes() {
    fetch('http://localhost:8000/api/get-images-list')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var noGraphMsg = document.getElementById('noGraphMessage');
            if (data.images && data.images.length > 0) {
                afficherGraphiques(data.images);
                if (noGraphMsg) noGraphMsg.style.display = 'none';
            } else {
                document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Aucun graphique généré</p></div>';
                if (noGraphMsg) noGraphMsg.style.display = 'block';
            }
        })
        .catch(function(e) {
            document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de connexion</p></div>';
            var noGraphMsg = document.getElementById('noGraphMessage');
            if (noGraphMsg) noGraphMsg.style.display = 'block';
            var btn = document.getElementById('runGrapheXGBoostBtn');
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span>📈</span> Crée Graphes';
            }
        });
}

// Fonction pour générer les graphiques (sans suppression préalable)
function runGrapheXGBoostModel() {
    var btn = document.getElementById('runGrapheXGBoostBtn');
    if (btn.disabled) {
        console.log('[Graphe] Bouton désactivé, action ignorée.');
        return;
    }
    
    btn.disabled = true;
    btn.innerHTML = '⏳ Génération en cours...';
    document.getElementById('imagesContainer').style.display = 'block';

    console.log('[Graphe] Lancement de la génération des graphiques...');

    // Failsafe timeout: always reset button after 50s
    var failsafeTimeout = setTimeout(function() {
        btn.disabled = false;
        btn.innerHTML = '<span>📈</span> Crée Graphes';
        console.warn('[Graphe] Failsafe: bouton réinitialisé après 35s');
    }, 35000);

    fetch('http://localhost:8000/api/run/graphe-xgboost-model', {
           method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(function(r) { return r.json(); })
    .then(function(res) {
        console.log('[Graphe] Réponse API:', res);
        setTimeout(function() {
            fetch('http://localhost:8000/api/get-images-list')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var noGraphMsg = document.getElementById('noGraphMessage');
                    if (data.images && data.images.length > 0) {
                        afficherGraphiques(data.images);
                        if (noGraphMsg) noGraphMsg.style.display = 'none';
                    } else {
                        document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Aucun graphique généré</p></div>';
                        if (noGraphMsg) noGraphMsg.style.display = 'block';
                    }
                    setTimeout(function() {
                        btn.disabled = false;
                        btn.innerHTML = '<span>📈</span> Crée Graphes';
                        clearTimeout(failsafeTimeout);
                    }, 100);
                });
        }, 1500);
        if (res.status !== 'success') {
            document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de génération</p><p style="font-size:14px;">Réessayez</p></div>';
            btn.disabled = false;
            btn.innerHTML = '<span>📈</span> Crée Graphes';
            clearTimeout(failsafeTimeout);
            console.error('[Graphe] Erreur lors de la génération:', res);
        }
    })
    .catch(function(e) {
        document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de connexion</p></div>';
        btn.disabled = false;
        btn.innerHTML = '<span>📈</span> Crée Graphes';
        clearTimeout(failsafeTimeout);
    });
}

// Fonction d'affichage des graphiques
function afficherGraphiques(images) {
    var order = [
        {match: function(img) { return img.filename.startsWith('partial_dependence_all'); }, label: '📊 Partial Dependence', large: false},
        {match: function(img) { return img.filename.startsWith('response_time_projection'); }, label: '⏱️ Projection Temps de Réponse', large: false},
        {match: function(img) { return img.filename.startsWith('saturation_evolution'); }, label: '⚡ Saturation', large: true},
        {match: function(img) { return img.filename.startsWith('charge_horaire'); }, label: '🕒 Charge horaire', large: false},
        {match: function(img) { return img.filename.startsWith('charge_par_type'); }, label: '🏷️ Charge par type de site', large: false},
        // Ajout d'autres types si besoin
        {match: function(img) { return img.type === 'residus'; }, label: '📉 Résidus', large: false},
        {match: function(img) { return img.type === 'correlation'; }, label: '🔗 Corrélation', large: false},
        {match: function(img) { return img.type === 'feature_importance'; }, label: '⭐ Feature Importance', large: false},
        {match: function(img) { return img.type === 'tree'; }, label: '🌳 Arbre XGBoost', large: false}
    ];
    
    var html = '<div class="images-grid">';
    
    order.forEach(function(o) {
        var img = images.find(o.match);
        if (img) {
            var cardClass = o.large ? 'image-card saturation-large' : 'image-card';
            html += '<div class="' + cardClass + '">';
            html += '<h4>' + o.label + '</h4>';
            html += '<a href="' + img.url + '" target="_blank" style="display:block;text-align:center;">';
            html += '<img src="' + img.url + '" style="width:100%;cursor:pointer;" alt="' + o.label + '">';
            html += '</a>';
            html += '<p class="image-name">';
            html += '<a href="' + img.url + '" target="_blank" style="color:#2563eb;text-decoration:underline;">Lien direct API</a> — ' + img.filename;
            html += '</p>';
            html += '</div>';
        }
    });
    
    html += '</div>';
    
    
    document.getElementById('imagesDisplay').innerHTML = html;
    console.log('[Graphe] Images affichées:', images);
}
</script>

</body>
</html>
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
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        .image-card.saturation-large {
            grid-column: span 2;
        }
        .image-card.saturation-large img {
            width: 100%;
            max-width: 100%;
            max-height: 500px;
            object-fit: contain;
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
    <div class="tree-links">
        <a href="http://localhost:8000/download/graphe/tree0" class="tree-link" download>📥 Tree 0</a>
        <a href="http://localhost:8000/download/graphe/treefinal" class="tree-link" download>📥 Tree Final</a>
        <a href="http://localhost:8000/download/graphe/feature_importance" class="tree-link" download>📥 Feature Importance</a>
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
            prediction = JSON.parse(sessionStorage.getItem('lastPrediction'));
        } catch(e) {}
        
        if (prediction && prediction.xgboost_score && prediction.recommendation) {
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
    fetch('http://localhost:8000/get-images-list')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.images && data.images.length > 0) {
                afficherGraphiques(data.images);
                document.getElementById('noGraphMessage').style.display = 'none';
            }
        })
        .catch(function(e) {
            console.log('[Graphe] Aucune image existante à charger');
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
    
    fetch('http://localhost:8000/run/graphe-xgboost-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(function(r) { return r.json(); })
    .then(function(res) {
        console.log('[Graphe] Réponse API:', res);
        
        if (res.status === 'success') {
            // Afficher les images générées après un court délai
            setTimeout(function() {
                fetch('http://localhost:8000/get-images-list')
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.images && data.images.length > 0) {
                            afficherGraphiques(data.images);
                            document.getElementById('noGraphMessage').style.display = 'none';
                        }
                        btn.disabled = false;
                        btn.innerHTML = '<span>📈</span> Crée Graphes';
                    });
            }, 1500);
        } else {
            document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de génération</p><p style="font-size:14px;">Réessayez</p></div>';
            btn.disabled = false;
            btn.innerHTML = '<span>📈</span> Crée Graphes';
            console.error('[Graphe] Erreur lors de la génération:', res);
        }
    })
    .catch(function(e) {
        console.error('[Graphe] Erreur de connexion à l\'API:', e);
        document.getElementById('imagesDisplay').innerHTML = '<div style="text-align:center;padding:40px;color:#888;"><p style="font-size:18px;">Erreur de connexion</p></div>';
        btn.disabled = false;
        btn.innerHTML = '<span>📈</span> Crée Graphes';
    });
}

// Fonction d'affichage des graphiques
function afficherGraphiques(images) {
    var order = [
        {match: function(img) { return img.filename.startsWith('learning_curve'); }, label: "📈 Courbe d'apprentissage", large: false},
        {match: function(img) { return img.filename.startsWith('partial_dependence_all'); }, label: '📊 Partial Dependence', large: false},
        {match: function(img) { return img.type === 'residus'; }, label: '📉 Résidus', large: false},
        {match: function(img) { return img.type === 'correlation'; }, label: '🔗 Corrélation', large: false},
        {match: function(img) { return img.filename.startsWith('saturation_evolution'); }, label: '⚡ Saturation', large: true},
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
    
    // Ajouter info sur le nombre d'images
    html += '<div style="margin-top:15px;padding:8px 15px;background:#f0f9ff;border-left:4px solid #0284c7;color:#0369a1;border-radius:4px;font-size:0.9em;">';
    html += '✅ ' + images.length + ' image(s) affichée(s) - Les images sont conservées lors de la navigation';
    html += '</div>';
    
    document.getElementById('imagesDisplay').innerHTML = html;
    console.log('[Graphe] Images affichées:', images);
}
</script>

</body>
</html>
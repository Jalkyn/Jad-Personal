# Guide de Prédiction par Lot avec CSV

## Vue d'ensemble

L'application de classification d'exoplanètes permet désormais de traiter des fichiers CSV entiers contenant plusieurs observations d'exoplanètes et d'obtenir un fichier CSV avec toutes les prédictions.

## Fonctionnalités

### 1. Téléchargement du Template CSV
- Bouton: **"Download CSV Template"**
- Génère un fichier CSV exemple avec toutes les 20 colonnes requises
- Contient 2 lignes d'exemples de données réelles du dataset NASA Kepler

### 2. Upload de Dataset CSV
- Bouton: **"Upload CSV Dataset for Batch Prediction"**
- Accepte uniquement les fichiers `.csv`
- Valide automatiquement que toutes les colonnes requises sont présentes

### 3. Traitement Automatique
Une fois le fichier uploadé, le système :
1. Parse et valide les données
2. Effectue les prédictions avec le modèle ML sélectionné
3. Génère un nouveau CSV avec les résultats
4. Télécharge automatiquement le fichier `predictions_[timestamp].csv`
5. Navigue vers l'onglet "Results" pour visualiser les prédictions

## Format du CSV

### Colonnes Requises (Input)
Les 20 features du NASA Kepler Dataset :
- `koi_score` - Detection Score [0–1]
- `planet_density_proxy` - Planet Density (proxy) [g/cm³]
- `koi_model_snr` - Transit SNR
- `koi_fpflag_ss` - FP Flag (Stellar Variability) [0/1]
- `koi_prad` - Planet Radius [Earth radii (R⊕)]
- `koi_duration_err1` - Transit Duration Error (+) [hours]
- `habitability_index` - Habitability Index
- `duration_period_ratio` - Duration/Period Ratio
- `koi_fpflag_co` - FP Flag (Contamination) [0/1]
- `koi_prad_err1` - Planet Radius Error (+) [Earth radii (R⊕)]
- `koi_time0bk_err1` - Transit Epoch Error (+) [days]
- `koi_period` - Orbital Period [days]
- `koi_steff_err2` - Stellar Temp Error (–) [K]
- `koi_steff_err1` - Stellar Temp Error (+) [K]
- `koi_period_err1` - Orbital Period Error (+) [days]
- `koi_depth` - Transit Depth [ppm]
- `koi_fpflag_nt` - FP Flag (Non-Transit) [0/1]
- `koi_impact` - Impact Parameter
- `koi_slogg_err2` - log(g) Error (–) [log(cm/s²)]
- `koi_insol` - Insolation Flux [Earth flux (S⊕)]

### Colonnes Générées (Output)
Le CSV de résultats contient toutes les colonnes d'input PLUS :
- `prediction` - Classification numérique (0, 1, ou 2)
  - 0 = False Positive (Faux Positif)
  - 1 = Candidate (Candidat)
  - 2 = Confirmed (Confirmé)
- `prediction_label` - Label textuel de la prédiction
- `confidence` - Niveau de confiance de la prédiction (0.0 à 1.0)
- `model` - Nom du modèle ML utilisé (XGBoost, Random Forest, etc.)
- `timestamp` - Date et heure ISO de la prédiction

## Exemple d'Utilisation

### Étape 1 : Télécharger le Template
```
Cliquer sur "Download CSV Template" → fichier "example_dataset.csv" téléchargé
```

### Étape 2 : Ajouter vos Données
Ouvrir le fichier CSV et ajouter vos observations :
```csv
koi_score,planet_density_proxy,koi_model_snr,koi_fpflag_ss,koi_prad,...
0.846,4.2,35.7,0,2.26,...
0.692,1.8,16.6,0,4.89,...
...vos données...
```

### Étape 3 : Upload et Prédiction
```
1. Sélectionner le modèle ML souhaité (par défaut: XGBoost)
2. Cliquer sur "Upload CSV Dataset for Batch Prediction"
3. Sélectionner votre fichier CSV
4. Attendre le traitement (toast de confirmation)
5. Le fichier predictions_xxxxx.csv se télécharge automatiquement
6. L'application navigue vers l'onglet "Results"
```

### Étape 4 : Analyser les Résultats
Le fichier téléchargé contient :
```csv
koi_score,planet_density_proxy,koi_model_snr,...,prediction,prediction_label,confidence,model,timestamp
0.846,4.2,35.7,...,2,Confirmed,0.9245,XGBoost,2025-09-30T...
0.692,1.8,16.6,...,1,Candidate,0.7832,XGBoost,2025-09-30T...
```

## Messages de Validation

L'application affiche des messages toast pour :
- ✅ Succès : Nombre de lignes chargées, prédictions téléchargées
- ❌ Erreurs : 
  - Format de fichier invalide (pas CSV)
  - Colonnes manquantes
  - Fichier vide
  - Erreurs de parsing

## Navigation Automatique

Après une prédiction (unique ou batch), l'application change automatiquement d'onglet vers **"Results"** pour afficher :
- Toutes les prédictions récentes
- La possibilité de donner un feedback (correct/incorrect)
- L'historique des 50 dernières prédictions

## Limitations

- Maximum 50 prédictions gardées en mémoire
- Les fichiers très volumineux peuvent prendre quelques secondes à traiter
- Le système simule un délai de 1.5s pour les prédictions batch (pour imiter un vrai ML backend)

## Notes Techniques

- Parser CSV personnalisé gérant les guillemets et virgules
- Validation stricte des colonnes requises
- Case-insensitive pour les noms de colonnes
- Support des lignes vides (ignorées automatiquement)
- Téléchargement via Blob API pour compatibilité navigateur
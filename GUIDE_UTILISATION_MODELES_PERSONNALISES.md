# Guide d'Utilisation - Modèles Personnalisés

## 🎯 Vue d'Ensemble

Votre système de classification d'exoplanètes permet maintenant d'entraîner des modèles personnalisés et de les utiliser immédiatement pour faire des prédictions. Voici comment procéder :

## 📋 Processus Complet

### 1️⃣ **Entraîner un Modèle Personnalisé**

1. **Aller dans l'onglet "Hyperparamètres"**
2. **Sélectionner un type de modèle** (RandomForest, XGBoost, SVM, KNN, Régression Logistique)
3. **Ajuster les hyperparamètres** selon vos besoins :
   - Utilisez les sliders pour les paramètres numériques
   - Sélectionnez les options dans les listes déroulantes
   - Activez/désactivez les switches selon les besoins

4. **Cliquer sur "Entraîner le Modèle"**
   - Une barre de progression s'affiche avec les étapes d'entraînement
   - Le temps estimé restant (ETA) est affiché
   - 7 étapes sont simulées : chargement → préparation → division → entraînement → validation → métriques → finalisation

5. **Consulter les résultats** :
   - Métriques complètes (Accuracy, F1-Macro, ROC-AUC, AUC Score)
   - Matrice de confusion détaillée
   - Temps d'entraînement

### 2️⃣ **Utiliser le Modèle pour des Prédictions**

1. **Aller dans l'onglet "Prédire"**
2. **Le modèle personnalisé apparaît automatiquement** dans la liste des modèles disponibles
3. **Identifier les modèles personnalisés** :
   - Badge "Personnalisé" violet
   - Nom du type + "Personnalisé" + ID court
   - F1-Score affiché

4. **Sélectionner votre modèle personnalisé**
5. **Saisir les caractéristiques** de l'exoplanète
6. **Cliquer sur "Classify Exoplanet"**
7. **Voir le résultat** avec la classe prédite et le niveau de confiance

### 3️⃣ **Gérer Vos Modèles**

Dans l'onglet "Hyperparamètres", section "Modèles Personnalisés Entraînés" :

- **Voir l'historique** de tous vos modèles entraînés
- **Supprimer des modèles** avec le bouton 🗑️
- **Revenir aux modèles par défaut** avec le bouton "Modèles Par Défaut"

## 🔄 Workflow Typique

```
1. Hyperparamètres → Configurer → Entraîner
                         ↓
2. Prédire → Sélectionner modèle personnalisé → Prédire
                         ↓
3. Résultats → Voir historique des prédictions
```

## 📊 Types de Modèles Disponibles

### **Random Forest**
- `n_estimators` : Nombre d'arbres (50-500)
- `max_depth` : Profondeur maximale (5-20 ou None)
- `min_samples_split` : Échantillons minimum pour diviser (2-10)
- `min_samples_leaf` : Échantillons minimum par feuille (1-4)
- `max_features` : Caractéristiques maximum ('sqrt', 'log2', None)

### **XGBoost**
- `n_estimators` : Nombre d'estimateurs (50-200)
- `max_depth` : Profondeur maximale (3-10)
- `learning_rate` : Taux d'apprentissage (0.01-0.2)
- `subsample` : Sous-échantillonnage (0.8-1.0)
- `colsample_bytree` : Échantillonnage des colonnes (0.8-1.0)

### **SVM**
- `C` : Paramètre de régularisation (0.1-100)
- `kernel` : Type de noyau ('rbf', 'poly', 'linear')
- `gamma` : Coefficient du noyau ('scale', 'auto')
- `degree` : Degré du polynôme (2-4)

### **KNN**
- `n_neighbors` : Nombre de voisins (3-20)
- `weights` : Pondération ('uniform', 'distance')
- `metric` : Métrique de distance ('euclidean', 'manhattan', 'minkowski')
- `p` : Paramètre pour Minkowski (1-3)

### **Régression Logistique**
- `C` : Inverse de la régularisation (0.1-100)
- `penalty` : Type de régularisation ('l1', 'l2', 'elasticnet')
- `solver` : Algorithme d'optimisation ('liblinear', 'saga')
- `max_iter` : Itérations maximum (1000-5000)

## 💡 Conseils d'Optimisation

### **Pour Améliorer la Performance :**

1. **Random Forest** :
   - Augmentez `n_estimators` pour plus de stabilité
   - Ajustez `max_depth` pour éviter le surapprentissage
   - Utilisez `max_features='sqrt'` pour de meilleures performances

2. **XGBoost** :
   - Diminuez `learning_rate` et augmentez `n_estimators`
   - Ajustez `max_depth` entre 3-7 pour équilibrer biais/variance
   - Utilisez `subsample` < 1.0 pour réduire le surapprentissage

3. **SVM** :
   - Commencez avec `C=1` et `kernel='rbf'`
   - Ajustez `C` pour contrôler le surapprentissage
   - Testez différents noyaux selon vos données

4. **KNN** :
   - Testez différentes valeurs de `n_neighbors` (impaires)
   - Utilisez `weights='distance'` pour pondérer par distance
   - Choisissez la bonne métrique selon vos données

5. **Régression Logistique** :
   - Utilisez `penalty='l1'` pour sélection de caractéristiques
   - Augmentez `C` si le modèle sous-apprend
   - Diminuez `C` si le modèle sur-apprend

## 🎯 Métriques à Surveiller

### **F1-Macro Score** (Principal)
- Moyenne non pondérée des F1-scores par classe
- Idéal pour datasets déséquilibrés
- **Objectif : > 85%**

### **ROC-AUC**
- Capacité discriminante du modèle
- Insensible au déséquilibre des classes
- **Objectif : > 95%**

### **AUC Score**
- Sensible aux classes minoritaires
- Complément important au ROC-AUC
- **Objectif : > 95%**

## 🚀 Exemples de Configurations Optimales

### **Pour Haute Précision :**
```
XGBoost:
- n_estimators: 200
- max_depth: 7
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
```

### **Pour Rapidité :**
```
Random Forest:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- max_features: 'sqrt'
```

### **Pour Robustesse :**
```
SVM:
- C: 10
- kernel: 'rbf'
- gamma: 'scale'
```

## 🔧 Dépannage

### **Si le F1-Score est faible :**
- Ajustez les hyperparamètres de régularisation
- Testez un autre type de modèle
- Vérifiez les données d'entrée

### **Si l'entraînement est lent :**
- Diminuez `n_estimators` ou `max_depth`
- Utilisez des modèles plus simples (KNN, Régression Logistique)

### **Si le modèle sur-apprend :**
- Augmentez la régularisation (diminuez `C`, ajoutez `penalty`)
- Diminuez la complexité du modèle
- Utilisez `subsample` < 1.0 pour XGBoost

## 📈 Surveillance des Performances

1. **Comparez avec les modèles par défaut**
2. **Testez sur différents datasets**
3. **Surveillez la cohérence des prédictions**
4. **Utilisez l'historique pour identifier les tendances**

---

Le système est maintenant complètement intégré : **entraînez → utilisez → gérez** vos modèles personnalisés en toute simplicité ! 🎉

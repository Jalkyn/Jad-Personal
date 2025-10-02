
# Explication des Métriques

## Métriques de Base

### Accuracy (Exactitude)
Proportion de prédictions correctes parmi toutes les prédictions.
Formule: (VP + VN) / (VP + VN + FP + FN)
Utile quand les classes sont équilibrées.

### Precision (Précision)
Proportion de prédictions positives qui sont correctes.
Formule: VP / (VP + FP)
Répond à: "Parmi les exoplanètes prédites, combien sont vraiment des exoplanètes?"

### Recall (Rappel/Sensibilité)
Proportion de vrais positifs détectés parmi tous les positifs réels.
Formule: VP / (VP + FN)
Répond à: "Parmi toutes les vraies exoplanètes, combien avons-nous détectées?"

### F1-Score
Moyenne harmonique de la précision et du rappel.
Formule: 2 * (Precision * Recall) / (Precision + Recall)
Équilibre entre précision et rappel.

#### F1 Macro
Moyenne non pondérée des F1-scores de chaque classe.
Traite toutes les classes de manière égale, même si déséquilibrées.

#### F1 Weighted
Moyenne pondérée des F1-scores par le nombre d'échantillons de chaque classe.
Prend en compte le déséquilibre des classes.

## Métriques Avancées

### ROC Score (Receiver Operating Characteristic)
Mesure la capacité du modèle à distinguer entre les classes.
Calcule l'aire sous la courbe ROC (sensibilité vs 1-spécificité).
Valeur entre 0 et 1, où 1 = classification parfaite.
Interprétation:
- 0.9-1.0: Excellent
- 0.8-0.9: Très bon
- 0.7-0.8: Bon
- 0.6-0.7: Moyen
- 0.5-0.6: Faible

### AUC Score (Area Under the Curve)
Aire sous la courbe de performance du classifieur.
Mesure la performance globale indépendamment du seuil de décision.
Valeur entre 0 et 1:
- 1.0: Classifieur parfait
- 0.5: Classifieur aléatoire
- < 0.5: Pire qu'aléatoire

Pour la classification multiclasse:
- OVR (One-vs-Rest): Compare chaque classe vs toutes les autres
- OVO (One-vs-One): Compare chaque paire de classes

### Matrice de Confusion
Tableau montrant les prédictions vs les vraies valeurs.
Structure:
              Prédit Négatif  Prédit Positif
Vrai Négatif       VN              FP
Vrai Positif       FN              VP

Permet de visualiser:
- Vrais Positifs (VP): Correctement classés comme positifs
- Vrais Négatifs (VN): Correctement classés comme négatifs
- Faux Positifs (FP): Incorrectement classés comme positifs (Erreur Type I)
- Faux Négatifs (FN): Incorrectement classés comme négatifs (Erreur Type II)

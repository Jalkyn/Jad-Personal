import { ExoplanetFeatures, PredictionResult } from '../types/exoplanet';
import HyperparameterTuningService, { TrainingResult } from './hyperparameterTuning';

// Service pour gérer les prédictions avec les modèles pré-entraînés et personnalisés
export class PredictionService {
  private static instance: PredictionService;
  private modelsLoaded = false;
  private models: Map<string, any> = new Map();
  private metricsData: any[] = [];
  private customModels: Map<string, TrainingResult> = new Map();

  private constructor() {}

  static getInstance(): PredictionService {
    if (!PredictionService.instance) {
      PredictionService.instance = new PredictionService();
    }
    return PredictionService.instance;
  }

  async loadModels(): Promise<void> {
    if (this.modelsLoaded) return;

    try {
      // Charger les métriques depuis le fichier JSON
      const response = await fetch('/ml/metrics/all_models_metrics.json');
      this.metricsData = await response.json();
      
      // Simuler le chargement des modèles (en production, ils seraient chargés via un API Python)
      const modelTypes = ['RandomForest', 'XGBoost', 'SVM', 'KNN', 'LogisticRegression'];
      
      for (const modelType of modelTypes) {
        for (let i = 1; i <= 5; i++) {
          const modelName = `${modelType}_Top${i}`;
          // En production, ceci ferait appel à un service Python pour charger le modèle pickle
          this.models.set(modelName, {
            name: modelName,
            type: modelType,
            loaded: true
          });
        }
      }
      
      this.modelsLoaded = true;
      console.log('Modèles chargés avec succès');
    } catch (error) {
      console.error('Erreur lors du chargement des modèles:', error);
      throw error;
    }
  }

  async predict(features: ExoplanetFeatures, modelName?: string): Promise<PredictionResult> {
    if (!this.modelsLoaded) {
      await this.loadModels();
    }

    // Si aucun modèle spécifique n'est demandé, utiliser le meilleur modèle (XGBoost_Top1)
    const selectedModel = modelName || 'XGBoost_Top1';
    
    // Vérifier si c'est un modèle personnalisé
    if (selectedModel.startsWith('custom_')) {
      return this.predictWithCustomModel(features, selectedModel);
    }
    
    if (!this.models.has(selectedModel)) {
      throw new Error(`Modèle ${selectedModel} non trouvé`);
    }

    try {
      // Simuler la prédiction (en production, ceci ferait appel à un service Python)
      const prediction = await this.makePrediction(features, selectedModel);
      
      return {
        prediction: prediction.class as 0 | 1 | 2,
        confidence: prediction.confidence,
        model: selectedModel,
        timestamp: new Date(),
        features
      };
    } catch (error) {
      console.error('Erreur lors de la prédiction:', error);
      throw error;
    }
  }

  async predictBatch(featuresArray: ExoplanetFeatures[], modelName?: string): Promise<PredictionResult[]> {
    const results: PredictionResult[] = [];
    
    for (const features of featuresArray) {
      try {
        const result = await this.predict(features, modelName);
        results.push(result);
      } catch (error) {
        console.error('Erreur lors de la prédiction pour:', features, error);
        // Continuer avec les autres prédictions même si une échoue
      }
    }
    
    return results;
  }

  private async makePrediction(features: ExoplanetFeatures, modelName: string): Promise<{class: number, confidence: number, probabilities: number[]}> {
    // En production, ceci ferait appel à une API Python Flask/FastAPI
    // Pour la démonstration, nous utilisons une logique simulée basée sur les caractéristiques
    
    const featureValues = Object.values(features);
    
    // Logique de prédiction simulée basée sur des heuristiques réalistes
    let score = 0;
    
    // Facteurs favorisant une planète confirmée (classe 2)
    if (features.koi_score > 0.5) score += 2;
    if (features.koi_model_snr > 0.3) score += 1.5;
    if (features.koi_fpflag_ss === 0 && features.koi_fpflag_co === 0 && features.koi_fpflag_nt === 0) score += 2;
    if (features.habitability_index > 0.5) score += 1;
    if (features.koi_depth > 0.1) score += 1;
    
    // Facteurs favorisant un faux positif (classe 0)
    if (features.koi_score < 0.2) score -= 2;
    if (features.koi_fpflag_ss === 1 || features.koi_fpflag_co === 1 || features.koi_fpflag_nt === 1) score -= 3;
    if (features.koi_model_snr < 0) score -= 1.5;
    
    // Ajouter du bruit aléatoire pour simuler la variabilité du modèle
    score += (Math.random() - 0.5) * 2;
    
    // Convertir le score en prédiction
    let prediction: number;
    let probabilities: number[];
    
    if (score > 2) {
      prediction = 2; // Confirmed
      probabilities = [0.1, 0.2, 0.7];
    } else if (score > 0) {
      prediction = 1; // Candidate
      probabilities = [0.2, 0.6, 0.2];
    } else {
      prediction = 0; // False Positive
      probabilities = [0.7, 0.2, 0.1];
    }
    
    // Ajouter de la variabilité aux probabilités
    const noise = (Math.random() - 0.5) * 0.2;
    probabilities = probabilities.map((p, i) => 
      Math.max(0.05, Math.min(0.95, p + (i === prediction ? noise : -noise/2)))
    );
    
    // Normaliser les probabilités
    const sum = probabilities.reduce((a, b) => a + b, 0);
    probabilities = probabilities.map(p => p / sum);
    
    const confidence = probabilities[prediction];
    
    return {
      class: prediction,
      confidence,
      probabilities
    };
  }

  getModelMetrics(modelName?: string): any {
    const selectedModel = modelName || 'XGBoost_Top1';
    return this.metricsData.find(m => m.model_name === selectedModel);
  }

  getAllMetrics(): any[] {
    return this.metricsData;
  }

  getTopModels(modelType?: string): any[] {
    if (modelType) {
      return this.metricsData.filter(m => m.model_name.startsWith(modelType));
    }
    return this.metricsData;
  }

  getAvailableModels(): string[] {
    const defaultModels = Array.from(this.models.keys());
    const customModels = Array.from(this.customModels.keys());
    return [...defaultModels, ...customModels];
  }

  // Méthodes pour gérer les modèles personnalisés
  addCustomModel(trainingResult: TrainingResult): void {
    this.customModels.set(trainingResult.modelId, trainingResult);
  }

  removeCustomModel(modelId: string): void {
    this.customModels.delete(modelId);
  }

  getCustomModels(): TrainingResult[] {
    return Array.from(this.customModels.values());
  }

  getAllAvailableModels(): Array<{ value: string; label: string; f1Score: number; isCustom: boolean }> {
    const models: Array<{ value: string; label: string; f1Score: number; isCustom: boolean }> = [];
    
    // Modèles par défaut
    const defaultModels = [
      { value: 'XGBoost_Top1', label: 'XGBoost (Top 1) - Recommandé', f1Score: 0.905 },
      { value: 'XGBoost_Top2', label: 'XGBoost (Top 2)', f1Score: 0.902 },
      { value: 'XGBoost_Top3', label: 'XGBoost (Top 3)', f1Score: 0.901 },
      { value: 'RandomForest_Top1', label: 'Random Forest (Top 1)', f1Score: 0.893 },
      { value: 'RandomForest_Top2', label: 'Random Forest (Top 2)', f1Score: 0.893 },
      { value: 'SVM_Top1', label: 'SVM (Top 1)', f1Score: 0.859 },
      { value: 'KNN_Top1', label: 'KNN (Top 1)', f1Score: 0.811 },
      { value: 'LogisticRegression_Top1', label: 'Régression Logistique (Top 1)', f1Score: 0.836 }
    ];

    models.push(...defaultModels.map(m => ({ ...m, isCustom: false })));

    // Modèles personnalisés
    const customModels = Array.from(this.customModels.values());
    models.push(...customModels.map(model => ({
      value: model.modelId,
      label: `${model.modelId.split('_')[1]} Personnalisé (${model.modelId.slice(-8)})`,
      f1Score: model.metrics.f1_macro,
      isCustom: true
    })));

    return models.sort((a, b) => b.f1Score - a.f1Score);
  }

  private async predictWithCustomModel(features: ExoplanetFeatures, modelId: string): Promise<PredictionResult> {
    const customModel = this.customModels.get(modelId);
    if (!customModel) {
      throw new Error(`Modèle personnalisé ${modelId} non trouvé`);
    }

    // Utiliser les hyperparamètres du modèle personnalisé pour ajuster la prédiction
    const prediction = await this.makePredictionWithCustomParams(features, customModel);
    
    return {
      prediction: prediction.class as 0 | 1 | 2,
      confidence: prediction.confidence,
      model: modelId,
      timestamp: new Date(),
      features
    };
  }

  private async makePredictionWithCustomParams(
    features: ExoplanetFeatures, 
    customModel: TrainingResult
  ): Promise<{class: number, confidence: number, probabilities: number[]}> {
    // Ajuster la logique de prédiction basée sur les performances du modèle personnalisé
    const featureValues = Object.values(features);
    
    let score = 0;
    
    // Facteurs favorisant une planète confirmée (classe 2)
    if (features.koi_score > 0.5) score += 2;
    if (features.koi_model_snr > 0.3) score += 1.5;
    if (features.koi_fpflag_ss === 0 && features.koi_fpflag_co === 0 && features.koi_fpflag_nt === 0) score += 2;
    if (features.habitability_index > 0.5) score += 1;
    if (features.koi_depth > 0.1) score += 1;
    
    // Facteurs favorisant un faux positif (classe 0)
    if (features.koi_score < 0.2) score -= 2;
    if (features.koi_fpflag_ss === 1 || features.koi_fpflag_co === 1 || features.koi_fpflag_nt === 1) score -= 3;
    if (features.koi_model_snr < 0) score -= 1.5;
    
    // Ajuster le score basé sur la performance du modèle personnalisé
    const performanceMultiplier = customModel.metrics.f1_macro / 0.9; // Normaliser par rapport à 90%
    score *= performanceMultiplier;
    
    // Ajouter du bruit aléatoire pour simuler la variabilité du modèle
    score += (Math.random() - 0.5) * 2;
    
    // Convertir le score en prédiction
    let prediction: number;
    let probabilities: number[];
    
    if (score > 2) {
      prediction = 2; // Confirmed
      probabilities = [0.1, 0.2, 0.7];
    } else if (score > 0) {
      prediction = 1; // Candidate
      probabilities = [0.2, 0.6, 0.2];
    } else {
      prediction = 0; // False Positive
      probabilities = [0.7, 0.2, 0.1];
    }
    
    // Ajuster les probabilités basées sur la précision du modèle
    const confidence_adjustment = customModel.metrics.accuracy;
    probabilities = probabilities.map((p, i) => 
      i === prediction ? 
        Math.max(0.05, Math.min(0.95, p * confidence_adjustment)) : 
        p * (1 - confidence_adjustment * 0.3)
    );
    
    // Normaliser les probabilités
    const sum = probabilities.reduce((a, b) => a + b, 0);
    probabilities = probabilities.map(p => p / sum);
    
    const confidence = probabilities[prediction];
    
    return {
      class: prediction,
      confidence,
      probabilities
    };
  }

  // Synchroniser avec le service d'entraînement
  syncWithTuningService(): void {
    const tuningService = HyperparameterTuningService.getInstance();
    const trainedModels = tuningService.getAllTrainedModels();
    
    // Ajouter tous les modèles entraînés
    trainedModels.forEach(model => {
      this.customModels.set(model.modelId, model);
    });
  }
}

export default PredictionService;

import { ExoplanetFeatures } from '../types/exoplanet';

export interface HyperparameterConfig {
  [key: string]: number | string | boolean;
}

export interface ModelConfig {
  type: 'RandomForest' | 'XGBoost' | 'SVM' | 'KNN' | 'LogisticRegression';
  hyperparameters: HyperparameterConfig;
}

export interface TrainingProgress {
  progress: number; // 0-100
  stage: string;
  eta: number; // temps estimé restant en secondes
}

export interface TrainingResult {
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_macro: number;
    f1_weighted: number;
    roc_auc: number;
    auc_score: number;
    confusion_matrix: number[][];
  };
  hyperparameters: HyperparameterConfig;
  trainingTime: number;
  modelId: string;
}

export class HyperparameterTuningService {
  private static instance: HyperparameterTuningService;
  private isTraining = false;
  private currentTraining: any = null;
  private trainedModels: Map<string, TrainingResult> = new Map();

  private constructor() {}

  static getInstance(): HyperparameterTuningService {
    if (!HyperparameterTuningService.instance) {
      HyperparameterTuningService.instance = new HyperparameterTuningService();
    }
    return HyperparameterTuningService.instance;
  }

  // Configurations par défaut pour chaque type de modèle
  getDefaultHyperparameters(modelType: string): HyperparameterConfig {
    switch (modelType) {
      case 'RandomForest':
        return {
          n_estimators: 100,
          max_depth: 10,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: 'sqrt'
        };
      case 'XGBoost':
        return {
          n_estimators: 100,
          max_depth: 6,
          learning_rate: 0.1,
          subsample: 0.8,
          colsample_bytree: 0.8
        };
      case 'SVM':
        return {
          C: 1.0,
          kernel: 'rbf',
          gamma: 'scale',
          degree: 3
        };
      case 'KNN':
        return {
          n_neighbors: 5,
          weights: 'uniform',
          metric: 'minkowski',
          p: 2
        };
      case 'LogisticRegression':
        return {
          C: 1.0,
          penalty: 'l2',
          solver: 'liblinear',
          max_iter: 1000
        };
      default:
        return {};
    }
  }

  // Plages de valeurs possibles pour l'optimisation automatique
  getHyperparameterRanges(modelType: string): { [key: string]: any } {
    switch (modelType) {
      case 'RandomForest':
        return {
          n_estimators: [50, 100, 200, 300],
          max_depth: [5, 10, 15, 20, null],
          min_samples_split: [2, 5, 10],
          min_samples_leaf: [1, 2, 4],
          max_features: ['sqrt', 'log2', null]
        };
      case 'XGBoost':
        return {
          n_estimators: [50, 100, 200],
          max_depth: [3, 5, 7, 10],
          learning_rate: [0.01, 0.05, 0.1, 0.2],
          subsample: [0.8, 0.9, 1.0],
          colsample_bytree: [0.8, 0.9, 1.0]
        };
      case 'SVM':
        return {
          C: [0.1, 1, 10, 100],
          kernel: ['rbf', 'poly', 'linear'],
          gamma: ['scale', 'auto'],
          degree: [2, 3, 4]
        };
      case 'KNN':
        return {
          n_neighbors: [3, 5, 7, 9, 15, 20],
          weights: ['uniform', 'distance'],
          metric: ['euclidean', 'manhattan', 'minkowski'],
          p: [1, 2, 3]
        };
      case 'LogisticRegression':
        return {
          C: [0.1, 1, 10, 100],
          penalty: ['l1', 'l2', 'elasticnet'],
          solver: ['liblinear', 'saga'],
          max_iter: [1000, 2000, 5000]
        };
      default:
        return {};
    }
  }

  async trainCustomModel(
    config: ModelConfig,
    progressCallback?: (progress: TrainingProgress) => void
  ): Promise<TrainingResult> {
    if (this.isTraining) {
      throw new Error('Un entraînement est déjà en cours');
    }

    this.isTraining = true;
    const startTime = Date.now();

    try {
      // Simuler l'entraînement avec des étapes progressives
      const stages = [
        'Chargement des données',
        'Préparation des caractéristiques',
        'Division train/test',
        'Entraînement du modèle',
        'Validation croisée',
        'Calcul des métriques',
        'Finalisation'
      ];

      let totalProgress = 0;
      const stageProgress = 100 / stages.length;

      for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        
        if (progressCallback) {
          progressCallback({
            progress: totalProgress,
            stage,
            eta: this.estimateETA(totalProgress, startTime)
          });
        }

        // Simuler le temps d'entraînement variable selon le type de modèle
        const stageTime = this.getStageTime(config.type, stage);
        await this.sleep(stageTime);

        totalProgress += stageProgress;
      }

      // Simuler les métriques finales basées sur les hyperparamètres
      const metrics = this.simulateTrainingMetrics(config);
      
      const trainingTime = Date.now() - startTime;
      const modelId = `custom_${config.type}_${Date.now()}`;

      const result: TrainingResult = {
        metrics,
        hyperparameters: config.hyperparameters,
        trainingTime,
        modelId
      };

      // Sauvegarder le modèle entraîné
      this.trainedModels.set(modelId, result);

      if (progressCallback) {
        progressCallback({
          progress: 100,
          stage: 'Terminé',
          eta: 0
        });
      }

      return result;

    } catch (error) {
      console.error('Erreur lors de l\'entraînement:', error);
      throw error;
    } finally {
      this.isTraining = false;
    }
  }

  private getStageTime(modelType: string, stage: string): number {
    // Temps de base pour chaque étape (en millisecondes)
    const baseTimes: { [key: string]: number } = {
      'Chargement des données': 500,
      'Préparation des caractéristiques': 300,
      'Division train/test': 200,
      'Entraînement du modèle': 2000,
      'Validation croisée': 1500,
      'Calcul des métriques': 500,
      'Finalisation': 300
    };

    // Multiplicateurs selon le type de modèle
    const modelMultipliers: { [key: string]: number } = {
      'RandomForest': 1.2,
      'XGBoost': 1.5,
      'SVM': 2.0,
      'KNN': 0.8,
      'LogisticRegression': 0.6
    };

    const baseTime = baseTimes[stage] || 500;
    const multiplier = modelMultipliers[modelType] || 1.0;
    
    return baseTime * multiplier;
  }

  private estimateETA(currentProgress: number, startTime: number): number {
    if (currentProgress <= 0) return 0;
    
    const elapsedTime = Date.now() - startTime;
    const totalEstimatedTime = (elapsedTime / currentProgress) * 100;
    const remainingTime = totalEstimatedTime - elapsedTime;
    
    return Math.max(0, Math.round(remainingTime / 1000));
  }

  private simulateTrainingMetrics(config: ModelConfig): TrainingResult['metrics'] {
    // Simuler des métriques réalistes basées sur le type de modèle et les hyperparamètres
    const baseMetrics: { [key: string]: any } = {
      'RandomForest': { accuracy: 0.91, precision: 0.92, recall: 0.91, f1_macro: 0.89, f1_weighted: 0.91, roc_auc: 0.986, auc_score: 0.978 },
      'XGBoost': { accuracy: 0.93, precision: 0.93, recall: 0.93, f1_macro: 0.91, f1_weighted: 0.93, roc_auc: 0.988, auc_score: 0.982 },
      'SVM': { accuracy: 0.89, precision: 0.90, recall: 0.89, f1_macro: 0.86, f1_weighted: 0.89, roc_auc: 0.980, auc_score: 0.971 },
      'KNN': { accuracy: 0.86, precision: 0.85, recall: 0.86, f1_macro: 0.81, f1_weighted: 0.85, roc_auc: 0.965, auc_score: 0.953 },
      'LogisticRegression': { accuracy: 0.88, precision: 0.88, recall: 0.88, f1_macro: 0.84, f1_weighted: 0.88, roc_auc: 0.974, auc_score: 0.963 }
    };

    const base = baseMetrics[config.type];
    
    // Ajouter de la variabilité basée sur les hyperparamètres et du bruit aléatoire
    const variation = (Math.random() - 0.5) * 0.05; // ±2.5%
    
    const metrics = {
      accuracy: Math.max(0.7, Math.min(0.95, base.accuracy + variation)),
      precision: Math.max(0.7, Math.min(0.95, base.precision + variation)),
      recall: Math.max(0.7, Math.min(0.95, base.recall + variation)),
      f1_macro: Math.max(0.65, Math.min(0.92, base.f1_macro + variation)),
      f1_weighted: Math.max(0.7, Math.min(0.95, base.f1_weighted + variation)),
      roc_auc: Math.max(0.85, Math.min(0.99, base.roc_auc + variation)),
      auc_score: Math.max(0.85, Math.min(0.99, base.auc_score + variation)),
      confusion_matrix: this.generateConfusionMatrix()
    };

    return metrics;
  }

  private generateConfusionMatrix(): number[][] {
    // Générer une matrice de confusion réaliste pour 3 classes
    const total = 1913; // Taille approximative du jeu de test
    const distribution = [0.21, 0.29, 0.50]; // Distribution approximative des classes
    
    const matrix: number[][] = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];

    for (let true_class = 0; true_class < 3; true_class++) {
      const class_total = Math.round(total * distribution[true_class]);
      let remaining = class_total;

      for (let pred_class = 0; pred_class < 3; pred_class++) {
        if (pred_class === 2 && true_class === 2) {
          // Dernière cellule, assigner le reste
          matrix[true_class][pred_class] = remaining;
        } else {
          let probability;
          if (true_class === pred_class) {
            // Diagonale principale - haute probabilité de prédiction correcte
            probability = 0.85 + Math.random() * 0.1;
          } else {
            // Erreurs de classification
            probability = Math.random() * 0.15;
          }
          
          const count = Math.min(remaining, Math.round(class_total * probability));
          matrix[true_class][pred_class] = count;
          remaining -= count;
        }
      }
    }

    return matrix;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getTrainedModel(modelId: string): TrainingResult | undefined {
    return this.trainedModels.get(modelId);
  }

  getAllTrainedModels(): TrainingResult[] {
    return Array.from(this.trainedModels.values());
  }

  isCurrentlyTraining(): boolean {
    return this.isTraining;
  }

  cancelTraining(): void {
    if (this.currentTraining) {
      // En production, ceci annulerait le processus d'entraînement
      this.currentTraining = null;
    }
    this.isTraining = false;
  }
}

export default HyperparameterTuningService;

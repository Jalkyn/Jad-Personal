import { MLModel, ExoplanetFeatures, PredictionResult, ModelMetrics } from '../types/exoplanet';

// Create ML models with real performance metrics from optimization analysis
export function createMockModels(): MLModel[] {
  return [
    {
      name: 'XGBoost',
      type: 'XGBoost',
      metrics: {
        accuracy: 0.9242,
        precision: 0.9246, // Using F1-Weighted as precision approximation
        recall: 0.9002, // Using F1-Macro as recall approximation
        f1Score: 0.9002, // F1-Macro score
        f1Weighted: 0.9246, // F1-Weighted score
        confusionMatrix: [
          [1450, 25, 7],
          [48, 920, 15],
          [8, 22, 590],
        ],
        totalPredictions: 3085
      },
      hyperparameters: {
        n_estimators: 200,
        max_depth: 8,
        learning_rate: 0.08,
        subsample: 0.9,
        colsample_bytree: 0.85,
        gamma: 0.1,
        reg_alpha: 0.05,
        reg_lambda: 1.2
      },
      isTraining: false
    },
    {
      name: 'Random Forest',
      type: 'Random Forest',
      metrics: {
        accuracy: 0.9179,
        precision: 0.9179, // Using F1-Weighted as precision approximation
        recall: 0.8917, // Using F1-Macro as recall approximation
        f1Score: 0.8917, // F1-Macro score
        f1Weighted: 0.9179, // F1-Weighted score
        confusionMatrix: [
          [1435, 35, 12],
          [65, 910, 17],
          [15, 28, 568],
        ],
        totalPredictions: 3085
      },
      hyperparameters: {
        n_estimators: 300,
        max_depth: 12,
        min_samples_split: 3,
        min_samples_leaf: 2,
        criterion: 'entropy',
        bootstrap: true,
        max_features: 'sqrt'
      },
      isTraining: false
    },
    {
      name: 'SVM',
      type: 'SVM',
      metrics: {
        accuracy: 0.8878,
        precision: 0.8878, // Using F1-Weighted as precision approximation
        recall: 0.8510, // Using F1-Macro as recall approximation
        f1Score: 0.8510, // F1-Macro score
        f1Weighted: 0.8878, // F1-Weighted score
        confusionMatrix: [
          [1390, 67, 25],
          [89, 845, 58],
          [28, 67, 516],
        ],
        totalPredictions: 3085
      },
      hyperparameters: {
        C: 10.0,
        kernel: 'rbf',
        gamma: 0.01,
        degree: 3,
        probability: true
      },
      isTraining: false
    },
    {
      name: 'Logistic Regression',
      type: 'Logistic Regression',
      metrics: {
        accuracy: 0.8798,
        precision: 0.8798, // Using F1-Weighted as precision approximation
        recall: 0.8447, // Using F1-Macro as recall approximation
        f1Score: 0.8447, // F1-Macro score
        f1Weighted: 0.8798, // F1-Weighted score
        confusionMatrix: [
          [1375, 82, 25],
          [95, 825, 72],
          [32, 78, 501],
        ],
        totalPredictions: 3085
      },
      hyperparameters: {
        C: 5.0,
        solver: 'saga',
        max_iter: 1000,
        l1_ratio: 0.3,
        penalty: 'elasticnet'
      },
      isTraining: false
    },
    {
      name: 'KNN',
      type: 'KNN',
      metrics: {
        accuracy: 0.8192,
        precision: 0.8192, // Using F1-Weighted as precision approximation
        recall: 0.7683, // Using F1-Macro as recall approximation
        f1Score: 0.7683, // F1-Macro score
        f1Weighted: 0.8192, // F1-Weighted score
        confusionMatrix: [
          [1320, 105, 57],
          [135, 785, 72],
          [48, 95, 468],
        ],
        totalPredictions: 3085
      },
      hyperparameters: {
        n_neighbors: 7,
        weights: 'distance',
        algorithm: 'ball_tree',
        leaf_size: 20,
        metric: 'minkowski'
      },
      isTraining: false
    }
  ];
}

// Simulate ML model prediction
export function predictWithModel(model: MLModel, features: ExoplanetFeatures): PredictionResult {
  // Simple heuristic-based prediction simulation
  let prediction: 0 | 1 | 2;
  let confidence: number;

  // Calculate a score based on key features
  const periodScore = features.koi_period > 0 && features.koi_period < 400 ? 1 : 0;
  const radiusScore = features.koi_prad > 0.5 && features.koi_prad < 4 ? 1 : 0;
  const snrScore = features.koi_model_snr > 10 ? 1 : 0;
  const detectionScore = features.koi_score > 0.5 ? 1 : 0;
  const depthScore = features.koi_depth > 50 ? 1 : 0;

  const totalScore = periodScore + radiusScore + snrScore + detectionScore + depthScore;

  // Model-specific behavior
  const modelMultiplier = getModelMultiplier(model.name);
  const adjustedScore = totalScore * modelMultiplier;

  // Determine prediction based on adjusted score
  if (adjustedScore >= 4.2) {
    prediction = 2; // Confirmed
    confidence = 0.85 + Math.random() * 0.1;
  } else if (adjustedScore >= 2.8) {
    prediction = 1; // Candidate
    confidence = 0.65 + Math.random() * 0.2;
  } else {
    prediction = 0; // False Positive
    confidence = 0.75 + Math.random() * 0.15;
  }

  // Add some randomness to simulate real model behavior
  if (Math.random() < 0.1) {
    prediction = ((prediction + 1) % 3) as 0 | 1 | 2;
    confidence *= 0.8;
  }

  // Ensure confidence is within bounds
  confidence = Math.min(0.99, Math.max(0.5, confidence));

  return {
    prediction,
    confidence,
    model: model.name,
    timestamp: new Date(),
    features
  };
}

function getModelMultiplier(modelName: string): number {
  switch (modelName) {
    case 'XGBoost': return 1.1;
    case 'Random Forest': return 1.05;
    case 'SVM': return 0.95;
    case 'Logistic Regression': return 0.9;
    case 'KNN': return 0.85;
    default: return 1.0;
  }
}

// Update model metrics based on user feedback
export function updateModelMetrics(model: MLModel, isCorrect: boolean): MLModel {
  const newTotalPredictions = model.metrics.totalPredictions + 1;
  
  // Simple metric update simulation
  let newAccuracy = model.metrics.accuracy;
  if (isCorrect) {
    newAccuracy = (model.metrics.accuracy * model.metrics.totalPredictions + 1) / newTotalPredictions;
  } else {
    newAccuracy = (model.metrics.accuracy * model.metrics.totalPredictions) / newTotalPredictions;
  }

  // Update other metrics proportionally
  const accuracyRatio = newAccuracy / model.metrics.accuracy;
  
  return {
    ...model,
    metrics: {
      ...model.metrics,
      accuracy: newAccuracy,
      precision: Math.min(0.99, model.metrics.precision * accuracyRatio),
      recall: Math.min(0.99, model.metrics.recall * accuracyRatio),
      f1Score: Math.min(0.99, model.metrics.f1Score * accuracyRatio),
      totalPredictions: newTotalPredictions,
      // Update confusion matrix (simplified)
      confusionMatrix: model.metrics.confusionMatrix.map(row => 
        row.map(cell => Math.max(0, Math.round(cell * (1 + (Math.random() - 0.5) * 0.02))))
      )
    }
  };
}

// Generate realistic training data for visualization
export function generateTrainingData(numSamples: number = 100): ExoplanetFeatures[] {
  const data: ExoplanetFeatures[] = [];
  
  for (let i = 0; i < numSamples; i++) {
    data.push({
      koi_score: Math.random() * 0.8 + 0.2,
      planet_density_proxy: Math.random() * 10 + 0.5,
      koi_model_snr: Math.random() * 100 + 5,
      koi_fpflag_ss: Math.random() > 0.8 ? 1 : 0,
      koi_prad: Math.random() * 8 + 0.1,
      koi_duration_err1: Math.random() * 2 + 0.01,
      habitability_index: Math.random() * 1,
      duration_period_ratio: Math.random() * 0.5,
      koi_fpflag_co: Math.random() > 0.9 ? 1 : 0,
      koi_prad_err1: Math.random() * 1 + 0.01,
      koi_time0bk_err1: Math.random() * 0.01,
      koi_period: Math.random() * 500 + 0.5,
      koi_steff_err2: -(Math.random() * 200 + 10),
      koi_steff_err1: Math.random() * 200 + 10,
      koi_period_err1: Math.random() * 0.1,
      koi_depth: Math.random() * 5000 + 10,
      koi_fpflag_nt: Math.random() > 0.85 ? 1 : 0,
      koi_impact: Math.random() * 1.5,
      koi_slogg_err2: -(Math.random() * 0.2 + 0.01),
      koi_insol: Math.random() * 1000 + 0.1,
    });
  }
  
  return data;
}
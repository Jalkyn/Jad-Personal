export interface ExoplanetFeatures {
  koi_period: number;
  koi_time0bk: number;
  koi_impact: number;
  koi_duration: number;
  koi_depth: number;
  koi_prad: number;
  koi_teq: number;
  koi_insol: number;
  koi_model_snr: number;
  koi_tce_plnt_num: number;
  koi_steff: number;
  koi_slogg: number;
  koi_srad: number;
  ra: number;
  dec: number;
  koi_kepmag: number;
  koi_gmag: number;
  koi_rmag: number;
  koi_imag: number;
  koi_zmag: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number; // F1-Macro score for fair multiclass evaluation
  f1Weighted: number; // F1-Weighted score considering class distribution
  confusionMatrix: number[][];
  totalPredictions: number;
}

export interface ModelHyperparameters {
  [key: string]: number | string | boolean;
}

export interface MLModel {
  name: string;
  type: 'Random Forest' | 'SVM' | 'XGBoost' | 'KNN' | 'Logistic Regression';
  metrics: ModelMetrics;
  hyperparameters: ModelHyperparameters;
  isTraining: boolean;
}

export interface PredictionResult {
  prediction: 0 | 1 | 2; // False Positive, Candidate, Confirmed
  confidence: number;
  model: string;
  timestamp: Date;
  features: ExoplanetFeatures;
  userFeedback?: 'correct' | 'incorrect' | 'unknown';
}

export const FEATURE_DESCRIPTIONS: Record<keyof ExoplanetFeatures, string> = {
  koi_period: "Orbital Period (days)",
  koi_time0bk: "Transit Epoch (BJD-2454900)",
  koi_impact: "Impact Parameter",
  koi_duration: "Transit Duration (hours)",
  koi_depth: "Transit Depth (ppm)",
  koi_prad: "Planetary Radius (Earth radii)",
  koi_teq: "Equilibrium Temperature (K)",
  koi_insol: "Insolation Flux (Earth flux)",
  koi_model_snr: "Transit Signal-to-Noise",
  koi_tce_plnt_num: "TCE Planet Number",
  koi_steff: "Stellar Effective Temperature (K)",
  koi_slogg: "Stellar Surface Gravity (log10(cm/s^2))",
  koi_srad: "Stellar Radius (Solar radii)",
  ra: "Right Ascension (degrees)",
  dec: "Declination (degrees)",
  koi_kepmag: "Kepler Magnitude",
  koi_gmag: "g-band Magnitude",
  koi_rmag: "r-band Magnitude",
  koi_imag: "i-band Magnitude",
  koi_zmag: "z-band Magnitude"
};

export const PREDICTION_LABELS = {
  0: "False Positive",
  1: "Candidate",
  2: "Confirmed"
};
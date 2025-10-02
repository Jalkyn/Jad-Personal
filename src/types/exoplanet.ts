export interface ExoplanetFeatures {
  koi_score: number;
  planet_density_proxy: number;
  koi_model_snr: number;
  koi_fpflag_ss: number;
  koi_prad: number;
  koi_duration_err1: number;
  habitability_index: number;
  duration_period_ratio: number;
  koi_fpflag_co: number;
  koi_prad_err1: number;
  koi_time0bk_err1: number;
  koi_period: number;
  koi_steff_err2: number;
  koi_steff_err1: number;
  koi_period_err1: number;
  koi_depth: number;
  koi_fpflag_nt: number;
  koi_impact: number;
  koi_slogg_err2: number;
  koi_insol: number;
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
  [key: string]: number | string;
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
  koi_score: "Detection Score [0–1]",
  planet_density_proxy: "Planet Density (proxy) [g/cm³]",
  koi_model_snr: "Transit SNR",
  koi_fpflag_ss: "FP Flag (Stellar Variability) [0/1]",
  koi_prad: "Planet Radius [Earth radii (R⊕)]",
  koi_duration_err1: "Transit Duration Error (+) [hours]",
  habitability_index: "Habitability Index",
  duration_period_ratio: "Duration/Period Ratio",
  koi_fpflag_co: "FP Flag (Contamination) [0/1]",
  koi_prad_err1: "Planet Radius Error (+) [Earth radii (R⊕)]",
  koi_time0bk_err1: "Transit Epoch Error (+) [days]",
  koi_period: "Orbital Period [days]",
  koi_steff_err2: "Stellar Temp Error (–) [K]",
  koi_steff_err1: "Stellar Temp Error (+) [K]",
  koi_period_err1: "Orbital Period Error (+) [days]",
  koi_depth: "Transit Depth [ppm]",
  koi_fpflag_nt: "FP Flag (Non-Transit) [0/1]",
  koi_impact: "Impact Parameter",
  koi_slogg_err2: "log(g) Error (–) [log(cm/s²)]",
  koi_insol: "Insolation Flux [Earth flux (S⊕)]"
};

export const PREDICTION_LABELS = {
  0: "False Positive",
  1: "Candidate",
  2: "Confirmed"
};
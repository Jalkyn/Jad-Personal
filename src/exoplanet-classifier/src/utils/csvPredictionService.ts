import { ExoplanetFeatures, PredictionResult } from '../types/exoplanet';
import PredictionService from './predictionService';

export interface CSVPredictionRow {
  id?: string;
  features: ExoplanetFeatures;
  originalRow: any;
}

export interface CSVBatchResult {
  predictions: PredictionResult[];
  processed: number;
  errors: Array<{ row: number; error: string; data?: any }>;
  summary: {
    totalRows: number;
    successfulPredictions: number;
    errorCount: number;
    predictionCounts: { [key: number]: number };
  };
}

export class CSVPredictionService {
  private static instance: CSVPredictionService;
  private predictionService: PredictionService;

  private constructor() {
    this.predictionService = PredictionService.getInstance();
  }

  static getInstance(): CSVPredictionService {
    if (!CSVPredictionService.instance) {
      CSVPredictionService.instance = new CSVPredictionService();
    }
    return CSVPredictionService.instance;
  }

  /**
   * Parse un fichier CSV et extrait les caractéristiques d'exoplanètes
   */
  async parseCSVFile(file: File): Promise<CSVPredictionRow[]> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        try {
          const csvText = event.target?.result as string;
          const rows = this.parseCSVText(csvText);
          resolve(rows);
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => {
        reject(new Error('Erreur lors de la lecture du fichier'));
      };
      
      reader.readAsText(file);
    });
  }

  /**
   * Parse le texte CSV et convertit en objets ExoplanetFeatures
   */
  private parseCSVText(csvText: string): CSVPredictionRow[] {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) {
      throw new Error('Le fichier CSV doit contenir au moins un en-tête et une ligne de données');
    }

    const headers = this.parseCSVLine(lines[0]);
    const featureColumns = this.mapHeadersToFeatures(headers);
    const rows: CSVPredictionRow[] = [];

    for (let i = 1; i < lines.length; i++) {
      try {
        const values = this.parseCSVLine(lines[i]);
        if (values.length !== headers.length) {
          console.warn(`Ligne ${i + 1}: Nombre de colonnes incorrect (${values.length} vs ${headers.length})`);
          continue;
        }

        const features = this.extractFeatures(headers, values, featureColumns);
        if (features) {
          const originalRow: any = {};
          headers.forEach((header, index) => {
            originalRow[header] = values[index];
          });

          rows.push({
            id: `row_${i}`,
            features,
            originalRow
          });
        }
      } catch (error) {
        console.warn(`Erreur ligne ${i + 1}:`, error);
      }
    }

    return rows;
  }

  /**
   * Parse une ligne CSV en prenant en compte les guillemets
   */
  private parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"' && !inQuotes) {
        inQuotes = true;
      } else if (char === '"' && inQuotes) {
        if (line[i + 1] === '"') {
          current += '"';
          i++; // Skip next quote
        } else {
          inQuotes = false;
        }
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    
    result.push(current.trim());
    return result;
  }

  /**
   * Map les en-têtes CSV aux noms de caractéristiques d'exoplanètes
   */
  private mapHeadersToFeatures(headers: string[]): { [header: string]: keyof ExoplanetFeatures } {
    const mapping: { [header: string]: keyof ExoplanetFeatures } = {};
    
    // Mapping exact des noms de colonnes
    const exactMappings: { [key: string]: keyof ExoplanetFeatures } = {
      'koi_score': 'koi_score',
      'planet_density_proxy': 'planet_density_proxy',
      'koi_model_snr': 'koi_model_snr',
      'koi_fpflag_ss': 'koi_fpflag_ss',
      'koi_prad': 'koi_prad',
      'koi_duration_err1': 'koi_duration_err1',
      'habitability_index': 'habitability_index',
      'duration_period_ratio': 'duration_period_ratio',
      'koi_fpflag_co': 'koi_fpflag_co',
      'koi_prad_err1': 'koi_prad_err1',
      'koi_time0bk_err1': 'koi_time0bk_err1',
      'koi_period': 'koi_period',
      'koi_steff_err2': 'koi_steff_err2',
      'koi_steff_err1': 'koi_steff_err1',
      'koi_period_err1': 'koi_period_err1',
      'koi_depth': 'koi_depth',
      'koi_fpflag_nt': 'koi_fpflag_nt',
      'koi_impact': 'koi_impact',
      'koi_slogg_err2': 'koi_slogg_err2',
      'koi_insol': 'koi_insol'
    };

    // Mapping par similarité pour être plus flexible
    headers.forEach(header => {
      const cleanHeader = header.toLowerCase().trim();
      
      if (exactMappings[cleanHeader]) {
        mapping[header] = exactMappings[cleanHeader];
      } else {
        // Recherche par similarité
        for (const [key, value] of Object.entries(exactMappings)) {
          if (cleanHeader.includes(key) || key.includes(cleanHeader)) {
            mapping[header] = value;
            break;
          }
        }
      }
    });

    return mapping;
  }

  /**
   * Extrait les caractéristiques d'exoplanètes à partir d'une ligne CSV
   */
  private extractFeatures(
    headers: string[], 
    values: string[], 
    featureColumns: { [header: string]: keyof ExoplanetFeatures }
  ): ExoplanetFeatures | null {
    const features: Partial<ExoplanetFeatures> = {};
    let validFeatureCount = 0;
    
    headers.forEach((header, index) => {
      const featureName = featureColumns[header];
      if (featureName && values[index] !== undefined && values[index] !== '') {
        const numericValue = parseFloat(values[index]);
        if (!isNaN(numericValue)) {
          features[featureName] = numericValue;
          validFeatureCount++;
        }
      }
    });

    // Vérifier que nous avons suffisamment de caractéristiques
    const requiredFeatures = Object.keys({
      koi_score: 0,
      planet_density_proxy: 0,
      koi_model_snr: 0,
      koi_fpflag_ss: 0,
      koi_prad: 0,
      koi_duration_err1: 0,
      habitability_index: 0,
      duration_period_ratio: 0,
      koi_fpflag_co: 0,
      koi_prad_err1: 0,
      koi_time0bk_err1: 0,
      koi_period: 0,
      koi_steff_err2: 0,
      koi_steff_err1: 0,
      koi_period_err1: 0,
      koi_depth: 0,
      koi_fpflag_nt: 0,
      koi_impact: 0,
      koi_slogg_err2: 0,
      koi_insol: 0
    } as ExoplanetFeatures);

    if (validFeatureCount < requiredFeatures.length * 0.8) { // Au moins 80% des caractéristiques
      return null;
    }

    // Compléter les caractéristiques manquantes avec des valeurs par défaut
    requiredFeatures.forEach(feature => {
      if (features[feature as keyof ExoplanetFeatures] === undefined) {
        features[feature as keyof ExoplanetFeatures] = 0;
      }
    });

    return features as ExoplanetFeatures;
  }

  /**
   * Effectue des prédictions par lot sur les données CSV
   */
  async processBatchPredictions(
    rows: CSVPredictionRow[],
    modelName?: string,
    progressCallback?: (progress: number, processed: number, total: number) => void
  ): Promise<CSVBatchResult> {
    const result: CSVBatchResult = {
      predictions: [],
      processed: 0,
      errors: [],
      summary: {
        totalRows: rows.length,
        successfulPredictions: 0,
        errorCount: 0,
        predictionCounts: { 0: 0, 1: 0, 2: 0 }
      }
    };

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      
      try {
        const prediction = await this.predictionService.predict(row.features, modelName);
        result.predictions.push(prediction);
        result.summary.successfulPredictions++;
        result.summary.predictionCounts[prediction.prediction]++;
      } catch (error) {
        result.errors.push({
          row: i + 1,
          error: error instanceof Error ? error.message : 'Erreur inconnue',
          data: row.originalRow
        });
        result.summary.errorCount++;
      }

      result.processed++;
      
      if (progressCallback) {
        const progress = (result.processed / rows.length) * 100;
        progressCallback(progress, result.processed, rows.length);
      }

      // Ajouter une petite pause pour ne pas bloquer l'interface
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 1));
      }
    }

    return result;
  }

  /**
   * Exporte les résultats de prédiction en CSV
   */
  exportPredictionsToCSV(results: CSVBatchResult, originalRows: CSVPredictionRow[]): string {
    const headers = [
      'ID',
      'Prediction',
      'Prediction_Label',
      'Confidence',
      'Model',
      'Timestamp',
      ...Object.keys(originalRows[0]?.originalRow || {})
    ];

    const csvLines = [headers.join(',')];

    results.predictions.forEach((prediction, index) => {
      const originalRow = originalRows[index];
      const predictionLabel = prediction.prediction === 0 ? 'False Positive' :
                            prediction.prediction === 1 ? 'Candidate' : 'Confirmed';
      
      const row = [
        originalRow?.id || `row_${index + 1}`,
        prediction.prediction,
        predictionLabel,
        prediction.confidence.toFixed(4),
        prediction.model,
        prediction.timestamp.toISOString(),
        ...Object.values(originalRow?.originalRow || {})
      ];

      csvLines.push(row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(','));
    });

    return csvLines.join('\n');
  }

  /**
   * Télécharge les résultats en tant que fichier CSV
   */
  downloadPredictions(results: CSVBatchResult, originalRows: CSVPredictionRow[], filename?: string): void {
    const csvContent = this.exportPredictionsToCSV(results, originalRows);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename || `exoplanet_predictions_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
  }

  /**
   * Valide le format du fichier CSV
   */
  validateCSVFormat(headers: string[]): { isValid: boolean; issues: string[] } {
    const issues: string[] = [];
    const requiredFeatures = Object.keys({
      koi_score: 0,
      planet_density_proxy: 0,
      koi_model_snr: 0,
      koi_fpflag_ss: 0,
      koi_prad: 0,
      koi_duration_err1: 0,
      habitability_index: 0,
      duration_period_ratio: 0,
      koi_fpflag_co: 0,
      koi_prad_err1: 0,
      koi_time0bk_err1: 0,
      koi_period: 0,
      koi_steff_err2: 0,
      koi_steff_err1: 0,
      koi_period_err1: 0,
      koi_depth: 0,
      koi_fpflag_nt: 0,
      koi_impact: 0,
      koi_slogg_err2: 0,
      koi_insol: 0
    } as ExoplanetFeatures);

    const featureColumns = this.mapHeadersToFeatures(headers);
    const foundFeatures = Object.values(featureColumns);
    
    const missingFeatures = requiredFeatures.filter(feature => 
      !foundFeatures.includes(feature as keyof ExoplanetFeatures)
    );

    if (missingFeatures.length > requiredFeatures.length * 0.2) { // Plus de 20% manquant
      issues.push(`Caractéristiques manquantes: ${missingFeatures.join(', ')}`);
    }

    if (headers.length === 0) {
      issues.push('Aucun en-tête trouvé dans le fichier CSV');
    }

    if (foundFeatures.length < requiredFeatures.length * 0.5) { // Moins de 50% des caractéristiques
      issues.push('Trop peu de caractéristiques reconnues dans le fichier');
    }

    return {
      isValid: issues.length === 0,
      issues
    };
  }
}

export default CSVPredictionService;

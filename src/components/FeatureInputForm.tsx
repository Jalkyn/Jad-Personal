import React, { useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Loader2, Rocket, Info, Upload, Download } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { ExoplanetFeatures, FEATURE_DESCRIPTIONS, MLModel, PredictionResult, PREDICTION_LABELS } from '../types/exoplanet';
import { toast } from 'sonner@2.0.3';
import { CSVUploadHelp } from './CSVUploadHelp';

interface FeatureInputFormProps {
  models: Array<{ value: string; label: string; f1Score: number; isCustom?: boolean }>;
  selectedModel: string;
  onModelSelect: (model: string) => void;
  onPredict: (features: ExoplanetFeatures, navigateToResults?: boolean) => void;
  onBatchPredict: (features: ExoplanetFeatures[]) => Promise<PredictionResult[]>;
  isTraining: boolean;
}

export function FeatureInputForm({ 
  models, 
  selectedModel, 
  onModelSelect, 
  onPredict, 
  onBatchPredict,
  isTraining
}: FeatureInputFormProps) {
  const [features, setFeatures] = useState<ExoplanetFeatures>({
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
    koi_insol: 0,
  });

  const handleInputChange = (key: keyof ExoplanetFeatures, value: string) => {
    setFeatures(prev => ({
      ...prev,
      [key]: parseFloat(value) || 0
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(features);
  };

  const generateRandomFeatures = (): ExoplanetFeatures => {
    // Generate realistic random values based on Kepler dataset ranges
    return {
      koi_score: Math.random() * 0.5 + 0.5, // 0.5-1.0
      planet_density_proxy: Math.random() * 8 + 0.5, // 0.5-8.5
      koi_model_snr: Math.random() * 50 + 5, // 5-55
      koi_fpflag_ss: Math.random() > 0.9 ? 1 : 0, // mostly 0
      koi_prad: Math.random() * 15 + 0.5, // 0.5-15.5 Earth radii
      koi_duration_err1: Math.random() * 0.8 + 0.05, // 0.05-0.85
      habitability_index: Math.random() * 0.9 + 0.1, // 0.1-1.0
      duration_period_ratio: Math.random() * 0.5, // 0-0.5
      koi_fpflag_co: Math.random() > 0.85 ? 1 : 0, // mostly 0
      koi_prad_err1: Math.random() * 0.5 + 0.05, // 0.05-0.55
      koi_time0bk_err1: Math.random() * 0.002, // 0-0.002
      koi_period: Math.random() * 300 + 1, // 1-301 days
      koi_steff_err2: -Math.random() * 150 - 20, // -170 to -20
      koi_steff_err1: Math.random() * 150 + 20, // 20-170
      koi_period_err1: Math.random() * 0.001, // 0-0.001
      koi_depth: Math.random() * 2000 + 50, // 50-2050 ppm
      koi_fpflag_nt: Math.random() > 0.9 ? 1 : 0, // mostly 0
      koi_impact: Math.random() * 1.2, // 0-1.2
      koi_slogg_err2: -Math.random() * 0.15 - 0.01, // -0.16 to -0.01
      koi_insol: Math.random() * 200 + 0.1 // 0.1-200.1 flux
    };
  };

  const loadExampleData = () => {
    const randomData = generateRandomFeatures();
    setFeatures(randomData);
    toast.success('Données aléatoires chargées!');
  };

  const fileInputRef = useRef<HTMLInputElement>(null);

  const downloadExampleCSV = () => {
    const featureKeys = Object.keys(FEATURE_DESCRIPTIONS);
    
    // Generate 7-10 random rows
    const numRows = Math.floor(Math.random() * 4) + 7; // 7-10 rows
    const exampleData = Array.from({ length: numRows }, () => generateRandomFeatures());

    const csvLines = [featureKeys.join(',')];
    exampleData.forEach(row => {
      csvLines.push(featureKeys.map(key => {
        const value = row[key as keyof ExoplanetFeatures];
        return typeof value === 'number' ? value.toFixed(6) : value;
      }).join(','));
    });

    const csvContent = csvLines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'kepler_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast.success(`Template CSV avec ${numRows} exemples aléatoires téléchargé!`);
  };

  const parseCSVLine = (line: string): string[] => {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current.trim());
    return result;
  };

  const handleCSVUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      toast.error('Veuillez télécharger un fichier CSV');
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split('\n').filter(line => line.trim());
        
        if (lines.length < 2) {
          toast.error('Le fichier CSV doit contenir au moins une ligne de données');
          return;
        }

        const headers = parseCSVLine(lines[0]).map(h => h.toLowerCase().trim());
        const featureKeys = Object.keys(FEATURE_DESCRIPTIONS);
        
        // Verify all required features are present
        const missingFeatures = featureKeys.filter(key => !headers.includes(key.toLowerCase()));
        if (missingFeatures.length > 0) {
          toast.error(`Colonnes manquantes: ${missingFeatures.join(', ')}`);
          return;
        }

        // Parse data rows
        const featuresArray: ExoplanetFeatures[] = [];
        for (let i = 1; i < lines.length; i++) {
          const values = parseCSVLine(lines[i]);
          if (values.length !== headers.length) continue;

          const rowFeatures: any = {};
          featureKeys.forEach(key => {
            const index = headers.indexOf(key.toLowerCase());
            if (index !== -1) {
              rowFeatures[key] = parseFloat(values[index]) || 0;
            }
          });
          featuresArray.push(rowFeatures as ExoplanetFeatures);
        }

        if (featuresArray.length === 0) {
          toast.error('Aucune donnée valide trouvée dans le CSV');
          return;
        }

        toast.success(`${featuresArray.length} lignes chargées. Traitement en cours...`);

        // Get predictions
        const results = await onBatchPredict(featuresArray);

        // Generate CSV with predictions automatically
        // featureKeys already declared above, so we reuse it
        const outputHeaders = [...featureKeys, 'prediction', 'prediction_label', 'confidence', 'model', 'timestamp'];
        const csvLines = [outputHeaders.join(',')];

        results.forEach((result, idx) => {
          const row = [
            ...featureKeys.map(key => featuresArray[idx][key as keyof ExoplanetFeatures]),
            result.prediction,
            `"${PREDICTION_LABELS[result.prediction]}"`,
            result.confidence.toFixed(4),
            `"${result.model}"`,
            result.timestamp.toISOString()
          ];
          csvLines.push(row.join(','));
        });

        const csvContent = csvLines.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Use original filename with _predictions suffix
        const originalName = file.name.replace('.csv', '');
        a.download = `${originalName}_predictions.csv`;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        toast.success(`${results.length} prédictions téléchargées: ${originalName}_predictions.csv`);
      } catch (error) {
        console.error('Error processing CSV:', error);
        toast.error('Erreur lors du traitement du fichier CSV');
      }
    };

    reader.readAsText(file);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const currentModel = models.find(m => m.value === selectedModel);

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2 space-y-6">
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Model Selection</CardTitle>
            <CardDescription className="text-slate-300">
              Choose a machine learning model for exoplanet classification
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Select value={selectedModel} onValueChange={onModelSelect}>
                <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent className="bg-slate-700 border-slate-600">
                  {models.map((model) => (
                    <SelectItem key={model.value} value={model.value} className="text-white">
                      <div className="flex items-center gap-2">
                        <span>{model.label}</span>
                        <div className="flex items-center gap-1">
                          {model.isCustom && (
                            <Badge variant="secondary" className="text-xs bg-purple-100 text-purple-800">
                              Personnalisé
                            </Badge>
                          )}
                          <Badge variant="outline" className="text-xs">
                            F1: {(model.f1Score * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <div className="grid gap-3 mt-4">
                <div className="grid grid-cols-2 gap-3">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={loadExampleData}
                    className="border-purple-600 text-purple-400 hover:bg-purple-900"
                  >
                    <Rocket className="w-4 h-4 mr-2" />
                    Load Example
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={downloadExampleCSV}
                    className="border-green-600 text-green-400 hover:bg-green-900"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    CSV Template
                  </Button>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isTraining}
                  className="border-blue-600 text-blue-400 hover:bg-blue-900 w-full"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload CSV for Batch Prediction
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleCSVUpload}
                  className="hidden"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Kepler Features Input</CardTitle>
            <CardDescription className="text-slate-300">
              Enter the 20 features from NASA Kepler dataset
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                {Object.entries(FEATURE_DESCRIPTIONS).map(([key, description]) => (
                  <div key={key} className="space-y-2">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label htmlFor={key} className="text-slate-300 flex items-center gap-1">
                            {description}
                            <Info className="w-3 h-3 text-slate-500" />
                          </Label>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs">{description}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <Input
                      id={key}
                      type="number"
                      step="any"
                      value={features[key as keyof ExoplanetFeatures]}
                      onChange={(e) => handleInputChange(key as keyof ExoplanetFeatures, e.target.value)}
                      className="bg-slate-700 border-slate-600 text-white"
                      placeholder="0.0"
                    />
                  </div>
                ))}
              </div>
              
              <Button
                type="submit"
                disabled={isTraining}
                className="w-full bg-purple-700 hover:bg-purple-600"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Classifying...
                  </>
                ) : (
                  <>
                    <Rocket className="w-4 h-4 mr-2" />
                    Classify Exoplanet
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-6">
        {currentModel && (
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                Modèle Sélectionné
                {currentModel.isCustom && (
                  <Badge variant="secondary" className="bg-purple-100 text-purple-800">
                    Personnalisé
                  </Badge>
                )}
              </CardTitle>
              <CardDescription className="text-slate-300">
                {currentModel.label}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-400">
                  {(currentModel.f1Score * 100).toFixed(2)}%
                </div>
                <div className="text-sm text-slate-400">F1-Macro Score</div>
              </div>
              
              <div className="text-xs text-slate-400 text-center">
                {currentModel.isCustom 
                  ? "Modèle personnalisé entraîné avec vos hyperparamètres"
                  : "Modèle optimisé pour la classification d'exoplanètes du dataset NASA Kepler"
                }
              </div>
            </CardContent>
          </Card>
        )}
        
        <CSVUploadHelp />
      </div>
    </div>
  );
}
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Settings, Save, RefreshCw, Play, RotateCcw, TrendingUp, Trash2 } from 'lucide-react';
import HyperparameterTuningService, { ModelConfig, TrainingProgress, TrainingResult } from '../utils/hyperparameterTuning';
import PredictionService from '../utils/predictionService';

export const HyperparameterTuning: React.FC = () => {
  const [selectedModelType, setSelectedModelType] = useState<string>('RandomForest');
  const [hyperparameters, setHyperparameters] = useState<any>({});
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [customModels, setCustomModels] = useState<TrainingResult[]>([]);
  const [useDefaultMetrics, setUseDefaultMetrics] = useState(true);

  const tuningService = HyperparameterTuningService.getInstance();
  const predictionService = PredictionService.getInstance();

  useEffect(() => {
    // Charger les hyperparamètres par défaut
    const defaults = tuningService.getDefaultHyperparameters(selectedModelType);
    setHyperparameters(defaults);
  }, [selectedModelType]);

  useEffect(() => {
    // Charger les modèles personnalisés existants
    const trained = tuningService.getAllTrainedModels();
    setCustomModels(trained);
  }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    setTrainingResult(null);
    
    const config: ModelConfig = {
      type: selectedModelType as any,
      hyperparameters
    };

    try {
      const result = await tuningService.trainCustomModel(config, (progress) => {
        setTrainingProgress(progress);
      });
      
      setTrainingResult(result);
      setCustomModels(prev => [...prev, result]);
      
      // Synchroniser avec le service de prédiction
      predictionService.addCustomModel(result);
    } catch (error) {
      console.error('Erreur lors de l\'entraînement:', error);
    } finally {
      setIsTraining(false);
      setTrainingProgress(null);
    }
  };

  const handleReset = () => {
    const defaults = tuningService.getDefaultHyperparameters(selectedModelType);
    setHyperparameters(defaults);
  };

  const resetToDefaults = () => {
    setUseDefaultMetrics(true);
    setTrainingResult(null);
    setCustomModels([]);
    
    // Nettoyer les modèles personnalisés du service de prédiction
    const customModels = predictionService.getCustomModels();
    customModels.forEach(model => {
      predictionService.removeCustomModel(model.modelId);
    });
  };

  const handleDeleteCustomModel = (modelId: string) => {
    setCustomModels(prev => prev.filter(model => model.modelId !== modelId));
    predictionService.removeCustomModel(modelId);
  };

  const updateHyperparameter = (key: string, value: any) => {
    setHyperparameters((prev: any) => ({
      ...prev,
      [key]: value
    }));
  };

  const renderHyperparameterInput = (key: string, value: any, type: string) => {
    switch (type) {
      case 'slider':
        return (
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>{key.replace(/_/g, ' ').toUpperCase()}</Label>
              <span className="text-sm text-muted-foreground">{value}</span>
            </div>
            <Slider
              value={[value]}
              onValueChange={([newValue]) => updateHyperparameter(key, newValue)}
              max={key === 'n_estimators' ? 500 : key === 'max_depth' ? 20 : 1}
              min={key === 'n_estimators' ? 10 : key === 'max_depth' ? 1 : 0}
              step={key === 'n_estimators' ? 10 : key === 'max_depth' ? 1 : 0.01}
              className="w-full"
            />
          </div>
        );
      
      case 'select':
        const options = key === 'kernel' 
          ? ['rbf', 'linear', 'poly', 'sigmoid']
          : key === 'criterion'
          ? ['gini', 'entropy']
          : key === 'solver'
          ? ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
          : key === 'weights'
          ? ['uniform', 'distance']
          : key === 'algorithm'
          ? ['auto', 'ball_tree', 'kd_tree', 'brute']
          : key === 'metric'
          ? ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
          : key === 'max_features'
          ? ['auto', 'sqrt', 'log2']
          : key === 'penalty'
          ? ['l1', 'l2', 'elasticnet', 'none']
          : ['auto', 'sqrt', 'log2'];
        
        return (
          <div className="space-y-2">
            <Label>{key.replace(/_/g, ' ').toUpperCase()}</Label>
            <Select value={String(value)} onValueChange={(newValue) => updateHyperparameter(key, newValue)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {options.map(option => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        );
      
      case 'switch':
        return (
          <div className="flex items-center justify-between">
            <Label>{key.replace(/_/g, ' ').toUpperCase()}</Label>
            <Switch
              checked={value}
              onCheckedChange={(checked) => updateHyperparameter(key, checked)}
            />
          </div>
        );
      
      default:
        return (
          <div className="space-y-2">
            <Label>{key.replace(/_/g, ' ').toUpperCase()}</Label>
            <Input
              type="number"
              step="any"
              value={value}
              onChange={(e) => updateHyperparameter(key, parseFloat(e.target.value) || 0)}
            />
          </div>
        );
    }
  };

  const getHyperparameterInputs = () => {
    const ranges = tuningService.getHyperparameterRanges(selectedModelType);
    return Object.keys(ranges).map(key => {
      const value = hyperparameters[key];
      let type = 'input';
      
      if (key === 'n_estimators' || key === 'max_depth' || key === 'n_neighbors') {
        type = 'slider';
      } else if (Array.isArray(ranges[key]) && typeof ranges[key][0] === 'string') {
        type = 'select';
      } else if (typeof value === 'boolean') {
        type = 'switch';
      }
      
      return { key, value, type };
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Réglage des Hyperparamètres</h2>
          <p className="text-muted-foreground">
            Entraînez vos propres modèles avec des hyperparamètres personnalisés
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={resetToDefaults}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Modèles Par Défaut
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Configuration du Modèle
          </CardTitle>
          <CardDescription>
            Sélectionnez le type de modèle et ajustez ses hyperparamètres
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <Label>Type de Modèle</Label>
              <Select value={selectedModelType} onValueChange={setSelectedModelType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="RandomForest">Random Forest</SelectItem>
                  <SelectItem value="XGBoost">XGBoost</SelectItem>
                  <SelectItem value="SVM">SVM</SelectItem>
                  <SelectItem value="KNN">K-Nearest Neighbors</SelectItem>
                  <SelectItem value="LogisticRegression">Régression Logistique</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {getHyperparameterInputs().map(({ key, value, type }) => (
                <div key={key}>
                  {renderHyperparameterInput(key, value, type)}
                </div>
              ))}
            </div>
            
            <div className="flex gap-4 pt-4 border-t">
              <Button
                onClick={handleTrain}
                disabled={isTraining}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {isTraining ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Entraînement...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Entraîner le Modèle
                  </>
                )}
              </Button>
              
              <Button variant="outline" onClick={handleReset}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Réinitialiser
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {isTraining && trainingProgress && (
        <Card>
          <CardHeader>
            <CardTitle>Progression de l'Entraînement</CardTitle>
            <CardDescription>{trainingProgress.stage}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Progress value={trainingProgress.progress} className="w-full" />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>{trainingProgress.progress.toFixed(1)}% terminé</span>
                <span>ETA: {trainingProgress.eta}s</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {trainingResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Résultats de l'Entraînement
            </CardTitle>
            <CardDescription>
              Performance du modèle personnalisé (Temps d'entraînement: {(trainingResult.trainingTime / 1000).toFixed(2)}s)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 border rounded">
                <div className="text-2xl font-bold text-green-600">
                  {(trainingResult.metrics.accuracy * 100).toFixed(2)}%
                </div>
                <div className="text-sm text-muted-foreground">Précision</div>
              </div>
              <div className="text-center p-4 border rounded">
                <div className="text-2xl font-bold text-blue-600">
                  {(trainingResult.metrics.f1_macro * 100).toFixed(2)}%
                </div>
                <div className="text-sm text-muted-foreground">F1-Macro</div>
              </div>
              <div className="text-center p-4 border rounded">
                <div className="text-2xl font-bold text-purple-600">
                  {(trainingResult.metrics.roc_auc * 100).toFixed(2)}%
                </div>
                <div className="text-sm text-muted-foreground">ROC-AUC</div>
              </div>
              <div className="text-center p-4 border rounded">
                <div className="text-2xl font-bold text-orange-600">
                  {(trainingResult.metrics.auc_score * 100).toFixed(2)}%
                </div>
                <div className="text-sm text-muted-foreground">AUC Score</div>
              </div>
            </div>

            <div className="mt-6">
              <h4 className="text-lg font-semibold mb-3">Matrice de Confusion</h4>
              <div className="grid grid-cols-4 gap-2 text-center text-sm">
                <div></div>
                <div className="font-medium">Faux Positif</div>
                <div className="font-medium">Candidat</div>
                <div className="font-medium">Confirmé</div>
                
                {trainingResult.metrics.confusion_matrix.map((row, i) => (
                  <React.Fragment key={i}>
                    <div className="font-medium py-2">
                      {['Faux Positif', 'Candidat', 'Confirmé'][i]}
                    </div>
                    {row.map((value, j) => (
                      <div
                        key={j}
                        className={`p-3 rounded text-sm ${
                          i === j 
                            ? 'bg-green-100 text-green-800 font-medium' 
                            : 'bg-red-50 text-red-600'
                        }`}
                      >
                        {value}
                      </div>
                    ))}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {customModels.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Modèles Personnalisés Entraînés</CardTitle>
            <CardDescription>
              Historique de vos modèles entraînés avec hyperparamètres personnalisés
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {customModels.map((model, index) => (
                <div key={model.modelId} className="flex items-center justify-between p-4 border rounded">
                  <div className="flex items-center gap-3">
                    <Badge variant="outline">#{customModels.length - index}</Badge>
                    <div>
                      <div className="font-medium">{model.modelId.split('_')[1]} Personnalisé</div>
                      <div className="text-sm text-muted-foreground">
                        ID: {model.modelId.slice(-8)}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="font-medium">{(model.metrics.f1_macro * 100).toFixed(2)}%</div>
                      <div className="text-xs text-muted-foreground">F1-Macro</div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDeleteCustomModel(model.modelId)}
                      className="text-red-600 hover:text-red-700 hover:bg-red-50"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
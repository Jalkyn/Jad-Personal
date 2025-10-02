import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Activity, Brain, Settings, Database, Upload } from 'lucide-react';
import { FeatureInputForm } from './FeatureInputForm';
import { ModelMetricsDisplay } from './ModelMetricsDisplay';
import { PredictionResults } from './PredictionResults';
import { HyperparameterTuning } from './HyperparameterTuning';
import { F1MetricsExplanation } from './F1MetricsExplanation';
import { AIAssistant } from './AIAssistant';
import { CSVBatchPrediction } from './CSVBatchPrediction';
import { ExoplanetFeatures, PredictionResult } from '../types/exoplanet';
import PredictionService from '../utils/predictionService';

export function ExoplanetClassifier() {
  const [selectedModel, setSelectedModel] = useState<string>('XGBoost_Top1');
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('predict');
  const [modelCount, setModelCount] = useState(25); // 5 types * 5 top models each
  const [availableModels, setAvailableModels] = useState<Array<{ value: string; label: string; f1Score: number; isCustom: boolean }>>([]);

  const predictionService = PredictionService.getInstance();

  useEffect(() => {
    // Charger les modèles au démarrage
    predictionService.loadModels()
      .then(() => {
        updateAvailableModels();
      })
      .catch(console.error);
  }, []);

  const updateAvailableModels = () => {
    predictionService.syncWithTuningService();
    const models = predictionService.getAllAvailableModels();
    setAvailableModels(models);
    setModelCount(models.length);
  };

  const handlePrediction = async (features: ExoplanetFeatures, navigateToResults = true) => {
    setIsLoading(true);
    
    try {
      const result = await predictionService.predict(features, selectedModel);
      setPredictions(prev => [result, ...prev.slice(0, 199)]); // Keep last 200 predictions
      
      // Navigate to results tab after prediction only if requested
      if (navigateToResults) {
        setActiveTab('results');
      }
    } catch (error) {
      console.error('Erreur lors de la prédiction:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchPrediction = async (featuresArray: ExoplanetFeatures[]) => {
    setIsLoading(true);
    
    try {
      const results = await predictionService.predictBatch(featuresArray, selectedModel);
      // Note: batch predictions are not added to history for CSV processing
      setActiveTab('results');
      return results;
    } catch (error) {
      console.error('Erreur lors des prédictions par lot:', error);
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = (predictionIndex: number, feedback: 'correct' | 'incorrect' | 'unknown') => {
    if (feedback === 'unknown') return;

    setPredictions(prev => {
      const updated = [...prev];
      if (updated[predictionIndex]) {
        updated[predictionIndex].userFeedback = feedback;
      }
      return updated;
    });

    // In a real implementation, this feedback would be sent to the backend
    // to improve model performance
    console.log(`Feedback received for prediction ${predictionIndex}: ${feedback}`);
  };

  const handleClearHistory = () => {
    setPredictions([]);
  };

  // Mettre à jour les modèles quand on change d'onglet
  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    if (tab === 'predict') {
      updateAvailableModels();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-white mb-4">
            Système de Classification d'Exoplanètes
          </h1>
          <p className="text-xl text-purple-200 mb-6">
            Plateforme ML basée sur le Dataset NASA Kepler - Modèles Optimisés F1-Macro
          </p>
          <div className="flex justify-center gap-4">
            <Badge variant="secondary" className="bg-purple-800 text-purple-100">
              <Database className="w-4 h-4 mr-1" />
              {predictions.length} Prédictions Effectuées
            </Badge>
            <Badge variant="secondary" className="bg-blue-800 text-blue-100">
              <Brain className="w-4 h-4 mr-1" />
              {modelCount} Modèles ML Disponibles
            </Badge>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 bg-slate-800 border-slate-700">
            <TabsTrigger value="predict" className="data-[state=active]:bg-purple-700">
              <Activity className="w-4 h-4 mr-2" />
              Prédire
            </TabsTrigger>
            <TabsTrigger value="batch" className="data-[state=active]:bg-purple-700">
              <Upload className="w-4 h-4 mr-2" />
              CSV Batch
            </TabsTrigger>
            <TabsTrigger value="metrics" className="data-[state=active]:bg-purple-700">
              <Brain className="w-4 h-4 mr-2" />
              Métriques
            </TabsTrigger>
            <TabsTrigger value="results" className="data-[state=active]:bg-purple-700">
              <Database className="w-4 h-4 mr-2" />
              Résultats
            </TabsTrigger>
            <TabsTrigger value="tuning" className="data-[state=active]:bg-purple-700">
              <Settings className="w-4 h-4 mr-2" />
              Hyperparamètres
            </TabsTrigger>
          </TabsList>

          <TabsContent value="predict" className="space-y-6">
            <FeatureInputForm
              models={availableModels}
              selectedModel={selectedModel}
              onModelSelect={setSelectedModel}
              onPredict={handlePrediction}
              onBatchPredict={handleBatchPrediction}
              isTraining={isLoading}
            />
          </TabsContent>

          <TabsContent value="batch" className="space-y-6">
            <CSVBatchPrediction />
          </TabsContent>

          <TabsContent value="metrics" className="space-y-6">
            <F1MetricsExplanation />
            <ModelMetricsDisplay />
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <PredictionResults
              predictions={predictions}
              onFeedback={handleFeedback}
              onClearHistory={handleClearHistory}
            />
          </TabsContent>

          <TabsContent value="tuning" className="space-y-6">
            <HyperparameterTuning />
          </TabsContent>
        </Tabs>

        {/* AI Assistant */}
        <AIAssistant />
      </div>
    </div>
  );
}
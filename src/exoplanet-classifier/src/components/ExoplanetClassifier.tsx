import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Activity, Brain, Settings, Database } from 'lucide-react';
import { FeatureInputForm } from './FeatureInputForm';
import { ModelMetricsDisplay } from './ModelMetricsDisplay';
import { PredictionResults } from './PredictionResults';
import { HyperparameterTuning } from './HyperparameterTuning';
import { F1MetricsExplanation } from './F1MetricsExplanation';
import { MLModel, ExoplanetFeatures, PredictionResult } from '../types/exoplanet';
import { createMockModels, predictWithModel, updateModelMetrics } from '../utils/mockMLModels';

export function ExoplanetClassifier() {
  const [models, setModels] = useState<MLModel[]>(createMockModels());
  const [selectedModel, setSelectedModel] = useState<string>('XGBoost'); // Start with best performing model
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  const handlePrediction = async (features: ExoplanetFeatures) => {
    const model = models.find(m => m.name === selectedModel);
    if (!model) return;

    setIsTraining(true);
    
    // Simulate prediction delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const result = predictWithModel(model, features);
    setPredictions(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 predictions
    
    setIsTraining(false);
  };

  const handleFeedback = (predictionIndex: number, feedback: 'correct' | 'incorrect' | 'unknown') => {
    if (feedback === 'unknown') return;

    setPredictions(prev => {
      const updated = [...prev];
      updated[predictionIndex].userFeedback = feedback;
      return updated;
    });

    // Update model metrics based on feedback
    const prediction = predictions[predictionIndex];
    const isCorrect = feedback === 'correct';
    
    setModels(prev => prev.map(model => {
      if (model.name === prediction.model) {
        return updateModelMetrics(model, isCorrect);
      }
      return model;
    }));
  };

  const handleHyperparameterUpdate = (modelName: string, hyperparameters: any) => {
    setModels(prev => prev.map(model => {
      if (model.name === modelName) {
        return { ...model, hyperparameters };
      }
      return model;
    }));
  };

  const currentModel = models.find(m => m.name === selectedModel);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-white mb-4">
            Exoplanet Classification System
          </h1>
          <p className="text-xl text-purple-200 mb-6">
            NASA Kepler Dataset ML Platform - F1-Macro Optimized Models
          </p>
          <div className="flex justify-center gap-4">
            <Badge variant="secondary" className="bg-purple-800 text-purple-100">
              <Database className="w-4 h-4 mr-1" />
              {predictions.length} Predictions Made
            </Badge>
            <Badge variant="secondary" className="bg-blue-800 text-blue-100">
              <Brain className="w-4 h-4 mr-1" />
              5 ML Models Available
            </Badge>
          </div>
        </div>

        <Tabs defaultValue="predict" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-slate-800 border-slate-700">
            <TabsTrigger value="predict" className="data-[state=active]:bg-purple-700">
              <Activity className="w-4 h-4 mr-2" />
              Predict
            </TabsTrigger>
            <TabsTrigger value="metrics" className="data-[state=active]:bg-purple-700">
              <Brain className="w-4 h-4 mr-2" />
              Model Metrics
            </TabsTrigger>
            <TabsTrigger value="results" className="data-[state=active]:bg-purple-700">
              <Database className="w-4 h-4 mr-2" />
              Results
            </TabsTrigger>
            <TabsTrigger value="tuning" className="data-[state=active]:bg-purple-700">
              <Settings className="w-4 h-4 mr-2" />
              Hyperparameters
            </TabsTrigger>
          </TabsList>

          <TabsContent value="predict" className="space-y-6">
            <FeatureInputForm
              models={models}
              selectedModel={selectedModel}
              onModelSelect={setSelectedModel}
              onPredict={handlePrediction}
              isTraining={isTraining}
            />
          </TabsContent>

          <TabsContent value="metrics" className="space-y-6">
            <F1MetricsExplanation />
            <ModelMetricsDisplay models={models} />
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <PredictionResults
              predictions={predictions}
              onFeedback={handleFeedback}
            />
          </TabsContent>

          <TabsContent value="tuning" className="space-y-6">
            <HyperparameterTuning
              models={models}
              onUpdate={handleHyperparameterUpdate}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
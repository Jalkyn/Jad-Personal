import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line } from 'recharts';
import { Download, Refresh, Info, RotateCcw } from 'lucide-react';
import PredictionService from '../utils/predictionService';

interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_macro: number;
  f1_weighted: number;
  roc_auc: number;
  auc_score: number;
  confusion_matrix: number[][];
  hyperparameters: { [key: string]: any };
  rank: number;
}

export const ModelMetricsDisplay: React.FC = () => {
  const [allMetrics, setAllMetrics] = useState<ModelMetrics[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [comparisonView, setComparisonView] = useState<'overview' | 'detailed' | 'roc_analysis'>('overview');
  const [loading, setLoading] = useState(true);
  const [useDefaultMetrics, setUseDefaultMetrics] = useState(true);

  const predictionService = PredictionService.getInstance();

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    setLoading(true);
    try {
      await predictionService.loadModels();
      const metrics = predictionService.getAllMetrics();
      setAllMetrics(metrics);
      
      if (metrics.length > 0 && !selectedModel) {
        setSelectedModel(metrics[0].model_name);
      }
    } catch (error) {
      console.error('Erreur lors du chargement des métriques:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCurrentModel = (): ModelMetrics | undefined => {
    return allMetrics.find(model => model.model_name === selectedModel);
  };

  const getComparisonData = () => {
    return allMetrics.map(model => ({
      name: model.model_name.replace(/_Top\d+/, ''),
      type: model.model_name.split('_')[0],
      accuracy: (model.accuracy * 100).toFixed(1),
      f1_macro: (model.f1_macro * 100).toFixed(1),
      f1_weighted: (model.f1_weighted * 100).toFixed(1),
      precision: (model.precision * 100).toFixed(1),
      recall: (model.recall * 100).toFixed(1),
      roc_auc: (model.roc_auc * 100).toFixed(1),
      auc_score: (model.auc_score * 100).toFixed(1),
      rank: model.rank
    }));
  };

  const getModelTypeComparison = () => {
    const types = ['RandomForest', 'XGBoost', 'SVM', 'KNN', 'LogisticRegression'];
    
    return types.map(type => {
      const modelsOfType = allMetrics.filter(m => m.model_name.startsWith(type));
      const bestModel = modelsOfType.reduce((best, current) => 
        current.f1_macro > best.f1_macro ? current : best
      );
      
      return {
        type,
        best_accuracy: (bestModel.accuracy * 100).toFixed(1),
        best_f1_macro: (bestModel.f1_macro * 100).toFixed(1),
        best_roc_auc: (bestModel.roc_auc * 100).toFixed(1),
        model_count: modelsOfType.length
      };
    });
  };

  const getRadarData = (metrics: ModelMetrics) => {
    return [
      { metric: 'Accuracy', value: metrics.accuracy * 100 },
      { metric: 'Precision', value: metrics.precision * 100 },
      { metric: 'Recall', value: metrics.recall * 100 },
      { metric: 'F1-Macro', value: metrics.f1_macro * 100 },
      { metric: 'F1-Weighted', value: metrics.f1_weighted * 100 },
      { metric: 'ROC-AUC', value: metrics.roc_auc * 100 }
    ];
  };

  const refreshMetrics = async () => {
    await loadMetrics();
  };

  const exportMetrics = () => {
    const data = allMetrics.map(model => ({
      name: model.model_name,
      accuracy: model.accuracy,
      precision: model.precision,
      recall: model.recall,
      f1_macro: model.f1_macro,
      f1_weighted: model.f1_weighted,
      roc_auc: model.roc_auc,
      auc_score: model.auc_score,
      hyperparameters: model.hyperparameters,
      confusion_matrix: model.confusion_matrix
    }));
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'exoplanet_model_metrics.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetToDefaults = () => {
    setUseDefaultMetrics(true);
    loadMetrics();
  };

  const currentModel = getCurrentModel();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-muted-foreground">Chargement des métriques...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Métriques de Performance des Modèles</h2>
          <p className="text-muted-foreground">
            Analysez et comparez les performances des modèles ML pour la classification d'exoplanètes
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={resetToDefaults}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Défaut
          </Button>
          <Button variant="outline" onClick={refreshMetrics}>
            <Refresh className="w-4 h-4 mr-2" />
            Actualiser
          </Button>
          <Button variant="outline" onClick={exportMetrics}>
            <Download className="w-4 h-4 mr-2" />
            Exporter
          </Button>
        </div>
      </div>

      <Tabs value={comparisonView} onValueChange={(value) => setComparisonView(value as any)}>
        <TabsList>
          <TabsTrigger value="overview">Vue d'ensemble</TabsTrigger>
          <TabsTrigger value="detailed">Analyse détaillée</TabsTrigger>
          <TabsTrigger value="roc_analysis">Analyse ROC/AUC</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Comparaison par Type de Modèle</CardTitle>
                <CardDescription>Meilleure performance par type de modèle</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={getModelTypeComparison()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="best_f1_macro" fill="#8884d8" name="F1-Macro %" />
                    <Bar dataKey="best_roc_auc" fill="#82ca9d" name="ROC-AUC %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top 5 Modèles Globaux</CardTitle>
                <CardDescription>Classés par F1-Macro Score</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {allMetrics
                    .sort((a, b) => b.f1_macro - a.f1_macro)
                    .slice(0, 5)
                    .map((model, index) => (
                    <div key={model.model_name} className="flex items-center justify-between p-3 border rounded">
                      <div className="flex items-center gap-3">
                        <Badge variant={index === 0 ? "default" : "secondary"}>#{index + 1}</Badge>
                        <div>
                          <div className="font-medium">{model.model_name}</div>
                          <div className="text-sm text-muted-foreground">
                            {model.model_name.split('_')[0]}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">{(model.f1_macro * 100).toFixed(2)}%</div>
                        <div className="text-xs text-muted-foreground">F1-Macro</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Meilleur Modèle Global</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {allMetrics.reduce((best, current) => 
                    current.f1_macro > best.f1_macro ? current : best
                  ).model_name}
                </div>
                <p className="text-xs text-muted-foreground">
                  F1-Macro: {(allMetrics.reduce((best, current) => 
                    current.f1_macro > best.f1_macro ? current : best
                  ).f1_macro * 100).toFixed(2)}%
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Précision Moyenne</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(allMetrics.reduce((acc, model) => acc + model.accuracy, 0) / allMetrics.length * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">
                  Sur {allMetrics.length} modèles
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">ROC-AUC Moyen</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(allMetrics.reduce((acc, model) => acc + model.roc_auc, 0) / allMetrics.length * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">
                  Performance discriminante
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Types de Modèles</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {new Set(allMetrics.map(m => m.model_name.split('_')[0])).size}
                </div>
                <p className="text-xs text-muted-foreground">
                  Algorithmes différents
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="detailed" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Analyse Détaillée du Modèle</CardTitle>
                  <CardDescription>Performance complète et hyperparamètres</CardDescription>
                </div>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-80">
                    <SelectValue placeholder="Sélectionner un modèle" />
                  </SelectTrigger>
                  <SelectContent>
                    {allMetrics.map(model => (
                      <SelectItem key={model.model_name} value={model.model_name}>
                        {model.model_name} (Rang: {model.rank})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              {currentModel && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold mb-4">Radar des Performances</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <RadarChart data={getRadarData(currentModel)}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis domain={[0, 100]} />
                        <Radar
                          name={currentModel.model_name}
                          dataKey="value"
                          stroke="#8884d8"
                          fill="#8884d8"
                          fillOpacity={0.6}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>

                  <div>
                    <h4 className="text-lg font-semibold mb-4">Métriques Détaillées</h4>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <div className="text-sm text-muted-foreground">Type de Modèle</div>
                          <Badge variant="outline">{currentModel.model_name.split('_')[0]}</Badge>
                        </div>
                        <div className="space-y-2">
                          <div className="text-sm text-muted-foreground">Rang</div>
                          <Badge variant={currentModel.rank <= 3 ? "default" : "secondary"}>
                            #{currentModel.rank}
                          </Badge>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-sm">Précision (Accuracy)</span>
                          <span className="font-medium">{(currentModel.accuracy * 100).toFixed(3)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Précision (Precision)</span>
                          <span className="font-medium">{(currentModel.precision * 100).toFixed(3)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Rappel (Recall)</span>
                          <span className="font-medium">{(currentModel.recall * 100).toFixed(3)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">F1-Score Macro</span>
                          <span className="font-medium">{(currentModel.f1_macro * 100).toFixed(3)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">F1-Score Weighted</span>
                          <span className="font-medium">{(currentModel.f1_weighted * 100).toFixed(3)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">ROC-AUC</span>
                          <span className="font-medium">{(currentModel.roc_auc * 100).toFixed(3)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">AUC Score</span>
                          <span className="font-medium">{(currentModel.auc_score * 100).toFixed(3)}%</span>
                        </div>
                      </div>

                      <div className="pt-4">
                        <h5 className="text-sm font-medium mb-2">Hyperparamètres</h5>
                        <div className="text-xs text-muted-foreground space-y-1 max-h-32 overflow-y-auto">
                          {Object.entries(currentModel.hyperparameters).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span>{key}:</span>
                              <span className="font-mono">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="w-4 h-4" />
                Matrice de Confusion
              </CardTitle>
              <CardDescription>Analyse détaillée des prédictions par classe</CardDescription>
            </CardHeader>
            <CardContent>
              {currentModel && (
                <div className="space-y-4">
                  <div className="grid grid-cols-4 gap-2 text-center text-sm">
                    <div></div>
                    <div className="font-medium">Faux Positif</div>
                    <div className="font-medium">Candidat</div>
                    <div className="font-medium">Confirmé</div>
                    
                    {currentModel.confusion_matrix.map((row, i) => (
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
                  
                  <div className="text-xs text-muted-foreground">
                    Les lignes représentent les classes réelles, les colonnes les prédictions.
                    Les valeurs diagonales indiquent les prédictions correctes.
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="roc_analysis" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Analyse ROC-AUC et AUC Score</CardTitle>
              <CardDescription>
                Compréhension des métriques de performance discriminante
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-semibold mb-4">Comparaison ROC-AUC</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={getComparisonData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[85, 100]} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="roc_auc" fill="#8884d8" name="ROC-AUC %" />
                      <Bar dataKey="auc_score" fill="#82ca9d" name="AUC Score %" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="space-y-6">
                  <div>
                    <h4 className="text-lg font-semibold mb-3">Explication ROC-AUC</h4>
                    <div className="bg-blue-50 p-4 rounded-lg space-y-2 text-sm">
                      <p><strong>ROC-AUC (Receiver Operating Characteristic - Area Under Curve)</strong></p>
                      <p>• Mesure la capacité du modèle à distinguer entre les classes</p>
                      <p>• Valeur entre 0 et 1 (1 = discrimination parfaite)</p>
                      <p>• Insensible au déséquilibre des classes</p>
                      <p>• ROC-AUC > 0.9 = Excellent</p>
                      <p>• ROC-AUC > 0.8 = Bon</p>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-lg font-semibold mb-3">Explication AUC Score</h4>
                    <div className="bg-green-50 p-4 rounded-lg space-y-2 text-sm">
                      <p><strong>AUC Score (Area Under Curve)</strong></p>
                      <p>• Aire sous la courbe de précision-rappel</p>
                      <p>• Plus sensible aux classes minoritaires</p>
                      <p>• Utile pour les datasets déséquilibrés</p>
                      <p>• Complément important au ROC-AUC</p>
                      <p>• AUC > 0.95 = Performance excellente</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6">
                <h4 className="text-lg font-semibold mb-4">Classement par Performance ROC-AUC</h4>
                <div className="space-y-2">
                  {allMetrics
                    .sort((a, b) => b.roc_auc - a.roc_auc)
                    .slice(0, 10)
                    .map((model, index) => (
                    <div key={model.model_name} className="flex items-center justify-between p-3 border rounded">
                      <div className="flex items-center gap-3">
                        <Badge variant={index < 3 ? "default" : "secondary"}>#{index + 1}</Badge>
                        <div>
                          <div className="font-medium">{model.model_name}</div>
                          <div className="text-sm text-muted-foreground">
                            {model.model_name.split('_')[0]}
                          </div>
                        </div>
                      </div>
                      <div className="flex gap-4 text-right">
                        <div>
                          <div className="font-medium">{(model.roc_auc * 100).toFixed(3)}%</div>
                          <div className="text-xs text-muted-foreground">ROC-AUC</div>
                        </div>
                        <div>
                          <div className="font-medium">{(model.auc_score * 100).toFixed(3)}%</div>
                          <div className="text-xs text-muted-foreground">AUC Score</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
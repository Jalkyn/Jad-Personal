import React, { useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { Upload, Download, FileText, AlertCircle, CheckCircle, Loader2, BarChart3 } from 'lucide-react';
import CSVPredictionService, { CSVPredictionRow, CSVBatchResult } from '../utils/csvPredictionService';
import PredictionService from '../utils/predictionService';

export const CSVBatchPrediction: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<CSVPredictionRow[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [results, setResults] = useState<CSVBatchResult | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('XGBoost_Top1');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [processingStatus, setProcessingStatus] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const csvService = CSVPredictionService.getInstance();
  const predictionService = PredictionService.getInstance();

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setValidationErrors(['Veuillez sélectionner un fichier CSV']);
      return;
    }

    setSelectedFile(file);
    setValidationErrors([]);
    setCsvData([]);
    setResults(null);

    try {
      const rows = await csvService.parseCSVFile(file);
      setCsvData(rows);

      if (rows.length === 0) {
        setValidationErrors(['Aucune donnée valide trouvée dans le fichier CSV']);
        return;
      }

      // Valider le format
      const firstRow = rows[0];
      const headers = Object.keys(firstRow.originalRow);
      const validation = csvService.validateCSVFormat(headers);
      
      if (!validation.isValid) {
        setValidationErrors(validation.issues);
      } else {
        setValidationErrors([]);
      }

    } catch (error) {
      console.error('Erreur lors du parsing CSV:', error);
      setValidationErrors([
        error instanceof Error ? error.message : 'Erreur lors de la lecture du fichier'
      ]);
    }
  };

  const handleBatchPrediction = async () => {
    if (!csvData.length) return;

    setIsProcessing(true);
    setProcessingProgress(0);
    setResults(null);
    setProcessingStatus('Initialisation...');

    try {
      // Charger les modèles si nécessaire
      await predictionService.loadModels();
      
      const batchResults = await csvService.processBatchPredictions(
        csvData,
        selectedModel,
        (progress, processed, total) => {
          setProcessingProgress(progress);
          setProcessingStatus(`Traitement: ${processed}/${total} échantillons`);
        }
      );

      setResults(batchResults);
      setProcessingStatus('Terminé !');
    } catch (error) {
      console.error('Erreur lors des prédictions par lot:', error);
      setValidationErrors([
        error instanceof Error ? error.message : 'Erreur lors du traitement'
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadResults = () => {
    if (!results || !csvData.length) return;

    csvService.downloadPredictions(
      results,
      csvData,
      `predictions_${selectedFile?.name?.replace('.csv', '')}_${new Date().toISOString().split('T')[0]}.csv`
    );
  };

  const handleSelectFile = () => {
    fileInputRef.current?.click();
  };

  const resetForm = () => {
    setSelectedFile(null);
    setCsvData([]);
    setResults(null);
    setValidationErrors([]);
    setProcessingProgress(0);
    setProcessingStatus('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getAvailableModels = () => {
    // Retourner la liste des modèles disponibles
    return [
      { value: 'XGBoost_Top1', label: 'XGBoost (Top 1) - Recommandé' },
      { value: 'XGBoost_Top2', label: 'XGBoost (Top 2)' },
      { value: 'XGBoost_Top3', label: 'XGBoost (Top 3)' },
      { value: 'RandomForest_Top1', label: 'Random Forest (Top 1)' },
      { value: 'RandomForest_Top2', label: 'Random Forest (Top 2)' },
      { value: 'SVM_Top1', label: 'SVM (Top 1)' },
      { value: 'KNN_Top1', label: 'KNN (Top 1)' },
      { value: 'LogisticRegression_Top1', label: 'Régression Logistique (Top 1)' }
    ];
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Prédictions par Lot (CSV)</h2>
          <p className="text-muted-foreground">
            Téléchargez un fichier CSV pour obtenir des prédictions sur plusieurs exoplanètes
          </p>
        </div>
        <Button variant="outline" onClick={resetForm}>
          Réinitialiser
        </Button>
      </div>

      {/* Sélection de fichier */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Téléchargement de Fichier CSV
          </CardTitle>
          <CardDescription>
            Téléchargez un fichier CSV contenant les caractéristiques d'exoplanètes pour des prédictions par lot
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <Button onClick={handleSelectFile} variant="outline">
                <FileText className="w-4 h-4 mr-2" />
                Sélectionner un fichier CSV
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                className="hidden"
              />
              {selectedFile && (
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  <span className="text-sm">{selectedFile.name}</span>
                  <Badge variant="secondary">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </Badge>
                </div>
              )}
            </div>

            {validationErrors.length > 0 && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  <div className="space-y-1">
                    {validationErrors.map((error, index) => (
                      <div key={index}>• {error}</div>
                    ))}
                  </div>
                </AlertDescription>
              </Alert>
            )}

            {csvData.length > 0 && validationErrors.length === 0 && (
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertDescription>
                  Fichier CSV chargé avec succès ! {csvData.length} échantillons trouvés.
                </AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Configuration et traitement */}
      {csvData.length > 0 && validationErrors.length === 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Configuration des Prédictions</CardTitle>
            <CardDescription>
              Sélectionnez le modèle à utiliser pour les prédictions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">
                  Modèle de Prédiction
                </label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {getAvailableModels().map((model) => (
                      <SelectItem key={model.value} value={model.value}>
                        {model.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex justify-between items-center pt-4 border-t">
                <div className="text-sm text-muted-foreground">
                  Prêt à traiter {csvData.length} échantillons
                </div>
                <Button 
                  onClick={handleBatchPrediction}
                  disabled={isProcessing || csvData.length === 0}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Traitement...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="w-4 h-4 mr-2" />
                      Lancer les Prédictions
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Progression */}
      {isProcessing && (
        <Card>
          <CardHeader>
            <CardTitle>Progression du Traitement</CardTitle>
            <CardDescription>{processingStatus}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Progress value={processingProgress} className="w-full" />
              <div className="text-center text-sm text-muted-foreground">
                {processingProgress.toFixed(1)}% terminé
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Résultats */}
      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              Résultats des Prédictions
            </CardTitle>
            <CardDescription>
              Traitement terminé avec succès
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Résumé */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 border rounded">
                  <div className="text-2xl font-bold text-blue-600">
                    {results.summary.totalRows}
                  </div>
                  <div className="text-sm text-muted-foreground">Total</div>
                </div>
                <div className="text-center p-4 border rounded">
                  <div className="text-2xl font-bold text-green-600">
                    {results.summary.successfulPredictions}
                  </div>
                  <div className="text-sm text-muted-foreground">Succès</div>
                </div>
                <div className="text-center p-4 border rounded">
                  <div className="text-2xl font-bold text-red-600">
                    {results.summary.errorCount}
                  </div>
                  <div className="text-sm text-muted-foreground">Erreurs</div>
                </div>
                <div className="text-center p-4 border rounded">
                  <div className="text-2xl font-bold text-purple-600">
                    {((results.summary.successfulPredictions / results.summary.totalRows) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Taux de Succès</div>
                </div>
              </div>

              <Separator />

              {/* Répartition des prédictions */}
              <div>
                <h4 className="text-lg font-semibold mb-3">Répartition des Prédictions</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 border rounded">
                    <div className="text-3xl font-bold text-red-500">
                      {results.summary.predictionCounts[0]}
                    </div>
                    <div className="text-sm text-muted-foreground">Faux Positifs</div>
                    <div className="text-xs text-muted-foreground">
                      ({((results.summary.predictionCounts[0] / results.summary.successfulPredictions) * 100).toFixed(1)}%)
                    </div>
                  </div>
                  <div className="text-center p-4 border rounded">
                    <div className="text-3xl font-bold text-orange-500">
                      {results.summary.predictionCounts[1]}
                    </div>
                    <div className="text-sm text-muted-foreground">Candidats</div>
                    <div className="text-xs text-muted-foreground">
                      ({((results.summary.predictionCounts[1] / results.summary.successfulPredictions) * 100).toFixed(1)}%)
                    </div>
                  </div>
                  <div className="text-center p-4 border rounded">
                    <div className="text-3xl font-bold text-green-500">
                      {results.summary.predictionCounts[2]}
                    </div>
                    <div className="text-sm text-muted-foreground">Confirmés</div>
                    <div className="text-xs text-muted-foreground">
                      ({((results.summary.predictionCounts[2] / results.summary.successfulPredictions) * 100).toFixed(1)}%)
                    </div>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Actions */}
              <div className="flex gap-4">
                <Button onClick={handleDownloadResults} className="bg-green-600 hover:bg-green-700">
                  <Download className="w-4 h-4 mr-2" />
                  Télécharger les Résultats CSV
                </Button>
                <Button variant="outline" onClick={resetForm}>
                  Nouveau Fichier
                </Button>
              </div>

              {/* Erreurs détaillées */}
              {results.errors.length > 0 && (
                <div>
                  <h4 className="text-lg font-semibold mb-3">Erreurs Détaillées</h4>
                  <div className="max-h-48 overflow-y-auto space-y-2">
                    {results.errors.slice(0, 10).map((error, index) => (
                      <Alert key={index} variant="destructive">
                        <AlertDescription>
                          <span className="font-medium">Ligne {error.row}:</span> {error.error}
                        </AlertDescription>
                      </Alert>
                    ))}
                    {results.errors.length > 10 && (
                      <div className="text-sm text-muted-foreground text-center">
                        ... et {results.errors.length - 10} erreurs supplémentaires
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Instructions */}
      <Card>
        <CardHeader>
          <CardTitle>Format de Fichier CSV Requis</CardTitle>
          <CardDescription>
            Instructions pour préparer votre fichier CSV
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-2">Colonnes Requises:</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                <div>koi_score</div>
                <div>planet_density_proxy</div>
                <div>koi_model_snr</div>
                <div>koi_fpflag_ss</div>
                <div>koi_prad</div>
                <div>koi_duration_err1</div>
                <div>habitability_index</div>
                <div>duration_period_ratio</div>
                <div>koi_fpflag_co</div>
                <div>koi_prad_err1</div>
                <div>koi_time0bk_err1</div>
                <div>koi_period</div>
                <div>koi_steff_err2</div>
                <div>koi_steff_err1</div>
                <div>koi_period_err1</div>
                <div>koi_depth</div>
                <div>koi_fpflag_nt</div>
                <div>koi_impact</div>
                <div>koi_slogg_err2</div>
                <div>koi_insol</div>
              </div>
            </div>
            <div>
              <h4 className="font-medium mb-2">Notes Importantes:</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Les en-têtes doivent correspondre exactement aux noms des colonnes</li>
                <li>• Les valeurs doivent être numériques</li>
                <li>• Les valeurs manquantes seront remplacées par 0</li>
                <li>• Le fichier doit contenir au moins 80% des colonnes requises</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Info, Download, Upload, FileCheck } from 'lucide-react';
import { Alert, AlertDescription } from './ui/alert';

export function CSVUploadHelp() {
  return (
    <Card className="bg-slate-800 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-400" />
          Guide d'Upload CSV
        </CardTitle>
        <CardDescription className="text-slate-300">
          Comment utiliser la fonctionnalité de prédiction par lot
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert className="bg-blue-900/20 border-blue-700/30">
          <AlertDescription className="text-slate-300">
            <div className="space-y-3">
              <div className="flex items-start gap-2">
                <Download className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-green-400 mb-1">Étape 1: Télécharger le template CSV</h4>
                  <p className="text-sm">
                    Cliquez sur "Download CSV Template" pour obtenir un fichier CSV exemple avec toutes les colonnes requises.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-2">
                <FileCheck className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-yellow-400 mb-1">Étape 2: Remplir vos données</h4>
                  <p className="text-sm">
                    Ouvrez le fichier CSV et ajoutez vos données d'exoplanètes. Assurez-vous que toutes les 20 colonnes sont présentes.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-2">
                <Upload className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-blue-400 mb-1">Étape 3: Upload et prédiction</h4>
                  <p className="text-sm">
                    Cliquez sur "Upload CSV Dataset for Batch Prediction". Le système va automatiquement :
                  </p>
                  <ul className="text-sm mt-2 space-y-1 ml-4">
                    <li>• Charger et valider vos données</li>
                    <li>• Effectuer les prédictions avec le modèle sélectionné</li>
                    <li>• Générer un nouveau CSV avec les résultats</li>
                    <li>• Télécharger automatiquement le fichier avec les prédictions</li>
                  </ul>
                </div>
              </div>
            </div>
          </AlertDescription>
        </Alert>

        <div className="p-4 bg-purple-900/20 border border-purple-700/30 rounded-lg">
          <h4 className="font-semibold text-purple-400 mb-2">Colonnes dans le CSV de résultats :</h4>
          <ul className="text-sm text-slate-300 space-y-1">
            <li>• Toutes les 20 features originales</li>
            <li>• <strong>prediction</strong> : 0 (Faux Positif), 1 (Candidat), 2 (Confirmé)</li>
            <li>• <strong>prediction_label</strong> : Label textuel de la prédiction</li>
            <li>• <strong>confidence</strong> : Niveau de confiance (0-1)</li>
            <li>• <strong>model</strong> : Nom du modèle utilisé</li>
            <li>• <strong>timestamp</strong> : Date et heure de la prédiction</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
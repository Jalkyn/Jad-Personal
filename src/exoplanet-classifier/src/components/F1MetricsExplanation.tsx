import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Info, Target, BarChart3, Users } from 'lucide-react';
import { Badge } from './ui/badge';

export function F1MetricsExplanation() {
  return (
    <Card className="bg-slate-800 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-400" />
          Guide des Métriques F1
        </CardTitle>
        <CardDescription className="text-slate-300">
          Comprendre F1-Macro vs F1-Weighted pour la classification d'exoplanètes
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* F1-Macro Section */}
        <div className="p-4 bg-orange-900/20 border border-orange-700/30 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <Target className="w-5 h-5 text-orange-400" />
            <h3 className="text-lg font-semibold text-orange-400">F1-Macro</h3>
            <Badge variant="outline" className="text-orange-400 border-orange-400">
              Équité des classes
            </Badge>
          </div>
          <div className="space-y-2 text-sm">
            <p className="text-slate-300">
              <strong>Calcul :</strong> Moyenne arithmétique simple des F1 de chaque classe
            </p>
            <p className="text-slate-300">
              <strong>Formule :</strong> (F1_classe0 + F1_classe1 + F1_classe2) / 3
            </p>
            <p className="text-slate-300">
              <strong>Avantage :</strong> Traite toutes les classes de manière égale, même les classes rares
            </p>
            <p className="text-slate-300">
              <strong>Usage :</strong> Idéal quand on veut une performance équilibrée sur toutes les classes
            </p>
          </div>
        </div>

        {/* F1-Weighted Section */}
        <div className="p-4 bg-pink-900/20 border border-pink-700/30 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <Users className="w-5 h-5 text-pink-400" />
            <h3 className="text-lg font-semibold text-pink-400">F1-Weighted</h3>
            <Badge variant="outline" className="text-pink-400 border-pink-400">
              Pondération par taille
            </Badge>
          </div>
          <div className="space-y-2 text-sm">
            <p className="text-slate-300">
              <strong>Calcul :</strong> Moyenne pondérée des F1 par la fréquence de chaque classe
            </p>
            <p className="text-slate-300">
              <strong>Formule :</strong> Σ(F1_classe_i × support_classe_i) / total_échantillons
            </p>
            <p className="text-slate-300">
              <strong>Avantage :</strong> Reflète la performance globale en tenant compte de la distribution réelle
            </p>
            <p className="text-slate-300">
              <strong>Usage :</strong> Mieux adapté quand les classes majoritaires sont plus importantes
            </p>
          </div>
        </div>

        {/* Key Differences */}
        <div className="p-4 bg-blue-900/20 border border-blue-700/30 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            <h3 className="text-lg font-semibold text-blue-400">Différences Clés</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-semibold text-orange-400 mb-2">F1-Macro</h4>
              <ul className="space-y-1 text-slate-300">
                <li>• Équité parfaite entre classes</li>
                <li>• Sensible aux classes minoritaires</li>
                <li>• Peut être "pénalisé" par une classe rare mal prédite</li>
                <li>• Idéal pour détecter tous types d'exoplanètes</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-pink-400 mb-2">F1-Weighted</h4>
              <ul className="space-y-1 text-slate-300">
                <li>• Reflète la performance "réelle" du dataset</li>
                <li>• Proche de l'accuracy</li>
                <li>• Moins sensible aux classes rares</li>
                <li>• Orienté vers les classes majoritaires</li>
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
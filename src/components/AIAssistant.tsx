import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Bot, X, Send, Sparkles, HelpCircle, FileUp, Download, Brain, Settings } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const ASSISTANT_KNOWLEDGE = {
  general: {
    welcome: "Bonjour! Je suis votre assistant intelligent pour le système de classification d'exoplanètes NASA Kepler. Je peux vous aider à utiliser l'application, comprendre les fonctionnalités, et répondre à vos questions. Comment puis-je vous aider aujourd'hui?",
    capabilities: "Je peux vous aider avec:\n• 📊 Prédictions manuelles et par CSV\n• 🤖 Sélection et configuration des modèles ML\n• 📈 Interprétation des métriques\n• 🔧 Ajustement des hyperparamètres\n• 📥 Import/export de données CSV\n• ✅ Système de feedback"
  },
  predict: {
    manual: "Pour faire une prédiction manuelle:\n1. Sélectionnez un modèle ML (XGBoost recommandé, F1: 94%)\n2. Remplissez les 20 caractéristiques Kepler ou cliquez 'Load Example Data'\n3. Cliquez 'Classify Exoplanet'\n4. Vous serez automatiquement redirigé vers l'onglet Results\n\nLes résultats incluent: Confirmé (2), Candidat (1), ou Faux Positif (0)",
    csv: "Pour prédictions par lot CSV:\n1. Téléchargez le template via 'CSV Template' (génère des données aléatoires)\n2. Ajoutez vos données (20 colonnes requises)\n3. Cliquez 'Upload CSV for Batch Prediction'\n4. Le fichier avec prédictions se télécharge automatiquement (nom_original_predictions.csv)\n\n⚠️ Les prédictions CSV ne sont pas ajoutées à l'historique Results (uniquement pour export).",
    features: "Les 20 caractéristiques NASA Kepler incluent:\n• koi_score: Score de détection [0-1]\n• koi_prad: Rayon planétaire [R⊕]\n• koi_period: Période orbitale [jours]\n• koi_insol: Flux d'insolation [S⊕]\n• Et 16 autres métriques de transit et d'erreur\n\nToutes sont essentielles pour la classification précise."
  },
  models: {
    overview: "5 modèles ML disponibles:\n\n🥇 XGBoost (Recommandé)\n• F1-Macro: 94.5%\n• Meilleur équilibre\n\n🥈 Random Forest\n• F1-Macro: 93.2%\n• Robuste aux outliers\n\n🥉 SVM\n• F1-Macro: 91.8%\n• Bon pour données linéaires\n\n4️⃣ KNN\n• F1-Macro: 89.7%\n• Simple et interprétable\n\n5️⃣ Logistic Regression\n• F1-Macro: 87.3%\n• Baseline rapide",
    metrics: "Métriques de performance:\n\n• Accuracy: Précision globale\n• Precision: Proportion de vrais positifs\n• Recall: Capacité à détecter les positifs\n• F1-Macro: Moyenne harmonique non pondérée (équitable multiclasse)\n• F1-Weighted: Moyenne pondérée par classe\n• Confusion Matrix: Détails par classe\n\nF1-Macro est notre métrique principale pour éviter le biais de classe."
  },
  hyperparameters: {
    overview: "L'onglet Hyperparameters permet d'ajuster:\n\n• XGBoost: n_estimators, learning_rate, max_depth\n• Random Forest: n_estimators, max_depth, min_samples_split\n• SVM: C, kernel, gamma\n• KNN: n_neighbors, weights\n• Logistic Regression: C, penalty\n\nModifiez les paramètres et observez l'impact sur les métriques!",
    tuning: "Conseils d'ajustement:\n\n📈 Overfitting? Réduisez max_depth ou augmentez regularization (C)\n📉 Underfitting? Augmentez n_estimators ou réduisez regularization\n⚡ Trop lent? Réduisez n_estimators\n🎯 Meilleur équilibre? Expérimentez avec learning_rate"
  },
  feedback: {
    usage: "Le système de feedback améliore les modèles:\n\n1. Allez dans l'onglet Results\n2. Consultez vos prédictions récentes\n3. Cliquez 'Correct' ou 'Incorrect' selon votre expertise\n4. Les métriques du modèle se mettent à jour en temps réel\n\nVos feedbacks aident à affiner la précision continuellement!"
  },
  csv: {
    format: "Format CSV requis:\n\n✅ Header obligatoire avec les 20 noms de colonnes\n✅ Colonnes séparées par virgules\n✅ Valeurs numériques (pas de texte)\n✅ Une ligne = une observation\n\n❌ Colonnes manquantes\n❌ Formats de date ou texte\n❌ Fichiers Excel (.xlsx)\n\nUtilisez 'Download CSV Template' pour le format correct!",
    output: "Le CSV de résultats contient:\n\n• Toutes vos colonnes d'entrée (20 features)\n• prediction: 0/1/2 (numérique)\n• prediction_label: False Positive/Candidate/Confirmed\n• confidence: Niveau de confiance [0-1]\n• model: Nom du modèle utilisé\n• timestamp: Date/heure ISO de prédiction\n\nParfait pour analyses ultérieures dans Excel, Python, R!"
  }
};

const getAssistantResponse = (userMessage: string): string => {
  const msg = userMessage.toLowerCase();

  // Greetings
  if (msg.match(/^(bonjour|salut|hello|hi|hey|coucou)/)) {
    return ASSISTANT_KNOWLEDGE.general.welcome;
  }

  // Capabilities
  if (msg.match(/aide|help|capacité|fonction|peut faire|capable/)) {
    return ASSISTANT_KNOWLEDGE.general.capabilities;
  }

  // Manual prediction
  if (msg.match(/prédiction manuelle|comment prédire|faire.*prédiction|classify/)) {
    return ASSISTANT_KNOWLEDGE.predict.manual;
  }

  // CSV prediction
  if (msg.match(/csv|batch|lot|fichier|upload|télécharger.*données/)) {
    if (msg.match(/format|structure|colonnes|requises|template/)) {
      return ASSISTANT_KNOWLEDGE.csv.format;
    }
    if (msg.match(/résultat|output|sortie|download/)) {
      return ASSISTANT_KNOWLEDGE.csv.output;
    }
    return ASSISTANT_KNOWLEDGE.predict.csv;
  }

  // Features
  if (msg.match(/caractéristique|feature|koi_|20.*colonnes|données.*kepler/)) {
    return ASSISTANT_KNOWLEDGE.predict.features;
  }

  // Models
  if (msg.match(/modèle|model|xgboost|random.*forest|svm|knn|logistic|quel.*modèle|meilleur/)) {
    if (msg.match(/métrique|performance|accuracy|f1|precision|recall/)) {
      return ASSISTANT_KNOWLEDGE.models.metrics;
    }
    return ASSISTANT_KNOWLEDGE.models.overview;
  }

  // Metrics
  if (msg.match(/métrique|f1|accuracy|precision|recall|performance|confusion.*matrix/)) {
    return ASSISTANT_KNOWLEDGE.models.metrics;
  }

  // Hyperparameters
  if (msg.match(/hyperparamètre|hyperparameter|ajuster|tuning|paramètre|configuration/)) {
    if (msg.match(/conseil|tip|optimiser|améliorer|overfitting|underfitting/)) {
      return ASSISTANT_KNOWLEDGE.hyperparameters.tuning;
    }
    return ASSISTANT_KNOWLEDGE.hyperparameters.overview;
  }

  // Feedback
  if (msg.match(/feedback|retour|correct|incorrect|améliorer.*modèle/)) {
    return ASSISTANT_KNOWLEDGE.feedback.usage;
  }

  // Results tab
  if (msg.match(/résultat|result|voir.*prédiction|historique|clear.*history/)) {
    return "L'onglet Results affiche:\n\n• Vos 200 dernières prédictions manuelles\n• Horodatage et modèle utilisé\n• Niveau de confiance\n• Caractéristiques clés (période, rayon, etc.)\n• Boutons de feedback\n• Bouton 'Clear History' pour vider l'historique\n\n⚠️ Les prédictions CSV batch ne sont PAS ajoutées à l'historique (uniquement export direct).";
  }

  // Navigation
  if (msg.match(/navigation|onglet|tab|page|aller/)) {
    return "4 onglets disponibles:\n\n🎯 Predict: Saisie manuelle ou upload CSV\n📊 Model Metrics: Comparer les 5 modèles\n📋 Results: Historique (max 200) et feedback\n⚙️ Hyperparameters: Ajuster les paramètres\n\nNote: Navigation auto vers Results SEULEMENT pour prédictions manuelles. CSV batch télécharge automatiquement sans navigation.";
  }

  // Default - suggest topics
  return "Je peux vous aider avec:\n\n• 'Comment faire une prédiction manuelle?'\n• 'Comment utiliser les fichiers CSV?'\n• 'Quel modèle choisir?'\n• 'Comment ajuster les hyperparamètres?'\n• 'Comment donner du feedback?'\n• 'Expliquer les métriques F1'\n\nPosez-moi une question ou choisissez un sujet!";
};

export function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: ASSISTANT_KNOWLEDGE.general.welcome,
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Simulate typing delay
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));

    const response = getAssistantResponse(input);
    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, assistantMessage]);
    setIsTyping(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const quickActions = [
    { icon: FileUp, label: "Upload CSV", query: "Comment utiliser les fichiers CSV?" },
    { icon: Brain, label: "Modèles", query: "Quel modèle choisir?" },
    { icon: Settings, label: "Hyperparamètres", query: "Comment ajuster les hyperparamètres?" },
    { icon: HelpCircle, label: "Guide", query: "aide" }
  ];

  const handleQuickAction = (query: string) => {
    setInput(query);
    setTimeout(() => handleSend(), 100);
  };

  return (
    <>
      {/* Floating button */}
      <AnimatePresence>
        {!isOpen && (
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            className="fixed bottom-6 right-6 z-50"
          >
            <Button
              onClick={() => setIsOpen(true)}
              size="lg"
              className="h-14 w-14 rounded-full bg-gradient-to-br from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 shadow-lg hover:shadow-xl transition-all"
            >
              <Sparkles className="h-6 w-6" />
            </Button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-6 right-6 z-50 w-[400px] max-h-[600px]"
          >
            <Card className="bg-slate-800 border-slate-700 shadow-2xl">
              <CardHeader className="pb-3 bg-gradient-to-r from-purple-900/50 to-blue-900/50 border-b border-slate-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center">
                      <Bot className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-white flex items-center gap-2">
                        Assistant AI
                        <Badge variant="outline" className="text-xs text-green-400 border-green-600">
                          En ligne
                        </Badge>
                      </CardTitle>
                      <CardDescription className="text-slate-400 text-xs">
                        Votre guide intelligent
                      </CardDescription>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsOpen(false)}
                    className="h-8 w-8 p-0 text-slate-400 hover:text-white hover:bg-slate-700"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>

              <CardContent className="p-0">
                <ScrollArea className="h-[380px] p-4" ref={scrollRef}>
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[85%] rounded-lg px-4 py-2 ${
                            message.role === 'user'
                              ? 'bg-purple-600 text-white'
                              : 'bg-slate-700 text-slate-100'
                          }`}
                        >
                          <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                          <p className="text-xs mt-1 opacity-60">
                            {message.timestamp.toLocaleTimeString('fr-FR', { 
                              hour: '2-digit', 
                              minute: '2-digit' 
                            })}
                          </p>
                        </div>
                      </div>
                    ))}
                    {isTyping && (
                      <div className="flex justify-start">
                        <div className="bg-slate-700 rounded-lg px-4 py-2">
                          <div className="flex gap-1">
                            <motion.div
                              animate={{ opacity: [0.4, 1, 0.4] }}
                              transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                              className="w-2 h-2 bg-slate-400 rounded-full"
                            />
                            <motion.div
                              animate={{ opacity: [0.4, 1, 0.4] }}
                              transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                              className="w-2 h-2 bg-slate-400 rounded-full"
                            />
                            <motion.div
                              animate={{ opacity: [0.4, 1, 0.4] }}
                              transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                              className="w-2 h-2 bg-slate-400 rounded-full"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>

                {/* Quick actions */}
                {messages.length <= 2 && (
                  <div className="px-4 py-2 border-t border-slate-700 bg-slate-800/50">
                    <p className="text-xs text-slate-400 mb-2">Actions rapides:</p>
                    <div className="grid grid-cols-2 gap-2">
                      {quickActions.map((action) => (
                        <Button
                          key={action.label}
                          variant="outline"
                          size="sm"
                          onClick={() => handleQuickAction(action.query)}
                          className="border-slate-600 text-slate-300 hover:bg-slate-700 text-xs h-8"
                        >
                          <action.icon className="h-3 w-3 mr-1" />
                          {action.label}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Input */}
                <div className="p-4 border-t border-slate-700 bg-slate-800">
                  <div className="flex gap-2">
                    <Input
                      ref={inputRef}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Posez une question..."
                      className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
                      disabled={isTyping}
                    />
                    <Button
                      onClick={handleSend}
                      disabled={!input.trim() || isTyping}
                      className="bg-purple-600 hover:bg-purple-500"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
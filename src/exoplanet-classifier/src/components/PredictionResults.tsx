import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { CheckCircle, XCircle, HelpCircle, Clock, Target, Trash2 } from 'lucide-react';
import { PredictionResult, PREDICTION_LABELS } from '../types/exoplanet';
import { toast } from 'sonner';

interface PredictionResultsProps {
  predictions: PredictionResult[];
  onFeedback: (index: number, feedback: 'correct' | 'incorrect' | 'unknown') => void;
  onClearHistory?: () => void;
}

export function PredictionResults({ predictions, onFeedback, onClearHistory }: PredictionResultsProps) {
  const getPredictionColor = (prediction: 0 | 1 | 2) => {
    switch (prediction) {
      case 0: return 'bg-red-900 text-red-200 border-red-700';
      case 1: return 'bg-yellow-900 text-yellow-200 border-yellow-700';
      case 2: return 'bg-green-900 text-green-200 border-green-700';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getFeedbackIcon = (feedback?: string) => {
    switch (feedback) {
      case 'correct': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'incorrect': return <XCircle className="w-4 h-4 text-red-400" />;
      default: return <HelpCircle className="w-4 h-4 text-slate-400" />;
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(timestamp);
  };

  if (predictions.length === 0) {
    return (
      <Card className="bg-slate-800 border-slate-700">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Target className="w-16 h-16 text-slate-600 mb-4" />
          <h3 className="text-xl font-semibold text-slate-300 mb-2">
            No Predictions Yet
          </h3>
          <p className="text-slate-500 text-center max-w-md">
            Enter exoplanet features and select a model to start making predictions. 
            Your results will appear here.
          </p>
        </CardContent>
      </Card>
    );
  }

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all prediction history? This action cannot be undone.')) {
      onClearHistory?.();
      toast.success('Prediction history cleared');
    }
  };

  return (
    <div className="space-y-6">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Recent Predictions ({predictions.length})
              </CardTitle>
              <CardDescription className="text-slate-300">
                Review and provide feedback on model predictions to improve accuracy
              </CardDescription>
            </div>
            {predictions.length > 0 && onClearHistory && (
              <Button
                variant="outline"
                onClick={handleClearHistory}
                className="border-red-600 text-red-400 hover:bg-red-900"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear History
              </Button>
            )}
          </div>
        </CardHeader>
      </Card>

      <div className="space-y-4">
        {predictions.map((prediction, index) => (
          <Card key={index} className="bg-slate-800 border-slate-700">
            <CardContent className="p-6">
              <div className="grid gap-4 md:grid-cols-4">
                <div className="md:col-span-2 space-y-3">
                  <div className="flex items-center gap-3">
                    <Badge className={`px-3 py-1 ${getPredictionColor(prediction.prediction)}`}>
                      {PREDICTION_LABELS[prediction.prediction]}
                    </Badge>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-400">Confidence:</span>
                      <span className={`font-semibold ${getConfidenceColor(prediction.confidence)}`}>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-slate-400">Model:</span>
                      <Badge variant="outline" className="text-purple-400 border-purple-600">
                        {prediction.model}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-slate-400">Time:</span>
                      <span className="text-slate-300">{formatTimestamp(prediction.timestamp)}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <span className="text-sm text-slate-400">Key Features:</span>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-slate-500">Period:</span>
                        <span className="text-slate-300 ml-1">{prediction.features.koi_period.toFixed(2)} days</span>
                      </div>
                      <div>
                        <span className="text-slate-500">Radius:</span>
                        <span className="text-slate-300 ml-1">{prediction.features.koi_prad.toFixed(2)} R⊕</span>
                      </div>
                      <div>
                        <span className="text-slate-500">Insolation:</span>
                        <span className="text-slate-300 ml-1">{prediction.features.koi_insol.toFixed(2)} S⊕</span>
                      </div>
                      <div>
                        <span className="text-slate-500">SNR:</span>
                        <span className="text-slate-300 ml-1">{prediction.features.koi_model_snr.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="md:col-span-2 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-400">Feedback:</span>
                    {getFeedbackIcon(prediction.userFeedback)}
                    {prediction.userFeedback && (
                      <span className={`text-sm font-medium ${
                        prediction.userFeedback === 'correct' 
                          ? 'text-green-400' 
                          : 'text-red-400'
                      }`}>
                        {prediction.userFeedback === 'correct' ? 'Correct' : 'Incorrect'}
                      </span>
                    )}
                  </div>

                  {!prediction.userFeedback && (
                    <div className="space-y-2">
                      <p className="text-sm text-slate-400">
                        Was this prediction correct?
                      </p>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          onClick={() => onFeedback(index, 'correct')}
                          className="bg-green-700 hover:bg-green-600 text-green-100"
                        >
                          <CheckCircle className="w-4 h-4 mr-1" />
                          Correct
                        </Button>
                        <Button
                          size="sm"
                          onClick={() => onFeedback(index, 'incorrect')}
                          className="bg-red-700 hover:bg-red-600 text-red-100"
                        >
                          <XCircle className="w-4 h-4 mr-1" />
                          Incorrect
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => onFeedback(index, 'unknown')}
                          className="border-slate-600 text-slate-400 hover:bg-slate-700"
                        >
                          <HelpCircle className="w-4 h-4 mr-1" />
                          Don't Know
                        </Button>
                      </div>
                    </div>
                  )}

                  {prediction.userFeedback && (
                    <div className="text-sm text-slate-500">
                      Thank you for your feedback! This helps improve model accuracy.
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
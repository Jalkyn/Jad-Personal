import React from 'react';
import { ExoplanetClassifier } from './components/ExoplanetClassifier';
import { Toaster } from './components/ui/sonner';

export default function App() {
  return (
    <div className="min-h-screen">
      <ExoplanetClassifier />
      <Toaster />
    </div>
  );
}
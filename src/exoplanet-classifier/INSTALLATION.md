# Installation Guide

## 📋 Prerequisites

Before installing, ensure you have:

- **Node.js 18 or higher** ([Download here](https://nodejs.org/))
- **npm** or **yarn** package manager
- **Git** (optional, for cloning)

## 🚀 Quick Installation

### Method 1: Using npm

1. **Extract the project** (if downloaded as ZIP) or clone:
   ```bash
   git clone <repository-url>
   cd exoplanet-classifier
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5173
   ```

### Method 2: Using yarn

1. **Extract/clone the project**:
   ```bash
   cd exoplanet-classifier
   ```

2. **Install dependencies**:
   ```bash
   yarn install
   ```

3. **Start the development server**:
   ```bash
   yarn dev
   ```

## 🔧 Build for Production

To create a production build:

```bash
npm run build
# or
yarn build
```

The build files will be generated in the `dist/` directory.

## 🏗️ Project Structure

```
exoplanet-classifier/
├── public/                  # Static assets
├── src/                     # Source code
│   ├── components/          # React components
│   │   ├── ui/             # ShadCN UI components
│   │   ├── ExoplanetClassifier.tsx
│   │   ├── FeatureInputForm.tsx
│   │   ├── ModelMetricsDisplay.tsx
│   │   ├── PredictionResults.tsx
│   │   ├── HyperparameterTuning.tsx
│   │   └── F1MetricsExplanation.tsx
│   ├── types/              # TypeScript definitions
│   ├── utils/              # Utility functions
│   ├── styles/             # CSS styles
│   └── App.tsx             # Main component
├── package.json            # Dependencies
├── vite.config.ts          # Vite configuration
├── tailwind.config.js      # Tailwind configuration
└── README.md              # Documentation
```

## 🎯 Features

- **5 ML Models**: XGBoost, Random Forest, SVM, Logistic Regression, KNN
- **Real-time Classification**: Instant exoplanet predictions
- **F1-Macro vs F1-Weighted**: Comprehensive metrics comparison
- **Interactive Charts**: Performance visualization with Recharts
- **Hyperparameter Tuning**: Model optimization interface
- **Feedback System**: Continuous learning capability

## 📊 Model Performance Overview

| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| XGBoost | 92.42% | 90.02% | 92.46% |
| Random Forest | 91.79% | 89.17% | 91.79% |
| SVM | 88.78% | 85.10% | 88.78% |
| Logistic Regression | 87.98% | 84.47% | 87.98% |
| KNN | 81.92% | 76.83% | 81.92% |

## 🛠️ Tech Stack

- **React 18** + **TypeScript** - Modern frontend framework
- **Vite** - Fast build tool and dev server
- **Tailwind CSS v4** - Utility-first styling
- **ShadCN/UI** - Beautiful component library
- **Recharts** - Interactive data visualization
- **Lucide React** - Modern icon library
- **Motion** - Smooth animations

## 🔍 Usage Tips

1. **Model Selection**: XGBoost is pre-selected as the best performer
2. **Feature Input**: Use tooltips to understand each NASA Kepler parameter
3. **Example Data**: Click "Load Example Data" for quick testing
4. **Metrics Analysis**: Check the F1 guide to understand scoring differences
5. **Hyperparameters**: Experiment with model tuning for optimization

## 🐛 Troubleshooting

### Common Issues:

**Port 5173 already in use:**
```bash
npm run dev -- --port 3000
```

**Dependencies not installing:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Build errors:**
```bash
npm run lint
npm run build
```

## 📞 Support

If you encounter any issues:

1. Check the [README.md](README.md) for detailed information
2. Verify all dependencies are installed correctly
3. Ensure you're using Node.js 18+
4. Open an issue with error details

## 📈 Performance Notes

- **Development**: Hot reloading enabled for fast development
- **Production**: Optimized build with tree-shaking and minification
- **Bundle Size**: ~2MB total (including charts and UI components)
- **Browser Support**: Modern browsers (Chrome 90+, Firefox 88+, Safari 14+)

---

**Happy exoplanet hunting! 🚀**
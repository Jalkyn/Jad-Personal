# 🚀 Exoplanet Classification Web Application

A comprehensive web application for classifying exoplanets using NASA Kepler dataset with multiple machine learning models. Built for researchers and enthusiasts to interact with advanced ML algorithms for exoplanet discovery and classification.

![Exoplanet Classifier](https://images.unsplash.com/photo-1446776653964-20c1d3a81b06?w=800&h=400&fit=crop&crop=center)

## ✨ Features

### 🤖 Multiple ML Models
- **XGBoost** (Best Performance: 92.42% accuracy, 90.02% F1-Macro)
- **Random Forest** (91.79% accuracy, 89.17% F1-Macro)
- **Support Vector Machine** (88.78% accuracy, 85.10% F1-Macro)
- **Logistic Regression** (87.98% accuracy, 84.47% F1-Macro)
- **K-Nearest Neighbors** (81.92% accuracy, 76.83% F1-Macro)

### 📊 Advanced Metrics & Visualization
- **F1-Macro vs F1-Weighted** comparison with detailed explanations
- Interactive confusion matrices for each model
- Real-time performance charts and statistics
- Comprehensive model comparison dashboard

### 🔧 Interactive Features
- **20 Kepler Features Input** with tooltips and descriptions
- **Hyperparameter Tuning** interface for model optimization
- **Real-time Classification** with confidence scores
- **Feedback System** to improve model accuracy
- **Example Data Loading** for quick testing

### 🎯 Classification Categories
- **Confirmed (2)**: Verified exoplanets
- **Candidate (1)**: Potential exoplanets requiring validation
- **False Positive (0)**: Non-planetary signals

## 🛠️ Technology Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS v4 + ShadCN UI Components
- **Charts**: Recharts for interactive visualizations
- **Icons**: Lucide React
- **Animations**: Motion (formerly Framer Motion)
- **Forms**: React Hook Form
- **Build Tool**: Vite

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd exoplanet-classifier
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:5173`

### Build for Production

```bash
npm run build
# or
yarn build
```

## 📖 Usage Guide

### 1. Model Selection
- Choose from 5 pre-trained machine learning models
- View real-time performance metrics for each model
- XGBoost is set as default (best performance)

### 2. Feature Input
- Enter 20 NASA Kepler dataset features
- Use tooltips to understand each parameter
- Load example data for quick testing

### 3. Classification
- Click "Classify Exoplanet" to get predictions
- View confidence scores and probability distributions
- Get detailed classification results

### 4. Model Analysis
- Compare all models side-by-side
- Understand F1-Macro vs F1-Weighted metrics
- Analyze confusion matrices and performance charts

### 5. Hyperparameter Tuning
- Adjust model parameters in real-time
- See how changes affect model performance
- Optimize for your specific use case

## 📊 Model Performance

| Model | Accuracy | F1-Macro | F1-Weighted | Total Predictions |
|-------|----------|----------|-------------|-------------------|
| XGBoost | 92.42% | 90.02% | 92.46% | 3,085 |
| Random Forest | 91.79% | 89.17% | 91.79% | 3,085 |
| SVM | 88.78% | 85.10% | 88.78% | 3,085 |
| Logistic Regression | 87.98% | 84.47% | 87.98% | 3,085 |
| KNN | 81.92% | 76.83% | 81.92% | 3,085 |

## 🔬 Understanding the Metrics

### F1-Macro Score
- **Fair evaluation** across all classes (Confirmed, Candidate, False Positive)
- **Equal weight** to each class regardless of size
- **Best for**: Balanced performance across rare and common exoplanet types

### F1-Weighted Score  
- **Weighted by class frequency** in the dataset
- **Reflects real-world** dataset distribution
- **Best for**: Overall system performance in production

### Key Insight
F1-Macro ≠ Recall! F1-Macro combines both precision and recall for each class, while Macro Recall only measures the ability to find true positives.

## 🗂️ Project Structure

```
exoplanet-classifier/
├── src/
│   ├── components/           # React components
│   │   ├── ui/              # ShadCN UI components
│   │   ├── ExoplanetClassifier.tsx
│   │   ├── FeatureInputForm.tsx
│   │   ├── ModelMetricsDisplay.tsx
│   │   ├── PredictionResults.tsx
│   │   ├── HyperparameterTuning.tsx
│   │   └── F1MetricsExplanation.tsx
│   ├── types/               # TypeScript definitions
│   ├── utils/               # Utility functions & mock data
│   ├── styles/              # Global CSS styles
│   └── App.tsx              # Main application component
├── public/                  # Static assets
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Kepler Mission** for providing the exoplanet dataset
- **ShadCN** for the beautiful UI component library
- **Recharts** for interactive data visualizations
- **The exoplanet research community** for advancing our understanding of planetary systems

## 📬 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out to the development team.

---

**Built with ❤️ for the exoplanet research community**
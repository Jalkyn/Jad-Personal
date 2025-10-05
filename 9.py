import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from ast import literal_eval
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuration de base
st.set_page_config(page_title="NASA Exoplanet Prediction", layout="wide")

# CSS amélioré
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .prediction-result {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .feedback-section {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .feature-input {
        margin-bottom: 1rem;
    }
    .feature-group {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.1rem;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .metric-item {
        background: rgba(255,255,255,0.05);
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        border-left: 3px solid #667eea;
    }
    .metrics-container {
        background: rgba(255,255,255,0.02);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.25rem;
    }
    .model-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .hyperparam-info {
        background: rgba(255,255,255,0.02);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .batch-prediction-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .retrain-info {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Noms réels des features avec organisation par catégorie
FEATURE_GROUPS = {
    "Detection & Scoring": [
        ("koi_score", "Detection Score [0-1]", 0.0, 1.0, 0.5),
        ("koi_model_snr", "Transit SNR", 0.0, 50.0, 15.0),
        ("habitability_index", "Habitability Index", 0.0, 1.0, 0.3)
    ],
    "Planet Characteristics": [
        ("planet_density_proxy", "Planet Density (proxy) [g/cm³]", 0.0, 10.0, 1.3),
        ("koi_prad", "Planet Radius [Earth radii]", 0.5, 20.0, 2.0),
        ("koi_prad_err1", "Planet Radius Error (+) [Earth radii]", 0.0, 5.0, 0.5)
    ],
    "False Positive Flags": [
        ("koi_fpflag_nt", "FP Flag (Non-Transit) [0/1]", 0, 1, 0),
        ("koi_fpflag_ss", "FP Flag (Stellar Variability) [0/1]", 0, 1, 0),
        ("koi_fpflag_co", "FP Flag (Contamination) [0/1]", 0, 1, 0)
    ],
    "Transit Parameters": [
        ("koi_duration_err1", "Transit Error (+) [hours]", 0.0, 5.0, 0.1),
        ("duration_period_ratio", "Duration/Period Ratio", 0.0, 0.1, 0.01),
        ("koi_time0bk_err1", "Transit Epoch Error (+) [days]", 0.0, 1.0, 0.01),
        ("koi_period", "Orbital Period [days]", 0.5, 500.0, 10.5),
        ("koi_depth", "Transit Depth [ppm]", 0.0, 10000.0, 500.0)
    ],
    "Orbital Parameters": [
        ("koi_impact", "Impact Parameter", 0.0, 1.0, 0.5),
        ("koi_period_err1", "Orbital Period Error (+) [days]", 0.0, 1.0, 0.01)
    ],
    "Stellar Parameters": [
        ("koi_steff_err1", "Stellar Temp Error (-) [K]", 0.0, 500.0, 50.0),
        ("koi_steff_err2", "Stellar Temp Error (+) [K]", 0.0, 500.0, 50.0),
        ("koi_slogg_err2", "log(g) Error (-) [log(cm/s^2)]", 0.0, 1.0, 0.1),
        ("koi_insol", "Insolation Flux [Earth flux]", 0.0, 10000.0, 1.0)
    ]
}

# Mapping des classes
CLASS_MAPPING = {
    0: "Faux Positif",
    1: "Candidat", 
    2: "Exoplanète"
}

# Mapping pour l'encodage de koi_disposition
DISPOSITION_MAPPING = {
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1, 
    'CONFIRMED': 2
}

# ================================
# FONCTIONS DE PRÉPROCESSING AVANCÉ
# ================================

def apply_advanced_preprocessing(df):
    """
    Applique le préprocessing avancé du code original sur le dataset
    """
    st.info("🔄 Application du préprocessing avancé...")
    
    # Sauvegarde des données originales
    df_processed = df.copy()
    
    # ================================
    # 1. NETTOYAGE AVANCÉ DES DONNÉES
    # ================================
    st.write("**Étape 1:** Nettoyage avancé des données")
    
    # Suppression des colonnes avec trop de valeurs manquantes
    missing_threshold = 0.7
    cols_to_drop = []
    
    for col in df_processed.columns:
        missing_pct = df_processed[col].isnull().sum() / len(df_processed)
        if missing_pct > missing_threshold:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
        st.write(f"🗑️ {len(cols_to_drop)} colonnes avec >70% de valeurs manquantes supprimées")
    
    # Suppression des doublons
    duplicates = df_processed.duplicated().sum()
    if duplicates > 0:
        df_processed = df_processed.drop_duplicates()
        st.write(f"🗑️ {duplicates} doublons supprimés")
    
    # Suppression des colonnes avec une seule valeur unique
    single_value_cols = []
    for col in df_processed.columns:
        if df_processed[col].nunique() <= 1:
            single_value_cols.append(col)
    
    if single_value_cols:
        df_processed = df_processed.drop(columns=single_value_cols)
        st.write(f"🗑️ Colonnes à valeur unique supprimées: {len(single_value_cols)}")
    
    # ================================
    # 2. IMPUTATION INTELLIGENTE
    # ================================
    st.write("**Étape 2:** Imputation des valeurs manquantes")
    
    # Identification des types de colonnes
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Imputation des variables numériques avec KNN
    if numeric_cols:
        missing_before = df_processed[numeric_cols].isnull().sum().sum()
        if missing_before > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            df_processed[numeric_cols] = knn_imputer.fit_transform(df_processed[numeric_cols])
            st.write(f"✅ {missing_before} valeurs numériques imputées avec KNN")
    
    # Imputation des variables catégorielles
    if categorical_cols:
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col] = df_processed[col].fillna(mode_value)
    
    # ================================
    # 3. ENCODAGE DE KOI_DISPOSITION
    # ================================
    st.write("**Étape 3:** Encodage de la variable cible koi_disposition")
    
    if 'koi_disposition' in df_processed.columns:
        # Encodage de koi_disposition
        le = LabelEncoder()
        df_processed['koi_disposition_encoded'] = le.fit_transform(df_processed['koi_disposition'])
        
        # Sauvegarde du mapping pour interprétation
        target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        st.write(f"✅ koi_disposition encodée. Mapping: {target_mapping}")
        
        # Suppression de la colonne originale koi_disposition
        df_processed = df_processed.drop(columns=['koi_disposition'])
        st.write("🗑️ Colonne koi_disposition originale supprimée")
    else:
        st.warning("⚠️ Colonne koi_disposition non trouvée - vérifiez que la variable cible est présente")
    
    # ================================
    # 4. FEATURE ENGINEERING
    # ================================
    st.write("**Étape 4:** Feature engineering")
    
    # Vérification des colonnes nécessaires pour le feature engineering
    required_features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_duration']
    available_features = [f for f in required_features if f in df_processed.columns]
    
    if len(available_features) == len(required_features):
        # Création de nouvelles features basées sur le domaine astronomique
        # Densité relative de la planète
        df_processed['planet_density_proxy'] = df_processed['koi_prad'] / (df_processed['koi_period'] ** (2/3))
        
        # Indice d'habitabilité simplifié
        df_processed['habitability_index'] = (df_processed['koi_teq'] / 288) * (df_processed['koi_prad'] / 1.0)
        
        # Ratio durée/période
        df_processed['duration_period_ratio'] = df_processed['koi_duration'] / df_processed['koi_period']
        
        # Binning de variables continues
        df_processed['planet_size_category'] = pd.cut(df_processed['koi_prad'],
                                           bins=[0, 1.25, 2.0, 4.0, float('inf')],
                                           labels=['Earth-like', 'Super-Earth', 'Neptune-like', 'Jupiter-like'])
        
        st.write("✅ Features d'ingénierie créées: planet_density_proxy, habitability_index, duration_period_ratio, planet_size_category")
    else:
        st.warning(f"⚠️ Features manquantes pour l'ingénierie: {set(required_features) - set(available_features)}")
    
    # ================================
    # 5. ENCODAGE DES VARIABLES CATÉGORIELLES
    # ================================
    st.write("**Étape 5:** Encodage des variables catégorielles")
    
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        unique_values = df_processed[col].nunique()
        if unique_values <= 10:  # One-hot encoding
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed = df_processed.drop(columns=[col])
        else:  # Label encoding
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    st.success(f"✅ Préprocessing terminé. Dimensions finales: {df_processed.shape}")
    
    return df_processed

def validate_and_preprocess_uploaded_data(uploaded_df, base_dataset_features):
    """
    Valide et prétraite le dataset uploadé en suivant exactement le même processus
    que le préprocessing original
    """
    st.info("🔍 Validation et prétraitement du dataset uploadé...")
    
    # Appliquer le préprocessing avancé
    processed_df = apply_advanced_preprocessing(uploaded_df)
    
    # Vérifier que toutes les features du base_dataset sont présentes
    missing_features = [f for f in base_dataset_features if f not in processed_df.columns]
    
    if missing_features:
        st.error(f"❌ Features manquantes après préprocessing: {missing_features}")
        return None
    
    # Sélectionner uniquement les features du base_dataset
    final_df = processed_df[base_dataset_features].copy()
    
    st.success(f"✅ Dataset prétraité: {final_df.shape[0]} échantillons, {final_df.shape[1]} features")
    
    return final_df

# FONCTION DE CORRESPONDANCE DES NOMS DE MODÈLES
def find_matching_model_name(selected_model, comparison_df):
    """Trouve le nom correspondant dans le DataFrame des métriques"""
    if comparison_df.empty:
        return None
        
    cleaned_selected = selected_model.lower().replace('_', '').replace('-', '').replace(' ', '')
    
    for model_name_in_df in comparison_df['model_name']:
        cleaned_df_name = model_name_in_df.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        if cleaned_selected == cleaned_df_name:
            return model_name_in_df
            
        if cleaned_selected in cleaned_df_name or cleaned_df_name in cleaned_selected:
            return model_name_in_df
            
        if re.sub(r'[^a-z0-9]', '', cleaned_selected) == re.sub(r'[^a-z0-9]', '', cleaned_df_name):
            return model_name_in_df
    
    return None

def get_model_family(model_name):
    """Détermine la famille du modèle à partir de son nom"""
    model_name_lower = model_name.lower()
    if 'randomforest' in model_name_lower or 'rf' in model_name_lower:
        return 'RandomForest'
    elif 'xgboost' in model_name_lower or 'xgb' in model_name_lower:
        return 'XGBoost'
    elif 'svm' in model_name_lower:
        return 'SVM'
    elif 'knn' in model_name_lower:
        return 'KNN'
    elif 'logistic' in model_name_lower or 'lr' in model_name_lower:
        return 'LogisticRegression'
    else:
        return 'Autre'

# CHARGEMENT DES DONNÉES
@st.cache_data
def load_model_metrics():
    """Charge les métriques des modèles depuis le fichier JSON"""
    try:
        with open('all_models_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return []

@st.cache_data
def load_model_comparison():
    """Charge la comparaison des modèles depuis le CSV"""
    try:
        df = pd.read_csv('models_comparison.csv')
        df['confusion_matrix'] = df['confusion_matrix'].apply(literal_eval)
        return df
    except Exception as e:
        st.error(f"❌ Impossible de charger models_comparison.csv: {e}")
        return pd.DataFrame()

@st.cache_data
def load_dataset():
    """Charge le dataset pour l'entraînement"""
    try:
        df = pd.read_csv('kepler_preprocessed.csv')
        return df
    except Exception as e:
        st.error(f"❌ Impossible de charger kepler_preprocessed.csv: {e}")
        return None

@st.cache_data
def load_base_dataset_for_retraining():
    """Charge le dataset de base sans scaling pour le réentraînement"""
    try:
        df = pd.read_csv('kepler_before_scaling_selected.csv')
        return df
    except Exception as e:
        st.error(f"❌ Impossible de charger kepler_before_scaling_selected.csv: {e}")
        return None

# CHARGEMENT DES MODELES
def load_models_with_dict():
    """Charge les modèles depuis des dictionnaires .pkl"""
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    models = {}
    model_info = {}
    
    if not pkl_files:
        st.error("❌ Aucun fichier .pkl trouvé")
        return models, model_info
    
    for pkl_file in pkl_files:
        try:
            model_name = pkl_file.replace('.pkl', '')
            
            # Ne pas charger les modèles personnalisés ou réentraînés ici, ils seront gérés séparément
            if is_custom_model(model_name) or is_retrained_model(model_name):
                continue
                
            with open(pkl_file, 'rb') as f:
                loaded_data = pickle.load(f)
            
            if isinstance(loaded_data, dict):
                model_object = None
                model_metadata = {}
                
                priority_keys = ['model', 'classifier', 'estimator', 'clf', 'best_estimator_']
                
                for key in priority_keys:
                    if key in loaded_data and hasattr(loaded_data[key], 'predict'):
                        model_object = loaded_data[key]
                        break
                
                if model_object is None:
                    for key, value in loaded_data.items():
                        if hasattr(value, 'predict'):
                            model_object = value
                            break
                
                for key, value in loaded_data.items():
                    model_metadata[key] = {
                        'type': type(value).__name__,
                        'has_predict': hasattr(value, 'predict'),
                        'has_predict_proba': hasattr(value, 'predict_proba') if hasattr(value, 'predict') else False
                    }
                
                models[model_name] = model_object
                model_info[model_name] = {
                    'metadata': model_metadata,
                    'has_model': model_object is not None,
                    'model_type': type(model_object).__name__ if model_object else 'Aucun',
                    'has_predict_proba': hasattr(model_object, 'predict_proba') if model_object else False,
                    'model_family': get_model_family(model_name)
                }
                
            else:
                models[model_name] = loaded_data
                model_info[model_name] = {
                    'metadata': {'direct': type(loaded_data).__name__},
                    'has_model': hasattr(loaded_data, 'predict'),
                    'model_type': type(loaded_data).__name__,
                    'has_predict_proba': hasattr(loaded_data, 'predict_proba'),
                    'model_family': get_model_family(model_name)
                }
            
            if model_info[model_name]['has_model']:
                test_features = np.random.randn(1, 20)
                try:
                    prediction = models[model_name].predict(test_features)
                    if model_info[model_name]['has_predict_proba']:
                        probabilities = models[model_name].predict_proba(test_features)
                    model_info[model_name]['test_success'] = True
                except Exception as e:
                    model_info[model_name]['test_success'] = False
                    model_info[model_name]['test_error'] = str(e)
            
        except Exception as e:
            st.error(f"❌ Erreur avec {pkl_file}: {str(e)}")
    
    return models, model_info

# FONCTIONS POUR IDENTIFIER LES TYPES DE MODÈLES
def is_custom_model(model_name):
    """Vérifie si un modèle est un modèle personnalisé"""
    # Logique plus flexible pour identifier les modèles custom
    custom_indicators = ['custom', 'personnalisé', 'tuning', 'hyperparam']
    model_lower = model_name.lower()
    
    # Vérifier les indicateurs dans le nom
    for indicator in custom_indicators:
        if indicator in model_lower:
            return True
    
    # Vérifier s'il est dans les métriques custom
    custom_metrics = load_custom_model_metrics()
    for metric in custom_metrics:
        if metric['model_name'] == model_name:
            return True
    
    return False

def is_retrained_model(model_name):
    """Vérifie si un modèle est un modèle réentraîné"""
    # Logique plus flexible pour identifier les modèles réentraînés
    retrained_indicators = ['retrained', 'reentraine', 'reentrainé', 'retrain', 'fine_tuned']
    model_lower = model_name.lower()
    
    # Vérifier les indicateurs dans le nom
    for indicator in retrained_indicators:
        if indicator in model_lower:
            return True
    
    # Vérifier s'il est dans les métriques retrained
    retrained_metrics = load_retrained_model_metrics()
    for metric in retrained_metrics:
        if metric['model_name'] == model_name:
            return True
    
    return False

# FONCTIONS POUR L'HYPERPARAMETER TUNING AMÉLIORÉES
def create_custom_model(model_type, hyperparams):
    """Crée un modèle personnalisé avec les hyperparamètres spécifiés"""
    try:
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                max_features=hyperparams.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "XGBoost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 3),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                subsample=hyperparams.get('subsample', 1.0),
                colsample_bytree=hyperparams.get('colsample_bytree', 1.0),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "SVM":
            from sklearn.svm import SVC
            model = SVC(
                C=hyperparams.get('C', 1.0),
                kernel=hyperparams.get('kernel', 'rbf'),
                gamma=hyperparams.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        else:
            return None
        
        return model
    except Exception as e:
        st.error(f"Erreur lors de la création du modèle: {e}")
        return None

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Entraîne et évalue un modèle, retourne les métriques"""
    try:
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Métriques détaillées
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_macro = report['macro avg']['f1-score']
        f1_weighted = report['weighted avg']['f1-score']
        
        # ROC AUC (si probabilités disponibles)
        roc_auc = None
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc if roc_auc else 0.5,
            'auc_score': roc_auc if roc_auc else 0.5,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement et évaluation: {e}")
        return None

def load_training_data():
    """Charge les données d'entraînement depuis kepler_preprocessed.csv"""
    try:
        # Charger le vrai dataset
        df = pd.read_csv('kepler_preprocessed.csv')
        st.success(f"✅ Dataset chargé: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
        
        # Vérifier la structure du dataset
        if df.shape[1] < 21:  # 20 features + target
            st.error(f"❌ Le dataset doit contenir au moins 21 colonnes (20 features + target). Actuel: {df.shape[1]}")
            return None, None, None, None
        
        # Séparer features et target
        X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
        y = df.iloc[:, -1].values   # Dernière colonne comme target
        
        # Vérifier les dimensions
        if X.shape[1] != 20:
            st.warning(f"⚠️ Nombre de features différent de 20. Utilisation des {X.shape[1]} premières colonnes.")
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y  # Pour maintenir la distribution des classes
        )
        
        st.success(f"✅ Split effectué: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        st.info(f"📊 Distribution des classes: {np.unique(y, return_counts=True)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        return None, None, None, None

def save_custom_model_metrics(model_name, metrics, hyperparams):
    """Sauvegarde les métriques du modèle custom dans un fichier JSON"""
    try:
        # Charger les métriques existantes
        custom_metrics_file = 'custom_models_metrics.json'
        if os.path.exists(custom_metrics_file):
            with open(custom_metrics_file, 'r') as f:
                all_custom_metrics = json.load(f)
        else:
            all_custom_metrics = []
        
        # Ajouter les nouvelles métriques
        model_metrics = {
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'roc_auc': metrics['roc_auc'],
            'auc_score': metrics['auc_score'],
            'confusion_matrix': metrics['confusion_matrix'],
            'hyperparameters': hyperparams,
            'creation_time': pd.Timestamp.now().isoformat(),
            'model_type': 'Custom'
        }
        
        all_custom_metrics.append(model_metrics)
        
        # Sauvegarder
        with open(custom_metrics_file, 'w') as f:
            json.dump(all_custom_metrics, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde des métriques: {e}")
        return False

def load_custom_model_metrics():
    """Charge les métriques des modèles custom"""
    try:
        custom_metrics_file = 'custom_models_metrics.json'
        if os.path.exists(custom_metrics_file):
            with open(custom_metrics_file, 'r') as f:
                return json.load(f)
        return []
    except:
        return []

def delete_custom_model(model_name):
    """Supprime un modèle personnalisé et toutes ses données"""
    try:
        # Supprimer le fichier .pkl
        pkl_file = f"{model_name}.pkl"
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
            st.write(f"🗑️ Fichier modèle supprimé: {pkl_file}")
        
        # Supprimer les métriques du fichier JSON
        custom_metrics_file = 'custom_models_metrics.json'
        if os.path.exists(custom_metrics_file):
            with open(custom_metrics_file, 'r') as f:
                all_custom_metrics = json.load(f)
            
            # Filtrer pour garder seulement les autres modèles
            updated_metrics = [m for m in all_custom_metrics if m['model_name'] != model_name]
            
            with open(custom_metrics_file, 'w') as f:
                json.dump(updated_metrics, f, indent=2)
            
            st.write(f"🗑️ Métriques supprimées pour: {model_name}")
        
        # Supprimer du session_state
        if model_name in st.session_state.custom_models:
            del st.session_state.custom_models[model_name]
        
        return True
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la suppression du modèle {model_name}: {e}")
        return False

def delete_all_custom_models():
    """Supprime tous les modèles personnalisés"""
    try:
        # Identifier tous les modèles custom
        custom_models_to_delete = []
        for model_name in list(st.session_state.custom_models.keys()):
            if is_custom_model(model_name):
                custom_models_to_delete.append(model_name)
        
        # Supprimer chaque modèle custom
        for model_name in custom_models_to_delete:
            delete_custom_model(model_name)
        
        st.success(f"✅ Tous les modèles personnalisés ont été supprimés ({len(custom_models_to_delete)} modèles)")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la suppression: {e}")

# FONCTIONS POUR LE REENTRAINEMENT AVEC PRÉPROCESSING AVANCÉ
def validate_dataset_for_retraining(uploaded_df, base_dataset):
    """Valide le dataset uploadé pour le réentraînement"""
    try:
        # Vérifier que les features nécessaires pour le feature engineering sont présentes
        required_engineering_features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_duration']
        missing_engineering_features = [f for f in required_engineering_features if f not in uploaded_df.columns]
        
        if missing_engineering_features:
            st.warning(f"⚠️ Features manquantes pour l'ingénierie: {missing_engineering_features}")
            st.info("Le feature engineering sera limité sans ces features")
        
        # Vérifier si koi_disposition est présente
        if 'koi_disposition' not in uploaded_df.columns:
            st.warning("⚠️ Colonne koi_disposition non trouvée - vérifiez que la variable cible est présente")
        
        # Vérifier les dimensions
        if uploaded_df.shape[1] < 10:  # Minimum raisonnable
            st.error(f"Le dataset doit contenir au moins 10 colonnes")
            return False
        
        if uploaded_df.isnull().sum().sum() > 0:
            st.warning("Le dataset contient des valeurs manquantes - elles seront traitées lors du prétraitement")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors de la validation du dataset: {e}")
        return False

def preprocess_uploaded_dataset(uploaded_df, base_dataset):
    """Prétraite le dataset uploadé en utilisant les mêmes étapes que le preprocessing original"""
    try:
        st.info("🔄 Début du prétraitement avancé du dataset uploadé...")
        
        # Appliquer le préprocessing avancé
        processed_df = apply_advanced_preprocessing(uploaded_df)
        
        # Vérifier que toutes les features du base_dataset sont présentes
        base_features = base_dataset.columns.tolist()
        missing_features = [feature for feature in base_features if feature not in processed_df.columns]
        
        if missing_features:
            st.error(f"❌ Features manquantes après préprocessing: {missing_features}")
            return None, None
        
        # Sélectionner uniquement les colonnes présentes dans le base_dataset
        df_final = processed_df[base_features].copy()
        
        st.success(f"✅ Sélection des features: {len(base_features)} colonnes conservées")
        
        # Statistiques de prétraitement
        preprocessing_stats = {
            'original_samples': len(uploaded_df),
            'final_samples': len(df_final),
            'original_features': len(uploaded_df.columns),
            'final_features': len(df_final.columns)
        }
        
        return df_final, preprocessing_stats
        
    except Exception as e:
        st.error(f"❌ Erreur lors du prétraitement: {e}")
        return None, None

def merge_and_clean_datasets(base_df, new_df, remove_duplicates=True, remove_na=True):
    """Fusionne et nettoie les datasets"""
    try:
        # Vérifier que les datasets ont la même structure
        if base_df.shape[1] != new_df.shape[1]:
            st.error(f"Les datasets ont des structures différentes: {base_df.shape[1]} vs {new_df.shape[1]} colonnes")
            return None, None
        
        # Fusionner les datasets
        merged_data = pd.concat([base_df, new_df], ignore_index=True)
        original_size = len(merged_data)
        
        stats = {
            'original_size': original_size,
            'duplicates_removed': 0,
            'na_removed': 0,
            'final_size': 0
        }
        
        # Supprimer les doublons
        if remove_duplicates:
            before_dedup = len(merged_data)
            merged_data = merged_data.drop_duplicates()
            stats['duplicates_removed'] = before_dedup - len(merged_data)
        
        # Supprimer les valeurs manquantes
        if remove_na:
            before_na = len(merged_data)
            merged_data = merged_data.dropna()
            stats['na_removed'] = before_na - len(merged_data)
        
        stats['final_size'] = len(merged_data)
        
        return merged_data, stats
        
    except Exception as e:
        st.error(f"Erreur lors de la fusion des datasets: {e}")
        return None, None

def apply_robust_scaling(merged_data):
    """Applique le RobustScaler comme dans le preprocessing original"""
    try:
        # Identifier les colonnes numériques
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclure la variable cible (supposée être la dernière colonne)
        target_col = merged_data.columns[-1]
        numeric_cols_to_scale = [col for col in numeric_cols if col != target_col]
        
        # Appliquer RobustScaler
        scaler = RobustScaler()
        scaled_data = merged_data.copy()
        scaled_data[numeric_cols_to_scale] = scaler.fit_transform(merged_data[numeric_cols_to_scale])
        
        st.success(f"✅ Scaling appliqué avec RobustScaler sur {len(numeric_cols_to_scale)} features")
        return scaled_data
        
    except Exception as e:
        st.error(f"❌ Erreur lors du scaling: {e}")
        return merged_data

def retrain_model_on_merged_data(model, merged_data):
    """Réentraîne un modèle sur les données fusionnées"""
    try:
        # Séparer features et target
        X = merged_data.iloc[:, :-1].values
        y = merged_data.iloc[:, -1].values
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Métriques complètes
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # ROC AUC si disponible
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                metrics['auc_score'] = metrics['roc_auc']
            except:
                metrics['roc_auc'] = 0.5
                metrics['auc_score'] = 0.5
        
        return model, metrics, None
        
    except Exception as e:
        return model, None, f"Erreur lors du réentraînement: {e}"

def save_retrained_model_metrics(model_name, metrics, original_model_name, dataset_stats):
    """Sauvegarde les métriques du modèle réentraîné"""
    try:
        retrained_metrics_file = 'retrained_models_metrics.json'
        
        # Charger les métriques existantes
        if os.path.exists(retrained_metrics_file):
            with open(retrained_metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Ajouter les nouvelles métriques
        model_metrics = {
            'model_name': model_name,
            'original_model': original_model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'roc_auc': metrics.get('roc_auc', 0.5),
            'auc_score': metrics.get('auc_score', 0.5),
            'confusion_matrix': metrics['confusion_matrix'],
            'dataset_stats': dataset_stats,
            'retrain_time': pd.Timestamp.now().isoformat(),
            'model_type': 'Retrained'
        }
        
        all_metrics.append(model_metrics)
        
        # Sauvegarder
        with open(retrained_metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde des métriques: {e}")
        return False

def load_retrained_model_metrics():
    """Charge les métriques des modèles réentraînés"""
    try:
        retrained_metrics_file = 'retrained_models_metrics.json'
        if os.path.exists(retrained_metrics_file):
            with open(retrained_metrics_file, 'r') as f:
                return json.load(f)
        return []
    except:
        return []

def delete_retrained_model(model_name):
    """Supprime un modèle réentraîné et toutes ses données"""
    try:
        # Supprimer le fichier .pkl
        pkl_file = f"{model_name}.pkl"
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
            st.write(f"🗑️ Fichier modèle supprimé: {pkl_file}")
        
        # Supprimer les métriques du fichier JSON
        retrained_metrics_file = 'retrained_models_metrics.json'
        if os.path.exists(retrained_metrics_file):
            with open(retrained_metrics_file, 'r') as f:
                all_retrained_metrics = json.load(f)
            
            # Filtrer pour garder seulement les autres modèles
            updated_metrics = [m for m in all_retrained_metrics if m['model_name'] != model_name]
            
            with open(retrained_metrics_file, 'w') as f:
                json.dump(updated_metrics, f, indent=2)
            
            st.write(f"🗑️ Métriques supprimées pour: {model_name}")
        
        # Supprimer du session_state
        if model_name in st.session_state.custom_models:
            del st.session_state.custom_models[model_name]
        
        return True
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la suppression du modèle {model_name}: {e}")
        return False

def delete_all_retrained_models():
    """Supprime tous les modèles réentraînés"""
    try:
        # Identifier tous les modèles réentraînés
        retrained_models_to_delete = []
        for model_name in list(st.session_state.custom_models.keys()):
            if is_retrained_model(model_name):
                retrained_models_to_delete.append(model_name)
        
        # Supprimer chaque modèle réentraîné
        for model_name in retrained_models_to_delete:
            delete_retrained_model(model_name)
        
        st.success(f"✅ Tous les modèles réentraînés ont été supprimés ({len(retrained_models_to_delete)} modèles)")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la suppression: {e}")

def get_retrained_models_count():
    """Retourne le nombre de modèles réentraînés"""
    count = 0
    for model_name in st.session_state.custom_models.keys():
        if is_retrained_model(model_name):
            count += 1
    return count

def get_custom_models_count():
    """Retourne le nombre de modèles personnalisés"""
    count = 0
    for model_name in st.session_state.custom_models.keys():
        if is_custom_model(model_name):
            count += 1
    return count

# FONCTIONS D'AFFICHAGE
def display_all_model_metrics_real_time(selected_model, comparison_df, model_info):
    """Affiche TOUTES les métriques du modèle sélectionné en temps réel"""
    matching_model_name = find_matching_model_name(selected_model, comparison_df)
    
    # Vérifier si c'est un modèle réentraîné
    if is_retrained_model(selected_model):
        retrained_metrics = load_retrained_model_metrics()
        retrained_model_data = None
        for metric in retrained_metrics:
            if metric['model_name'] == selected_model:
                retrained_model_data = metric
                break
        
        if retrained_model_data:
            display_retrained_metrics_from_json(retrained_model_data, selected_model)
        else:
            st.warning(f"ℹ️ Aucune métrique disponible pour le modèle réentraîné '{selected_model}'")
        return
    
    # Vérifier si c'est un modèle custom
    if is_custom_model(selected_model):
        custom_metrics = load_custom_model_metrics()
        custom_model_data = None
        for metric in custom_metrics:
            if metric['model_name'] == selected_model:
                custom_model_data = metric
                break
        
        if custom_model_data:
            display_custom_metrics_from_json(custom_model_data, selected_model)
        else:
            st.warning(f"ℹ️ Aucune métrique disponible pour le modèle personnalisé '{selected_model}'")
        return
    
    # Sinon, c'est un modèle de base
    if not matching_model_name:
        st.warning(f"ℹ️ Aucune métrique disponible pour le modèle '{selected_model}'")
        return
    
    # Afficher les métriques du modèle de base
    model_data = comparison_df[comparison_df['model_name'] == matching_model_name].iloc[0]
    display_base_model_metrics(model_data, matching_model_name, model_info)

def display_retrained_metrics_from_json(model_data, model_name):
    """Affiche les métriques d'un modèle réentraîné depuis le JSON"""
    st.markdown(f"### 📊 Métriques Complètes du Modèle - {model_name}")
    
    with st.container():
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        
        # En-tête avec informations de réentraînement
        st.info(f"🔄 Modèle réentraîné à partir de: **{model_data['original_model']}**")
        st.write(f"**Date de réentraînement:** {pd.to_datetime(model_data['retrain_time']).strftime('%Y-%m-%d %H:%M')}")
        
        # 1. MÉTRIQUES PRINCIPALES
        st.subheader("🎯 Métriques Principales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model_data['accuracy']:.4f}")
            st.metric("Precision", f"{model_data['precision']:.4f}")
        with col2:
            st.metric("Recall", f"{model_data['recall']:.4f}")
            st.metric("F1 Macro", f"{model_data['f1_macro']:.4f}")
        with col3:
            st.metric("F1 Weighted", f"{model_data['f1_weighted']:.4f}")
            st.metric("ROC AUC", f"{model_data.get('roc_auc', 0):.4f}")
        with col4:
            st.metric("AUC Score", f"{model_data.get('auc_score', 0):.4f}")
            st.metric("Type", "Réentraîné")
        
        # 2. STATISTIQUES DU DATASET
        st.markdown("---")
        st.subheader("📈 Statistiques du Dataset")
        
        dataset_stats = model_data.get('dataset_stats', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taille originale", f"{dataset_stats.get('original_size', 0)}")
        with col2:
            st.metric("Doublons supprimés", f"{dataset_stats.get('duplicates_removed', 0)}")
        with col3:
            st.metric("NA supprimés", f"{dataset_stats.get('na_removed', 0)}")
        with col4:
            st.metric("Taille finale", f"{dataset_stats.get('final_size', 0)}")
        
        # 3. MATRICE DE CONFUSION
        st.markdown("---")
        st.subheader("🎯 Matrice de Confusion")
        
        cm = np.array(model_data['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Faux Positif", "Candidat", "Exoplanète"],
                    yticklabels=["Faux Positif", "Candidat", "Exoplanète"])
        ax.set_xlabel('Prédit')
        ax.set_ylabel('Réel')
        ax.set_title(f'Matrice de Confusion - {model_name}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_custom_metrics_from_json(model_data, model_name):
    """Affiche les métriques d'un modèle custom depuis le JSON"""
    st.markdown(f"### 📊 Métriques Complètes du Modèle - {model_name}")
    
    with st.container():
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        
        # 1. MÉTRIQUES PRINCIPALES
        st.subheader("🎯 Métriques Principales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model_data['accuracy']:.4f}")
            st.metric("Precision", f"{model_data['precision']:.4f}")
        with col2:
            st.metric("Recall", f"{model_data['recall']:.4f}")
            st.metric("F1 Macro", f"{model_data['f1_macro']:.4f}")
        with col3:
            st.metric("F1 Weighted", f"{model_data['f1_weighted']:.4f}")
            st.metric("ROC AUC", f"{model_data['roc_auc']:.4f}")
        with col4:
            st.metric("AUC Score", f"{model_data['auc_score']:.4f}")
            st.metric("Type", "Personnalisé")
        
        # 2. STATISTIQUES DÉTAILLÉES
        st.markdown("---")
        st.subheader("📈 Analyse de la Matrice de Confusion")
        
        cm = np.array(model_data['confusion_matrix'])
        
        total_samples = np.sum(cm)
        true_positives = np.diag(cm)
        false_positives = np.sum(cm, axis=0) - true_positives
        false_negatives = np.sum(cm, axis=1) - true_positives
        
        precision_per_class = true_positives / (true_positives + false_positives)
        recall_per_class = true_positives / (true_positives + false_negatives)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        
        classes = ["Faux Positif", "Candidat", "Exoplanète"]
        
        col1, col2, col3 = st.columns(3)
        
        for i, class_name in enumerate(classes):
            if i == 0:
                with col1:
                    st.markdown(f"**{class_name}**")
                    st.metric("Précision", f"{precision_per_class[i]:.3f}")
                    st.metric("Rappel", f"{recall_per_class[i]:.3f}")
                    st.metric("F1-Score", f"{f1_per_class[i]:.3f}")
            elif i == 1:
                with col2:
                    st.markdown(f"**{class_name}**")
                    st.metric("Précision", f"{precision_per_class[i]:.3f}")
                    st.metric("Rappel", f"{recall_per_class[i]:.3f}")
                    st.metric("F1-Score", f"{f1_per_class[i]:.3f}")
            else:
                with col3:
                    st.markdown(f"**{class_name}**")
                    st.metric("Précision", f"{precision_per_class[i]:.3f}")
                    st.metric("Rappel", f"{recall_per_class[i]:.3f}")
                    st.metric("F1-Score", f"{f1_per_class[i]:.3f}")
        
        # 3. MATRICE DE CONFUSION VISUELLE
        st.markdown("---")
        st.subheader("🎯 Matrice de Confusion")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=classes,
                    yticklabels=classes)
        ax.set_xlabel('Prédit')
        ax.set_ylabel('Réel')
        ax.set_title(f'Matrice de Confusion - {model_name}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # 4. HYPERPARAMÈTRES
        st.markdown("---")
        st.subheader("⚙️ Hyperparamètres du Modèle")
        st.json(model_data['hyperparameters'])
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_base_model_metrics(model_data, model_name, model_info):
    """Affiche les métriques d'un modèle de base"""
    st.markdown(f"### 📊 Métriques Complètes du Modèle - {model_name}")
    
    with st.container():
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        
        # 1. MÉTRIQUES PRINCIPALES
        st.subheader("🎯 Métriques Principales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model_data['accuracy']:.4f}")
            st.metric("Precision", f"{model_data['precision']:.4f}")
        with col2:
            st.metric("Recall", f"{model_data['recall']:.4f}")
            st.metric("F1 Macro", f"{model_data['f1_macro']:.4f}")
        with col3:
            st.metric("F1 Weighted", f"{model_data['f1_weighted']:.4f}")
            st.metric("ROC AUC", f"{model_data['roc_auc']:.4f}")
        with col4:
            st.metric("AUC Score", f"{model_data['auc_score']:.4f}")
            st.metric("Rang", f"{model_data['rank']:.0f}")
        
        # 2. STATISTIQUES DÉTAILLÉES
        st.markdown("---")
        st.subheader("📈 Analyse de la Matrice de Confusion")
        
        cm = np.array(model_data['confusion_matrix'])
        
        total_samples = np.sum(cm)
        true_positives = np.diag(cm)
        false_positives = np.sum(cm, axis=0) - true_positives
        false_negatives = np.sum(cm, axis=1) - true_positives
        
        precision_per_class = true_positives / (true_positives + false_positives)
        recall_per_class = true_positives / (true_positives + false_negatives)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        
        classes = ["Faux Positif", "Candidat", "Exoplanète"]
        
        col1, col2, col3 = st.columns(3)
        
        for i, class_name in enumerate(classes):
            if i == 0:
                with col1:
                    st.markdown(f"**{class_name}**")
                    st.metric("Précision", f"{precision_per_class[i]:.3f}")
                    st.metric("Rappel", f"{recall_per_class[i]:.3f}")
                    st.metric("F1-Score", f"{f1_per_class[i]:.3f}")
            elif i == 1:
                with col2:
                    st.markdown(f"**{class_name}**")
                    st.metric("Précision", f"{precision_per_class[i]:.3f}")
                    st.metric("Rappel", f"{recall_per_class[i]:.3f}")
                    st.metric("F1-Score", f"{f1_per_class[i]:.3f}")
            else:
                with col3:
                    st.markdown(f"**{class_name}**")
                    st.metric("Précision", f"{precision_per_class[i]:.3f}")
                    st.metric("Rappel", f"{recall_per_class[i]:.3f}")
                    st.metric("F1-Score", f"{f1_per_class[i]:.3f}")
        
        # 3. MATRICE DE CONFUSION VISUELLE
        st.markdown("---")
        st.subheader("🎯 Matrice de Confusion")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=classes,
                    yticklabels=classes)
        ax.set_xlabel('Prédit')
        ax.set_ylabel('Réel')
        ax.set_title(f'Matrice de Confusion - {model_name}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # 4. HYPERPARAMÈTRES
        st.markdown("---")
        st.subheader("⚙️ Hyperparamètres du Modèle")
        
        if isinstance(model_data['hyperparameters'], str):
            try:
                hyperparams = literal_eval(model_data['hyperparameters'])
                st.json(hyperparams)
            except:
                st.write(model_data['hyperparameters'])
        else:
            st.json(model_data['hyperparameters'])
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_model_comparison(df):
    """Affiche la comparaison des modèles"""
    st.subheader("📊 Comparaison des Modèles")
    
    display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_macro', 'f1_weighted', 'roc_auc', 'auc_score', 'rank']
    display_df = df[display_cols].copy()
    
    for col in ['accuracy', 'precision', 'recall', 'f1_macro', 'f1_weighted', 'roc_auc', 'auc_score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("📈 Graphiques de Comparaison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        models_plot = df.head(10)
        ax.barh(models_plot['model_name'], models_plot['accuracy'])
        ax.set_xlabel('Accuracy')
        ax.set_title('Comparaison de l\'Accuracy par Modèle')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(models_plot['model_name'], models_plot['f1_macro'])
        ax.set_xlabel('F1 Macro Score')
        ax.set_title('Comparaison du F1-Score par Modèle')
        plt.tight_layout()
        st.pyplot(fig)

def display_model_metrics(selected_model, df):
    """Affiche les métriques détaillées d'un modèle"""
    matching_model_name = find_matching_model_name(selected_model, df)
    
    if not matching_model_name:
        st.warning(f"Données non disponibles pour le modèle '{selected_model}'")
        return
    
    st.subheader(f"📊 Métriques Détaillées - {matching_model_name}")
    
    model_data = df[df['model_name'] == matching_model_name].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{model_data['accuracy']:.4f}")
        st.metric("Precision", f"{model_data['precision']:.4f}")
    with col2:
        st.metric("Recall", f"{model_data['recall']:.4f}")
        st.metric("F1 Macro", f"{model_data['f1_macro']:.4f}")
    with col3:
        st.metric("F1 Weighted", f"{model_data['f1_weighted']:.4f}")
        st.metric("ROC AUC", f"{model_data['roc_auc']:.4f}")
    with col4:
        st.metric("AUC Score", f"{model_data['auc_score']:.4f}")
        st.metric("Rang", f"{model_data['rank']:.0f}")
    
    st.subheader("📈 Matrice de Confusion")
    cm = np.array(model_data['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=["Faux Positif", "Candidat", "Exoplanète"],
                yticklabels=["Faux Positif", "Candidat", "Exoplanète"])
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title(f'Matrice de Confusion - {matching_model_name}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    st.subheader("⚙️ Hyperparamètres")
    if isinstance(model_data['hyperparameters'], str):
        try:
            hyperparams = literal_eval(model_data['hyperparameters'])
            st.json(hyperparams)
        except:
            st.write(model_data['hyperparameters'])
    else:
        st.json(model_data['hyperparameters'])

# FONCTIONS POUR LA GESTION DES FEATURES
def initialize_features():
    """Initialise les valeurs des features dans session_state"""
    if 'feature_values' not in st.session_state:
        default_values = {}
        for group_name, features in FEATURE_GROUPS.items():
            for feature_key, feature_name, min_val, max_val, default_val in features:
                default_values[feature_key] = default_val
        st.session_state.feature_values = default_values

def get_feature_description(feature_name):
    """Retourne la description d'une feature"""
    descriptions = {
        "koi_score": "Score de détection de l'exoplanète (0-1)",
        "koi_model_snr": "Rapport signal/bruit du transit",
        "habitability_index": "Indice d'habitabilité de la planète",
        "planet_density_proxy": "Densité estimée de la planète",
        "koi_prad": "Rayon de la planète en rayons terrestres",
        "koi_prad_err1": "Erreur sur le rayon de la planète",
        "koi_fpflag_nt": "Indicateur de faux positif non lié au transit",
        "koi_fpflag_ss": "Indicateur de faux positif dû à la variabilité stellaire",
        "koi_fpflag_co": "Indicateur de faux positif dû à la contamination",
        "koi_duration_err1": "Erreur sur la durée du transit",
        "duration_period_ratio": "Ratio durée du transit / période orbitale",
        "koi_time0bk_err1": "Erreur sur l'époque du transit",
        "koi_period": "Période orbitale en jours",
        "koi_depth": "Profondeur du transit en parties par million",
        "koi_impact": "Paramètre d'impact du transit",
        "koi_period_err1": "Erreur sur la période orbitale",
        "koi_steff_err1": "Erreur négative sur la température stellaire",
        "koi_steff_err2": "Erreur positive sur la température stellaire",
        "koi_slogg_err2": "Erreur sur la gravité de surface stellaire",
        "koi_insol": "Flux d'insolation en unités terrestres"
    }
    return descriptions.get(feature_name, "Caractéristique d'exoplanète")

# FONCTIONS POUR LA PRÉDICTION PAR LOT AMÉLIORÉE
def validate_batch_csv(uploaded_df):
    """Valide le fichier CSV pour la prédiction par lot"""
    try:
        # Vérifier le nombre de colonnes
        if uploaded_df.shape[1] < 20:
            return False, f"Le fichier doit contenir au moins 20 colonnes de caractéristiques. Actuel: {uploaded_df.shape[1]}"
        
        # Vérifier les valeurs manquantes
        if uploaded_df.isnull().any().any():
            missing_cols = uploaded_df.columns[uploaded_df.isnull().any()].tolist()
            return False, f"Valeurs manquantes détectées dans les colonnes: {missing_cols}"
        
        # Vérifier les types de données (doivent être numériques)
        for col in uploaded_df.columns[:20]:  # Vérifier les 20 premières colonnes
            if not pd.api.types.is_numeric_dtype(uploaded_df[col]):
                return False, f"La colonne '{col}' doit contenir des valeurs numériques"
        
        return True, "Fichier valide"
    
    except Exception as e:
        return False, f"Erreur lors de la validation: {str(e)}"

def process_batch_prediction(model, batch_df, model_name):
    """Traite la prédiction par lot et génère les résultats"""
    try:
        # Extraire les features (20 premières colonnes)
        batch_features = batch_df.iloc[:, :20].values
        
        # Faire les prédictions
        predictions_numeric = model.predict(batch_features)
        predictions_text = [CLASS_MAPPING.get(pred, "Inconnu") for pred in predictions_numeric]
        
        # Créer le DataFrame de résultats
        result_df = batch_df.copy()
        result_df['Prediction_Numérique'] = predictions_numeric
        result_df['Prediction_Classe'] = predictions_text
        
        # Ajouter les probabilités si disponibles
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(batch_features)
                for i in range(probabilities.shape[1]):
                    class_name = CLASS_MAPPING.get(i, f"Classe_{i}")
                    result_df[f'Confiance_{class_name}'] = probabilities[:, i]
                
                # Ajouter la confiance maximale
                result_df['Confiance_Maximale'] = np.max(probabilities, axis=1)
                result_df['Classe_Plus_Probable'] = [CLASS_MAPPING.get(np.argmax(prob), "Inconnu") for prob in probabilities]
                
            except Exception as proba_error:
                st.warning(f"⚠️ Impossible d'ajouter les probabilités: {proba_error}")
        
        # Statistiques des prédictions
        prediction_stats = pd.DataFrame({
            'Classe': list(CLASS_MAPPING.values()),
            'Nombre': [np.sum(predictions_numeric == i) for i in range(3)],
            'Pourcentage': [f"{(np.sum(predictions_numeric == i) / len(predictions_numeric)) * 100:.2f}%" 
                          for i in range(3)]
        })
        
        return result_df, prediction_stats, None
        
    except Exception as e:
        return None, None, f"Erreur lors du traitement par lot: {str(e)}"

# CHARGEMENT DES MODELES PERSONNALISES ET REENTRAINES
def load_custom_and_retrained_models():
    """Charge les modèles personnalisés et réentraînés depuis les fichiers .pkl"""
    custom_models = {}
    
    # Charger tous les fichiers .pkl
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    for model_file in pkl_files:
        try:
            model_name = model_file.replace('.pkl', '')
            
            # Ignorer les modèles de base (déjà chargés séparément)
            if model_name in st.session_state.models:
                continue
                
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Vérifier que c'est bien un modèle (a une méthode predict)
            if hasattr(model, 'predict'):
                custom_models[model_name] = model
                
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de {model_file}: {e}")
    
    return custom_models

# INITIALISATION
if 'models' not in st.session_state:
    st.session_state.models, st.session_state.model_info = load_models_with_dict()

if 'custom_models' not in st.session_state:
    st.session_state.custom_models = load_custom_and_retrained_models()

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

initialize_features()
metrics_data = load_model_metrics()
comparison_df = load_model_comparison()
dataset = load_dataset()

# INTERFACE UTILISATEUR PRINCIPALE
st.markdown("""
<div class="main-header">
    <h1>🚀 NASA Exoplanet Prediction</h1>
    <p>Plateforme avancée d'analyse et de prédiction d'exoplanètes</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR - DIAGNOSTIC
with st.sidebar:
    st.markdown("## 📦 Statut des Modèles")
    st.write(f"**Modèles de base:** {len(st.session_state.models)}")
    st.write(f"**Modèles personnalisés:** {get_custom_models_count()}")
    st.write(f"**Modèles réentraînés:** {get_retrained_models_count()}")
    
    if st.session_state.models and not comparison_df.empty:
        st.markdown("### 🔍 Correspondances")
        for model_name in st.session_state.models.keys():
            matching_name = find_matching_model_name(model_name, comparison_df)
            status = "✅" if matching_name else "❌"
            st.write(f"{status} {model_name} → {matching_name if matching_name else 'Aucune'}")

    st.markdown("---")
    st.subheader("🔗 Navigation Rapide")
    if st.sidebar.button("🎯 Aller à la Prédiction", use_container_width=True):
        st.rerun()

    if st.session_state.last_prediction is not None:
        if st.sidebar.button("📊 Voir les Résultats", use_container_width=True):
            st.rerun()

    if st.sidebar.button("🔄 Recharger tous les modèles", use_container_width=True):
        st.session_state.models, st.session_state.model_info = load_models_with_dict()
        st.session_state.custom_models = load_custom_and_retrained_models()
        st.rerun()

    st.markdown("---")
    st.subheader("💾 Gestion des Fichiers")
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    for pkl_file in pkl_files:
        st.write(f"• {pkl_file}")

# ONGLETS PRINCIPAUX
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prédiction", 
    "📊 Résultat & Feedback", 
    "📈 Modèles & Métriques", 
    "⚙️ Hyperparameter Tuning", 
    "🔄 Réentraînement"
])

# ONGLET 1: PRÉDICTION - CORRIGÉ
with tab1:
    st.header("🎯 Prédiction d'Exoplanètes")
    
    # Combiner tous les modèles disponibles
    all_models = {**st.session_state.models, **st.session_state.custom_models}
    
    working_models = [name for name, model in all_models.items() if model is not None and hasattr(model, 'predict')]
    
    if not working_models:
        st.error("❌ Aucun modèle fonctionnel disponible")
    else:
        selected_model = st.selectbox(
            "Choisir le modèle pour la prédiction:",
            working_models,
            help="Sélectionnez un modèle parmi les modèles disponibles",
            key="model_selector_prediction"
        )
        
        if selected_model:
            model = all_models[selected_model]
            
            # Déterminer le type de modèle
            is_custom_model_flag = is_custom_model(selected_model)
            is_retrained_model_flag = is_retrained_model(selected_model)
            
            # Afficher les métriques appropriées
            if is_retrained_model_flag:
                # Afficher les métriques du modèle réentraîné
                retrained_metrics = load_retrained_model_metrics()
                retrained_model_data = None
                for metric in retrained_metrics:
                    if metric['model_name'] == selected_model:
                        retrained_model_data = metric
                        break
                
                if retrained_model_data:
                    display_retrained_metrics_from_json(retrained_model_data, selected_model)
                else:
                    st.info("ℹ️ Modèle réentraîné - Métriques détaillées non disponibles")
            elif is_custom_model_flag:
                # Afficher les métriques du modèle personnalisé
                custom_metrics = load_custom_model_metrics()
                custom_model_data = None
                for metric in custom_metrics:
                    if metric['model_name'] == selected_model:
                        custom_model_data = metric
                        break
                
                if custom_model_data:
                    display_custom_metrics_from_json(custom_model_data, selected_model)
                else:
                    st.info("ℹ️ Modèle personnalisé - Métriques détaillées non disponibles")
            else:
                # Afficher les métriques du modèle de base
                model_info = st.session_state.model_info.get(selected_model, {})
                if not comparison_df.empty:
                    display_all_model_metrics_real_time(selected_model, comparison_df, model_info)
                else:
                    st.warning("ℹ️ Aucune donnée de métriques disponible")
                    st.info(f"Modèle: {selected_model}")
            
            st.markdown("---")
            st.subheader("📊 Caractéristiques d'Entrée")
            st.write("Veuillez saisir les valeurs pour les caractéristiques d'exoplanètes:")
            
            features = []
            feature_values_dict = {}
            
            for group_name, group_features in FEATURE_GROUPS.items():
                st.markdown(f'<div class="feature-group"><h4>{group_name}</h4></div>', unsafe_allow_html=True)
                
                cols = st.columns(3)
                for i, (feature_key, feature_name, min_val, max_val, default_val) in enumerate(group_features):
                    with cols[i % 3]:
                        current_value = st.session_state.feature_values.get(feature_key, default_val)
                        new_value = st.number_input(
                            label=feature_name,
                            value=float(current_value),
                            min_value=float(min_val),
                            max_value=float(max_val),
                            step=0.1 if max_val > 10 else 0.01,
                            format="%.6f",
                            key=f"pred_{feature_key}",
                            help=get_feature_description(feature_key)
                        )
                        st.session_state.feature_values[feature_key] = new_value
                        features.append(new_value)
                        feature_values_dict[feature_key] = new_value
            
            if len(features) != 20:
                st.error(f"❌ Erreur: Nombre de features incorrect. Attendu: 20, Obtenu: {len(features)}")
            else:
                if st.button("🚀 Lancer la Prédiction", type="primary", use_container_width=True):
                    try:
                        features_array = np.array(features).reshape(1, -1)
                        prediction = model.predict(features_array)
                        
                        probabilities = None
                        if hasattr(model, 'predict_proba'):
                            try:
                                probabilities = model.predict_proba(features_array)
                            except Exception as proba_error:
                                st.warning(f"⚠️ Impossible de récupérer les probabilités: {proba_error}")
                        
                        prediction_label = CLASS_MAPPING.get(prediction[0], "Inconnu")
                        
                        st.session_state.last_prediction = {
                            'model': selected_model,
                            'prediction': prediction[0],
                            'prediction_label': prediction_label,
                            'probabilities': probabilities[0] if probabilities is not None else None,
                            'features': features,
                            'feature_values': feature_values_dict,
                            'model_type': 'Retrained' if is_retrained_model_flag else 'Custom' if is_custom_model_flag else 'Base'
                        }
                        st.session_state.feedback_given = False
                        
                        st.success("✅ Prédiction effectuée avec succès! Redirection vers les résultats...")
                        js = """
                        <script>
                            const tabs = window.parent.document.querySelectorAll('[data-testid="stTab"]');
                            tabs.forEach(tab => {
                                if (tab.textContent.includes('📊 Résultat')) {
                                    tab.click();
                                }
                            });
                        </script>
                        """
                        st.components.v1.html(js, height=0)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            
            # SECTION PRÉDICTION PAR LOT AMÉLIORÉE
            st.markdown("---")
            st.subheader("📁 Prédiction par Lot (CSV)")
            
            # Instructions détaillées
            st.markdown("""
            <div class="batch-prediction-info">
            <h4>📋 Instructions pour la prédiction par lot</h4>
            <p><strong>Format requis:</strong> Fichier CSV avec au moins 20 colonnes de caractéristiques dans l'ordre suivant :</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher l'ordre des colonnes attendues
            col_order = []
            for group_name, group_features in FEATURE_GROUPS.items():
                for feature_key, feature_name, _, _, _ in group_features:
                    col_order.append(feature_name)
            
            st.info(f"**Ordre des colonnes attendues:** {', '.join(col_order[:5])}...")
            
            with st.expander("📋 Voir l'ordre complet des 20 colonnes"):
                for i, col_name in enumerate(col_order, 1):
                    st.write(f"{i}. {col_name}")
            
            uploaded_file = st.file_uploader("Charger un fichier CSV pour les prédictions par lot", 
                                           type=['csv'], 
                                           key="batch_prediction")
            
            if uploaded_file is not None:
                try:
                    # Charger et valider le fichier
                    batch_df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Fichier chargé: {batch_df.shape[0]} lignes, {batch_df.shape[1]} colonnes")
                    
                    # Validation du fichier
                    is_valid, validation_message = validate_batch_csv(batch_df)
                    
                    if not is_valid:
                        st.error(f"❌ {validation_message}")
                    else:
                        st.success("✅ Fichier validé avec succès")
                        
                        # Aperçu des données
                        with st.expander("👀 Aperçu des données chargées"):
                            st.dataframe(batch_df.head(10))
                            
                            # Statistiques descriptives
                            st.subheader("📊 Statistiques descriptives")
                            st.dataframe(batch_df.describe())
                        
                        # Bouton de prédiction par lot
                        if st.button("🎯 Lancer les Prédictions par Lot", 
                                   use_container_width=True, 
                                   key="batch_predict",
                                   type="primary"):
                            
                            with st.spinner("Traitement des prédictions par lot en cours..."):
                                # Traitement des prédictions
                                result_df, prediction_stats, error = process_batch_prediction(model, batch_df, selected_model)
                                
                                if error:
                                    st.error(f"❌ {error}")
                                else:
                                    st.success(f"✅ Prédictions terminées! {len(result_df)} lignes traitées")
                                    
                                    # Afficher les statistiques des prédictions
                                    st.subheader("📈 Statistiques des Prédictions")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Total des prédictions", len(result_df))
                                    with col2:
                                        exoplanets_count = len(result_df[result_df['Prediction_Classe'] == 'Exoplanète'])
                                        st.metric("Exoplanètes détectées", exoplanets_count)
                                    with col3:
                                        candidates_count = len(result_df[result_df['Prediction_Classe'] == 'Candidat'])
                                        st.metric("Candidats identifiés", candidates_count)
                                    
                                    # Graphique de distribution des classes
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    class_counts = result_df['Prediction_Classe'].value_counts()
                                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                                    bars = ax.bar(class_counts.index, class_counts.values, color=colors)
                                    ax.set_title('Distribution des Classes Prédites')
                                    ax.set_ylabel('Nombre d\'occurrences')
                                    
                                    # Ajouter les valeurs sur les barres
                                    for bar in bars:
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                                f'{int(height)}', ha='center', va='bottom')
                                    
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Aperçu des résultats
                                    st.subheader("👀 Aperçu des Résultats")
                                    st.dataframe(result_df.head(15))
                                    
                                    # Téléchargement des résultats
                                    st.subheader("📥 Téléchargement des Résultats")
                                    csv = result_df.to_csv(index=False)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            label="💾 Télécharger CSV Complet",
                                            data=csv,
                                            file_name=f"predictions_complete_{selected_model}.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                                    with col2:
                                        # Version simplifiée pour analyse rapide
                                        simple_cols = ['Prediction_Classe', 'Confiance_Maximale', 'Classe_Plus_Probable']
                                        available_cols = [col for col in simple_cols if col in result_df.columns]
                                        simple_df = result_df[available_cols]
                                        simple_csv = simple_df.to_csv(index=False)
                                        
                                        st.download_button(
                                            label="📄 Télécharger Résumé",
                                            data=simple_csv,
                                            file_name=f"predictions_summary_{selected_model}.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                                    
                                    # Section d'analyse avancée
                                    if 'Confiance_Maximale' in result_df.columns:
                                        st.markdown("---")
                                        st.subheader("🔍 Analyse de Confiance")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Distribution de la confiance
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            ax.hist(result_df['Confiance_Maximale'], bins=20, alpha=0.7, color='#667eea')
                                            ax.set_xlabel('Confiance Maximale')
                                            ax.set_ylabel('Fréquence')
                                            ax.set_title('Distribution de la Confiance des Prédictions')
                                            st.pyplot(fig)
                                        
                                        with col2:
                                            # Seuil de confiance
                                            confidence_threshold = st.slider(
                                                "Seuil de confiance pour filtrer les résultats:",
                                                0.5, 1.0, 0.8, 0.05
                                            )
                                            
                                            high_confidence = result_df[result_df['Confiance_Maximale'] >= confidence_threshold]
                                            st.metric(
                                                f"Prédictions avec confiance ≥ {confidence_threshold}",
                                                f"{len(high_confidence)} ({len(high_confidence)/len(result_df)*100:.1f}%)"
                                            )
                                            
                                            if len(high_confidence) > 0:
                                                st.dataframe(high_confidence[['Prediction_Classe', 'Confiance_Maximale']].head(10))
                            
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement du fichier CSV: {str(e)}")

# ONGLET 2: RÉSULTAT & FEEDBACK
with tab2:
    st.header("📊 Résultat de la Prédiction")
    
    if st.session_state.last_prediction is None:
        st.info("ℹ️ Aucune prédiction récente. Veuillez d'abord faire une prédiction dans l'onglet 'Prédiction'.")
    else:
        prediction_data = st.session_state.last_prediction
        
        st.markdown(f"""
        <div class="prediction-result">
            <h2>🎯 Résultat de la Prédiction</h2>
            <h3>Classe Prédite: {prediction_data['prediction_label']}</h3>
            <p><strong>Modèle utilisé:</strong> {prediction_data['model']}</p>
            <p><strong>Type de modèle:</strong> {prediction_data.get('model_type', 'Inconnu')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📋 Valeurs des Caractéristiques Utilisées")
        
        feature_display_data = []
        for group_name, group_features in FEATURE_GROUPS.items():
            for feature_key, feature_name, min_val, max_val, default_val in group_features:
                if feature_key in prediction_data['feature_values']:
                    feature_display_data.append({
                        'Groupe': group_name,
                        'Caractéristique': feature_name,
                        'Valeur': prediction_data['feature_values'][feature_key]
                    })
        
        feature_df = pd.DataFrame(feature_display_data)
        st.dataframe(feature_df, use_container_width=True)
        
        if prediction_data['probabilities'] is not None:
            st.subheader("📊 Probabilités par Classe")
            
            prob_data = []
            for i, prob in enumerate(prediction_data['probabilities']):
                class_name = CLASS_MAPPING.get(i, f"Classe {i}")
                prob_data.append({'Classe': class_name, 'Probabilité': prob})
            
            prob_df = pd.DataFrame(prob_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            bars = ax.bar(prob_df['Classe'], prob_df['Probabilité'], color=colors[:len(prob_df)])
            ax.set_ylabel('Probabilité')
            ax.set_title('Probabilités de Prédiction par Classe')
            ax.set_ylim(0, 1)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(prob_df)
        else:
            st.warning("⚠️ Les probabilités ne sont pas disponibles pour ce modèle")
        
        if not st.session_state.feedback_given:
            st.markdown("---")
            st.subheader("💬 Feedback sur la Prédiction")
            
            st.write("Cette prédiction vous semble-t-elle correcte?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("👍 Correct", use_container_width=True, key="feedback_correct"):
                    st.session_state.feedback_given = True
                    st.success("✅ Merci pour votre feedback!")
            with col2:
                if st.button("👎 Incorrect", use_container_width=True, key="feedback_incorrect"):
                    st.session_state.feedback_given = True
                    st.warning("⚠️ Merci pour votre feedback, nous améliorons constamment nos modèles.")
            with col3:
                if st.button("🤔 Incertain", use_container_width=True, key="feedback_uncertain"):
                    st.session_state.feedback_given = True
                    st.info("ℹ️ Merci pour votre retour!")
        else:
            st.success("✅ Feedback déjà donné pour cette prédiction.")

# ONGLET 3: MODÈLES & MÉTRIQUES
with tab3:
    st.header("📈 Analyse des Modèles")
    
    if comparison_df.empty:
        st.error("❌ Données de comparaison non disponibles")
    else:
        display_model_comparison(comparison_df)
        
        st.markdown("---")
        st.subheader("🔍 Analyse Détaillée par Modèle")
        
        selected_model_detail = st.selectbox(
            "Choisir un modèle pour voir les détails:",
            comparison_df['model_name'].tolist(),
            key="model_detail_select"
        )
        
        if selected_model_detail:
            display_model_metrics(selected_model_detail, comparison_df)
        
        st.markdown("---")
        st.subheader("🔧 Diagnostic des Modèles Chargés")
        
        if st.session_state.models:
            st.write(f"**Modèles chargés avec succès:** {len([m for m in st.session_state.model_info.values() if m['has_model']])}")
            st.write(f"**Modèles avec probabilités:** {len([m for m in st.session_state.model_info.values() if m.get('has_predict_proba', False)])}")
            
            with st.expander("📋 Détails du diagnostic des modèles"):
                for model_name, info in st.session_state.model_info.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        status = "✅" if info['has_model'] else "❌"
                        proba_status = "✅" if info.get('has_predict_proba', False) else "❌"
                        st.write(f"{status} **{model_name}** - {info['model_type']} - Probabilités: {proba_status}")
                    with col2:
                        if info.get('test_error'):
                            st.error("Erreur test")
                        elif info.get('test_success'):
                            st.success("Test OK")

# ONGLET 4: HYPERPARAMETER TUNING - CORRIGÉ
with tab4:
    st.header("⚙️ Personnalisation des Hyperparamètres")
    
    st.markdown("""
    <div class="metrics-container">
    <h3>🎯 Création de Modèles Personnalisés</h3>
    <p>Configurez vos propres hyperparamètres pour créer un modèle personnalisé. Le modèle sera automatiquement 
    entraîné sur le dataset Kepler et évalué. Maximum 10 modèles personnalisés.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sélection du type de modèle
    model_type = st.selectbox(
        "Type de modèle à personnaliser:",
        ["RandomForest", "XGBoost", "SVM"],
        key="model_type_select"
    )
    
    st.subheader("🎛️ Configuration des Hyperparamètres")
    
    # Configuration selon le type de modèle
    if model_type == "RandomForest":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 100, 50, 
                                   help="Nombre d'arbres dans la forêt")
            max_depth = st.slider("max_depth", 5, 50, 20, 
                                help="Profondeur maximale des arbres")
        with col2:
            min_samples_split = st.slider("min_samples_split", 2, 10, 2, 
                                        help="Nombre minimum d'échantillons pour diviser un nœud")
            min_samples_leaf = st.slider("min_samples_leaf", 1, 5, 1, 
                                       help="Nombre minimum d'échantillons dans une feuille")
        
        max_features = st.selectbox("max_features", ["sqrt", "log2", "auto"], 
                                  help="Nombre de features à considérer pour la meilleure division")
        
        hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }
    
    elif model_type == "XGBoost":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 100, 50,
                                   help="Nombre d'arbres boosting")
            max_depth = st.slider("max_depth", 3, 15, 7,
                                help="Profondeur maximale des arbres")
        with col2:
            learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01,
                                    help="Taux d'apprentissage pour le boosting")
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.1,
                                help="Fraction d'échantillons utilisés pour l'entraînement")
        
        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.1,
                                   help="Fraction de features utilisées pour chaque arbre")
        
        hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }
    
    elif model_type == "SVM":
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("C", 0.1, 10.0, 1.0, 0.1,
                        help="Paramètre de régularisation")
            kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"],
                               help="Type de noyau à utiliser")
        with col2:
            gamma = st.selectbox("gamma", ["scale", "auto"],
                               help="Coefficient du noyau")
            degree = st.slider("degree", 2, 5, 3,
                             help="Degré du polynôme (si kernel='poly')") if kernel == "poly" else 3
        
        hyperparams = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'degree': degree
        }
    
    # Nom du modèle personnalisé
    st.subheader("🏷️ Configuration du Modèle")
    
    # Générer un nom unique pour le modèle
    existing_custom_names = [k for k in st.session_state.custom_models.keys() if is_custom_model(k)]
    base_name = f"Custom_{model_type}"
    counter = 1
    while f"{base_name}_{counter}" in existing_custom_names:
        counter += 1
    default_name = f"{base_name}_{counter}"
    
    custom_model_name = st.text_input("Nom du modèle personnalisé:", 
                                    default_name, 
                                    key="custom_model_name",
                                    help="Donnez un nom unique à votre modèle personnalisé")
    
    # Informations sur les hyperparamètres
    st.markdown("---")
    st.subheader("📚 Explications des Hyperparamètres")
    
    if model_type == "RandomForest":
        st.markdown("""
        <div class="hyperparam-info">
        <strong>Random Forest - Explications:</strong><br>
        • <strong>n_estimators</strong>: Nombre d'arbres dans la forêt. Plus d'arbres améliore la performance mais augmente le temps de calcul.<br>
        • <strong>max_depth</strong>: Profondeur maximale des arbres. Évite le sur-apprentissage en limitant la complexité.<br>
        • <strong>min_samples_split</strong>: Nombre minimum d'échantillons requis pour diviser un nœud interne.<br>
        • <strong>min_samples_leaf</strong>: Nombre minimum d'échantillons requis dans un nœud feuille.<br>
        • <strong>max_features</strong>: Nombre de features à considérer pour chercher la meilleure division.
        </div>
        """, unsafe_allow_html=True)
    
    elif model_type == "XGBoost":
        st.markdown("""
        <div class="hyperparam-info">
        <strong>XGBoost - Explications:</strong><br>
        • <strong>n_estimators</strong>: Nombre d'arbres de boosting. Contrôle le nombre de rounds de boosting.<br>
        • <strong>max_depth</strong>: Profondeur maximale des arbres. Une valeur plus élevée permet des modèles plus complexes.<br>
        • <strong>learning_rate</strong>: Taux d'apprentissage. Réduit l'impact de chaque arbre pour éviter le sur-apprentissage.<br>
        • <strong>subsample</strong>: Fraction d'échantillons utilisés pour l'entraînement. Prévention du sur-apprentissage.<br>
        • <strong>colsample_bytree</strong>: Fraction de features utilisées pour construire chaque arbre.
        </div>
        """, unsafe_allow_html=True)
    
    elif model_type == "SVM":
        st.markdown("""
        <div class="hyperparam-info">
        <strong>SVM - Explications:</strong><br>
        • <strong>C</strong>: Paramètre de régularisation. Contrôle le trade-off entre marge d'erreur et complexité du modèle.<br>
        • <strong>kernel</strong>: Fonction noyau utilisée pour transformer les données (linéaire, RBF, polynomial, sigmoïde).<br>
        • <strong>gamma</strong>: Coefficient du noyau. Définit l'influence d'un seul exemple d'entraînement.<br>
        • <strong>degree</strong>: Degré de la fonction noyau polynomial (si kernel='poly').
        </div>
        """, unsafe_allow_html=True)
    
    # Boutons de création et gestion
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Créer et Entraîner le Modèle", type="primary", use_container_width=True, key="create_custom_model"):
            if not custom_model_name.strip():
                st.error("❌ Veuillez donner un nom à votre modèle")
            elif custom_model_name in st.session_state.models or custom_model_name in st.session_state.custom_models:
                st.error("❌ Ce nom de modèle existe déjà")
            else:
                with st.spinner("Création et entraînement du modèle personnalisé en cours..."):
                    # Vérifier et gérer la limite de 10 modèles
                    custom_models_count = get_custom_models_count()
                    if custom_models_count >= 10:
                        # Trouver le modèle custom le plus ancien
                        oldest_custom = None
                        for model_name in st.session_state.custom_models.keys():
                            if is_custom_model(model_name):
                                oldest_custom = model_name
                                break
                        
                        if oldest_custom:
                            # Supprimer le modèle le plus ancien
                            delete_custom_model(oldest_custom)
                            st.info(f"🗑️ Modèle '{oldest_custom}' supprimé (limite de 10 modèles atteinte)")
                    
                    # Créer le modèle
                    custom_model = create_custom_model(model_type, hyperparams)
                    
                    if custom_model is not None:
                        # Charger les données d'entraînement réelles
                        X_train, X_test, y_train, y_test = load_training_data()
                        
                        if X_train is not None:
                            # Entraîner et évaluer le modèle
                            metrics = train_and_evaluate_model(custom_model, X_train, X_test, y_train, y_test)
                            
                            if metrics:
                                # Sauvegarder les métriques dans JSON
                                save_success = save_custom_model_metrics(custom_model_name, metrics, hyperparams)
                                
                                # Stocker directement l'objet modèle
                                st.session_state.custom_models[custom_model_name] = custom_model
                                
                                # Sauvegarde du modèle
                                try:
                                    with open(f"{custom_model_name}.pkl", 'wb') as f:
                                        pickle.dump(custom_model, f)
                                    
                                    st.success(f"✅ Modèle personnalisé '{custom_model_name}' créé et entraîné avec succès!")
                                    st.metric("Accuracy obtenue", f"{metrics['accuracy']:.4f}")
                                    
                                    # Afficher les métriques immédiatement
                                    display_custom_metrics_from_json({
                                        'model_name': custom_model_name,
                                        **metrics,
                                        'hyperparameters': hyperparams
                                    }, custom_model_name)
                                    
                                    # Recharger la page pour mettre à jour la liste des modèles
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la sauvegarde: {e}")
                        else:
                            st.error("❌ Impossible de charger les données d'entraînement")
    
    with col2:
        custom_models_count = get_custom_models_count()
        if custom_models_count > 0:
            if st.button("🗑️ Supprimer Tous les Modèles Personnalisés", type="secondary", use_container_width=True, key="delete_all_custom"):
                delete_all_custom_models()
                st.rerun()
    
    # Affichage des modèles personnalisés existants
    custom_models_list = [k for k in st.session_state.custom_models.keys() if is_custom_model(k)]
    if custom_models_list:
        st.markdown("---")
        st.subheader("📋 Modèles Personnalisés Existants")
        
        for model_name in custom_models_list[:10]:  # Limite à 10
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{model_name}**")
                # Charger les métriques depuis le fichier JSON pour affichage
                custom_metrics = load_custom_model_metrics()
                model_metrics = None
                for metric in custom_metrics:
                    if metric['model_name'] == model_name:
                        model_metrics = metric
                        break
                
                if model_metrics:
                    st.write(f"Accuracy: {model_metrics.get('accuracy', 0):.4f}")
            
            with col2:
                # Essayer de récupérer le timestamp depuis les métriques
                if model_metrics and 'creation_time' in model_metrics:
                    try:
                        creation_time = pd.to_datetime(model_metrics['creation_time'])
                        st.write(f"Créé le: {creation_time.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.write("Date inconnue")
                else:
                    st.write("Date inconnue")
            
            with col3:
                if st.button("🗑️", key=f"delete_{model_name}"):
                    success = delete_custom_model(model_name)
                    if success:
                        st.success(f"✅ Modèle '{model_name}' supprimé")
                        st.rerun()

# ONGLET 5: RÉENTRAÎNEMENT AVEC PRÉPROCESSING AVANCÉ
with tab5:
    # Initialize session_state variables
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'merge_stats' not in st.session_state:
        st.session_state.merge_stats = None
    if 'scaled_data' not in st.session_state:
        st.session_state.scaled_data = None
    
    st.header("🔄 Réentraînement des Modèles avec Nouvelles Données")
    
    st.markdown("""
    <div class="retrain-info">
    <h3>🎯 Réentraînement Avancé avec Préprocessing</h3>
    <p>Chargez de nouvelles données pour réentraîner un modèle existant. Les nouvelles données subiront le même 
    préprocessing avancé que le dataset original (nettoyage, imputation, encodage de koi_disposition, feature engineering) 
    avant d'être fusionnées avec le dataset Kepler original.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger le dataset de base pour le réentraînement
    base_dataset = load_base_dataset_for_retraining()
    
    if base_dataset is None:
        st.error("❌ Impossible de charger le dataset de base pour le réentraînement")
        st.info("ℹ️ Assurez-vous que le fichier 'kepler_before_scaling_selected.csv' est présent dans le répertoire")
    else:
        st.success(f"✅ Dataset de base chargé: {base_dataset.shape[0]} échantillons, {base_dataset.shape[1]} caractéristiques")
        st.info(f"📋 Features attendues: {list(base_dataset.columns)}")
    
    # Section de chargement des données
    st.subheader("📁 Chargement des Nouvelles Données")
    
    uploaded_file = st.file_uploader(
        "Charger un nouveau dataset CSV", 
        type=['csv'], 
        key="retrain_upload",
        help="Le dataset subira un préprocessing avancé avant la fusion (incluant l'encodage de koi_disposition)"
    )
    
    if uploaded_file is not None and base_dataset is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Nouveau dataset chargé: {new_data.shape[0]} lignes, {new_data.shape[1]} colonnes")
            
            # Validation du dataset
            if validate_dataset_for_retraining(new_data, base_dataset):
                st.success("✅ Structure du dataset validée")
                
                # Aperçu des données
                with st.expander("👀 Aperçu du nouveau dataset"):
                    st.dataframe(new_data.head())
                    st.write(f"**Dimensions:** {new_data.shape[0]} lignes × {new_data.shape[1]} colonnes")
                    
                    # Vérification des features nécessaires pour le feature engineering
                    required_engineering_features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_duration']
                    available_features = [f for f in required_engineering_features if f in new_data.columns]
                    missing_features = set(required_engineering_features) - set(available_features)
                    
                    if missing_features:
                        st.warning(f"⚠️ Features manquantes pour l'ingénierie: {missing_features}")
                    else:
                        st.success("✅ Toutes les features nécessaires pour l'ingénierie sont présentes")
                    
                    # Vérification de koi_disposition
                    if 'koi_disposition' in new_data.columns:
                        st.success("✅ Colonne koi_disposition trouvée - elle sera encodée et supprimée")
                        # Afficher la distribution des valeurs
                        disposition_distribution = new_data['koi_disposition'].value_counts()
                        st.write("📊 Distribution de koi_disposition:")
                        for value, count in disposition_distribution.items():
                            st.write(f"  - {value}: {count} échantillons")
                    else:
                        st.warning("⚠️ Colonne koi_disposition non trouvée - vérifiez que la variable cible est présente")
                    
                    # Distribution des classes dans le nouveau dataset
                    if new_data.shape[1] > 0 and 'koi_disposition' in new_data.columns:
                        target_col = new_data['koi_disposition']
                        class_distribution = target_col.value_counts()
                        st.write("📊 Distribution des classes dans le nouveau dataset:")
                        for class_val, count in class_distribution.items():
                            st.write(f"  - {class_val}: {count} échantillons")
                
                # Section de prétraitement avancé
                st.subheader("🔄 Prétraitement Avancé du Dataset Uploadé")
                
                # Options de prétraitement
                col1, col2 = st.columns(2)
                with col1:
                    remove_duplicates = st.checkbox("Supprimer les doublons", value=True, 
                                                   help="Supprime les lignes en double")
                with col2:
                    remove_na = st.checkbox("Supprimer les valeurs manquantes résiduelles", value=True,
                                           help="Supprime les lignes avec des valeurs manquantes après imputation")
                
                # Bouton de prétraitement avancé
                if st.button("🔍 Appliquer le Préprocessing Avancé", key="preprocess_uploaded"):
                    with st.spinner("Prétraitement avancé en cours..."):
                        # Prétraiter le dataset uploadé avec le préprocessing avancé
                        processed_data, preprocess_stats = preprocess_uploaded_dataset(new_data, base_dataset)
                        
                        if processed_data is not None:
                            # Stocker dans session_state pour utilisation ultérieure
                            st.session_state.processed_data = processed_data
                            st.session_state.preprocess_stats = preprocess_stats
                            st.success("✅ Prétraitement avancé terminé avec succès!")
                            
                            # Afficher les statistiques de prétraitement
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Échantillons originaux", preprocess_stats['original_samples'])
                            with col2:
                                st.metric("Échantillons finaux", preprocess_stats['final_samples'])
                            with col3:
                                st.metric("Features originelles", preprocess_stats['original_features'])
                            with col4:
                                st.metric("Features finales", preprocess_stats['final_features'])
                            
                            # Vérifier si koi_disposition_encoded est présente
                            if 'koi_disposition_encoded' in processed_data.columns:
                                st.success("✅ koi_disposition encodée avec succès en koi_disposition_encoded")
                                
                                # Afficher la distribution des classes encodées
                                encoded_distribution = processed_data['koi_disposition_encoded'].value_counts().sort_index()
                                st.write("📊 Distribution des classes après encodage:")
                                for class_val, count in encoded_distribution.items():
                                    class_name = CLASS_MAPPING.get(int(class_val), f"Classe {class_val}")
                                    st.write(f"  - {class_name} ({class_val}): {count} échantillons")
                            
                # Section de fusion avec le dataset original
                st.subheader("🔄 Fusion avec le Dataset Original")
                
                st.info(f"📊 Dataset original Kepler: {base_dataset.shape[0]} échantillons")
                
                # Simulation de la fusion
                if st.button("🔍 Simuler la fusion", key="simulate_merge"):
                    if 'processed_data' not in st.session_state:
                        st.error("❌ Veuillez d'abord appliquer le préprocessing avancé")
                    else:
                        with st.spinner("Simulation de la fusion en cours..."):
                            try:
                                # Fusionner les datasets
                                merged_data, stats = merge_and_clean_datasets(
                                    base_dataset, st.session_state.processed_data, remove_duplicates, remove_na
                                )
                                
                                if merged_data is not None:
                                    st.session_state.merged_data = merged_data
                                    st.session_state.merge_stats = stats
                                    st.success(f"✅ Fusion simulée réussie!")
                                    
                                    # Afficher les statistiques de fusion
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Taille originale", f"{stats['original_size']}")
                                    with col2:
                                        st.metric("Doublons supprimés", f"{stats['duplicates_removed']}")
                                    with col3:
                                        st.metric("NA supprimés", f"{stats['na_removed']}")
                                    with col4:
                                        st.metric("Taille finale", f"{stats['final_size']}")
                                    
                                    # Distribution des classes après fusion
                                    target_col = merged_data.iloc[:, -1]
                                    class_distribution = target_col.value_counts().sort_index()
                                    st.write("📊 Distribution des classes après fusion:")
                                    for class_val, count in class_distribution.items():
                                        class_name = CLASS_MAPPING.get(int(class_val), f"Classe {class_val}")
                                        st.write(f"  - {class_name}: {count} échantillons")
                                    
                                    # Appliquer le scaling
                                    st.subheader("⚖️ Application du Scaling")
                                    scaled_data = apply_robust_scaling(merged_data)
                                    
                                    if scaled_data is not None:
                                        st.session_state.scaled_data = scaled_data
                                        st.success("✅ Scaling appliqué avec succès!")
                                        
                                else:
                                    st.error("❌ Erreur lors de la simulation de fusion")
                                
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la simulation: {str(e)}")
                
                # Afficher les résultats de simulation s'ils existent
                if 'merge_stats' in st.session_state:
                    st.info("📋 Résultats de simulation disponibles")
                
                # Section de sélection du modèle
                st.subheader("🎯 Sélection du Modèle à Réentraîner")
                
                # Combiner tous les modèles disponibles (base + personnalisés + réentraînés)
                all_available_models = {**st.session_state.models, **st.session_state.custom_models}
                working_models = {name: model for name, model in all_available_models.items() 
                                if model is not None and hasattr(model, 'predict') and hasattr(model, 'fit')}
                
                if not working_models:
                    st.error("❌ Aucun modèle fonctionnel disponible pour le réentraînement")
                else:
                    selected_retrain_model = st.selectbox(
                        "Modèle à réentraîner:",
                        list(working_models.keys()),
                        key="retrain_model_select",
                        help="Sélectionnez le modèle que vous souhaitez réentraîner avec les nouvelles données"
                    )
                    
                    if selected_retrain_model:
                        # Afficher les informations du modèle sélectionné
                        model_obj = working_models[selected_retrain_model]
                        st.info(f"**Type de modèle:** {type(model_obj).__name__}")
                        
                        # Générer un nom pour le modèle réentraîné
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        retrained_model_name = f"Retrained_{selected_retrain_model}_{timestamp}"
                        
                        st.write(f"**Nouveau nom du modèle:** {retrained_model_name}")
                        
                        # Bouton de réentraînement complet
                        if st.button("🔄 Lancer le Réentraînement Complet", type="primary", use_container_width=True, key="retrain_button"):
                            with st.spinner("Réentraînement en cours... Cela peut prendre quelques minutes."):
                                try:
                                    # Vérifier que les données nécessaires sont dans session_state
                                    if ('processed_data' not in st.session_state or 
                                        'merge_stats' not in st.session_state or 
                                        'scaled_data' not in st.session_state):
                                        st.error("❌ Veuillez d'abord simuler la fusion complète")
                                        st.stop()

                                    processed_data = st.session_state.processed_data
                                    merged_data = st.session_state.merged_data
                                    scaled_data = st.session_state.scaled_data
                                    
                                    # Cloner le modèle original (important pour ne pas le modifier)
                                    original_model = working_models[selected_retrain_model]
                                    
                                    # Pour scikit-learn models, nous pouvons utiliser clone
                                    from sklearn.base import clone
                                    try:
                                        model_to_retrain = clone(original_model)
                                    except:
                                        # Fallback: utiliser le même type de modèle avec mêmes paramètres
                                        st.warning("⚠️ Impossible de cloner le modèle, utilisation du modèle original")
                                        model_to_retrain = original_model
                                    
                                    # Réentraîner le modèle sur les données fusionnées et scaled
                                    retrained_model, metrics, error = retrain_model_on_merged_data(
                                        model_to_retrain, scaled_data
                                    )
                                    
                                    if error:
                                        st.error(f"❌ {error}")
                                    else:
                                        st.success("✅ Réentraînement terminé avec succès!")
                                        
                                        # Sauvegarder le modèle réentraîné
                                        with open(f"{retrained_model_name}.pkl", 'wb') as f:
                                            pickle.dump(retrained_model, f)
                                        
                                        # Ajouter au session_state
                                        st.session_state.custom_models[retrained_model_name] = retrained_model
                                        
                                        # Combiner les statistiques
                                        combined_stats = {
                                            **st.session_state.merge_stats,
                                            'preprocessing_stats': st.session_state.preprocess_stats,
                                            'base_dataset_size': len(base_dataset),
                                            'uploaded_dataset_size': len(new_data),
                                            'final_dataset_size': len(scaled_data)
                                        }
                                        
                                        # Sauvegarder les métriques
                                        save_success = save_retrained_model_metrics(
                                            retrained_model_name, metrics, selected_retrain_model, combined_stats
                                        )
                                        
                                        if save_success:
                                            st.success(f"💾 Modèle réentraîné sauvegardé sous '{retrained_model_name}.pkl'")
                                            
                                            # Afficher les métriques complètes
                                            display_retrained_metrics_from_json({
                                                'model_name': retrained_model_name,
                                                'original_model': selected_retrain_model,
                                                **metrics,
                                                'dataset_stats': combined_stats,
                                                'retrain_time': pd.Timestamp.now().isoformat()
                                            }, retrained_model_name)
                                            
                                            st.balloons()
                                            st.rerun()
                                            st.markdown("---")
                                            st.success("🎉 Réentraînement terminé avec succès! Le modèle est maintenant disponible dans la liste des modèles.")
                                            st.info(f"🔍 Le modèle '{retrained_model_name}' est maintenant sélectionnable dans l'onglet 'Prédiction'")
                                    
                                except Exception as e:
                                    st.error(f"❌ Erreur lors du réentraînement: {str(e)}")                                    # 4. Cloner le modèle original (important pour ne pas le modifier)
                                    original_model = working_models[selected_retrain_model]
                                    
                                    # Pour scikit-learn models, nous pouvons utiliser clone
                                    from sklearn.base import clone
                                    try:
                                        model_to_retrain = clone(original_model)
                                    except:
                                        # Fallback: utiliser le même type de modèle avec mêmes paramètres
                                        st.warning("⚠️ Impossible de cloner le modèle, utilisation du modèle original")
                                        model_to_retrain = original_model
                                    
                                    # 5. Réentraîner le modèle sur les données fusionnées et scaled
                                    retrained_model, metrics, error = retrain_model_on_merged_data(
                                        model_to_retrain, scaled_data
                                    )
                                    
                                    if error:
                                        st.error(f"❌ {error}")
                                    else:
                                        st.success("✅ Réentraînement terminé avec succès!")
                                        
                                        # Afficher les métriques
                                        st.subheader("📊 Métriques du Modèle Réentraîné")
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        with col1:
                                            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                                        with col2:
                                            st.metric("Precision", f"{metrics['precision']:.4f}")
                                        with col3:
                                            st.metric("Recall", f"{metrics['recall']:.4f}")
                                        with col4:
                                            st.metric("F1 Score", f"{metrics['f1_weighted']:.4f}")
                                        
                                        # Sauvegarder le modèle réentraîné
                                        try:
                                            with open(f"{retrained_model_name}.pkl", 'wb') as f:
                                                pickle.dump(retrained_model, f)
                                            
                                            # Ajouter au session_state
                                            st.session_state.custom_models[retrained_model_name] = retrained_model
                                            
                                            # Combiner les statistiques
                                            combined_stats = {
                                                **stats,
                                                'preprocessing_stats': preprocess_stats,
                                                'base_dataset_size': len(base_dataset),
                                                'uploaded_dataset_size': len(new_data),
                                                'final_dataset_size': len(scaled_data)
                                            }
                                            
                                            # Sauvegarder les métriques
                                            save_success = save_retrained_model_metrics(
                                                retrained_model_name, metrics, selected_retrain_model, combined_stats
                                            )
                                            
                                            if save_success:
                                                st.success(f"💾 Modèle réentraîné sauvegardé sous '{retrained_model_name}.pkl'")
                                                
                                                # Afficher les métriques complètes
                                                display_retrained_metrics_from_json({
                                                    'model_name': retrained_model_name,
                                                    'original_model': selected_retrain_model,
                                                    **metrics,
                                                    'dataset_stats': combined_stats,
                                                    'retrain_time': pd.Timestamp.now().isoformat()
                                                }, retrained_model_name)
                                                
                                                st.balloons()
                                                st.markdown("---")
                                                st.success("🎉 Réentraînement terminé avec succès! Le modèle est maintenant disponible dans la liste des modèles.")
                                                st.info(f"🔍 Le modèle '{retrained_model_name}' est maintenant sélectionnable dans l'onglet 'Prédiction'")
                                                
                                            
                                        except Exception as e:
                                            st.error(f"❌ Erreur lors de la sauvegarde: {e}")
                                
                                except Exception as e:
                                    st.error(f"❌ Erreur lors du réentraînement: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
    else:
        if base_dataset is not None:
            st.info("ℹ️ Veuillez charger un fichier CSV pour commencer le réentraînement")
    
    # Section de gestion des modèles réentraînés
    st.markdown("---")
    st.subheader("📋 Gestion des Modèles Réentraînés")
    
    # Afficher les modèles réentraînés existants
    retrained_models_list = [k for k in st.session_state.custom_models.keys() if is_retrained_model(k)]
    
    if retrained_models_list:
        st.write(f"**{len(retrained_models_list)} modèle(s) réentraîné(s) disponible(s):**")
        
        for model_name in retrained_models_list:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                # Trouver les métriques correspondantes
                model_metrics = None
                for metric in load_retrained_model_metrics():
                    if metric['model_name'] == model_name:
                        model_metrics = metric
                        break
                
                if model_metrics:
                    original_model = model_metrics.get('original_model', 'Inconnu')
                    accuracy = model_metrics.get('accuracy', 0)
                    st.write(f"**{model_name}**")
                    st.write(f"Original: {original_model} | Accuracy: {accuracy:.4f}")
                else:
                    st.write(f"**{model_name}**")
            
            with col2:
                if model_metrics and 'retrain_time' in model_metrics:
                    try:
                        retrain_time = pd.to_datetime(model_metrics['retrain_time'])
                        st.write(f"Réentraîné le: {retrain_time.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.write("Date inconnue")
                else:
                    st.write("Date inconnue")
            
            with col3:
                if st.button("🗑️", key=f"delete_retrained_{model_name}"):
                    success = delete_retrained_model(model_name)
                    if success:
                        st.success(f"✅ Modèle réentraîné '{model_name}' supprimé")
                        st.rerun()
        
        # Bouton pour supprimer tous les modèles réentraînés
        if st.button("🗑️ Supprimer Tous les Modèles Réentraînés", type="secondary", use_container_width=True):
            delete_all_retrained_models()
            st.rerun()
    else:
        st.info("ℹ️ Aucun modèle réentraîné disponible")
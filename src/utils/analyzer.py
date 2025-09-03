"""
Analysis utilities for fundus disease classification
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


class StatisticalAnalyzer:
    """Statistical analysis utilities"""
    
    def __init__(self, df):
        self.df = df
        
    def demographic_analysis(self):
        """Analyze demographic characteristics"""
        print("=== DEMOGRAPHIC ANALYSIS ===")
        
        results = {}
        
        # Age analysis
        print("\n--- Age Distribution ---")
        age_stats = self.df.groupby('disease_type')['age'].describe()
        print(age_stats)
        
        inflammatory_age = self.df[self.df['disease_type'] == 'Inflammatory_ON']['age'].dropna()
        ischemic_age = self.df[self.df['disease_type'] == 'Ischemic_ON']['age'].dropna()
        
        age_ttest = stats.ttest_ind(inflammatory_age, ischemic_age)
        print(f"Age comparison (t-test): t={age_ttest.statistic:.3f}, p={age_ttest.pvalue:.6f}")
        
        # Sex distribution
        print("\n--- Sex Distribution ---")
        sex_crosstab = pd.crosstab(self.df['disease_type'], self.df['sex'], margins=True)
        print(sex_crosstab)
        
        sex_chi2 = stats.chi2_contingency(pd.crosstab(self.df['disease_type'], self.df['sex']))
        print(f"Sex distribution (chi-square): χ²={sex_chi2[0]:.3f}, p={sex_chi2[1]:.6f}")
        
        # Affected side distribution
        print("\n--- Affected Side Distribution ---")
        side_crosstab = pd.crosstab(self.df['disease_type'], self.df['affected_side'], margins=True)
        print(side_crosstab)
        
        side_chi2 = stats.chi2_contingency(pd.crosstab(self.df['disease_type'], self.df['affected_side']))
        print(f"Affected side (chi-square): χ²={side_chi2[0]:.3f}, p={side_chi2[1]:.6f}")
        
        results.update({
            'age_stats': age_stats,
            'age_ttest': age_ttest,
            'sex_crosstab': sex_crosstab,
            'sex_chi2': sex_chi2,
            'side_crosstab': side_crosstab,
            'side_chi2': side_chi2
        })
        
        return results
    
    def clinical_features_analysis(self):
        """Analyze clinical features"""
        print("\n=== CLINICAL FEATURES ANALYSIS ===")
        
        results = {}
        
        # Pain analysis
        print("\n--- Pain Analysis ---")
        pain_crosstab = pd.crosstab(self.df['disease_type'], self.df['pain_binary'], margins=True)
        print(pain_crosstab)
        print("Note: 0=No, 1=Yes")
        
        pain_chi2 = stats.chi2_contingency(pd.crosstab(self.df['disease_type'], self.df['pain_binary']))
        print(f"Pain association (chi-square): χ²={pain_chi2[0]:.3f}, p={pain_chi2[1]:.6f}")
        
        # Visual Acuity analysis
        print("\n--- Visual Acuity Analysis ---")
        va_stats = self.df.groupby('disease_type')['affected_eye_va'].describe()
        print("Affected Eye Visual Acuity:")
        print(va_stats)
        
        inflammatory_va = self.df[self.df['disease_type'] == 'Inflammatory_ON']['affected_eye_va'].dropna()
        ischemic_va = self.df[self.df['disease_type'] == 'Ischemic_ON']['affected_eye_va'].dropna()
        
        if len(inflammatory_va) > 0 and len(ischemic_va) > 0:
            va_ttest = stats.ttest_ind(inflammatory_va, ischemic_va)
            print(f"VA comparison (t-test): t={va_ttest.statistic:.3f}, p={va_ttest.pvalue:.6f}")
            
            va_mannwhitney = stats.mannwhitneyu(inflammatory_va, ischemic_va, alternative='two-sided')
            print(f"VA comparison (Mann-Whitney U): U={va_mannwhitney.statistic:.3f}, p={va_mannwhitney.pvalue:.6f}")
        
        # IOP analysis
        print("\n--- Intraocular Pressure Analysis ---")
        iop_stats = self.df.groupby('disease_type')['affected_eye_iop'].describe()
        print("Affected Eye IOP:")
        print(iop_stats)
        
        inflammatory_iop = self.df[self.df['disease_type'] == 'Inflammatory_ON']['affected_eye_iop'].dropna()
        ischemic_iop = self.df[self.df['disease_type'] == 'Ischemic_ON']['affected_eye_iop'].dropna()
        
        if len(inflammatory_iop) > 0 and len(ischemic_iop) > 0:
            iop_ttest = stats.ttest_ind(inflammatory_iop, ischemic_iop)
            print(f"IOP comparison (t-test): t={iop_ttest.statistic:.3f}, p={iop_ttest.pvalue:.6f}")
        
        # RNFL analysis
        print("\n--- RNFL Thickness Analysis ---")
        rnfl_stats = self.df.groupby('disease_type')['affected_eye_rnfl'].describe()
        print("Affected Eye RNFL:")
        print(rnfl_stats)
        
        inflammatory_rnfl = self.df[self.df['disease_type'] == 'Inflammatory_ON']['affected_eye_rnfl'].dropna()
        ischemic_rnfl = self.df[self.df['disease_type'] == 'Ischemic_ON']['affected_eye_rnfl'].dropna()
        
        if len(inflammatory_rnfl) > 0 and len(ischemic_rnfl) > 0:
            rnfl_ttest = stats.ttest_ind(inflammatory_rnfl, ischemic_rnfl)
            print(f"RNFL comparison (t-test): t={rnfl_ttest.statistic:.3f}, p={rnfl_ttest.pvalue:.6f}")
        
        # RAPD analysis
        print("\n--- RAPD Analysis ---")
        rapd_crosstab = pd.crosstab(self.df['disease_type'], self.df['rapd_code'], margins=True)
        print("RAPD Code Distribution:")
        print(rapd_crosstab)
        
        return results


class MLAnalyzer:
    """Machine learning analysis utilities"""
    
    def __init__(self, df):
        self.df = df
        self.model = None
        self.scaler = None
        
    def prepare_features(self, feature_columns=None):
        """Prepare features for ML analysis"""
        if feature_columns is None:
            feature_columns = ['age', 'pain_binary', 'affected_eye_va', 'affected_eye_iop', 'affected_eye_rnfl']
        
        X = self.df[feature_columns].copy()
        
        # Handle missing values with median imputation
        for col in feature_columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"Imputed {X[col].isnull().sum()} missing values in {col} with median {median_val:.3f}")
        
        # Create target variable (0=Inflammatory, 1=Ischemic)
        y = (self.df['disease_type'] == 'Ischemic_ON').astype(int)
        
        return X, y, feature_columns
    
    def train_random_forest(self, X, y, feature_columns, test_size=0.3, random_state=42):
        """Train Random Forest classifier"""
        print("\n=== MACHINE LEARNING ANALYSIS ===")
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                          random_state=random_state, stratify=y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Test set performance
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Test set accuracy: {test_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Classification report
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Inflammatory_ON', 'Ischemic_ON']))
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': feature_importance,
            'cv_scores': cv_scores,
            'test_score': test_score,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred
        }

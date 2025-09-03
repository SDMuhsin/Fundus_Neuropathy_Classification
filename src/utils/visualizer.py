"""
Visualization utilities for fundus disease classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Visualizer:
    """Comprehensive visualization utilities"""
    
    def __init__(self, df):
        self.df = df
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_comprehensive_plots(self, results_dir="results"):
        """Create comprehensive visualization suite"""
        print("\n=== CREATING COMPREHENSIVE VISUALIZATIONS ===")
        
        # Set up the plotting area
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Age distribution
        plt.subplot(4, 4, 1)
        inflammatory_age = self.df[self.df['disease_type'] == 'Inflammatory_ON']['age'].dropna()
        ischemic_age = self.df[self.df['disease_type'] == 'Ischemic_ON']['age'].dropna()
        
        plt.hist([inflammatory_age, ischemic_age], bins=20, alpha=0.7, 
                 label=['Inflammatory ON', 'Ischemic ON'], color=['skyblue', 'lightcoral'])
        plt.xlabel('Age (years)')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Disease Type')
        plt.legend()
        
        # 2. Pain distribution
        plt.subplot(4, 4, 2)
        pain_counts = pd.crosstab(self.df['disease_type'], self.df['pain_binary'])
        pain_counts.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
        plt.title('Pain Presence by Disease Type')
        plt.xlabel('Disease Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Pain (0=No, 1=Yes)')
        
        # 3. Visual Acuity distribution (log scale)
        plt.subplot(4, 4, 3)
        inflammatory_va = self.df[self.df['disease_type'] == 'Inflammatory_ON']['affected_eye_va'].dropna()
        ischemic_va = self.df[self.df['disease_type'] == 'Ischemic_ON']['affected_eye_va'].dropna()
        
        if len(inflammatory_va) > 0 and len(ischemic_va) > 0:
            plt.hist([np.log10(inflammatory_va + 0.0001), np.log10(ischemic_va + 0.0001)], 
                     bins=20, alpha=0.7, label=['Inflammatory ON', 'Ischemic ON'])
            plt.xlabel('Log10(Visual Acuity + 0.0001)')
            plt.ylabel('Frequency')
            plt.title('Visual Acuity Distribution (Log Scale)')
            plt.legend()
        
        # 4. RAPD certainty distribution
        plt.subplot(4, 4, 4)
        if 'rapd_certainty' in self.df.columns:
            rapd_counts = pd.crosstab(self.df['disease_type'], self.df['rapd_certainty'])
            rapd_counts.plot(kind='bar', ax=plt.gca())
            plt.title('RAPD Certainty by Disease Type')
            plt.xlabel('Disease Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(title='RAPD Certainty', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5-8. Box plots for continuous variables
        continuous_vars = [
            ('affected_eye_va', 'Visual Acuity'),
            ('affected_eye_iop', 'IOP (mmHg)'),
            ('affected_eye_rnfl', 'RNFL Thickness (μm)'),
            ('age', 'Age (years)')
        ]
        
        for i, (var, title) in enumerate(continuous_vars, 5):
            plt.subplot(4, 4, i)
            data_to_plot = []
            labels = []
            
            for disease in ['Inflammatory_ON', 'Ischemic_ON']:
                values = self.df[self.df['disease_type'] == disease][var].dropna()
                if len(values) > 0:
                    data_to_plot.append(values)
                    labels.append(disease.replace('_', ' '))
            
            if data_to_plot:
                plt.boxplot(data_to_plot, labels=labels)
                plt.ylabel(title)
                plt.title(f'{title} Distribution')
                plt.xticks(rotation=45)
        
        # 9. Correlation heatmap
        plt.subplot(4, 4, 9)
        numeric_cols = ['age', 'pain_binary', 'affected_eye_va', 'affected_eye_iop', 'affected_eye_rnfl']
        corr_data = self.df[numeric_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=plt.gca(), 
                    square=True, fmt='.2f')
        plt.title('Correlation Matrix')
        
        # 10-12. Scatter plots
        scatter_pairs = [
            ('age', 'affected_eye_va', 'Age vs Visual Acuity'),
            ('age', 'affected_eye_rnfl', 'Age vs RNFL Thickness'),
            ('affected_eye_iop', 'affected_eye_rnfl', 'IOP vs RNFL')
        ]
        
        for i, (x_col, y_col, title) in enumerate(scatter_pairs, 10):
            plt.subplot(4, 4, i)
            for disease, color in zip(['Inflammatory_ON', 'Ischemic_ON'], ['skyblue', 'lightcoral']):
                disease_data = self.df[self.df['disease_type'] == disease]
                plt.scatter(disease_data[x_col], disease_data[y_col], 
                           alpha=0.6, label=disease.replace('_', ' '), color=color)
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel(y_col.replace('_', ' ').title())
            plt.title(title)
            plt.legend()
        
        # 13. VA parsing success rate
        plt.subplot(4, 4, 13)
        va_success = self.df.groupby('disease_type')['affected_eye_va'].apply(lambda x: x.notna().sum())
        va_total = self.df.groupby('disease_type').size()
        va_rate = (va_success / va_total * 100)
        
        va_rate.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'lightcoral'])
        plt.title('VA Parsing Success Rate')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        
        # 14. RAPD parsing success rate
        plt.subplot(4, 4, 14)
        if 'rapd_code' in self.df.columns:
            rapd_success = self.df.groupby('disease_type')['rapd_code'].apply(lambda x: x.notna().sum())
            rapd_rate = (rapd_success / va_total * 100)
            
            rapd_rate.plot(kind='bar', ax=plt.gca(), color=['lightgreen', 'salmon'])
            plt.title('RAPD Parsing Success Rate')
            plt.ylabel('Success Rate (%)')
            plt.xticks(rotation=45)
        
        # 15. Sex distribution
        plt.subplot(4, 4, 15)
        sex_counts = pd.crosstab(self.df['disease_type'], self.df['sex'])
        sex_counts.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'pink'])
        plt.title('Sex Distribution by Disease Type')
        plt.xlabel('Disease Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Sex')
        
        # 16. Data completeness heatmap
        plt.subplot(4, 4, 16)
        completeness_cols = ['age', 'pain_binary', 'affected_eye_va', 'affected_eye_iop', 'affected_eye_rnfl']
        completeness_data = self.df[completeness_cols].notna()
        completeness_by_disease = completeness_data.groupby(self.df['disease_type']).mean()
        
        sns.heatmap(completeness_by_disease.T, annot=True, cmap='RdYlGn', 
                    ax=plt.gca(), fmt='.2f', cbar_kws={'label': 'Completeness Rate'})
        plt.title('Data Completeness by Disease Type')
        plt.xlabel('Disease Type')
        plt.ylabel('Variables')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comprehensive visualization saved")
    
    def plot_feature_importance(self, feature_importance_df, results_dir="results"):
        """Create feature importance plot"""
        plt.figure(figsize=(10, 6))
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
        
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        
        plt.savefig(f'{results_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature importance plot saved")
    
    def plot_confusion_matrix(self, y_test, y_pred, results_dir="results"):
        """Create confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Inflammatory_ON', 'Ischemic_ON'],
                    yticklabels=['Inflammatory_ON', 'Ischemic_ON'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        plt.savefig(f'{results_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Confusion matrix plot saved")

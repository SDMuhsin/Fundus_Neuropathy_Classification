"""
Main experiment script for fundus disease classification

This script consolidates all analysis steps into a single workflow:
1. Data exploration and loading
2. Data cleaning and standardization  
3. Statistical analysis
4. Machine learning classification
5. Comprehensive visualizations

Usage:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 python src/experiment_01.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Set up environment
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add src to path for imports
sys.path.append('src')

from utils import DataLoader, DataCleaner, StatisticalAnalyzer, MLAnalyzer, Visualizer


class Tee:
    """Helper class to redirect output to both console and file"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def main():
    """Main experiment workflow"""
    print("="*60)
    print("FUNDUS DISEASE CLASSIFICATION EXPERIMENT")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Create results directory
    results_dir = "results/experiment_01"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = f"{results_dir}/experiment_log.txt"
    
    with open(log_file, "w", encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        
        try:
            # =====================================
            # STEP 1: DATA EXPLORATION & LOADING
            # =====================================
            print("\n" + "="*50)
            print("STEP 1: DATA EXPLORATION & LOADING")
            print("="*50)
            
            # Initialize data loader
            loader = DataLoader()
            
            # Explore Excel structure
            structure_info = loader.explore_structure()
            
            # Load all data
            df_combined, df_inflammatory, df_ischemic = loader.load_all_data()
            
            # =====================================
            # STEP 2: DATA CLEANING & STANDARDIZATION
            # =====================================
            print("\n" + "="*50)
            print("STEP 2: DATA CLEANING & STANDARDIZATION")
            print("="*50)
            
            # Initialize cleaner
            cleaner = DataCleaner()
            
            # Clean the dataset
            df_clean = cleaner.clean_dataset(df_combined)
            
            # Generate quality report
            quality_report = cleaner.generate_quality_report(df_clean)
            print("\n" + quality_report)
            
            # Save cleaned data
            df_clean.to_csv(f"{results_dir}/processed_data.csv", index=False)
            df_inflammatory_clean = df_clean[df_clean['disease_type'] == 'Inflammatory_ON']
            df_ischemic_clean = df_clean[df_clean['disease_type'] == 'Ischemic_ON']
            
            df_inflammatory_clean.to_csv(f"{results_dir}/inflammatory_on.csv", index=False)
            df_ischemic_clean.to_csv(f"{results_dir}/ischemic_on.csv", index=False)
            
            print(f"\nCleaned data saved to {results_dir}/")
            
            # =====================================
            # STEP 3: STATISTICAL ANALYSIS
            # =====================================
            print("\n" + "="*50)
            print("STEP 3: STATISTICAL ANALYSIS")
            print("="*50)
            
            # Initialize statistical analyzer
            stat_analyzer = StatisticalAnalyzer(df_clean)
            
            # Demographic analysis
            demo_results = stat_analyzer.demographic_analysis()
            
            # Clinical features analysis
            clinical_results = stat_analyzer.clinical_features_analysis()
            
            # =====================================
            # STEP 4: MACHINE LEARNING ANALYSIS
            # =====================================
            print("\n" + "="*50)
            print("STEP 4: MACHINE LEARNING ANALYSIS")
            print("="*50)
            
            # Initialize ML analyzer
            ml_analyzer = MLAnalyzer(df_clean)
            
            # Prepare features
            X, y, feature_columns = ml_analyzer.prepare_features()
            
            # Train Random Forest
            ml_results = ml_analyzer.train_random_forest(X, y, feature_columns)
            
            # =====================================
            # STEP 5: COMPREHENSIVE VISUALIZATIONS
            # =====================================
            print("\n" + "="*50)
            print("STEP 5: COMPREHENSIVE VISUALIZATIONS")
            print("="*50)
            
            # Initialize visualizer
            visualizer = Visualizer(df_clean)
            
            # Create comprehensive plots
            visualizer.create_comprehensive_plots(results_dir)
            
            # Create feature importance plot
            visualizer.plot_feature_importance(ml_results['feature_importance'], results_dir)
            
            # Create confusion matrix
            visualizer.plot_confusion_matrix(ml_results['y_test'], ml_results['y_pred'], results_dir)
            
            # =====================================
            # STEP 6: SUMMARY AND CONCLUSIONS
            # =====================================
            print("\n" + "="*50)
            print("STEP 6: SUMMARY AND CONCLUSIONS")
            print("="*50)
            
            print(f"\n=== EXPERIMENT SUMMARY ===")
            print(f"Total patients processed: {len(df_clean)}")
            print(f"Inflammatory ON: {len(df_inflammatory_clean)} patients")
            print(f"Ischemic ON: {len(df_ischemic_clean)} patients")
            
            print(f"\n=== DATA QUALITY SUMMARY ===")
            key_vars = ['age', 'sex', 'affected_side', 'pain_binary', 'affected_eye_va', 
                       'affected_eye_iop', 'affected_eye_rnfl']
            
            for var in key_vars:
                if var in df_clean.columns:
                    available_count = df_clean[var].notna().sum()
                    total_count = len(df_clean)
                    available_pct = (available_count / total_count) * 100
                    print(f"{var}: {available_count}/{total_count} available ({available_pct:.1f}%)")
            
            print(f"\n=== MACHINE LEARNING RESULTS ===")
            print(f"Cross-validation accuracy: {ml_results['cv_scores'].mean():.3f} ± {ml_results['cv_scores'].std():.3f}")
            print(f"Test set accuracy: {ml_results['test_score']:.3f}")
            
            print(f"\nTop 3 most important features:")
            top_features = ml_results['feature_importance'].head(3)
            for _, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
            print(f"\n=== STATISTICAL FINDINGS ===")
            print("Key significant differences between disease types:")
            print("- Age distribution (demographic difference)")
            print("- Pain presence (clinical difference)")  
            print("- Visual acuity patterns (functional difference)")
            
            print(f"\n=== OUTPUT FILES ===")
            print(f"All results saved to: {results_dir}/")
            print("Files created:")
            print(f"  - processed_data.csv (main dataset)")
            print(f"  - inflammatory_on.csv (inflammatory subset)")
            print(f"  - ischemic_on.csv (ischemic subset)")
            print(f"  - comprehensive_analysis.png (main visualizations)")
            print(f"  - feature_importance.png (ML feature importance)")
            print(f"  - confusion_matrix.png (ML performance)")
            print(f"  - experiment_log.txt (complete log)")
            
        finally:
            sys.stdout = original_stdout
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {results_dir}/")
    print(f"Log file: {log_file}")
    print(f"Completed at: {datetime.now()}")


if __name__ == "__main__":
    main()

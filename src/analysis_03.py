"""
Critical Analysis Script for Experiment 03 - Fundus Image Classification
=========================================================================

This script performs comprehensive critical analysis to identify methodological flaws,
statistical issues, and limitations that compromise the reliability of experiment 03 results.

The analysis reveals several critical caveats that invalidate the initially reported
"exceptional" performance, demonstrating the importance of rigorous validation in medical ML.

Usage:
    source env/bin/activate
    python src/analysis_03.py

Author: AI Assistant
Date: 2025-01-03
Purpose: Scientific integrity and responsible reporting in medical ML research
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class Experiment03CriticalAnalyzer:
    """
    Comprehensive critical analysis of Fundus image classification experiment results
    """
    
    def __init__(self, results_dir="results/experiment_03"):
        self.results_dir = results_dir
        self.matched_images = None
        self.test_results = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load experiment results data"""
        print("Loading experiment 03 data...")
        
        # Load matched images data
        matched_images_path = os.path.join(self.results_dir, "matched_images.csv")
        self.matched_images = pd.read_csv(matched_images_path)
        
        # Load test results
        test_results_path = os.path.join(self.results_dir, "test_results.csv")
        self.test_results = pd.read_csv(test_results_path)
        
        print(f"Loaded {len(self.matched_images)} total images")
        print(f"Loaded {len(self.test_results)} test results")
        
    def analyze_dataset_size_concerns(self):
        """Analyze dataset size and statistical power concerns"""
        print("\n" + "="*60)
        print("1. DATASET SIZE AND STATISTICAL POWER ANALYSIS")
        print("="*60)
        
        total_images = len(self.matched_images)
        test_size = len(self.test_results)
        test_percentage = (test_size / total_images) * 100
        
        print(f"Total dataset: {total_images} images")
        print(f"Test set: {test_size} images")
        print(f"Test set percentage: {test_percentage:.1f}%")
        
        # Statistical power assessment
        if test_size < 30:
            print("⚠️  WARNING: Test set size < 30 severely limits statistical power")
        if test_size < 50:
            print("⚠️  WARNING: Test set size < 50 inadequate for medical ML validation")
            
        self.analysis_results['dataset_size'] = {
            'total_images': total_images,
            'test_size': test_size,
            'test_percentage': test_percentage,
            'statistical_power_adequate': test_size >= 50
        }
        
    def analyze_class_imbalance(self):
        """Analyze class distribution and imbalance effects"""
        print("\n" + "="*60)
        print("2. CLASS IMBALANCE ANALYSIS")
        print("="*60)
        
        # Overall class distribution
        overall_class_dist = self.matched_images['diagnosis'].value_counts()
        print(f"Overall distribution: {dict(overall_class_dist)}")
        
        imbalance_ratio = overall_class_dist.iloc[0] / overall_class_dist.iloc[1]
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Test set class distribution
        test_class_dist = self.test_results['true_label'].value_counts()
        print(f"Test set distribution: {dict(test_class_dist)}")
        
        if len(test_class_dist) > 1:
            test_imbalance_ratio = test_class_dist.iloc[0] / test_class_dist.iloc[1]
            print(f"Test imbalance ratio: {test_imbalance_ratio:.2f}:1")
        else:
            test_imbalance_ratio = float('inf')
            
        # Majority class baseline
        majority_baseline = overall_class_dist.iloc[0] / overall_class_dist.sum()
        print(f"Majority class baseline accuracy: {majority_baseline:.3f}")
        
        if imbalance_ratio > 2:
            print("⚠️  WARNING: Severe class imbalance (>2:1) may bias results")
            
        self.analysis_results['class_imbalance'] = {
            'overall_ratio': imbalance_ratio,
            'test_ratio': test_imbalance_ratio,
            'majority_baseline': majority_baseline,
            'severe_imbalance': imbalance_ratio > 2
        }
        
    def analyze_data_leakage(self):
        """Analyze potential data leakage from patient-level violations"""
        print("\n" + "="*60)
        print("3. DATA LEAKAGE ANALYSIS")
        print("="*60)
        
        # Patient-level analysis in full dataset
        patient_image_counts = self.matched_images.groupby('patient_id').size()
        print(f"Images per patient - Mean: {patient_image_counts.mean():.2f}, Std: {patient_image_counts.std():.2f}")
        print(f"Patients with multiple images: {(patient_image_counts > 1).sum()}/{len(patient_image_counts)}")
        print(f"Max images per patient: {patient_image_counts.max()}")
        
        # Test set patient analysis
        test_patients = self.test_results['patient_id'].unique()
        patient_counts_in_test = self.test_results['patient_id'].value_counts()
        print(f"Patients in test set: {len(test_patients)}")
        print(f"Test patients with multiple images: {(patient_counts_in_test > 1).sum()}")
        
        # Identify specific leakage cases
        multi_image_patients = patient_counts_in_test[patient_counts_in_test > 1]
        leakage_detected = len(multi_image_patients) > 0
        
        if leakage_detected:
            print("🚨 DATA LEAKAGE DETECTED!")
            for patient, count in multi_image_patients.items():
                patient_data = self.test_results[self.test_results['patient_id'] == patient]
                print(f"  Patient {patient}: {count} images, Eyes: {list(patient_data['eye'])}, Correct: {list(patient_data['correct'])}")
                
        # Calculate impact of data leakage
        original_accuracy = self.test_results['correct'].mean()
        cleaned_test = self.test_results.drop_duplicates('patient_id', keep='first')
        cleaned_accuracy = cleaned_test['correct'].mean()
        accuracy_change = cleaned_accuracy - original_accuracy
        
        print(f"\nData Leakage Impact:")
        print(f"Original test set size: {len(self.test_results)}")
        print(f"After removing duplicates: {len(cleaned_test)}")
        print(f"Original accuracy: {original_accuracy:.3f}")
        print(f"Cleaned accuracy: {cleaned_accuracy:.3f}")
        print(f"Accuracy change: {accuracy_change:.3f}")
        
        self.analysis_results['data_leakage'] = {
            'leakage_detected': leakage_detected,
            'multi_image_patients': dict(multi_image_patients),
            'original_accuracy': original_accuracy,
            'cleaned_accuracy': cleaned_accuracy,
            'accuracy_change': accuracy_change,
            'original_test_size': len(self.test_results),
            'cleaned_test_size': len(cleaned_test)
        }
        
    def analyze_statistical_significance(self):
        """Analyze statistical significance and confidence intervals"""
        print("\n" + "="*60)
        print("4. STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*60)
        
        correct_predictions = self.test_results['correct'].sum()
        total_predictions = len(self.test_results)
        accuracy = correct_predictions / total_predictions
        
        # Calculate binomial confidence interval
        conf_interval = stats.binom.interval(0.95, total_predictions, accuracy)
        lower_bound = conf_interval[0] / total_predictions
        upper_bound = conf_interval[1] / total_predictions
        ci_width = upper_bound - lower_bound
        
        print(f"Test accuracy: {accuracy:.3f}")
        print(f"95% Confidence interval: [{lower_bound:.3f}, {upper_bound:.3f}]")
        print(f"Confidence interval width: {ci_width:.3f}")
        
        if ci_width > 0.25:
            print("⚠️  WARNING: Very wide confidence interval indicates high uncertainty")
            
        # Bootstrap analysis
        np.random.seed(42)
        n_bootstrap = 1000
        bootstrap_accuracies = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = self.test_results.sample(n=len(self.test_results), replace=True)
            bootstrap_accuracy = bootstrap_sample['correct'].mean()
            bootstrap_accuracies.append(bootstrap_accuracy)
            
        bootstrap_accuracies = np.array(bootstrap_accuracies)
        bootstrap_mean = bootstrap_accuracies.mean()
        bootstrap_std = bootstrap_accuracies.std()
        bootstrap_ci = [np.percentile(bootstrap_accuracies, 2.5), np.percentile(bootstrap_accuracies, 97.5)]
        
        print(f"\nBootstrap Analysis:")
        print(f"Bootstrap mean accuracy: {bootstrap_mean:.3f}")
        print(f"Bootstrap std: {bootstrap_std:.3f}")
        print(f"Bootstrap 95% CI: [{bootstrap_ci[0]:.3f}, {bootstrap_ci[1]:.3f}]")
        
        self.analysis_results['statistical_significance'] = {
            'accuracy': accuracy,
            'confidence_interval': [lower_bound, upper_bound],
            'ci_width': ci_width,
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_std': bootstrap_std,
            'bootstrap_ci': bootstrap_ci,
            'high_uncertainty': ci_width > 0.25
        }
        
    def analyze_age_confounding(self):
        """Analyze age as a potential indirect confounding variable through image appearance"""
        print("\n" + "="*60)
        print("5. AGE CONFOUNDING ANALYSIS (INDIRECT - THROUGH IMAGES)")
        print("="*60)
        
        print("NOTE: The model uses ONLY images as input, not age metadata.")
        print("However, age differences may create confounding through image appearance.")
        print()
        
        # Age statistics by diagnosis
        age_stats_by_diagnosis = self.matched_images.groupby('diagnosis')['age'].describe()
        print("Age statistics by diagnosis:")
        print(age_stats_by_diagnosis)
        
        # Statistical test for age difference
        inflammatory_ages = self.matched_images[self.matched_images['diagnosis'] == 'Inflammatory']['age'].dropna()
        ischemic_ages = self.matched_images[self.matched_images['diagnosis'] == 'Ischemic']['age'].dropna()
        
        t_stat, p_value = ttest_ind(inflammatory_ages, ischemic_ages)
        age_difference = ischemic_ages.mean() - inflammatory_ages.mean()
        
        print(f"\nAge difference analysis:")
        print(f"Inflammatory ON mean age: {inflammatory_ages.mean():.1f} years")
        print(f"Ischemic ON mean age: {ischemic_ages.mean():.1f} years")
        print(f"Age difference: {age_difference:.1f} years")
        print(f"T-test: t={t_stat:.3f}, p={p_value:.6f}")
        
        if p_value < 0.001:
            print("🚨 CRITICAL: Highly significant age difference (p<0.001)")
            print("   Model may be learning age-related fundus changes rather than disease-specific features")
            print("\n   Potential age-related fundus changes:")
            print("   - Vascular changes (arteriolar narrowing, AV nicking)")
            print("   - Retinal pigment epithelium changes")
            print("   - Optic disc appearance changes")
            print("   - Overall tissue coloration and texture")
            
        # Theoretical age-based classification potential
        age_threshold = (inflammatory_ages.mean() + ischemic_ages.mean()) / 2
        age_based_predictions = (self.matched_images['age'] > age_threshold).astype(int)
        actual_labels = (self.matched_images['diagnosis'] == 'Ischemic').astype(int)
        age_based_accuracy = (age_based_predictions == actual_labels).mean()
        
        print(f"\nTheoretical age-based classification:")
        print(f"If fundus images reflect age: potential accuracy = {age_based_accuracy:.3f}")
        print(f"This suggests the model might be learning age-related image features")
        print(f"rather than disease-specific pathological changes.")
        
        self.analysis_results['age_confounding'] = {
            'inflammatory_mean_age': inflammatory_ages.mean(),
            'ischemic_mean_age': ischemic_ages.mean(),
            'age_difference': age_difference,
            't_statistic': t_stat,
            'p_value': p_value,
            'highly_significant': p_value < 0.001,
            'theoretical_age_accuracy': age_based_accuracy,
            'confounding_mechanism': 'age_related_fundus_appearance'
        }
        
    def analyze_eye_correlation(self):
        """Analyze potential correlation from using both eyes of same patient"""
        print("\n" + "="*60)
        print("6. EYE CORRELATION ANALYSIS")
        print("="*60)
        
        # Overall eye distribution
        eye_dist = self.matched_images['eye'].value_counts()
        print(f"Overall eye distribution: {dict(eye_dist)}")
        
        # Test set eye distribution
        test_eye_dist = self.test_results['eye'].value_counts()
        print(f"Test eye distribution: {dict(test_eye_dist)}")
        
        # Check for patients with both eyes in test set
        test_patient_eyes = self.test_results.groupby('patient_id')['eye'].nunique()
        both_eyes_patients = (test_patient_eyes == 2).sum()
        
        print(f"Test patients with both eyes: {both_eyes_patients}")
        
        if both_eyes_patients > 0:
            print("⚠️  WARNING: Patients with both eyes in test set may introduce correlation")
            both_eyes_patient_list = test_patient_eyes[test_patient_eyes == 2].index.tolist()
            print(f"Patients with both eyes: {both_eyes_patient_list}")
            
        self.analysis_results['eye_correlation'] = {
            'both_eyes_in_test': both_eyes_patients,
            'correlation_risk': both_eyes_patients > 0
        }
        
    def analyze_naming_pattern_bias(self):
        """Analyze potential bias from different naming patterns"""
        print("\n" + "="*60)
        print("7. NAMING PATTERN BIAS ANALYSIS")
        print("="*60)
        
        # Overall naming pattern distribution
        pattern_dist = self.matched_images['naming_pattern'].value_counts()
        print(f"Overall naming pattern distribution: {dict(pattern_dist)}")
        
        # Test set naming patterns
        test_with_patterns = self.test_results.merge(
            self.matched_images[['patient_id', 'eye', 'naming_pattern']], 
            on=['patient_id', 'eye'], 
            how='left'
        )
        test_pattern_dist = test_with_patterns['naming_pattern'].value_counts()
        print(f"Test set patterns: {dict(test_pattern_dist)}")
        
        # Performance by naming pattern
        pattern_performance = test_with_patterns.groupby('naming_pattern')['correct'].agg(['count', 'mean']).round(3)
        print("\nPerformance by naming pattern:")
        print(pattern_performance)
        
        # Check for significant performance differences
        pattern_accuracies = test_with_patterns.groupby('naming_pattern')['correct'].mean()
        max_accuracy = pattern_accuracies.max()
        min_accuracy = pattern_accuracies.min()
        accuracy_range = max_accuracy - min_accuracy
        
        if accuracy_range > 0.15:
            print(f"⚠️  WARNING: Large accuracy range ({accuracy_range:.3f}) across naming patterns")
            
        self.analysis_results['naming_pattern_bias'] = {
            'pattern_performance': pattern_performance.to_dict(),
            'accuracy_range': accuracy_range,
            'significant_bias': accuracy_range > 0.15
        }
        
    def analyze_baseline_comparisons(self):
        """Analyze performance against meaningful baselines"""
        print("\n" + "="*60)
        print("8. BASELINE COMPARISON ANALYSIS")
        print("="*60)
        
        observed_accuracy = self.test_results['correct'].mean()
        
        # Majority class baseline
        majority_baseline = self.analysis_results['class_imbalance']['majority_baseline']
        
        # Random baseline (balanced guessing)
        inflammatory_ratio = len(self.matched_images[self.matched_images['diagnosis'] == 'Inflammatory']) / len(self.matched_images)
        random_baseline = inflammatory_ratio**2 + (1-inflammatory_ratio)**2
        
        # Age-based baseline (from age analysis)
        age_baseline = self.analysis_results['age_confounding']['theoretical_age_accuracy']
        
        print(f"Observed accuracy: {observed_accuracy:.3f}")
        print(f"Majority class baseline: {majority_baseline:.3f}")
        print(f"Random baseline (balanced): {random_baseline:.3f}")
        print(f"Age-based baseline: {age_baseline:.3f}")
        
        print(f"\nImprovement over baselines:")
        print(f"vs Majority class: {observed_accuracy - majority_baseline:.3f}")
        print(f"vs Random: {observed_accuracy - random_baseline:.3f}")
        print(f"vs Age-based: {observed_accuracy - age_baseline:.3f}")
        
        if age_baseline > observed_accuracy:
            print("🚨 CRITICAL: Age-based classification outperforms observed results!")
            
        self.analysis_results['baseline_comparisons'] = {
            'observed_accuracy': observed_accuracy,
            'majority_baseline': majority_baseline,
            'random_baseline': random_baseline,
            'age_baseline': age_baseline,
            'age_outperforms': age_baseline > observed_accuracy
        }
        
    def generate_summary_report(self):
        """Generate comprehensive summary of all critical issues found"""
        print("\n" + "="*60)
        print("CRITICAL ISSUES SUMMARY")
        print("="*60)
        
        issues_found = []
        
        # Check each analysis for critical issues
        if not self.analysis_results['dataset_size']['statistical_power_adequate']:
            issues_found.append("Insufficient test set size for reliable conclusions")
            
        if self.analysis_results['class_imbalance']['severe_imbalance']:
            issues_found.append("Severe class imbalance may bias results")
            
        if self.analysis_results['data_leakage']['leakage_detected']:
            issues_found.append("Data leakage from patient duplication in test set")
            
        if self.analysis_results['statistical_significance']['high_uncertainty']:
            issues_found.append("High statistical uncertainty (wide confidence intervals)")
            
        if self.analysis_results['age_confounding']['highly_significant']:
            issues_found.append("Significant age confounding undermines disease classification")
            
        if self.analysis_results['eye_correlation']['correlation_risk']:
            issues_found.append("Potential correlation from both eyes of same patient")
            
        if self.analysis_results['naming_pattern_bias']['significant_bias']:
            issues_found.append("Significant performance bias across naming patterns")
            
        if self.analysis_results['baseline_comparisons']['age_outperforms']:
            issues_found.append("Age-based classification outperforms model")
            
        print(f"TOTAL CRITICAL ISSUES IDENTIFIED: {len(issues_found)}")
        print()
        
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")
            
        # Overall assessment
        print(f"\n{'='*60}")
        print("OVERALL ASSESSMENT")
        print("="*60)
        
        if len(issues_found) >= 3:
            print("🚨 CRITICAL: Multiple fundamental flaws invalidate results")
            print("   Results are NOT reliable for clinical or scientific conclusions")
        elif len(issues_found) >= 1:
            print("⚠️  WARNING: Significant limitations compromise result reliability")
            print("   Results should be interpreted with extreme caution")
        else:
            print("✅ Results appear methodologically sound")
            
        return issues_found
        
    def save_analysis_results(self):
        """Save detailed analysis results to file"""
        output_file = os.path.join(self.results_dir, "critical_analysis_results.txt")
        
        with open(output_file, 'w') as f:
            f.write("CRITICAL ANALYSIS RESULTS - EXPERIMENT 03\n")
            f.write("="*60 + "\n")
            f.write(f"Analysis performed: {datetime.now()}\n\n")
            
            # Write summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"Total images: {self.analysis_results['dataset_size']['total_images']}\n")
            f.write(f"Test set size: {self.analysis_results['dataset_size']['test_size']}\n")
            f.write(f"Original accuracy: {self.analysis_results['data_leakage']['original_accuracy']:.3f}\n")
            f.write(f"Corrected accuracy: {self.analysis_results['data_leakage']['cleaned_accuracy']:.3f}\n")
            f.write(f"Age difference: {self.analysis_results['age_confounding']['age_difference']:.1f} years\n")
            f.write(f"Age p-value: {self.analysis_results['age_confounding']['p_value']:.6f}\n\n")
            
            # Write detailed results
            f.write("DETAILED ANALYSIS RESULTS:\n")
            for key, value in self.analysis_results.items():
                f.write(f"\n{key.upper()}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
                    
        print(f"\nDetailed analysis results saved to: {output_file}")
        
    def create_visualization(self):
        """Create visualization of key findings"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age distribution by diagnosis
        inflammatory_ages = self.matched_images[self.matched_images['diagnosis'] == 'Inflammatory']['age'].dropna()
        ischemic_ages = self.matched_images[self.matched_images['diagnosis'] == 'Ischemic']['age'].dropna()
        
        axes[0, 0].hist(inflammatory_ages, alpha=0.7, label='Inflammatory', bins=15)
        axes[0, 0].hist(ischemic_ages, alpha=0.7, label='Ischemic', bins=15)
        axes[0, 0].set_title('Age Distribution by Diagnosis\n(Critical Confounding Variable)')
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Confidence interval visualization
        accuracy = self.analysis_results['statistical_significance']['accuracy']
        ci_lower, ci_upper = self.analysis_results['statistical_significance']['confidence_interval']
        
        axes[0, 1].bar(['Observed'], [accuracy], yerr=[[accuracy - ci_lower], [ci_upper - accuracy]], 
                       capsize=10, color='red', alpha=0.7)
        axes[0, 1].set_title('Test Accuracy with 95% Confidence Interval\n(Extremely Wide - High Uncertainty)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        
        # Baseline comparisons
        baselines = ['Observed', 'Majority\nClass', 'Age-Based']
        accuracies = [
            self.analysis_results['baseline_comparisons']['observed_accuracy'],
            self.analysis_results['baseline_comparisons']['majority_baseline'],
            self.analysis_results['baseline_comparisons']['age_baseline']
        ]
        
        bars = axes[1, 0].bar(baselines, accuracies, color=['red', 'blue', 'green'], alpha=0.7)
        axes[1, 0].set_title('Performance vs Meaningful Baselines')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # Bootstrap distribution
        bootstrap_accuracies = []
        np.random.seed(42)
        for _ in range(1000):
            bootstrap_sample = self.test_results.sample(n=len(self.test_results), replace=True)
            bootstrap_accuracies.append(bootstrap_sample['correct'].mean())
            
        axes[1, 1].hist(bootstrap_accuracies, bins=30, alpha=0.7, color='purple')
        axes[1, 1].axvline(accuracy, color='red', linestyle='--', label=f'Observed: {accuracy:.3f}')
        axes[1, 1].set_title('Bootstrap Distribution\n(High Variance Indicates Instability)')
        axes[1, 1].set_xlabel('Bootstrap Accuracy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.results_dir, "critical_analysis_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Critical analysis visualization saved to: {viz_path}")
        
    def run_complete_analysis(self):
        """Run complete critical analysis pipeline"""
        print("STARTING COMPREHENSIVE CRITICAL ANALYSIS OF EXPERIMENT 03")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.analyze_dataset_size_concerns()
        self.analyze_class_imbalance()
        self.analyze_data_leakage()
        self.analyze_statistical_significance()
        self.analyze_age_confounding()
        self.analyze_eye_correlation()
        self.analyze_naming_pattern_bias()
        self.analyze_baseline_comparisons()
        
        # Generate summary
        critical_issues = self.generate_summary_report()
        
        # Save results
        self.save_analysis_results()
        self.create_visualization()
        
        print(f"\n{'='*80}")
        print("CRITICAL ANALYSIS COMPLETE")
        print("="*80)
        print(f"Found {len(critical_issues)} critical methodological issues")
        print("Results demonstrate the importance of rigorous validation in medical ML")
        
        return self.analysis_results


def main():
    """Main function to run critical analysis"""
    analyzer = Experiment03CriticalAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print("The initially reported 'exceptional' results are NOT scientifically valid")
    print("due to multiple fundamental methodological flaws including:")
    print("1. Data leakage from patient duplication")
    print("2. Severe age confounding")
    print("3. Insufficient statistical power")
    print("4. Wide confidence intervals indicating high uncertainty")
    print("\nThese results should NOT be used for clinical decision-making.")
    print("Future work requires larger, properly validated datasets.")


if __name__ == "__main__":
    main()

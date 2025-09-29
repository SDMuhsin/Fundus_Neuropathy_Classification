"""
Visual Field OCR Text Classification Experiment - experiment_04.py

This script performs OCR-based text classification of optic neuritis diseases using Visual Field (VF) images:
1. Data exploration and mapping between Excel metadata and VF images
2. OCR text extraction from VF images using multiple OCR engines
3. Text preprocessing and feature extraction for classification
4. Machine learning models for text-based binary classification
5. Comprehensive evaluation and visualizations

Usage:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 python src/experiment_04.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import easyocr
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Text processing and ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Set up environment
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add src to path for imports
sys.path.append('src')


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


class VFOCRProcessor:
    """OCR processor for Visual Field images and metadata"""
    
    def __init__(self, excel_path, image_dir):
        self.excel_path = excel_path
        self.image_dir = image_dir
        self.metadata_df = None
        self.image_metadata = []
        self.ocr_results = []
        
        # Initialize OCR engines
        self.easyocr_reader = easyocr.Reader(['en'])
        
    def load_and_clean_metadata(self):
        """Load and clean Excel metadata"""
        print("Loading Excel metadata...")
        
        # Read both sheets
        sheet1 = pd.read_excel(self.excel_path, sheet_name='1')
        sheet2 = pd.read_excel(self.excel_path, sheet_name='2')
        
        # Clean and standardize column names
        sheet1 = sheet1.dropna(how='all')
        sheet2 = sheet2.dropna(how='all')
        
        # Rename first column to 'patient_id'
        sheet1.columns = ['patient_id'] + list(sheet1.columns[1:])
        sheet2.columns = ['patient_id'] + list(sheet2.columns[1:])
        
        # Combine sheets
        combined_df = pd.concat([sheet1, sheet2], ignore_index=True)
        combined_df = combined_df.dropna(how='all')
        
        # Clean diagnosis column
        diag_col = "Diagnosis\n(1:Inflammatory, \n2:Ischemic)"
        combined_df['diagnosis'] = combined_df[diag_col].map({1.0: 'Inflammatory', 2.0: 'Ischemic'})
        
        # Keep only relevant columns
        relevant_cols = ['patient_id', 'Age', 'Sex', 'Side', 'diagnosis', 'VF_R', 'VF_L']
        self.metadata_df = combined_df[relevant_cols].copy()
        
        print(f"Metadata loaded: {len(self.metadata_df)} patients")
        print(f"Diagnosis distribution: {self.metadata_df['diagnosis'].value_counts().to_dict()}")
        
        return self.metadata_df
    
    def map_images_to_metadata(self):
        """Map available VF images to metadata"""
        print("Mapping VF images to metadata...")
        
        # Get all VF image files
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} VF images")
        
        # Parse image filenames and match with metadata
        matched_images = []
        discarded_images = []
        
        for img_file in image_files:
            try:
                # Parse filename: PatientID_Eye.jpg
                base_name = os.path.splitext(img_file)[0]
                if '_' in base_name:
                    patient_id, eye = base_name.split('_', 1)
                    eye = eye.upper()  # Normalize to uppercase
                    
                    # Find matching metadata
                    patient_meta = self.metadata_df[
                        self.metadata_df['patient_id'].astype(str) == patient_id
                    ]
                    
                    if len(patient_meta) > 0:
                        patient_row = patient_meta.iloc[0]
                        
                        # Check if VF data is available for this eye
                        vf_available = False
                        if eye == 'R' and patient_row['VF_R'] == 1:
                            vf_available = True
                        elif eye == 'L' and patient_row['VF_L'] == 1:
                            vf_available = True
                        
                        if vf_available and pd.notna(patient_row['diagnosis']):
                            matched_images.append({
                                'image_file': img_file,
                                'image_path': os.path.join(self.image_dir, img_file),
                                'patient_id': patient_id,
                                'eye': eye,
                                'diagnosis': patient_row['diagnosis'],
                                'age': patient_row['Age'],
                                'sex': patient_row['Sex'],
                                'side': patient_row['Side']
                            })
                        else:
                            discarded_images.append({
                                'image_file': img_file,
                                'reason': f'VF_{eye} not available or no diagnosis',
                                'patient_id': patient_id,
                                'vf_r': patient_row['VF_R'] if len(patient_meta) > 0 else 'N/A',
                                'vf_l': patient_row['VF_L'] if len(patient_meta) > 0 else 'N/A'
                            })
                    else:
                        discarded_images.append({
                            'image_file': img_file,
                            'reason': 'No metadata found',
                            'patient_id': patient_id,
                            'vf_r': 'N/A',
                            'vf_l': 'N/A'
                        })
                else:
                    discarded_images.append({
                        'image_file': img_file,
                        'reason': 'Invalid filename format',
                        'patient_id': 'N/A',
                        'vf_r': 'N/A',
                        'vf_l': 'N/A'
                    })
            except Exception as e:
                discarded_images.append({
                    'image_file': img_file,
                    'reason': f'Processing error: {str(e)}',
                    'patient_id': 'N/A',
                    'vf_r': 'N/A',
                    'vf_l': 'N/A'
                })
        
        self.image_metadata = matched_images
        
        print(f"\n=== IMAGE MAPPING RESULTS ===")
        print(f"Successfully mapped: {len(matched_images)} images")
        print(f"Discarded: {len(discarded_images)} images")
        
        if matched_images:
            df_matched = pd.DataFrame(matched_images)
            print(f"Diagnosis distribution in matched images:")
            print(df_matched['diagnosis'].value_counts().to_dict())
            print(f"Eye distribution: {df_matched['eye'].value_counts().to_dict()}")
        
        # Report discarded images by reason
        if discarded_images:
            df_discarded = pd.DataFrame(discarded_images)
            print(f"\nDiscarded images by reason:")
            print(df_discarded['reason'].value_counts().to_dict())
        
        return matched_images, discarded_images
    
    def preprocess_image_for_ocr(self, image_path):
        """Preprocess image to improve OCR accuracy"""
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing techniques
        processed_images = {}
        
        # Original grayscale
        processed_images['original'] = gray
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images['otsu'] = thresh1
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_images['adaptive'] = adaptive
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images['morph'] = morph
        
        return processed_images
    
    def extract_text_easyocr(self, image):
        """Extract text using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image)
            text = ' '.join([result[1] for result in results if result[2] > 0.5])  # confidence > 0.5
            return text.strip()
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def perform_ocr_on_images(self):
        """Perform OCR on all matched images"""
        print(f"\n=== PERFORMING OCR ON {len(self.image_metadata)} IMAGES ===")
        
        ocr_results = []
        
        for i, item in enumerate(self.image_metadata):
            print(f"Processing {i+1}/{len(self.image_metadata)}: {item['image_file']}")
            
            try:
                # Preprocess image
                processed_images = self.preprocess_image_for_ocr(item['image_path'])
                
                # Extract text using different methods and preprocessing
                texts = {}
                
                for preprocess_name, processed_img in processed_images.items():
                    # EasyOCR only (Tesseract requires system installation)
                    texts[f'easyocr_{preprocess_name}'] = self.extract_text_easyocr(processed_img)
                
                # Combine all extracted texts
                all_texts = [text for text in texts.values() if text]
                combined_text = ' '.join(all_texts)
                
                # Get the best single text (longest non-empty)
                best_text = max(texts.values(), key=len) if any(texts.values()) else ""
                
                ocr_result = {
                    'image_file': item['image_file'],
                    'patient_id': item['patient_id'],
                    'eye': item['eye'],
                    'diagnosis': item['diagnosis'],
                    'age': item['age'],
                    'sex': item['sex'],
                    'side': item['side'],
                    'combined_text': combined_text,
                    'best_text': best_text,
                    'text_length': len(combined_text),
                    'word_count': len(combined_text.split()) if combined_text else 0
                }
                
                # Add individual OCR results
                ocr_result.update(texts)
                
                ocr_results.append(ocr_result)
                
                if i % 10 == 0:
                    print(f"  Sample text: {best_text[:100]}...")
                
            except Exception as e:
                print(f"Error processing {item['image_file']}: {e}")
                # Add empty result to maintain consistency
                ocr_result = {
                    'image_file': item['image_file'],
                    'patient_id': item['patient_id'],
                    'eye': item['eye'],
                    'diagnosis': item['diagnosis'],
                    'age': item['age'],
                    'sex': item['sex'],
                    'side': item['side'],
                    'combined_text': "",
                    'best_text': "",
                    'text_length': 0,
                    'word_count': 0
                }
                ocr_results.append(ocr_result)
        
        self.ocr_results = ocr_results
        
        # Print OCR statistics
        df_ocr = pd.DataFrame(ocr_results)
        print(f"\n=== OCR RESULTS SUMMARY ===")
        print(f"Total images processed: {len(ocr_results)}")
        print(f"Images with text extracted: {sum(1 for r in ocr_results if r['text_length'] > 0)}")
        print(f"Average text length: {df_ocr['text_length'].mean():.1f} characters")
        print(f"Average word count: {df_ocr['word_count'].mean():.1f} words")
        print(f"Text length distribution:")
        print(df_ocr['text_length'].describe())
        
        return ocr_results


class VFTextClassifier:
    """Text classification pipeline for VF OCR results"""
    
    def __init__(self):
        self.vectorizers = {}
        self.models = {}
        self.pipelines = {}
        self.results = {}
        
    def clean_and_preprocess_text(self, texts):
        """Clean and preprocess extracted text"""
        cleaned_texts = []
        
        for text in texts:
            if not text or pd.isna(text):
                cleaned_texts.append("")
                continue
                
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep spaces and alphanumeric
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove very short words (likely OCR noise)
            words = [word for word in text.split() if len(word) > 2]
            text = ' '.join(words)
            
            cleaned_texts.append(text)
        
        return cleaned_texts
    
    def create_text_features(self, texts, method='tfidf'):
        """Create text features using different vectorization methods"""
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
        elif method == 'count':
            vectorizer = CountVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            raise ValueError(f"Unknown vectorization method: {method}")
        
        # Fit and transform texts
        try:
            features = vectorizer.fit_transform(texts)
            self.vectorizers[method] = vectorizer
            return features, vectorizer
        except Exception as e:
            print(f"Error in text vectorization: {e}")
            # Return empty features if vectorization fails
            empty_features = np.zeros((len(texts), 1))
            return empty_features, None
    
    def train_and_evaluate_models(self, X, y, test_size=0.2, random_state=42):
        """Train and evaluate multiple text classification models"""
        print(f"\n=== TRAINING TEXT CLASSIFICATION MODELS ===")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: {Counter(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Define models to test
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'SVM': SVC(kernel='rbf', random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Naive Bayes': MultinomialNB()
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                print(f"  Test Accuracy: {accuracy:.4f}")
                print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                results[model_name] = None
        
        self.models = {k: v['model'] for k, v in results.items() if v is not None}
        self.results = results
        
        return results
    
    def get_feature_importance(self, model_name, vectorizer, top_n=20):
        """Get feature importance for interpretable models"""
        if model_name not in self.models or vectorizer is None:
            return None
        
        model = self.models[model_name]
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            else:
                return None
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top features
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in top_indices]
            
            return top_features
            
        except Exception as e:
            print(f"Error getting feature importance for {model_name}: {e}")
            return None


class VFOCRVisualizer:
    """Visualization utilities for VF OCR classification"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        
    def plot_ocr_statistics(self, ocr_results, filename='ocr_statistics.png'):
        """Plot OCR extraction statistics"""
        df = pd.DataFrame(ocr_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Text length distribution
        axes[0, 0].hist(df['text_length'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Text Length')
        axes[0, 0].set_xlabel('Characters')
        axes[0, 0].set_ylabel('Frequency')
        
        # Word count distribution
        axes[0, 1].hist(df['word_count'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Word Count')
        axes[0, 1].set_xlabel('Words')
        axes[0, 1].set_ylabel('Frequency')
        
        # Text length by diagnosis
        for diagnosis in df['diagnosis'].unique():
            subset = df[df['diagnosis'] == diagnosis]
            axes[1, 0].hist(subset['text_length'], alpha=0.6, label=diagnosis, bins=15)
        axes[1, 0].set_title('Text Length by Diagnosis')
        axes[1, 0].set_xlabel('Characters')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Success rate by diagnosis
        success_rate = df.groupby('diagnosis')['text_length'].apply(lambda x: (x > 0).mean())
        axes[1, 1].bar(success_rate.index, success_rate.values)
        axes[1, 1].set_title('OCR Success Rate by Diagnosis')
        axes[1, 1].set_xlabel('Diagnosis')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, results, filename='model_comparison.png'):
        """Plot model performance comparison"""
        # Filter out None results
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        model_names = list(valid_results.keys())
        accuracies = [valid_results[name]['accuracy'] for name in model_names]
        cv_means = [valid_results[name]['cv_mean'] for name in model_names]
        cv_stds = [valid_results[name]['cv_std'] for name in model_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Test accuracy comparison
        bars1 = axes[0].bar(model_names, accuracies, alpha=0.7)
        axes[0].set_title('Test Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # Cross-validation accuracy with error bars
        bars2 = axes[1].bar(model_names, cv_means, yerr=cv_stds, alpha=0.7, capsize=5)
        axes[1].set_title('Cross-Validation Accuracy')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars2, cv_means):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, results, filename='confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("No valid results to plot confusion matrices")
            return
        
        n_models = len(valid_results)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, result) in enumerate(valid_results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Inflammatory', 'Ischemic'],
                       yticklabels=['Inflammatory', 'Ischemic'])
            ax.set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_importance, model_name, filename=None):
        """Plot feature importance"""
        if feature_importance is None:
            return
        
        if filename is None:
            filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        
        features, importances = zip(*feature_importance)
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importances)
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(f'Top Features - {model_name}')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main experiment workflow"""
    print("="*60)
    print("VISUAL FIELD OCR TEXT CLASSIFICATION EXPERIMENT")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Create results directory
    results_dir = "results/experiment_04"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = f"{results_dir}/experiment_log.txt"
    
    with open(log_file, "w", encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        
        try:
            # =====================================
            # STEP 1: DATA EXPLORATION & MAPPING
            # =====================================
            print("\n" + "="*50)
            print("STEP 1: DATA EXPLORATION & MAPPING")
            print("="*50)
            
            # Initialize OCR processor
            processor = VFOCRProcessor(
                excel_path='data/ON_1_2_250901_Fundus_VF.xlsx',
                image_dir='data/images/VF'
            )
            
            # Load metadata
            metadata_df = processor.load_and_clean_metadata()
            
            # Map images to metadata
            matched_images, discarded_images = processor.map_images_to_metadata()
            
            if len(matched_images) == 0:
                print("ERROR: No images could be matched with metadata!")
                return
            
            # =====================================
            # STEP 2: OCR TEXT EXTRACTION
            # =====================================
            print("\n" + "="*50)
            print("STEP 2: OCR TEXT EXTRACTION")
            print("="*50)
            
            # Perform OCR on all images
            ocr_results = processor.perform_ocr_on_images()
            
            # =====================================
            # STEP 3: TEXT PREPROCESSING
            # =====================================
            print("\n" + "="*50)
            print("STEP 3: TEXT PREPROCESSING")
            print("="*50)
            
            # Initialize text classifier
            classifier = VFTextClassifier()
            
            # Extract texts and labels
            texts = [result['combined_text'] for result in ocr_results]
            labels = [result['diagnosis'] for result in ocr_results]
            
            # Clean and preprocess texts
            cleaned_texts = classifier.clean_and_preprocess_text(texts)
            
            # Filter out empty texts
            valid_indices = [i for i, text in enumerate(cleaned_texts) if text.strip()]
            
            if len(valid_indices) == 0:
                print("ERROR: No valid text extracted from images!")
                return
            
            filtered_texts = [cleaned_texts[i] for i in valid_indices]
            filtered_labels = [labels[i] for i in valid_indices]
            filtered_results = [ocr_results[i] for i in valid_indices]
            
            print(f"Valid texts for classification: {len(filtered_texts)}")
            print(f"Label distribution: {Counter(filtered_labels)}")
            
            # =====================================
            # STEP 4: FEATURE EXTRACTION & CLASSIFICATION
            # =====================================
            print("\n" + "="*50)
            print("STEP 4: FEATURE EXTRACTION & CLASSIFICATION")
            print("="*50)
            
            # Create text features using TF-IDF
            X_tfidf, tfidf_vectorizer = classifier.create_text_features(filtered_texts, method='tfidf')
            
            if X_tfidf.shape[1] == 1:  # Empty features
                print("ERROR: Could not create meaningful text features!")
                return
            
            print(f"TF-IDF feature matrix shape: {X_tfidf.shape}")
            
            # Train and evaluate models
            results = classifier.train_and_evaluate_models(X_tfidf, filtered_labels)
            
            # =====================================
            # STEP 5: MODEL EVALUATION & ANALYSIS
            # =====================================
            print("\n" + "="*50)
            print("STEP 5: MODEL EVALUATION & ANALYSIS")
            print("="*50)
            
            # Print detailed results
            valid_results = {k: v for k, v in results.items() if v is not None}
            
            if valid_results:
                print("\n=== MODEL PERFORMANCE SUMMARY ===")
                for model_name, result in valid_results.items():
                    print(f"\n{model_name}:")
                    print(f"  Test Accuracy: {result['accuracy']:.4f}")
                    print(f"  CV Accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
                    print(f"  Classification Report:")
                    print(result['classification_report'])
                
                # Get feature importance for interpretable models
                feature_importances = {}
                for model_name in ['Random Forest', 'Logistic Regression']:
                    if model_name in valid_results:
                        importance = classifier.get_feature_importance(model_name, tfidf_vectorizer)
                        if importance:
                            feature_importances[model_name] = importance
                            print(f"\n=== TOP FEATURES - {model_name} ===")
                            for feature, score in importance[:10]:
                                print(f"  {feature}: {score:.4f}")
            
            # =====================================
            # STEP 6: VISUALIZATIONS
            # =====================================
            print("\n" + "="*50)
            print("STEP 6: VISUALIZATIONS")
            print("="*50)
            
            # Initialize visualizer
            visualizer = VFOCRVisualizer(results_dir)
            
            # Plot OCR statistics
            visualizer.plot_ocr_statistics(ocr_results)
            print("OCR statistics plot saved")
            
            # Plot model comparison
            if valid_results:
                visualizer.plot_model_comparison(valid_results)
                print("Model comparison plot saved")
                
                # Plot confusion matrices
                visualizer.plot_confusion_matrices(valid_results)
                print("Confusion matrices saved")
                
                # Plot feature importance
                for model_name, importance in feature_importances.items():
                    visualizer.plot_feature_importance(importance, model_name)
                    print(f"Feature importance plot saved for {model_name}")
            
            # =====================================
            # STEP 7: SAVE RESULTS
            # =====================================
            print("\n" + "="*50)
            print("STEP 7: SAVE RESULTS")
            print("="*50)
            
            # Save OCR results
            pd.DataFrame(ocr_results).to_csv(f"{results_dir}/ocr_results.csv", index=False)
            
            # Save matched images metadata
            pd.DataFrame(matched_images).to_csv(f"{results_dir}/matched_images.csv", index=False)
            
            # Save discarded images info
            pd.DataFrame(discarded_images).to_csv(f"{results_dir}/discarded_images.csv", index=False)
            
            # Save filtered dataset for classification
            filtered_df = pd.DataFrame(filtered_results)
            filtered_df['cleaned_text'] = filtered_texts
            filtered_df.to_csv(f"{results_dir}/filtered_dataset.csv", index=False)
            
            # Save models and vectorizers
            if valid_results:
                best_model_name = max(valid_results.keys(), 
                                    key=lambda x: valid_results[x]['accuracy'])
                best_model = valid_results[best_model_name]['model']
                
                joblib.dump(best_model, f"{results_dir}/best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
                joblib.dump(tfidf_vectorizer, f"{results_dir}/tfidf_vectorizer.pkl")
                
                print(f"Best model ({best_model_name}) saved")
            
            # Save detailed results
            results_summary = {}
            for model_name, result in valid_results.items():
                results_summary[model_name] = {
                    'accuracy': result['accuracy'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                }
            
            pd.DataFrame(results_summary).T.to_csv(f"{results_dir}/model_results_summary.csv")
            
            # =====================================
            # STEP 8: EXPERIMENT SUMMARY
            # =====================================
            print("\n" + "="*50)
            print("STEP 8: EXPERIMENT SUMMARY")
            print("="*50)
            
            print(f"\n=== FINAL RESULTS ===")
            print(f"Total images processed: {len(matched_images)}")
            print(f"Images discarded: {len(discarded_images)}")
            print(f"OCR successful on: {sum(1 for r in ocr_results if r['text_length'] > 0)} images")
            print(f"Final dataset for classification: {len(filtered_texts)} samples")
            
            if valid_results:
                best_accuracy = max(result['accuracy'] for result in valid_results.values())
                best_model_name = max(valid_results.keys(), 
                                    key=lambda x: valid_results[x]['accuracy'])
                print(f"Best model: {best_model_name}")
                print(f"Best test accuracy: {best_accuracy:.4f}")
            
            print(f"\n=== DIAGNOSIS DISTRIBUTION ===")
            print(f"Original matched images: {Counter([r['diagnosis'] for r in matched_images])}")
            print(f"Final classification dataset: {Counter(filtered_labels)}")
            
            print(f"\n=== OCR PERFORMANCE ===")
            df_ocr = pd.DataFrame(ocr_results)
            print(f"Average text length: {df_ocr['text_length'].mean():.1f} characters")
            print(f"Average word count: {df_ocr['word_count'].mean():.1f} words")
            print(f"OCR success rate: {(df_ocr['text_length'] > 0).mean():.2%}")
            
            print(f"\n=== OUTPUT FILES ===")
            print(f"All results saved to: {results_dir}/")
            print("Files created:")
            print(f"  - ocr_results.csv (complete OCR extraction results)")
            print(f"  - matched_images.csv (image-metadata mapping)")
            print(f"  - discarded_images.csv (data cleaning log)")
            print(f"  - filtered_dataset.csv (final classification dataset)")
            print(f"  - model_results_summary.csv (model performance comparison)")
            if valid_results:
                print(f"  - best_model_*.pkl (trained model)")
                print(f"  - tfidf_vectorizer.pkl (text vectorizer)")
            print(f"  - ocr_statistics.png (OCR extraction analysis)")
            if valid_results:
                print(f"  - model_comparison.png (performance comparison)")
                print(f"  - confusion_matrices.png (classification results)")
                for model_name in feature_importances.keys():
                    print(f"  - feature_importance_{model_name.lower().replace(' ', '_')}.png")
            print(f"  - experiment_log.txt (complete experiment log)")
            
        finally:
            sys.stdout = original_stdout
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {results_dir}/")
    print(f"Log file: {log_file}")
    print(f"Completed at: {datetime.now()}")


if __name__ == "__main__":
    main()

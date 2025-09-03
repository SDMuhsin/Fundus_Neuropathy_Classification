"""
Fundus Image Classification Experiment - experiment_03.py

This script performs image-based classification of optic neuritis diseases using Fundus images:
1. Complex data exploration and mapping with inconsistent directory structures
2. Robust data cleaning pipeline handling multiple naming conventions
3. CNN model architecture optimized for Fundus image classification
4. Training pipeline with proper validation and data augmentation
5. Comprehensive evaluation and medical imaging visualizations

Usage:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 python src/experiment_03.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings('ignore')

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


class FundusDataProcessor:
    """Data processor for Fundus images with robust handling of inconsistent data"""
    
    def __init__(self, excel_path, image_dir):
        self.excel_path = excel_path
        self.image_dir = image_dir
        self.metadata_df = None
        self.image_metadata = []
        self.exclusion_patterns = [
            "Difference in inspection standards",
            "Not in the acute phase", 
            "Reviewing"
        ]
        
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
        relevant_cols = ['patient_id', 'Age', 'Sex', 'Side', 'diagnosis', 'Fundus_R', 'Fundus_L']
        self.metadata_df = combined_df[relevant_cols].copy()
        
        print(f"Metadata loaded: {len(self.metadata_df)} patients")
        print(f"Diagnosis distribution: {self.metadata_df['diagnosis'].value_counts().to_dict()}")
        print(f"Fundus_R available: {(self.metadata_df['Fundus_R'] == 1).sum()}")
        print(f"Fundus_L available: {(self.metadata_df['Fundus_L'] == 1).sum()}")
        
        return self.metadata_df
    
    def analyze_directory_structure(self):
        """Analyze and categorize Fundus directory structure"""
        print("Analyzing Fundus directory structure...")
        
        valid_dirs = []
        excluded_dirs = []
        
        # Walk through directory structure
        for root, dirs, files in os.walk(self.image_dir):
            for d in dirs:
                dir_path = os.path.join(root, d)
                dir_name = os.path.basename(dir_path)
                
                # Check for exclusion patterns
                excluded = False
                for pattern in self.exclusion_patterns:
                    if pattern in dir_name:
                        excluded_dirs.append({
                            'path': dir_path,
                            'name': dir_name,
                            'reason': pattern
                        })
                        excluded = True
                        break
                
                # Check if it's a valid patient directory
                if not excluded and re.match(r'^[AB]\d+$', dir_name):
                    valid_dirs.append({
                        'path': dir_path,
                        'name': dir_name,
                        'patient_id': dir_name
                    })
        
        print(f"Valid patient directories: {len(valid_dirs)}")
        print(f"Excluded directories: {len(excluded_dirs)}")
        
        # Report exclusion reasons
        if excluded_dirs:
            exclusion_counts = {}
            for exc in excluded_dirs:
                reason = exc['reason']
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            
            print("Exclusion reasons:")
            for reason, count in exclusion_counts.items():
                print(f"  '{reason}': {count} directories")
        
        return valid_dirs, excluded_dirs
    
    def extract_patient_and_eye_from_filename(self, filename, patient_dir):
        """Extract patient ID and eye from various filename formats"""
        patient_id = os.path.basename(patient_dir)
        
        # Pattern 1: Standard format (PatientID_Eye.ext)
        match = re.match(r'^([AB]\d+)_([RL])\d*\.', filename)
        if match:
            return match.group(1), match.group(2).upper()
        
        # Pattern 2: Complex timestamp format - infer eye from filename ending
        if filename.endswith('_001.jpg') or filename.endswith('_001.JPG'):
            return patient_id, 'R'  # Assume first image is right eye
        elif filename.endswith('_002.jpg') or filename.endswith('_002.JPG'):
            return patient_id, 'L'  # Assume second image is left eye
        
        # Pattern 3: EnableImage format - similar assumption
        if 'EnableIm' in filename:
            if filename.endswith('_001.JPG'):
                return patient_id, 'R'
            elif filename.endswith('_002.JPG'):
                return patient_id, 'L'
        
        # Pattern 4: Other formats - try to infer from patient directory
        # If we can't determine eye, we'll skip this image
        return patient_id, None
    
    def map_images_to_metadata(self):
        """Map available Fundus images to metadata with robust pattern matching"""
        print("Mapping Fundus images to metadata...")
        
        # Get valid directories
        valid_dirs, excluded_dirs = self.analyze_directory_structure()
        
        matched_images = []
        discarded_images = []
        
        # Process each valid patient directory
        for dir_info in valid_dirs:
            dir_path = dir_info['path']
            patient_id = dir_info['patient_id']
            
            try:
                # Get all image files in directory
                files = os.listdir(dir_path)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # Find matching metadata
                patient_meta = self.metadata_df[
                    self.metadata_df['patient_id'].astype(str) == patient_id
                ]
                
                if len(patient_meta) == 0:
                    for img_file in image_files:
                        discarded_images.append({
                            'image_file': img_file,
                            'image_path': os.path.join(dir_path, img_file),
                            'patient_id': patient_id,
                            'reason': 'No metadata found',
                            'fundus_r': 'N/A',
                            'fundus_l': 'N/A'
                        })
                    continue
                
                patient_row = patient_meta.iloc[0]
                
                # Process each image file
                for img_file in image_files:
                    try:
                        # Extract patient ID and eye
                        extracted_id, eye = self.extract_patient_and_eye_from_filename(img_file, dir_path)
                        
                        if eye is None:
                            discarded_images.append({
                                'image_file': img_file,
                                'image_path': os.path.join(dir_path, img_file),
                                'patient_id': patient_id,
                                'reason': 'Cannot determine eye from filename',
                                'fundus_r': patient_row['Fundus_R'],
                                'fundus_l': patient_row['Fundus_L']
                            })
                            continue
                        
                        # Check if Fundus data is available for this eye
                        fundus_available = False
                        if eye == 'R' and patient_row['Fundus_R'] == 1:
                            fundus_available = True
                        elif eye == 'L' and patient_row['Fundus_L'] == 1:
                            fundus_available = True
                        
                        if fundus_available and pd.notna(patient_row['diagnosis']):
                            # Verify image can be loaded
                            img_path = os.path.join(dir_path, img_file)
                            try:
                                with Image.open(img_path) as img:
                                    img.verify()  # Verify image integrity
                                
                                matched_images.append({
                                    'image_file': img_file,
                                    'image_path': img_path,
                                    'patient_id': patient_id,
                                    'eye': eye,
                                    'diagnosis': patient_row['diagnosis'],
                                    'age': patient_row['Age'],
                                    'sex': patient_row['Sex'],
                                    'side': patient_row['Side'],
                                    'naming_pattern': self._classify_naming_pattern(img_file)
                                })
                            except Exception as img_error:
                                discarded_images.append({
                                    'image_file': img_file,
                                    'image_path': img_path,
                                    'patient_id': patient_id,
                                    'reason': f'Image loading error: {str(img_error)}',
                                    'fundus_r': patient_row['Fundus_R'],
                                    'fundus_l': patient_row['Fundus_L']
                                })
                        else:
                            reason = 'No diagnosis' if pd.isna(patient_row['diagnosis']) else f'Fundus_{eye} not available'
                            discarded_images.append({
                                'image_file': img_file,
                                'image_path': os.path.join(dir_path, img_file),
                                'patient_id': patient_id,
                                'reason': reason,
                                'fundus_r': patient_row['Fundus_R'],
                                'fundus_l': patient_row['Fundus_L']
                            })
                    
                    except Exception as file_error:
                        discarded_images.append({
                            'image_file': img_file,
                            'image_path': os.path.join(dir_path, img_file),
                            'patient_id': patient_id,
                            'reason': f'File processing error: {str(file_error)}',
                            'fundus_r': 'N/A',
                            'fundus_l': 'N/A'
                        })
            
            except Exception as dir_error:
                print(f"Error processing directory {dir_path}: {dir_error}")
        
        self.image_metadata = matched_images
        
        print(f"\n=== FUNDUS IMAGE MAPPING RESULTS ===")
        print(f"Successfully mapped: {len(matched_images)} images")
        print(f"Discarded: {len(discarded_images)} images")
        
        if matched_images:
            df_matched = pd.DataFrame(matched_images)
            print(f"Diagnosis distribution in matched images:")
            print(df_matched['diagnosis'].value_counts().to_dict())
            print(f"Eye distribution: {df_matched['eye'].value_counts().to_dict()}")
            print(f"Naming pattern distribution:")
            print(df_matched['naming_pattern'].value_counts().to_dict())
        
        # Report discarded images by reason
        if discarded_images:
            df_discarded = pd.DataFrame(discarded_images)
            print(f"\nDiscarded images by reason:")
            print(df_discarded['reason'].value_counts().to_dict())
        
        return matched_images, discarded_images
    
    def _classify_naming_pattern(self, filename):
        """Classify the naming pattern of a file"""
        if re.match(r'^[AB]\d+_[RL]\d*\.', filename):
            return 'Standard'
        elif 'EnableIm' in filename:
            return 'EnableImage'
        elif len(filename.split('_')) > 5:
            return 'Complex timestamp'
        else:
            return 'Other'
    
    def get_dataset_info(self):
        """Get comprehensive dataset information"""
        if not self.image_metadata:
            return None
        
        df = pd.DataFrame(self.image_metadata)
        
        info = {
            'total_images': len(df),
            'unique_patients': df['patient_id'].nunique(),
            'diagnosis_distribution': df['diagnosis'].value_counts().to_dict(),
            'eye_distribution': df['eye'].value_counts().to_dict(),
            'naming_pattern_distribution': df['naming_pattern'].value_counts().to_dict(),
            'age_stats': df['age'].describe().to_dict() if 'age' in df.columns else None,
            'sex_distribution': df['sex'].value_counts().to_dict() if 'sex' in df.columns else None
        }
        
        return info


class FundusImageDataset(Dataset):
    """PyTorch Dataset for Fundus images"""
    
    def __init__(self, image_metadata, transform=None):
        self.image_metadata = image_metadata
        self.transform = transform
        
        # Create label mapping
        self.label_map = {'Inflammatory': 0, 'Ischemic': 1}
        
    def __len__(self):
        return len(self.image_metadata)
    
    def __getitem__(self, idx):
        item = self.image_metadata[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_map[item['diagnosis']]
        
        return image, label, item['patient_id'], item['eye']


class FundusClassificationCNN(nn.Module):
    """CNN architecture optimized for Fundus image classification"""
    
    def __init__(self, num_classes=2, input_size=224):
        super(FundusClassificationCNN, self).__init__()
        
        # Feature extraction layers - designed for medical imaging
        self.features = nn.Sequential(
            # First conv block - capture basic features
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            
            # Global average pooling to reduce parameters
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier with reduced parameters
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class FundusTrainer:
    """Training pipeline for Fundus image classification"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_model(self, train_loader, val_loader, num_epochs=40, learning_rate=0.0001):
        """Train the model with enhanced regularization"""
        print(f"Training on device: {self.device}")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        early_stop_patience = 15
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 20)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels, patient_ids, eyes) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 5 == 0:
                    print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels, patient_ids, eyes in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Early stopping and best model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {best_val_acc:.2f}%')
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history, best_val_acc
    
    def evaluate_model(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_patient_ids = []
        all_eyes = []
        
        with torch.no_grad():
            for images, labels, patient_ids, eyes in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_patient_ids.extend(patient_ids)
                all_eyes.extend(eyes)
        
        return all_predictions, all_labels, all_patient_ids, all_eyes


class FundusVisualizer:
    """Visualization utilities for Fundus classification"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        
    def plot_training_history(self, history, filename='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss - Fundus Classification')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Training Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy - Fundus Classification')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, filename='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Inflammatory', 'Ischemic'],
                   yticklabels=['Inflammatory', 'Ischemic'])
        plt.title('Confusion Matrix - Fundus Image Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sample_images(self, dataset, num_samples=8, filename='sample_images.png'):
        """Plot sample Fundus images from dataset"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # Get random samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        for i, idx in enumerate(indices):
            image, label, patient_id, eye = dataset[idx]
            
            # Convert tensor to numpy for plotting
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
                image = (image - image.min()) / (image.max() - image.min())  # Normalize
            
            axes[i].imshow(image)
            axes[i].set_title(f'{patient_id}_{eye}\n{"Inflammatory" if label == 0 else "Ischemic"}')
            axes[i].axis('off')
        
        plt.suptitle('Sample Fundus Images', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_data_distribution(self, matched_images, filename='data_distribution.png'):
        """Plot data distribution analysis"""
        df = pd.DataFrame(matched_images)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Diagnosis distribution
        axes[0, 0].pie(df['diagnosis'].value_counts().values, 
                       labels=df['diagnosis'].value_counts().index,
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Diagnosis Distribution')
        
        # Eye distribution
        axes[0, 1].pie(df['eye'].value_counts().values,
                       labels=df['eye'].value_counts().index,
                       autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Eye Distribution')
        
        # Naming pattern distribution
        pattern_counts = df['naming_pattern'].value_counts()
        axes[1, 0].bar(pattern_counts.index, pattern_counts.values)
        axes[1, 0].set_title('Naming Pattern Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Age distribution by diagnosis
        if 'age' in df.columns:
            for diagnosis in df['diagnosis'].unique():
                subset = df[df['diagnosis'] == diagnosis]['age'].dropna()
                axes[1, 1].hist(subset, alpha=0.7, label=diagnosis, bins=15)
            axes[1, 1].set_title('Age Distribution by Diagnosis')
            axes[1, 1].set_xlabel('Age')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main experiment workflow"""
    print("="*60)
    print("FUNDUS IMAGE CLASSIFICATION EXPERIMENT")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Create results directory
    results_dir = "results/experiment_03"
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
            
            # Initialize data processor
            processor = FundusDataProcessor(
                excel_path='data/ON_1_2_250901_Fundus_VF.xlsx',
                image_dir='data/images/Fundus'
            )
            
            # Load metadata
            metadata_df = processor.load_and_clean_metadata()
            
            # Map images to metadata
            matched_images, discarded_images = processor.map_images_to_metadata()
            
            # Get dataset info
            dataset_info = processor.get_dataset_info()
            print(f"\n=== FINAL DATASET INFO ===")
            for key, value in dataset_info.items():
                print(f"{key}: {value}")
            
            if len(matched_images) == 0:
                print("ERROR: No images could be matched with metadata!")
                return
            
            # =====================================
            # STEP 2: DATA PREPROCESSING
            # =====================================
            print("\n" + "="*50)
            print("STEP 2: DATA PREPROCESSING")
            print("="*50)
            
            # Define image transforms optimized for Fundus images
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Create dataset
            full_dataset = FundusImageDataset(matched_images, transform=train_transform)
            
            # Split dataset
            train_size = int(0.7 * len(full_dataset))
            val_size = int(0.15 * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
            
            # Apply different transforms to validation and test sets
            val_dataset.dataset.transform = val_transform
            test_dataset.dataset.transform = val_transform
            
            print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            # Create data loaders
            batch_size = 8  # Smaller batch size due to potentially limited data
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            # =====================================
            # STEP 3: MODEL ARCHITECTURE
            # =====================================
            print("\n" + "="*50)
            print("STEP 3: MODEL ARCHITECTURE")
            print("="*50)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Initialize model
            model = FundusClassificationCNN(num_classes=2, input_size=224)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # =====================================
            # STEP 4: MODEL TRAINING
            # =====================================
            print("\n" + "="*50)
            print("STEP 4: MODEL TRAINING")
            print("="*50)
            
            # Initialize trainer
            trainer = FundusTrainer(model, device)
            
            # Train model
            history, best_val_acc = trainer.train_model(
                train_loader, val_loader, 
                num_epochs=40, learning_rate=0.0001
            )
            
            # =====================================
            # STEP 5: MODEL EVALUATION
            # =====================================
            print("\n" + "="*50)
            print("STEP 5: MODEL EVALUATION")
            print("="*50)
            
            # Evaluate on test set
            predictions, labels, patient_ids, eyes = trainer.evaluate_model(test_loader)
            
            # Calculate metrics
            test_accuracy = accuracy_score(labels, predictions)
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            # Classification report
            label_names = ['Inflammatory', 'Ischemic']
            report = classification_report(labels, predictions, target_names=label_names)
            print("Classification Report:")
            print(report)
            
            # =====================================
            # STEP 6: VISUALIZATIONS
            # =====================================
            print("\n" + "="*50)
            print("STEP 6: VISUALIZATIONS")
            print("="*50)
            
            # Initialize visualizer
            visualizer = FundusVisualizer(results_dir)
            
            # Plot training history
            visualizer.plot_training_history(history)
            print("Training history plot saved")
            
            # Plot confusion matrix
            visualizer.plot_confusion_matrix(labels, predictions)
            print("Confusion matrix saved")
            
            # Plot sample images
            sample_dataset = FundusImageDataset(matched_images, transform=val_transform)
            visualizer.plot_sample_images(sample_dataset)
            print("Sample images saved")
            
            # Plot data distribution
            visualizer.plot_data_distribution(matched_images)
            print("Data distribution plot saved")
            
            # =====================================
            # STEP 7: SAVE RESULTS
            # =====================================
            print("\n" + "="*50)
            print("STEP 7: SAVE RESULTS")
            print("="*50)
            
            # Save model
            torch.save(model.state_dict(), f"{results_dir}/fundus_classification_model.pth")
            
            # Save matched images metadata
            pd.DataFrame(matched_images).to_csv(f"{results_dir}/matched_images.csv", index=False)
            
            # Save discarded images info
            pd.DataFrame(discarded_images).to_csv(f"{results_dir}/discarded_images.csv", index=False)
            
            # Save test results
            test_results = pd.DataFrame({
                'patient_id': patient_ids,
                'eye': eyes,
                'true_label': labels,
                'predicted_label': predictions,
                'correct': [t == p for t, p in zip(labels, predictions)]
            })
            test_results.to_csv(f"{results_dir}/test_results.csv", index=False)
            
            # =====================================
            # STEP 8: SUMMARY
            # =====================================
            print("\n" + "="*50)
            print("STEP 8: EXPERIMENT SUMMARY")
            print("="*50)
            
            print(f"\n=== FINAL RESULTS ===")
            print(f"Total images processed: {len(matched_images)}")
            print(f"Images discarded: {len(discarded_images)}")
            print(f"Final dataset size: {len(full_dataset)}")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            print(f"Test accuracy: {test_accuracy:.4f}")
            
            print(f"\n=== DIAGNOSIS DISTRIBUTION ===")
            df_matched = pd.DataFrame(matched_images)
            print(df_matched['diagnosis'].value_counts().to_dict())
            
            print(f"\n=== DATA QUALITY INSIGHTS ===")
            print(f"Naming pattern distribution:")
            print(df_matched['naming_pattern'].value_counts().to_dict())
            
            print(f"\n=== OUTPUT FILES ===")
            print(f"All results saved to: {results_dir}/")
            print("Files created:")
            print(f"  - fundus_classification_model.pth (trained model)")
            print(f"  - matched_images.csv (processed dataset)")
            print(f"  - discarded_images.csv (data cleaning log)")
            print(f"  - test_results.csv (evaluation results)")
            print(f"  - training_history.png (training curves)")
            print(f"  - confusion_matrix.png (performance matrix)")
            print(f"  - sample_images.png (data visualization)")
            print(f"  - data_distribution.png (dataset analysis)")
            print(f"  - experiment_log.txt (complete log)")
            
        finally:
            sys.stdout = original_stdout
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {results_dir}/")
    print(f"Log file: {log_file}")
    print(f"Completed at: {datetime.now()}")


if __name__ == "__main__":
    main()

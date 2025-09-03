"""
Visual Field Image Classification Experiment - experiment_02.py

This script performs image-based classification of optic neuritis diseases using Visual Field (VF) images:
1. Data exploration and mapping between Excel metadata and VF images
2. Data cleaning and preprocessing for image data
3. CNN model architecture for binary classification
4. Training pipeline with proper validation
5. Comprehensive evaluation and visualizations

Usage:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 python src/experiment_02.py
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


class VFDataProcessor:
    """Data processor for Visual Field images and metadata"""
    
    def __init__(self, excel_path, image_dir):
        self.excel_path = excel_path
        self.image_dir = image_dir
        self.metadata_df = None
        self.image_metadata = []
        
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
            'age_stats': df['age'].describe().to_dict() if 'age' in df.columns else None,
            'sex_distribution': df['sex'].value_counts().to_dict() if 'sex' in df.columns else None
        }
        
        return info


class VFImageDataset(Dataset):
    """PyTorch Dataset for Visual Field images"""
    
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


class VFClassificationCNN(nn.Module):
    """CNN architecture for VF image classification"""
    
    def __init__(self, num_classes=2, input_size=224):
        super(VFClassificationCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output_size(input_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _get_conv_output_size(self, input_size):
        """Calculate the output size of convolutional layers"""
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        dummy_output = self.features(dummy_input)
        return int(np.prod(dummy_output.size()[1:]))
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class VFTrainer:
    """Training pipeline for VF image classification"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_model(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        """Train the model"""
        print(f"Training on device: {self.device}")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
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
                
                if batch_idx % 10 == 0:
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
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {best_val_acc:.2f}%')
        
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


class VFVisualizer:
    """Visualization utilities for VF classification"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        
    def plot_training_history(self, history, filename='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Training Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
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
        plt.title('Confusion Matrix - VF Image Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sample_images(self, dataset, num_samples=8, filename='sample_images.png'):
        """Plot sample images from dataset"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # Get random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image, label, patient_id, eye = dataset[idx]
            
            # Convert tensor to numpy for plotting
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
                image = (image - image.min()) / (image.max() - image.min())  # Normalize
            
            axes[i].imshow(image)
            axes[i].set_title(f'{patient_id}_{eye}\n{"Inflammatory" if label == 0 else "Ischemic"}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main experiment workflow"""
    print("="*60)
    print("VISUAL FIELD IMAGE CLASSIFICATION EXPERIMENT")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Create results directory
    results_dir = "results/experiment_02"
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
            processor = VFDataProcessor(
                excel_path='data/ON_1_2_250901_Fundus_VF.xlsx',
                image_dir='data/images/VF'
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
            
            # Define image transforms
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
            full_dataset = VFImageDataset(matched_images, transform=train_transform)
            
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
            batch_size = 16
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
            model = VFClassificationCNN(num_classes=2, input_size=224)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # =====================================
            # STEP 4: MODEL TRAINING
            # =====================================
            print("\n" + "="*50)
            print("STEP 4: MODEL TRAINING")
            print("="*50)
            
            # Initialize trainer
            trainer = VFTrainer(model, device)
            
            # Train model
            history, best_val_acc = trainer.train_model(
                train_loader, val_loader, 
                num_epochs=30, learning_rate=0.001
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
            visualizer = VFVisualizer(results_dir)
            
            # Plot training history
            visualizer.plot_training_history(history)
            print("Training history plot saved")
            
            # Plot confusion matrix
            visualizer.plot_confusion_matrix(labels, predictions)
            print("Confusion matrix saved")
            
            # Plot sample images
            sample_dataset = VFImageDataset(matched_images, transform=val_transform)
            visualizer.plot_sample_images(sample_dataset)
            print("Sample images saved")
            
            # =====================================
            # STEP 7: SAVE RESULTS
            # =====================================
            print("\n" + "="*50)
            print("STEP 7: SAVE RESULTS")
            print("="*50)
            
            # Save model
            torch.save(model.state_dict(), f"{results_dir}/vf_classification_model.pth")
            
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
            
            print(f"\n=== OUTPUT FILES ===")
            print(f"All results saved to: {results_dir}/")
            print("Files created:")
            print(f"  - vf_classification_model.pth (trained model)")
            print(f"  - matched_images.csv (processed dataset)")
            print(f"  - discarded_images.csv (data cleaning log)")
            print(f"  - test_results.csv (evaluation results)")
            print(f"  - training_history.png (training curves)")
            print(f"  - confusion_matrix.png (performance matrix)")
            print(f"  - sample_images.png (data visualization)")
            print(f"  - experiment_log.txt (complete log)")
            
        finally:
            sys.stdout = original_stdout
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {results_dir}/")
    print(f"Log file: {log_file}")
    print(f"Completed at: {datetime.now()}")


if __name__ == "__main__":
    main()

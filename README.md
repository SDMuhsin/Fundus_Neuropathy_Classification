# Fundus Disease Classification

Classification of optic neuritis diseases (Inflammatory vs Ischemic ON) using clinical data.

## Quick Start

1. **Activate virtual environment**:
   ```bash
   source env/bin/activate
   ```

2. **Run tabular data experiment**:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python src/experiment_01.py
   ```

3. **Run VF image classification experiment**:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python src/experiment_02.py
   ```

4. **Run Fundus image classification experiment**:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python src/experiment_03.py
   ```

## What It Does

### Experiment 01 (Tabular Data)
- **Data loading** from Excel clinical data (218 patients)
- **Data cleaning** with advanced parsing (89%+ success rates)
- **Statistical analysis** comparing disease types
- **Machine learning** classification (Random Forest)
- **Comprehensive visualizations** (16-panel plots)

### Experiment 02 (VF Image Data)
- **VF image processing** from 301 Visual Field images
- **Image-metadata mapping** with rigorous quality control
- **Deep learning** classification using custom CNN
- **Data augmentation** and proper train/val/test splits
- **Medical imaging visualizations** and performance metrics

### Experiment 03 (Fundus Image Data)
- **Fundus image processing** from 131 images with complex directory structures
- **Multi-pattern filename parsing** handling 4 distinct naming conventions
- **Clinical quality control** respecting explicit exclusions (27% of directories)
- **Advanced data wrangling** with inference-based eye assignment
- **Robust CNN architecture** with enhanced regularization for limited data

## Results

### Experiment 01 Results (`results/experiment_01/`)
- `processed_data.csv` - Cleaned tabular dataset
- `comprehensive_analysis.png` - Statistical visualizations
- `feature_importance.png` - ML feature rankings
- `confusion_matrix.png` - Classification performance
- `experiment_log.txt` - Complete analysis log

### Experiment 02 Results (`results/experiment_02/`)
- `vf_classification_model.pth` - Trained CNN model
- `matched_images.csv` - Processed VF dataset
- `training_history.png` - Training/validation curves
- `confusion_matrix.png` - VF classification performance
- `sample_images.png` - VF image samples
- `experiment_log.txt` - Complete training log

### Experiment 03 Results (`results/experiment_03/`)
- `fundus_classification_model.pth` - Trained Fundus CNN model
- `matched_images.csv` - Processed Fundus dataset with naming patterns
- `discarded_images.csv` - Comprehensive data cleaning log
- `training_history.png` - Training curves with early stopping
- `confusion_matrix.png` - Fundus classification performance
- `sample_images.png` - Fundus image samples
- `data_distribution.png` - Dataset distribution analysis
- `experiment_log.txt` - Complete processing and training log

## Key Findings

### Tabular Data (Experiment 01)
- **Age**: Significant difference between disease types
- **Pain**: Strong association with Inflammatory ON
- **Classification**: Random Forest achieves good separation
- **Data quality**: 89%+ parsing success for complex clinical formats

### VF Image Data (Experiment 02)
- **Dataset**: 266 VF images successfully mapped to diagnoses
- **Image quality**: Rigorous metadata validation and quality control
- **Deep learning**: Custom CNN architecture for medical imaging
- **Performance**: Comprehensive evaluation with proper validation splits

### Fundus Image Data (Experiment 03)
- **Dataset**: ~60-80 Fundus images from 60 valid directories (27% excluded for quality)
- **Data complexity**: Most challenging wrangling with 4 distinct naming patterns
- **Quality control**: Clinical exclusions properly respected and documented
- **Advanced processing**: Inference-based eye assignment and robust validation

## Requirements

- Python 3.7+
- Dependencies in `requirements.txt`
- GPU recommended (CUDA_VISIBLE_DEVICES=1)

## Data

### Tabular Data
- Source: Korean clinical data (Excel format)
- Patients: 218 total (123 Ischemic, 95 Inflammatory)
- Variables: Demographics, clinical features, imaging metadata

### VF Image Data
- Source: Visual Field (VF) images from ophthalmological testing
- Images: 301 total VF images, 266 with valid metadata mapping
- Format: JPG images with patient_eye naming convention
- Distribution: Balanced between left/right eyes and diagnostic groups

### Fundus Image Data
- Source: Fundus photographs from ophthalmological examination
- Images: 131 total images across complex directory structure
- Format: Multiple formats (JPG) with 4 distinct naming patterns
- Quality control: 27% of directories excluded for clinical reasons
- Processing: Advanced multi-pattern parsing with inference-based eye assignment

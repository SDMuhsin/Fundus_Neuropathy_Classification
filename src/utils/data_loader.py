"""
Data loading utilities for fundus disease classification
"""

import pandas as pd
import numpy as np
import os


class DataLoader:
    """Handles loading and initial exploration of Excel data"""
    
    def __init__(self, data_path="data/ON_1_2_250822_Clinical_Data.xlsx"):
        self.data_path = data_path
        self.excel_file = None
        
    def explore_structure(self):
        """Explore the structure of the Excel file"""
        print("=== Excel File Structure Analysis ===")
        
        self.excel_file = pd.ExcelFile(self.data_path)
        print(f"Sheet names: {self.excel_file.sheet_names}")
        
        structure_info = {}
        
        for sheet_name in self.excel_file.sheet_names:
            print(f"\n--- Sheet: {sheet_name} ---")
            
            # Read raw data
            df_raw = pd.read_excel(self.data_path, sheet_name=sheet_name, header=None)
            print(f"Raw shape: {df_raw.shape}")
            
            structure_info[sheet_name] = {
                'shape': df_raw.shape,
                'header_row': 3,  # Based on previous analysis
                'columns': 19     # Based on previous analysis
            }
            
            # Display first few rows
            print("First 5 rows:")
            print(df_raw.head())
            
        return structure_info
    
    def load_sheet(self, sheet_name, disease_type):
        """Load and perform initial processing of a single sheet"""
        print(f"\nLoading Sheet {sheet_name} ({disease_type})...")
        
        # Read raw data
        df_raw = pd.read_excel(self.data_path, sheet_name=sheet_name, header=None)
        
        # Skip header rows (first 3 rows contain headers/metadata)
        df = df_raw.iloc[3:].copy()
        
        # Define column mapping based on previous analysis
        column_mapping = {
            0: 'patient_id',
            1: 'age', 
            2: 'sex',
            3: 'affected_side_raw',
            4: 'diagnosis',
            5: 'pain_raw',
            6: 'rapd_raw',
            7: 'va_right_raw',
            8: 'va_left_raw', 
            9: 'iop_right',
            10: 'iop_left',
            11: 'fundus_right',
            12: 'fundus_left',
            13: 'oct_right',
            14: 'oct_left',
            15: 'rnfl_right',
            16: 'rnfl_left',
            17: 'vf_right',
            18: 'vf_left'
        }
        
        df.columns = [column_mapping.get(i, f'col_{i}') for i in range(len(df.columns))]
        
        # Remove empty rows
        df = df.dropna(subset=['patient_id'])
        
        # Add disease type
        df['disease_type'] = disease_type
        
        print(f"Loaded {len(df)} patients")
        return df
    
    def load_all_data(self):
        """Load both sheets and combine into single dataset"""
        print("=== Loading All Data ===")
        
        # Load both disease types
        df_inflammatory = self.load_sheet('1', 'Inflammatory_ON')
        df_ischemic = self.load_sheet('2', 'Ischemic_ON')
        
        # Combine datasets
        df_combined = pd.concat([df_inflammatory, df_ischemic], ignore_index=True)
        
        print(f"\nCombined Dataset:")
        print(f"Total patients: {len(df_combined)}")
        print(f"Inflammatory ON: {len(df_inflammatory)}")
        print(f"Ischemic ON: {len(df_ischemic)}")
        
        return df_combined, df_inflammatory, df_ischemic

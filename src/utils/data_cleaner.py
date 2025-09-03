"""
Data cleaning utilities for fundus disease classification
"""

import pandas as pd
import numpy as np
import re


class DataCleaner:
    """Comprehensive data cleaning and standardization"""
    
    @staticmethod
    def parse_visual_acuity(va_string):
        """
        Parse visual acuity strings with comprehensive handling
        Returns tuple: (numeric_value, original_format, interpretation)
        """
        if pd.isna(va_string) or va_string == '':
            return np.nan, '', 'missing'
        
        va_str = str(va_string).strip()
        
        # Special cases mapping
        special_cases = {
            'HM': (0.001, 'HM', 'hand_motion'),
            'FC10': (0.002, 'FC10', 'finger_count_10cm'),
            'FC20': (0.003, 'FC20', 'finger_count_20cm'),
            'FC30': (0.004, 'FC30', 'finger_count_30cm'),
            'FC50': (0.005, 'FC50', 'finger_count_50cm'),
            'FC60': (0.006, 'FC60', 'finger_count_60cm'),
            'NLP': (0.0001, 'NLP', 'no_light_perception'),
            'LP': (0.0002, 'LP', 'light_perception'),
            '-': (np.nan, '-', 'not_recorded')
        }
        
        # Check exact matches with special cases
        va_upper = va_str.upper()
        for key, (value, format_type, interp) in special_cases.items():
            if key in va_upper:
                return value, format_type, interp
        
        # Handle patterns like 'X (1.2)' or 'X(0.4)'
        x_match = re.search(r'X\s*\(([0-9.]+)\)', va_str, re.IGNORECASE)
        if x_match:
            corrected = float(x_match.group(1))
            return corrected, f'X({corrected})', 'corrected_only'
        
        # Handle patterns like '0.04(0.3)' - corrected VA in parentheses
        numeric_paren_match = re.search(r'([0-9.]+)\s*\(\s*([0-9.]+)\s*\)', va_str)
        if numeric_paren_match:
            uncorrected = float(numeric_paren_match.group(1))
            corrected = float(numeric_paren_match.group(2))
            return corrected, f'{uncorrected}({corrected})', 'corrected_numeric'
        
        # Handle simple decimal like '0.15', '1.2'
        simple_numeric_match = re.search(r'^([0-9.]+)$', va_str)
        if simple_numeric_match:
            value = float(simple_numeric_match.group(1))
            return value, str(value), 'simple_numeric'
        
        # If nothing matches, return NaN
        return np.nan, va_str, 'unparseable'
    
    @staticmethod
    def standardize_rapd(rapd_string):
        """
        Standardize RAPD (Relative Afferent Pupillary Defect) field
        Returns tuple: (standardized_code, side, certainty)
        """
        if pd.isna(rapd_string) or rapd_string == '':
            return np.nan, np.nan, np.nan
        
        rapd_str = str(rapd_string).strip()
        
        # Handle numeric codes
        if rapd_str in ['1', '2']:
            return int(rapd_str), np.nan, 'definite'
        
        # Korean text mappings
        korean_mappings = {
            '확인불가': (np.nan, np.nan, 'unconfirmable'),
            '확인안됨': (np.nan, np.nan, 'unconfirmed'),
            '기록X': (np.nan, np.nan, 'not_recorded'),
            '좌측 의안': (1, 'L', 'definite')
        }
        
        for korean, (code, side, cert) in korean_mappings.items():
            if korean in rapd_str:
                return code, side, cert
        
        # English patterns
        rapd_lower = rapd_str.lower()
        
        # Extract side information
        side = np.nan
        if 'rt' in rapd_lower or 'right' in rapd_lower:
            side = 'R'
        elif 'lt' in rapd_lower or 'left' in rapd_lower:
            side = 'L'
        elif 'both' in rapd_lower:
            side = 'B'
        
        # Extract certainty
        certainty = 'definite'
        if 'equivocal' in rapd_lower or 'eqivocal' in rapd_lower:
            certainty = 'equivocal'
        elif 'unknown' in rapd_lower:
            certainty = 'unknown'
        
        # Extract numeric code
        code_match = re.search(r'(\d+)', rapd_str)
        code = int(code_match.group(1)) if code_match else 1
        
        return code, side, certainty
    
    @staticmethod
    def standardize_affected_side(side_string):
        """Standardize affected side field"""
        if pd.isna(side_string) or side_string == '':
            return np.nan
        
        side_str = str(side_string).strip().upper()
        
        # Handle complex cases with line breaks and dates
        if '\n' in side_str:
            if 'LT' in side_str or 'L' in side_str:
                return 'L'
            elif 'RT' in side_str or 'R' in side_str:
                return 'R'
            else:
                return 'B'
        
        # Remove spaces
        side_str = side_str.replace(' ', '')
        
        # Standard mappings
        if side_str in ['R', 'RT', 'RIGHT']:
            return 'R'
        elif side_str in ['L', 'LT', 'LEFT']:
            return 'L'
        elif side_str in ['B', 'BOTH', 'BILATERAL']:
            return 'B'
        
        return side_str
    
    @staticmethod
    def standardize_pain(pain_value):
        """Standardize pain field to binary (0=No, 1=Yes)"""
        if pd.isna(pain_value) or pain_value == '':
            return np.nan
        
        pain_str = str(pain_value).strip()
        
        # Handle string values
        if pain_str.lower() in ['unknown', 'unclear', 'nan']:
            return np.nan
        
        # Convert to float
        try:
            pain_val = float(pain_str)
        except (ValueError, TypeError):
            return np.nan
        
        # Standard mapping: 1=Yes, 2=No
        if pain_val == 1.0:
            return 1  # Yes
        elif pain_val == 2.0:
            return 0  # No
        elif pain_val == 0.0:
            return 0  # Treat as No
        elif pain_val == 3.0:
            return np.nan  # Unclear
        else:
            return np.nan
    
    def clean_dataset(self, df):
        """Comprehensive cleaning of the entire dataset"""
        print("=== Comprehensive Data Cleaning ===")
        
        df_clean = df.copy()
        
        # Basic field cleaning
        df_clean['patient_id'] = df_clean['patient_id'].astype(str).str.strip()
        df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
        df_clean['sex'] = df_clean['sex'].astype(str).str.strip().str.upper()
        df_clean['sex'] = df_clean['sex'].replace({'NAN': np.nan})
        
        # Advanced field standardization
        df_clean['affected_side'] = df_clean['affected_side_raw'].apply(self.standardize_affected_side)
        df_clean['diagnosis'] = pd.to_numeric(df_clean['diagnosis'], errors='coerce')
        df_clean['pain_binary'] = df_clean['pain_raw'].apply(self.standardize_pain)
        
        # RAPD parsing
        rapd_results = df_clean['rapd_raw'].apply(self.standardize_rapd)
        df_clean['rapd_code'] = [r[0] if isinstance(r, tuple) else np.nan for r in rapd_results]
        df_clean['rapd_side'] = [r[1] if isinstance(r, tuple) else np.nan for r in rapd_results]
        df_clean['rapd_certainty'] = [r[2] if isinstance(r, tuple) else np.nan for r in rapd_results]
        
        # Visual Acuity parsing
        for side in ['right', 'left']:
            va_results = df_clean[f'va_{side}_raw'].apply(self.parse_visual_acuity)
            df_clean[f'va_{side}_numeric'] = [r[0] if isinstance(r, tuple) else np.nan for r in va_results]
            df_clean[f'va_{side}_format'] = [r[1] if isinstance(r, tuple) else '' for r in va_results]
            df_clean[f'va_{side}_type'] = [r[2] if isinstance(r, tuple) else '' for r in va_results]
        
        # IOP cleaning
        df_clean['iop_right'] = pd.to_numeric(df_clean['iop_right'], errors='coerce')
        df_clean['iop_left'] = pd.to_numeric(df_clean['iop_left'], errors='coerce')
        
        # Test availability (1=available, 2=not available)
        test_columns = ['fundus_right', 'fundus_left', 'oct_right', 'oct_left', 'vf_right', 'vf_left']
        for col in test_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # RNFL thickness
        df_clean['rnfl_right'] = pd.to_numeric(df_clean['rnfl_right'], errors='coerce')
        df_clean['rnfl_left'] = pd.to_numeric(df_clean['rnfl_left'], errors='coerce')
        
        # Create affected eye variables
        df_clean['affected_eye_va'] = np.where(df_clean['affected_side'] == 'R', 
                                             df_clean['va_right_numeric'], 
                                             np.where(df_clean['affected_side'] == 'L',
                                                    df_clean['va_left_numeric'],
                                                    np.nan))
        
        df_clean['affected_eye_iop'] = np.where(df_clean['affected_side'] == 'R', 
                                              df_clean['iop_right'], 
                                              np.where(df_clean['affected_side'] == 'L',
                                                     df_clean['iop_left'],
                                                     np.nan))
        
        df_clean['affected_eye_rnfl'] = np.where(df_clean['affected_side'] == 'R', 
                                               df_clean['rnfl_right'], 
                                               np.where(df_clean['affected_side'] == 'L',
                                                      df_clean['rnfl_left'],
                                                      np.nan))
        
        print(f"Cleaned dataset: {len(df_clean)} patients")
        return df_clean
    
    def generate_quality_report(self, df):
        """Generate comprehensive data quality report"""
        report = []
        report.append("=== DATA QUALITY REPORT ===\n")
        
        # Basic statistics
        report.append(f"Total patients: {len(df)}")
        report.append(f"Disease distribution:")
        for disease, count in df['disease_type'].value_counts().items():
            report.append(f"  {disease}: {count}")
        
        # Missing data analysis
        report.append("\n=== MISSING DATA ANALYSIS ===")
        key_columns = ['age', 'sex', 'affected_side', 'pain_binary', 'affected_eye_va', 
                       'affected_eye_iop', 'affected_eye_rnfl']
        
        for col in key_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                report.append(f"{col}: {missing_count} missing ({missing_pct:.1f}%)")
        
        # Parsing success rates
        report.append("\n=== PARSING SUCCESS RATES ===")
        if 'va_right_type' in df.columns:
            va_types = df['va_right_type'].value_counts()
            report.append("Visual Acuity parsing types:")
            for va_type, count in va_types.items():
                if va_type and va_type != 'missing':
                    report.append(f"  {va_type}: {count}")
        
        return '\n'.join(report)

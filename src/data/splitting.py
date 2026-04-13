import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_by_patient(df, train_ratio=0.8):
    """
    Splits pairs into train/val/test sets based on SampleID to prevent leakage.
    ID format: [SampleID]_patch_[X]_[Y]
    """
    # Extract SampleID (everything before the first '_patch_')
    df['sample_id'] = df['id'].apply(lambda x: x.split('_patch_')[0])
    
    unique_patients = df['sample_id'].unique()
    train_patients, test_patients = train_test_split(unique_patients, train_size=train_ratio, random_state=42)
    
    train_df = df[df['sample_id'].isin(train_patients)]
    test_df = df[df['sample_id'].isin(test_patients)]
    
    # Further split test into val and test (50/50)
    val_patients, test_final_patients = train_test_split(test_patients, train_size=0.5, random_state=42)
    
    val_df = test_df[test_df['sample_id'].isin(val_patients)]
    test_df = test_df[test_df['sample_id'].isin(test_final_patients)]
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Simple test
    data = {'id': ['P1_patch_1_1', 'P1_patch_1_2', 'P2_patch_1_1', 'P3_patch_1_1']}
    df = pd.DataFrame(data)
    train, val, test = split_by_patient(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

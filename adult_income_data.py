import pandas as pd
from sklearn.preprocessing import LabelEncoder
import urllib.request
import os
import sys

def download_and_prepare_adult():
    print("="*60)
    print("ADULT INCOME DATASET DOWNLOADER")
    print("="*60)
    
    os.makedirs('data', exist_ok=True)
    print("\n✓ Created 'data' directory")
    
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
    ]
    
    print("\n📥 Downloading Adult Income dataset from UCI...")
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    try:
        print("  - Downloading training data...")
        urllib.request.urlretrieve(train_url, 'adult_train_raw.csv')
        print("  ✓ Training data downloaded")
        
        print("  - Downloading test data...")
        urllib.request.urlretrieve(test_url, 'adult_test_raw.csv')
        print("  ✓ Test data downloaded")
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)
    
    print("\n📊 Loading and preprocessing data...")
    df_train = pd.read_csv('adult_train_raw.csv', header=None, names=columns, skipinitialspace=True)
    df_test = pd.read_csv('adult_test_raw.csv', header=None, names=columns, skipinitialspace=True, skiprows=1)
    
    print(f"✓ Loaded {len(df_train)} training records")
    print(f"✓ Loaded {len(df_test)} test records")
    
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    print(f"✓ Combined: {len(df_combined)} total records")
    
    print("\n🧹 Handling missing values...")
    missing_before = len(df_combined)
    df_combined = df_combined.replace('?', None)
    df_combined = df_combined.dropna()
    missing_count = missing_before - len(df_combined)
    if missing_count > 0:
        print(f"✓ Removed {missing_count} rows with missing values")
    
    print("\n🎯 Processing target column...")
    df_combined['target'] = df_combined['target'].str.strip()
    df_combined['target'] = (df_combined['target'] == '>50K').astype(int)
    
    print("\n🔢 Encoding categorical features...")
    categorical_cols = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    
    for col in categorical_cols:
        df_combined[col] = df_combined[col].str.strip()
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col])
    
    print(f"✓ Encoded {len(categorical_cols)} categorical features")
    
    print("\n📋 Selecting important features...")
    important_features = [
        'age', 'education-num', 'marital-status', 'occupation', 
        'relationship', 'sex', 'capital-gain', 'capital-loss', 
        'hours-per-week', 'target'
    ]
    df_combined = df_combined[important_features]
    
    print("\n✂️  Splitting data (80% train, 20% test)...")
    train_size = int(0.8 * len(df_combined))
    df_train_final = df_combined[:train_size]
    df_test_final = df_combined[train_size:]
    print(f"✓ Training: {len(df_train_final)} samples")
    print(f"✓ Test: {len(df_test_final)} samples")
    
    print("\n💾 Saving CSV files...")
    df_train_final.to_csv('data/file.csv', index=False)
    df_test_final.drop(columns=['target']).to_csv('data/test.csv', index=False)
    print("✓ Saved data/file.csv")
    print("✓ Saved data/test.csv")
    
    os.remove('adult_train_raw.csv')
    os.remove('adult_test_raw.csv')
    
    print("\n✨ SETUP COMPLETE!")
    print("="*60 + "\n")

if __name__ == "__main__":
    download_and_prepare_adult()
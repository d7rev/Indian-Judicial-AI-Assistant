import pandas as pd
import os

# Find any csv file in the current folder
files = [f for f in os.listdir('.') if f.endswith('.csv')]

if not files:
    print("Error: No CSV file found in this folder!")
else:
    target_file = files[0]
    print(f"Found file: {target_file}. Loading now...")
    
    try:
        df = pd.read_csv(target_file)
        
        print("\n--- Project Statistics ---")
        print(f"Total rows in dataset: {len(df)}")
        print(f"Columns found: {list(df.columns)}")
        
        # This will show us the first few rows so we can see the 'labels'
        print("\n--- Data Sample (First 3 rows) ---")
        print(df.head(3))
        
    except Exception as e:
        print(f"An error occurred: {e}")
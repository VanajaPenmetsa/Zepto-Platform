import pandas as pd
import seaborn as sns

def run_eda():
    df = sns.load_dataset('titanic')
    df.to_csv('analytics/titanic.csv', index=False)
    
    # Cleaning
    df = df.dropna(subset=['embarked'])
    df['age'] = df['age'].fillna(df['age'].median())
    if 'deck' in df.columns:
        df = df.drop(columns=['deck'])
        
    print("Dataset saved to analytics/titanic.csv")
    print(f"Dataset shape after cleaning: {df.shape}")
    print("EDA completed successfully.")

if __name__ == "__main__":
    run_eda()
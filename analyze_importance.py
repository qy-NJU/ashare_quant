import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.xgboost_model import XGBoostWrapper

def analyze_importance(config_path="pipeline_config.yaml"):
    print("--- Analyzing Feature Importance ---")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)['pipeline']
        
    model_path = config['model'].get('save_path')
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please run pipeline first.")
        return
        
    # Load model
    model = XGBoostWrapper()
    model.load(model_path)
    
    if not model.booster:
        print("Failed to load booster.")
        return
        
    # Get feature importance (weight: number of times a feature is used to split the data)
    # Alternatively 'gain' represents the average gain across all splits the feature is used in.
    importance_type = 'gain'
    scores = model.booster.get_score(importance_type=importance_type)
    
    if not scores:
        print("No features found or model is empty.")
        return
        
    # Convert to DataFrame for easier manipulation
    df_imp = pd.DataFrame(list(scores.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print(f"\nTop 20 Features by {importance_type.capitalize()}:")
    print(df_imp.head(20))
    
    print("\nBottom 10 Features (Candidates for Removal):")
    print(df_imp.tail(10))
    
    # Save full list to CSV
    os.makedirs('data/analysis', exist_ok=True)
    out_file = 'data/analysis/feature_importance.csv'
    df_imp.to_csv(out_file, index=False)
    print(f"\nFull feature importance saved to {out_file}")
    
    # Extract top 30 as 'common' feature set
    top_features = df_imp['Feature'].head(30).tolist()
    print("\nRecommended Common Feature Set (Top 30):")
    print(top_features)

if __name__ == "__main__":
    analyze_importance()

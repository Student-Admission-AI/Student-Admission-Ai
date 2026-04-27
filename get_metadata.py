import joblib
import pandas as pd

try:
    # Load your existing masters model
    model = joblib.load('Models/masters_regression.pkl')

    # Ask the model: "What columns were you trained on?"
    features = model.get_booster().feature_names
    
    # Save that list to a file
    joblib.dump(features, 'Models/feature_list.pkl')
    print(f"✅ SUCCESS: Mapped {len(features)} features to Models/feature_list.pkl")
except Exception as e:
    print(f"❌ ERROR: {e}")
